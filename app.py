"""
Overvalued Stock Screen: ACCT 3343 Analysis App
================================================

SELL (overvalued / short candidates), three steps:
  1. Forensic red flags (Chapter 3): scaled total accruals (STA), scaled net operating
     assets (SNOA) and the Beneish manipulation probability (PMAN).
  2. Distress flag (Chapter 4): Altman Z-score.
  3. Valuation check (CAPM class): how much of today's price current earnings cannot
     support at the CAPM required return.

BUY (a different analysis, assignment rule 2):
  - financial strength: Piotroski F-score (Chapter 6)
  - franchise quality: CFOA and Max-Margin (Chapter 5)
  - ranked on gross profitability + book-to-market (Chapter 2)

Every formula follows the instructor's chapter notebooks
(github.com/x2003xuy/acct3343-public). Missing inputs stay missing (no fillna(0)), so a
company can never be flagged by a measure it has no data for.

Data: universe.csv (one row per company), built with yfinance in
build_universe.ipynb. The latest fiscal year has no suffix; earlier years end in
_prev, _prev2 and _prev3.
"""
from pathlib import Path
from statistics import NormalDist
from string import Template

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ============================================================================
# 1. SETTINGS
# ============================================================================
DATA_FILE = Path(__file__).with_name("universe.csv")

# Assignment rule 1: my 60 Second Pitch company may not appear anywhere in the app.
PITCH_TICKER = "UBER"                 # Uber Technologies: my 60 Second Pitch (a long), dropped below

YEARS = ["", "_prev", "_prev2", "_prev3"]   # column suffixes, latest fiscal year first
FINANCIALS_EXCLUDED = 695      # Financial Services firms the screener query left out (build log)

RISK_FREE_RATE = 0.0409        # 3-month T-bill (FRED GS3M), Sep 2026: CAPM Example.xlsx
MARKET_RETURN = 0.09           # workbook's teaching assumption for E(Rm); a sidebar slider
ALTMAN_DISTRESS = 1.81         # Chapter 4 distress boundary
ALTMAN_ORIGINAL = 2.675        # Altman's original cutoff (Chapter 4)
BENEISH_CUTOFF = -1.78         # Beneish (1999) PROBM cutoff
MIN_SECTOR_FIRMS = 3           # course rule: a sector mean needs at least 3 valid firms
NORMAL = NormalDist()          # standard normal CDF for PMAN (Chapter 3 notebook)
F_SIGNALS = ["F_ROA", "F_CFO", "F_DROA", "F_ACCRUAL", "F_DLEV",
             "F_DLIQ", "F_EQ", "F_DGM", "F_DAT"]


# ============================================================================
# 2. HELPERS
# ============================================================================
def safe_divide(numerator, denominator, positive_denominator=False):
    """Divide two columns, leaving impossible cases missing instead of 0 or infinity.

    Same rule as the Chapter 3 notebook: the denominator must be non-zero (or
    positive when `positive_denominator=True`) and both inputs must be known.
    """
    valid = numerator.notna() & denominator.notna()
    valid &= denominator.gt(0) if positive_denominator else denominator.ne(0)
    result = numerator.where(valid) / denominator.where(valid)
    return result.replace([np.inf, -np.inf], np.nan)


def is_prior_year(df, newer, older, min_days, max_days):
    """True when the `older` fiscal period is the year right before the `newer` one.

    Chapter 3 accepts 330-400 days between fiscal year-ends; Chapters 5 and 6 accept
    350-380 days. Both also require the fiscal-year label to increase by exactly 1.
    """
    days = (df["fiscal_period_end" + newer] - df["fiscal_period_end" + older]).dt.days
    one_year = (df["fiscal_year" + newer] - df["fiscal_year" + older]).eq(1)
    return days.between(min_days, max_days) & one_year


def indicator(test, valid):
    """1 if a test passes, 0 if it fails, missing if its inputs are unknown (Chapter 6)."""
    return pd.Series(np.where(valid, test.astype(float), np.nan), index=test.index)


# ============================================================================
# 3. COURSE MEASURES: one function per chapter, ported from the class notebooks
# ============================================================================
def add_market_equity(d):
    """Market value of equity at the fiscal year-end (Chapters 2 and 4).

    Year-end shares x year-end closing price, so the price matches the date of the
    balance sheet. Our own safety check: statement shares x today's price should be
    within 0.5-2x of today's market cap. A mismatch means the share count is in
    different units (e.g. an ADR) or a split happened after year-end, so the value is
    left missing rather than off by a large factor.
    """
    d["market_equity"] = d["ordinary_shares_number"] * d["fiscal_period_end_close"]
    d["share_count_check"] = safe_divide(
        d["ordinary_shares_number"] * d["current_share_price"], d["current_market_cap"], True)
    mismatch = d["share_count_check"].notna() & ~d["share_count_check"].between(0.5, 2.0)
    d["market_equity"] = d["market_equity"].mask(mismatch)
    d["market_equity_note"] = np.where(
        mismatch, "Statement share count does not match market cap (ADR or recent split)", "")
    return d


def add_chapter2(d):
    """Chapter 2: gross profitability (GPA) and book-to-market (B/M)."""
    d["GPA"] = safe_divide(d["gross_profit"], d["total_assets"], True)
    d["BM"] = safe_divide(d["stockholders_equity"], d["market_equity"], True)
    d["BM_today"] = safe_divide(d["stockholders_equity"], d["current_market_cap"], True)
    return d


def add_chapter3_accruals(d):
    """Chapter 3: scaled total accruals (STA) and scaled net operating assets (SNOA).

    STA  = (net income - cash flow from operations) / total assets
    SNOA = (operating assets - operating liabilities) / total assets, where
      operating assets      = total assets - cash
      operating liabilities = total assets - short-term debt - long-term debt
                              - minority interest - preferred stock - common equity
    Input preparation follows the Chapter 3 notebook exactly.
    """
    assets_ok = d["total_assets"].gt(0)
    d["STA"] = safe_divide(d["net_income"] - d["operating_cash_flow"], d["total_assets"], True)
    d["STA_reason"] = np.select(
        [~assets_ok, d["STA"].isna()],
        ["Total assets missing or not positive", "Missing net income or operating cash flow"],
        default="")

    # Short-term debt (with leases); rebuilt from total debt only when it is missing.
    current_debt = d["current_debt_and_capital_lease_obligation"]
    residual_debt = d["total_debt"] - d["long_term_debt_and_capital_lease_obligation"]
    rebuild_debt = current_debt.isna() & residual_debt.ge(-d["total_assets"].abs() * 1e-9)
    current_debt = current_debt.mask(rebuild_debt, residual_debt.clip(lower=0))

    # Minority interest: reported value, else the balance-sheet identity (within 0.01%).
    minority = d["minority_interest"]
    identity_gap = (d["total_assets"] - d["total_liabilities_net_minority_interest"]
                    - d["stockholders_equity"])
    tolerance = d["total_assets"].abs() * 1e-4
    rebuild_minority = minority.isna() & identity_gap.ge(-tolerance)
    minority = minority.mask(rebuild_minority, identity_gap.where(identity_gap.abs().gt(tolerance), 0.0))

    # Preferred stock: zero only when it is structurally absent (core equity lines known).
    preferred = d["preferred_stock"]
    absent = preferred.isna() & d[["total_assets", "common_stock_equity",
                                   "stockholders_equity"]].notna().all(axis=1)
    preferred = preferred.mask(absent, 0.0)

    d["operating_assets"] = d["total_assets"] - d["cash_and_cash_equivalents"]
    d["operating_liabilities"] = (d["total_assets"] - current_debt
                                  - d["long_term_debt_and_capital_lease_obligation"]
                                  - minority - preferred - d["common_stock_equity"])
    snoa_inputs_ok = pd.concat([d["cash_and_cash_equivalents"], current_debt,
                                d["long_term_debt_and_capital_lease_obligation"],
                                minority, preferred, d["common_stock_equity"]],
                               axis=1).notna().all(axis=1)
    d["SNOA"] = safe_divide(d["operating_assets"] - d["operating_liabilities"],
                            d["total_assets"].where(snoa_inputs_ok), True)
    d["SNOA_reason"] = np.select(
        [~assets_ok, ~snoa_inputs_ok],
        ["Total assets missing or not positive", "Missing debt, minority interest or equity input"],
        default="")
    return d


def add_chapter3_beneish(d):
    """Chapter 3 extension: Beneish (1999) eight-variable PROBM and PMAN = N(PROBM).

    Requires the immediately preceding fiscal year (330-400 days earlier) and all eight
    indices. Financial companies are not applicable. TATA uses the course's
    balance-sheet accrual formula. A labelled cash-flow version (Beneish, Lee and
    Nichols 2013) is also computed because Yahoo often leaves taxes payable blank.
    """
    p = "_prev"
    rev, rev_p = d["total_revenue"], d["total_revenue" + p]
    margin = safe_divide(rev - d["cost_of_revenue"], rev, True)
    margin_p = safe_divide(rev_p - d["cost_of_revenue" + p], rev_p, True)
    quality = 1 - safe_divide(d["current_assets"] + d["net_property_plant_equipment"],
                              d["total_assets"], True)
    quality_p = 1 - safe_divide(d["current_assets" + p] + d["net_property_plant_equipment" + p],
                                d["total_assets" + p], True)
    dep_rate = safe_divide(d["depreciation_and_amortization"],
                           d["depreciation_and_amortization"] + d["net_property_plant_equipment"], True)
    dep_rate_p = safe_divide(d["depreciation_and_amortization" + p],
                             d["depreciation_and_amortization" + p]
                             + d["net_property_plant_equipment" + p], True)
    leverage = safe_divide(d["long_term_debt"] + d["current_liabilities"], d["total_assets"], True)
    leverage_p = safe_divide(d["long_term_debt" + p] + d["current_liabilities" + p],
                             d["total_assets" + p], True)

    d["DSRI"] = safe_divide(safe_divide(d["accounts_receivable"], rev, True),
                            safe_divide(d["accounts_receivable" + p], rev_p, True))
    d["GMI"] = safe_divide(margin_p, margin)
    d["AQI"] = safe_divide(quality, quality_p)
    d["SGI"] = safe_divide(rev, rev_p, True)
    d["DEPI"] = safe_divide(dep_rate_p, dep_rate)
    d["SGAI"] = safe_divide(safe_divide(d["selling_general_and_administration"], rev, True),
                            safe_divide(d["selling_general_and_administration" + p], rev_p, True))
    d["LVGI"] = safe_divide(leverage, leverage_p)
    accruals = ((d["current_assets"] - d["current_assets" + p])
                - (d["cash_and_cash_equivalents"] - d["cash_and_cash_equivalents" + p])
                - ((d["current_liabilities"] - d["current_liabilities" + p])
                   - (d["current_debt"] - d["current_debt" + p])
                   - (d["taxes_payable"] - d["taxes_payable" + p]))
                - d["depreciation_and_amortization"])
    d["TATA"] = safe_divide(accruals, d["total_assets"], True)
    d["TATA_CF"] = safe_divide(d["net_income_continuing_operations"] - d["operating_cash_flow"],
                               d["total_assets"], True)

    applicable = is_prior_year(d, "", p, 330, 400) & d["sector"].ne("Financial Services")
    others = ["DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "LVGI"]
    for suffix, tata in [("", "TATA"), ("_CF", "TATA_CF")]:
        eligible = applicable & d[others + [tata]].notna().all(axis=1)
        probm = (-4.84 + 0.920 * d["DSRI"] + 0.528 * d["GMI"] + 0.404 * d["AQI"]
                 + 0.892 * d["SGI"] + 0.115 * d["DEPI"] - 0.172 * d["SGAI"]
                 + 4.679 * d[tata] - 0.327 * d["LVGI"])
        d["PROBM" + suffix] = probm.where(eligible)
        d["PMAN" + suffix] = d["PROBM" + suffix].map(
            lambda v: NORMAL.cdf(v) if pd.notna(v) else np.nan)
    d["PMAN_reason"] = np.select(
        [d["sector"].eq("Financial Services"), ~is_prior_year(d, "", p, 330, 400),
         d["PROBM"].isna()],
        ["Not applicable to financial companies", "No valid preceding fiscal year",
         "Missing Beneish input (often taxes payable)"], default="")
    return d


def add_chapter4_altman(d):
    """Chapter 4: Altman Z-score, course formula.

    Z = 0.012 X1 + 0.014 X2 + 0.033 X3 + 0.006 X4 + 0.999 X5, with X1-X4 in percentage
    points (as the Chapter 4 notebook specifies) and X5 as a ratio:
      X1 = working capital / TA       X2 = retained earnings / TA
      X3 = EBIT / TA                  X4 = market equity / book total liabilities
      X5 = sales / TA
    Working capital falls back to current assets - current liabilities. Yahoo's total
    liabilities field is blank, so the notebook's net-of-minority total is used.
    """
    working_capital = d["working_capital"].fillna(d["current_assets"] - d["current_liabilities"])
    liabilities = d["total_liabilities"].fillna(d["total_liabilities_net_minority_interest"])
    inputs = pd.concat([working_capital, d["retained_earnings"], d["ebit"], d["market_equity"],
                        liabilities, d["total_revenue"], d["total_assets"]], axis=1)
    eligible = inputs.notna().all(axis=1) & d["total_assets"].gt(0) & liabilities.gt(0)
    assets = d["total_assets"].where(eligible)
    d["X1"] = working_capital / assets
    d["X2"] = d["retained_earnings"] / assets
    d["X3"] = d["ebit"] / assets
    d["X4"] = d["market_equity"] / liabilities.where(eligible)
    d["X5"] = d["total_revenue"] / assets
    d["Altman_Z"] = (0.012 * (100 * d["X1"]) + 0.014 * (100 * d["X2"]) + 0.033 * (100 * d["X3"])
                     + 0.006 * (100 * d["X4"]) + 0.999 * d["X5"])
    d["Altman_zone"] = np.select(
        [d["Altman_Z"].lt(ALTMAN_DISTRESS), d["Altman_Z"].lt(ALTMAN_ORIGINAL),
         d["Altman_Z"].ge(ALTMAN_ORIGINAL)],
        ["Below 1.81 (distress)", "Between 1.81 and 2.675", "At or above 2.675"], default="")
    d["Altman_reason"] = np.where(
        d["Altman_Z"].isna(),
        np.where(d["market_equity"].isna(), "Missing fiscal-date market equity",
                 "Missing working capital, retained earnings, EBIT, liabilities or sales"), "")
    return d


def add_chapter5_franchise(d):
    """Chapter 5: CFOA, margin growth, margin stability, gross profitability and ROC.

    Uses the latest four fiscal years, which must be consecutive (350-380 days apart).
      CFOA              = sum of 4 years of free cash flow / latest total assets
      Gross margin      = (revenue - cost of revenue) / revenue
      Margin growth     = (latest GM / first GM) ** (1/3) - 1  (needs positive GM all 4 years)
      Margin stability  = mean GM / sample standard deviation of GM
    """
    history_ok = pd.Series(True, index=d.index)
    for newer, older in zip(YEARS[:-1], YEARS[1:]):
        history_ok &= is_prior_year(d, newer, older, 350, 380)

    fcf = d[["free_cash_flow" + y for y in YEARS]]
    cfoa_ok = history_ok & fcf.notna().all(axis=1) & d["total_assets"].gt(0)
    d["CFOA"] = (fcf.sum(axis=1) / d["total_assets"]).where(cfoa_ok)

    margins = pd.concat([safe_divide(d["total_revenue" + y] - d["cost_of_revenue" + y],
                                     d["total_revenue" + y], True) for y in YEARS], axis=1)
    margin_ok = history_ok & margins.notna().all(axis=1)
    growth_ok = margin_ok & margins.gt(0).all(axis=1)
    first, last = margins.iloc[:, -1], margins.iloc[:, 0]        # oldest, latest
    d["Margin_Growth"] = ((last / first.where(growth_ok)) ** (1 / 3) - 1).where(growth_ok)
    spread = margins.std(axis=1, ddof=1)
    stable_ok = margin_ok & spread.gt(0) & margins.nunique(axis=1).gt(1)
    d["Margin_Stability"] = (margins.mean(axis=1) / spread.where(stable_ok)).where(stable_ok)
    d["Gross_Margin"] = margins.iloc[:, 0]
    d["Franchise_reason"] = np.where(history_ok, "",
                                     "Needs four consecutive fiscal years of statements")

    d["Gross_Profitability"] = safe_divide(d["total_revenue"] - d["cost_of_revenue"],
                                           d["total_assets"], True)
    capital = d["total_debt"] + d["stockholders_equity"] - d["cash_and_cash_equivalents"]
    d["ROC_PreTax"] = safe_divide(d["ebit"], capital, True)
    tax_ok = d["tax_rate_for_calcs"].between(0, 1)
    d["ROC_AfterTax"] = safe_divide(d["ebit"] * (1 - d["tax_rate_for_calcs"]),
                                    capital, True).where(tax_ok)
    return d


def add_chapter6_fscore(d):
    """Chapter 6: classroom Piotroski F-score (nine 0/1 signals; complete scores only).

    ROA and CFO use beginning-of-year assets; leverage = long-term debt / ending
    assets; liquidity = current assets / current liabilities; margin = gross margin;
    turnover = sales / beginning assets. Changes compare the latest year with the year
    before (350-380 days apart). Equity uses net common-stock issuance.
    """
    def ratio(numerator, denominator):
        return numerator.where(np.isfinite(numerator)) / \
            denominator.where(np.isfinite(denominator) & denominator.gt(0))

    pair0 = is_prior_year(d, "", "_prev", 350, 380)          # latest vs prior year
    pair1 = is_prior_year(d, "_prev", "_prev2", 350, 380)    # prior year vs the one before
    begin0 = d["total_assets_prev"].where(pair0)             # beginning assets, latest year
    begin1 = d["total_assets_prev2"].where(pair1)            # beginning assets, prior year

    roa0 = ratio(d["net_income"], begin0)
    roa1 = ratio(d["net_income_prev"], begin1)
    cfo0 = ratio(d["operating_cash_flow"], begin0)
    lev0 = ratio(d["long_term_debt"].where(d["long_term_debt"].ge(0)), d["total_assets"])
    lev1 = ratio(d["long_term_debt_prev"].where(d["long_term_debt_prev"].ge(0)), d["total_assets_prev"])
    liq0 = ratio(d["current_assets"].where(d["current_assets"].ge(0)), d["current_liabilities"])
    liq1 = ratio(d["current_assets_prev"].where(d["current_assets_prev"].ge(0)),
                 d["current_liabilities_prev"])
    gm0 = ratio(d["total_revenue"] - d["cost_of_revenue"], d["total_revenue"])
    gm1 = ratio(d["total_revenue_prev"] - d["cost_of_revenue_prev"], d["total_revenue_prev"])
    turn0 = ratio(d["total_revenue"].where(d["total_revenue"].ge(0)), begin0)
    turn1 = ratio(d["total_revenue_prev"].where(d["total_revenue_prev"].ge(0)), begin1)

    d["ROA"], d["CFO_to_assets"] = roa0, cfo0
    d["dROA"] = roa0 - roa1.where(pair0)
    d["dLeverage"] = lev0 - lev1.where(pair0)
    d["dLiquidity"] = liq0 - liq1.where(pair0)
    d["dGrossMargin"] = gm0 - gm1.where(pair0)
    d["dTurnover"] = turn0 - turn1.where(pair0)

    # Net common-stock issuance; fallback = gross issuance + signed repurchases (labelled).
    issuance = d["net_common_stock_issuance"]
    fallback = issuance.isna() & d["issuance_of_capital_stock"].ge(0) & \
        d["repurchase_of_capital_stock"].le(0)
    d["net_issuance"] = issuance.mask(
        fallback, d["issuance_of_capital_stock"] + d["repurchase_of_capital_stock"])

    known = np.isfinite
    d["F_ROA"] = indicator(roa0.gt(0), known(roa0))
    d["F_CFO"] = indicator(cfo0.gt(0), known(cfo0))
    d["F_DROA"] = indicator(d["dROA"].gt(0), known(d["dROA"]))
    d["F_ACCRUAL"] = indicator(cfo0.gt(roa0), known(cfo0) & known(roa0))
    d["F_DLEV"] = indicator(d["dLeverage"].lt(0), known(d["dLeverage"]))
    d["F_DLIQ"] = indicator(d["dLiquidity"].gt(0), known(d["dLiquidity"]))
    d["F_EQ"] = indicator(d["net_issuance"].le(0), d["net_issuance"].notna())
    d["F_DGM"] = indicator(d["dGrossMargin"].gt(0), known(d["dGrossMargin"]))
    d["F_DAT"] = indicator(d["dTurnover"].gt(0), known(d["dTurnover"]))

    signals = d[F_SIGNALS]
    d["F_Score"] = signals.sum(axis=1, min_count=9)       # complete scores only
    d["F_Score_low"] = signals.sum(axis=1, min_count=0)   # if every unknown signal failed
    d["F_Score_high"] = d["F_Score_low"] + 9 - signals.count(axis=1)
    d["F_reason"] = np.where(d["F_Score"].isna(),
                             "Missing signal(s) or no consecutive prior year", "")
    return d


def add_capm_inputs(d):
    """Inputs for the CAPM valuation check (class CAPM workbook).

    Beta: 60 monthly returns vs the S&P 500, the workbook method (it reproduces Apple's
    1.07). Yahoo's beta is used only when fewer than 36 months exist, and is labelled.
    E/P uses the latest fiscal year's net income over today's market cap.
    """
    own = d["beta_60m"].where(d["beta_60m_months"].ge(36))
    d["beta"] = own.fillna(d["beta_yahoo"])
    d["beta_source"] = np.where(own.notna(), "60-mo regression",
                                np.where(d["beta"].notna(), "Yahoo 5Y monthly", "unavailable"))
    d["earnings_yield"] = safe_divide(d["net_income"], d["current_market_cap"], True)
    years_ok = is_prior_year(d, "", "_prev", 350, 380) & \
        is_prior_year(d, "_prev", "_prev2", 350, 380) & is_prior_year(d, "_prev2", "_prev3", 350, 380)
    growth = safe_divide(d["total_revenue"], d["total_revenue_prev3"], True)
    d["revenue_cagr_3y"] = (growth ** (1 / 3) - 1).where(years_ok & d["total_revenue"].gt(0))
    return d


def compute_measures(raw):
    """Run every chapter's calculations on the company file (one row per company)."""
    d = raw.copy()
    for y in YEARS:
        d["fiscal_period_end" + y] = pd.to_datetime(d["fiscal_period_end" + y], errors="coerce")
    for step in (add_market_equity, add_chapter2, add_chapter3_accruals, add_chapter3_beneish,
                 add_chapter4_altman, add_chapter5_franchise, add_chapter6_fscore, add_capm_inputs):
        d = step(d).copy()   # copy() packs the new columns together (keeps pandas fast)
    return d


# ============================================================================
# 4. SCREENING TOOLS (rankings and the CAPM check)
# ============================================================================
def sector_demean(frame, column):
    """Value minus its sector's mean (Chapters 3 and 4: 'compare with broad peers').

    A sector mean is used only when the sector has at least 3 valid values; otherwise
    the demeaned value stays missing.
    """
    groups = frame.groupby("sector")[column]
    enough = groups.transform("count").ge(MIN_SECTOR_FIRMS)
    return (frame[column] - groups.transform("mean")).where(frame[column].notna() & enough)


def course_rank_sum(frame, columns):
    """Chapter 3 ranking: rank each measure (1 = most concerning, ties share the lowest
    rank) among firms that have all of them, add the ranks, then rank the sum."""
    complete = frame[columns].notna().all(axis=1)
    ranks = frame.loc[complete, columns].rank(ascending=False, method="min")
    return ranks.sum(axis=1).rank(method="min").reindex(frame.index)


def percentile(series):
    """Percentile rank from 0 to 100 within the companies currently screened."""
    return series.rank(pct=True) * 100


def capm_required_return(beta, market_return, risk_free=RISK_FREE_RATE):
    """CAPM (class workbook): r = Rf + beta x (E(Rm) - Rf). Negative betas are floored
    at 0, so no stock is required to earn less than the risk-free rate."""
    return risk_free + beta.clip(lower=0) * (market_return - risk_free)


def growth_share_of_price(earnings_yield, required_return):
    """Share of today's price that current earnings, valued with no growth at r, can't
    explain: 1 - (E/P) / r. It also equals implied growth g* / r, where g* = r - E/P.
    Loss-makers (E <= 0) have no earnings support, so they are capped at 100%."""
    return (1 - earnings_yield / required_return).clip(upper=1.0)


# ============================================================================
# 5. THE SCREEN: one function turns the sidebar settings into SELL / BUY lists
# ============================================================================
TATA_OPTIONS = ["Course formula; cash-flow TATA only where inputs are missing",
                "Course formula only (balance-sheet TATA)"]
SELL_RANKINGS = ["Most red flags first", "Course rank-sum (STA + SNOA)", "Highest growth share"]
BUY_RANKINGS = ["Course rank-sum (GPA + B/M)", "Weighted percentiles"]
FLAGS = {"flag_STA": "Accruals", "flag_SNOA": "Bloated NOA", "flag_PMAN": "Beneish",
         "flag_Z": "Distress"}


def run_screen(measures, s):
    """Apply one set of settings `s` (a dict of sidebar values) to every company.

    Returns a dict with the screened universe `u` (every company in scope, with its flags
    and scores) plus the ordered `sell`, `watch` and `buy` tables.
    """
    in_scope = (measures["exclusion_reason"].isna()
                & measures["sector"].isin(s["sectors"])
                & measures["current_market_cap"].ge(s["min_cap"] * 1e9))
    u = measures[in_scope].copy()

    # SELL step 1: forensic flags (Chapter 3). Percentiles are recomputed among the
    # companies in scope, so every universe filter can change who is flagged.
    sector_relative = s["benchmark"] == "Sector-relative"
    for m in ("STA", "SNOA"):
        u[m + "_basis"] = sector_demean(u, m) if sector_relative else u[m]
        u[m + "_pct"] = percentile(u[m + "_basis"])
        u["flag_" + m] = u[m + "_pct"].ge(s[m.lower() + "_pct"])     # sta_pct / snoa_pct sliders
    # Beneish: the course's balance-sheet TATA wherever its inputs exist. Yahoo often leaves
    # taxes payable blank, so by default those companies use the published cash-flow TATA.
    u["PROBM_used"], u["PMAN_used"] = u["PROBM"], u["PMAN"]
    u["TATA_source"] = np.where(u["PROBM"].notna(), "Course balance-sheet TATA", "")
    if s["tata_version"] == TATA_OPTIONS[0]:
        fallback = u["PROBM"].isna() & u["PROBM_CF"].notna()
        u["PROBM_used"] = u["PROBM"].fillna(u["PROBM_CF"])
        u["PMAN_used"] = u["PMAN"].fillna(u["PMAN_CF"])
        u.loc[fallback, "TATA_source"] = "Cash-flow TATA (fallback)"
    u["flag_PMAN"] = u["PROBM_used"].gt(s["beneish_cut"])

    # SELL step 2: distress flag (Chapter 4). The sector-relative Z (the course's second
    # stage) is kept as context; it doesn't change the flag.
    u["Altman_relative"] = sector_demean(u, "Altman_Z")
    u["flag_Z"] = u["Altman_Z"].lt(s["altman_cut"])

    u["red_flags"] = u[list(FLAGS)].sum(axis=1)
    u["flags_assessable"] = u[["STA_basis", "SNOA_basis", "PROBM_used", "Altman_Z"]].notna().sum(axis=1)
    u["flagged"] = u["red_flags"].ge(s["flags_needed"])
    u["accrual_rank"] = course_rank_sum(u, ["STA_basis", "SNOA_basis"])

    # SELL step 3: valuation check (CAPM). A company passes when at least the threshold
    # share of its price depends on growth that current earnings do not show.
    u["required_return"] = capm_required_return(u["beta"], s["market_return"] / 100)
    u["growth_share"] = growth_share_of_price(u["earnings_yield"], u["required_return"])
    u["implied_growth"] = (u["required_return"] - u["earnings_yield"]).where(u["earnings_yield"].gt(0))
    if s["valuation_on"]:
        u["priced_for_growth"] = u["growth_share"].ge(s["growth_share_min"] / 100)
    else:
        u["priced_for_growth"] = True
    u["SELL"] = u["flagged"] & u["priced_for_growth"]
    u["WATCH"] = u["flagged"] & ~u["priced_for_growth"]

    sell_order = {
        SELL_RANKINGS[0]: (["red_flags", "accrual_rank", "growth_share"], [False, True, False]),
        SELL_RANKINGS[1]: (["accrual_rank", "red_flags", "growth_share"], [True, False, False]),
        SELL_RANKINGS[2]: (["growth_share", "red_flags", "accrual_rank"], [False, False, True]),
    }[s["sell_rank"]]
    sell = u[u["SELL"]].sort_values(sell_order[0], ascending=sell_order[1], na_position="last")
    watch = u[u["WATCH"]].sort_values(["red_flags", "accrual_rank"], ascending=[False, True],
                                      na_position="last")

    # BUY: a different analysis. Gates: Chapter 6 F-score, then Chapter 5 franchise
    # measures; ranking: Chapter 2 gross profitability + book-to-market.
    u["CFOA_pct"] = percentile(u["CFOA"])
    growth_pct, stability_pct = percentile(u["Margin_Growth"]), percentile(u["Margin_Stability"])
    u["Max_Margin"] = pd.concat([growth_pct, stability_pct], axis=1).max(axis=1, skipna=False)
    pool = u["red_flags"].eq(0) if s["buy_guard"] else ~u["SELL"]   # never both SELL and BUY
    strong = pool & u["F_Score"].ge(s["fscore_min"])
    franchise = strong & u["CFOA_pct"].ge(s["franchise_floor"]) & \
        u["Max_Margin"].ge(s["franchise_floor"])
    passers = u[franchise].copy()
    if s["buy_rank"] == BUY_RANKINGS[0]:
        passers["buy_score"] = course_rank_sum(passers, ["GPA", "BM"])          # 1 = best
        buy = passers.sort_values("buy_score", na_position="last")
    else:
        passers["buy_score"] = (s["buy_weight"] * percentile(passers["GPA"])
                                + (1 - s["buy_weight"]) * percentile(passers["BM"]))
        buy = passers.sort_values("buy_score", ascending=False, na_position="last")
    buy = buy[buy["buy_score"].notna()].head(s["buy_n"])
    u["BUY"] = u.index.isin(buy.index)

    funnel = {
        "sell": [("In scope", len(u)), ("At least 1 red flag", int(u["red_flags"].ge(1).sum())),
                 (f"At least {s['flags_needed']} red flags", int(u["flagged"].sum())),
                 ("Priced for growth (SELL)", int(u["SELL"].sum()))],
        "buy": [("No red flags" if s["buy_guard"] else "In scope", int(pool.sum())),
                (f"F-score at least {s['fscore_min']}", int(strong.sum())),
                ("Franchise gates", int(franchise.sum())), ("BUY list", len(buy))],
    }
    return {"u": u, "sell": sell, "watch": watch, "buy": buy, "funnel": funnel}


# ============================================================================
# 6. PRESENTATION HELPERS (formatting and charts)
# ============================================================================
SELL_COLOR, BUY_COLOR, OTHER_COLOR = "#eb6834", "#2a78d6", "#b8bfcc"   # validated palette slots


def fmt_pct(x, digits=1):
    if pd.isna(x):
        return "n/a"
    value = round(x * 100, digits) + 0.0          # + 0.0 turns -0.0 into 0.0
    return f"{value:.{digits}f}%".replace("-", "−")


def fmt_num(x, digits=2):
    return "n/a" if pd.isna(x) else f"{x:,.{digits}f}"


def fmt_money(x):
    if pd.isna(x):
        return "n/a"
    sign, x = ("−" if x < 0 else ""), abs(x)
    if x >= 1e12:
        return f"{sign}${x / 1e12:,.2f}T"
    return f"{sign}${x / 1e9:,.1f}B" if x >= 1e9 else f"{sign}${x / 1e6:,.0f}M"


def ordinal(n):
    n = int(round(n))
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def group_of(u):
    return pd.Series(np.select([u["SELL"], u["WATCH"], u["BUY"]], ["SELL", "Watchlist", "BUY"],
                               default="Other"), index=u.index)


def finish(fig, title, height=380):
    """Shared chart styling: title, legend on top, thin margins."""
    fig.update_layout(title=dict(text=title, font=dict(size=15)), height=height,
                      font=dict(family="Inter, system-ui, sans-serif"),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=10, r=10, t=60, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, title=None),
                      hoverlabel=dict(font_size=12))
    return fig


def group_scatter(u, x, y, x_title, y_title, title, hline=None, vline=None, y_pct=False):
    """Every screened company as a dot: SELL, Watchlist, BUY and the rest in grey."""
    fig = go.Figure()
    groups = group_of(u)
    styles = {"Other": dict(color=OTHER_COLOR, size=7, opacity=0.45, symbol="circle"),
              "BUY": dict(color=BUY_COLOR, size=10, opacity=0.95, symbol="circle"),
              "Watchlist": dict(color=SELL_COLOR, size=10, opacity=0.95, symbol="circle-open"),
              "SELL": dict(color=SELL_COLOR, size=11, opacity=0.95, symbol="circle")}
    for name, style in styles.items():
        part = u[groups.eq(name) & u[x].notna() & u[y].notna()]
        if part.empty:
            continue
        fig.add_trace(go.Scatter(
            x=part[x], y=part[y], mode="markers", name=f"{name} ({len(part)})",
            marker=dict(color=style["color"], size=style["size"], opacity=style["opacity"],
                        symbol=style["symbol"], line=dict(width=1.5 if name == "Watchlist" else 0.5,
                                                          color=style["color"] if name == "Watchlist" else "white")),
            customdata=np.stack([part["ticker"], part["company_name"]], axis=-1),
            hovertemplate="<b>%{customdata[0]}</b> · %{customdata[1]}<br>"
                          f"{x_title}: %{{x:.3g}}<br>{y_title}: %{{y:.3g}}<extra>{name}</extra>"))
    if hline is not None:
        fig.add_hline(y=hline, line_width=1, line_color="#898781")
    if vline is not None:
        fig.add_vline(x=vline, line_width=1, line_color="#898781")
    fig.update_xaxes(title=x_title, range=view_range(u[x], vline))
    fig.update_yaxes(title=y_title, tickformat=".0%" if y_pct else None, range=view_range(u[y], hline))
    return finish(fig, title, height=430)


def view_range(values, line=None):
    """Axis range covering the middle 96% of companies (plus any cutoff line), so a few
    extreme values don't squash the chart. Outliers stay reachable with the zoom tools."""
    low, high = values.quantile([0.02, 0.98])
    if line is not None:
        low, high = min(low, line), max(high, line)
    pad = (high - low) * 0.06 or 1
    return [low - pad, high + pad]


def funnel_chart(stages, title):
    labels, counts = zip(*stages)
    fig = go.Figure(go.Bar(x=counts, y=labels, orientation="h", marker_color=BUY_COLOR,
                           marker_cornerradius=4, text=[f"{c:,}" for c in counts],
                           textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}: %{x:,}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(showticklabels=False, showgrid=False, range=[0, max(counts) * 1.15])
    return finish(fig, title, height=260)


# Beneish (1999), Table 2: sample means for manipulators vs non-manipulators.
BENEISH_MEANS = {"DSRI": (1.465, 1.031), "GMI": (1.193, 1.014), "AQI": (1.254, 1.039),
                 "SGI": (1.607, 1.134), "DEPI": (1.077, 1.001), "SGAI": (1.041, 1.054),
                 "LVGI": (1.111, 1.037)}


def why_flagged(row, s):
    """Plain-English reasons, generated from the numbers, for one company."""
    peer = " versus sector peers" if s["benchmark"] == "Sector-relative" else ""
    notes = []
    if row["flag_STA"]:
        gap = row["net_income"] - row["operating_cash_flow"]
        if gap > 0:
            detail = (f"net income beat operating cash flow by {fmt_money(gap)}, so reported "
                      "profit is running ahead of cash")
        else:
            detail = (f"cash flow still exceeds net income, but by much less than at sector peers "
                      f"(STA {fmt_pct(row['STA'])} vs a sector average of "
                      f"{fmt_pct(row['STA'] - row['STA_basis'])})")
        notes.append(f"**Accruals (Ch 3):** {detail}. STA {fmt_pct(row['STA'])} is in the "
                     f"{ordinal(row['STA_pct'])} percentile{peer}.")
    if row["flag_SNOA"]:
        notes.append(f"**Bloated balance sheet (Ch 3):** net operating assets are "
                     f"{fmt_pct(row['SNOA'], 0)} of total assets ({ordinal(row['SNOA_pct'])} "
                     f"percentile{peer}), a sign of accumulated accruals.")
    if row["flag_PMAN"]:
        source = " using the cash-flow TATA fallback" if "fallback" in row["TATA_source"] else ""
        notes.append(f"**Beneish (Ch 3):** PROBM {fmt_num(row['PROBM_used'])} is above the "
                     f"{fmt_num(s['beneish_cut'])} cutoff (PMAN {fmt_pct(row['PMAN_used'])}{source}).")
    if row["flag_Z"]:
        relative = row["Altman_relative"]
        versus = ("" if pd.isna(relative) else
                  f", {fmt_num(abs(relative))} {'below' if relative < 0 else 'above'} its sector average")
        notes.append(f"**Distress (Ch 4):** Altman Z {fmt_num(row['Altman_Z'])} is below the "
                     f"{fmt_num(s['altman_cut'])} cutoff{versus}.")
    if pd.notna(row["growth_share"]):
        if row["earnings_yield"] <= 0:
            notes.append("**Valuation (CAPM):** the company lost money, so none of today's price is "
                         "supported by current earnings.")
        else:
            notes.append(f"**Valuation (CAPM):** at a {fmt_pct(row['required_return'])} required "
                         f"return, {fmt_pct(row['growth_share'], 0)} of today's price depends on "
                         f"growth: earnings must grow about {fmt_pct(row['implied_growth'])} a year "
                         f"forever (revenue grew {fmt_pct(row['revenue_cagr_3y'])} a year over 3 years).")
    return notes


# ============================================================================
# 7. MY WRITTEN ANALYSIS (investment conclusions are my own words; numbers are live)
# ============================================================================
CONCLUSIONS = {
    "SELL": [
        {"ticker": "NVMI", "verdict": "SELL · low conviction", "text": """
**Thesis.** Nova is my first SELL candidate because its stock price still depends heavily on future
growth, while its reported earnings are slightly less cash-backed than those of its technology peers.
The company is profitable and financially stable, and its high SNOA turns out to be mostly mechanical
(see below), but I think the valuation leaves limited room for disappointing results.

**What the financial statements show.** Nova's revenue increased from $571 million in 2022 to
$881 million in 2025, while net income increased from $140 million to $259 million. Operating cash
flow also improved from $120 million to $246 million. However, net income exceeded operating cash flow
by approximately $14 million in 2025.

The screen flags Nova's scaled net operating assets (SNOA), which reached 81% of total assets, placing
it in the 97th percentile compared with sector peers. Its scaled total accruals (STA) were only 0.6%,
but still ranked in the 94th percentile compared with sector peers.

Receivables increased from $109 million in 2022 to $152 million in 2025, although revenue grew faster
over that period. This suggests that receivables alone do not explain the accounting concerns. The
Beneish asset quality index of 1.31 points to a change in the composition of assets, which I examined
further.

**Alternative explanations.** The balance sheet explains most of the SNOA flag. Nova's total debt rose
from $236 million to $799 million in 2025, and it holds $834 million of short-term investments and
$585 million of longer-term investment securities, but only $214 million of cash and cash equivalents.
The course's SNOA formula subtracts only cash, so these $1.42 billion of investments count as operating
assets. Treating them as the financial assets they are, Nova's SNOA would be about 21% rather than 81%.
The same investments also explain the higher asset quality index.

Nova's strong growth and stable gross margin (55.5% in 2022 to 57.4% in 2025) suggest that the business
itself is performing well. In addition, operating cash flow exceeded net income in 2024, which weakens
the argument that earnings are consistently unsupported by cash. I therefore view the accounting
signals as weak, and the SELL case rests mainly on valuation.

**What the price assumes.** Using a beta of 1.78, the CAPM model estimates a required return of 12.8%.
Nova's earnings yield is only 2.5% (a P/E of about 41), meaning approximately 81% of its share price is
attributed to future growth under the app's simplified valuation model. The implied perpetual growth
rate is 10.4%, compared with actual three-year revenue growth of 15.6% annually. Although recent growth
supports some of the valuation, maintaining strong growth indefinitely is much harder than achieving it
over three years.

**Catalyst, risks and verdict.** Over the next 6–12 months, weaker revenue growth or lower margins in
the semiconductor equipment cycle could cause investors to reconsider the valuation. I would reconsider
the SELL thesis if operating cash flow consistently exceeded net income and revenue growth stayed
comfortably above the roughly 10% a year that the price implies. Short interest is 5.0% of float, with
2.8 days to cover, creating some short-squeeze risk.

**Verdict: SELL, low conviction.** Nova is a strong business. Once its investment portfolio is accounted
for, the balance-sheet warning largely disappears, and the case depends on its valuation being too high.
"""},
        {"ticker": "SNPS", "verdict": "SELL · medium conviction", "text": """
**Thesis.** Synopsys is a SELL candidate because its operating assets have expanded significantly,
while its valuation still depends heavily on future growth. Its accounting indicators suggest that the
balance sheet deserves closer examination.

**Financial evidence and valuation.** Synopsys reported $7.1 billion in revenue in 2025, compared with
$4.6 billion in 2022. However, total assets increased from $9.4 billion to $48.2 billion over the same
period. Its SNOA reached 82%, placing it in the 98th percentile compared with sector peers.

The jump reflects the acquisition of Ansys, completed during fiscal 2025: goodwill and intangible assets
rose from $3.6 billion to $39.6 billion (82% of total assets), funded largely by about $13.5 billion of
new long-term debt. This explains the Beneish asset quality index of 2.02 and leverage index of 1.75.
Receivables also increased from $941 million in 2024 to $1.55 billion in 2025, partly because the
acquired company's receivables were added.

Because the asset increase comes from an acquisition rather than from accruals in the existing
business, it is not evidence of manipulation. The risk is different: Synopsys now has to earn an
adequate return on almost $40 billion of acquired assets. Operating cash flow also exceeded net income
in 2025, which is an important counterargument.

At a 10.0% required return, approximately 85% of the stock price depends on future growth. The implied
perpetual growth rate is 8.5%, compared with three-year revenue growth of 15.2%.

**Catalyst, risks and verdict.** Over the next 6–12 months, acquisition integration results, operating
margins and cash generation could affect the valuation. I would reconsider the SELL thesis if the
enlarged asset base produced sustainable earnings growth while operating cash flow stayed stronger than
net income. Short interest is 2.6% of float, with 2.5 days to cover.

**Verdict: SELL, medium conviction.** The valuation is demanding and the company must now justify a
much larger, debt-funded asset base, but the strong cash flow means the accounting concerns should not
be overstated.
"""},
        {"ticker": "COHR", "verdict": "SELL · high conviction", "text": """
**Thesis.** Coherent is a SELL candidate because its recent earnings improvement has not translated into
stronger operating cash flow, while its stock price assumes substantial future growth.

**Financial evidence and valuation.** Coherent's net income increased from $49 million in 2025 to
$805 million in 2026. However, operating cash flow declined from $634 million to only $80 million. This
created a $725 million gap between reported profit and operating cash flow. Free cash flow also declined
from positive $193 million to approximately negative $1.0 billion. This is particularly concerning
because the company's share price increased approximately 97% over the previous 12 months. Its STA
reached 4.0%, ranking in the 98th percentile compared with sector peers, while SNOA reached 74%, in the
92nd percentile.

The cash-flow statement explains the gap. Working capital absorbed $1.1 billion: inventory rose from
$1.44 billion to $2.58 billion (up 79%, while revenue grew 23%) and receivables increased by
$368 million. Net income also included a $124 million gain on the sale of a business, whose
$437 million of proceeds appear in investing rather than operating cash flow. Capital spending of
$1.1 billion then turned free cash flow negative.

One possible explanation is that Coherent is building inventory and capacity for strong demand: revenue
increased from $5.8 billion to $7.1 billion, and gross margin improved from 35.2% to 37.5%. However,
inventory growing more than three times faster than sales is a classic accrual warning sign. If demand
slows, some of that inventory may have to be written down.

At a 14.5% required return, approximately 88% of the share price depends on growth. The implied
perpetual growth rate is 12.7%, above its historical three-year revenue growth of 11.3%.

**Catalyst, risks and verdict.** The next annual or quarterly financial results could reveal whether
operating cash flow recovers and whether the inventory build turns into sales. I would reconsider the
SELL thesis if operating cash flow exceeded net income for two consecutive quarters, free cash flow
returned to positive territory, and inventory fell back toward its 2025 level relative to sales. Short
interest is 5.6% of float, with 1.8 days to cover.

**Verdict: SELL, high conviction.** Coherent has the clearest recent disconnect between reported
earnings and cash generation among these three candidates, driven by a large inventory build, although
continued growth and improving margins remain important risks to the thesis.
"""},
    ],
    "BUY": [
        {"ticker": "KFY", "verdict": "BUY · high conviction", "text": """
**Thesis.** Korn Ferry is my first BUY candidate because it combines strong financial performance,
consistent cash generation and a relatively attractive valuation. The company is not growing rapidly,
but its profitability and financial strength suggest that the market may be undervaluing its ability to
generate sustainable returns.

**What the financial statements show.** Korn Ferry's revenue remained relatively stable, increasing from
$2.8 billion in 2024 to $2.9 billion in 2026. Over the same period, net income increased from
$169 million to $277 million. The company also generated $414 million in operating cash flow and
$324 million in free cash flow in 2026, both improvements over the previous year. This demonstrates
that the business continues to produce cash despite relatively modest revenue growth.

Its Piotroski F-score is 8 out of 9, showing strong overall financial health. The company passes the
profitability, cash generation, improving return on assets, cash-backed earnings, leverage, liquidity,
share issuance and asset turnover tests. The only failed signal is gross margin improvement.

**Business quality and valuation.** Korn Ferry's Chapter 5 cash flow on assets (CFOA) is 0.28, placing
it in the 64th percentile. Its Max-Margin measure ranks in the 88th percentile, suggesting relatively
strong margin stability. The company's return on capital after tax is 20.1%, indicating that it
generates attractive returns relative to the capital invested in the business.

From Chapter 2, Korn Ferry has gross profitability of 0.64 and a book-to-market ratio of 0.59 at its
fiscal year-end (the figure the screen ranks on) and 0.48 at today's price. This means its book equity
represents approximately 48% of its current market value.

The valuation may reflect concerns about slower hiring activity, economic uncertainty and limited
revenue growth. These are reasonable concerns for a staffing and consulting business. However, the
improvement in profitability and continued cash generation suggest that the company may be better
positioned than its modest revenue growth implies.

**Alternative explanations and risks.** One concern is that gross margin declined from 90.7% in 2023 to
88.1% in 2026. The company's margins are not improving consistently, and an economic slowdown could
reduce demand for its services. I also recognize that a high F-score does not automatically mean a
stock is undervalued. The investment case depends on Korn Ferry maintaining its profitability and
generating sufficient future returns.

**Catalyst and time horizon.** Over the next 6–12 months, improving hiring demand, stronger consulting
activity or continued earnings growth could encourage investors to assign a higher valuation to the
business. I would reconsider the BUY thesis if the F-score fell below 7, operating cash flow declined
below $300 million annually, or return on capital fell below 15%.

**Verdict: BUY, high conviction.** Korn Ferry offers a strong combination of financial quality, stable
margins and a valuation that appears reasonable relative to its profitability.
"""},
        {"ticker": "TILE", "verdict": "BUY · medium-to-high conviction", "text": """
**Thesis.** Interface is a BUY candidate because its profitability has improved consistently while its
valuation remains relatively moderate. Its financial statements suggest that operational improvements
are producing stronger results.

**Financial evidence and valuation.** Interface's revenue increased from approximately $1.3 billion in
2022 to $1.4 billion in 2025. More importantly, net income increased from $20 million to $116 million
over the same period. Gross margin improved every year, rising from 33.7% to 38.7%. This suggests
better operating performance rather than growth driven entirely by higher sales.

The company has a Piotroski F-score of 8 out of 9. Its CFOA is 0.31, ranking in the 69th percentile,
while its Max-Margin ranks in the 79th percentile. Return on capital after tax is 15.9%.

Interface's current book-to-market ratio is 0.31, compared with 0.40 at fiscal year-end. The stock has
become more expensive relative to book value, but its profitability improvements still support further
investigation. A possible reason for the valuation is concern about commercial construction and
renovation demand. However, the improvement in margins suggests the company has become more efficient.

**Catalyst, risks and verdict.** Over the next 6–12 months, continued margin improvement and earnings
growth could support a higher valuation. Short interest is high at 10.0% of float (7.7 days to cover),
a sign that some investors doubt the recovery. I would reconsider the BUY thesis if gross margin fell
below 36%, the F-score declined below 7, or annual free cash flow fell below $100 million.

**Verdict: BUY, medium-to-high conviction.** Interface has demonstrated improving profitability,
although demand sensitivity and its higher current valuation remain risks.
"""},
        {"ticker": "BBY", "verdict": "BUY · medium conviction", "text": """
**Thesis.** Best Buy is a BUY candidate because it continues to generate substantial cash flow and
attractive returns despite weaker revenue. The company offers financial strength, although its
valuation is less compelling than the other BUY candidates.

**Financial evidence and valuation.** Best Buy's revenue declined from $46.3 billion in 2023 to
$41.7 billion in 2026. However, operating cash flow remained strong at approximately $2.0 billion in
2026, while free cash flow reached $1.3 billion. Gross margin also improved from 21.4% in 2023 to 22.5%
in 2026, suggesting that the company has maintained some profitability despite lower sales, although
it slipped slightly in the latest year (from 22.6% to 22.5%).

Best Buy has a Piotroski F-score of 7 out of 9, indicating relatively strong financial health; the two
tests it fails are that leverage rose and that gross margin dipped in the latest year. Its CFOA is 0.29
(66th percentile), while its Max-Margin ranks in the 75th percentile. Return on capital after tax is
20.6%.

The company's gross profitability is 0.64, but its book-to-market ratio is 0.22 at fiscal year-end and
only 0.15 at today's price, after a 20% rise in the share price over the past year. This is a less
attractive valuation signal than Korn Ferry or Interface, making the BUY argument more dependent on
continued cash generation.

**Catalyst, risks and verdict.** Over the next 6–12 months, stabilizing consumer electronics demand and
continued free cash flow generation could improve investor confidence. I would reconsider the BUY
thesis if annual free cash flow fell below $1.0 billion, gross margin declined below 21%, or the F-score
dropped below 7 (it would then fail the screen's F-score gate).

**Verdict: BUY, medium conviction.** Best Buy is financially strong, but declining revenue and its
relatively low book-to-market ratio make it a less convincing value opportunity.
"""},
    ],
    "portfolio": """
The three SELL candidates are all technology companies with high net operating assets and valuations
that depend heavily on future growth. This creates sector concentration and makes the short portfolio
vulnerable if technology stocks continue performing well. None of them is a distress short: their
Altman Z-scores (7.96, 3.17 and 7.51) are well above the 1.81 cutoff, so the case is about earnings
quality and valuation, not bankruptcy risk. The screen's signals are also only moderately robust: all
three are flagged under the course defaults and the Broad preset, but none under the Strict preset or
when accruals are measured in absolute terms. After reviewing the statements, my conviction order
differs from the screen's ranking: Coherent first, then Synopsys, with Nova last.

The BUY candidates are more diversified across staffing services, manufacturing and retail. They share
strong F-scores, positive cash generation and relatively stable or improving profitability. However,
their valuations are not equally attractive, which is why Korn Ferry is my preferred BUY. It is also the
most robust result: Korn Ferry ranks first and Interface ranks in the top three under all six standard
settings, while Best Buy makes the top three only under the course defaults.

The screen is useful for identifying potential opportunities, but it cannot fully assess management
quality, competitive advantages, acquisition details or events occurring after the latest financial
statements. Nova also shows that a measure can be distorted by how a formula classifies assets. These
recommendations should therefore be treated as starting points for further research rather than
automatic trading decisions.
""",
}


# ============================================================================
# 8. PAGE SETUP AND DATA
# ============================================================================
st.set_page_config(page_title="Paper Profits · Overvalued Stock Screen", page_icon="🧾",
                   layout="wide")

# Look and feel. The deck allows only three files, so instead of a separate theme file the
# colours and fonts are set here with CSS: an indigo accent instead of Streamlit's red, a
# navy sidebar, serif headings, and card-style panels. Light and dark versions follow the
# viewer's Streamlit theme.
THEMES = {
    "light": dict(page="#f3f5fa", card="#ffffff", border="#e1e6f0", heading="#0f1b2d", muted="#5b6476",
                  accent="#4f46e5", button="#4f46e5", tint="rgba(79,70,229,.08)", ring="rgba(79,70,229,.22)",
                  tag_bg="#e6e5fc", tag_text="#3730a3", shadow="rgba(15,27,45,.06)"),
    "dark": dict(page="#0b1120", card="#111a2e", border="#1e2a44", heading="#eef2fa", muted="#9aa6bd",
                 accent="#818cf8", button="#4f46e5", tint="rgba(129,140,248,.12)", ring="rgba(129,140,248,.28)",
                 tag_bg="#2b3170", tag_text="#e0e3ff", shadow="rgba(0,0,0,.35)"),
}
STYLE = Template("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600&family=Inter:wght@400;500;600;700&display=swap');

/* Fonts: Inter for text, Fraunces for headings; icons keep their own icon font */
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp button, [data-testid="stMarkdownContainer"],
[data-testid="stMetricValue"], [data-testid="stWidgetLabel"], [data-testid="stTab"] {
    font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif; }
.stApp h1, .stApp h2, .stApp h3 { font-family: 'Fraunces', Georgia, serif !important; color: $heading;
    letter-spacing: -0.01em; }
[data-testid="stIconMaterial"] { font-family: 'Material Symbols Rounded' !important; }

/* Page */
.stApp { background: $page; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"] { padding-top: 2.4rem; max-width: 1500px; }

/* Header banner */
.pp-hero { border-radius: 20px; padding: 28px 34px 24px; margin-bottom: 1.3rem; color: #f5f7fc;
    background: radial-gradient(900px 380px at 88% -30%, rgba(129,140,248,.38), transparent 60%),
                linear-gradient(135deg, #0f1b2d 0%, #182747 55%, #272f6b 100%);
    box-shadow: 0 18px 40px rgba(15,27,45,.18); }
.pp-eyebrow { font-size: .74rem; font-weight: 700; letter-spacing: .16em; text-transform: uppercase;
    color: #a5b4fc; }
.pp-title { font-family: 'Fraunces', Georgia, serif; font-size: 2.9rem; font-weight: 600; line-height: 1.05;
    margin: .35rem 0 .55rem; color: #ffffff; }
.pp-title span { color: #a5b4fc; }
.pp-lede { font-size: 1.04rem; line-height: 1.55; color: #cbd5e8; max-width: 880px; margin: 0 0 1rem; }
.pp-chips { display: flex; flex-wrap: wrap; gap: 8px; }
.pp-chips span { font-size: .78rem; padding: 4px 11px; border-radius: 999px; color: #e2e8f5;
    background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.14); }
.pp-footer { margin-top: 2.6rem; padding-top: 1rem; border-top: 1px solid $border; color: $muted;
    font-size: .8rem; }
@media (max-width: 640px) {
    .pp-hero { padding: 20px 20px 18px; border-radius: 16px; }
    .pp-title { font-size: 2.2rem; }
    .pp-lede { font-size: .95rem; } }

/* Cards: headline numbers, charts, expanders */
[data-testid="stMetric"] { background: $card; border: 1px solid $border !important; border-radius: 16px !important;
    padding: 16px 18px 14px !important; box-shadow: 0 1px 2px $shadow, 0 10px 24px $shadow; }
[data-testid="stMetricLabel"] p { font-size: .74rem !important; font-weight: 600; letter-spacing: .08em;
    text-transform: uppercase; color: $muted !important; }
[data-testid="stMetricValue"] { font-weight: 700; color: $heading !important; }
[data-testid="stExpander"] [data-testid="stMetricLabel"] p { font-size: .66rem !important; letter-spacing: .03em; }
[data-testid="stExpander"] [data-testid="stMetricValue"] { font-size: 1.7rem !important; }
[data-testid="stPlotlyChart"] { background: $card; border: 1px solid $border; border-radius: 16px;
    padding: 6px 8px; box-shadow: 0 1px 2px $shadow; }
[data-testid="stExpander"] details { background: $card; border: 1px solid $border !important;
    border-radius: 14px !important; }

/* Tabs */
[data-testid="stTabs"] [role="tablist"] { gap: 6px; border-bottom: 1px solid $border; }
[data-testid="stTab"] { padding: 10px 16px !important; border-radius: 10px 10px 0 0; color: $muted; }
[data-testid="stTab"] p { font-weight: 600; }
[data-testid="stTab"]:hover { color: $accent !important; }
[data-testid="stTab"][aria-selected="true"] { color: $accent !important; background: $tint; }
[data-testid="stTab"] [class*="SelectionIndicator"] { background-color: $accent !important; height: 3px !important;
    border-radius: 3px; }

/* Indigo accent on every control that Streamlit colours red by default */
[data-testid="stBaseButton-primary"] { background: $button !important; border-color: $button !important;
    color: #ffffff !important; }
[data-testid="stBaseButton-primary"]:hover { filter: brightness(1.1); }
[data-testid="stMultiSelectTagsContainer"] [data-tag] { background: $tag_bg !important; color: $tag_text !important; }
[data-testid="stSlider"] [role="group"] > div > div:first-child { filter: hue-rotate(243deg) saturate(.85); }
[data-testid="stSlider"] [role="group"] > div > div[style*="absolute"] { background: $accent !important;
    box-shadow: 0 0 0 4px $ring; }
[data-testid="stSliderThumbValue"], [data-testid="stSliderThumbValue"] p { color: $accent !important; font-weight: 600; }
[data-testid="stRadioOption"][data-selected="true"] > div > div:first-child { background: $accent !important;
    border-color: $accent !important; }
[data-testid="stCheckbox"][data-selected="true"] label > div:first-of-type { background: $accent !important; }
[data-testid="stDownloadButton"] button { border-color: $accent !important; color: $accent !important; }

/* Sidebar: a deep navy panel in both light and dark mode */
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f1b2d 0%, #18264a 100%);
    border-right: 1px solid rgba(255,255,255,.06); }
[data-testid="stSidebar"] h1 { font-family: 'Inter', sans-serif !important; font-size: .78rem !important;
    font-weight: 700; letter-spacing: .16em; text-transform: uppercase; color: #a5b4fc !important; }
[data-testid="stSidebar"] h3 { font-family: 'Inter', sans-serif !important; font-size: .98rem !important;
    color: #ffffff !important; border-top: 1px solid rgba(255,255,255,.09); padding-top: .9rem !important;
    margin-top: .4rem; }
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p, [data-testid="stSidebar"] [data-testid="stRadioOption"] p,
[data-testid="stSidebar"] [data-testid="stCheckbox"] p, [data-testid="stSidebar"] summary p { color: #dce3f1 !important; }
[data-testid="stSidebar"] [data-testid="stSliderTickBar"] p { color: rgba(220,227,241,.6) !important; }
[data-testid="stSidebar"] [data-testid="stSliderThumbValue"],
[data-testid="stSidebar"] [data-testid="stSliderThumbValue"] p { color: #c7d2fe !important; }
[data-testid="stSidebar"] [data-testid="stSlider"] [role="group"] > div > div[style*="absolute"] {
    background: #a5b4fc !important; box-shadow: 0 0 0 4px rgba(165,180,252,.25); }
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] > div > div:first-child,
[data-testid="stSidebar"] [data-testid="stCheckbox"][data-selected="true"] label > div:first-of-type {
    background: #818cf8 !important; border-color: #818cf8 !important; }
[data-testid="stSidebar"] [data-testid="stTooltipIcon"], [data-testid="stSidebar"] [data-testid="stTooltipIcon"] button {
    color: #b4c2dd !important; opacity: 1 !important; }
[data-testid="stSidebar"] [data-testid="stTooltipIcon"] svg * { stroke: #b4c2dd !important; }
[data-testid="stSidebar"] [data-testid="stIconMaterial"] { color: #c7d2fe; }
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] { background: rgba(255,255,255,.06) !important;
    border: 1px solid rgba(255,255,255,.18) !important; color: #e6ebf5 !important; }
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover { border-color: #a5b4fc !important; }
[data-testid="stSidebar"] [data-testid="stExpander"] details { background: rgba(255,255,255,.04) !important;
    border-color: rgba(255,255,255,.12) !important; }
</style>
""")
st.markdown(STYLE.substitute(THEMES["dark" if st.context.theme.type == "dark" else "light"]),
            unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading company data…")
def load_data(file_version, pitch_ticker):
    """Read universe.csv and run every course calculation. Cached until the file changes."""
    raw = pd.read_csv(DATA_FILE)
    if pitch_ticker:
        raw.loc[raw["ticker"].eq(pitch_ticker), "exclusion_reason"] = \
            "My 60 Second Pitch company (assignment rule 1)"
    return compute_measures(raw), list(raw.columns)


measures, RAW_COLUMNS = load_data(DATA_FILE.stat().st_mtime, PITCH_TICKER)
usable = measures[measures["exclusion_reason"].isna()]
ALL_SECTORS = sorted(usable["sector"].dropna().unique())
caps = usable["current_market_cap"].dropna() / 1e9
CAP_STEPS = [0.0] + [float(f"{caps.quantile(q):.2g}") for q in (0.25, 0.5, 0.75, 0.9)]

DEFAULTS = dict(
    sectors=[x for x in ALL_SECTORS if x not in ("Real Estate", "Utilities")], min_cap=0.0,
    benchmark="Sector-relative", sta_pct=90, snoa_pct=90, beneish_cut=BENEISH_CUTOFF,
    altman_cut=ALTMAN_DISTRESS, flags_needed=2, valuation_on=True,
    growth_share_min=50, market_return=MARKET_RETURN * 100, sell_rank=SELL_RANKINGS[0],
    buy_guard=True, fscore_min=7, franchise_floor=50, buy_rank=BUY_RANKINGS[0],
    buy_weight=0.5, buy_n=20, tata_version=TATA_OPTIONS[0])
PRESETS = {
    "Strict": dict(sta_pct=95, snoa_pct=95, growth_share_min=70, fscore_min=8, franchise_floor=60,
                   buy_n=15),
    "Broad": dict(sta_pct=80, snoa_pct=80, altman_cut=ALTMAN_ORIGINAL, beneish_cut=-2.22,
                  growth_share_min=30, fscore_min=6, franchise_floor=40, buy_n=30),
}

for key, value in DEFAULTS.items():               # first visit: start from the course defaults
    st.session_state.setdefault(key, value)


def apply_settings(values):
    """Preset and reset buttons: overwrite the sidebar values (runs before widgets draw)."""
    st.session_state.update(values)


# ============================================================================
# 9. SIDEBAR CONTROLS (every value feeds run_screen)
# ============================================================================
sb = st.sidebar
sb.title("Screen settings")
sb.button("Reset to course defaults", on_click=apply_settings, args=(DEFAULTS,),
          type="primary", width="stretch", key="preset_default")
p1, p2 = sb.columns(2)
p1.button("Strict", on_click=apply_settings, args=({**DEFAULTS, **PRESETS["Strict"]},),
          width="stretch", key="preset_strict", help="Higher-conviction cutoffs")
p2.button("Broad", on_click=apply_settings, args=({**DEFAULTS, **PRESETS["Broad"]},),
          width="stretch", key="preset_broad", help="Cast a wider net")

sb.subheader("1 · Universe")
sb.multiselect("Sectors", ALL_SECTORS, key="sectors",
               help="Real Estate and Utilities start excluded: their asset-heavy, regulated "
                    "balance sheets make accrual and Altman scores hard to compare.")
sb.select_slider("Minimum market cap", CAP_STEPS, key="min_cap",
                 format_func=lambda v: "All sizes" if v == 0 else f"≥ ${v:,.3g}B")

sb.subheader("2 · SELL: red flags")
sb.radio("Compare accruals with", ["Sector-relative", "Absolute"], key="benchmark", horizontal=True,
         help="Sector-relative subtracts each sector's average first (the course's demeaned "
              "ranking), so asset-heavy industries are not flagged just for their business model.")
sb.slider("STA percentile cutoff (accruals)", 50, 99, key="sta_pct",
          help="Flag a company when its scaled total accruals are at or above this percentile. The "
               "default 90 flags the top decile, the group Sloan's accrual anomaly study (Chapter 3) "
               "found most overvalued.")
sb.slider("SNOA percentile cutoff (net operating assets)", 50, 99, key="snoa_pct",
          help="Flag a company when its scaled net operating assets are at or above this percentile: "
               "the Chapter 3 'bloated balance sheet' signal.")
sb.slider("Beneish PROBM above", -3.0, -1.0, step=0.01, key="beneish_cut",
          help="Beneish (1999) cutoff −1.78 ≈ a 3.8% manipulation probability.")
sb.slider("Altman Z below", 0.5, 3.5, step=0.01, key="altman_cut",
          help="Course distress boundary 1.81; Altman's original cutoff 2.675.")
sb.slider("Red flags required (of 4)", 1, 4, key="flags_needed")

sb.subheader("3 · SELL: valuation check (CAPM)")
sb.toggle("Require the price to rely on growth", key="valuation_on")
sb.slider("Growth share of price at least (%)", 0, 90, step=5, key="growth_share_min",
          disabled=not st.session_state["valuation_on"],
          help="Share of today's price that current earnings can't support at the CAPM "
               "required return. 50% ≈ a P/E above 22 for an average-risk stock.")
sb.slider("Expected market return E(Rm) (%)", 6.0, 12.0, step=0.25, key="market_return",
          disabled=not st.session_state["valuation_on"],
          help="CAPM: r = 4.09% + β × (E(Rm) − 4.09%). The class workbook assumes 9%.")
sb.radio("Order the SELL list by", SELL_RANKINGS, key="sell_rank")

sb.subheader("4 · BUY: quality at a fair price")
sb.toggle("Exclude companies with any red flag", key="buy_guard")
sb.slider("Piotroski F-score at least", 5, 9, key="fscore_min")
sb.slider("Franchise floor: CFOA and Max-Margin percentile", 0, 90, step=5, key="franchise_floor")
sb.radio("Rank BUY candidates by", BUY_RANKINGS, key="buy_rank")
sb.slider("Weight on profitability vs value", 0.0, 1.0, step=0.05, key="buy_weight",
          disabled=st.session_state["buy_rank"] != BUY_RANKINGS[1])
sb.slider("Number of BUY names", 5, 50, step=5, key="buy_n")

with sb.expander("Advanced"):
    st.radio("Beneish TATA (total accruals) definition", TATA_OPTIONS, key="tata_version",
             help="The course formula uses balance-sheet accruals, which need taxes payable. "
                  "Yahoo leaves that blank for most companies, so by default those use the "
                  "published cash-flow TATA (Beneish, Lee & Nichols 2013), labelled per company.")

settings = {key: st.session_state[key] for key in DEFAULTS}


# ============================================================================
# 10. RUN THE SCREEN AND SHOW THE HEADLINE NUMBERS
# ============================================================================
@st.cache_data
def course_default_lists(_measures, data_version):
    """Counts and SELL tickers under the course defaults, for the 'vs defaults' comparisons."""
    r = run_screen(_measures, DEFAULTS)
    return {"u": len(r["u"]), "sell": r["sell"]["ticker"].tolist(), "watch": len(r["watch"]),
            "buy": len(r["buy"])}


@st.cache_data
def robustness(_measures, data_version, sectors, min_cap):
    """In how many of 6 standard settings is each company a SELL candidate?
    (Course defaults, Strict and Broad, each with sector-relative and absolute accruals.)"""
    counts = pd.Series(0, index=_measures["ticker"].values)
    for preset in ({}, PRESETS["Strict"], PRESETS["Broad"]):
        for benchmark in ("Sector-relative", "Absolute"):
            r = run_screen(_measures, {**DEFAULTS, **preset, "benchmark": benchmark,
                                       "sectors": list(sectors), "min_cap": min_cap})
            counts[r["sell"]["ticker"].values] += 1
    return counts


# The two cached comparisons above skip hashing the big table (the leading underscore), so
# their cache key must change whenever anything that shapes their result changes: the data
# file, the excluded pitch company, the course defaults or the presets.
data_version = (DATA_FILE.stat().st_mtime, PITCH_TICKER, repr(DEFAULTS), repr(PRESETS))
result = run_screen(measures, settings)
baseline = course_default_lists(measures, data_version)
robust_by_ticker = robustness(measures, data_version, tuple(settings["sectors"]), settings["min_cap"])
u, sell, watch, buy = result["u"], result["sell"], result["watch"], result["buy"]

as_of = usable["data_pulled"].dropna().max()
st.markdown(f"""
<div class="pp-hero">
  <div class="pp-eyebrow">ACCT 3343 · Analysis App · Overvalued stock screen</div>
  <div class="pp-title">Paper <span>Profits</span></div>
  <p class="pp-lede">Finds companies whose reported profits run ahead of their cash, whose balance
  sheets are bloating or whose finances are fragile, but whose share prices still assume growth.
  Then, as a separate analysis, the financially strong franchises trading at a fair price.</p>
  <div class="pp-chips"><span>{len(usable):,} US-listed companies</span>
  <span>Yahoo Finance data pulled {as_of}</span><span>Chapters 2–6 + CAPM</span>
  <span>Not investment advice</span></div>
</div>""", unsafe_allow_html=True)


def kpi(column, label, now, then, tip=None):
    """A headline number with its change versus the course-default settings."""
    change = now - then
    column.metric(label, f"{now:,}", f"{change:+,} vs defaults" if change else "same as defaults",
                  delta_color="off", delta_arrow="auto" if change else "off", border=True, help=tip)


k1, k2, k3, k4 = st.columns(4)
kpi(k1, "Companies screened", len(u), baseline["u"])
kpi(k2, "SELL candidates", len(sell), len(baseline["sell"]), "Enough red flags AND priced for growth")
kpi(k3, "Watchlist", len(watch), baseline["watch"],
    "Enough red flags, but current earnings already support most of the price")
kpi(k4, "BUY candidates", len(buy), baseline["buy"],
    "Passed the F-score and franchise gates, ranked on GPA + B/M")

if u.empty:
    st.warning("No companies match the universe filters. Add a sector or lower the market-cap floor.")
    st.stop()

tabs = st.tabs(["Overview", "SELL screen", "BUY screen", "Company deep dive",
                "Investment conclusions", "Methodology & data"])


# ============================================================================
# 11. OVERVIEW TAB
# ============================================================================
with tabs[0]:
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("How the screen works")
        valuation_line = (f"at least **{settings['growth_share_min']}%** of its price must depend on "
                          "growth that current earnings don't show (CAPM)"
                          if settings["valuation_on"] else "valuation check switched off")
        st.markdown(
            f"**SELL = overvalued.** A company needs at least **{settings['flags_needed']} of 4** "
            "red flags: high accruals (STA), a bloated balance sheet (SNOA), a high Beneish "
            f"manipulation score (Chapter 3), or Altman distress (Chapter 4). Then {valuation_line}. "
            f"Flagged companies that fail the price test go on the **Watchlist** instead.\n\n"
            "**BUY = a different analysis.** Companies with no red flags must pass the Chapter 6 "
            f"F-score (≥ {settings['fscore_min']}) and Chapter 5 franchise gates, then are ranked "
            "on Chapter 2 gross profitability and book-to-market.")
    with right:
        st.subheader("What changed")
        now_sell = set(sell["ticker"])
        default_sell = set(baseline["sell"])
        added, removed = sorted(now_sell - default_sell), sorted(default_sell - now_sell)
        if not added and not removed:
            st.markdown("The SELL list is the same as under the course defaults.")
        else:
            st.markdown(f"Compared with the course defaults:  \n"
                        f"**Entered SELL ({len(added)}):** {', '.join(added[:25]) or 'none'}"
                        f"{' …' if len(added) > 25 else ''}  \n"
                        f"**Left SELL ({len(removed)}):** {', '.join(removed[:25]) or 'none'}"
                        f"{' …' if len(removed) > 25 else ''}")
        last = st.session_state.get("_last_sell")
        if last is not None and last != now_sell:
            st.caption(f"Since your last change: +{len(now_sell - last)} entered, "
                       f"−{len(last - now_sell)} left.")
        st.session_state["_last_sell"] = now_sell

    c1, c2 = st.columns(2)
    c1.plotly_chart(funnel_chart(result["funnel"]["sell"], "SELL funnel"), width="stretch")
    c2.plotly_chart(funnel_chart(result["funnel"]["buy"], "BUY funnel"), width="stretch")

    mix = pd.DataFrame({"Universe": u["sector"].value_counts(normalize=True),
                        "SELL": sell["sector"].value_counts(normalize=True),
                        "BUY": buy["sector"].value_counts(normalize=True)}).fillna(0)
    mix = mix.sort_values("Universe")
    fig = go.Figure()
    for name, color in (("Universe", OTHER_COLOR), ("SELL", SELL_COLOR), ("BUY", BUY_COLOR)):
        fig.add_trace(go.Bar(y=mix.index, x=mix[name], name=name, orientation="h",
                             marker_color=color, marker_cornerradius=3,
                             hovertemplate="%{y}: %{x:.0%}<extra>" + name + "</extra>"))
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
    fig.update_xaxes(tickformat=".0%", title="Share of the group")
    st.plotly_chart(finish(fig, "Sector mix: where the SELL and BUY names come from", 460),
                    width="stretch")


# ============================================================================
# 12. SELL TAB
# ============================================================================
with tabs[1]:
    st.subheader(f"SELL candidates ({len(sell)})")
    st.caption(f"At least {settings['flags_needed']} of 4 red flags"
               + (f" and a growth share of price of at least {settings['growth_share_min']}%"
                  if settings["valuation_on"] else "")
               + f". Ordered by: {settings['sell_rank'][0].lower() + settings['sell_rank'][1:]}.")
    if sell.empty:
        st.info("No company meets the SELL rule with these settings. Try the Broad preset or "
                "lower the number of red flags required.")
    else:
        top = sell.head(15)
        vs = " vs sector" if settings["benchmark"] == "Sector-relative" else ""
        columns = [f"Accruals{vs} (STA)", f"NOA{vs} (SNOA)", "Beneish PROBM", "Distress (low Z)",
                   "Growth share of price"]
        concern = pd.DataFrame(dict(zip(columns, [
            top["STA_pct"], top["SNOA_pct"], percentile(u["PROBM_used"]).reindex(top.index),
            (100 - percentile(u["Altman_Z"])).reindex(top.index),
            percentile(u["growth_share"]).reindex(top.index)])))
        labels = pd.DataFrame(dict(zip(columns, [
            top["STA"].map(fmt_pct), top["SNOA"].map(lambda v: fmt_pct(v, 0)),
            top["PROBM_used"].map(fmt_num), top["Altman_Z"].map(fmt_num),
            top["growth_share"].map(lambda v: fmt_pct(v, 0))])))
        heat = go.Figure(go.Heatmap(
            z=concern.values, x=concern.columns, y=top["ticker"], text=labels.values,
            texttemplate="%{text}", colorscale="Oranges", zmin=0, zmax=100, xgap=2, ygap=2,
            colorbar=dict(title="Concern<br>percentile"),
            hovertemplate="%{y} · %{x}<br>value %{text}<br>concern percentile %{z:.0f}<extra></extra>"))
        heat.update_yaxes(autorange="reversed")
        heat.update_xaxes(side="top")
        heat = finish(heat, "Red-flag heatmap: top 15 SELL names", 170 + 32 * len(top))
        st.plotly_chart(heat.update_layout(margin=dict(l=10, r=10, t=110, b=10)), width="stretch")
        st.caption("Colour = how concerning each measure is compared with every company in scope "
                   "(percentile; darker = worse" + (", after subtracting each sector's average" if vs else "")
                   + "). The number in each cell is the company's actual value.")

        table = pd.DataFrame({
            "Rank": range(1, len(sell) + 1), "Ticker": sell["ticker"], "Company": sell["company_name"],
            "Sector": sell["sector"], "Red flags": sell["red_flags"],
            "Assessable": sell["flags_assessable"],
            **{label: sell[flag] for flag, label in FLAGS.items()},
            "STA %": sell["STA"] * 100, "SNOA %": sell["SNOA"] * 100,
            "PROBM": sell["PROBM_used"], "Altman Z": sell["Altman_Z"],
            "Growth share %": sell["growth_share"] * 100, "Implied growth %": sell["implied_growth"] * 100,
            "Revenue CAGR 3y %": sell["revenue_cagr_3y"] * 100,
            "Short % float": sell["current_short_percent_of_float"] * 100,
            "12-mo return %": sell["return_12m"] * 100,
            "Market cap $B": sell["current_market_cap"] / 1e9})
        number = st.column_config.NumberColumn
        st.dataframe(table, hide_index=True, width="stretch", height=min(38 + 35 * len(table), 560),
                     column_config={
                         **{label: st.column_config.CheckboxColumn(label, width="small")
                            for label in FLAGS.values()},
                         "STA %": number(format="%.1f%%"), "SNOA %": number(format="%.0f%%"),
                         "PROBM": number(format="%.2f"), "Altman Z": number(format="%.2f"),
                         "Growth share %": number(format="%.0f%%"),
                         "Implied growth %": number(format="%.1f%%"),
                         "Revenue CAGR 3y %": number(format="%.1f%%"),
                         "Short % float": number(format="%.1f%%"),
                         "12-mo return %": number(format="%.0f%%"),
                         "Market cap $B": number(format="$%.1f")})
        st.download_button("Download the SELL list (CSV)", table.to_csv(index=False),
                           file_name="sell_candidates.csv", mime="text/csv")

    axes = {"Accruals STA percentile": "STA_pct", "Accruals STA": "STA", "SNOA percentile": "SNOA_pct",
            "Beneish PROBM": "PROBM_used", "Altman Z": "Altman_Z", "Growth share of price": "growth_share",
            "12-month return": "return_12m"}
    a1, a2 = st.columns(2)
    x_name = a1.selectbox("Chart x-axis", list(axes), index=0, key="sell_x")
    y_name = a2.selectbox("Chart y-axis", list(axes), index=5, key="sell_y")
    lines = {"STA_pct": settings["sta_pct"], "SNOA_pct": settings["snoa_pct"],
             "PROBM_used": settings["beneish_cut"], "Altman_Z": settings["altman_cut"],
             "growth_share": settings["growth_share_min"] / 100}
    st.plotly_chart(group_scatter(u, axes[x_name], axes[y_name], x_name, y_name,
                                  f"{y_name} vs {x_name}: every company in scope",
                                  hline=lines.get(axes[y_name]), vline=lines.get(axes[x_name]),
                                  y_pct=axes[y_name] in ("growth_share", "return_12m")),
                    width="stretch")
    st.caption("Grey lines mark the current cutoffs. The view covers the middle 96% of companies; "
               "drag on the chart to zoom, double-click to reset. Hover over a dot for its name.")

    with st.expander(f"Watchlist ({len(watch)}): red flags, but the price already looks reasonable"):
        st.dataframe(pd.DataFrame({
            "Ticker": watch["ticker"], "Company": watch["company_name"], "Sector": watch["sector"],
            "Red flags": watch["red_flags"], "Growth share %": watch["growth_share"] * 100,
            "Earnings yield %": watch["earnings_yield"] * 100}),
            hide_index=True, width="stretch",
            column_config={"Growth share %": st.column_config.NumberColumn(format="%.0f%%"),
                           "Earnings yield %": st.column_config.NumberColumn(format="%.1f%%")})


# ============================================================================
# 13. BUY TAB
# ============================================================================
with tabs[2]:
    st.subheader(f"BUY candidates ({len(buy)})")
    rules = ["No red flags"] if settings["buy_guard"] else ["Not a SELL candidate"]
    rules += [f"F-score ≥ {settings['fscore_min']}",
              f"CFOA and Max-Margin at or above the {ordinal(max(settings['franchise_floor'], 1))} "
              "percentile", f"ranked by {settings['buy_rank'][0].lower() + settings['buy_rank'][1:]}"]
    st.caption(" · ".join(rules) + ".")
    if buy.empty:
        st.info("No company passes every BUY gate. Lower the F-score minimum or the franchise floor.")
    else:
        number = st.column_config.NumberColumn
        buy_table = pd.DataFrame({
            "Rank": range(1, len(buy) + 1), "Ticker": buy["ticker"], "Company": buy["company_name"],
            "Sector": buy["sector"], "F-score": buy["F_Score"], "CFOA": buy["CFOA"],
            "CFOA pct": buy["CFOA_pct"], "Max-Margin pct": buy["Max_Margin"],
            "GPA": buy["GPA"], "B/M (fiscal)": buy["BM"], "B/M (today)": buy["BM_today"],
            "ROC after tax %": buy["ROC_AfterTax"] * 100,
            "Market cap $B": buy["current_market_cap"] / 1e9})
        st.dataframe(buy_table, hide_index=True, width="stretch",
                     column_config={
                         "F-score": number(format="%d"), "CFOA": number(format="%.2f"),
                         "CFOA pct": number(format="%.0f"), "Max-Margin pct": number(format="%.0f"),
                         "GPA": number(format="%.2f"), "B/M (fiscal)": number(format="%.2f"),
                         "B/M (today)": number(format="%.2f"), "ROC after tax %": number(format="%.1f%%"),
                         "Market cap $B": number(format="$%.1f")})
        st.download_button("Download the BUY list (CSV)", buy_table.to_csv(index=False),
                           file_name="buy_candidates.csv", mime="text/csv")

    b1, b2 = st.columns(2)
    quality = u.assign(GPA_pct=percentile(u["GPA"]), BM_pct=percentile(u["BM"]))
    b1.plotly_chart(group_scatter(quality, "BM_pct", "GPA_pct", "Book-to-market percentile",
                                  "Gross profitability percentile",
                                  "Quality (GPA) vs price (B/M): top-right is cheap and profitable",
                                  hline=50, vline=50), width="stretch")
    scores = u["F_Score"].dropna().astype(int).value_counts().reindex(range(10), fill_value=0)
    passing = scores.index >= settings["fscore_min"]
    fig = go.Figure(go.Bar(x=scores.index, y=scores.values, marker_cornerradius=4,
                           marker_color=np.where(passing, BUY_COLOR, OTHER_COLOR),
                           hovertemplate="F-score %{x}: %{y} companies<extra></extra>"))
    fig.update_xaxes(title="Piotroski F-score (complete scores only)", dtick=1)
    fig.update_yaxes(title="Companies")
    b2.plotly_chart(finish(fig, f"F-score distribution (blue = passes ≥ {settings['fscore_min']})",
                           430), width="stretch")


# ============================================================================
# 14. COMPANY DEEP DIVE TAB
# ============================================================================
with tabs[3]:
    by_ticker = u.set_index("ticker")
    options = u.sort_values("company_name")["ticker"].tolist()
    if st.session_state.get("company") not in options:
        st.session_state["company"] = sell["ticker"].iloc[0] if not sell.empty else options[0]
    pick = st.selectbox("Choose a company", options, key="company",
                        format_func=lambda t: f"{t} · {by_ticker.at[t, 'company_name']}")
    row = by_ticker.loc[pick]
    status = ("SELL" if row["SELL"] else "Watchlist" if row["WATCH"] else
              "BUY" if row["BUY"] else "Not on either list")

    st.markdown(f"### {row['company_name']} ({pick})")
    st.caption(f"{row['sector']} · {row['industry']} · market cap {fmt_money(row['current_market_cap'])}"
               f" · latest fiscal year ended {row['fiscal_period_end']:%b %d, %Y}")
    with st.container(horizontal=True):
        st.badge(status, color={"SELL": "orange", "Watchlist": "violet", "BUY": "blue"}.get(status, "gray"))
        st.badge(f"{int(row['red_flags'])} red flags ({int(row['flags_assessable'])} of 4 assessable)",
                 color="gray")
        st.badge(f"SELL in {int(robust_by_ticker.get(pick, 0))} of 6 standard settings", color="gray",
                 help="Course defaults, Strict and Broad, each with sector-relative and absolute "
                      "accrual percentiles, using your current universe.")

    reasons = why_flagged(row, settings)
    if reasons:
        st.markdown("\n".join(f"- {r}" for r in reasons))

    m = st.columns(6)
    m[0].metric("STA", fmt_pct(row["STA"]), help=row["STA_reason"] or None)
    m[1].metric("SNOA", fmt_pct(row["SNOA"], 0), help=row["SNOA_reason"] or None)
    m[2].metric("Beneish PROBM", fmt_num(row["PROBM_used"]),
                help=row["TATA_source"] or row["PMAN_reason"] or None)
    m[3].metric("Altman Z", fmt_num(row["Altman_Z"]), help=row["Altman_reason"] or row["Altman_zone"])
    m[4].metric("F-score", "n/a" if pd.isna(row["F_Score"]) else f"{int(row['F_Score'])} / 9",
                help=row["F_reason"] or None)
    m[5].metric("Growth share of price", fmt_pct(row["growth_share"], 0))

    history = [y for y in reversed(YEARS) if pd.notna(row["fiscal_period_end" + y])]
    years = [f"FY{int(row['fiscal_year' + y])}" for y in history]
    scale, unit = (1e9, "$B") if row[["net_income", "operating_cash_flow"]].abs().max() >= 1e9 else (1e6, "$M")
    d1, d2 = st.columns(2)
    fig = go.Figure()
    for column, label, color in (("net_income", "Net income", SELL_COLOR),
                                 ("operating_cash_flow", "Operating cash flow", BUY_COLOR)):
        fig.add_trace(go.Bar(x=years, y=[row[column + y] / scale for y in history], name=label,
                             marker_color=color, marker_cornerradius=4,
                             hovertemplate="%{x}: %{y:,.2f}" + unit + "<extra>" + label + "</extra>"))
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
    fig.update_yaxes(title=unit)
    d1.plotly_chart(finish(fig, "Profit vs cash: is net income backed by operating cash flow?"),
                    width="stretch")
    margins = [safe_divide(pd.Series([row["total_revenue" + y] - row["cost_of_revenue" + y]]),
                           pd.Series([row["total_revenue" + y]]), True).iloc[0] for y in history]
    fig = go.Figure(go.Scatter(x=years, y=margins, mode="lines+markers", line=dict(color=BUY_COLOR, width=2),
                               marker=dict(size=9), name="Gross margin",
                               hovertemplate="%{x}: %{y:.1%}<extra></extra>"))
    fig.update_yaxes(tickformat=".0%", title="Gross margin")
    d2.plotly_chart(finish(fig, "Gross margin trend (Chapter 5 franchise test)"), width="stretch")

    d3, d4 = st.columns(2)
    if pd.notna(row["Altman_Z"]):
        parts = {"1.2 × working capital / TA": 1.2 * row["X1"], "1.4 × retained earnings / TA": 1.4 * row["X2"],
                 "3.3 × EBIT / TA": 3.3 * row["X3"], "0.6 × market equity / liabilities": 0.6 * row["X4"],
                 "0.999 × sales / TA": 0.999 * row["X5"]}
        fig = go.Figure(go.Bar(x=list(parts.values()), y=list(parts), orientation="h", marker_color=BUY_COLOR,
                               marker_cornerradius=4, hovertemplate="%{y}: %{x:.2f}<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(title="Contribution to Z")
        d3.plotly_chart(finish(fig, f"Altman Z = {row['Altman_Z']:.2f}: what drives it ({row['Altman_zone']})"),
                        width="stretch")
    else:
        d3.info(f"Altman Z unavailable: {row['Altman_reason']}.")
    indices = [k for k in BENEISH_MEANS if pd.notna(row[k])]
    if indices:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[BENEISH_MEANS[k][1] for k in indices], y=indices, mode="markers",
                                 name="Non-manipulators (mean)", marker=dict(color=OTHER_COLOR, size=11)))
        fig.add_trace(go.Scatter(x=[BENEISH_MEANS[k][0] for k in indices], y=indices, mode="markers",
                                 name="Manipulators (mean)", marker=dict(color="#52514e", size=11, symbol="x")))
        fig.add_trace(go.Scatter(x=[row[k] for k in indices], y=indices, mode="markers", name=pick,
                                 marker=dict(color=SELL_COLOR, size=13, symbol="diamond")))
        fig.update_xaxes(title="Index value (1.0 = no change from last year)")
        fig.update_yaxes(autorange="reversed")
        d4.plotly_chart(finish(fig, "Beneish indices vs Beneish (1999) sample means"), width="stretch")
    else:
        d4.info("Beneish indices unavailable for this company.")

    t1, t2 = st.columns(2)
    with t1:
        st.markdown("**Piotroski F-score checklist (Chapter 6)**")
        checks = [("F_ROA", "ROA > 0", row["ROA"], "pct"), ("F_CFO", "CFO / assets > 0", row["CFO_to_assets"], "pct"),
                  ("F_DROA", "ROA improved", row["dROA"], "pct"),
                  ("F_ACCRUAL", "CFO > ROA (cash-backed earnings)", row["CFO_to_assets"] - row["ROA"], "pct"),
                  ("F_DLEV", "Leverage fell", row["dLeverage"], "pct"),
                  ("F_DLIQ", "Current ratio rose", row["dLiquidity"], "num"),
                  ("F_EQ", "No net share issuance", row["net_issuance"], "money"),
                  ("F_DGM", "Gross margin rose", row["dGrossMargin"], "pct"),
                  ("F_DAT", "Asset turnover rose", row["dTurnover"], "num")]
        show = {"pct": lambda v: fmt_pct(v, 2), "num": fmt_num, "money": fmt_money}
        st.dataframe(pd.DataFrame({
            "Signal": [c[1] for c in checks],
            "Value": [show[c[3]](c[2]) for c in checks],
            "Point": ["✓ 1" if row[c[0]] == 1 else "✗ 0" if row[c[0]] == 0 else "? unknown" for c in checks]}),
            hide_index=True, width="stretch")
    with t2:
        st.markdown("**CAPM valuation check**")
        pe = 1 / row["earnings_yield"] if row["earnings_yield"] > 0 else np.nan
        st.dataframe(pd.DataFrame({"Item": [
            "Beta", "Required return r = 4.09% + β × (E(Rm) − 4.09%)", "Earnings yield E/P (P/E)",
            "Growth share of price = 1 − (E/P) / r", "Implied perpetual growth g* = r − E/P",
            "Actual revenue growth (3-year CAGR)", "Short interest (% of float)", "12-month return"],
            "Value": [f"{fmt_num(row['beta'])} ({row['beta_source']})", fmt_pct(row["required_return"]),
                      f"{fmt_pct(row['earnings_yield'])} ({fmt_num(pe, 1)}×)", fmt_pct(row["growth_share"], 0),
                      fmt_pct(row["implied_growth"]), fmt_pct(row["revenue_cagr_3y"]),
                      fmt_pct(row["current_short_percent_of_float"]), fmt_pct(row["return_12m"], 0)]}),
            hide_index=True, width="stretch")


# ============================================================================
# 15. INVESTMENT CONCLUSIONS TAB (my written case; the evidence updates live)
# ============================================================================
# Each side shows only its own measures, so the cards respect rule 2 too.
EVIDENCE = {
    "SELL": [("STA", lambda r: fmt_pct(r["STA"])), ("SNOA", lambda r: fmt_pct(r["SNOA"], 0)),
             ("Altman Z", lambda r: fmt_num(r["Altman_Z"])),
             ("Growth share", lambda r: fmt_pct(r.get("growth_share", np.nan), 0)),
             ("Short % float", lambda r: fmt_pct(r["current_short_percent_of_float"])),
             ("12-mo return", lambda r: fmt_pct(r["return_12m"], 0))],
    "BUY": [("F-score", lambda r: "n/a" if pd.isna(r["F_Score"]) else f"{int(r['F_Score'])} / 9"),
            ("CFOA", lambda r: fmt_num(r["CFOA"])),
            ("Max-Margin", lambda r: fmt_num(r.get("Max_Margin", np.nan), 0)),
            ("GPA", lambda r: fmt_num(r["GPA"])), ("B/M (fiscal)", lambda r: fmt_num(r["BM"])),
            ("ROC after tax", lambda r: fmt_pct(r["ROC_AfterTax"], 0))],
}

with tabs[4]:
    st.subheader("Investment conclusions")
    st.caption("My written case for the top three SELL and BUY names under the course-default "
               "settings. The numbers in each card are calculated live, and the badge shows whether "
               "the company is still selected under the settings chosen in the sidebar.")
    st.info(f"**Live view, under your current sidebar settings:** top SELL names "
            f"{', '.join(sell['ticker'].head(3)) or 'none'}; top BUY names "
            f"{', '.join(buy['ticker'].head(3)) or 'none'}. The written cases below were prepared "
            "for the course-default settings.")
    all_firms = measures.set_index("ticker")
    for side in ("SELL", "BUY"):
        st.markdown(f"### {side} picks")
        for rank, item in enumerate(CONCLUSIONS[side], start=1):
            ticker = item["ticker"]
            in_scope = ticker in by_ticker.index
            r = by_ticker.loc[ticker] if in_scope else all_firms.loc[ticker]
            still = in_scope and bool(r[side])
            with st.expander(f"{side} #{rank} · {r['company_name']} ({ticker}) · {item['verdict']}",
                             expanded=rank == 1):
                with st.container(horizontal=True):
                    st.badge(item["verdict"], color="orange" if side == "SELL" else "blue")
                    st.badge(f"Still a {side} under current settings" if still
                             else f"Not a {side} under current settings", color="gray")
                cells = [cell for _ in range(2) for cell in st.columns(3)]   # 2 rows of 3, so numbers fit
                for column, (label, value) in zip(cells, EVIDENCE[side]):
                    column.metric(label, value(r))
                st.markdown(item["text"].replace("$", "\\$"))   # "$" would start a math formula
    st.markdown("### Putting it together")
    st.markdown(CONCLUSIONS["portfolio"].replace("$", "\\$"))


# ============================================================================
# 16. METHODOLOGY & DATA TAB
# ============================================================================
def formula_checks():
    """Re-run the course's worked examples through this app's own functions."""
    def run(values):
        frame = pd.DataFrame([{**dict.fromkeys(RAW_COLUMNS, np.nan), "ticker": "CHECK",
                               "sector": "Consumer Defensive", **values}])
        return compute_measures(frame).iloc[0]

    empire2 = run(dict(total_assets=17080, gross_profit=32167 - 23469, stockholders_equity=5003,
                       ordinary_shares_number=10299, fiscal_period_end_close=1.0))
    empire3 = run(dict(net_income=219_000, operating_cash_flow=2_099_000, total_assets=17_080_000,
                       cash_and_cash_equivalents=291_000, current_debt_and_capital_lease_obligation=831_000,
                       long_term_debt_and_capital_lease_obligation=7_026_000, minority_interest=149_000,
                       preferred_stock=0, common_stock_equity=5_003_000, stockholders_equity=5_003_000))
    empire4 = run(dict(working_capital=-738, retained_earnings=3334, ebit=625, ordinary_shares_number=10793,
                       fiscal_period_end_close=1.0, total_liabilities_net_minority_interest=11929,
                       total_revenue=32167, total_assets=17080))
    empire6 = run(dict(fiscal_period_end="2025-05-03", fiscal_year=2025, fiscal_period_end_prev="2024-05-04",
                       fiscal_year_prev=2024, fiscal_period_end_prev2="2023-05-06", fiscal_year_prev2=2023,
                       net_income=198, operating_cash_flow=1911, total_assets=17080, total_assets_prev=17019,
                       total_assets_prev2=16790, net_income_prev=0.0417 * 16790, long_term_debt=7026,
                       long_term_debt_prev=6642, current_assets=3390, current_liabilities=4128,
                       current_assets_prev=3134, current_liabilities_prev=4096, total_revenue=31950,
                       cost_of_revenue=31950 - 8664, total_revenue_prev=30733,
                       cost_of_revenue_prev=30733 * (1 - 8382 / 31277), net_common_stock_issuance=-1.0))
    apple = measures.loc[measures["ticker"].eq("AAPL"), "beta_60m"]
    apple_r = capm_required_return(pd.Series([1.0704]), 0.09).iloc[0]
    apple_beta = f"{apple.iloc[0]:.4f}" if len(apple) and pd.notna(apple.iloc[0]) else "n/a"
    rows = [("Ch 2 · Empire GPA", "0.509", f"{empire2['GPA']:.3f}"),
            ("Ch 2 · Empire B/M", "0.486", f"{empire2['BM']:.3f}"),
            ("Ch 3 · Empire STA", "−11%", fmt_pct(empire3["STA"], 0)),
            ("Ch 3 · Empire SNOA", "74%", fmt_pct(empire3["SNOA"], 0)),
            ("Ch 4 · Empire Altman Z (notebook scaling; slide shows 1.89, see note)", "2.77",
             f"{empire4['Altman_Z']:.2f}"),
            ("Ch 6 · Empire F-score", "7 of 9", f"{empire6['F_Score']:.0f} of 9"),
            ("CAPM · Apple 60-month beta (workbook)", "1.0704", apple_beta),
            ("CAPM · Apple required return at E(Rm) = 9% (workbook)", "9.35%", fmt_pct(apple_r, 2))]
    checks = pd.DataFrame(rows, columns=["Course worked example", "Course answer", "This app's functions"])
    checks["Match"] = np.where(checks["Course answer"] == checks["This app's functions"], "✓", "✗")
    return checks


with tabs[5]:
    st.subheader("Data")
    excluded = measures["exclusion_reason"].value_counts()
    st.markdown(
        f"- **Source:** Yahoo Finance via `yfinance`, pulled {as_of}. Statements are the latest four "
        "annual fiscal years; prices are as of the pull date. The app only reads `universe.csv`; it never "
        "downloads data.\n"
        "- **Universe:** the instructor's Yahoo screener query (US-listed, NYSE and Nasdaq, market cap "
        f"≥ $300M), largest first. Financial Services firms ({FINANCIALS_EXCLUDED:,} at the pull) are "
        "left out of the query because accrual, Beneish and Altman models do not apply to banks.\n"
        "- **Default scope:** Real Estate and Utilities start outside the screen because their asset-heavy, "
        "regulated balance sheets make accrual (SNOA) and Altman scores hard to compare with other sectors. "
        "Add them back in the sidebar's Sectors box.\n"
        f"- **{len(measures):,} companies in the file → {len(usable):,} usable.** Excluded, with reasons:")
    st.dataframe(excluded.rename_axis("Reason").reset_index(name="Companies"), hide_index=True)

    st.subheader("Rule 2: SELL and BUY use different analyses")
    st.markdown(
        "| Side | Measures | Course chapters | Direction |\n|---|---|---|---|\n"
        "| **SELL** | STA, SNOA, Beneish PMAN, Altman Z, CAPM growth share | Ch 3, Ch 4, CAPM | high accruals, "
        "bloated NOA, high PROBM, low Z, price relies on growth |\n"
        "| **BUY** | Piotroski F-score, CFOA, Max-Margin (margin growth/stability), GPA, B/M | Ch 6, Ch 5, Ch 2 "
        "| strong, persistent and cheap |\n\n"
        "No measure is used on both sides. The F-score includes one accrual test (CFO > ROA) among its "
        "nine signals; the BUY side never ranks on accruals. Excluding red-flagged companies from BUY is a "
        "consistency guard (as in the class example app), not the BUY analysis, and it can be switched off.")

    st.subheader("Formulas (as implemented in the course notebooks)")
    st.markdown("**Ch 3 · Scaled total accruals** (higher = earnings less backed by cash)")
    st.latex(r"STA = \frac{NI - CFO}{TA}")
    st.markdown("**Ch 3 · Scaled net operating assets** (higher = more accumulated operating assets)")
    st.latex(r"SNOA = \frac{(TA - Cash) - (TA - STD - LTD - MI - PS - CE)}{TA}")
    st.markdown("**Ch 3 · Beneish probability of manipulation** (TATA from balance-sheet accrual changes)")
    st.latex(r"PROBM = -4.84 + 0.920\,DSRI + 0.528\,GMI + 0.404\,AQI + 0.892\,SGI + 0.115\,DEPI"
             r" - 0.172\,SGAI + 4.679\,TATA - 0.327\,LVGI \qquad PMAN = \Phi(PROBM)")
    st.markdown("**Ch 4 · Altman Z** (X1–X4 in percentage points, as in the Chapter 4 notebook; "
                "market equity at the fiscal year-end)")
    st.latex(r"Z = 0.012X_1 + 0.014X_2 + 0.033X_3 + 0.006X_4 + 0.999X_5")
    st.caption("X1 = working capital/TA, X2 = retained earnings/TA, X3 = EBIT/TA, X4 = market equity/"
               "book liabilities, X5 = sales/TA. Equivalent to 1.2, 1.4, 3.3, 0.6 on decimal ratios. "
               "Distress below 1.81; Altman's original cutoff 2.675.")
    st.markdown("**CAPM valuation check** (class CAPM workbook: β from 60 monthly returns vs the S&P 500)")
    st.latex(r"r = R_f + \beta\,(E[R_m] - R_f) \qquad \text{Growth share} = 1 - \frac{E/P}{r} = "
             r"\frac{g^*}{r},\quad g^* = r - E/P")
    st.markdown("**Ch 6 · Piotroski F-score:** nine 0/1 tests (ROA > 0, CFO > 0, ΔROA > 0, CFO > ROA, "
                "Δleverage < 0, Δcurrent ratio > 0, no net issuance, Δgross margin > 0, Δturnover > 0).")
    st.markdown("**Ch 5 · Franchise**")
    st.latex(r"CFOA = \frac{\sum_{4\,yrs} FCF}{TA} \qquad MG = \left(\frac{GM_{latest}}{GM_{first}}\right)^{1/3} - 1"
             r" \qquad MS = \frac{\overline{GM}}{\sigma_{GM}}")
    st.caption("Max-Margin = the higher of the margin-growth and margin-stability percentiles (textbook).")
    st.markdown("**Ch 2 · Quality and price**")
    st.latex(r"GPA = \frac{Revenue - COGS}{TA} \qquad B/M = \frac{Book\ equity}{Market\ equity}"
             r" \qquad \text{BUY rank} = \operatorname{rank}(\operatorname{rank}_{GPA} + \operatorname{rank}_{B/M})")

    st.subheader("Missing data: coverage of each measure")
    coverage = []
    for label, column, reason in [("STA", "STA", "STA_reason"), ("SNOA", "SNOA", "SNOA_reason"),
                                  ("Beneish PMAN (course TATA)", "PMAN", "PMAN_reason"),
                                  ("Beneish PMAN (cash-flow TATA)", "PMAN_CF", None),
                                  ("Beneish PMAN (as used now: course + fallback)", "PMAN_used", None),
                                  ("Altman Z", "Altman_Z", "Altman_reason"), ("CFOA", "CFOA", "Franchise_reason"),
                                  ("Margin growth", "Margin_Growth", None), ("Margin stability", "Margin_Stability", None),
                                  ("F-score (complete)", "F_Score", "F_reason"), ("GPA", "GPA", None),
                                  ("B/M (fiscal date)", "BM", None), ("Beta", "beta", None)]:
        frame = usable if column in usable.columns else u    # "as used now" exists only in scope
        valid = frame[column].notna()
        top_reason = (frame.loc[~valid & frame[reason].astype(str).ne(""), reason].value_counts()
                      .index[:1].tolist() if reason else [])
        coverage.append({"Measure": label, "Companies with a value": int(valid.sum()),
                         "Coverage": valid.mean() * 100, "Most common reason when missing":
                         top_reason[0] if top_reason else ""})
    st.dataframe(pd.DataFrame(coverage), hide_index=True, width="stretch",
                 column_config={"Coverage": st.column_config.ProgressColumn(
                     min_value=0, max_value=100, format="%.0f%%")})
    st.caption("Coverage is out of all usable companies (the 'as used now' row: companies in scope). "
               "Missing inputs are never replaced with zero. A company cannot be flagged by a measure it "
               "has no data for; each company shows how many of the 4 red flags could be assessed.")

    st.subheader("Formula checks: the course's worked examples, re-run live by this app's functions")
    st.dataframe(formula_checks(), hide_index=True, width="stretch")
    st.caption("Empire Co. Altman Z: the slide's 1.89 applies the 0.012…0.006 coefficients to decimal "
               "ratios. The Chapter 4 notebook expresses X1–X4 in percentage points, which gives 2.77 "
               "(the slide also rounds X3 to 0.038; it is 0.0366). The Chapter 6 slide's CFO/assets of "
               "1.12% is 11.23%. Every measure also matches the instructor's notebooks to 12 decimal places "
               "on the class dataset (automated tests).")

    st.subheader("Limitations")
    st.markdown(
        "- Yahoo provides about five years of annual statements, so the Chapter 5 franchise measures use "
        "four years (three growth periods) instead of the textbook's eight.\n"
        "- Yahoo leaves taxes payable blank for most companies, so the course's balance-sheet Beneish TATA "
        "can be computed for only about one in five. By default those companies use the published cash-flow "
        "TATA (Beneish, Lee & Nichols 2013), labelled per company; 'Course formula only' is under Advanced.\n"
        "- Statements may be restated and carry no filing dates, so this is a current screen, not a "
        "point-in-time backtest.\n"
        "- Market equity uses fiscal-year-end shares and price, as in the course. It is left missing when "
        "the statement share count doesn't match today's market cap (ADRs, post-year-end splits).\n"
        "- A red flag starts an investigation; it does not prove manipulation or predict bankruptcy.")

st.markdown(f'<div class="pp-footer">Paper Profits · ACCT 3343 Analysis App · data: Yahoo Finance via '
            f'yfinance ({as_of}) · methods: <i>Quantitative Value</i> (Gray &amp; Carlisle), Chapters 2–6, '
            f'and the class CAPM workbook · for coursework, not investment advice</div>', unsafe_allow_html=True)
