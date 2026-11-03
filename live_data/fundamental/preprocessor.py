"""
Quarterly Fundamental Stock Pre-Processor  (corrected)
======================================================

Same I/O contract as the original. Changes are marked `# FIX:` and are confined to
constants + build_stock_quarter_record. Summary of fixes:

  1. negative_equity_distortion: keyed off equity SIGN (and a small-equity guard for
     non-financials), NOT off de>500 / roe>300. Those branches were circular and fired
     for normally-levered banks (GS, MS) and real high-ROE names.
  2. ev_sales: status now not_applicable_financials for banks (margin/EV multiples don't
     translate) — was only flagged for ev_ebitda.
  3. earnings_quality (opcf/fcf/accruals/capex): status not_meaningful_financials for banks
     — cash-conversion is structurally noisy for financials (BAC opcf -35x is not a signal).
  4. interest_coverage: status "net_cash" ONLY when net debt is actually negative; a missing
     value is "not_available", not strength.
  5. eps_revision_3m: both endpoints pinned to the SAME series (BEST_EPS_MEDIAN) and guarded
     against a non-positive base (revision off negative EPS is meaningless).
  6. extreme_pe: also true for negative forward P/E (loss-making forward — P/E meaningless).
  7. sector_benchmark_caveat: surfaces the two benchmark-reliability problems —
     Communication Services before the 2018 GICS reclass (telecom proxy), and the payment
     networks (V/MA) measured against the bank-dominated Financials benchmark.
"""

import argparse
import json
import math
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


warnings.filterwarnings(
    "ignore",
    message=r"Cell .* is marked as a date but the serial value .* is outside the limits for dates.*",
    category=UserWarning,
    module=r"openpyxl\.worksheet\._reader",
)


EXCEL_EPOCH = datetime(1899, 12, 30)
FINANCIALS_SECTOR = "Financials"

# FIX (7): S&P applied the Communication Services GICS reclassification on 2018-09-28.
# Before this, the S5TELS index (mapped to Communication Services below) was telecom-only,
# so its aggregate fundamentals (high dividend yield, telecom ROE) are NOT a valid benchmark
# for GOOGL/META/NFLX/DIS. Rows dated before this are flagged.
GICS_COMSERV_RECLASS = pd.Timestamp("2018-10-01")

# FIX (7): Visa and Mastercard are tagged Financials in the workbook but are payment networks
# (asset-light, ~50%+ margins, P/B 5-10). They are benchmarked against the bank-dominated
# S5FINL index, so sector-relative valuation/quality reads are a business-model mismatch.
PAYMENT_NETWORK_TICKERS = {"V US EQUITY", "MA US EQUITY"}

# FIX (1): a non-financial whose book equity is below this fraction of assets has ratios that
# explode without being meaningful (e.g. ABBV post-spinoff). Tunable against your distribution.
SMALL_EQUITY_TO_ASSETS = 0.03

SECTOR_INDEX = {
    "Communication Services": "S5TELS Index",
    "Consumer Discretionary": "S5COND Index",
    "Consumer Staples": "S5CONS Index",
    "Energy": "S5ENRS Index",
    "Financials": "S5FINL Index",
    "Health Care": "S5HLTH Index",
    "Industrials": "S5INDU Index",
    "Information Technology": "S5INFT Index",
    "Materials": "S5MATR Index",
    "Real Estate": "S5RLST Index",
    "Utilities": "S5UTIL Index",
}

VALUATION_FIELDS = [
    "PE_RATIO",
    "BEST_PE_RATIO",
    "PX_TO_BOOK_RATIO",
    "BEST_CURRENT_EV_BEST_SALES",
    "BEST_CUR_EV_TO_EBITDA",
    "FREE_CASH_FLOW_YIELD",
    "EQY_DVD_YLD_12M",
    "PX_LAST",
]

MISSING_RATIO_FIELDS = [
    "ENTERPRISE_VALUE",
    "PX_LAST",
    "EBITDA",
    "CUR_MKT_CAP",
    "EQY_SH_OUT",
    "SALES_REV_TURN",
    "BEST_EPS",
    "BEST_PE_RATIO",
    "EQY_DVD_YLD_IND",
    "EQY_DVD_YLD_12M",
    "DIVIDEND_PER_SHARE_12M",
    "TRAIL_12M_EPS",
    "IS_EPS",
    "TOT_DEBT_TO_TOT_EQY",
    "NET_DEBT",
    "TOT_COMMON_EQY",
    "TOT_DEBT_TO_TOT_CAP",
]

QUALITY_FIELDS = [
    "RETURN_ON_ASSET",
    "GROSS_MARGIN",
    "EBITDA_MARGIN",
    "CF_FREE_CASH_FLOW",
    "CF_CASH_FROM_OPER",
    "NET_INCOME",
    "NET_DEBT_TO_EBITDA",
    "INTEREST_COVERAGE_RATIO",
    "CASH_CONVERSION_CYCLE",
    "BS_TOT_ASSET",
    "BS_CASH_NEAR_CASH_ITEM",
    "BS_TOT_LIAB2",
    "SHORT_AND_LONG_TERM_DEBT",
    "RETURN_COM_EQY",
    "TOT_DEBT_TO_TOT_EQY",
    "CUR_RATIO",
    "CF_DEPR_AMORT",
    "CAPITAL_EXPEND",
]

EARNINGS_QUARTERLY_FIELDS = [
    "IS_EPS",
    "SALES_GROWTH",
    "EPS_GROWTH",
    "ANNOUNCEMENT_DT",
]

EARNINGS_MONTHLY_FIELDS = [
    "BEST_ANALYST_RATING",
    "BEST_EPS_MEDIAN",
    "BEST_TARGET_PRICE",
    "TOT_ANALYST_REC",
]

SECTOR_BENCHMARK_FIELDS = [
    "BEST_PE_RATIO",
    "PE_RATIO",
    "PX_TO_BOOK_RATIO",
    "BEST_CUR_EV_TO_EBITDA",
    "BEST_CURRENT_EV_BEST_SALES",
    "FREE_CASH_FLOW_YIELD",
    "EQY_DVD_YLD_12M",
    "PROF_MARGIN",
    "RETURN_COM_EQY",
]


# ---------------------------------------------------------------------
# Basic helpers (unchanged)
# ---------------------------------------------------------------------
def cell_text(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value).strip()


def parse_date(value):
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).normalize()
    if value is None:
        return pd.NaT
    try:
        if pd.isna(value):
            return pd.NaT
    except TypeError:
        pass
    if isinstance(value, (int, float, np.integer, np.floating)):
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return pd.NaT
        iv = int(f)
        if abs(f - iv) < 1e-9 and 19000101 <= iv <= 21001231:
            return pd.to_datetime(str(iv), format="%Y%m%d", errors="coerce")
        if 20000 <= f <= 60000:
            return (pd.Timestamp(EXCEL_EPOCH) + pd.to_timedelta(f, unit="D")).normalize()
        return pd.NaT
    text = str(value).strip()
    if text == "" or text.upper().startswith("#N/A"):
        return pd.NaT
    try:
        f = float(text)
        iv = int(f)
        if abs(f - iv) < 1e-9 and 19000101 <= iv <= 21001231:
            return pd.to_datetime(str(iv), format="%Y%m%d", errors="coerce")
        if 20000 <= f <= 60000:
            return (pd.Timestamp(EXCEL_EPOCH) + pd.to_timedelta(f, unit="D")).normalize()
        return pd.NaT
    except ValueError:
        pass
    return pd.to_datetime(text, errors="coerce").normalize()


def decode_number(value):
    if isinstance(value, (pd.Timestamp, datetime)):
        try:
            return (pd.Timestamp(value).to_pydatetime() - EXCEL_EPOCH).total_seconds() / 86400.0
        except Exception:
            return np.nan
    if value is None:
        return np.nan
    if isinstance(value, str):
        text = value.strip()
        if text == "" or text.upper().startswith("#N/A"):
            return np.nan
        try:
            return float(text)
        except ValueError:
            return np.nan
    try:
        f = float(value)
        return f if not math.isnan(f) else np.nan
    except (TypeError, ValueError):
        return np.nan


def safe_float(value, nd=4):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return round(f, nd)


def safe_int(value):
    v = safe_float(value, 0)
    return None if v is None else int(v)


def json_clean(obj):
    if isinstance(obj, dict):
        return {str(k): json_clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_clean(v) for v in obj]
    if isinstance(obj, (pd.Timestamp, datetime)):
        return pd.Timestamp(obj).strftime("%Y-%m-%d")
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        f = float(obj)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    return obj


# ---------------------------------------------------------------------
# Workbook parsing (unchanged)
# ---------------------------------------------------------------------
def find_security_blocks(raw, ticker_row, include_index=False):
    blocks = []
    for col, value in enumerate(raw.iloc[ticker_row].tolist()):
        text = cell_text(value)
        if "EQUITY" in text or (include_index and "Index" in text):
            blocks.append((col, text))
    return blocks


def detect_ticker_row(raw, include_index=False, max_scan_rows=8):
    best_row = None
    best_count = 0
    for row in range(min(max_scan_rows, raw.shape[0])):
        count = 0
        for value in raw.iloc[row].tolist():
            text = cell_text(value)
            if "EQUITY" in text or (include_index and "Index" in text):
                count += 1
        if count > best_count:
            best_row = row
            best_count = count
    if best_row is None or best_count == 0:
        raise ValueError("Could not find ticker/index header row.")
    return best_row


def detect_field_row(raw, wanted_fields, max_scan_rows=8):
    wanted = set(wanted_fields)
    best_row = None
    best_count = 0
    for row in range(min(max_scan_rows, raw.shape[0])):
        count = 0
        for value in raw.iloc[row].tolist():
            text = cell_text(value)
            if text in wanted or text.upper() in {"DATES", "DATE"}:
                count += 1
        if count > best_count:
            best_row = row
            best_count = count
    if best_row is None or best_count == 0:
        raise ValueError("Could not find Bloomberg field-code row.")
    return best_row


def find_shared_date_col(raw, field_row):
    for col in range(raw.shape[1]):
        if cell_text(raw.iat[field_row, col]).upper() in {"DATES", "DATE"}:
            return col
    return 0


def parse_bloomberg_panel(excel_path, sheet_name, wanted_fields, include_index=False):
    raw = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)
    ticker_row = detect_ticker_row(raw, include_index=include_index)
    field_row = detect_field_row(raw, wanted_fields)
    shared_date_col = find_shared_date_col(raw, field_row)
    blocks = find_security_blocks(raw, ticker_row, include_index=include_index)
    wanted = set(wanted_fields)
    rows = []
    data_start = field_row + 1
    for block_idx, (start_col, security) in enumerate(blocks):
        end_col = blocks[block_idx + 1][0] if block_idx + 1 < len(blocks) else raw.shape[1]
        start_field = cell_text(raw.iat[field_row, start_col]).upper()
        if start_field in {"DATES", "DATE"}:
            date_col = start_col
            first_field_col = start_col + 1
        else:
            date_col = shared_date_col
            first_field_col = start_col
        dates = raw.iloc[data_start:, date_col].apply(parse_date)
        for col in range(first_field_col, end_col):
            field = cell_text(raw.iat[field_row, col])
            if field not in wanted:
                continue
            values = raw.iloc[data_start:, col]
            for dt, raw_value in zip(dates, values):
                if pd.isna(dt):
                    continue
                if field == "ANNOUNCEMENT_DT":
                    value = parse_date(raw_value)
                else:
                    value = decode_number(raw_value)
                rows.append((security.strip(), pd.Timestamp(dt), field, value))
    return pd.DataFrame(rows, columns=["ticker", "date", "field", "value"])


def parse_macro(excel_path):
    raw = pd.read_excel(excel_path, sheet_name="6_Macro", header=None)
    dates = raw.iloc[3:, 0].apply(parse_date)
    rows = []
    for col in range(1, raw.shape[1]):
        name = cell_text(raw.iat[0, col])
        if not name:
            continue
        key = name.replace(" Index", "").replace(" ", "_")
        values = raw.iloc[3:, col]
        for dt, value in zip(dates, values):
            if pd.isna(dt):
                continue
            rows.append((pd.Timestamp(dt), key, decode_number(value)))
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows, columns=["date", "series", "value"])
        .pivot_table(index="date", columns="series", values="value", aggfunc="first")
        .sort_index()
    )


def derive_regimes(macro):
    if macro.empty:
        return pd.DataFrame()
    m = macro.sort_index().copy()
    out = pd.DataFrame(index=m.index)
    ff = m["FEDL01"] if "FEDL01" in m.columns else pd.Series(index=m.index, dtype=float)
    ff_chg_6m = ff - ff.shift(6)

    def rate_label(x):
        if pd.isna(x):
            return None
        if x > 0.25:
            return "tightening"
        if x < -0.25:
            return "easing"
        return "stable"

    out["rate_environment"] = ff_chg_6m.apply(rate_label)
    vix = m["VIX"] if "VIX" in m.columns else pd.Series(index=m.index, dtype=float)
    oas = m["LF98OAS"] if "LF98OAS" in m.columns else pd.Series(index=m.index, dtype=float)

    def risk_label(v, o):
        if pd.isna(v) or pd.isna(o):
            return None
        if v > 25 or o > 5.0:
            return "stressed"
        if v < 16 and o < 3.5:
            return "calm"
        return "normal"

    out["risk_regime"] = [risk_label(v, o) for v, o in zip(vix, oas)]
    pmi = m["NAPMPMI"] if "NAPMPMI" in m.columns else pd.Series(index=m.index, dtype=float)
    y10 = m["USGG10YR"] if "USGG10YR" in m.columns else pd.Series(index=m.index, dtype=float)
    y2 = m["USGG2YR"] if "USGG2YR" in m.columns else pd.Series(index=m.index, dtype=float)
    spread = y10 - y2
    inverted_3m = (spread < 0).rolling(3).sum() >= 3

    def cycle_label(p, inv):
        if pd.isna(p):
            return None
        if p < 48 or inv:
            return "contraction"
        if p > 52:
            return "expansion"
        return "mid_cycle"

    out["cycle"] = [cycle_label(p, inv) for p, inv in zip(pmi, inverted_3m)]
    out["yield_curve_10y2y"] = spread
    return out


def load_all(excel_path):
    tickers = pd.read_excel(excel_path, sheet_name="1_Tickers")
    if "ticker" not in tickers.columns or "sector" not in tickers.columns:
        raise ValueError("Sheet 1_Tickers must have columns named 'ticker' and 'sector'.")
    tickers = tickers[["ticker", "sector"]].copy()
    tickers["ticker"] = tickers["ticker"].astype(str).str.strip()
    tickers["sector"] = tickers["sector"].astype(str).str.strip()
    tickers = tickers[
        (tickers["ticker"] != "") & (tickers["ticker"].str.lower() != "nan")
    ].reset_index(drop=True)
    tickers["stock_id"] = [f"STOCK_{i:03d}" for i in range(1, len(tickers) + 1)]

    valuation = parse_bloomberg_panel(excel_path, "2_Valuation", VALUATION_FIELDS)
    try:
        missing_ratios = parse_bloomberg_panel(excel_path, "missing_ratios", MISSING_RATIO_FIELDS)
    except ValueError:
        missing_ratios = pd.DataFrame(columns=["ticker", "date", "field", "value"])
    quality_q = parse_bloomberg_panel(excel_path, "3_Quality_Quarterly", QUALITY_FIELDS)
    earnings_q = parse_bloomberg_panel(excel_path, "4_Earnings_Quarterly", EARNINGS_QUARTERLY_FIELDS)
    earnings_m = parse_bloomberg_panel(excel_path, "4_Earnings_Monthly", EARNINGS_MONTHLY_FIELDS)
    sector_benchmarks = parse_bloomberg_panel(
        excel_path, "5_Sector_Benchmarks", SECTOR_BENCHMARK_FIELDS, include_index=True
    ).rename(columns={"ticker": "index"})
    macro = parse_macro(excel_path)
    regimes = derive_regimes(macro)
    return (tickers, valuation, missing_ratios, quality_q, earnings_q, earnings_m,
            sector_benchmarks, regimes)


# ---------------------------------------------------------------------
# Time-series helpers (unchanged)
# ---------------------------------------------------------------------
def wide(long_df, field):
    if long_df is None or long_df.empty:
        return pd.DataFrame()
    sub = long_df[long_df["field"] == field]
    if sub.empty:
        return pd.DataFrame()
    return sub.pivot_table(index="date", columns="ticker", values="value", aggfunc="first").sort_index()


def yoy_change(long_df, field):
    w = wide(long_df, field)
    if w.empty:
        return w
    return w - w.shift(4)


def add_eps_cagr(earnings_quarterly):
    eps = wide(earnings_quarterly, "IS_EPS")
    if eps.empty:
        return eps, eps
    ttm = eps.rolling(4).sum()
    cagr = (ttm / ttm.shift(12)).pow(1 / 3) - 1
    cagr = cagr.where(ttm.shift(12) > 0) * 100
    return ttm, cagr


def asof_wide(w, ticker, as_of):
    if w is None or w.empty or ticker not in w.columns:
        return np.nan
    dates = w.index[w.index <= as_of]
    if len(dates) == 0:
        return np.nan
    return w.loc[dates[-1], ticker]


def asof_q(w, ticker, last_public_quarter):
    if w is None or w.empty or ticker not in w.columns or last_public_quarter is None:
        return np.nan
    dates = w.index[w.index <= last_public_quarter]
    if len(dates) == 0:
        return np.nan
    return w.loc[dates[-1], ticker]


def first_valid(*values):
    for value in values:
        try:
            if pd.notna(value):
                return value
        except ValueError:
            continue
    return np.nan


def safe_ratio(numerator, denominator, multiplier=1.0):
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return np.nan
    try:
        value = float(numerator) / float(denominator) * multiplier
    except (TypeError, ValueError, ZeroDivisionError):
        return np.nan
    if math.isnan(value) or math.isinf(value):
        return np.nan
    return value


# ---------------------------------------------------------------------
# Context builder (unchanged)
# ---------------------------------------------------------------------
def build_context(excel_path):
    (tickers, valuation, missing_ratios, quality_q, earnings_q, earnings_m,
     sector_benchmarks, regimes) = load_all(excel_path)
    if quality_q.empty and earnings_q.empty:
        raise ValueError("Parsed 0 rows from quarterly sheets.")
    _, eps_cagr = add_eps_cagr(earnings_q)
    ann_wide = wide(earnings_q, "ANNOUNCEMENT_DT")
    eq_dates = sorted(pd.Timestamp(d) for d in earnings_q["date"].dropna().unique())
    q_ann_by_ticker = {}
    for ticker in tickers["ticker"]:
        if not ann_wide.empty and ticker in ann_wide.columns:
            ann_s = ann_wide[ticker].copy()
        else:
            ann_s = pd.Series(index=pd.to_datetime(eq_dates), dtype="datetime64[ns]")
        missing = ann_s.isna()
        if len(ann_s.index):
            ann_s.loc[missing] = ann_s.index[missing] + pd.Timedelta(days=45)
        q_ann_by_ticker[ticker] = ann_s.sort_index()
    sec_by_idx = {}
    if not sector_benchmarks.empty:
        for idx, group in sector_benchmarks.groupby("index"):
            sec_by_idx[idx] = group.pivot_table(
                index="date", columns="field", values="value", aggfunc="first"
            ).sort_index()
    quarter_dates = sorted(
        set(pd.Timestamp(d) for d in quality_q["date"].dropna().unique()).union(
            set(pd.Timestamp(d) for d in earnings_q["date"].dropna().unique())
        )
    )
    return {
        "tickers": tickers,
        "ticker_sector": dict(zip(tickers["ticker"], tickers["sector"])),
        "ticker_stock_id": dict(zip(tickers["ticker"], tickers["stock_id"])),
        "quarter_dates": quarter_dates,
        "regimes": regimes,
        "q_ann_by_ticker": q_ann_by_ticker,
        "qwide": {field: wide(quality_q, field) for field in QUALITY_FIELDS},
        "ewide": {field: wide(earnings_q, field) for field in EARNINGS_QUARTERLY_FIELDS if field != "ANNOUNCEMENT_DT"},
        "vwide": {field: wide(valuation, field) for field in VALUATION_FIELDS},
        "mrwide": {field: wide(missing_ratios, field) for field in MISSING_RATIO_FIELDS},
        "mwide": {field: wide(earnings_m, field) for field in EARNINGS_MONTHLY_FIELDS},
        "sec_by_idx": sec_by_idx,
        "eps_cagr": eps_cagr,
        "gm_trend": yoy_change(quality_q, "GROSS_MARGIN"),
        "ebm_trend": yoy_change(quality_q, "EBITDA_MARGIN"),
        "nde_trend": yoy_change(quality_q, "NET_DEBT_TO_EBITDA"),
        "cr_trend": yoy_change(quality_q, "CUR_RATIO"),
    }


def pit_quarter_and_announcement(ctx, ticker, as_of):
    ann_s = ctx["q_ann_by_ticker"].get(ticker)
    if ann_s is None or ann_s.empty:
        return None, None
    public = ann_s[ann_s <= as_of]
    if public.empty:
        return None, None
    last_q = public.index.max()
    return pd.Timestamp(last_q), public.loc[last_q]


def get_regime(ctx, as_of):
    regimes = ctx["regimes"]
    if regimes is None or regimes.empty:
        return {}
    rsub = regimes[regimes.index <= as_of]
    if rsub.empty:
        return {}
    row = rsub.iloc[-1]
    return {
        "rate_environment": row.get("rate_environment"),
        "risk_regime": row.get("risk_regime"),
        "cycle": row.get("cycle"),
        "yield_curve_10y2y": safe_float(row.get("yield_curve_10y2y"), 4),
    }


# ---------------------------------------------------------------------
# Record builder
# ---------------------------------------------------------------------
def build_stock_quarter_record(ctx, ticker, as_of):
    sector = ctx["ticker_sector"].get(ticker)
    stock_id = ctx["ticker_stock_id"].get(ticker)
    is_financial = sector == FINANCIALS_SECTOR

    sector_benchmark_row = None
    sector_index = SECTOR_INDEX.get(sector)
    if sector_index in ctx["sec_by_idx"]:
        sb = ctx["sec_by_idx"][sector_index]
        s = sb[sb.index <= as_of]
        if not s.empty:
            sector_benchmark_row = s.iloc[-1]

    def bench(field):
        if sector_benchmark_row is None or field not in sector_benchmark_row:
            return None
        return safe_float(sector_benchmark_row[field])

    last_public_quarter, last_announcement_date = pit_quarter_and_announcement(ctx, ticker, as_of)

    qwide = ctx["qwide"]; ewide = ctx["ewide"]; vwide = ctx["vwide"]
    mrwide = ctx["mrwide"]; mwide = ctx["mwide"]

    roe = asof_q(qwide.get("RETURN_COM_EQY"), ticker, last_public_quarter)
    roa = asof_q(qwide.get("RETURN_ON_ASSET"), ticker, last_public_quarter)
    gm = asof_q(qwide.get("GROSS_MARGIN"), ticker, last_public_quarter)
    ebm = asof_q(qwide.get("EBITDA_MARGIN"), ticker, last_public_quarter)

    de_from_quality = asof_q(qwide.get("TOT_DEBT_TO_TOT_EQY"), ticker, last_public_quarter)
    de_from_missing = asof_wide(mrwide.get("TOT_DEBT_TO_TOT_EQY"), ticker, as_of)
    de = first_valid(de_from_missing, de_from_quality)
    de_source = (
        "missing_ratios.TOT_DEBT_TO_TOT_EQY" if pd.notna(de_from_missing)
        else ("3_Quality_Quarterly.TOT_DEBT_TO_TOT_EQY" if pd.notna(de_from_quality) else None)
    )

    nde = asof_q(qwide.get("NET_DEBT_TO_EBITDA"), ticker, last_public_quarter)
    ic = asof_q(qwide.get("INTEREST_COVERAGE_RATIO"), ticker, last_public_quarter)
    cr = asof_q(qwide.get("CUR_RATIO"), ticker, last_public_quarter)
    ccc = asof_q(qwide.get("CASH_CONVERSION_CYCLE"), ticker, last_public_quarter)
    fcf = asof_q(qwide.get("CF_FREE_CASH_FLOW"), ticker, last_public_quarter)
    cfo = asof_q(qwide.get("CF_CASH_FROM_OPER"), ticker, last_public_quarter)
    ni = asof_q(qwide.get("NET_INCOME"), ticker, last_public_quarter)
    assets = asof_q(qwide.get("BS_TOT_ASSET"), ticker, last_public_quarter)
    capex = asof_q(qwide.get("CAPITAL_EXPEND"), ticker, last_public_quarter)
    depr = asof_q(qwide.get("CF_DEPR_AMORT"), ticker, last_public_quarter)

    eps_growth = asof_q(ewide.get("EPS_GROWTH"), ticker, last_public_quarter)
    sales_growth = asof_q(ewide.get("SALES_GROWTH"), ticker, last_public_quarter)
    eps_3y_cagr = asof_q(ctx["eps_cagr"], ticker, last_public_quarter)

    gm_trend = asof_q(ctx["gm_trend"], ticker, last_public_quarter)
    ebm_trend = asof_q(ctx["ebm_trend"], ticker, last_public_quarter)
    nde_trend = asof_q(ctx["nde_trend"], ticker, last_public_quarter)
    cr_trend = asof_q(ctx["cr_trend"], ticker, last_public_quarter)

    price_valuation = asof_wide(vwide.get("PX_LAST"), ticker, as_of)
    pe_trailing_valuation = asof_wide(vwide.get("PE_RATIO"), ticker, as_of)
    pe_forward_valuation = asof_wide(vwide.get("BEST_PE_RATIO"), ticker, as_of)
    pb = asof_wide(vwide.get("PX_TO_BOOK_RATIO"), ticker, as_of)
    ev_sales = asof_wide(vwide.get("BEST_CURRENT_EV_BEST_SALES"), ticker, as_of)
    ev_ebitda = asof_wide(vwide.get("BEST_CUR_EV_TO_EBITDA"), ticker, as_of)
    fcf_yield = asof_wide(vwide.get("FREE_CASH_FLOW_YIELD"), ticker, as_of)
    div_yield_valuation = asof_wide(vwide.get("EQY_DVD_YLD_12M"), ticker, as_of)

    price_missing = asof_wide(mrwide.get("PX_LAST"), ticker, as_of)
    trailing_eps = asof_wide(mrwide.get("TRAIL_12M_EPS"), ticker, as_of)
    best_eps_missing = asof_wide(mrwide.get("BEST_EPS"), ticker, as_of)
    pe_forward_missing = asof_wide(mrwide.get("BEST_PE_RATIO"), ticker, as_of)
    div_yield_missing = asof_wide(mrwide.get("EQY_DVD_YLD_12M"), ticker, as_of)
    dividend_per_share_12m = asof_wide(mrwide.get("DIVIDEND_PER_SHARE_12M"), ticker, as_of)
    net_debt = asof_wide(mrwide.get("NET_DEBT"), ticker, as_of)
    total_common_equity = asof_wide(mrwide.get("TOT_COMMON_EQY"), ticker, as_of)
    debt_to_capital = asof_wide(mrwide.get("TOT_DEBT_TO_TOT_CAP"), ticker, as_of)

    price = first_valid(price_missing, price_valuation)
    pe_trailing_calc = safe_ratio(price, trailing_eps)
    pe_trailing = first_valid(pe_trailing_valuation, pe_trailing_calc)

    best_eps = first_valid(asof_wide(mwide.get("BEST_EPS_MEDIAN"), ticker, as_of), best_eps_missing)
    pe_forward_calc = safe_ratio(price, best_eps)
    pe_forward = first_valid(pe_forward_valuation, pe_forward_missing, pe_forward_calc)

    div_yield_calc = safe_ratio(dividend_per_share_12m, price, 100)
    div_yield = first_valid(div_yield_missing, div_yield_valuation, div_yield_calc)
    if pd.isna(div_yield) and (pd.isna(dividend_per_share_12m) or dividend_per_share_12m == 0):
        div_yield = 0.0

    net_debt_to_equity = safe_ratio(net_debt, total_common_equity)

    target_price = asof_wide(mwide.get("BEST_TARGET_PRICE"), ticker, as_of)
    analyst_rating = asof_wide(mwide.get("BEST_ANALYST_RATING"), ticker, as_of)
    n_analysts = asof_wide(mwide.get("TOT_ANALYST_REC"), ticker, as_of)

    # FIX (5): pin both endpoints of the 3-month revision to the SAME series (BEST_EPS_MEDIAN)
    # and require a positive base. The old code's `best_eps` could come from a fallback series
    # while the prior point used BEST_EPS_MEDIAN, mixing series; and a revision off negative EPS
    # is meaningless.
    rev_series = mwide.get("BEST_EPS_MEDIAN")
    eps_now = asof_wide(rev_series, ticker, as_of)
    eps_prev = asof_wide(rev_series, ticker, as_of - pd.DateOffset(months=3))
    eps_revision_3m = None
    if pd.notna(eps_now) and pd.notna(eps_prev) and eps_prev > 0:
        eps_revision_3m = round((eps_now / eps_prev - 1) * 100, 2)

    implied_upside = None
    if pd.notna(target_price) and pd.notna(price) and price != 0:
        implied_upside = round((target_price / price - 1) * 100, 2)

    peg = None
    peg_note = "not_meaningful"
    if pd.notna(pe_forward) and pd.notna(eps_3y_cagr) and pe_forward > 0 and eps_3y_cagr > 0:
        peg = safe_float(pe_forward / eps_3y_cagr, 2)
        peg_note = "ok"
    elif pd.isna(pe_forward):
        peg_note = "forward_pe_not_available"

    opcf_conversion = safe_float(cfo / ni, 3) if (pd.notna(cfo) and pd.notna(ni) and ni != 0) else None
    fcf_conversion = safe_float(fcf / ni, 3) if (pd.notna(fcf) and pd.notna(ni) and ni != 0) else None
    accruals_ratio = (
        safe_float((ni - cfo) / assets, 4)
        if (pd.notna(ni) and pd.notna(cfo) and pd.notna(assets) and assets != 0) else None
    )
    capex_pct_assets = (
        safe_float(abs(capex) / assets * 100, 2)
        if (pd.notna(capex) and pd.notna(assets) and assets != 0) else None
    )
    capex_depr_ratio = (
        safe_float(abs(capex) / abs(depr), 2)
        if (pd.notna(capex) and pd.notna(depr) and depr != 0) else None
    )

    # FIX (1): equity-sign based; non-financial small-equity guard; NO de>500 / roe>300.
    equity_to_assets = safe_ratio(total_common_equity, assets)
    negative_equity_distortion = bool(
        (pd.notna(total_common_equity) and total_common_equity <= 0)
        or (
            (not is_financial)
            and pd.notna(equity_to_assets)
            and 0 < equity_to_assets < SMALL_EQUITY_TO_ASSETS
        )
    )

    # FIX (6): also meaningless when forward P/E is negative (loss-making forward).
    extreme_pe = bool(pd.notna(pe_forward) and (pe_forward > 150 or pe_forward < 0))

    # FIX (4): net cash only when net debt is actually negative.
    if pd.isna(ic):
        ic_status = (
            "net_cash"
            if ((pd.notna(nde) and nde < 0) or (pd.notna(net_debt) and net_debt < 0))
            else "not_available"
        )
    else:
        ic_status = "ok"

    # FIX (2/3): financial-aware statuses for EV/Sales and cash-quality metrics.
    ev_sales_status = (
        "not_applicable_financials" if is_financial
        else ("not_available" if pd.isna(ev_sales) else "ok")
    )
    fin_cash_status = "not_meaningful_financials" if is_financial else "ok"

    # FIX (7): surface benchmark-reliability problems.
    if sector == "Communication Services" and pd.Timestamp(as_of) < GICS_COMSERV_RECLASS:
        sector_benchmark_caveat = "stale_pre_2018_gics_reclass"
    elif ticker in PAYMENT_NETWORK_TICKERS:
        sector_benchmark_caveat = "payment_network_vs_bank_benchmark"
    else:
        sector_benchmark_caveat = None

    is_earnings_event = False
    if last_announcement_date is not None and not pd.isna(last_announcement_date):
        days_from_announcement = abs((pd.Timestamp(as_of) - pd.Timestamp(last_announcement_date)).days)
        is_earnings_event = days_from_announcement <= 7

    return {
        "task": "fundamental_signal_generation",
        "as_of_date": as_of.strftime("%Y-%m-%d"),
        "horizon": "1M",
        "stock_id": stock_id,
        "ticker": ticker,
        "sector": sector,
        "regime_context": get_regime(ctx, as_of),
        "data_visibility": {
            "last_public_quarter_end": (
                None if last_public_quarter is None else last_public_quarter.strftime("%Y-%m-%d")
            ),
            "is_earnings_event": is_earnings_event,
        },
        "data_quality_flags": {
            "negative_equity_distortion": negative_equity_distortion,
            "extreme_pe": extreme_pe,
            "sector_benchmark_caveat": sector_benchmark_caveat,  # FIX (7)
        },
        "valuation": {
            "pe_trailing": {"value": safe_float(pe_trailing), "sector_benchmark": bench("PE_RATIO")},
            "pe_forward": {
                "value": safe_float(pe_forward),
                "sector_benchmark": bench("BEST_PE_RATIO"),
                "status": "extreme_not_meaningful" if extreme_pe else ("not_available" if pd.isna(pe_forward) else "ok"),
            },
            "peg": {"value": peg, "note": peg_note},
            "pb": {"value": safe_float(pb), "sector_benchmark": bench("PX_TO_BOOK_RATIO")},
            "ev_ebitda": {
                "value": safe_float(ev_ebitda),
                "sector_benchmark": bench("BEST_CUR_EV_TO_EBITDA"),
                "basis": "best_forward",
                "status": (
                    "not_applicable_financials" if is_financial
                    else ("not_available" if pd.isna(ev_ebitda) else "ok")
                ),
            },
            "ev_sales": {
                "value": safe_float(ev_sales),
                "sector_benchmark": bench("BEST_CURRENT_EV_BEST_SALES"),
                "basis": "best_forward",
                "status": ev_sales_status,  # FIX (2)
            },
            "fcf_yield": {"value": safe_float(fcf_yield), "sector_benchmark": bench("FREE_CASH_FLOW_YIELD")},
            "div_yield": {
                "value": safe_float(div_yield),
                "sector_benchmark": bench("EQY_DVD_YLD_12M"),
                "status": "structural_zero" if div_yield == 0 else ("not_available" if pd.isna(div_yield) else "ok"),
                "numerator": {"metric": "DIVIDEND_PER_SHARE_12M", "value": safe_float(dividend_per_share_12m)},
                "denominator": {"metric": "PX_LAST", "value": safe_float(price)},
            },
        },
        "profitability": {
            "roe": {
                "value": safe_float(roe),
                "sector_benchmark": bench("RETURN_COM_EQY"),
                "status": "distorted_negative_equity" if negative_equity_distortion else "ok",
            },
            "roa": {"value": safe_float(roa)},
            "gross_margin": {"value": safe_float(gm), "trend_yoy_pp": safe_float(gm_trend, 2)},
            "ebitda_margin": {"value": safe_float(ebm), "trend_yoy_pp": safe_float(ebm_trend, 2)},
        },
        "earnings_quality": {
            "opcf_conversion": {"value": opcf_conversion, "note": "operating_cf_over_net_income", "status": fin_cash_status},  # FIX (3)
            "fcf_conversion": {"value": fcf_conversion, "note": "fcf_after_capex_over_net_income", "status": fin_cash_status},  # FIX (3)
            "accruals_ratio": {"value": accruals_ratio, "note": "(NI-OpCF)/assets; low=high quality", "status": fin_cash_status},  # FIX (3)
            "capex_pct_assets": {"value": capex_pct_assets, "status": fin_cash_status},  # FIX (3)
            "capex_depr_ratio": {"value": capex_depr_ratio, "note": ">1 investing; <1 underinvesting", "status": fin_cash_status},  # FIX (3)
        },
        "growth": {
            "eps_growth_yoy": {"value": safe_float(eps_growth, 2)},
            "eps_3y_cagr": {"value": safe_float(eps_3y_cagr, 2)},
            "rev_growth_yoy": {"value": safe_float(sales_growth, 2)},
        },
        "health": {
            "debt_to_equity": {
                "value": safe_float(de),
                "unit": "percent",
                "source": de_source,
                "status": "not_available" if pd.isna(de) else ("distorted_negative_equity" if negative_equity_distortion else "ok"),
            },
            "net_debt_to_equity": {
                "value": safe_float(net_debt_to_equity, 4),
                "unit": "ratio",
                "status": "missing_components" if pd.isna(net_debt_to_equity) else "ok",
                "numerator": {"metric": "NET_DEBT", "value": safe_float(net_debt)},
                "denominator": {"metric": "TOT_COMMON_EQY", "value": safe_float(total_common_equity)},
            },
            "debt_to_capital": {
                "value": safe_float(debt_to_capital),
                "unit": "percent",
                "status": "not_available" if pd.isna(debt_to_capital) else "ok",
            },
            "net_debt_ebitda": {"value": safe_float(nde), "trend_yoy": safe_float(nde_trend, 3)},
            "interest_coverage": {"value": safe_float(ic), "status": ic_status},  # FIX (4)
            "current_ratio": {"value": safe_float(cr), "trend_yoy": safe_float(cr_trend, 3)},
            "cash_conversion_cycle_days": {"value": safe_float(ccc)},
        },
        "estimates": {
            "best_eps_fwd": safe_float(best_eps),
            "target_price": safe_float(target_price),
            "implied_upside_pct": implied_upside,
            "eps_revision_3m_pct": eps_revision_3m,
            "analyst_rating": safe_float(analyst_rating),
            "n_analysts": safe_int(n_analysts),
        },
    }


def count_real_feature_values(record):
    feature_blocks = ["valuation", "profitability", "earnings_quality", "growth", "health", "estimates"]
    count = 0

    def walk(value):
        nonlocal count
        if isinstance(value, dict):
            for v in value.values():
                walk(v)
            return
        if isinstance(value, list):
            for v in value:
                walk(v)
            return
        if value is None:
            return
        try:
            if pd.isna(value):
                return
        except TypeError:
            pass
        if isinstance(value, str) and value.strip() == "":
            return
        count += 1

    for block in feature_blocks:
        walk(record.get(block, {}))
    return count


def should_write_stock_quarter(record, min_real_values=5):
    if record["data_visibility"]["last_public_quarter_end"] is None:
        return False
    return count_real_feature_values(record) >= min_real_values


# ---------------------------------------------------------------------
# Writer (unchanged)
# ---------------------------------------------------------------------
def write_quarterly_jsonl(excel_path, out_path, start=None, end="2025-01-31", min_real_values=5):
    ctx = build_context(excel_path)
    start_ts = pd.Timestamp(start) if start else min(ctx["quarter_dates"])
    end_ts = pd.Timestamp(end)
    quarter_dates = [d for d in ctx["quarter_dates"] if start_ts <= d <= end_ts]
    if not quarter_dates:
        raise ValueError(
            f"No quarter dates between {start_ts.date()} and {end_ts.date()}. "
            f"Available quarters run from {min(ctx['quarter_dates']).date()} to {max(ctx['quarter_dates']).date()}."
        )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_written = 0
    total_skipped = 0
    skipped_by_ticker = {}
    with open(out_path, "w", encoding="utf-8") as f:
        for as_of in quarter_dates:
            for ticker in ctx["tickers"]["ticker"]:
                record = build_stock_quarter_record(ctx, ticker, as_of)
                if not should_write_stock_quarter(record, min_real_values=min_real_values):
                    total_skipped += 1
                    skipped_by_ticker[ticker] = skipped_by_ticker.get(ticker, 0) + 1
                    continue
                f.write(json.dumps(json_clean(record), separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n")
                total_written += 1
    return {
        "total_written": total_written,
        "total_skipped": total_skipped,
        "n_quarters": len(quarter_dates),
        "n_stocks": len(ctx["tickers"]),
        "first_quarter": quarter_dates[0],
        "last_quarter": quarter_dates[-1],
        "skipped_by_ticker": skipped_by_ticker,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--book", default="Book1.xlsx")
    parser.add_argument("--out", default="fundamental_quarterly.jsonl")
    parser.add_argument("--start", default="2013-01-01")
    parser.add_argument("--end", default="2025-01-31")
    parser.add_argument("--min-real-values", type=int, default=5)
    args = parser.parse_args()
    summary = write_quarterly_jsonl(
        excel_path=args.book, out_path=args.out, start=args.start,
        end=args.end, min_real_values=args.min_real_values,
    )
    print(
        f"Wrote {summary['total_written']} stock-quarter rows to {args.out} "
        f"and skipped {summary['total_skipped']} mostly-empty rows "
        f"({summary['n_stocks']} stocks x {summary['n_quarters']} possible quarters, "
        f"{summary['first_quarter'].date()} to {summary['last_quarter'].date()})"
    )


if __name__ == "__main__":
    main()