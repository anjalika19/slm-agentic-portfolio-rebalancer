import io
import json
import math
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


# Kenneth French five-factor model. RF is used to form excess returns and is
# deliberately not included in the regressor list.
FF5_FACTOR_NAMES = ["mkt_rf", "smb", "hml", "rmw", "cma"]
RISK_FREE_COLUMN = "rf"
DEFAULT_FACTOR_NAMES = FF5_FACTOR_NAMES

# Kept only for reading or comparing historical JKP-generated artifacts.
JKP_FACTOR_NAMES = [
    "accruals",
    "debt_issuance",
    "div12m_me",
    "investment",
    "low_leverage",
    "low_risk",
    "momentum",
    "profit_growth",
    "profitability",
    "quality",
    "rvol_21d",
    "seasonality",
    "short_term_reversal",
    "size",
    "value",
]


def ensure_dir(path: Union[str, Path]) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def clean_factor_name(name: str) -> str:
    return (
        str(name)
        .lower()
        .strip()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("%", "pct")
        .replace("(", "")
        .replace(")", "")
    )


def month_end_from_string(x: str) -> pd.Timestamp:
    x = str(x).strip()
    if len(x) == 7:
        return pd.Period(x, freq="M").to_timestamp("M")
    return pd.Timestamp(x).to_period("M").to_timestamp("M")


def month_file_key(month_end: pd.Timestamp) -> str:
    return pd.Timestamp(month_end).strftime("%Y-%m-%d")


def safe_float(x: Any, ndigits: int = 6):
    if x is None:
        return None
    if isinstance(x, (float, np.floating)):
        if np.isnan(x) or np.isinf(x):
            return None
        return round(float(x), ndigits)
    if isinstance(x, (int, np.integer)):
        return float(x)
    return x


def round_nested(obj: Any, ndigits: int = 6):
    if isinstance(obj, dict):
        return {k: round_nested(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_nested(v, ndigits) for v in obj]
    return safe_float(obj, ndigits)


def write_json(path: Union[str, Path], obj: Dict) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(round_nested(obj), f, indent=2)


def read_json(path: Union[str, Path]) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compound_return(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return np.nan
    return float((1.0 + x).prod() - 1.0)


def max_drawdown_from_prices(prices: pd.Series) -> float:
    prices = pd.to_numeric(prices, errors="coerce").dropna()
    if len(prices) < 2:
        return np.nan
    running_max = prices.cummax()
    dd = prices / running_max - 1.0
    return float(dd.min())


def zscore(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    mu = values.mean(skipna=True)
    sd = values.std(skipna=True, ddof=0)

    if not np.isfinite(sd) or sd < 1e-12:
        return pd.Series(0.0, index=values.index)

    return ((values - mu) / sd).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def normal_p_value_from_t(t_stat: float) -> float:
    if not np.isfinite(t_stat):
        return np.nan
    return float(math.erfc(abs(t_stat) / math.sqrt(2.0)))


def load_prices(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"date", "ticker", "PX_LAST", "PX_VOLUME"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Price file missing columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

    for col in ["PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST", "PX_VOLUME"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["ticker", "date"])
    df = df.drop_duplicates(["ticker", "date"], keep="last")

    df["daily_return"] = df.groupby("ticker")["PX_LAST"].pct_change()
    df["dollar_volume"] = df["PX_LAST"] * df["PX_VOLUME"]

    return df


def _read_factor_text(path: Union[str, Path]) -> str:
    """Read a raw Kenneth French CSV from either a CSV or one-file ZIP."""
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if len(csv_names) != 1:
                raise ValueError(
                    f"Expected exactly one CSV in {path}, found {len(csv_names)}."
                )
            return zf.read(csv_names[0]).decode("utf-8-sig")
    return path.read_text(encoding="utf-8-sig")


def _load_kenneth_french_ff5(path: Union[str, Path]) -> pd.DataFrame:
    """Load the raw French daily FF5 file and convert percent values to decimals."""
    text = _read_factor_text(path)
    lines = text.splitlines()

    header_idx = next(
        (
            i
            for i, line in enumerate(lines)
            if "Mkt-RF" in line and "SMB" in line and "HML" in line and "RF" in line
        ),
        None,
    )
    if header_idx is None:
        raise ValueError("Could not find the Kenneth French FF5 header row.")

    df = pd.read_csv(io.StringIO(text), skiprows=header_idx)
    df = df.rename(columns={df.columns[0]: "date"})

    # Remove the copyright/footer lines after the last YYYYMMDD observation.
    date_text = df["date"].astype(str).str.strip()
    df = df.loc[date_text.str.fullmatch(r"\d{8}")].copy()
    df["date"] = pd.to_datetime(
        df["date"].astype(str).str.strip(), format="%Y%m%d", errors="coerce"
    )

    rename = {
        "Mkt-RF": "mkt_rf",
        "SMB": "smb",
        "HML": "hml",
        "RMW": "rmw",
        "CMA": "cma",
        "RF": "rf",
    }
    df = df.rename(columns=rename)

    required = ["date", *FF5_FACTOR_NAMES, RISK_FREE_COLUMN]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"French FF5 file missing columns: {sorted(missing)}")

    # French Library files are percentages: -0.67 means -0.67%, not -67%.
    for col in [*FF5_FACTOR_NAMES, RISK_FREE_COLUMN]:
        df[col] = pd.to_numeric(df[col], errors="coerce") / 100.0

    return (
        df[required]
        .dropna(subset=["date"])
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )


def load_factors(path: str) -> pd.DataFrame:
    """Load either raw Kenneth French FF5 data or the legacy JKP factor CSV."""
    path_obj = Path(path)

    # Raw French file: ZIP, or CSV with metadata before its real header.
    if path_obj.suffix.lower() == ".zip":
        return _load_kenneth_french_ff5(path_obj)

    with path_obj.open("r", encoding="utf-8-sig") as f:
        prefix = "".join(f.readline() for _ in range(12))
    if "Mkt-RF" in prefix and "SMB" in prefix and "RF" in prefix:
        return _load_kenneth_french_ff5(path_obj)

    df = pd.read_csv(path_obj)

    if {"date", "name", "ret"}.issubset(df.columns):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df["name"] = df["name"].map(clean_factor_name)
        df["ret"] = pd.to_numeric(df["ret"], errors="coerce")

        wide = df.pivot_table(
            index="date",
            columns="name",
            values="ret",
            aggfunc="mean",
        ).sort_index()

        wide = wide.reset_index()
        wide.columns = [
            "date" if c == "date" else clean_factor_name(c)
            for c in wide.columns
        ]
        return wide

    if "date" not in df.columns:
        raise ValueError("Factor file must have date column.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    rename = {c: clean_factor_name(c) for c in df.columns if c != "date"}
    df = df.rename(columns=rename)

    for c in df.columns:
        if c != "date":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.sort_values("date")


def load_benchmark(path: Optional[str]) -> Optional[pd.DataFrame]:
    if path is None:
        return None

    raw = pd.read_csv(path, header=None)

    dates_row = None
    for i in range(min(30, len(raw))):
        if str(raw.iloc[i, 0]).strip().lower() in {"date", "dates"}:
            dates_row = i
            break

    if dates_row is None:
        df = pd.read_csv(path)
        date_cols = [c for c in df.columns if str(c).lower() in {"date", "dates"}]
        if not date_cols:
            return None

        date_col = date_cols[0]
        price_cols = [c for c in df.columns if c != date_col]
        if not price_cols:
            return None

        out = df[[date_col, price_cols[0]]].rename(
            columns={date_col: "date", price_cols[0]: "benchmark_price"}
        )
    else:
        tickers_row = max(0, dates_row - 2)
        fields_row = dates_row

        tickers = raw.iloc[tickers_row].ffill()
        fields = raw.iloc[fields_row]
        data = raw.iloc[dates_row + 1:].copy()
        data = data.rename(columns={0: "date"})

        candidates = []
        for col in range(1, raw.shape[1]):
            ticker = str(tickers.iloc[col]).strip()
            field = str(fields.iloc[col]).strip()
            if ticker.lower() == "nan" or field.lower() == "nan":
                continue
            candidates.append((col, ticker, field))

        chosen = None

        for col, ticker, field in candidates:
            if ticker.upper() == "SPXT INDEX" and field.upper() == "PX_LAST":
                chosen = (col, ticker, field)
                break

        if chosen is None:
            for col, ticker, field in candidates:
                if ticker.upper() == "SPXT INDEX" and field.upper() in {"PX_LAST", "PX_OPEN"}:
                    chosen = (col, ticker, field)
                    break

        if chosen is None and candidates:
            chosen = candidates[0]

        if chosen is None:
            return None

        col, ticker, field = chosen
        print(f"[benchmark] using {ticker} {field}")

        out = data[["date", col]].rename(columns={col: "benchmark_price"})

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["benchmark_price"] = pd.to_numeric(out["benchmark_price"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date")
    out["benchmark_return"] = out["benchmark_price"].pct_change()

    return out[["date", "benchmark_price", "benchmark_return"]]


def monthly_stock_prices(prices: pd.DataFrame) -> pd.DataFrame:
    px = prices.pivot_table(
        index="date",
        columns="ticker",
        values="PX_LAST",
        aggfunc="last",
    ).sort_index()
    return px.resample("ME").last()


def monthly_stock_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return monthly_stock_prices(prices).pct_change()


def monthly_factor_returns(factors: pd.DataFrame) -> pd.DataFrame:
    factor_cols = [c for c in factors.columns if c != "date"]
    f = factors.sort_values("date").set_index("date")[factor_cols]

    if set([*FF5_FACTOR_NAMES, RISK_FREE_COLUMN]).issubset(f.columns):
        out = f[["smb", "hml", "rmw", "cma"]].resample("ME").apply(
            compound_return
        )
        monthly_rf = f["rf"].resample("ME").apply(compound_return)
        monthly_market = (f["mkt_rf"] + f["rf"]).resample("ME").apply(
            compound_return
        )
        out.insert(0, "mkt_rf", monthly_market - monthly_rf)
        out["rf"] = monthly_rf
        return out[[*FF5_FACTOR_NAMES, RISK_FREE_COLUMN]]

    return f.resample("ME").apply(compound_return)


def build_ff5_regression_frame(
    stock_returns: pd.Series,
    factors: pd.DataFrame,
) -> pd.DataFrame:
    """Align monthly stock returns with FF5 and create the correct excess-return y."""
    monthly_ff5 = monthly_factor_returns(factors)
    stock = pd.to_numeric(stock_returns, errors="coerce").rename("stock_return")
    frame = stock.to_frame().join(monthly_ff5, how="inner").dropna()
    frame["stock_excess_return"] = frame["stock_return"] - frame["rf"]
    return frame


def daily_return_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pivot_table(
        index="date",
        columns="ticker",
        values="daily_return",
        aggfunc="last",
    ).sort_index()


def available_month_ends_from_prices(
    prices: pd.DataFrame,
    start: str,
    end: str,
) -> List[pd.Timestamp]:
    monthly_px = monthly_stock_prices(prices)

    return [
        pd.Timestamp(x)
        for x in monthly_px.index
        if pd.Timestamp(start) <= pd.Timestamp(x) <= pd.Timestamp(end)
    ]