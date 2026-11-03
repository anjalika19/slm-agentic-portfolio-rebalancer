"""Deterministic synthetic FF5 market backing the FixtureProvider.

The fixture mirrors the live quant specification:
- daily Kenneth-French-style Mkt-RF, SMB, HML, RMW, CMA, and RF series;
- stock total return = RF + alpha + FF5 exposures + idiosyncratic return;
- the benchmark is the synthetic total market return RF + Mkt-RF.

This keeps fixture regressions mathematically consistent with the live path while
remaining deterministic and point-in-time safe.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import BENCHMARK, FF5_FACTORS, FF5_STYLE_FACTORS, RISK_FREE_FACTOR, UNIVERSE


_SEED = 20240930
_START = "2017-01-01"
_END = "2026-06-30"


def _stable_hash(*parts: str) -> int:
    value = 0
    for part in parts:
        for character in part:
            value = (value * 131 + ord(character)) & 0x7FFFFFFF
    return value


def _true_betas() -> Dict[str, Dict[str, float]]:
    betas: Dict[str, Dict[str, float]] = {}
    for ticker in UNIVERSE:
        rng = np.random.default_rng(_stable_hash("beta", ticker))
        exposure = {
            "mkt_rf": float(np.round(1.0 + rng.normal(0.0, 0.18), 3)),
        }
        exposure.update(
            {
                factor: float(np.round(rng.normal(0.0, 0.35), 3))
                for factor in FF5_STYLE_FACTORS
            }
        )
        betas[ticker] = exposure
    return betas


_ALPHA = {
    "GOOGL": 0.05,
    "UNH": 0.03,
    "AMD": 0.025,
    "LLY": 0.025,
    "COST": 0.015,
    "CAT": 0.015,
    "JPM": 0.01,
    "PEP": 0.0,
    "BA": -0.02,
    "BAC": -0.02,
    "XOM": -0.03,
}
_IDIO_VOL = {
    "GOOGL": 0.22,
    "UNH": 0.18,
    "AMD": 0.34,
    "LLY": 0.24,
    "COST": 0.18,
    "JPM": 0.22,
    "BA": 0.32,
    "BAC": 0.26,
    "CAT": 0.24,
    "PEP": 0.14,
    "XOM": 0.28,
}


def _alpha_for(ticker: str) -> float:
    return _ALPHA.get(ticker, 0.0)


def _idio_vol_for(ticker: str) -> float:
    if ticker in _IDIO_VOL:
        return _IDIO_VOL[ticker]
    rng = np.random.default_rng(_stable_hash("idio_vol", ticker))
    return float(0.18 + 0.10 * rng.random())


@lru_cache(maxsize=1)
def _build() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    dates = pd.bdate_range(_START, _END)
    n_days = len(dates)

    # Approximate long-run annual premia and volatility for the synthetic fixture.
    annual_means = {
        "mkt_rf": 0.06,
        "smb": 0.01,
        "hml": 0.02,
        "rmw": 0.02,
        "cma": 0.015,
    }
    annual_vols = {
        "mkt_rf": 0.16,
        "smb": 0.09,
        "hml": 0.10,
        "rmw": 0.08,
        "cma": 0.08,
    }

    factor_returns = {}
    for factor in FF5_FACTORS:
        rng = np.random.default_rng(_stable_hash("factor", factor))
        shocks = rng.normal(
            annual_means[factor] / 252.0,
            annual_vols[factor] / np.sqrt(252.0),
            n_days,
        )
        series = np.zeros(n_days)
        series[0] = shocks[0]
        for index in range(1, n_days):
            series[index] = 0.04 * series[index - 1] + shocks[index]
        factor_returns[factor] = series

    rf_rng = np.random.default_rng(_stable_hash("factor", RISK_FREE_FACTOR))
    rf = np.maximum(
        0.0,
        rf_rng.normal(0.04 / 252.0, 0.002 / np.sqrt(252.0), n_days),
    )

    factors = pd.DataFrame(factor_returns, index=dates)
    factors[RISK_FREE_FACTOR] = rf
    factors = factors[FF5_FACTORS + [RISK_FREE_FACTOR]]

    betas = _true_betas()
    prices = {}
    factor_matrix = factors[FF5_FACTORS].to_numpy()
    for ticker in UNIVERSE:
        beta_vector = np.array([betas[ticker][factor] for factor in FF5_FACTORS])
        systematic = factor_matrix @ beta_vector
        rng = np.random.default_rng(_stable_hash("idio", ticker))
        idiosyncratic = rng.normal(
            0.0,
            _idio_vol_for(ticker) / np.sqrt(252.0),
            n_days,
        )
        total_return = rf + _alpha_for(ticker) / 252.0 + systematic + idiosyncratic
        total_return = np.clip(total_return, -0.95, None)
        prices[ticker] = 100.0 * np.cumprod(1.0 + total_return)

    market_total_return = rf + factors["mkt_rf"].to_numpy()
    prices[BENCHMARK] = 100.0 * np.cumprod(1.0 + market_total_return)

    price_frame = pd.DataFrame(prices, index=dates)
    alpha_series = pd.Series({ticker: _alpha_for(ticker) for ticker in UNIVERSE})
    return price_frame, factors, alpha_series


def daily_prices() -> pd.DataFrame:
    return _build()[0].copy()


def daily_factor_returns() -> pd.DataFrame:
    return _build()[1].copy()


def true_alpha() -> pd.Series:
    return _build()[2].copy()


def prices_through(as_of: str | pd.Timestamp) -> pd.DataFrame:
    timestamp = pd.Timestamp(as_of)
    prices = daily_prices()
    return prices.loc[prices.index <= timestamp]


def factor_returns_through(as_of: str | pd.Timestamp) -> pd.DataFrame:
    timestamp = pd.Timestamp(as_of)
    factors = daily_factor_returns()
    return factors.loc[factors.index <= timestamp]


def forward_total_return(
    ticker: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> float:
    prices = daily_prices()[ticker]
    start_values = prices.loc[prices.index <= pd.Timestamp(start)]
    end_values = prices.loc[prices.index <= pd.Timestamp(end)]
    if start_values.empty or end_values.empty:
        return 0.0
    return float(end_values.iloc[-1] / start_values.iloc[-1] - 1.0)


def month_end_dates(start: str, end: str) -> List[pd.Timestamp]:
    return list(pd.date_range(start, end, freq="ME"))
