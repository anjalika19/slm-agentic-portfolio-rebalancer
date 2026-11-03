"""Shared point-in-time Quant package builder for fixture and live providers.

The factor leg uses Kenneth French daily FF5 + RF. Stock factor regressions are
run on monthly stock excess returns (stock return minus RF), while benchmark-
relative price returns remain available as separate short-horizon evidence.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from config import COV_WINDOW_DAYS, FF5_FACTORS, RISK_FREE_FACTOR
from factor_model import (
    run_factor_regression,
    to_monthly_ff5_returns,
    to_monthly_returns,
)


def _rank_int(values: Dict[str, float], higher_is_better: bool = True) -> Dict[str, int]:
    series = pd.Series(values, dtype=float)
    ranks = series.rank(
        ascending=not higher_is_better,
        method="min",
        na_option="bottom",
    )
    return {key: int(value) for key, value in ranks.items()}


def covariance_block(
    prices: pd.DataFrame,
    universe: List[str],
    window: int = COV_WINDOW_DAYS,
) -> Dict:
    returns = prices[universe].pct_change().dropna().tail(window)
    covariance = (returns.cov() * 252.0).loc[universe, universe]
    return {
        "order": list(universe),
        "window_days": window,
        "annualized": True,
        "matrix": np.round(covariance.values, 8).tolist(),
    }


def factor_block(
    prices: pd.DataFrame,
    universe: List[str],
    bench: str,
    factor_daily: pd.DataFrame,
    as_of: str,
) -> Dict:
    """Build monthly FF5 regressions without using benchmark returns as y."""
    del bench  # benchmark-relative returns are price features, not the FF5 dependent variable
    as_of_timestamp = pd.Timestamp(as_of)
    price_history = prices.loc[prices.index <= as_of_timestamp, universe]
    factor_history = factor_daily.loc[factor_daily.index <= as_of_timestamp]

    stock_monthly = to_monthly_returns(price_history)
    ff5_monthly = to_monthly_ff5_returns(factor_history)
    stock_excess = stock_monthly.sub(ff5_monthly[RISK_FREE_FACTOR], axis=0)
    stock_excess = stock_excess.dropna(how="all")
    return run_factor_regression(stock_excess, ff5_monthly, as_of=as_of)


def _diff(a, b):
    if a is None or b is None:
        return None
    return round(float(a - b), 6)


def _safe_annualized_vol(series: pd.Series, window: int) -> float | None:
    values = series.dropna().tail(window)
    if len(values) < 2:
        return None
    return round(float(values.std(ddof=1) * np.sqrt(252.0)), 6)


def build_quant_package(
    portfolio_id: str,
    as_of: str,
    horizon: str,
    universe: List[str],
    prices: pd.DataFrame,
    bench: str,
    factor_daily: pd.DataFrame,
    macro_regime: str,
) -> Dict:
    """Assemble the Quant package consumed by the specialist and validator."""
    cov_block = covariance_block(prices, universe)
    factor_results = factor_block(prices, universe, bench, factor_daily, as_of)

    returns = prices.pct_change()
    n_stocks = len(universe)
    weights = {ticker: 1.0 / n_stocks for ticker in universe}
    benchmark_prices = prices[bench]

    def window_return(series: pd.Series, lag: int):
        clean = series.dropna()
        if len(clean) <= lag:
            return None
        return round(float(clean.iloc[-1] / clean.iloc[-1 - lag] - 1.0), 6)

    stock_features: Dict[str, Dict] = {}
    ret_3m, ret_12m, vol_21d, style_support, r_squared = {}, {}, {}, {}, {}

    benchmark_1m = window_return(benchmark_prices, 21)
    benchmark_3m = window_return(benchmark_prices, 63)
    benchmark_6m = window_return(benchmark_prices, 126)

    for ticker in universe:
        stock_prices = prices[ticker]
        stock_1m = window_return(stock_prices, 21)
        stock_3m = window_return(stock_prices, 63)
        stock_6m = window_return(stock_prices, 126)
        stock_12m = window_return(stock_prices, 252)

        daily_stock_returns = returns[ticker].dropna()
        vol_21 = _safe_annualized_vol(daily_stock_returns, 21)
        vol_252 = _safe_annualized_vol(daily_stock_returns, 252)
        tracking_returns = (returns[ticker] - returns[bench]).dropna()
        tracking_vol = _safe_annualized_vol(tracking_returns, 63)

        paired = pd.concat([returns[ticker], returns[bench]], axis=1).dropna().tail(252)
        if len(paired) >= 2 and float(np.var(paired.iloc[:, 1], ddof=1)) > 1e-12:
            beta_to_benchmark = round(
                float(
                    np.cov(paired.iloc[:, 0], paired.iloc[:, 1], ddof=1)[0, 1]
                    / np.var(paired.iloc[:, 1], ddof=1)
                ),
                4,
            )
        else:
            beta_to_benchmark = None

        short_window = stock_prices.dropna().tail(63)
        max_drawdown = (
            round(float((short_window / short_window.cummax() - 1.0).min()), 6)
            if not short_window.empty
            else None
        )

        factor_model = factor_results["by_ticker"].get(ticker, {})
        stock_features[ticker] = {
            "portfolio_weight": round(weights[ticker], 6),
            "returns": {
                "ret_1m": stock_1m,
                "ret_3m": stock_3m,
                "ret_6m": stock_6m,
                "ret_12m": stock_12m,
                "excess_ret_1m_vs_benchmark": _diff(stock_1m, benchmark_1m),
                "excess_ret_3m_vs_benchmark": _diff(stock_3m, benchmark_3m),
                "excess_ret_6m_vs_benchmark": _diff(stock_6m, benchmark_6m),
            },
            "risk": {
                "vol_21d": vol_21,
                "vol_252d": vol_252,
                "tracking_vol_63d": tracking_vol,
                "beta_to_benchmark_252d": beta_to_benchmark,
                "max_drawdown_63d": max_drawdown,
                "adv_30d_usd": None,
            },
            "factor_model": factor_model,
        }

        ret_3m[ticker] = stock_3m if stock_3m is not None else np.nan
        ret_12m[ticker] = stock_12m if stock_12m is not None else np.nan
        vol_21d[ticker] = vol_21 if vol_21 is not None else np.nan
        style_support[ticker] = factor_model.get("net_style_factor_support", 0.0)
        r_squared[ticker] = factor_model.get("r_squared", np.nan)

    ranks = {
        "ret_3m": _rank_int(ret_3m, True),
        "ret_12m": _rank_int(ret_12m, True),
        "vol_21d": _rank_int(vol_21d, False),
        "net_style_factor_support": _rank_int(style_support, True),
        "r_squared": _rank_int(r_squared, True),
    }
    for ticker in universe:
        stock_features[ticker]["cross_sectional_ranks"] = {
            "rank_ret_3m": ranks["ret_3m"][ticker],
            "rank_ret_12m": ranks["ret_12m"][ticker],
            "rank_vol_21d": ranks["vol_21d"][ticker],
            "rank_net_style_factor_support": ranks["net_style_factor_support"][ticker],
            # Backward-compatible alias for historical UI/readers.
            "rank_net_factor_support": ranks["net_style_factor_support"][ticker],
            "rank_r_squared": ranks["r_squared"][ticker],
        }

    portfolio_weighted_betas = {factor: 0.0 for factor in FF5_FACTORS}
    for ticker in universe:
        exposures = stock_features[ticker]["factor_model"].get("factor_exposures", [])
        for exposure in exposures:
            factor = exposure.get("factor")
            beta = exposure.get("beta")
            if factor in portfolio_weighted_betas and beta is not None:
                portfolio_weighted_betas[factor] += weights[ticker] * float(beta)

    finite_betas = {
        factor: beta
        for factor, beta in portfolio_weighted_betas.items()
        if np.isfinite(beta)
    }
    dominant_factor = (
        max(finite_betas, key=lambda factor: abs(finite_betas[factor]))
        if finite_betas
        else None
    )

    factor_dates = factor_daily.index[factor_daily.index <= pd.Timestamp(as_of)]
    latest_factor_date = (
        pd.Timestamp(factor_dates.max()).strftime("%Y-%m-%d")
        if len(factor_dates)
        else None
    )

    return {
        "task": "quant_portfolio_signal_generation",
        "as_of_date": as_of,
        "horizon": horizon,
        "portfolio_id": portfolio_id,
        "macro_regime": macro_regime,
        "n_stocks": n_stocks,
        "portfolio_universe": list(universe),
        "portfolio_weights": {ticker: round(weight, 6) for ticker, weight in weights.items()},
        "factor_model_spec": {
            "name": "fama_french_5_factor",
            "dependent_variable": "stock_return_minus_rf",
            "regressors": FF5_FACTORS,
            "risk_free_column": RISK_FREE_FACTOR,
            "latest_factor_date": latest_factor_date,
        },
        "stock_features": stock_features,
        "factor_regime": factor_results["factor_regime"],
        "portfolio_factor_summary": {
            "portfolio_weighted_betas": {
                factor: round(beta, 4)
                for factor, beta in portfolio_weighted_betas.items()
            },
            "dominant_factor": dominant_factor,
            "dominant_factor_portfolio_beta": (
                round(portfolio_weighted_betas[dominant_factor], 4)
                if dominant_factor
                else None
            ),
            "market_beta_is_systematic_not_alpha": True,
        },
        "covariance_matrix": cov_block,
    }
