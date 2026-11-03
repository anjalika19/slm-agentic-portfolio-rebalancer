"""Kenneth French five-factor model used by the Quant specialist.

The live source is the daily Fama/French 5 Factors (2x3) file. Daily returns are
aggregated to month-end, stock returns are converted to excess returns with RF,
and each ticker is fit on a trailing point-in-time window:

    R_i - RF = alpha + beta_mkt(Mkt-RF) + beta_smb SMB + beta_hml HML
                     + beta_rmw RMW + beta_cma CMA + error

Inference uses a small Newey-West/HAC covariance estimate so monthly
heteroskedasticity/autocorrelation does not make the factor t-statistics look more
precise than they are. Mkt-RF is reported as systematic exposure, while only the
four style factors contribute to ``net_style_factor_support``.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from config import (
    FACTOR_HAC_LAGS,
    FACTOR_LOOKBACK_MONTHS,
    FACTOR_MIN_OBS,
    FACTOR_SIGNAL_P_MAX,
    FF5_FACTORS,
    FF5_STYLE_FACTORS,
    RISK_FREE_FACTOR,
)


def to_monthly_returns(daily_prices: pd.DataFrame) -> pd.DataFrame:
    """Month-end simple returns from a daily price panel."""
    monthly_prices = daily_prices.resample("ME").last()
    return monthly_prices.pct_change().dropna(how="all")


def _compound(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float((1.0 + values).prod() - 1.0)


def to_monthly_ff5_returns(factor_daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily FF5 + RF returns into internally consistent monthly returns.

    The French daily file reports Mkt-RF and RF separately. We first recover the
    daily total market return, compound market and RF independently, then subtract
    monthly RF to obtain monthly Mkt-RF. The four long-short style factors are
    compounded directly.
    """
    required = set(FF5_FACTORS + [RISK_FREE_FACTOR])
    missing = required - set(factor_daily.columns)
    if missing:
        raise ValueError(f"FF5 factor data missing columns: {sorted(missing)}")

    daily = factor_daily[FF5_FACTORS + [RISK_FREE_FACTOR]].copy().sort_index()
    for column in daily.columns:
        daily[column] = pd.to_numeric(daily[column], errors="coerce")

    out = pd.DataFrame(index=daily.resample("ME").size().index)
    out[RISK_FREE_FACTOR] = daily[RISK_FREE_FACTOR].resample("ME").apply(_compound)

    market_total_daily = daily["mkt_rf"] + daily[RISK_FREE_FACTOR]
    market_total_monthly = market_total_daily.resample("ME").apply(_compound)
    out["mkt_rf"] = market_total_monthly - out[RISK_FREE_FACTOR]

    for factor in FF5_STYLE_FACTORS:
        out[factor] = daily[factor].resample("ME").apply(_compound)

    return out[FF5_FACTORS + [RISK_FREE_FACTOR]].dropna(how="all")


def _factor_regime(factor_monthly: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    regime: Dict[str, Dict[str, float]] = {}
    for factor in factor_monthly.columns:
        series = factor_monthly[factor].dropna()
        if len(series) < 6:
            regime[factor] = {
                "ret_3m": None,
                "ret_12m": None,
                "z_score_window": None,
                "trend": "neutral",
                "momentum_score": None,
            }
            continue

        ret_3m = _compound(series.tail(3))
        ret_12m = _compound(series.tail(12))
        window = series.tail(FACTOR_LOOKBACK_MONTHS)
        mean = float(window.mean())
        std = float(window.std(ddof=1))
        if not np.isfinite(std) or std < 1e-12:
            std = 1e-12
        z_score = (float(series.iloc[-1]) - mean) / std
        trend = "up" if ret_3m > ret_12m / 4.0 else ("down" if ret_3m < 0 else "flat")
        momentum_score = mean / std
        regime[factor] = {
            "ret_3m": round(float(ret_3m), 6),
            "ret_12m": round(float(ret_12m), 6),
            "z_score_window": round(float(z_score), 4),
            "trend": trend,
            "momentum_score": round(float(momentum_score), 4),
        }
    return regime


def _ols_with_hac_stats(y: np.ndarray, X: np.ndarray, max_lags: int):
    """OLS coefficients with Newey-West/HAC standard errors.

    Returns coefficients, t-statistics, p-values, R², adjusted R², and residual
    variance. ``X`` must already include an intercept.
    """
    n, k = X.shape
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    residual = y - X @ beta
    dof = max(n - k, 1)
    rss = float(residual @ residual)
    sigma2 = rss / dof

    bread = np.linalg.pinv(X.T @ X)
    meat = np.zeros((k, k), dtype=float)
    for t in range(n):
        xt = X[t][:, None]
        meat += float(residual[t] ** 2) * (xt @ xt.T)

    lags = min(max(int(max_lags), 0), max(n - 1, 0))
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        cross = np.zeros((k, k), dtype=float)
        for t in range(lag, n):
            xt = X[t][:, None]
            xl = X[t - lag][:, None]
            cross += float(residual[t] * residual[t - lag]) * (xt @ xl.T)
        meat += weight * (cross + cross.T)

    covariance = bread @ meat @ bread
    covariance *= n / dof  # small-sample correction
    variances = np.maximum(np.diag(covariance), 0.0)
    standard_errors = np.sqrt(variances)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stats = np.where(standard_errors > 0, beta / standard_errors, 0.0)
    p_values = 2.0 * stats.t.sf(np.abs(t_stats), dof)

    tss = float(((y - y.mean()) ** 2).sum()) or 1e-12
    r_squared = 1.0 - rss / tss
    adj_r_squared = 1.0 - (1.0 - r_squared) * (n - 1) / dof
    return beta, t_stats, p_values, r_squared, adj_r_squared, sigma2


def _empty_result(n_obs: int) -> Dict:
    return {
        "model": "fama_french_5_factor",
        "dependent_variable": "stock_return_minus_rf",
        "n_obs": int(n_obs),
        "insufficient_history": True,
        "factor_exposures": [],
        "selected_factors": [],
        "selected_style_factors": [],
        "alpha": None,
        "alpha_annualized": None,
        "alpha_t_stat": None,
        "alpha_p_value": None,
        "r_squared": None,
        "adj_r_squared": None,
        "residual_variance": None,
        "market_beta": None,
        "market_interaction_3m": None,
        "positive_style_factor_support": 0.0,
        "negative_style_factor_pressure": 0.0,
        "net_style_factor_support": 0.0,
        # Backward-compatible aliases for old UI/run readers.
        "positive_factor_support": 0.0,
        "negative_factor_pressure": 0.0,
        "net_factor_support": 0.0,
    }


def run_factor_regression(
    stock_monthly_excess: pd.DataFrame,
    factor_monthly: pd.DataFrame,
    as_of: Optional[str] = None,
    factors: Optional[List[str]] = None,
    lookback_months: int = FACTOR_LOOKBACK_MONTHS,
    min_obs: int = FACTOR_MIN_OBS,
    hac_lags: int = FACTOR_HAC_LAGS,
) -> Dict[str, Dict]:
    """Run point-in-time FF5 regressions for every stock.

    ``stock_monthly_excess`` must contain monthly stock returns minus monthly RF.
    ``factor_monthly`` must contain Mkt-RF, SMB, HML, RMW, and CMA; RF may also be
    present but is never included as a regressor.
    """
    factors = list(factors or FF5_FACTORS)
    missing = set(factors) - set(factor_monthly.columns)
    if missing:
        raise ValueError(f"Monthly FF5 data missing regressors: {sorted(missing)}")

    if as_of is not None:
        timestamp = pd.Timestamp(as_of)
        stock_monthly_excess = stock_monthly_excess.loc[
            stock_monthly_excess.index <= timestamp
        ]
        factor_monthly = factor_monthly.loc[factor_monthly.index <= timestamp]

    factor_history = factor_monthly[factors]
    regime = _factor_regime(factor_history)
    factor_window = factor_history.tail(lookback_months)
    factor_3m = {factor: regime[factor]["ret_3m"] for factor in factors}

    by_ticker: Dict[str, Dict] = {}
    for ticker in stock_monthly_excess.columns:
        joined = pd.concat(
            [stock_monthly_excess[ticker].rename("stock_excess_return"), factor_window],
            axis=1,
            join="inner",
        ).dropna()
        n_obs = len(joined)
        if n_obs < min_obs:
            by_ticker[ticker] = _empty_result(n_obs)
            continue

        y = joined["stock_excess_return"].to_numpy(dtype=float)
        X_factor = joined[factors].to_numpy(dtype=float)
        X = np.column_stack([np.ones(n_obs), X_factor])
        beta, t_stats, p_values, r2, adj_r2, sigma2 = _ols_with_hac_stats(
            y, X, hac_lags
        )

        alpha = float(beta[0])
        alpha_t = float(t_stats[0])
        alpha_p = float(p_values[0])
        exposures = []
        positive_style_support = 0.0
        negative_style_pressure = 0.0
        market_beta = None
        market_interaction = None

        for index, factor in enumerate(factors):
            factor_beta = float(beta[index + 1])
            factor_t = float(t_stats[index + 1])
            factor_p = float(p_values[index + 1])
            recent_factor_return = float(factor_3m.get(factor) or 0.0)
            interaction = factor_beta * recent_factor_return
            statistically_meaningful = factor_p <= FACTOR_SIGNAL_P_MAX

            if factor == "mkt_rf":
                strength = "systematic" if statistically_meaningful else "uncertain_systematic"
                market_beta = factor_beta
                market_interaction = interaction
            elif statistically_meaningful and interaction > 0:
                strength = "supporting"
                positive_style_support += abs(interaction)
            elif statistically_meaningful and interaction < 0:
                strength = "adverse"
                negative_style_pressure += abs(interaction)
            else:
                strength = "neutral"

            exposures.append(
                {
                    "factor": factor,
                    "beta": round(factor_beta, 4),
                    "t_stat": round(factor_t, 3),
                    "p_value": round(factor_p, 4),
                    "factor_return_3m": round(recent_factor_return, 6),
                    "interaction_3m": round(interaction, 6),
                    "signal_strength": strength,
                }
            )

        net_style_support = positive_style_support - negative_style_pressure
        selected_style = [
            exposure["factor"]
            for exposure in exposures
            if exposure["factor"] in FF5_STYLE_FACTORS
            and exposure["signal_strength"] != "neutral"
        ]
        result = {
            "model": "fama_french_5_factor",
            "dependent_variable": "stock_return_minus_rf",
            "lookback_months": int(lookback_months),
            "hac_lags": int(hac_lags),
            "n_obs": int(n_obs),
            "insufficient_history": False,
            "selected_factors": selected_style,
            "selected_style_factors": selected_style,
            "factor_exposures": exposures,
            "alpha": round(alpha, 6),
            "alpha_annualized": round(alpha * 12.0, 6),
            "alpha_t_stat": round(alpha_t, 3),
            "alpha_p_value": round(alpha_p, 4),
            "r_squared": round(float(r2), 4),
            "adj_r_squared": round(float(adj_r2), 4),
            "residual_variance": round(float(sigma2), 8),
            "market_beta": round(float(market_beta), 4) if market_beta is not None else None,
            "market_interaction_3m": (
                round(float(market_interaction), 6)
                if market_interaction is not None
                else None
            ),
            "positive_style_factor_support": round(positive_style_support, 6),
            "negative_style_factor_pressure": round(negative_style_pressure, 6),
            "net_style_factor_support": round(net_style_support, 6),
            # Backward-compatible aliases. New code/prompts should use style names.
            "positive_factor_support": round(positive_style_support, 6),
            "negative_factor_pressure": round(negative_style_pressure, 6),
            "net_factor_support": round(net_style_support, 6),
        }
        by_ticker[ticker] = result

    return {
        "model": "fama_french_5_factor",
        "regressors": factors,
        "risk_free_column": RISK_FREE_FACTOR,
        "factor_regime": regime,
        "by_ticker": by_ticker,
    }
