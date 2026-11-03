"""Performance metrics for the monthly backtest."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Dict, List

import numpy as np


def equity_curve(monthly_returns: List[float], start: float = 1.0) -> List[float]:
    curve = [start]
    for value in monthly_returns:
        curve.append(curve[-1] * (1.0 + value))
    return curve


def max_drawdown(curve: List[float]) -> float:
    peak = -np.inf
    drawdown = 0.0
    for value in curve:
        peak = max(peak, value)
        drawdown = min(drawdown, value / peak - 1.0)
    return float(drawdown)


def annualized_return(monthly_returns: List[float]) -> float:
    if not monthly_returns:
        return 0.0
    growth = np.prod([1.0 + value for value in monthly_returns])
    return float(growth ** (12.0 / len(monthly_returns)) - 1.0)


def _risk_free_array(rf_monthly, n: int) -> np.ndarray:
    if (isinstance(rf_monthly, Sequence) or isinstance(rf_monthly, np.ndarray)) and not isinstance(rf_monthly, (str, bytes)):
        values = np.asarray(list(rf_monthly), dtype=float)
        if len(values) != n:
            raise ValueError(
                f"risk-free series has {len(values)} observations; expected {n}"
            )
        return values
    return np.full(n, float(rf_monthly), dtype=float)


def excess_returns(monthly_returns: List[float], rf_monthly=0.0) -> np.ndarray:
    returns = np.asarray(monthly_returns, dtype=float)
    return returns - _risk_free_array(rf_monthly, len(returns))


def sharpe(monthly_returns: List[float], rf_monthly=0.0) -> float:
    excess = excess_returns(monthly_returns, rf_monthly)
    if len(excess) < 2:
        return 0.0
    std = excess.std(ddof=1)
    if not np.isfinite(std) or std == 0:
        return 0.0
    return float(excess.mean() / std * np.sqrt(12.0))


def sortino(monthly_returns: List[float], rf_monthly=0.0) -> float:
    excess = excess_returns(monthly_returns, rf_monthly)
    downside = excess[excess < 0]
    downside_std = downside.std(ddof=1) if len(downside) > 1 else 0.0
    if not np.isfinite(downside_std) or downside_std == 0:
        return 0.0
    return float(excess.mean() / downside_std * np.sqrt(12.0))


def calmar(monthly_returns: List[float], curve: List[float]) -> float:
    drawdown = abs(max_drawdown(curve))
    if drawdown == 0:
        return 0.0
    return float(annualized_return(monthly_returns) / drawdown)


def summarize(
    monthly_returns: List[float],
    turnovers: List[float],
    rf_monthly=0.0,
) -> Dict[str, float]:
    curve = equity_curve(monthly_returns)
    rf_array = _risk_free_array(rf_monthly, len(monthly_returns))
    return {
        "cumulative_return": round(curve[-1] - 1.0, 6),
        "annualized_return": round(annualized_return(monthly_returns), 6),
        "sharpe": round(sharpe(monthly_returns, rf_array), 4),
        "sortino": round(sortino(monthly_returns, rf_array), 4),
        "max_drawdown": round(max_drawdown(curve), 6),
        "calmar": round(calmar(monthly_returns, curve), 4),
        "avg_turnover": round(float(np.mean(turnovers)) if turnovers else 0.0, 6),
        "avg_monthly_rf": round(float(rf_array.mean()) if len(rf_array) else 0.0, 6),
        "n_months": len(monthly_returns),
    }
