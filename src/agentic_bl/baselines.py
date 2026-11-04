"""Baseline portfolios for comparison (spec section 6): 1/N, MVO, Risk Parity.

All long-only, computed Python-only from a covariance (and, for MVO, expected
returns) so the UI can show the debate portfolio against them on the same dates.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

try:
    import cvxpy as cp
except Exception:  # pragma: no cover
    cp = None


def equal_weight(universe: List[str]) -> Dict[str, float]:
    n = len(universe)
    return {t: 1.0 / n for t in universe}


def max_sharpe_mvo(
    universe: List[str], mu: np.ndarray, Sigma: np.ndarray, max_weight: float = 1.0
) -> Dict[str, float]:
    """Long-only maximum-Sharpe portfolio via the standard convex reformulation.

    Maximise mu^T y s.t. y >= 0, y^T Sigma y <= 1 ... then renormalise y to sum 1.
    Falls back to min-variance if expected excess returns are all <= 0.
    """
    n = len(universe)
    Sig = 0.5 * (Sigma + Sigma.T)
    if cp is None or np.all(mu <= 0):
        return min_variance(universe, Sigma, max_weight)
    y = cp.Variable(n)
    prob = cp.Problem(
        cp.Maximize(mu @ y),
        [y >= 0, cp.quad_form(y, cp.psd_wrap(Sig)) <= 1.0],
    )
    try:
        prob.solve(solver=cp.CLARABEL)
    except Exception:
        prob.solve()
    if y.value is None or y.value.sum() <= 1e-9:
        return min_variance(universe, Sigma, max_weight)
    w = np.clip(np.asarray(y.value).reshape(-1), 0, None)
    w = w / w.sum()
    return {t: float(w[i]) for i, t in enumerate(universe)}


def min_variance(universe: List[str], Sigma: np.ndarray, max_weight: float = 1.0) -> Dict[str, float]:
    n = len(universe)
    Sig = 0.5 * (Sigma + Sigma.T)
    if cp is None:
        w = np.ones(n) / n
        return {t: float(w[i]) for i, t in enumerate(universe)}
    w = cp.Variable(n)
    cons = [w >= 0, cp.sum(w) == 1]
    if max_weight < 1.0 and n * max_weight >= 1.0:
        cons.append(w <= max_weight)
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(Sig))), cons)
    try:
        prob.solve(solver=cp.CLARABEL)
    except Exception:
        prob.solve()
    wv = np.ones(n) / n if w.value is None else np.clip(np.asarray(w.value).reshape(-1), 0, None)
    wv = wv / wv.sum()
    return {t: float(wv[i]) for i, t in enumerate(universe)}


def risk_parity(universe: List[str], Sigma: np.ndarray, iters: int = 500) -> Dict[str, float]:
    """Equal risk-contribution portfolio via a simple fixed-point iteration."""
    n = len(universe)
    Sig = 0.5 * (Sigma + Sigma.T)
    w = np.ones(n) / n
    target = 1.0 / n
    for _ in range(iters):
        mrc = Sig @ w                      # marginal risk contribution
        port_var = float(w @ Sig @ w) or 1e-12
        rc = w * mrc / port_var            # risk contribution share
        # multiplicative update toward equal contribution
        w = w * (target / np.maximum(rc, 1e-12)) ** 0.5
        w = np.clip(w, 0, None)
        w = w / w.sum()
    return {t: float(w[i]) for i, t in enumerate(universe)}


def all_baselines(
    universe: List[str], mu: np.ndarray, Sigma: np.ndarray
) -> Dict[str, Dict[str, float]]:
    return {
        "equal_weight": equal_weight(universe),
        "mvo_max_sharpe": max_sharpe_mvo(universe, mu, Sigma),
        "risk_parity": risk_parity(universe, Sigma),
    }
