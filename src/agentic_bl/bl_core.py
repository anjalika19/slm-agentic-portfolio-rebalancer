"""Black-Litterman core: covariance, Omega scaling, posterior, constrained MVO.

This module is where the three central correctness fixes live (spec section 3):

  3b  Omega is tied to the prior the Idzorek way --
      Omega_ii = (1/confidence - 1) * (P_i . tauSigma . P_i^T), floored.
      NOT (1 - confidence)^2 on an arbitrary scale (which dwarfs tauSigma and
      collapses the posterior back to equilibrium).

  3c  The optimizer is a real constrained QP (cvxpy): maximise
      mu^T w - (delta/2) w^T Sigma w subject to common min/max weights,
      per-name and total active-weight limits, and sector caps. Absolute
      volatility is reported but is only constrained when an explicit mandate
      supplies ``vol_cap``. Infeasible solves fall back loudly.

  3d  The covariance builder verifies symmetry + PSD and never silently returns
      identity; it ridge-corrects only as a last resort and reports the reason
      and condition number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from agentic_bl import config
import numpy as np

try:
    import cvxpy as cp
except Exception:  # pragma: no cover - cvxpy is a hard dependency for optimize()
    cp = None

from agentic_bl.config import (
    BL_TAU,
    MAX_WEIGHT,
    MIN_OMEGA,
    RISK_AVERSION,
    SECTOR_CAP,
    VOL_CAP,
)


# --------------------------------------------------------------------------- #
# covariance (3d)
# --------------------------------------------------------------------------- #
@dataclass
class CovarianceResult:
    order: List[str]
    matrix: np.ndarray          # annualized, symmetric PSD
    condition_number: float
    ridge_applied: bool
    note: str


def build_covariance(
    daily_returns: np.ndarray,
    order: List[str],
    *,
    annualization: int = 252,
    strict: bool = True,
    min_eig: float = 1e-10,
) -> CovarianceResult:
    """Sample covariance over a daily-return window, annualized + PSD-verified.

    ``daily_returns`` is shape (n_days, n_assets). Raises (strict) if the block is
    missing/degenerate rather than silently returning identity.
    """
    R = np.asarray(daily_returns, dtype=float)
    if R.ndim != 2 or R.shape[1] != len(order):
        raise ValueError(f"returns shape {R.shape} incompatible with order len {len(order)}")
    R = R[~np.isnan(R).any(axis=1)]
    if R.shape[0] < max(20, R.shape[1] + 1):
        msg = f"covariance window too short: {R.shape[0]} usable rows for {R.shape[1]} assets"
        if strict:
            raise ValueError(msg)
        # non-strict: still do NOT fabricate identity -- use a diagonal of variances
        var = np.nanvar(R, axis=0) if R.shape[0] > 1 else np.ones(len(order))
        cov = np.diag(np.maximum(var, 1e-8)) * annualization
        cond = float(np.linalg.cond(cov))
        return CovarianceResult(order, cov, cond, True, "diag_fallback:" + msg)

    cov = np.cov(R, rowvar=False) * annualization
    cov = 0.5 * (cov + cov.T)  # force symmetry

    eigvals = np.linalg.eigvalsh(cov)
    ridge_applied = False
    note = "sample_cov_psd"
    if eigvals.min() < min_eig:
        # ridge to nearest PSD instead of discarding the real covariance structure
        bump = (min_eig - eigvals.min()) + 1e-8
        cov = cov + bump * np.eye(cov.shape[0])
        ridge_applied = True
        note = f"ridge(+{bump:.2e}) applied; min_eig was {eigvals.min():.2e}"

    cond = float(np.linalg.cond(cov))
    return CovarianceResult(order, cov, cond, ridge_applied, note)


def covariance_from_block(
    block: Dict, order: List[str], *, strict: bool = True
) -> CovarianceResult:
    """Build a CovarianceResult from a pre-computed covariance block (already annualized).

    Used when the provider hands us a covariance matrix directly (fixture/live).
    Verifies symmetry/PSD; never falls back to identity silently.
    """
    mat = block.get("matrix")
    blk_order = block.get("order", order)
    if mat is None:
        msg = f"covariance block missing 'matrix' for order={order}"
        if strict:
            raise ValueError(msg)
        cov = np.eye(len(order)) * 0.04
        return CovarianceResult(order, cov, float(np.linalg.cond(cov)), True, "MISSING:" + msg)

    M = np.asarray(mat, dtype=float)
    idx = [blk_order.index(t) for t in order]
    M = M[np.ix_(idx, idx)]
    M = 0.5 * (M + M.T)
    eig = np.linalg.eigvalsh(M)
    ridge = False
    note = "provider_cov_psd"
    if eig.min() < 1e-10:
        bump = (1e-10 - eig.min()) + 1e-8
        M = M + bump * np.eye(M.shape[0])
        ridge = True
        note = f"ridge(+{bump:.2e}); min_eig was {eig.min():.2e}"
    return CovarianceResult(order, M, float(np.linalg.cond(M)), ridge, note)


# --------------------------------------------------------------------------- #
# equilibrium prior
# --------------------------------------------------------------------------- #
def equilibrium_returns(
    Sigma: np.ndarray, w_mkt: np.ndarray, risk_aversion: float = RISK_AVERSION
) -> np.ndarray:
    """Reverse-optimized implied equilibrium excess returns: pi = delta * Sigma w_mkt."""
    return risk_aversion * Sigma @ w_mkt


# --------------------------------------------------------------------------- #
# views: P rows + Idzorek Omega (3b)
# --------------------------------------------------------------------------- #
def p_row_absolute(ticker: str, universe_order: List[str]) -> List[float]:
    """Absolute view picks out one asset (P row is a unit vector)."""
    return [1.0 if t == ticker else 0.0 for t in universe_order]


def p_row_relative(long_tickers, short_tickers, universe_order: List[str]) -> List[float]:
    """Relative (zero-sum) view: the long basket outperforms the short basket.

    Reference formulation (unchanged BL math, spec A6):
        E[R] = [(tau*Sigma)^-1 + P^T Omega^-1 P]^-1 [(tau*Sigma)^-1 Pi + P^T Omega^-1 Q]
        Pi = delta * Sigma * w,  tau = 0.05,  delta = 2.5
    An ABSOLUTE view is a unit P-row [0..1..0] with Q the asset's expected return.
    A RELATIVE view is a zero-sum P-row (+1/k on the favored basket of k names, -1/m on
    the avoided basket of m names) with Q the expected spread. Single-stock specialists
    use absolute views; the Economist expresses favored-vs-avoided sectors as relative
    views. Both stack into the SAME P/Q/Omega and the same posterior formula above.
    """
    row = [0.0] * len(universe_order)
    longs = [t for t in long_tickers if t in universe_order]
    shorts = [t for t in short_tickers if t in universe_order]
    if not longs or not shorts:
        return row
    for t in longs:
        row[universe_order.index(t)] += 1.0 / len(longs)
    for t in shorts:
        row[universe_order.index(t)] -= 1.0 / len(shorts)
    return row


def p_row_vs_complement(ticker: str, universe_order: List[str]) -> List[float]:
    """Zero-sum view: *ticker* vs the equal-weight complement.

    Row = [+1, -1/(n-1), -1/(n-1), …] for the given ticker's position.
    Q = expected outperformance of *ticker* over the rest of the portfolio.
    This is the canonical single-stock relative view (Idzorek 2005 §3.2).
    """
    n = len(universe_order)
    complement_weight = -1.0 / (n - 1) if n > 1 else 0.0
    return [
        1.0 if t == ticker else complement_weight
        for t in universe_order
    ]


def idzorek_omega(
    P: np.ndarray,
    Sigma: np.ndarray,
    confidences: np.ndarray,
    tau: float = BL_TAU,
    widen: np.ndarray = None,
) -> np.ndarray:
    """Idzorek-style Omega tied to the prior (spec 3b / D2):

        Omega_kk = widen_k * (1/c_k - 1) * (P_k . tauSigma . P_k^T), floored to MIN_OMEGA.

    A confidence near 1 -> tiny Omega (view dominates); near 0 -> huge Omega.
    ``widen_k`` (default 1) inflates the uncertainty of standalone ABSOLUTE views
    (use ~3) so a single conviction informs but cannot dominate the prior, while
    PAIRWISE rotation views keep widen=1.
    """
    P = np.atleast_2d(P)
    c = np.clip(np.asarray(confidences, dtype=float), 1e-4, 0.9999)
    base = np.einsum("ij,jk,ik->i", P, tau * Sigma, P)  # P_k tauSigma P_k^T
    w = np.ones_like(c) if widen is None else np.asarray(widen, dtype=float)
    diag = w * (1.0 / c - 1.0) * base
    diag = np.maximum(diag, MIN_OMEGA)
    return np.diag(diag)


def classify_view_row(row, tol: float = 1e-6) -> str:
    """Classify a P row as:
      'absolute'  — unit-sum with a single +1 entry  (e.g. [0,1,0,0,0,0])
      'pairwise'  — zero-sum with exactly one +1 and one -1  ([+1,0,-1,0,0,0])
      'relative'  — zero-sum fractional  ([+1,-0.2,-0.2,-0.2,-0.2,-0.2])
      'invalid'   — anything else (non-zero-sum AND not unit-sum-single)

    We allow 'relative' rows because debate-derived views pair one long vs the
    equal-weight complement; banning them would block all pairwise signal from
    debates where no clean +1/-1 opposing pair exists.
    """
    r = np.asarray(row, dtype=float)
    s = float(r.sum())
    nz = np.abs(r) > tol
    if abs(s - 1.0) < tol and nz.sum() == 1 and abs(r[nz][0] - 1.0) < tol:
        return "absolute"
    if abs(s) < tol and nz.sum() == 2 and sorted(np.round(r[nz], 6)) == [-1.0, 1.0]:
        return "pairwise"
    if abs(s) < tol:          # zero-sum fractional (e.g. vs-complement rows)
        return "relative"
    return "invalid"


def validate_view_rows(P) -> list:
    """Return the classification of each row; raises if any row is truly invalid
    (not zero-sum and not a single-+1 absolute view).

    Allowed shapes:
      absolute — [0…+1…0]              (single-stock level view)
      pairwise — [0…+1…-1…0]           (clean long-short pair)
      relative — zero-sum fractional    (long vs equal-weight complement)
    """
    P = np.atleast_2d(P)
    kinds = [classify_view_row(r) for r in P]
    bad = [i for i, k in enumerate(kinds) if k == "invalid"]
    if bad:
        raise ValueError(
            f"invalid view rows (not zero-sum, not single +1) at indices {bad}: "
            f"P={P[bad].tolist()}"
        )
    return kinds


def parametric_var_cvar(mu_p: float, sigma_p: float, level: float = 0.95):
    """Parametric monthly VaR/CVaR readout (spec D5). Returns positive loss numbers.
    VaR95 ~= 1.645*sigma - mu ; CVaR95 ~= 2.063*sigma - mu (monthly inputs)."""
    z = {0.95: 1.645, 0.99: 2.326}.get(level, 1.645)
    phi = {0.95: 2.063, 0.99: 2.665}.get(level, 2.063)  # E[Z|Z>z]/(1-level) tail factor
    return {
        "level": level,
        "var": round(float(z * sigma_p - mu_p), 6),
        "cvar": round(float(phi * sigma_p - mu_p), 6),
        "sigma_monthly": round(float(sigma_p), 6),
        "mu_monthly": round(float(mu_p), 6),
    }


# --------------------------------------------------------------------------- #
# posterior
# --------------------------------------------------------------------------- #
@dataclass
class BLPosterior:
    mu_bl: np.ndarray
    cov_bl: np.ndarray
    pi: np.ndarray


def bl_posterior(
    pi: np.ndarray,
    Sigma: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    Omega: np.ndarray,
    tau: float = BL_TAU,
) -> BLPosterior:
    """Canonical Black-Litterman posterior (He & Litterman 1999; Idzorek 2005):

        E[R] = [(tau*Sigma)^-1 + P^T Omega^-1 P]^-1 [(tau*Sigma)^-1 Pi + P^T Omega^-1 Q]

    Note: Omega is INVERTED here. (Some texts, e.g. Chen & Tsui 2008 eq. 2, print this
    without the inverse on Omega -- that is a typo; a more-uncertain view has larger
    Omega and must get LESS weight, which only holds with Omega^-1.)
    Pi is the reverse-optimized equilibrium prior Pi = delta * Sigma * w (see
    equilibrium_returns)."""
    P = np.atleast_2d(P)
    Q = np.asarray(Q, dtype=float).reshape(-1)
    tau_sigma = tau * Sigma
    tau_sigma_inv = np.linalg.pinv(tau_sigma)
    omega_inv = np.linalg.pinv(Omega)
    precision = tau_sigma_inv + P.T @ omega_inv @ P
    rhs = tau_sigma_inv @ pi + P.T @ omega_inv @ Q
    cov_post_views = np.linalg.pinv(precision)
    mu_bl = cov_post_views @ rhs
    cov_bl = Sigma + cov_post_views
    return BLPosterior(mu_bl=mu_bl, cov_bl=cov_bl, pi=pi)


# --------------------------------------------------------------------------- #
# constrained long-only optimizer (3c)
# --------------------------------------------------------------------------- #
@dataclass
class OptimizeResult:
    weights: np.ndarray
    status: str
    objective: Optional[float]
    used_fallback: bool
    note: str
    portfolio_vol: float
    cap_adjusted: Optional[Dict] = None  # set iff the per-name cap had to be raised


def optimize_long_only(
    mu: np.ndarray,
    Sigma: np.ndarray,
    order: List[str],
    sector_of: Dict[str, str],
    *,
    risk_aversion: float = RISK_AVERSION,
    min_weight: float = config.MIN_WEIGHT,
    max_weight: float = MAX_WEIGHT,
    max_active_deviation: float = config.MAX_ACTIVE_DEVIATION,
    sector_cap: float = SECTOR_CAP,
    vol_cap: Optional[float] = VOL_CAP,
    benchmark_weights: Optional[np.ndarray] = None,
    active_weight_budget: float = config.ACTIVE_WEIGHT_BUDGET,
    benchmark_penalty: float = config.BENCHMARK_PENALTY,
    concentration_penalty: float = config.CONCENTRATION_PENALTY,
) -> OptimizeResult:
    """Maximise mu^T w - (delta/2) w^T Sigma w under the long-only risk gate.

    Constraints: min_weight<=w<=max_weight, sum w=1, per-name active
    deviation<=max_active_deviation, total active L1<=active_weight_budget,
    and per-sector<=sector_cap. If ``vol_cap`` is explicitly supplied, it is
    added as a hard mandate; the research default is no absolute-volatility cap.
    Infeasible or failed solves use a loudly logged min-variance fallback.
    """
    if cp is None:
        raise RuntimeError("cvxpy is required for optimize_long_only")
    n = len(order)
    Sig = 0.5 * (Sigma + Sigma.T)

    if benchmark_weights is None:
        benchmark = np.ones(n, dtype=float) / n
    else:
        benchmark = np.asarray(benchmark_weights, dtype=float).reshape(-1)
        if benchmark.shape != (n,):
            raise ValueError(
                f"benchmark_weights must have shape ({n},), got {benchmark.shape}"
            )
        if not np.all(np.isfinite(benchmark)) or np.any(benchmark < 0):
            raise ValueError("benchmark_weights must be finite and non-negative")
        benchmark_sum = float(benchmark.sum())
        if benchmark_sum <= 0:
            raise ValueError("benchmark_weights must sum to a positive value")
        benchmark = benchmark / benchmark_sum

    # Feasibility guards. These preserve the requested risk limits whenever they
    # are compatible with the selected benchmark, and make any necessary change
    # explicit in the run log instead of silently falling back to equal weight.
    adjustments = {}

    effective_min = max(0.0, float(min_weight))
    if effective_min * n > 1.0 + 1e-9:
        adjusted = max(0.0, (1.0 / n) - 1e-6)
        adjustments["min_weight"] = {
            "configured": round(effective_min, 4),
            "effective": round(adjusted, 4),
            "n": n,
            "reason": "MIN_WEIGHT * N exceeds 1; lowered for feasibility",
        }
        effective_min = adjusted

    effective_max = float(max_weight)
    if effective_max < 1.0 / n + 1e-9:
        adjusted = min(1.0, max(effective_max, 1.5 / n))
        adjustments["max_weight"] = {
            "configured": round(effective_max, 4),
            "effective": round(adjusted, 4),
            "n": n,
            "min_feasible": round(1.0 / n, 4),
            "reason": "MAX_WEIGHT below 1/N; raised for feasibility",
        }
        effective_max = adjusted

    if effective_min > effective_max:
        adjusted = max(0.0, min(effective_min, (1.0 / n) - 1e-6))
        adjustments["min_weight"] = {
            "configured": round(float(min_weight), 4),
            "effective": round(adjusted, 4),
            "reason": "MIN_WEIGHT exceeded effective MAX_WEIGHT; lowered",
        }
        effective_min = adjusted

    # A per-name active bound can conflict with hard min/max weights when the
    # benchmark is concentrated. Equal-weight research portfolios need no
    # adjustment; market-cap experiments get the smallest explicit relaxation.
    effective_max_active_deviation = max(0.0, float(max_active_deviation))
    required_active_deviation = max(
        float(np.max(np.maximum(benchmark - effective_max, 0.0))),
        float(np.max(np.maximum(effective_min - benchmark, 0.0))),
    )
    if required_active_deviation > effective_max_active_deviation + 1e-9:
        adjusted = required_active_deviation + 1e-6
        adjustments["max_active_deviation"] = {
            "configured": round(effective_max_active_deviation, 4),
            "effective": round(adjusted, 4),
            "min_required_by_weight_bounds": round(required_active_deviation, 4),
            "reason": "benchmark conflicts with min/max weight bounds",
        }
        effective_max_active_deviation = adjusted

    # Minimum total L1 movement required to bring the benchmark inside the hard
    # box. Because active weights sum to zero, the minimum L1 is twice the larger
    # of required total overweight reduction and required total underweight lift.
    excess = float(np.maximum(benchmark - effective_max, 0.0).sum())
    deficit = float(np.maximum(effective_min - benchmark, 0.0).sum())
    min_l1_for_weight_bounds = 2.0 * max(excess, deficit)
    effective_active_budget = float(active_weight_budget)
    if min_l1_for_weight_bounds > effective_active_budget + 1e-9:
        adjusted = min(2.0, min_l1_for_weight_bounds + 1e-6)
        adjustments["active_weight_budget"] = {
            "configured": round(effective_active_budget, 4),
            "effective": round(adjusted, 4),
            "min_required_by_weight_bounds": round(min_l1_for_weight_bounds, 4),
            "reason": "benchmark conflicts with hard min/max weights",
        }
        effective_active_budget = adjusted

    # ``Unknown`` is not a real sector. Grouping every unmapped runtime ticker
    # into one Unknown bucket and applying SECTOR_CAP to that bucket makes
    # sum(w)=1 incompatible with Unknown<=0.40. Skip the sector cap for unmapped
    # names; per-name and active-weight limits still constrain them. The batch
    # runner now validates known research tickers, but this guard prevents a
    # silent equal-weight fallback for any future custom ticker.
    unknown_sector_tickers = [
        t for t in order
        if not sector_of.get(t) or str(sector_of.get(t)).strip().lower() == "unknown"
    ]
    if unknown_sector_tickers:
        adjustments["unknown_sector_tickers"] = {
            "tickers": list(unknown_sector_tickers),
            "reason": "sector cap skipped for unmapped tickers",
        }

    cap_adjusted = adjustments or None
    relax_bits = []
    if "min_weight" in adjustments:
        relax_bits.append(f"min adjusted {min_weight:.3f}->{effective_min:.3f}")
    if "max_weight" in adjustments:
        relax_bits.append(f"cap raised {max_weight:.3f}->{effective_max:.3f}")
    if "max_active_deviation" in adjustments:
        relax_bits.append(
            f"active/name raised {max_active_deviation:.3f}->{effective_max_active_deviation:.3f}"
        )
    if "active_weight_budget" in adjustments:
        relax_bits.append(
            f"active budget raised {active_weight_budget:.3f}->{effective_active_budget:.3f}"
        )
    if unknown_sector_tickers:
        relax_bits.append(
            "sector cap skipped for unmapped=" + ",".join(unknown_sector_tickers)
        )
    relax_note = (" [" + "; ".join(relax_bits) + "]") if relax_bits else ""

    sectors: Dict[str, List[int]] = {}
    for i, t in enumerate(order):
        sector = sector_of.get(t)
        if not sector or str(sector).strip().lower() == "unknown":
            continue
        sectors.setdefault(str(sector), []).append(i)

    def base_constraints(w):
        active_weights = w - benchmark
        cons = [
            w >= effective_min,
            cp.sum(w) == 1,
            w <= effective_max,
            cp.abs(active_weights) <= effective_max_active_deviation,
            # Two-sided active L1 budget: total overweight must be matched by
            # the same total underweight because portfolio weights sum to one.
            cp.norm1(active_weights) <= effective_active_budget,
        ]
        for _, idxs in sectors.items():
            cons.append(cp.sum(w[idxs]) <= sector_cap)
        return cons

    # Primary objective: expected return minus total risk, benchmark departure,
    # and concentration. The hard active budget prevents pathological portfolios;
    # the soft penalties make boundary solutions uncommon.
    w = cp.Variable(n)
    active_weights = w - benchmark
    util = (
        mu @ w
        - 0.5 * risk_aversion * cp.quad_form(w, cp.psd_wrap(Sig))
        - benchmark_penalty * cp.sum_squares(active_weights)
        - concentration_penalty * cp.sum_squares(w)
    )
    cons = base_constraints(w)
    if vol_cap is not None:
        if not np.isfinite(float(vol_cap)) or float(vol_cap) <= 0:
            raise ValueError("vol_cap must be a positive finite number or None")
        cons.append(cp.quad_form(w, cp.psd_wrap(Sig)) <= float(vol_cap) ** 2)
    prob = cp.Problem(cp.Maximize(util), cons)
    try:
        prob.solve(solver=cp.CLARABEL)
    except Exception:
        prob.solve()  # let cvxpy pick a solver

    if w.value is not None and prob.status in ("optimal", "optimal_inaccurate"):
        wv = _clean_weights(np.asarray(w.value).reshape(-1))
        vol = float(np.sqrt(max(wv @ Sig @ wv, 0.0)))
        objective_note = "utility_max+min_max_weight+per_name_active+active_budget+benchmark_penalty"
        if vol_cap is not None:
            objective_note += "+explicit_vol_cap"
        return OptimizeResult(
            wv, prob.status, float(prob.value), False,
            objective_note + relax_note, vol, cap_adjusted,
        )

    # Fallback: minimum variance under the base investability constraints.
    # With the research-default vol_cap=None this is only a numerical/base-gate
    # fallback; it is no longer triggered by an arbitrary absolute-volatility cliff.
    w2 = cp.Variable(n)
    prob2 = cp.Problem(cp.Minimize(cp.quad_form(w2, cp.psd_wrap(Sig))), base_constraints(w2))
    try:
        prob2.solve(solver=cp.CLARABEL)
    except Exception:
        prob2.solve()
    if w2.value is not None and prob2.status in ("optimal", "optimal_inaccurate"):
        wv = _clean_weights(np.asarray(w2.value).reshape(-1))
        vol = float(np.sqrt(max(wv @ Sig @ wv, 0.0)))
        return OptimizeResult(
            wv, prob2.status, None, True,
            f"INFEASIBLE primary ({prob.status}); min-variance fallback" + relax_note, vol,
            cap_adjusted,
        )

    # last resort: equal weight (loudly flagged)
    wv = np.ones(n) / n
    vol = float(np.sqrt(max(wv @ Sig @ wv, 0.0)))
    return OptimizeResult(wv, "fallback_equal_weight", None, True,
                          "both QPs infeasible; equal-weight", vol, cap_adjusted)


def _clean_weights(w: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    w = np.where(w < floor, 0.0, w)
    s = w.sum()
    return w / s if s > 0 else np.ones_like(w) / len(w)
