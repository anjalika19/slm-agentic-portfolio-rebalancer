#!/usr/bin/env python3
"""Replay saved model views through Black--Litterman without new model calls.

The online runs may use a skill-gated calibration file that shrinks every view
close to zero.  This utility recovers the *stored raw* stock opinions, applies
one common architecture-neutral transformation to every model configuration,
and reruns only:

    raw stock views -> adjacent pairwise BL views -> BL posterior -> optimizer

It then recomputes transaction-cost-aware monthly returns and exports analysis-
ready tables at the stock-month, portfolio-month, and portfolio levels.

The default rule intentionally matches the project's pre-existing no-file
fallback:

* q = raw_q * 0.25 for every model and architecture;
* confidence uses the same conservative default mapping for every view;
* the saved point-in-time covariance matrix, prior, realized returns, risk-free
  return, optimizer limits, and transaction cost are preserved.

No Gemma, Gemini, data-provider, news, or debate calls are made.

Examples
--------
Replay the latest completed backtest for each portfolio and factorial cell::

    python scripts/research/replay_fixed_scaling.py \
      --backtests 'runs/backtest_*.json' \
      --out-dir runs/fixed_scaling_replay

Process explicit files, including an incomplete set while LLM-multi is running::

    python scripts/research/replay_fixed_scaling.py \
      --backtests runs/backtest_*_gemma_single_*.json \
                  runs/backtest_*_gemma_debate_*.json \
                  runs/backtest_*_gemini_c1_*.json \
                  runs/backtest_*_gemini_debate_*.json
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from agentic_bl import bl_core
from agentic_bl import calibration
from agentic_bl import config
from agentic_bl.metrics import summarize


MODE_TO_CELL = {
    "gemma_single": "slm_single",
    "gemma_debate": "slm_multi",
    "gemini_c1": "llm_single",
    "gemini_debate": "llm_multi",
}

CELL_TO_STRATEGY = {
    "slm_single": "single_gemma",
    "slm_multi": "debate",
    "llm_single": "single_gemini",
    "llm_multi": "multi_gemini",
}

STRATEGY_TO_CELL = {v: k for k, v in CELL_TO_STRATEGY.items()}

CELL_METADATA = {
    "slm_single": {"model_family": "gemma", "architecture": "single_agent"},
    "slm_multi": {"model_family": "gemma", "architecture": "multi_agent"},
    "llm_single": {"model_family": "gemini", "architecture": "single_agent"},
    "llm_multi": {"model_family": "gemini", "architecture": "multi_agent"},
}


@dataclass(frozen=True)
class ReplayRule:
    q_multiplier: float = 0.25
    confidence_mode: str = "conservative"
    fixed_confidence: float = 0.55
    view_mode: str = "absolute_return"
    eligibility_domain: str = "raw"
    raw_spread_floor: float = 0.0
    scaled_spread_floor: float = config.VIEW_SPREAD_FLOOR

    def transform(self, raw_q: float, raw_confidence: float) -> Tuple[float, float]:
        raw_q = _finite_float(raw_q, 0.0)
        raw_confidence = _finite_float(raw_confidence, 0.5)
        q = raw_q * self.q_multiplier

        if self.confidence_mode == "raw":
            confidence = raw_confidence
        elif self.confidence_mode == "fixed":
            confidence = self.fixed_confidence
        elif self.confidence_mode == "conservative":
            # Exact no-history behavior of calibration.ViewCalibrator.default():
            # high self-reported confidence cannot exceed the conservative prior,
            # while low confidence remains a downward veto.
            confidence = (
                config.DEFAULT_CONFIDENCE_CENTER
                + config.DEFAULT_CONFIDENCE_SLOPE * (raw_confidence - 0.5)
            )
            confidence = min(confidence, raw_confidence)
        else:  # pragma: no cover - argparse protects this
            raise ValueError(f"unknown confidence mode: {self.confidence_mode}")

        confidence = float(
            np.clip(
                confidence,
                config.MIN_VIEW_CONFIDENCE,
                config.MAX_VIEW_CONFIDENCE,
            )
        )
        return float(q), confidence




@dataclass
class _OptimizerOutput:
    weights: np.ndarray
    status: str
    used_fallback: bool
    note: str
    portfolio_vol: float
    cap_adjusted: Optional[Dict[str, Any]] = None


def _optimize_portfolio(
    mu: np.ndarray,
    sigma: np.ndarray,
    universe: Sequence[str],
    sectors: Mapping[str, str],
    benchmark: np.ndarray,
    saved: Mapping[str, Any],
) -> _OptimizerOutput:
    """Use the project's cvxpy optimizer, with a SciPy fallback for portability."""
    kwargs = {
        "risk_aversion": _config_float(saved, "RISK_AVERSION", config.RISK_AVERSION),
        "min_weight": _config_float(saved, "MIN_WEIGHT", config.MIN_WEIGHT),
        "max_weight": _config_float(saved, "MAX_WEIGHT", config.MAX_WEIGHT),
        "max_active_deviation": _config_float(
            saved, "MAX_ACTIVE_DEVIATION", config.MAX_ACTIVE_DEVIATION
        ),
        "sector_cap": _config_float(saved, "SECTOR_CAP", config.SECTOR_CAP),
        "vol_cap": _config_optional_float(saved, "VOL_CAP", config.VOL_CAP),
        "benchmark_weights": benchmark,
        "active_weight_budget": _config_float(
            saved, "ACTIVE_WEIGHT_BUDGET", config.ACTIVE_WEIGHT_BUDGET
        ),
        "benchmark_penalty": _config_float(
            saved, "BENCHMARK_PENALTY", config.BENCHMARK_PENALTY
        ),
        "concentration_penalty": _config_float(
            saved, "CONCENTRATION_PENALTY", config.CONCENTRATION_PENALTY
        ),
    }
    if getattr(bl_core, "cp", None) is not None:
        result = bl_core.optimize_long_only(
            mu, sigma, list(universe), dict(sectors), **kwargs
        )
        return _OptimizerOutput(
            weights=np.asarray(result.weights, dtype=float),
            status=str(result.status),
            used_fallback=bool(result.used_fallback),
            note=str(result.note),
            portfolio_vol=float(result.portfolio_vol),
            cap_adjusted=result.cap_adjusted,
        )

    # The user project installs cvxpy. This fallback makes replay usable in a
    # minimal analysis environment and solves the same smooth utility with the
    # same constraints through SLSQP.
    n = len(universe)
    risk_aversion = float(kwargs["risk_aversion"])
    min_weight = max(0.0, float(kwargs["min_weight"]))
    max_weight = float(kwargs["max_weight"])
    max_active = max(0.0, float(kwargs["max_active_deviation"]))
    active_budget = max(0.0, float(kwargs["active_weight_budget"]))
    sector_cap = float(kwargs["sector_cap"])
    vol_cap = kwargs["vol_cap"]
    benchmark_penalty = float(kwargs["benchmark_penalty"])
    concentration_penalty = float(kwargs["concentration_penalty"])

    lower = np.maximum(min_weight, benchmark - max_active)
    upper = np.minimum(max_weight, benchmark + max_active)
    if lower.sum() > 1.0 + 1e-9 or upper.sum() < 1.0 - 1e-9:
        raise ValueError("saved optimizer bounds are infeasible")
    bounds = list(zip(lower, upper))

    sector_indexes: Dict[str, List[int]] = {}
    for index, ticker in enumerate(universe):
        sector = sectors.get(ticker)
        if sector and str(sector).strip().lower() != "unknown":
            sector_indexes.setdefault(str(sector), []).append(index)

    def objective(weights: np.ndarray) -> float:
        active = weights - benchmark
        utility = (
            float(mu @ weights)
            - 0.5 * risk_aversion * float(weights @ sigma @ weights)
            - benchmark_penalty * float(active @ active)
            - concentration_penalty * float(weights @ weights)
        )
        return -utility

    constraints: List[Dict[str, Any]] = [
        {"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0)},
        {
            "type": "ineq",
            "fun": lambda weights: float(active_budget - np.abs(weights - benchmark).sum()),
        },
    ]
    for indexes in sector_indexes.values():
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda weights, idx=np.asarray(indexes, dtype=int): float(
                    sector_cap - weights[idx].sum()
                ),
            }
        )
    if vol_cap is not None:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda weights: float(
                    float(vol_cap) ** 2 - weights @ sigma @ weights
                ),
            }
        )

    result = minimize(
        objective,
        benchmark.copy(),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"SciPy optimizer failed: {result.message}")
    weights = np.asarray(result.x, dtype=float)
    weights = np.where(weights < 1e-10, 0.0, weights)
    weights /= weights.sum()
    volatility = float(np.sqrt(max(weights @ sigma @ weights, 0.0)))
    return _OptimizerOutput(
        weights=weights,
        status="optimal_scipy_slsqp",
        used_fallback=False,
        note="utility_max+saved_constraints (SciPy SLSQP portability fallback)",
        portfolio_vol=volatility,
    )


@dataclass
class ReplayResult:
    source_path: Path
    portfolio_id: str
    cell: str
    strategy: str
    model_family: str
    architecture: str
    universe: List[str]
    start: str
    end: str
    monthly_rows: List[Dict[str, Any]]
    stock_rows: List[Dict[str, Any]]
    pairwise_rows: List[Dict[str, Any]]
    weight_rows: List[Dict[str, Any]]
    summary_row: Dict[str, Any]
    json_output: Dict[str, Any]


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) else float(default)


def _events(log: Mapping[str, Any], event_type: str) -> List[Dict[str, Any]]:
    return [
        event
        for event in (log.get("events") or [])
        if event.get("type") == event_type
    ]


def _last_payload_by_ticker(
    log: Mapping[str, Any], event_type: str
) -> Dict[str, Dict[str, Any]]:
    output: Dict[str, Dict[str, Any]] = {}
    for event in _events(log, event_type):
        ticker = str(event.get("ticker") or "").upper()
        if ticker:
            output[ticker] = dict(event.get("payload") or {})
    return output


def _extract_single_views(
    log: Mapping[str, Any], universe: Sequence[str]
) -> Dict[str, Dict[str, Any]]:
    payloads = _last_payload_by_ticker(log, "single_model_emitted")
    views: Dict[str, Dict[str, Any]] = {}
    for ticker in universe:
        payload = payloads.get(ticker) or {}
        if payload.get("raw_q") is None:
            continue
        views[ticker] = {
            "raw_q": _finite_float(payload.get("raw_q")),
            "raw_confidence": _finite_float(payload.get("raw_confidence"), 0.5),
            "source": payload.get("source") or "single_model_emitted",
            "resolved": payload.get("resolved", True),
            "raw_direction": payload.get("signal_label") or payload.get("direction"),
            "rationale": payload.get("rationale"),
            "used_mock": bool(payload.get("used_mock", False)),
        }
    return views


def _extract_multi_views(
    log: Mapping[str, Any], universe: Sequence[str]
) -> Dict[str, Dict[str, Any]]:
    # Current matched multi-agent runs persist the final pre-calibration q and
    # confidence directly in final_view.  This is the cleanest replay boundary.
    payloads = _last_payload_by_ticker(log, "final_view")
    views: Dict[str, Dict[str, Any]] = {}
    for ticker in universe:
        payload = payloads.get(ticker) or {}
        raw_q = payload.get("raw_q")
        if raw_q is None:
            continue
        views[ticker] = {
            "raw_q": _finite_float(raw_q),
            "raw_confidence": _finite_float(payload.get("raw_confidence"), 0.5),
            "source": payload.get("source") or "final_view",
            "resolved": payload.get("resolved", True),
            "raw_direction": _direction(_finite_float(raw_q)),
            "rationale": payload.get("reasoning_summary"),
            "used_mock": bool(payload.get("used_mock", False)),
        }
    return views


def _extract_raw_views(
    log: Mapping[str, Any], architecture: str, universe: Sequence[str]
) -> Dict[str, Dict[str, Any]]:
    if architecture == "single_agent":
        return _extract_single_views(log, universe)
    if architecture == "multi_agent":
        return _extract_multi_views(log, universe)
    raise ValueError(f"unsupported architecture: {architecture}")


def _covariance(log: Mapping[str, Any], universe: Sequence[str]) -> np.ndarray:
    block = (
        (log.get("meta") or {})
        .get("agent_inputs", {})
        .get("quant", {})
        .get("covariance_matrix", {})
    )
    source_order = list(block.get("order") or universe)
    matrix = np.asarray(block.get("matrix"), dtype=float)
    if matrix.shape != (len(source_order), len(source_order)):
        raise ValueError("saved run log has no usable covariance matrix")
    try:
        indexes = [source_order.index(ticker) for ticker in universe]
    except ValueError as exc:
        raise ValueError("covariance matrix order does not cover the portfolio") from exc
    matrix = matrix[np.ix_(indexes, indexes)]
    return 0.5 * (matrix + matrix.T)


def _prior_weights(log: Mapping[str, Any], universe: Sequence[str]) -> np.ndarray:
    for event in _events(log, "baseline_view"):
        payload = event.get("payload") or {}
        if payload.get("type") != "bl_prior_weights":
            continue
        weights = payload.get("weights") or {}
        vector = np.asarray([_finite_float(weights.get(t)) for t in universe], dtype=float)
        if np.all(vector >= 0) and vector.sum() > 0:
            return vector / vector.sum()
    return np.ones(len(universe), dtype=float) / len(universe)


def _sectors(log: Mapping[str, Any], universe: Sequence[str]) -> Dict[str, str]:
    mapping = (
        (log.get("meta") or {})
        .get("agent_inputs", {})
        .get("economist", {})
        .get("universe_sector_mapping", {})
    )
    output = {}
    for ticker in universe:
        saved = mapping.get(ticker)
        if saved and str(saved).strip().lower() != "unknown":
            output[ticker] = str(saved)
        else:
            output[ticker] = config.SECTOR_BY_TICKER.get(ticker, "Unknown")
    return output


def _saved_config(log: Mapping[str, Any]) -> Dict[str, Any]:
    return dict((log.get("meta") or {}).get("config") or {})


def _config_float(saved: Mapping[str, Any], name: str, default: float) -> float:
    return _finite_float(saved.get(name), default)


def _config_optional_float(
    saved: Mapping[str, Any], name: str, default: Optional[float]
) -> Optional[float]:
    value = saved.get(name, default)
    if value is None or value == "":
        return None
    return _finite_float(value, default if default is not None else 0.0)


def _build_pairwise_views(
    transformed: Mapping[str, Mapping[str, Any]],
    universe: Sequence[str],
    pi: np.ndarray,
    rule: ReplayRule,
    *,
    max_pairs: int,
) -> List[Dict[str, Any]]:
    """Build adjacent pairwise rows while keeping ranking and BL semantics separate.

    ``raw_q`` is the saved model score.  It determines the ordinal ranking and,
    by default, whether two adjacent names are genuinely distinguishable.
    ``q`` is the commonly scaled score and controls the magnitude of the active
    view.  This prevents a small calibration multiplier from silently deleting
    a non-tied model ordering.

    Two BL interpretations are supported:

    ``absolute_return``
        The scaled long-short spread is the complete BL view Q. This reproduces
        the original fixed-scaling replay.

    ``prior_relative_alpha``
        The scaled long-short spread is an active alpha adjustment relative to
        the equilibrium prior, so Q = P*pi + alpha. This guarantees that a
        positive model ordering asks BL to increase the long-over-short spread
        relative to the prior rather than accidentally weaken it.
    """
    order_index = {ticker: index for index, ticker in enumerate(universe)}
    ranked: List[Tuple[str, Mapping[str, Any], float, float]] = []
    for ticker in universe:
        view = transformed.get(ticker) or {}
        if view.get("resolved") is False:
            continue
        raw_q = _finite_float(view.get("raw_q"), float("nan"))
        scaled_q = _finite_float(view.get("q"), float("nan"))
        if not np.isfinite(raw_q) or not np.isfinite(scaled_q):
            continue
        ranked.append((ticker, view, raw_q, scaled_q))

    # Ranking comes from the model output itself, not from the prior and not from
    # confidence. A common positive multiplier leaves this ordering unchanged.
    ranked.sort(key=lambda item: (-item[2], order_index[item[0]]))
    if len(ranked) < 2:
        return []

    natural_count = len(ranked) - 1
    limit = natural_count if max_pairs <= 0 else min(max_pairs, natural_count)
    pairwise: List[Dict[str, Any]] = []

    for rank_index, (long_item, short_item) in enumerate(
        zip(ranked[:-1], ranked[1:]), start=1
    ):
        if len(pairwise) >= limit:
            break
        long_ticker, long_view, long_raw_q, long_scaled_q = long_item
        short_ticker, short_view, short_raw_q, short_scaled_q = short_item
        raw_spread = max(0.0, float(long_raw_q - short_raw_q))
        scaled_spread = max(0.0, float(long_scaled_q - short_scaled_q))

        if rule.eligibility_domain == "raw":
            eligibility_spread = raw_spread
            eligibility_floor = rule.raw_spread_floor
        elif rule.eligibility_domain == "scaled":
            eligibility_spread = scaled_spread
            eligibility_floor = rule.scaled_spread_floor
        else:  # pragma: no cover - argparse validates
            raise ValueError(f"unknown eligibility domain: {rule.eligibility_domain}")

        # A zero spread is a tie, not a useful Q=0 equality view. Preserve the
        # historical floor convention: a spread exactly equal to a positive
        # floor survives, while a strictly smaller spread is removed.
        if eligibility_spread <= 1e-12 or eligibility_spread + 1e-12 < eligibility_floor:
            continue

        p_row = np.asarray(
            bl_core.p_row_relative([long_ticker], [short_ticker], list(universe)),
            dtype=float,
        )
        prior_spread = float(p_row @ np.asarray(pi, dtype=float))
        model_alpha_spread = scaled_spread
        if rule.view_mode == "absolute_return":
            bl_total_q = model_alpha_spread
        elif rule.view_mode == "prior_relative_alpha":
            bl_total_q = prior_spread + model_alpha_spread
        else:  # pragma: no cover - argparse validates
            raise ValueError(f"unknown view mode: {rule.view_mode}")

        long_confidence = _finite_float(long_view.get("confidence"), 0.5)
        short_confidence = _finite_float(short_view.get("confidence"), 0.5)
        confidence = float(
            np.clip(
                0.5 * (long_confidence + short_confidence),
                config.MIN_VIEW_CONFIDENCE,
                config.MAX_VIEW_CONFIDENCE,
            )
        )
        pairwise.append(
            {
                "long_ticker": long_ticker,
                "short_ticker": short_ticker,
                "q": float(bl_total_q),
                "bl_total_q": float(bl_total_q),
                "raw_q": raw_spread,
                "raw_spread": raw_spread,
                "scaled_spread": scaled_spread,
                "model_alpha_spread": model_alpha_spread,
                "prior_spread": prior_spread,
                "q_minus_prior": float(bl_total_q - prior_spread),
                "confidence": confidence,
                "p_row": p_row.tolist(),
                "view_type": "adjacent_chain_from_stock_views",
                "view_mode": rule.view_mode,
                "eligibility_domain": rule.eligibility_domain,
                "eligibility_spread": eligibility_spread,
                "eligibility_floor": eligibility_floor,
                "source": "fixed_common_scaling_adjacent_chain",
                "rank_long": rank_index,
                "rank_short": rank_index + 1,
                "long_source": long_view.get("source"),
                "short_source": short_view.get("source"),
                "long_stock_q": long_scaled_q,
                "short_stock_q": short_scaled_q,
                "long_raw_q": long_raw_q,
                "short_raw_q": short_raw_q,
            }
        )
    return pairwise


def _replay_month(
    log: Mapping[str, Any],
    universe: Sequence[str],
    architecture: str,
    rule: ReplayRule,
) -> Dict[str, Any]:
    raw_views = _extract_raw_views(log, architecture, universe)
    missing = [ticker for ticker in universe if ticker not in raw_views]
    if missing:
        raise ValueError(f"saved run is missing raw stock views for {missing}")

    transformed: Dict[str, Dict[str, Any]] = {}
    for ticker in universe:
        raw = raw_views[ticker]
        q, confidence = rule.transform(raw["raw_q"], raw["raw_confidence"])
        transformed[ticker] = {
            "ticker": ticker,
            "q": q,
            "confidence": confidence,
            "raw_q": raw["raw_q"],
            "raw_confidence": raw["raw_confidence"],
            "source": raw.get("source"),
            "resolved": raw.get("resolved", True),
        }

    saved = _saved_config(log)
    max_pairs = int(saved.get("MAX_PAIRWISE_VIEWS", config.MAX_PAIRWISE_VIEWS) or 0)
    sigma = _covariance(log, universe)
    benchmark = _prior_weights(log, universe)
    risk_aversion = _config_float(saved, "RISK_AVERSION", config.RISK_AVERSION)
    tau = _config_float(saved, "BL_TAU", config.BL_TAU)
    pi = bl_core.equilibrium_returns(sigma, benchmark, risk_aversion)

    pairwise = _build_pairwise_views(
        transformed,
        universe,
        pi,
        rule,
        max_pairs=max_pairs,
    )

    if pairwise:
        p = np.asarray([view["p_row"] for view in pairwise], dtype=float)
        q = np.asarray([view["q"] for view in pairwise], dtype=float)
        confidences = np.asarray(
            [view["confidence"] for view in pairwise], dtype=float
        )
        omega = bl_core.idzorek_omega(p, sigma, confidences, tau=tau)
        posterior = bl_core.bl_posterior(pi, sigma, p, q, omega, tau=tau)
        omega_diag = np.diag(omega)
    else:
        p = np.zeros((0, len(universe)), dtype=float)
        q = np.zeros(0, dtype=float)
        omega_diag = np.zeros(0, dtype=float)
        posterior = bl_core.BLPosterior(
            mu_bl=pi.copy(), cov_bl=sigma.copy(), pi=pi.copy()
        )

    result = _optimize_portfolio(
        posterior.mu_bl,
        sigma,
        universe,
        _sectors(log, universe),
        benchmark,
        saved,
    )

    weights = {
        ticker: float(result.weights[index])
        for index, ticker in enumerate(universe)
    }
    benchmark_weights = {
        ticker: float(benchmark[index])
        for index, ticker in enumerate(universe)
    }

    aligned_count = 0
    for view in pairwise:
        long_ticker = view["long_ticker"]
        short_ticker = view["short_ticker"]
        active_tilt = (
            weights[long_ticker]
            - benchmark_weights[long_ticker]
            - weights[short_ticker]
            + benchmark_weights[short_ticker]
        )
        view["active_long_short_tilt"] = float(active_tilt)
        view["active_tilt_aligned"] = bool(active_tilt > 1e-10)
        aligned_count += int(view["active_tilt_aligned"])

    alignment_rate = (
        float(aligned_count / len(pairwise)) if pairwise else None
    )
    return {
        "weights": weights,
        "benchmark_weights": benchmark_weights,
        "raw_stock_views": raw_views,
        "transformed_stock_views": transformed,
        "pairwise_views": pairwise,
        "bl_inputs": {
            "P": p.tolist(),
            "Q": q.tolist(),
            "omega_diag": omega_diag.tolist(),
            "pi": pi.tolist(),
            "mu_bl": posterior.mu_bl.tolist(),
        },
        "optimizer": {
            "status": result.status,
            "used_fallback": bool(result.used_fallback),
            "note": result.note,
            "portfolio_vol": float(result.portfolio_vol),
            "cap_adjusted": result.cap_adjusted,
        },
        "view_diagnostics": {
            "view_mode": rule.view_mode,
            "eligibility_domain": rule.eligibility_domain,
            "raw_spread_floor": rule.raw_spread_floor,
            "scaled_spread_floor": rule.scaled_spread_floor,
            "pairwise_alignment_rate": alignment_rate,
        },
    }

def _turnover(
    previous: Mapping[str, float], current: Mapping[str, float], universe: Sequence[str]
) -> float:
    return float(
        sum(abs(current.get(ticker, 0.0) - previous.get(ticker, 0.0)) for ticker in universe)
    )


def _drift_weights(
    weights: Mapping[str, float],
    realized_returns: Mapping[str, float],
    universe: Sequence[str],
) -> Dict[str, float]:
    grown = {
        ticker: float(weights.get(ticker, 0.0))
        * (1.0 + float(realized_returns[ticker]))
        for ticker in universe
    }
    total = float(sum(grown.values()))
    if not np.isfinite(total) or total <= 0:
        raise ValueError("cannot compute drifted weights")
    return {ticker: grown[ticker] / total for ticker in universe}


def _direction(value: float, tolerance: float = 1e-12) -> str:
    if value > tolerance:
        return "long"
    if value < -tolerance:
        return "short"
    return "neutral"


def _rankdata(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=float)
    index = 0
    while index < len(array):
        end = index + 1
        while end < len(array) and array[order[end]] == array[order[index]]:
            end += 1
        ranks[order[index:end]] = 0.5 * (index + end - 1)
        index = end
    return ranks


def _correlation(x: Sequence[float], y: Sequence[float], *, rank: bool = False) -> Optional[float]:
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    if len(xx) < 3 or len(xx) != len(yy):
        return None
    if rank:
        xx = _rankdata(xx)
        yy = _rankdata(yy)
    if np.allclose(xx, xx[0]) or np.allclose(yy, yy[0]):
        return None
    result = float(np.corrcoef(xx, yy)[0, 1])
    return result if np.isfinite(result) else None


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(clean)) if clean else None


def _std(values: Sequence[float], ddof: int = 1) -> Optional[float]:
    if len(values) <= ddof:
        return None
    result = float(np.std(np.asarray(values, dtype=float), ddof=ddof))
    return result if np.isfinite(result) else None


def _active_metrics(strategy_returns: Sequence[float], benchmark_returns: Sequence[float]) -> Dict[str, Any]:
    active = np.asarray(strategy_returns, dtype=float) - np.asarray(benchmark_returns, dtype=float)
    tracking_error = float(active.std(ddof=1) * math.sqrt(12.0)) if len(active) > 1 else 0.0
    information_ratio = (
        float(active.mean() / active.std(ddof=1) * math.sqrt(12.0))
        if len(active) > 1 and active.std(ddof=1) > 0
        else 0.0
    )
    return {
        "mean_monthly_active_return": float(active.mean()) if len(active) else 0.0,
        "annualized_active_return": float(active.mean() * 12.0) if len(active) else 0.0,
        "tracking_error": tracking_error,
        "information_ratio": information_ratio,
        "monthly_win_rate_vs_equal_weight": float(np.mean(active > 0)) if len(active) else 0.0,
        "worst_monthly_active_return": float(active.min()) if len(active) else 0.0,
        "best_monthly_active_return": float(active.max()) if len(active) else 0.0,
    }


def _resolve_cell(backtest: Mapping[str, Any]) -> Tuple[str, str]:
    mode = str(backtest.get("run_mode") or "")
    if mode in MODE_TO_CELL:
        cell = MODE_TO_CELL[mode]
        return cell, CELL_TO_STRATEGY[cell]

    strategies = list(backtest.get("model_strategies") or [])
    if len(strategies) == 1 and strategies[0] in STRATEGY_TO_CELL:
        strategy = strategies[0]
        return STRATEGY_TO_CELL[strategy], strategy

    # Compatibility with the older dedicated outputs.
    if backtest.get("run_debate") and not backtest.get("run_gemini"):
        return "slm_multi", "debate"
    if backtest.get("run_gemini") and not backtest.get("run_debate"):
        return "llm_single", "single_gemini"
    raise ValueError(
        f"cannot identify a single factorial cell from run_mode={mode!r}; "
        "use one-mode backtest files rather than compare/factorial summaries"
    )


def _embedded_logs_by_date(
    backtest: Mapping[str, Any], strategy: str
) -> Dict[str, Dict[str, Any]]:
    output: Dict[str, Dict[str, Any]] = {}
    accepted_kinds = {strategy}
    # Legacy Gemma multi files call the strategy simply "debate".
    if strategy == "debate":
        accepted_kinds.add("debate")
    for entry in backtest.get("embedded_run_logs") or []:
        kind = str(entry.get("kind") or "")
        if kind not in accepted_kinds:
            continue
        log = entry.get("log") or {}
        date = str(entry.get("as_of_date") or log.get("as_of_date") or "")
        if date and log:
            output[date] = log
    return output


def replay_backtest(path: Path, rule: ReplayRule) -> ReplayResult:
    backtest = json.loads(path.read_text())
    cell, strategy = _resolve_cell(backtest)
    metadata = CELL_METADATA[cell]
    universe = [str(ticker).upper() for ticker in backtest.get("universe") or []]
    if not universe:
        raise ValueError(f"{path.name}: empty universe")

    logs_by_date = _embedded_logs_by_date(backtest, strategy)
    months = list(backtest.get("per_month") or [])
    if not months:
        raise ValueError(f"{path.name}: no per_month rows")

    portfolio_id = str(backtest.get("portfolio_id") or path.stem)
    cost_rate = _finite_float(backtest.get("txn_cost_bps"), config.TXN_COST_BPS) / 10000.0
    previous_strategy: Dict[str, float] = {}
    previous_benchmark: Dict[str, float] = {}

    gross_returns: List[float] = []
    net_returns: List[float] = []
    turnovers: List[float] = []
    benchmark_gross: List[float] = []
    benchmark_net: List[float] = []
    benchmark_turnovers: List[float] = []
    risk_free: List[float] = []
    monthly_rows: List[Dict[str, Any]] = []
    stock_rows: List[Dict[str, Any]] = []
    pairwise_rows: List[Dict[str, Any]] = []
    weight_rows: List[Dict[str, Any]] = []
    replay_months: List[Dict[str, Any]] = []
    cumulative_net = 1.0
    cumulative_benchmark_net = 1.0

    for month_index, month in enumerate(months, start=1):
        as_of = str(month.get("as_of_date") or "")
        log = logs_by_date.get(as_of)
        if not log:
            raise ValueError(f"{path.name} {as_of}: missing embedded {strategy} log")
        realized = {
            ticker: _finite_float((month.get("realized_returns") or {}).get(ticker))
            for ticker in universe
        }
        if set(realized) != set(universe):  # pragma: no cover - defensive
            raise ValueError(f"{path.name} {as_of}: incomplete realized returns")

        replayed = _replay_month(
            log,
            universe,
            metadata["architecture"],
            rule,
        )
        weights = replayed["weights"]
        benchmark = replayed["benchmark_weights"]

        gross_return = float(sum(weights[t] * realized[t] for t in universe))
        benchmark_gross_return = float(
            sum(benchmark[t] * realized[t] for t in universe)
        )
        turnover = _turnover(previous_strategy, weights, universe)
        benchmark_turnover = _turnover(previous_benchmark, benchmark, universe)
        net_return = gross_return - cost_rate * turnover
        benchmark_net_return = benchmark_gross_return - cost_rate * benchmark_turnover
        previous_strategy = _drift_weights(weights, realized, universe)
        previous_benchmark = _drift_weights(benchmark, realized, universe)
        month_rf = _finite_float(month.get("risk_free_return"), 0.0)

        gross_returns.append(gross_return)
        net_returns.append(net_return)
        turnovers.append(turnover)
        benchmark_gross.append(benchmark_gross_return)
        benchmark_net.append(benchmark_net_return)
        benchmark_turnovers.append(benchmark_turnover)
        risk_free.append(month_rf)
        cumulative_net *= 1.0 + net_return
        cumulative_benchmark_net *= 1.0 + benchmark_net_return

        raw_values: List[float] = []
        scaled_values: List[float] = []
        realized_targets: List[float] = []
        hits: List[int] = []
        active_share = 0.5 * sum(abs(weights[t] - benchmark[t]) for t in universe)

        for view_index, view in enumerate(replayed["pairwise_views"], start=1):
            long_ticker = str(view["long_ticker"])
            short_ticker = str(view["short_ticker"])
            realized_pair_monthly = realized[long_ticker] - realized[short_ticker]
            pairwise_rows.append(
                {
                    "source_backtest": str(path),
                    "portfolio_id": portfolio_id,
                    "cell": cell,
                    "strategy": strategy,
                    "model_family": metadata["model_family"],
                    "architecture": metadata["architecture"],
                    "month_index": month_index,
                    "as_of_date": as_of,
                    "hold_end": month.get("hold_end"),
                    "view_index": view_index,
                    "long_ticker": long_ticker,
                    "short_ticker": short_ticker,
                    "view_mode": rule.view_mode,
                    "eligibility_domain": rule.eligibility_domain,
                    "raw_spread": view["raw_spread"],
                    "scaled_spread": view["scaled_spread"],
                    "model_alpha_spread": view["model_alpha_spread"],
                    "prior_spread": view["prior_spread"],
                    "bl_total_q": view["bl_total_q"],
                    "q_minus_prior": view["q_minus_prior"],
                    "confidence": view["confidence"],
                    "active_long_short_tilt": view["active_long_short_tilt"],
                    "active_tilt_aligned": view["active_tilt_aligned"],
                    "realized_long_short_monthly": realized_pair_monthly,
                    "realized_long_short_annualized": 12.0 * realized_pair_monthly,
                    "direction_hit": int(realized_pair_monthly > 0.0)
                    if abs(realized_pair_monthly) > 1e-12
                    else None,
                }
            )

        for ticker in universe:
            raw = replayed["raw_stock_views"][ticker]
            transformed = replayed["transformed_stock_views"][ticker]
            rest = [realized[other] for other in universe if other != ticker]
            realized_vs_rest_monthly = realized[ticker] - float(np.mean(rest))
            realized_y = 12.0 * realized_vs_rest_monthly
            raw_q = float(raw["raw_q"])
            scaled_q = float(transformed["q"])
            is_directional = abs(raw_q) > 1e-12 and abs(realized_y) > 1e-12
            hit = int(np.sign(raw_q) == np.sign(realized_y)) if is_directional else None
            if hit is not None:
                hits.append(hit)
            raw_values.append(raw_q)
            scaled_values.append(scaled_q)
            realized_targets.append(realized_y)

            contribution = weights[ticker] * realized[ticker]
            benchmark_contribution = benchmark[ticker] * realized[ticker]
            stock_rows.append(
                {
                    "source_backtest": str(path),
                    "portfolio_id": portfolio_id,
                    "cell": cell,
                    "strategy": strategy,
                    "model_family": metadata["model_family"],
                    "architecture": metadata["architecture"],
                    "month_index": month_index,
                    "as_of_date": as_of,
                    "hold_end": month.get("hold_end"),
                    "ticker": ticker,
                    "raw_q_annualized": raw_q,
                    "scaled_q_annualized": scaled_q,
                    "raw_confidence": raw["raw_confidence"],
                    "replay_confidence": transformed["confidence"],
                    "raw_direction": _direction(raw_q),
                    "source": raw.get("source"),
                    "resolved": raw.get("resolved", True),
                    "used_mock": raw.get("used_mock", False),
                    "realized_stock_return": realized[ticker],
                    "realized_vs_rest_monthly": realized_vs_rest_monthly,
                    "realized_vs_rest_annualized": realized_y,
                    "direction_hit": hit,
                    "absolute_forecast_error": abs(raw_q - realized_y),
                    "weight": weights[ticker],
                    "benchmark_weight": benchmark[ticker],
                    "active_weight": weights[ticker] - benchmark[ticker],
                    "return_contribution": contribution,
                    "benchmark_return_contribution": benchmark_contribution,
                    "active_return_contribution": contribution - benchmark_contribution,
                }
            )
            weight_rows.append(
                {
                    "portfolio_id": portfolio_id,
                    "cell": cell,
                    "strategy": strategy,
                    "as_of_date": as_of,
                    "hold_end": month.get("hold_end"),
                    "ticker": ticker,
                    "weight": weights[ticker],
                    "benchmark_weight": benchmark[ticker],
                    "active_weight": weights[ticker] - benchmark[ticker],
                }
            )

        rank_ic = _correlation(raw_values, realized_targets, rank=True)
        pearson_ic = _correlation(raw_values, realized_targets, rank=False)
        monthly_hit_rate = float(np.mean(hits)) if hits else None
        monthly_mae = float(
            np.mean(np.abs(np.asarray(raw_values) - np.asarray(realized_targets)))
        )
        active_return = net_return - benchmark_net_return
        monthly_row = {
            "source_backtest": str(path),
            "portfolio_id": portfolio_id,
            "universe": ",".join(universe),
            "cell": cell,
            "strategy": strategy,
            "model_family": metadata["model_family"],
            "architecture": metadata["architecture"],
            "month_index": month_index,
            "as_of_date": as_of,
            "hold_end": month.get("hold_end"),
            "gross_return": gross_return,
            "net_return": net_return,
            "turnover": turnover,
            "risk_free_return": month_rf,
            "excess_return_over_rf": net_return - month_rf,
            "equal_weight_gross_return": benchmark_gross_return,
            "equal_weight_net_return": benchmark_net_return,
            "equal_weight_turnover": benchmark_turnover,
            "active_return_vs_equal_weight": active_return,
            "cumulative_net_return": cumulative_net - 1.0,
            "cumulative_equal_weight_net_return": cumulative_benchmark_net - 1.0,
            "cumulative_active_return": cumulative_net - cumulative_benchmark_net,
            "active_share": active_share,
            "n_pairwise_views": len(replayed["pairwise_views"]),
            "view_mode": rule.view_mode,
            "eligibility_domain": rule.eligibility_domain,
            "raw_spread_floor": rule.raw_spread_floor,
            "scaled_spread_floor": rule.scaled_spread_floor,
            "pairwise_alignment_rate": replayed["view_diagnostics"][
                "pairwise_alignment_rate"
            ],
            "mean_pairwise_raw_spread": _mean(
                float(view["raw_spread"])
                for view in replayed["pairwise_views"]
            ),
            "mean_pairwise_model_alpha_spread": _mean(
                float(view["model_alpha_spread"])
                for view in replayed["pairwise_views"]
            ),
            "mean_pairwise_prior_spread": _mean(
                float(view["prior_spread"])
                for view in replayed["pairwise_views"]
            ),
            "mean_pairwise_q_minus_prior": _mean(
                float(view["q_minus_prior"])
                for view in replayed["pairwise_views"]
            ),
            "optimizer_status": replayed["optimizer"]["status"],
            "optimizer_used_fallback": replayed["optimizer"]["used_fallback"],
            "portfolio_volatility": replayed["optimizer"]["portfolio_vol"],
            "raw_q_mean": float(np.mean(raw_values)),
            "raw_q_std": float(np.std(raw_values)),
            "raw_q_min": float(np.min(raw_values)),
            "raw_q_max": float(np.max(raw_values)),
            "scaled_q_std": float(np.std(scaled_values)),
            "directional_hit_rate": monthly_hit_rate,
            "spearman_rank_ic": rank_ic,
            "pearson_ic": pearson_ic,
            "forecast_mae_annualized": monthly_mae,
            "directional_coverage": float(np.mean(np.abs(raw_values) > 1e-12)),
        }
        monthly_rows.append(monthly_row)
        replay_months.append(
            {
                "as_of_date": as_of,
                "hold_end": month.get("hold_end"),
                "realized_returns": realized,
                "risk_free_return": month_rf,
                "gross_return": gross_return,
                "net_return": net_return,
                "turnover": turnover,
                "equal_weight_gross_return": benchmark_gross_return,
                "equal_weight_net_return": benchmark_net_return,
                "equal_weight_turnover": benchmark_turnover,
                "active_return_vs_equal_weight": active_return,
                **replayed,
                "signal_metrics": {
                    "directional_hit_rate": monthly_hit_rate,
                    "spearman_rank_ic": rank_ic,
                    "pearson_ic": pearson_ic,
                    "forecast_mae_annualized": monthly_mae,
                },
            }
        )

    gross_summary = summarize(gross_returns, turnovers, risk_free)
    net_summary = summarize(net_returns, turnovers, risk_free)
    benchmark_gross_summary = summarize(
        benchmark_gross, benchmark_turnovers, risk_free
    )
    benchmark_net_summary = summarize(
        benchmark_net, benchmark_turnovers, risk_free
    )
    active = _active_metrics(net_returns, benchmark_net)
    cell_stock_rows = stock_rows
    direction_hits = [
        row["direction_hit"]
        for row in cell_stock_rows
        if row["direction_hit"] is not None
    ]
    summary_row = {
        "source_backtest": str(path),
        "portfolio_id": portfolio_id,
        "universe": ",".join(universe),
        "cell": cell,
        "strategy": strategy,
        "model_family": metadata["model_family"],
        "architecture": metadata["architecture"],
        "start": backtest.get("start"),
        "end": backtest.get("end"),
        "n_months": len(months),
        "q_multiplier": rule.q_multiplier,
        "confidence_mode": rule.confidence_mode,
        "fixed_confidence": (
            rule.fixed_confidence if rule.confidence_mode == "fixed" else None
        ),
        "view_mode": rule.view_mode,
        "eligibility_domain": rule.eligibility_domain,
        "raw_spread_floor": rule.raw_spread_floor,
        "scaled_spread_floor": rule.scaled_spread_floor,
        **{f"gross_{key}": value for key, value in gross_summary.items()},
        **{f"net_{key}": value for key, value in net_summary.items()},
        **{
            f"equal_weight_net_{key}": value
            for key, value in benchmark_net_summary.items()
        },
        "cumulative_return_gap_vs_equal_weight": (
            net_summary["cumulative_return"]
            - benchmark_net_summary["cumulative_return"]
        ),
        "annualized_return_gap_vs_equal_weight": (
            net_summary["annualized_return"]
            - benchmark_net_summary["annualized_return"]
        ),
        "sharpe_gap_vs_equal_weight": (
            net_summary["sharpe"] - benchmark_net_summary["sharpe"]
        ),
        **active,
        "mean_monthly_directional_hit_rate": _mean(
            row["directional_hit_rate"] for row in monthly_rows
        ),
        "overall_directional_hit_rate": (
            float(np.mean(direction_hits)) if direction_hits else None
        ),
        "mean_monthly_spearman_rank_ic": _mean(
            row["spearman_rank_ic"] for row in monthly_rows
        ),
        "mean_monthly_pearson_ic": _mean(
            row["pearson_ic"] for row in monthly_rows
        ),
        "mean_forecast_mae_annualized": _mean(
            row["forecast_mae_annualized"] for row in monthly_rows
        ),
        "mean_active_share": _mean(row["active_share"] for row in monthly_rows),
        "mean_pairwise_views": _mean(
            float(row["n_pairwise_views"]) for row in monthly_rows
        ),
        "mean_pairwise_alignment_rate": _mean(
            row["pairwise_alignment_rate"] for row in monthly_rows
        ),
        "mean_pairwise_model_alpha_spread": _mean(
            row["mean_pairwise_model_alpha_spread"] for row in monthly_rows
        ),
        "mean_pairwise_prior_spread": _mean(
            row["mean_pairwise_prior_spread"] for row in monthly_rows
        ),
        "optimizer_fallback_months": sum(
            bool(row["optimizer_used_fallback"]) for row in monthly_rows
        ),
        "mock_stock_views": sum(bool(row["used_mock"]) for row in stock_rows),
    }

    json_output = {
        "source_backtest": str(path),
        "portfolio_id": portfolio_id,
        "universe": universe,
        "cell": cell,
        "strategy": strategy,
        "model_family": metadata["model_family"],
        "architecture": metadata["architecture"],
        "start": backtest.get("start"),
        "end": backtest.get("end"),
        "replay_rule": {
            "model_calls": False,
            "q_multiplier": rule.q_multiplier,
            "confidence_mode": rule.confidence_mode,
            "fixed_confidence": (
                rule.fixed_confidence if rule.confidence_mode == "fixed" else None
            ),
            "view_mode": rule.view_mode,
            "eligibility_domain": rule.eligibility_domain,
            "raw_spread_floor": rule.raw_spread_floor,
            "scaled_spread_floor": rule.scaled_spread_floor,
            "pairwise_conversion": "adjacent_chain_n_minus_1",
            "description": (
                "Common fixed scaling applied to stored raw model views. "
                "Ranking eligibility and Black-Litterman Q semantics are explicit; "
                "saved point-in-time data and optimizer settings are preserved."
            ),
        },
        "per_month": replay_months,
        "performance": {
            strategy: {"gross": gross_summary, "net": net_summary},
            "equal_weight": {
                "gross": benchmark_gross_summary,
                "net": benchmark_net_summary,
            },
        },
        "active_performance": active,
        "signal_summary": {
            key: value
            for key, value in summary_row.items()
            if key
            in {
                "mean_monthly_directional_hit_rate",
                "overall_directional_hit_rate",
                "mean_monthly_spearman_rank_ic",
                "mean_monthly_pearson_ic",
                "mean_forecast_mae_annualized",
            }
        },
    }

    return ReplayResult(
        source_path=path,
        portfolio_id=portfolio_id,
        cell=cell,
        strategy=strategy,
        model_family=metadata["model_family"],
        architecture=metadata["architecture"],
        universe=universe,
        start=str(backtest.get("start") or ""),
        end=str(backtest.get("end") or ""),
        monthly_rows=monthly_rows,
        stock_rows=stock_rows,
        pairwise_rows=pairwise_rows,
        weight_rows=weight_rows,
        summary_row=summary_row,
        json_output=json_output,
    )


def _factorial_rows(summary_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], Dict[str, Mapping[str, Any]]] = {}
    for row in summary_rows:
        key = (str(row["portfolio_id"]), str(row["start"]), str(row["end"]))
        grouped.setdefault(key, {})[str(row["cell"])] = row

    metrics = [
        "net_cumulative_return",
        "net_annualized_return",
        "net_sharpe",
        "net_sortino",
        "net_max_drawdown",
        "net_calmar",
        "net_avg_turnover",
        "mean_monthly_spearman_rank_ic",
        "overall_directional_hit_rate",
    ]
    output: List[Dict[str, Any]] = []
    required = set(CELL_TO_STRATEGY)
    for (portfolio_id, start, end), cells in sorted(grouped.items()):
        if not required.issubset(cells):
            continue
        for metric in metrics:
            values = {cell: cells[cell].get(metric) for cell in required}
            if any(value is None for value in values.values()):
                continue
            values = {key: float(value) for key, value in values.items()}
            gap_single = values["llm_single"] - values["slm_single"]
            gap_multi = values["llm_multi"] - values["slm_multi"]
            output.append(
                {
                    "portfolio_id": portfolio_id,
                    "start": start,
                    "end": end,
                    "metric": metric,
                    **values,
                    "gap_llm_single_minus_slm_single": gap_single,
                    "gap_llm_multi_minus_slm_multi": gap_multi,
                    "gap_llm_single_minus_slm_multi": (
                        values["llm_single"] - values["slm_multi"]
                    ),
                    "slm_multi_minus_single_gain": (
                        values["slm_multi"] - values["slm_single"]
                    ),
                    "llm_multi_minus_single_gain": (
                        values["llm_multi"] - values["llm_single"]
                    ),
                    "interaction": (
                        values["slm_multi"]
                        - values["slm_single"]
                        - values["llm_multi"]
                        + values["llm_single"]
                    ),
                    "absolute_gap_closure": abs(gap_single) - abs(gap_multi),
                    "relative_gap_closure": (
                        (abs(gap_single) - abs(gap_multi)) / abs(gap_single)
                        if abs(gap_single) > 1e-12
                        else None
                    ),
                }
            )
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expand_patterns(patterns: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if not matches and Path(pattern).exists():
            matches = [pattern]
        paths.extend(Path(match) for match in matches)
    return sorted(set(paths))


def _latest_one_mode_backtests(paths: Sequence[Path]) -> List[Path]:
    selected: Dict[Tuple[str, str, str, str], Path] = {}
    for path in paths:
        try:
            backtest = json.loads(path.read_text())
            cell, _ = _resolve_cell(backtest)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        key = (
            str(backtest.get("portfolio_id") or path.stem),
            cell,
            str(backtest.get("start") or ""),
            str(backtest.get("end") or ""),
        )
        previous = selected.get(key)
        if previous is None or path.stat().st_mtime > previous.stat().st_mtime:
            selected[key] = path
    return sorted(selected.values())


def _print_summary(results: Sequence[ReplayResult], failures: Sequence[str], out_dir: Path) -> None:
    print(f"\nReplayed {len(results)} portfolio-cell backtests without model calls.")
    cells: Dict[str, int] = {}
    for result in results:
        cells[result.cell] = cells.get(result.cell, 0) + 1
    for cell in ("slm_single", "slm_multi", "llm_single", "llm_multi"):
        print(f"  {cell:12s}: {cells.get(cell, 0)}")
    if failures:
        print(f"\nSkipped/failed {len(failures)} files:")
        for failure in failures[:20]:
            print(f"  - {failure}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")
    print(f"\nOutputs: {out_dir}")
    print("  monthly_metrics.csv       one row per portfolio/configuration/month")
    print("  stock_month_views.csv     one row per stock prediction and realized outcome")
    print("  pairwise_views.csv        BL semantics, prior spread, alpha, and active-tilt alignment")
    print("  monthly_weights.csv       one row per portfolio/configuration/month/ticker")
    print("  portfolio_summary.csv     gross/net/risk-adjusted/signal metrics")
    print("  factorial_effects.csv     2x2 gaps once all four cells are available")
    print("  reconstructed/*.json      full replay audit trail")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay stored raw model views under one common fixed scaling rule"
    )
    parser.add_argument(
        "--backtests",
        nargs="+",
        default=["runs/backtest_*.json"],
        help="Backtest JSON paths or glob patterns",
    )
    parser.add_argument("--out-dir", default="runs/fixed_scaling_replay")
    parser.add_argument("--q-multiplier", type=float, default=0.25)
    parser.add_argument(
        "--confidence-mode",
        choices=["conservative", "raw", "fixed"],
        default="conservative",
        help="Common confidence treatment; conservative matches the existing no-file fallback",
    )
    parser.add_argument("--fixed-confidence", type=float, default=0.55)
    parser.add_argument(
        "--view-mode",
        choices=["absolute_return", "prior_relative_alpha"],
        default="absolute_return",
        help=(
            "Interpret the scaled pairwise spread as the complete BL Q "
            "(absolute_return) or as alpha relative to P*pi "
            "(prior_relative_alpha)."
        ),
    )
    parser.add_argument(
        "--eligibility-domain",
        choices=["raw", "scaled"],
        default="raw",
        help=(
            "Decide whether adjacent rankings survive using the saved raw model "
            "spread or the post-multiplier scaled spread. Raw is recommended."
        ),
    )
    parser.add_argument(
        "--raw-spread-floor",
        type=float,
        default=0.0,
        help="Minimum raw annualized stock-score spread; zero removes ties only.",
    )
    parser.add_argument(
        "--scaled-spread-floor",
        type=float,
        default=config.VIEW_SPREAD_FLOOR,
        help="Legacy post-scaling spread floor used only with --eligibility-domain scaled.",
    )
    parser.add_argument(
        "--all-matches",
        action="store_true",
        help="Do not deduplicate; otherwise select the latest file per portfolio and factorial cell",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first malformed/incomplete backtest instead of skipping it",
    )
    args = parser.parse_args()

    if not np.isfinite(args.q_multiplier) or args.q_multiplier < 0:
        raise SystemExit("--q-multiplier must be finite and non-negative")
    if not 0 <= args.fixed_confidence <= 1:
        raise SystemExit("--fixed-confidence must be between 0 and 1")
    if not np.isfinite(args.raw_spread_floor) or args.raw_spread_floor < 0:
        raise SystemExit("--raw-spread-floor must be finite and non-negative")
    if not np.isfinite(args.scaled_spread_floor) or args.scaled_spread_floor < 0:
        raise SystemExit("--scaled-spread-floor must be finite and non-negative")

    paths = _expand_patterns(args.backtests)
    if not paths:
        raise SystemExit("No backtest files matched")
    if not args.all_matches:
        paths = _latest_one_mode_backtests(paths)
    if not paths:
        raise SystemExit("No single-mode factorial backtests matched")

    rule = ReplayRule(
        q_multiplier=args.q_multiplier,
        confidence_mode=args.confidence_mode,
        fixed_confidence=args.fixed_confidence,
        view_mode=args.view_mode,
        eligibility_domain=args.eligibility_domain,
        raw_spread_floor=args.raw_spread_floor,
        scaled_spread_floor=args.scaled_spread_floor,
    )
    out_dir = Path(args.out_dir)
    reconstructed_dir = out_dir / "reconstructed"
    reconstructed_dir.mkdir(parents=True, exist_ok=True)

    results: List[ReplayResult] = []
    failures: List[str] = []
    for path in paths:
        try:
            result = replay_backtest(path, rule)
            results.append(result)
            destination = reconstructed_dir / f"{path.stem}_fixed_scaling.json"
            destination.write_text(json.dumps(result.json_output, indent=2, default=float))
            print(f"replayed {path.name} -> {destination.name}")
        except Exception as exc:  # batch utility should preserve successful outputs
            message = f"{path}: {type(exc).__name__}: {exc}"
            if args.fail_fast:
                raise
            failures.append(message)
            print(f"SKIP {message}")

    if not results:
        raise SystemExit("No backtests could be replayed")

    monthly_rows = [row for result in results for row in result.monthly_rows]
    stock_rows = [row for result in results for row in result.stock_rows]
    pairwise_rows = [row for result in results for row in result.pairwise_rows]
    weight_rows = [row for result in results for row in result.weight_rows]
    summary_rows = [result.summary_row for result in results]
    factorial_rows = _factorial_rows(summary_rows)

    _write_csv(out_dir / "monthly_metrics.csv", monthly_rows)
    _write_csv(out_dir / "stock_month_views.csv", stock_rows)
    _write_csv(out_dir / "pairwise_views.csv", pairwise_rows)
    _write_csv(out_dir / "monthly_weights.csv", weight_rows)
    _write_csv(out_dir / "portfolio_summary.csv", summary_rows)
    _write_csv(out_dir / "factorial_effects.csv", factorial_rows)
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "rule": {
                    "q_multiplier": rule.q_multiplier,
                    "confidence_mode": rule.confidence_mode,
                    "fixed_confidence": rule.fixed_confidence,
                    "view_mode": rule.view_mode,
                    "eligibility_domain": rule.eligibility_domain,
                    "raw_spread_floor": rule.raw_spread_floor,
                    "scaled_spread_floor": rule.scaled_spread_floor,
                    "model_calls": False,
                },
                "processed": [
                    {
                        "path": str(result.source_path),
                        "sha256": _sha256_file(result.source_path),
                    }
                    for result in results
                ],
                "replay_script_sha256": _sha256_file(Path(__file__)),
                "failures": failures,
                "counts": {
                    "backtests": len(results),
                    "monthly_rows": len(monthly_rows),
                    "stock_rows": len(stock_rows),
                    "pairwise_rows": len(pairwise_rows),
                    "weight_rows": len(weight_rows),
                    "factorial_rows": len(factorial_rows),
                },
            },
            indent=2,
        )
    )
    _print_summary(results, failures, out_dir)


if __name__ == "__main__":
    main()
