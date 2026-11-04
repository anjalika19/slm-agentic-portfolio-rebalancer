#!/usr/bin/env python3
"""Rebuild portfolio weights from saved raw model views without calling models.

This script is intentionally limited to the three post-model changes:

* load a new q/confidence calibration file;
* convert stock views to the independent adjacent-chain BL representation;
* apply the current common optimizer risk limits.

The agent/debate outputs themselves are read from the embedded run logs and are
not regenerated or altered.

Examples
--------
Rebuild one diagnostic test:

    python scripts/research/reconstruct_saved_backtests.py \
      --backtests runs/backtest_GNJLXC_2025-*.json \
      --calibration runs/view_calibration_2024_v2.json

Rebuild all 2024 training portfolios after relearning calibration:

    python scripts/research/reconstruct_saved_backtests.py \
      --backtests 'runs/backtest_*_2024-*_2024-12-31_*.json' \
      --calibration runs/view_calibration_2024_v2.json \
      --out-dir runs/reconstructed_v2
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from agentic_bl import bl_core
from agentic_bl import calibration
from agentic_bl import config
from agentic_bl.metrics import summarize


def _events(log: Dict[str, Any], event_type: str) -> List[Dict[str, Any]]:
    return [e for e in (log.get("events") or []) if e.get("type") == event_type]


def _last_event(log: Dict[str, Any], event_type: str) -> Optional[Dict[str, Any]]:
    rows = _events(log, event_type)
    return rows[-1] if rows else None


def _optimized_payload(log: Dict[str, Any]) -> Dict[str, Any]:
    event = _last_event(log, "optimized_portfolio")
    return (event or {}).get("payload") or {}


def _family_from_source(kind: str, source: Optional[str]) -> str:
    source = str(source or "")
    if kind == "single_gemini" or "single_gemini" in source:
        return "gemini_stock_signal"
    if source.startswith("agent_consensus_r"):
        return f"gemma_debate_r{source.replace('agent_consensus_r', '')}"
    if source == "agent_phase1_consensus":
        return "gemma_phase1_consensus"
    if source == "moderator_end_only_no_consensus":
        return "gemma_moderator_end"
    return "default"


def _gemini_raw_views(log: Dict[str, Any], universe: List[str]) -> Dict[str, Dict[str, Any]]:
    views: Dict[str, Dict[str, Any]] = {}
    for event in _events(log, "single_model_emitted"):
        ticker = event.get("ticker")
        payload = event.get("payload") or {}
        if ticker not in universe:
            continue
        raw_q = payload.get("raw_q")
        raw_confidence = payload.get("raw_confidence")
        if raw_q is None or raw_confidence is None:
            continue
        views[ticker] = {
            "raw_q": float(raw_q),
            "raw_confidence": float(raw_confidence),
            "family": "gemini_stock_signal",
            "source": payload.get("source") or "single_gemini_stock_signal_calibrated",
            "resolved": payload.get("resolved", True),
        }
    return views


def _gemma_pairwise_list(log: Dict[str, Any]) -> List[Dict[str, Any]]:
    pairwise = _optimized_payload(log).get("pairwise_views") or []
    if isinstance(pairwise, dict):
        return list(pairwise.values())
    return list(pairwise)


def _gemma_final_status(log: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    status = {}
    for event in _events(log, "final_view"):
        ticker = event.get("ticker")
        payload = event.get("payload") or {}
        if ticker and not payload.get("long_ticker") and not payload.get("short_ticker"):
            status[ticker] = payload
    return status


def _find_phase1(log: Dict[str, Any], ticker: str) -> Optional[Dict[str, Any]]:
    for event in _events(log, "baseline_view"):
        payload = event.get("payload") or {}
        if event.get("ticker") == ticker and payload.get("source") == "agent_phase1_consensus":
            return payload
    return None


def _find_consensus_round(log: Dict[str, Any], ticker: str, round_number: int) -> Optional[Dict[str, Any]]:
    matches = []
    for event in _events(log, "agent_consensus_candidate"):
        if event.get("ticker") == ticker and int(event.get("round") or 0) == round_number:
            matches.append(event.get("payload") or {})
    return matches[-1] if matches else None


def _find_moderator_final(log: Dict[str, Any], ticker: str) -> Optional[Dict[str, Any]]:
    matches = []
    for event in _events(log, "moderator_final_decision"):
        payload = event.get("payload") or {}
        if event.get("ticker") == ticker or payload.get("ticker") == ticker:
            matches.append(payload)
    return matches[-1] if matches else None


def _gemma_raw_views(log: Dict[str, Any], universe: List[str]) -> Dict[str, Dict[str, Any]]:
    # Old saved runs contain raw stock q/source information inside the pairwise
    # audit rows. Recover each stock's final raw opinion from those rows.
    raw_q: Dict[str, float] = {}
    source: Dict[str, str] = {}
    for pair in _gemma_pairwise_list(log):
        long_ticker = pair.get("long_ticker")
        short_ticker = pair.get("short_ticker")
        if long_ticker in universe and pair.get("long_raw_q") is not None:
            raw_q[long_ticker] = float(pair["long_raw_q"])
            source[long_ticker] = str(pair.get("long_source") or "default")
        if short_ticker in universe and pair.get("short_raw_q") is not None:
            raw_q[short_ticker] = float(pair["short_raw_q"])
            source[short_ticker] = str(pair.get("short_source") or "default")

    final_status = _gemma_final_status(log)
    views: Dict[str, Dict[str, Any]] = {}

    for ticker in universe:
        src = source.get(ticker, "default")
        q = raw_q.get(ticker)
        conf = None

        if src == "agent_phase1_consensus":
            payload = _find_phase1(log, ticker) or {}
            q = payload.get("raw_q", q)
            conf = payload.get("raw_confidence")
        elif src.startswith("agent_consensus_r"):
            try:
                round_number = int(src.replace("agent_consensus_r", ""))
            except ValueError:
                round_number = 0
            payload = _find_consensus_round(log, ticker, round_number) or {}
            q = payload.get("raw_recommended_q", q)
            conf = payload.get("raw_recommended_confidence")
        elif src == "moderator_end_only_no_consensus":
            payload = _find_moderator_final(log, ticker) or {}
            q = payload.get("recommended_q", q)
            conf = payload.get("recommended_confidence")

        # Fallbacks are audit-visible. They are only needed for an older run that
        # omitted a raw confidence field; q is never regenerated from model text.
        if q is None:
            phase1 = _find_phase1(log, ticker) or {}
            q = phase1.get("raw_q")
            if conf is None:
                conf = phase1.get("raw_confidence")
            src = phase1.get("source") or src

        if conf is None:
            # Confidence calibration was not the main changed component. Using the
            # stored final confidence is preferable to dropping an otherwise valid
            # raw q when an old event omitted raw_confidence.
            conf = (final_status.get(ticker) or {}).get("confidence", 0.5)

        if q is None:
            continue

        views[ticker] = {
            "raw_q": float(q),
            "raw_confidence": float(conf),
            "family": _family_from_source("debate", src),
            "source": src,
            "resolved": (final_status.get(ticker) or {}).get("resolved", True),
        }

    return views


def _raw_views(log: Dict[str, Any], kind: str, universe: List[str]) -> Dict[str, Dict[str, Any]]:
    if kind == "single_gemini":
        return _gemini_raw_views(log, universe)
    if kind == "debate":
        return _gemma_raw_views(log, universe)
    raise ValueError(f"unsupported embedded log kind: {kind}")


def _covariance(log: Dict[str, Any], universe: List[str]) -> np.ndarray:
    block = (
        log.get("meta", {})
        .get("agent_inputs", {})
        .get("quant", {})
        .get("covariance_matrix", {})
    )
    source_order = block.get("order") or universe
    matrix = np.asarray(block.get("matrix"), dtype=float)
    if matrix.shape != (len(source_order), len(source_order)):
        raise ValueError("saved log has no usable covariance matrix")
    idx = [source_order.index(ticker) for ticker in universe]
    matrix = matrix[np.ix_(idx, idx)]
    return 0.5 * (matrix + matrix.T)


def _sectors(log: Dict[str, Any], universe: List[str]) -> Dict[str, str]:
    mapping = (
        log.get("meta", {})
        .get("agent_inputs", {})
        .get("economist", {})
        .get("universe_sector_mapping", {})
    )

    # Older batch logs may have persisted every runtime ticker as "Unknown"
    # because config.py only mapped the then-current six-stock portfolio. Treat
    # that saved placeholder as missing and recover the sector from the complete
    # current research-universe map. This allows optimizer-only reconstruction
    # without calling either model again.
    out: Dict[str, str] = {}
    for ticker in universe:
        saved = mapping.get(ticker)
        if saved and str(saved).strip().lower() != "unknown":
            out[ticker] = str(saved)
        else:
            out[ticker] = config.SECTOR_BY_TICKER.get(ticker, "Unknown")
    return out


def _prior_weights(log: Dict[str, Any], universe: List[str]) -> np.ndarray:
    for event in _events(log, "baseline_view"):
        payload = event.get("payload") or {}
        if payload.get("type") == "bl_prior_weights":
            weights = payload.get("weights") or {}
            vector = np.array([float(weights.get(t, 0.0)) for t in universe], dtype=float)
            if vector.sum() > 0:
                return vector / vector.sum()
    return np.ones(len(universe), dtype=float) / len(universe)


def _reconstruct_month(
    log: Dict[str, Any],
    kind: str,
    universe: List[str],
    calibrator: calibration.ViewCalibrator,
) -> Dict[str, Any]:
    raw_views = _raw_views(log, kind, universe)
    missing = [ticker for ticker in universe if ticker not in raw_views]
    if missing:
        raise ValueError(f"saved {kind} log is missing raw stock views for: {missing}")

    calibrated_views = {}
    for ticker in universe:
        raw = raw_views[ticker]
        calibrated = calibrator.calibrate(
            raw["raw_q"],
            raw["raw_confidence"],
            family=raw["family"],
        )
        calibrated_views[ticker] = {
            "ticker": ticker,
            "q": calibrated.q,
            "confidence": calibrated.confidence,
            "raw_q": calibrated.raw_q,
            "raw_confidence": calibrated.raw_confidence,
            "family": raw["family"],
            "source": raw["source"],
            "resolved": raw.get("resolved", True),
            "calibration": calibrated.as_metadata(),
        }

    pairwise = calibration.ticker_views_to_pairwise_bl_views(
        calibrated_views,
        universe,
        max_pairs=config.MAX_PAIRWISE_VIEWS,
        spread_floor=config.VIEW_SPREAD_FLOOR,
        source=f"reconstructed_{kind}_adjacent_chain",
    )

    Sigma = _covariance(log, universe)
    benchmark = _prior_weights(log, universe)
    log_config = log.get("meta", {}).get("config", {})
    risk_aversion = float(log_config.get("RISK_AVERSION", config.RISK_AVERSION))
    tau = float(log_config.get("BL_TAU", config.BL_TAU))
    pi = bl_core.equilibrium_returns(Sigma, benchmark, risk_aversion)

    if pairwise:
        P = np.asarray([view["p_row"] for view in pairwise], dtype=float)
        Q = np.asarray([view["q"] for view in pairwise], dtype=float)
        confidences = np.asarray([view["confidence"] for view in pairwise], dtype=float)
        Omega = bl_core.idzorek_omega(P, Sigma, confidences, tau=tau)
        posterior = bl_core.bl_posterior(pi, Sigma, P, Q, Omega, tau=tau)
        omega_diag = np.diag(Omega)
    else:
        P = np.zeros((0, len(universe)), dtype=float)
        Q = np.zeros(0, dtype=float)
        omega_diag = np.zeros(0, dtype=float)
        posterior = bl_core.BLPosterior(mu_bl=pi.copy(), cov_bl=Sigma.copy(), pi=pi.copy())

    result = bl_core.optimize_long_only(
        posterior.mu_bl,
        Sigma,
        universe,
        _sectors(log, universe),
        risk_aversion=risk_aversion,
        benchmark_weights=benchmark,
    )

    return {
        "weights": {ticker: round(float(result.weights[i]), 8) for i, ticker in enumerate(universe)},
        "raw_stock_views": raw_views,
        "calibrated_stock_views": calibrated_views,
        "pairwise_views": pairwise,
        "bl_inputs": {
            "P": P.tolist(),
            "Q": Q.tolist(),
            "omega_diag": omega_diag.tolist(),
            "pi": pi.tolist(),
            "mu_bl": posterior.mu_bl.tolist(),
        },
        "optimizer": {
            "status": result.status,
            "used_fallback": result.used_fallback,
            "note": result.note,
            "portfolio_vol": result.portfolio_vol,
            "cap_adjusted": result.cap_adjusted,
        },
    }


def _turnover(previous: Dict[str, float], current: Dict[str, float], universe: List[str]) -> float:
    return float(sum(abs(current.get(t, 0.0) - previous.get(t, 0.0)) for t in universe))


def _drift_weights(
    weights: Dict[str, float], realized_returns: Dict[str, float], universe: List[str]
) -> Dict[str, float]:
    grown = {
        ticker: float(weights.get(ticker, 0.0)) * (1.0 + float(realized_returns[ticker]))
        for ticker in universe
    }
    total = float(sum(grown.values()))
    if not np.isfinite(total) or total <= 0:
        raise ValueError("cannot compute drifted weights from non-positive portfolio value")
    return {ticker: grown[ticker] / total for ticker in universe}


def reconstruct_backtest(path: Path, calibration_path: str, out_dir: Path) -> Path:
    backtest = json.loads(path.read_text())
    universe = list(backtest["universe"])
    calibrator = calibration.ViewCalibrator.from_json_file(calibration_path)

    log_by_date_kind: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for entry in backtest.get("embedded_run_logs", []) or []:
        log = entry.get("log") or {}
        date = entry.get("as_of_date") or log.get("as_of_date")
        kind = entry.get("kind")
        if date and kind and log:
            log_by_date_kind[(date, kind)] = log

    kinds = []
    if backtest.get("run_debate"):
        kinds.append("debate")
    if backtest.get("run_gemini"):
        kinds.append("single_gemini")
    if not kinds:
        raise ValueError("backtest has neither debate nor Gemini embedded logs")

    strategy_name = {"debate": "debate_reconstructed_v2", "single_gemini": "single_gemini_reconstructed_v2"}
    pre_trade = {kind: {} for kind in kinds}
    gross = {kind: [] for kind in kinds}
    net = {kind: [] for kind in kinds}
    turnover = {kind: [] for kind in kinds}
    rf = []
    per_month = []
    cost_rate = float(backtest.get("txn_cost_bps", config.TXN_COST_BPS)) / 10000.0

    for month_index, month in enumerate(backtest.get("per_month", []) or []):
        date = month["as_of_date"]
        realized = {ticker: float(value) for ticker, value in (month.get("realized_returns") or {}).items()}
        if any(ticker not in realized for ticker in universe):
            raise ValueError(f"{path.name} {date}: missing realized_returns")
        month_rf = float(month.get("risk_free_return", 0.0))
        rf.append(month_rf)

        row = {
            "as_of_date": date,
            "hold_end": month.get("hold_end"),
            "realized_returns": realized,
            "strategies": {},
        }

        for kind in kinds:
            log = log_by_date_kind.get((date, kind))
            if not log:
                raise ValueError(f"{path.name} {date}: missing embedded {kind} log")
            rebuilt = _reconstruct_month(log, kind, universe, calibrator)
            weights = rebuilt["weights"]
            gr = float(sum(weights[t] * realized[t] for t in universe))
            to = _turnover(pre_trade[kind], weights, universe)
            nr = gr - cost_rate * to
            pre_trade[kind] = _drift_weights(weights, realized, universe)
            gross[kind].append(gr)
            net[kind].append(nr)
            turnover[kind].append(to)
            rebuilt.update(
                {
                    "strategy": strategy_name[kind],
                    "gross_return": gr,
                    "net_return": nr,
                    "turnover": to,
                }
            )
            row["strategies"][strategy_name[kind]] = rebuilt

        per_month.append(row)

    performance = {}
    for kind in kinds:
        name = strategy_name[kind]
        performance[name] = {
            "gross": summarize(gross[kind], turnover[kind], rf),
            "net": summarize(net[kind], turnover[kind], rf),
        }

    output = {
        "source_backtest": str(path),
        "source_run_mode": backtest.get("run_mode"),
        "universe": universe,
        "start": backtest.get("start"),
        "end": backtest.get("end"),
        "calibration_path": calibration_path,
        "reconstruction_method": {
            "model_calls": False,
            "q_calibration": "skill_gated_zero_centered_family_calibration_v3",
            "pairwise_conversion": "adjacent_chain_n_minus_1",
            "risk_limits": {
                "min_weight": config.MIN_WEIGHT,
                "max_weight": config.MAX_WEIGHT,
                "max_active_deviation": config.MAX_ACTIVE_DEVIATION,
                "active_weight_budget": config.ACTIVE_WEIGHT_BUDGET,
            },
        },
        "per_month": per_month,
        "performance": performance,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{path.stem}_reconstructed_v2.json"
    output_path.write_text(json.dumps(output, indent=2))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backtests", nargs="+", required=True, help="JSON files or glob patterns")
    parser.add_argument("--calibration", required=True, help="new calibration JSON")
    parser.add_argument("--out-dir", default="runs/reconstructed_v2")
    args = parser.parse_args()

    paths: List[str] = []
    for pattern in args.backtests:
        paths.extend(glob.glob(pattern))
    paths = sorted(set(paths))
    if not paths:
        raise SystemExit("No backtest files matched")

    for raw_path in paths:
        output_path = reconstruct_backtest(Path(raw_path), args.calibration, Path(args.out_dir))
        print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
