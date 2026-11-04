"""Learn q/confidence calibration constants from completed backtest files.

Example:
  python scripts/research/learn_calibration.py --backtests runs/backtest_live_*.json --out runs/view_calibration.json

Then run future backtests with:
  VIEW_CALIBRATION_PATH=runs/view_calibration.json agentic-bl-backtest --provider live --mode compare ...

The script uses only completed historical views and their realized next-month
returns. For publication-grade walk-forward tests, learn using only backtests or
months that ended before the rebalance date being tested.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional

from agentic_bl import calibration
from agentic_bl.calibration import ViewCalibrator, params_to_json


def _iter_logs(backtest: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for entry in backtest.get("embedded_run_logs", []) or []:
        log = entry.get("log") or {}
        if log.get("events"):
            yield log


def _month_by_date(backtest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {m.get("as_of_date"): m for m in backtest.get("per_month", []) if m.get("as_of_date")}


def _raw_view_samples_from_log(
    log: Dict[str, Any],
    realized_returns: Dict[str, float],
    kind: str,
) -> List[Dict[str, Any]]:
    """Extract stock-level raw views from saved logs.

    Multi-agent runs (Gemma or Gemini):
      - phase-1 views
      - accepted debate-round consensus views
      - end-only moderator views

    Single-agent runs (Gemma or Gemini):
      - stock-level views

    Pairwise final views are excluded because they repeat the same stock-level
    opinion across several pairs.
    """
    universe = (
        log.get("meta", {}).get("universe")
        or sorted(realized_returns)
    )

    events = log.get("events", []) or []
    samples: List[Dict[str, Any]] = []
    meta = log.get("meta", {}) or {}
    model_family = str(meta.get("model_family") or "").strip().lower()
    if model_family not in {"gemma", "gemini"}:
        model_family = "gemini" if "gemini" in str(kind) else "gemma"

    # Determine which Gemma consensus candidates were actually accepted.
    accepted_rounds = set()

    for event in events:
        if event.get("type") != "consensus_result":
            continue

        payload = event.get("payload") or {}

        if payload.get("resolved") is True:
            accepted_rounds.add(
                (
                    event.get("ticker"),
                    int(event.get("round") or 0),
                )
            )

    for event in events:
        event_type = event.get("type")
        ticker = event.get("ticker")
        payload = event.get("payload") or {}

        if not ticker or ticker not in realized_returns:
            continue

        family = None
        source = payload.get("source") or event_type
        raw_q = None
        raw_confidence = None

        # -------------------------------------------------------------
        # Single-model stock-level view
        # -------------------------------------------------------------
        if event_type == "single_model_emitted":
            if not str(kind).startswith("single_"):
                continue

            family = f"{model_family}_stock_signal"
            raw_q = payload.get("raw_q")
            raw_confidence = payload.get("raw_confidence")

            if raw_q is None:
                raw_q = payload.get("q")

            if raw_confidence is None:
                raw_confidence = payload.get("confidence")

        # -------------------------------------------------------------
        # Multi-agent phase-1 stock view
        # -------------------------------------------------------------
        elif (
            event_type == "baseline_view"
            and payload.get("source") == "agent_phase1_consensus"
        ):
            family = f"{model_family}_phase1_consensus"
            raw_q = payload.get("raw_q")
            raw_confidence = payload.get("raw_confidence")

            if raw_q is None:
                raw_q = payload.get("q")

            if raw_confidence is None:
                raw_confidence = payload.get("confidence")

        # -------------------------------------------------------------
        # Accepted multi-agent debate-round view
        # -------------------------------------------------------------
        elif event_type == "agent_consensus_candidate":
            round_number = int(event.get("round") or 0)

            # Do not calibrate rejected intermediate candidates.
            if (ticker, round_number) not in accepted_rounds:
                continue

            family = f"{model_family}_debate_r{round_number}"
            source = f"agent_consensus_r{round_number}"

            raw_q = payload.get("raw_recommended_q")
            raw_confidence = payload.get(
                "raw_recommended_confidence"
            )

            if raw_q is None:
                raw_q = payload.get("recommended_q")

            if raw_confidence is None:
                raw_confidence = payload.get(
                    "recommended_confidence"
                )

        # -------------------------------------------------------------
        # End-only moderator view
        # -------------------------------------------------------------
        elif event_type == "moderator_final_decision":
            family = f"{model_family}_moderator_end"
            source = "moderator_end_only_no_consensus"

            # The moderator schema stores its uncalibrated proposal in these
            # fields, while q/confidence are the calibrated values.
            raw_q = payload.get("recommended_q")
            raw_confidence = payload.get(
                "recommended_confidence"
            )

            if raw_q is None:
                raw_q = payload.get("raw_q")

            if raw_confidence is None:
                raw_confidence = payload.get(
                    "raw_confidence"
                )

            # Compatibility fallback for older logs.
            if raw_q is None:
                raw_q = payload.get("q")

            if raw_confidence is None:
                raw_confidence = payload.get("confidence")

        else:
            continue

        if raw_q is None or raw_confidence is None:
            continue

        others = [
            other
            for other in universe
            if other != ticker and other in realized_returns
        ]

        if not others:
            continue

        rest_return = sum(
            float(realized_returns[other])
            for other in others
        ) / len(others)

        realized_y = 12.0 * (
            float(realized_returns[ticker])
            - rest_return
        )

        samples.append(
            {
                "as_of_date": log.get("as_of_date"),
                "ticker": ticker,
                "family": family,
                "source": source,
                "raw_q": float(raw_q),
                "raw_confidence": float(raw_confidence),
                "realized_y": float(realized_y),
            }
        )

    return samples


def collect_samples(paths: List[str]) -> List[Dict[str, Any]]:
    samples = []
    for path in paths:
        with open(path, "r") as fh:
            bt = json.load(fh)
        months = _month_by_date(bt)
        for entry in bt.get("embedded_run_logs", []) or []:
            log = entry.get("log") or {}
            as_of = entry.get("as_of_date") or log.get("as_of_date")
            month = months.get(as_of) or {}
            realized = month.get("realized_returns") or {}
            if not realized:
                # Old backtest files did not save per-stock realized returns.
                # Re-run those backtests or patch them before learning calibration.
                continue
            samples.extend(_raw_view_samples_from_log(log, realized, entry.get("kind", "")))
    return samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backtests", nargs="+", required=True, help="Backtest JSON files or glob patterns")
    ap.add_argument("--out", default="runs/view_calibration.json")
    args = ap.parse_args()

    paths: List[str] = []
    for pat in args.backtests:
        paths.extend(glob.glob(pat))
    paths = sorted(set(paths))
    if not paths:
        raise SystemExit("No backtest files matched")

    samples = collect_samples(paths)
    by_family: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in samples:
        family = s["family"]
        if family != "default":
            by_family[family].append(s)
        # Every observation belongs to the shared/global pool exactly once.
        by_family["default"].append(s)

    # Learn one shared/global prior first, then shrink every family-specific
    # estimate toward it. This prevents a noisy family regression from receiving
    # a radically different q scale when its directional/ranking skill is weak.
    global_params = ViewCalibrator.learn(by_family.get("default", []), family="default")
    params = {"default": global_params}
    for family, rows in sorted(by_family.items()):
        if family == "default":
            continue
        params[family] = ViewCalibrator.learn(
            rows,
            family=family,
            prior=global_params,
        )

    out = params_to_json(params)
    out["calibration_method"] = "skill_gated_zero_centered_family_calibration_v3"
    out["n_samples_total"] = len(samples)
    out["source_backtests"] = paths

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)

    print(f"wrote {args.out}")
    print(f"samples: {len(samples)}")
    for family, p in sorted(params.items()):
        print(
            f"{family:24s} n={p.n_samples:4d} q_multiplier={p.q_multiplier:.3f} "
            f"conf_center={p.confidence_center:.3f} conf_slope={p.confidence_slope:.3f} "
            f"hit={p.hit_rate} rank_ic={p.rank_ic} reliability={p.reliability} "
            f"raw_slope={p.raw_q_slope} rmse={p.rmse}"
        )


if __name__ == "__main__":
    main()
