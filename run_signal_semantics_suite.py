#!/usr/bin/env python3
"""Run pre-declared replay specifications from saved backtests only.

This wrapper makes no model or data-provider calls. It executes the fixed-scaling
replay under three transparent signal-translation rules and writes compact
cross-specification summaries suitable for a robustness appendix.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


SPECIFICATIONS = {
    "legacy_absolute_conservative": [
        "--q-multiplier", "0.25",
        "--confidence-mode", "conservative",
        "--view-mode", "absolute_return",
        "--eligibility-domain", "scaled",
        "--scaled-spread-floor", "0.0025",
    ],
    "prior_alpha_fixed": [
        "--q-multiplier", "0.25",
        "--confidence-mode", "fixed",
        "--fixed-confidence", "0.55",
        "--view-mode", "prior_relative_alpha",
        "--eligibility-domain", "raw",
        "--raw-spread-floor", "0",
    ],
    "prior_alpha_conservative": [
        "--q-multiplier", "0.25",
        "--confidence-mode", "conservative",
        "--view-mode", "prior_relative_alpha",
        "--eligibility-domain", "raw",
        "--raw-spread-floor", "0",
    ],
}


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: Mapping[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fields.append(field)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(specification: str, rows: Sequence[Mapping[str, str]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("cell"))].append(row)

    output: List[Dict[str, Any]] = []
    for cell, cell_rows in sorted(grouped.items()):
        output.append(
            {
                "specification": specification,
                "cell": cell,
                "n_portfolios": len(cell_rows),
                "mean_net_cumulative_return": _mean(
                    _float(row, "net_cumulative_return") for row in cell_rows
                ),
                "mean_cumulative_gap_vs_equal_weight": _mean(
                    _float(row, "cumulative_return_gap_vs_equal_weight")
                    for row in cell_rows
                ),
                "mean_net_sharpe": _mean(
                    _float(row, "net_sharpe") for row in cell_rows
                ),
                "mean_active_share": _mean(
                    _float(row, "mean_active_share") for row in cell_rows
                ),
                "mean_pairwise_views": _mean(
                    _float(row, "mean_pairwise_views") for row in cell_rows
                ),
                "mean_pairwise_alignment_rate": _mean(
                    _float(row, "mean_pairwise_alignment_rate")
                    for row in cell_rows
                ),
            }
        )
    return output


def _matched(specification: str, rows: Sequence[Mapping[str, str]]) -> List[Dict[str, Any]]:
    by_portfolio: Dict[str, Dict[str, Mapping[str, str]]] = defaultdict(dict)
    for row in rows:
        by_portfolio[str(row.get("portfolio_id"))][str(row.get("cell"))] = row

    comparisons = [
        ("slm_multi_minus_llm_single", "slm_multi", "llm_single"),
        ("slm_single_minus_slm_multi", "slm_single", "slm_multi"),
        ("slm_single_minus_llm_single", "slm_single", "llm_single"),
    ]
    output: List[Dict[str, Any]] = []
    for name, left, right in comparisons:
        differences: List[float] = []
        for cells in by_portfolio.values():
            if left not in cells or right not in cells:
                continue
            left_value = _float(cells[left], "net_cumulative_return")
            right_value = _float(cells[right], "net_cumulative_return")
            if left_value is None or right_value is None:
                continue
            differences.append(left_value - right_value)
        output.append(
            {
                "specification": specification,
                "comparison": name,
                "n_matched_portfolios": len(differences),
                "mean_cumulative_return_difference": _mean(differences),
                "left_wins": sum(value > 0 for value in differences),
                "ties": sum(abs(value) <= 1e-12 for value in differences),
                "right_wins": sum(value < 0 for value in differences),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run three no-model-call signal-semantics replay specifications"
    )
    parser.add_argument(
        "--backtests",
        nargs="+",
        default=["runs/backtest_*.json"],
        help="Backtest paths or glob patterns passed to replay_fixed_scaling.py",
    )
    parser.add_argument(
        "--out-root",
        default="runs/signal_semantics_suite",
        help="Root directory for three replay outputs and comparison tables",
    )
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    root = Path(args.out_root)
    root.mkdir(parents=True, exist_ok=True)
    replay_script = Path(__file__).with_name("replay_fixed_scaling.py")
    aggregate_rows: List[Dict[str, Any]] = []
    matched_rows: List[Dict[str, Any]] = []
    executed: List[Dict[str, Any]] = []

    for name, specification_args in SPECIFICATIONS.items():
        out_dir = root / name
        command = [
            sys.executable,
            str(replay_script),
            "--backtests",
            *args.backtests,
            "--out-dir",
            str(out_dir),
            *specification_args,
        ]
        if args.fail_fast:
            command.append("--fail-fast")
        print("\nRUN", name)
        print(" ".join(command))
        subprocess.run(command, check=True)
        rows = _read_csv(out_dir / "portfolio_summary.csv")
        aggregate_rows.extend(_aggregate(name, rows))
        matched_rows.extend(_matched(name, rows))
        executed.append({"name": name, "arguments": specification_args})

    _write_csv(root / "specification_summary.csv", aggregate_rows)
    _write_csv(root / "matched_comparisons.csv", matched_rows)
    (root / "suite_manifest.json").write_text(
        json.dumps(
            {
                "model_calls": False,
                "backtests": args.backtests,
                "specifications": executed,
                "outputs": [
                    "specification_summary.csv",
                    "matched_comparisons.csv",
                ],
            },
            indent=2,
        )
    )
    print(f"\nWrote comparison tables to {root}")


if __name__ == "__main__":
    main()
