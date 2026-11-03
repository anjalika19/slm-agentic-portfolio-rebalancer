"""Monthly rebalancing driver + backtest reporting (spec section 6.5).

For each month-end rebalance date, point-in-time:
  build input dataset -> run the full agent/moderator/debate/BL+MVO pipeline -> w*.
Hold w* until the next rebalance; the realized return is computed from FORWARD
prices that are never exposed to the agents (strict PiT wall). Roll forward.

We persist one run JSON per rebalance (via run_rebalance) plus a backtest summary
JSON with equity curves, weight evolution, turnover/consensus-failure strips and
performance metrics for the debate portfolio and the three baselines on the same
dates.
"""

from __future__ import annotations

import argparse
import json
import os
import uuid

from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

import baselines as bl
import config
import evaluate
from metrics import summarize
from providers.base import DataProvider

try:
    from pipeline.runner import run_rebalance
except Exception:  # pragma: no cover - Gemma/LangGraph deps are optional for Gemini-only C1
    run_rebalance = None

try:
    from agents.gemini import run_single_model_rebalance
except Exception:  # pragma: no cover - Gemini deps are optional until C1 is requested
    run_single_model_rebalance = None

def _safe_filename_part(s: str) -> str:
    """Keep filenames simple and filesystem-safe."""
    return "".join(
        c if c.isalnum() or c in "-_" else "-"
        for c in str(s)
    ).strip("-")


def _portfolio_letters(universe: List[str]) -> str:
    """Example: GOOGL, UNH, BAC, CAT, PEP, XOM -> GUBCPX."""
    letters = "".join(
        str(t).strip().upper()[0]
        for t in universe
        if str(t).strip()
    )
    return letters or "PORT"


def _backtest_filename(universe: List[str], start: str, end: str, run_mode: Optional[str] = None) -> str:
    letters = _portfolio_letters(universe)
    start_s = _safe_filename_part(start)
    end_s = _safe_filename_part(end)
    mode_s = _safe_filename_part(run_mode or "")
    short_id = uuid.uuid4().hex[:8]

    mode_part = f"_{mode_s}" if mode_s else ""
    return f"backtest_{letters}_{start_s}_{end_s}{mode_part}_{short_id}.json"



MODEL_CONFIGS = {
    "debate": {"architecture": "multi_agent", "model_family": "gemma"},
    "single_gemma": {"architecture": "single_agent", "model_family": "gemma"},
    "single_gemini": {"architecture": "single_agent", "model_family": "gemini"},
    "multi_gemini": {"architecture": "multi_agent", "model_family": "gemini"},
}

RUN_MODE_STRATEGIES = {
    "gemma_debate": ["debate"],
    "gemma_single": ["single_gemma"],
    "gemini_c1": ["single_gemini"],
    "gemini_debate": ["multi_gemini"],
    # Backward-compatible headline comparison used by earlier experiments.
    "compare": ["debate", "single_gemini"],
    # Complete 2x2 factorial design: model scale x agent architecture.
    "factorial": ["single_gemma", "debate", "single_gemini", "multi_gemini"],
}


def _normalize_run_mode(run_mode: Optional[str], run_gemini: Optional[bool]) -> str:
    """Choose which model path this backtest should execute.

    - gemma_debate: run only the multi-agent Gemma/SLM debate pipeline.
    - gemini_c1: run only the single-agent Gemini C1 path.
    - gemma_single: run the single-agent Gemma/SLM control.
    - gemini_debate: run the multi-agent Gemini/LLM architecture.
    - compare: run the legacy SLM-multi vs LLM-single comparison.
    - factorial: run all four matched configurations.

    ``run_gemini=True`` is kept as backward-compatible compare mode for old CLI/UI
    calls, but new callers should pass ``run_mode`` explicitly.
    """
    if run_mode is None:
        return "compare" if bool(run_gemini) else "gemma_debate"

    mode = str(run_mode).strip().lower()
    aliases = {
        "debate": "gemma_debate",
        "gemma": "gemma_debate",
        "gemma_multi": "gemma_debate",
        "multi_agent": "gemma_debate",
        "slm_multi": "gemma_debate",
        "slm_single": "gemma_single",
        "single_gemma": "gemma_single",
        "c1": "gemini_c1",
        "gemini": "gemini_c1",
        "single_gemini": "gemini_c1",
        "gemini_only": "gemini_c1",
        "llm_single": "gemini_c1",
        "gemini_multi": "gemini_debate",
        "llm_multi": "gemini_debate",
        "both": "compare",
        "all": "factorial",
        "all_four": "factorial",
        "2x2": "factorial",
    }
    mode = aliases.get(mode, mode)
    if mode not in RUN_MODE_STRATEGIES:
        expected = ", ".join(RUN_MODE_STRATEGIES)
        raise ValueError(f"unknown run_mode={run_mode!r}; expected one of: {expected}")
    return mode


def _mode_runs_debate(run_mode: str) -> bool:
    return any(
        MODEL_CONFIGS[s]["architecture"] == "multi_agent"
        for s in RUN_MODE_STRATEGIES[run_mode]
    )


def _mode_runs_gemini(run_mode: str) -> bool:
    return any(
        MODEL_CONFIGS[s]["model_family"] == "gemini"
        for s in RUN_MODE_STRATEGIES[run_mode]
    )


def _mode_model_strategies(run_mode: str) -> List[str]:
    return list(RUN_MODE_STRATEGIES[run_mode])

def month_end_dates(start: str, end: str) -> List[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end, freq="ME")]


def _aggregate_debate_eval(per_month: list, strategy: Optional[str] = None) -> Dict:
    """Aggregate B1-B3 across the backtest months (the headline debate metrics)."""
    pre_h, post_h, pre_b, post_b = [], [], [], []
    correction = subversion = sycophancy = changes = 0
    per_round_hits: Dict[int, list] = {}
    for m in per_month:
        if strategy is None:
            s = m.get("debate_eval") or {}
        else:
            s = (m.get("debate_eval_by_strategy") or {}).get(strategy, {})
        b1 = s.get("B1", {})
        if b1.get("pre", {}).get("hit_rate") is not None:
            pre_h.append(b1["pre"]["hit_rate"]); pre_b.append(b1["pre"]["brier"])
        if b1.get("post", {}).get("hit_rate") is not None:
            post_h.append(b1["post"]["hit_rate"]); post_b.append(b1["post"]["brier"])
        b2 = s.get("B2", {})
        correction += b2.get("correction", 0); subversion += b2.get("subversion", 0)
        sycophancy += b2.get("sycophancy_flagged", 0); changes += b2.get("position_changes", 0)
        for r, mr in (s.get("B3") or {}).items():
            if mr.get("hit_rate") is not None:
                per_round_hits.setdefault(int(r), []).append(mr["hit_rate"])

    def avg(x):
        return round(float(np.mean(x)), 4) if x else None

    per_round = {r: {"mean_hit_rate": avg(v), "n_months": len(v)}
                 for r, v in sorted(per_round_hits.items())}
    return {
        "B1": {"pre_hit_rate": avg(pre_h), "post_hit_rate": avg(post_h),
               "pre_brier": avg(pre_b), "post_brier": avg(post_b),
               "delta_hit_rate": (round(avg(post_h) - avg(pre_h), 4) if pre_h and post_h else None),
               "debate_helped": (avg(post_h) or 0) > (avg(pre_h) or 0)},
        "B2": {"correction": correction, "subversion": subversion, "net": correction - subversion,
               "sycophancy_flagged": sycophancy, "position_changes": changes},
        "B3_per_round": per_round,
        "B3_martingale": evaluate.martingale_verdict(
            {r: {"hit_rate": d["mean_hit_rate"]} for r, d in per_round.items()}),
    }


FACTORIAL_CELLS = {
    "slm_single": "single_gemma",
    "slm_multi": "debate",
    "llm_single": "single_gemini",
    "llm_multi": "multi_gemini",
}


def _factorial_effects(performance: Dict[str, Dict]) -> Dict[str, Dict]:
    """Report scale, architecture, and interaction effects for the 2x2 design.

    Effects are simple paired differences over identical rebalance dates. Positive
    ``gap_closure`` means the LLM--SLM gap is smaller under the multi-agent
    architecture. The output is omitted when any factorial cell is unavailable.
    """
    required = set(FACTORIAL_CELLS.values())
    if not required.issubset(performance):
        return {}

    metrics = (
        "cumulative_return",
        "annualized_return",
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
        "avg_turnover",
    )
    out: Dict[str, Dict] = {}
    for basis in ("gross", "net"):
        basis_out: Dict[str, Dict] = {}
        for metric in metrics:
            values = {
                cell: performance[strategy].get(basis, {}).get(metric)
                for cell, strategy in FACTORIAL_CELLS.items()
            }
            if any(value is None for value in values.values()):
                continue
            values = {key: float(value) for key, value in values.items()}
            gap_single = values["llm_single"] - values["slm_single"]
            gap_multi = values["llm_multi"] - values["slm_multi"]
            slm_architecture_gain = values["slm_multi"] - values["slm_single"]
            llm_architecture_gain = values["llm_multi"] - values["llm_single"]
            signed_gap_change = gap_single - gap_multi
            gap_closure = abs(gap_single) - abs(gap_multi)
            relative_gap_closure = (
                gap_closure / abs(gap_single)
                if abs(gap_single) > 1e-12
                else None
            )
            basis_out[metric] = {
                "cells": values,
                "scale_gap_single": round(gap_single, 8),
                "scale_gap_multi": round(gap_multi, 8),
                "slm_architecture_gain": round(slm_architecture_gain, 8),
                "llm_architecture_gain": round(llm_architecture_gain, 8),
                "interaction": round(slm_architecture_gain - llm_architecture_gain, 8),
                "signed_gap_change": round(signed_gap_change, 8),
                "gap_closure": round(gap_closure, 8),
                "relative_gap_closure": (
                    round(relative_gap_closure, 8)
                    if relative_gap_closure is not None
                    else None
                ),
            }
        out[basis] = basis_out
    return out


def _next_month_end(date: str) -> str:
    return (pd.Timestamp(date) + pd.offsets.MonthEnd(1)).strftime("%Y-%m-%d")


def _turnover(prev: Dict[str, float], cur: Dict[str, float], universe: List[str]) -> float:
    return float(sum(abs(cur.get(t, 0.0) - prev.get(t, 0.0)) for t in universe))


def _drift_weights(
    weights: Dict[str, float], realized_returns: Dict[str, float], universe: List[str]
) -> Dict[str, float]:
    """Move target weights through the holding period to the next pre-trade state."""
    grown = {
        t: float(weights.get(t, 0.0)) * (1.0 + float(realized_returns.get(t, 0.0)))
        for t in universe
    }
    total = float(sum(grown.values()))
    if not np.isfinite(total) or total <= 0:
        raise ValueError("cannot compute drifted weights from non-positive portfolio value")
    return {t: grown[t] / total for t in universe}


def _realized_return(
    provider: DataProvider, weights: Dict[str, float], start: str, end: str, universe: List[str]
) -> float:
    return float(sum(weights.get(t, 0.0) * provider.forward_return(t, start, end) for t in universe))


def run_backtest(
    provider: DataProvider,
    universe: List[str],
    start: str,
    end: str,
    *,
    portfolio_id: str = "backtest",
    strict: Optional[bool] = None,
    txn_cost_bps: float = None,
    progress: Optional[Callable[[str], None]] = None,
    subscriber: Optional[Callable] = None,
    run_gemini: Optional[bool] = None,
    run_mode: Optional[str] = None,
    embed_run_logs: bool = True,
) -> Dict:
    strict = config.STRICT_MODEL if strict is None else strict
    if run_gemini is None:
        run_gemini = config.RUN_SINGLE_GEMINI_C1
    run_mode = _normalize_run_mode(run_mode, run_gemini)
    run_gemini = _mode_runs_gemini(run_mode)
    run_debate = _mode_runs_debate(run_mode)
    model_strategies = _mode_model_strategies(run_mode)
    if run_debate and run_rebalance is None:
        raise RuntimeError("a multi-agent configuration was requested, but pipeline.runner could not be imported")
    if any(MODEL_CONFIGS[s]["architecture"] == "single_agent" for s in model_strategies) and run_single_model_rebalance is None:
        raise RuntimeError("a single-agent configuration was requested, but agents.gemini could not be imported")
    txn_cost_bps = config.TXN_COST_BPS if txn_cost_bps is None else txn_cost_bps
    cost_rate = txn_cost_bps / 10000.0

    dates = month_end_dates(start, end)
    if not dates:
        raise SystemExit(f"no month-end dates in [{start}, {end}]")

    strategies = list(model_strategies)
    strategies.extend(["equal_weight", "mvo_max_sharpe", "risk_parity"])
    gross_returns: Dict[str, List[float]] = {s: [] for s in strategies}
    net_returns: Dict[str, List[float]] = {s: [] for s in strategies}
    turnovers: Dict[str, List[float]] = {s: [] for s in strategies}
    # Pre-trade weights at each rebalance. After every holding period these are
    # drifted by realized asset returns; comparing target-to-target understates
    # turnover for constant-weight strategies such as equal weight.
    pre_trade_w: Dict[str, Dict[str, float]] = {s: {} for s in strategies}
    risk_free_returns: List[float] = []

    per_month = []
    run_paths = []
    embedded_run_logs = []

    def _run_entry(kind: str, date: str, res: Dict) -> Dict:
        path = res.get("run_log_path")
        entry = {
            "kind": kind,
            "as_of_date": date,
            "path": path,
            "name": os.path.basename(path) if path else None,
            "n_events": len(res.get("emitter").events) if res.get("emitter") is not None else None,
        }
        # Keep the full replayable event stream inside the backtest file. This makes
        # completed backtests self-contained even if standalone run JSONs are moved,
        # deleted, or were written on a different machine.
        if embed_run_logs and res.get("emitter") is not None:
            entry["log"] = res["emitter"].to_run_log()
        return entry

    for i, date in enumerate(dates):
        hold_end = dates[i + 1] if i + 1 < len(dates) else _next_month_end(date)
        if progress:
            progress(f"rebalance {i+1}/{len(dates)}: {date} -> {hold_end} · universe={','.join(universe)}")

        bundle = provider.get_input_bundle(portfolio_id, date, universe, config.DEFAULT_HORIZON)
        holding_period_rf = float(provider.risk_free_return(date, hold_end))
        risk_free_returns.append(holding_period_rf)
        weights = {s: {} for s in strategies}

        model_results: Dict[str, Dict] = {}
        for strategy in model_strategies:
            spec = MODEL_CONFIGS[strategy]
            architecture = spec["architecture"]
            family = spec["model_family"]
            if progress:
                progress(
                    f"{strategy} {i+1}/{len(dates)}: {date} · universe={','.join(universe)}"
                )

            run_pid = f"{portfolio_id}_{family}_{'multi' if architecture == 'multi_agent' else 'single'}"
            if architecture == "multi_agent":
                result = run_rebalance(
                    provider,
                    run_pid,
                    date,
                    universe,
                    strict=strict,
                    subscriber=subscriber,
                    bundle=bundle,
                    model_family=family,
                )
            else:
                result = run_single_model_rebalance(
                    provider,
                    run_pid,
                    date,
                    universe,
                    strict=strict,
                    subscriber=subscriber,
                    bundle=bundle,
                    model_family=family,
                )

            model_results[strategy] = result
            weights[strategy] = result["weights"]
            if result.get("run_log_path"):
                run_paths.append(result["run_log_path"])
            embedded_run_logs.append(_run_entry(strategy, date, result))

        # B1-B3 is defined only for multi-agent strategies. Single-agent controls
        # are evaluated by their stock views and realized portfolio performance.
        realized = {t: provider.forward_return(t, date, hold_end) for t in universe}
        debate_eval_by_strategy = {
            strategy: evaluate.score_run(
                [e.to_dict() for e in result["emitter"].events], realized
            )
            for strategy, result in model_results.items()
            if MODEL_CONFIGS[strategy]["architecture"] == "multi_agent"
        }
        primary_multi = (
            "debate" if "debate" in model_results else
            next(iter(debate_eval_by_strategy), None)
        )
        primary_result = model_results.get(primary_multi) if primary_multi else None
        legacy_gemini_result = model_results.get("single_gemini")

        bi = provider.baseline_inputs(universe, date)
        if bi is not None:
            mu, Sigma = bi
            base = bl.all_baselines(universe, np.asarray(mu), np.asarray(Sigma))
            weights["equal_weight"] = base["equal_weight"]
            weights["mvo_max_sharpe"] = base["mvo_max_sharpe"]
            weights["risk_parity"] = base["risk_parity"]
        else:
            weights["equal_weight"] = bl.equal_weight(universe)
            weights["mvo_max_sharpe"] = bl.equal_weight(universe)
            weights["risk_parity"] = bl.equal_weight(universe)
        

        month_row = {
            "as_of_date": date,
            "hold_end": hold_end,
            "weights": weights,
            "consensus_failures": primary_result["consensus_failures"] if primary_result else [],
            "consensus_failures_by_strategy": {
                s: r.get("consensus_failures", [])
                for s, r in model_results.items()
                if MODEL_CONFIGS[s]["architecture"] == "multi_agent"
            },
            "n_rounds": primary_result["n_rounds"] if primary_result else 0,
            "n_rounds_by_strategy": {
                s: r.get("n_rounds", 0)
                for s, r in model_results.items()
                if MODEL_CONFIGS[s]["architecture"] == "multi_agent"
            },
            "any_mock": any(primary_result["used_mock"].values()) if primary_result else None,
            "model_any_mock": {
                s: any(r.get("used_mock", {}).values())
                for s, r in model_results.items()
            },
            "gross_return": {},
            "net_return": {},
            "turnover": {},
            # Strictly ex-post. This is saved only after the holding period for
            # calibration/evaluation and is never exposed to agents.
            "realized_returns": {t: round(float(realized[t]), 8) for t in universe},
            "risk_free_return": round(holding_period_rf, 6),
            "debate_eval": debate_eval_by_strategy.get(primary_multi, {}) if primary_multi else {},
            "debate_eval_by_strategy": debate_eval_by_strategy,
            "run_logs": {
                s: {
                    "path": r.get("run_log_path"),
                    "name": os.path.basename(r.get("run_log_path", "")) if r.get("run_log_path") else None,
                }
                for s, r in model_results.items()
            },
            # Backward-compatible fields used by older UI builds.
            "run_log_path": primary_result.get("run_log_path") if primary_result else None,
            "run_log_name": os.path.basename(primary_result.get("run_log_path", "")) if primary_result and primary_result.get("run_log_path") else None,
            "gemini_run_log_path": legacy_gemini_result.get("run_log_path") if legacy_gemini_result else None,
            "gemini_run_log_name": os.path.basename(legacy_gemini_result.get("run_log_path", "")) if legacy_gemini_result and legacy_gemini_result.get("run_log_path") else None,
            "gemini_any_mock": any(legacy_gemini_result.get("used_mock", {}).values()) if legacy_gemini_result else None,
        }

        for s in strategies:
            gr = _realized_return(provider, weights[s], date, hold_end, universe)
            to = (
                _turnover(pre_trade_w[s], weights[s], universe)
                if pre_trade_w[s]
                else _turnover({}, weights[s], universe)
            )
            net = gr - cost_rate * to
            gross_returns[s].append(gr)
            net_returns[s].append(net)
            turnovers[s].append(to)
            pre_trade_w[s] = _drift_weights(weights[s], realized, universe)
            month_row["gross_return"][s] = round(gr, 6)
            month_row["net_return"][s] = round(net, 6)
            month_row["turnover"][s] = round(to, 6)

        per_month.append(month_row)

    performance = {
        s: {
            "gross": summarize(gross_returns[s], turnovers[s], risk_free_returns),
            "net": summarize(net_returns[s], turnovers[s], risk_free_returns),
        }
        for s in strategies
    }

    multi_strategies = [
        s for s in model_strategies
        if MODEL_CONFIGS[s]["architecture"] == "multi_agent"
    ]
    debate_evaluation_by_strategy = {
        s: _aggregate_debate_eval(per_month, strategy=s)
        for s in multi_strategies
    }
    primary_multi = "debate" if "debate" in multi_strategies else (multi_strategies[0] if multi_strategies else None)
    debate_evaluation = debate_evaluation_by_strategy.get(
        primary_multi, _aggregate_debate_eval(per_month)
    )
    out_name = _backtest_filename(universe, start, end, run_mode)
    summary = {
        "portfolio_id": portfolio_id,
        "universe": list(universe),
        "start": start,
        "end": end,
        "rebalance_dates": dates,
        "txn_cost_bps": txn_cost_bps,
        "risk_free_source": "provider FF5 RF when available; otherwise 0",
        "risk_free_returns": [round(value, 8) for value in risk_free_returns],
        "strict_model": strict,
        "provider": provider.name,
        "run_gemini": run_gemini,
        "run_debate": run_debate,
        "run_mode": run_mode,
        "model_strategies": model_strategies,
        "model_configs": {s: MODEL_CONFIGS[s] for s in model_strategies},
        "embedded_run_logs": embedded_run_logs,
        "config": config.summary(),
        "per_month": per_month,
        "performance": performance,
        "debate_evaluation": debate_evaluation,
        "debate_evaluation_by_strategy": debate_evaluation_by_strategy,
        "factorial_effects": _factorial_effects(performance),
        "run_log_paths": run_paths,
        "portfolio_letters": _portfolio_letters(universe),
        "backtest_file_id": out_name.replace(".json", ""),
    }

    os.makedirs(config.RUNS_DIR, exist_ok=True)
    out_path = os.path.join(config.RUNS_DIR, out_name)
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    summary["summary_path"] = out_path
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="fixture", choices=["fixture", "live"])
    ap.add_argument("--start", default=config.START_DATE)
    ap.add_argument("--end", default=config.END_DATE)
    ap.add_argument("--portfolio-id", default="backtest")
    ap.add_argument("--universe", default=None, help="comma-separated tickers; defaults to config.UNIVERSE")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--gemini", action="store_true", help="backward compatible: compare Gemma debate and single-model Gemini C1")
    ap.add_argument(
        "--mode",
        choices=list(RUN_MODE_STRATEGIES),
        default=None,
        help=(
            "configuration to run: Gemma/Gemini x single/multi; "
            "factorial runs all four on the same point-in-time bundle"
        ),
    )
    args = ap.parse_args()

    if args.provider == "fixture":
        from providers.fixture import FixtureProvider

        provider = FixtureProvider()
    else:
        from providers.live import LiveProvider

        provider = LiveProvider()

    universe = [x.strip().upper() for x in args.universe.split(",") if x.strip()] if args.universe else list(config.UNIVERSE)

    summary = run_backtest(
        provider, universe, args.start, args.end,
        portfolio_id=args.portfolio_id,
        strict=True if args.strict else config.STRICT_MODEL,
        progress=print,
        run_gemini=args.gemini or config.RUN_SINGLE_GEMINI_C1,
        run_mode=args.mode,
    )

    print("\n================ BACKTEST PERFORMANCE (net of costs) ================")
    perf = summary["performance"]
    hdr = f"{'strategy':16s} {'cumRet':>9s} {'annRet':>9s} {'Sharpe':>7s} {'Sortino':>8s} {'MaxDD':>8s} {'Calmar':>7s} {'Turn':>6s}"
    print(hdr)
    for s, p in perf.items():
        n = p["net"]
        print(f"{s:16s} {n['cumulative_return']:>9.3f} {n['annualized_return']:>9.3f} "
              f"{n['sharpe']:>7.2f} {n['sortino']:>8.2f} {n['max_drawdown']:>8.3f} "
              f"{n['calmar']:>7.2f} {n['avg_turnover']:>6.2f}")
    cf = sum(len(m["consensus_failures"]) for m in summary["per_month"])
    print(f"\nmonths: {len(summary['per_month'])} | total consensus failures: {cf}")

    de = summary["debate_evaluation"]
    print("\n================ DEBATE EVALUATION (does the debate add info?) ================")
    b1 = de["B1"]
    print(f"B1 forward view accuracy:  Phase-1 hit={b1['pre_hit_rate']} (brier {b1['pre_brier']})  "
          f"-> post-debate hit={b1['post_hit_rate']} (brier {b1['post_brier']})  "
          f"delta={b1['delta_hit_rate']}  helped={b1['debate_helped']}")
    b2 = de["B2"]
    print(f"B2 correction={b2['correction']} subversion={b2['subversion']} net={b2['net']}  "
          f"sycophancy_flagged={b2['sycophancy_flagged']}/{b2['position_changes']}")
    print("B3 per-round mean hit-rate: " +
          ", ".join(f"R{r}={d['mean_hit_rate']}" for r, d in de["B3_per_round"].items()))
    print(f"   martingale verdict: {de['B3_martingale']}")
    print("\nsummary:", summary["summary_path"])


if __name__ == "__main__":
    main()
