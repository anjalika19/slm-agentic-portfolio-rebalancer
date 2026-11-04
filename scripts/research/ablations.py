"""Decomposition ablations (spec B4): isolate where the value comes from.

Each ablation is an alternative way of turning the SAME information into per-ticker
views; all are scored by the same forward-accuracy harness (evaluate.hit_and_brier)
against the realized next-month return.

  full_debate          -- the real pipeline (post-debate consensus).
  phase1_only          -- skip all debate, just aggregate the Phase-1 signals. If full
                          debate ~= this, the debate adds nothing.
  single_gemma_all_data-- one model, ALL FOUR data packages in one prompt (same
                          information, different organization). Isolates *specialization*
                          from *deliberation*. (Mock: a blended single-signal stand-in;
                          the real isolation needs Ollama.)
  sham_debate          -- negative control: scramble which agent says what. Real debate
                          must beat sham, or the deliberation is just noise.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from agentic_bl import config
from agentic_bl import evaluate
from agentic_bl.agents import mock
from agentic_bl.config import ANNUAL_RETURN_SCALE, NEUTRAL_BAND


def _dir(q: float) -> str:
    if abs(q) < NEUTRAL_BAND:
        return "neutral"
    return "long" if q > 0 else "short"


def _single_gemma_views(bundle, universe) -> Dict[str, Dict]:
    """Single 'all-data' agent stand-in: blend the four specialists' per-ticker signals
    (same information, one organization). With a real model this is one prompt holding
    all four packages; in mock we blend the deterministic specialist signals."""
    outs = {a: mock.mock_specialist_output(a, bundle.packages[a]) for a in config.SPECIALISTS}
    from agentic_bl.agents.normalize import normalize
    norm = {a: normalize(a, outs[a], universe, bundle.sectors) for a in config.SPECIALISTS}
    views = {}
    for t in universe:
        sig = float(np.mean([norm[a]["signals"].get(t, {}).get("signal", 0.0) for a in config.SPECIALISTS]))
        conf = float(np.mean([norm[a]["signals"].get(t, {}).get("confidence", 0.5) for a in config.SPECIALISTS]))
        views[t] = {"direction": _dir(sig * ANNUAL_RETURN_SCALE), "confidence": conf}
    return views


def run_ablations(provider, portfolio_id, as_of, universe, hold_end, strict=False) -> Dict:
    """Run every ablation for one date and score each against realized next-month return."""
    from agentic_bl.pipeline.runner import run_rebalance

    realized = {t: provider.forward_return(t, as_of, hold_end) for t in universe}
    bundle = provider.get_input_bundle(portfolio_id, as_of, universe)

    # full debate
    full = run_rebalance(provider, portfolio_id, as_of, universe, strict=strict, save=False, bundle=bundle)
    full_events = [e.to_dict() for e in full["emitter"].events]
    full_views = evaluate.final_views(full_events)
    phase1_views = evaluate.phase1_views(full_events)

    # sham debate (scrambled peer facts)
    sham = run_rebalance(provider, portfolio_id, as_of, universe, strict=strict, save=False,
                         bundle=bundle, sham=True)
    sham_views = evaluate.final_views([e.to_dict() for e in sham["emitter"].events])

    # single-Gemma all-data
    sg_views = _single_gemma_views(bundle, universe)

    configs = {
        "full_debate": full_views,
        "phase1_only": phase1_views,
        "single_gemma_all_data": sg_views,
        "sham_debate": sham_views,
    }
    scored = {name: evaluate.hit_and_brier(v, realized) for name, v in configs.items()}
    return {
        "as_of": as_of,
        "realized": realized,
        "scores": {name: {"hit_rate": m["hit_rate"], "brier": m["brier"], "n": m["n"]}
                   for name, m in scored.items()},
        "verdicts": _verdicts(scored),
    }


def _verdicts(scored: Dict) -> Dict:
    def hr(name):
        return scored[name]["hit_rate"]

    out = {}
    if hr("full_debate") is not None and hr("phase1_only") is not None:
        out["debate_beats_phase1"] = hr("full_debate") > hr("phase1_only")
    if hr("full_debate") is not None and hr("sham_debate") is not None:
        out["debate_beats_sham"] = hr("full_debate") > hr("sham_debate")
    if hr("full_debate") is not None and hr("single_gemma_all_data") is not None:
        out["specialization_beats_single_gemma"] = hr("full_debate") > hr("single_gemma_all_data")
    return out


def aggregate_ablations(provider, universe, start, end, strict=False) -> Dict:
    """Mean ablation scores across a backtest window (the meaningful, less-noisy view)."""
    import pandas as pd

    dates = [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end, freq="ME")]
    per_config: Dict[str, List[float]] = {}
    per_config_brier: Dict[str, List[float]] = {}
    rows = []
    for i, d in enumerate(dates):
        hold_end = (dates[i + 1] if i + 1 < len(dates)
                    else (pd.Timestamp(d) + pd.offsets.MonthEnd(1)).strftime("%Y-%m-%d"))
        res = run_ablations(provider, "ablate", d, universe, hold_end, strict=strict)
        rows.append(res)
        for name, m in res["scores"].items():
            if m["hit_rate"] is not None:
                per_config.setdefault(name, []).append(m["hit_rate"])
                per_config_brier.setdefault(name, []).append(m["brier"])

    def avg(x):
        return round(float(np.mean(x)), 4) if x else None

    mean_hit = {k: avg(v) for k, v in per_config.items()}
    mean_brier = {k: avg(v) for k, v in per_config_brier.items()}
    return {
        "start": start, "end": end, "n_months": len(dates),
        "mean_hit_rate": mean_hit, "mean_brier": mean_brier,
        "verdicts": {
            "debate_beats_phase1": (mean_hit.get("full_debate") or 0) >= (mean_hit.get("phase1_only") or 0),
            "debate_beats_sham": (mean_hit.get("full_debate") or 0) >= (mean_hit.get("sham_debate") or 0),
            "specialization_beats_single_gemma":
                (mean_hit.get("full_debate") or 0) >= (mean_hit.get("single_gemma_all_data") or 0),
        },
        "per_month": rows,
    }


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Run the B4 decomposition ablations.")
    ap.add_argument("--provider", default="fixture", choices=["fixture", "live"])
    ap.add_argument("--start", default=config.START_DATE)
    ap.add_argument("--end", default=config.END_DATE)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    if args.provider == "fixture":
        from agentic_bl.providers.fixture import FixtureProvider

        provider = FixtureProvider()
    else:
        from agentic_bl.providers.live import LiveProvider

        provider = LiveProvider()

    out = aggregate_ablations(provider, config.UNIVERSE, args.start, args.end,
                              strict=True if args.strict else config.STRICT_MODEL)
    print(f"\n=== B4 ablations, {args.start}..{args.end} ({out['n_months']} months) ===")
    print(f"{'config':24s} {'mean hit':>9s} {'mean brier':>11s}")
    for name in ("full_debate", "phase1_only", "single_gemma_all_data", "sham_debate"):
        print(f"{name:24s} {str(out['mean_hit_rate'].get(name)):>9s} {str(out['mean_brier'].get(name)):>11s}")
    print("\nverdicts:", out["verdicts"])
    print("(mock provider -> single_gemma ~= phase1; the real isolation needs Ollama.)")


if __name__ == "__main__":
    main()
