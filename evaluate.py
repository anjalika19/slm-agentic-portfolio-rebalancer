"""Debate evaluation harness (spec B): does the DEBATE add information?

Outcome Sharpe can't answer that -- a good Sharpe can come entirely from Phase-1
signals. So we score the *views* directly against the realized next-month return:

  B1  forward view accuracy, Phase-1 (pre) vs post-debate (post): hit-rate + Brier.
      If post <= pre, the debate is destroying information.
  B2  correction (wrong->right) vs subversion (right->wrong) across the debate, plus
      the sycophancy rate (unevidenced flips, from the anti-conformity guard).
  B3  martingale check: hit-rate per round. Flat => extra rounds add noise (cut them);
      rising with specialized info => that is the headline figure.

Everything is forward-looking: the ground truth is the realized next-month return,
which the agents never see (point-in-time wall).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from config import ANNUAL_RETURN_SCALE, NEUTRAL_BAND


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _dir_from_q(q: float, band: float = NEUTRAL_BAND) -> str:
    if abs(q) < band:
        return "neutral"
    return "long" if q > 0 else "short"


def _p_up(direction: str, confidence: float) -> float:
    """Predicted P(up next month) from a directional view + confidence."""
    c = max(0.0, min(1.0, confidence))
    if direction == "long":
        return 0.5 + 0.5 * c
    if direction == "short":
        return 0.5 - 0.5 * c
    return 0.5


def hit_and_brier(views: Dict[str, Dict], realized: Dict[str, float]) -> Dict:
    """views: {ticker: {direction, confidence}}. Score directional views with a
    non-zero realized return. Returns hit-rate (directional) + Brier (calibration)."""
    n = hits = 0
    brier = 0.0
    detail = {}
    for t, v in views.items():
        d = v.get("direction")
        r = realized.get(t)
        if d not in ("long", "short") or r is None or abs(r) < 1e-9:
            continue
        n += 1
        correct = (d == "long" and r > 0) or (d == "short" and r < 0)
        hits += int(correct)
        outcome = 1.0 if r > 0 else 0.0
        b = (_p_up(d, v.get("confidence", 0.5)) - outcome) ** 2
        brier += b
        detail[t] = {"direction": d, "realized": round(r, 4), "correct": correct}
    return {
        "n": n,
        "hits": hits,
        "hit_rate": round(hits / n, 4) if n else None,
        "brier": round(brier / n, 4) if n else None,
        "detail": detail,
    }


# --------------------------------------------------------------------------- #
# extract views from the event stream
# --------------------------------------------------------------------------- #
def phase1_views(events: List[dict]) -> Dict[str, Dict]:
    """Aggregated Phase-1 view per ticker (mean signal across the 4 specialists)."""
    sig: Dict[str, List[float]] = {}
    conf: Dict[str, List[float]] = {}
    for e in events:
        if e["type"] == "signal_emitted":
            sig.setdefault(e["ticker"], []).append(float(e["payload"].get("signal") or 0.0))
            conf.setdefault(e["ticker"], []).append(float(e["payload"].get("confidence") or 0.5))
    out = {}
    for t, s in sig.items():
        mean_sig = float(np.mean(s))
        out[t] = {"direction": _dir_from_q(mean_sig * ANNUAL_RETURN_SCALE),
                  "confidence": float(np.mean(conf[t]))}
    return out


def final_views(events: List[dict]) -> Dict[str, Dict]:
    out = {}
    for e in events:
        if e["type"] == "final_view" and e.get("ticker"):
            p = e["payload"]
            out[e["ticker"]] = {"direction": p.get("direction"),
                                "confidence": float(p.get("confidence") or 0.5)}
    return out


def per_round_views(events: List[dict]) -> Dict[int, Dict[str, Dict]]:
    """Per-round post-aggregation view per ticker (from the thin moderator's q)."""
    rounds: Dict[int, Dict[str, Dict]] = {}
    for e in events:
        if e["type"] == "moderator_synthesis":
            r = e.get("round")
            p = e["payload"]
            rounds.setdefault(r, {})[e["ticker"]] = {
                "direction": _dir_from_q(float(p.get("recommended_q") or 0.0)),
                "confidence": float(p.get("recommended_confidence") or 0.5),
            }
    return rounds


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def score_run(events: List[dict], realized: Dict[str, float]) -> Dict:
    pre = phase1_views(events)
    post = final_views(events)
    b1_pre = hit_and_brier(pre, realized)
    b1_post = hit_and_brier(post, realized)

    # B2: correction vs subversion at the ticker level (pre vs post direction)
    correction = subversion = unchanged = 0
    for t, pv in pre.items():
        po = post.get(t, {})
        if po.get("direction") == pv.get("direction"):
            unchanged += 1
            continue
        r = realized.get(t)
        if r is None or abs(r) < 1e-9:
            continue
        pre_correct = (pv["direction"] == "long" and r > 0) or (pv["direction"] == "short" and r < 0)
        post_correct = (po.get("direction") == "long" and r > 0) or (po.get("direction") == "short" and r < 0)
        if not pre_correct and post_correct:
            correction += 1
        elif pre_correct and not post_correct:
            subversion += 1

    # sycophancy rate: flagged unevidenced flips / total position changes
    flips = sum(1 for e in events if e["type"] == "rebuttal"
                and e["payload"].get("revised_direction") and e["payload"].get("agent_name"))
    sycophancy = sum(1 for e in events if e["type"] == "warning"
                     and e["payload"].get("kind") == "sycophancy")

    # B3: per-round hit-rate (martingale check)
    per_round = {r: hit_and_brier(v, realized) for r, v in sorted(per_round_views(events).items())}

    return {
        "B1": {"pre": b1_pre, "post": b1_post,
               "delta_hit_rate": _delta(b1_post["hit_rate"], b1_pre["hit_rate"]),
               "delta_brier": _delta(b1_post["brier"], b1_pre["brier"]),
               "debate_helped": _helped(b1_pre, b1_post)},
        "B2": {"correction": correction, "subversion": subversion, "unchanged": unchanged,
               "net": correction - subversion,
               "sycophancy_flagged": sycophancy, "position_changes": flips},
        "B3": {r: {"hit_rate": m["hit_rate"], "n": m["n"], "brier": m["brier"]}
               for r, m in per_round.items()},
    }


def _delta(a, b):
    if a is None or b is None:
        return None
    return round(a - b, 4)


def _helped(pre, post) -> bool:
    if pre["hit_rate"] is None or post["hit_rate"] is None:
        return False
    if post["hit_rate"] != pre["hit_rate"]:
        return post["hit_rate"] > pre["hit_rate"]
    # tie on hit-rate -> better-calibrated (lower Brier) wins
    return (post["brier"] or 1) < (pre["brier"] or 1)


def martingale_verdict(per_round: Dict[int, Dict]) -> str:
    """Flat across rounds => martingale (cut rounds). Rising => debate adds info."""
    rates = [m["hit_rate"] for m in per_round.values() if m.get("hit_rate") is not None]
    if len(rates) < 2:
        return "insufficient_rounds"
    if rates[-1] > rates[0] + 1e-9:
        return "rising (debate adds information)"
    if abs(rates[-1] - rates[0]) < 1e-9:
        return "flat (martingale -> consider cutting later rounds)"
    return "falling (later rounds destroy information)"


# --------------------------------------------------------------------------- #
# CLI: evaluate one rebalance date against its realized next-month return
# --------------------------------------------------------------------------- #
def evaluate_single(provider, portfolio_id, as_of, universe, hold_end, strict=False) -> Dict:
    from pipeline.runner import run_rebalance

    res = run_rebalance(provider, portfolio_id, as_of, universe, strict=strict, save=False)
    events = [e.to_dict() for e in res["emitter"].events]
    realized = {t: provider.forward_return(t, as_of, hold_end) for t in universe}
    return {"realized": realized, "score": score_run(events, realized)}


def main() -> None:
    import argparse
    import pandas as pd
    import config

    ap = argparse.ArgumentParser(description="Score whether the debate adds information (spec B1-B3).")
    ap.add_argument("--provider", default="fixture", choices=["fixture", "live"])
    ap.add_argument("--as-of", default=config.DEFAULT_AS_OF)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    if args.provider == "fixture":
        from providers.fixture import FixtureProvider

        provider = FixtureProvider()
    else:
        from providers.live import LiveProvider

        provider = LiveProvider()

    hold_end = (pd.Timestamp(args.as_of) + pd.offsets.MonthEnd(1)).strftime("%Y-%m-%d")
    out = evaluate_single(provider, "eval", args.as_of, config.UNIVERSE, hold_end,
                          strict=True if args.strict else config.STRICT_MODEL)
    s = out["score"]
    print(f"\n=== Debate evaluation @ {args.as_of} (realized over {args.as_of}->{hold_end}) ===")
    print(f"B1 forward view accuracy:")
    print(f"   Phase-1 (pre):  hit={s['B1']['pre']['hit_rate']}  brier={s['B1']['pre']['brier']}  (n={s['B1']['pre']['n']})")
    print(f"   post-debate:    hit={s['B1']['post']['hit_rate']}  brier={s['B1']['post']['brier']}  (n={s['B1']['post']['n']})")
    print(f"   delta hit={s['B1']['delta_hit_rate']}  delta brier={s['B1']['delta_brier']}  -> debate helped: {s['B1']['debate_helped']}")
    print(f"B2 correction={s['B2']['correction']}  subversion={s['B2']['subversion']}  net={s['B2']['net']}  "
          f"sycophancy_flagged={s['B2']['sycophancy_flagged']}/{s['B2']['position_changes']} changes")
    print(f"B3 per-round hit-rate: " + ", ".join(f"R{r}={m['hit_rate']}" for r, m in s["B3"].items()))
    print(f"   martingale verdict: {martingale_verdict(s['B3'])}")
    print("\n(single month -> noisy; run the backtest for aggregate B1-B3 across months.)")


if __name__ == "__main__":
    main()


def seed_variance_report(scored_runs: List[Dict]) -> Dict:
    """Summarize variance across repeated runs with different seeds.

    Caller passes a list of score_run(...) outputs or run summaries that contain
    B1/pre and B1/post. This keeps the function provider-agnostic and avoids
    leaking future returns into agents.
    """
    pre, post = [], []
    for s in scored_runs:
        b1 = s.get("B1", s.get("score", {}).get("B1", {}))
        if not b1:
            continue
        if b1.get("pre", {}).get("hit_rate") is not None:
            pre.append(float(b1["pre"]["hit_rate"]))
        if b1.get("post", {}).get("hit_rate") is not None:
            post.append(float(b1["post"]["hit_rate"]))

    def stat(xs):
        return {
            "n": len(xs),
            "mean": round(float(np.mean(xs)), 4) if xs else None,
            "variance": round(float(np.var(xs, ddof=1)), 6) if len(xs) > 1 else 0.0,
        }

    return {"phase1_hit_rate": stat(pre), "post_debate_hit_rate": stat(post)}
