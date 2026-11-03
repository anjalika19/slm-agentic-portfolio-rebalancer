"""CLI: run one fixture (or live) single-date debate end-to-end and print a summary.

  python run_single.py                      # fixture, mock-friendly demo
  STRICT_MODEL=1 python run_single.py       # require Gemma/Ollama
  python run_single.py --as-of 2024-06-30 --provider fixture
"""

from __future__ import annotations

import argparse
import json

import config
from pipeline.runner import run_rebalance


def _provider(name: str):
    if name == "fixture":
        from providers.fixture import FixtureProvider

        return FixtureProvider()
    if name == "live":
        from providers.live import LiveProvider

        return LiveProvider()
    raise SystemExit(f"unknown provider: {name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="fixture", choices=["fixture", "live"])
    ap.add_argument("--portfolio-id", default=config.DEFAULT_PORTFOLIO_ID)
    ap.add_argument("--as-of", default=config.DEFAULT_AS_OF)
    ap.add_argument("--strict", action="store_true", help="require real model (no mock fallback)")
    args = ap.parse_args()

    strict = True if args.strict else config.STRICT_MODEL

    def on_event(ev):
        tag = f"[{ev.seq:>3}] {ev.phase:9s} {ev.type:18s}"
        loc = " ".join(x for x in [ev.ticker or "", (f"r{ev.round}" if ev.round else ""), ev.agent or ""] if x)
        print(f"{tag} {loc}")

    res = run_rebalance(
        _provider(args.provider),
        args.portfolio_id,
        args.as_of,
        config.UNIVERSE,
        strict=strict,
        subscriber=on_event,
    )

    print("\n================ RESULT ================")
    print("rounds:", res["n_rounds"], "| consensus failures:", res["consensus_failures"])
    print("used_mock:", res["used_mock"])
    print("\nfinal weights vs equilibrium:")
    w, eq = res["weights"], res["equilibrium_weights"]
    for t in config.UNIVERSE:
        print(f"  {t:6s} weight={w.get(t,0):.4f}  eq={eq.get(t,0):.4f}  tilt={w.get(t,0)-eq.get(t,0):+.4f}")
    print("\noptimizer:", res["optimized"]["status"], "| vol=", res["optimized"]["portfolio_vol"],
          "| note:", res["optimized"]["note"])
    print("hallucination rates:", {a: res["validator"][a]["hallucination_rate"] for a in res["validator"]})
    print("\nrun log:", res.get("run_log_path"))


if __name__ == "__main__":
    main()
