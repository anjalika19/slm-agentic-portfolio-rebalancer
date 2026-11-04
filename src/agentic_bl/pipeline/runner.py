"""Single-date rebalance driver: build bundle -> run graph -> persist run JSON.

This is the one function the server, backtest and CLI all call. It owns the
EventEmitter (so every run is recorded to ``runs/<id>_<date>_<ts>.json`` and is
replayable) and returns a compact result with the final weights + metrics.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import random
import numpy as np

from agentic_bl import config
from agentic_bl import calibration
from agentic_bl.events import EventEmitter
from agentic_bl.pipeline.graph import build_graph
from agentic_bl.pipeline.state import RunContext, RunState
from agentic_bl.providers.base import DataProvider, InputBundle


def run_rebalance(
    provider: DataProvider,
    portfolio_id: str,
    as_of_date: str,
    universe: List[str],
    *,
    horizon: str = "1M",
    strict: Optional[bool] = None,
    emitter: Optional[EventEmitter] = None,
    subscriber: Optional[Callable] = None,
    save: bool = True,
    clock: Optional[Callable[[], datetime]] = None,
    bundle: Optional[InputBundle] = None,
    sham: bool = False,
    model_family: str = "gemma",
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the full flow for one date and return a result dict."""
    strict = config.STRICT_MODEL if strict is None else strict
    model_family = config.normalize_model_family(model_family)
    model_name = model_name or config.model_for_family(model_family)
    random.seed(config.PYTHON_RANDOM_SEED)
    np.random.seed(config.NUMPY_RANDOM_SEED)

    if bundle is None:
        bundle = provider.get_input_bundle(portfolio_id, as_of_date, universe, horizon)

    if emitter is None:
        emitter = EventEmitter(
            portfolio_id=portfolio_id,
            as_of_date=as_of_date,
            runs_dir=config.RUNS_DIR,
            meta={
                "provider": provider.name,
                "config": config.summary(),
                "strict_model": strict,
                "universe": list(universe),
                "horizon": horizon,
                "architecture": "multi_agent",
                "model_family": model_family,
                "model_name": model_name,
                "seeds": {
                    "python_random_seed": config.PYTHON_RANDOM_SEED,
                    "numpy_seed": config.NUMPY_RANDOM_SEED,
                    "model_seed": config.MODEL_SEED,
                    "ollama_seed": config.MODEL_SEED,  # legacy run-log field
                },
            },
            clock=clock,
        )
    # Embed the per-agent input packages + covariance in the run log so the UI can
    # show "what each agent saw" alongside its signal (replay-only; not streamed).
    emitter.meta.setdefault("agent_inputs", bundle.packages)
    emitter.meta.setdefault("covariance", bundle.covariance)
    emitter.meta.setdefault("sectors", bundle.sectors)

    if subscriber is not None:
        emitter.subscribe(subscriber)

    emitter.emit(
        "run_started",
        phase="system",
        payload={
            "portfolio_id": portfolio_id,
            "as_of_date": as_of_date,
            "universe": list(universe),
            "provider": provider.name,
            "strict_model": strict,
            "architecture": "multi_agent",
            "model_family": model_family,
            "model_name": model_name,
            "provider_warnings": bundle.warnings,
        },
    )
    for w in bundle.warnings:
        emitter.warning(w, phase="provider")

    rs = RunState(bundle=bundle, universe_order=list(universe))
    ctx = RunContext(
        emitter=emitter,
        strict=strict,
        scale=config.ANNUAL_RETURN_SCALE,
        max_rounds=config.MAX_DEBATE_ROUNDS,
        accept_threshold=config.CONSENSUS_ACCEPT_THRESHOLD,
        model_family=model_family,
        model_name=model_name,
        sham=sham,
        calibrator=calibration.load_view_calibrator(),
    )

    graph = build_graph()
    graph.invoke({"rs": rs, "ctx": ctx}, config={"recursion_limit": 50})

    result = {
        "portfolio_id": portfolio_id,
        "as_of_date": as_of_date,
        "architecture": "multi_agent",
        "model_family": model_family,
        "model_name": model_name,
        "weights": rs.optimized.get("weights", {}),
        "equilibrium_weights": rs.optimized.get("equilibrium_weights", {}),
        "optimized": rs.optimized,
        "final_views": rs.final_views,
        "consensus": rs.consensus,
        "validator": rs.validator,
        "used_mock": rs.used_mock,
        "seeds": {
            "python_random_seed": config.PYTHON_RANDOM_SEED,
            "numpy_seed": config.NUMPY_RANDOM_SEED,
            "model_seed": config.MODEL_SEED,
            "ollama_seed": config.MODEL_SEED,  # legacy result field
        },
        "n_rounds": rs.debate_round,
        "consensus_failures": [t for t, c in rs.consensus.items() if not c.get("resolved", True)],
    }

    emitter.emit(
        "run_finished",
        phase="system",
        payload={
            "weights": result["weights"],
            "n_rounds": rs.debate_round,
            "consensus_failures": result["consensus_failures"],
            "any_mock": any(rs.used_mock.values()),
        },
    )

    if save:
        path = emitter.save()
        result["run_log_path"] = path
    result["emitter"] = emitter
    return result
