"""FastAPI server: replay saved runs + stream a live debate over WebSocket (spec 5).

Endpoints
  GET  /                      -> the UI (static index.html)
  GET  /api/runs             -> list saved single-date run logs
  GET  /api/runs/{name}      -> full event log for replay
  GET  /api/backtests        -> list saved backtest summaries
  GET  /api/backtests/{name} -> a backtest summary (cross-month view)
  WS   /ws/run               -> start a live run; stream events as they emit

The UI is built against the replay endpoints first (record-then-replay), so it works
with zero models running; the WebSocket adds live streaming when Ollama is up (or in
mock mode).
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import threading
from typing import Any, List, Dict
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from agentic_bl import config
from agentic_bl.agents.gemini import run_single_model_rebalance
from agentic_bl.backtest import MODEL_CONFIGS, RUN_MODE_STRATEGIES, run_backtest
from agentic_bl.events import load_run_log
from agentic_bl.pipeline.runner import run_rebalance

UI_DIR = os.path.join(config.PROJECT_ROOT, "ui")

app = FastAPI(title="Agentic Black-Litterman Debate")

# In-memory live run state for the polling UI.
# Start a run in a worker thread, append emitted events, and let the browser poll.
ACTIVE_RUNS: Dict[str, dict] = {}
ACTIVE_RUNS_LOCK = threading.RLock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_provider(provider_name: str):
    if provider_name == "live":
        from agentic_bl.providers.live import LiveProvider

        return LiveProvider()

    from agentic_bl.providers.fixture import FixtureProvider

    return FixtureProvider()


def _public_run_state(run_id: str, state: dict, after: int = 0) -> dict:
    events = [e for e in state["events"] if int(e.get("seq", 0)) > after]
    last_seq = state["events"][-1]["seq"] if state["events"] else 0

    return {
        "run_id": run_id,
        "status": state["status"],
        "started_at": state["started_at"],
        "finished_at": state.get("finished_at"),
        "event_count": len(state["events"]),
        "next_seq": last_seq,
        "events": events,
        "last_event": state["events"][-1] if state["events"] else None,
        "run_log_path": state.get("run_log_path"),
        "error": state.get("error"),
    }


def _search_dirs() -> List[str]:
    """Saved runs first, then the read-only examples bundled with the repo."""
    return [config.RUNS_DIR, config.EXAMPLES_DIR]


def _list_json():
    os.makedirs(config.RUNS_DIR, exist_ok=True)
    out, seen = [], set()
    for d in _search_dirs():
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d), reverse=True):
            if not f.endswith(".json") or f in seen:
                continue
            seen.add(f)
            out.append({"name": f, "is_backtest": f.startswith("backtest_")})
    return out


def _load_backtest(name: str) -> dict | None:
    path = _safe_path(name)
    if path is None or not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def _strip_embedded_logs(bt: dict) -> dict:
    """Return a UI-friendly backtest summary without multi-MB embedded event logs."""
    out = dict(bt)
    entries = []
    for i, entry in enumerate(bt.get("embedded_run_logs", []) or []):
        shallow = {k: v for k, v in entry.items() if k != "log"}
        shallow["id"] = str(i)
        shallow["available"] = bool(entry.get("log")) or _run_path_exists(entry.get("path") or entry.get("name"))
        entries.append(shallow)
    out["embedded_run_logs"] = entries
    return out


def _run_path_candidates(pathish: str | None) -> List[str]:
    if not pathish:
        return []
    name = os.path.basename(pathish)
    candidates = []
    if os.path.isabs(pathish):
        candidates.append(pathish)
    for d in _search_dirs():
        candidates.append(os.path.join(d, name))
        candidates.append(os.path.join(d, pathish))
    # De-duplicate while preserving order.
    out = []
    for c in candidates:
        if c not in out:
            out.append(c)
    return out


def _run_path_exists(pathish: str | None) -> bool:
    return any(os.path.exists(c) for c in _run_path_candidates(pathish))


def _load_run_reference(pathish: str | None) -> dict | None:
    for c in _run_path_candidates(pathish):
        if os.path.exists(c):
            return load_run_log(c)
    return None


def _backtest_replay_entries(bt: dict) -> List[dict]:
    """List replayable debate/Gemini logs represented by a backtest file.

    New backtests are self-contained via ``embedded_run_logs``. Older summaries may
    only contain ``run_log_paths``; those are marked available only if the referenced
    standalone JSON still exists locally.
    """
    entries: List[dict] = []

    embedded = bt.get("embedded_run_logs", []) or []
    for i, entry in enumerate(embedded):
        log = entry.get("log") or {}
        pathish = entry.get("path") or entry.get("name")
        kind = entry.get("kind", "debate")
        as_of = entry.get("as_of_date") or log.get("as_of_date") or ""
        entries.append({
            "id": str(i),
            "kind": kind,
            "as_of_date": as_of,
            "label": f"{as_of} · {kind}",
            "name": entry.get("name") or os.path.basename(pathish or ""),
            "path": pathish,
            "n_events": entry.get("n_events") or log.get("n_events") or len(log.get("events", []) or []),
            "available": bool(entry.get("log")) or _run_path_exists(pathish),
            "source": "embedded" if entry.get("log") else "standalone_file",
        })

    if entries:
        return entries

    # Backward compatibility: old summaries saved only run_log_paths.
    paths = bt.get("run_log_paths", []) or []
    months = bt.get("per_month", []) or []
    for i, pathish in enumerate(paths):
        as_of = (months[i] or {}).get("as_of_date") if i < len(months) else ""
        entries.append({
            "id": str(i),
            "kind": "debate",
            "as_of_date": as_of or "",
            "label": f"{as_of or 'run '+str(i+1)} · debate",
            "name": os.path.basename(pathish or ""),
            "path": pathish,
            "n_events": None,
            "available": _run_path_exists(pathish),
            "source": "standalone_file",
        })
    return entries


def _load_backtest_replay(bt: dict, run_id: str) -> dict | None:
    entries = _backtest_replay_entries(bt)
    try:
        idx = int(run_id)
    except ValueError:
        idx = -1
    if idx < 0 or idx >= len(entries):
        return None

    embedded = bt.get("embedded_run_logs", []) or []
    if idx < len(embedded) and embedded[idx].get("log"):
        return embedded[idx]["log"]

    return _load_run_reference(entries[idx].get("path") or entries[idx].get("name"))

def _parse_universe_from_params(params: dict) -> List[str]:
    """Read universe from request params.

    Accepts either:
      - ["AAPL", "MSFT", "NVDA"]
      - "AAPL,MSFT,NVDA"

    Falls back to config.UNIVERSE if omitted.
    """
    raw = params.get("universe")

    if raw is None or raw == "":
        return list(config.UNIVERSE)

    if isinstance(raw, str):
        universe = [x.strip().upper() for x in raw.split(",") if x.strip()]
    elif isinstance(raw, list):
        universe = [str(x).strip().upper() for x in raw if str(x).strip()]
    else:
        raise ValueError("universe must be a comma-separated string or list of tickers")

    if not universe:
        raise ValueError("universe cannot be empty")

    return universe


def _run_mode_from_params(params: dict, *, default_compare_for_legacy_gemini: bool = False) -> str:
    """Resolve UI/API run mode.

    New UI sends run_mode explicitly. Legacy callers that only send run_gemini=True
    can still get the old compare behavior when requested by the caller.
    """
    raw = params.get("run_mode")
    if raw:
        mode = str(raw).strip().lower()
    elif bool(params.get("run_gemini", False)):
        mode = "compare" if default_compare_for_legacy_gemini else "gemini_c1"
    else:
        mode = "gemma_debate"

    aliases = {
        "debate": "gemma_debate",
        "gemma": "gemma_debate",
        "multi_agent": "gemma_debate",
        "gemma_multi": "gemma_debate",
        "slm_multi": "gemma_debate",
        "gemma_single": "gemma_single",
        "single_gemma": "gemma_single",
        "slm_single": "gemma_single",
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
        raise ValueError(f"run_mode must be one of: {', '.join(RUN_MODE_STRATEGIES)}")
    return mode

@app.get("/api/runs")
def list_runs():
    return [r for r in _list_json() if not r["is_backtest"]]


@app.get("/api/backtests")
def list_backtests():
    return [r for r in _list_json() if r["is_backtest"]]


@app.get("/api/runs/{name}")
def get_run(name: str):
    path = _safe_path(name)
    if path is None or not os.path.exists(path):
        return JSONResponse({"error": "not found"}, status_code=404)
    return load_run_log(path)


@app.get("/api/backtests/{name}/runs")
def list_backtest_runs(name: str):
    bt = _load_backtest(name)
    if bt is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return {"backtest": name, "runs": _backtest_replay_entries(bt)}


@app.get("/api/backtests/{name}/runs/{run_id}")
def get_backtest_run(name: str, run_id: str):
    bt = _load_backtest(name)
    if bt is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    log = _load_backtest_replay(bt, run_id)
    if log is None:
        return JSONResponse({"error": "run log not found or not embedded"}, status_code=404)
    return log


@app.get("/api/backtests/{name}")
def get_backtest(name: str, include_logs: bool = False):
    bt = _load_backtest(name)
    if bt is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return bt if include_logs else _strip_embedded_logs(bt)


@app.get("/api/config")
def get_config():
    return {
        "config": config.summary(),
        "default_as_of": config.DEFAULT_AS_OF,
        "default_start": config.START_DATE,
        "default_end": config.END_DATE,
        "universe": config.UNIVERSE,
        "run_single_gemini_c1": config.RUN_SINGLE_GEMINI_C1,
        "gemma_model": config.GEMMA_MODEL,
        "gemini_model": config.GEMINI_MODEL,
        "run_modes": list(RUN_MODE_STRATEGIES),
        "model_configs": MODEL_CONFIGS,
    }

@app.post("/api/live-runs")
def start_live_run(params: dict):
    """Start a rebalance in a background thread and return a pollable run id."""
    run_id = uuid.uuid4().hex[:10]

    portfolio_id = params.get("portfolio_id", config.DEFAULT_PORTFOLIO_ID)
    as_of = params.get("as_of", config.DEFAULT_AS_OF)
    provider_name = params.get("provider", "fixture")
    strict = bool(params.get("strict", False))
    run_gemini = bool(params.get("run_gemini", False))
    run_mode = _run_mode_from_params(params, default_compare_for_legacy_gemini=False)
    universe = _parse_universe_from_params(params)

    state = {
        "status": "running",
        "started_at": _utc_now(),
        "finished_at": None,
        "events": [],
        "run_log_path": None,
        "error": None,
        "universe": universe,
        "run_gemini": run_gemini,
        "run_mode": run_mode,
    }

    with ACTIVE_RUNS_LOCK:
        ACTIVE_RUNS[run_id] = state

    def subscriber(ev):
        with ACTIVE_RUNS_LOCK:
            state["events"].append(ev.to_dict())

    def worker():
        try:
            provider = _get_provider(provider_name)

            if run_mode in {"gemma_single", "gemini_c1"}:
                family = "gemma" if run_mode == "gemma_single" else "gemini"
                res = run_single_model_rebalance(
                    provider,
                    f"{portfolio_id}_{family}_single",
                    as_of,
                    universe,
                    strict=strict,
                    subscriber=subscriber,
                    model_family=family,
                )
            elif run_mode in {"gemma_debate", "gemini_debate"}:
                family = "gemma" if run_mode == "gemma_debate" else "gemini"
                res = run_rebalance(
                    provider,
                    f"{portfolio_id}_{family}_multi",
                    as_of,
                    universe,
                    strict=strict,
                    subscriber=subscriber,
                    model_family=family,
                )
            else:
                raise ValueError(
                    "the single-date polling endpoint accepts one configuration; "
                    "use the backtest WebSocket for compare or factorial mode"
                )

            with ACTIVE_RUNS_LOCK:
                state["status"] = "done"
                state["run_log_path"] = os.path.basename(res.get("run_log_path", ""))
                state["finished_at"] = _utc_now()

        except Exception as e:
            with ACTIVE_RUNS_LOCK:
                state["status"] = "error"
                state["error"] = f"{type(e).__name__}: {e}"
                state["finished_at"] = _utc_now()

    threading.Thread(target=worker, daemon=True).start()

    return _public_run_state(run_id, state)


@app.get("/api/live-runs/{run_id}")
def poll_live_run(run_id: str, after: int = 0):
    """Return events emitted after the caller's last seen sequence number."""
    with ACTIVE_RUNS_LOCK:
        state = ACTIVE_RUNS.get(run_id)

        if state is None:
            return JSONResponse({"error": "not found"}, status_code=404)

        return _public_run_state(run_id, state, after=after)

def _safe_path(name: str):
    if "/" in name or "\\" in name or ".." in name:
        return None
    for d in _search_dirs():
        path = os.path.join(d, name)
        if os.path.exists(path):
            return path
    return os.path.join(config.RUNS_DIR, name)


# --------------------------------------------------------------------------- #
# live run over WebSocket
# --------------------------------------------------------------------------- #
@app.websocket("/ws/run")
async def ws_run(ws: WebSocket):
    await ws.accept()
    try:
        params = await ws.receive_json()
    except Exception:
        await ws.close()
        return

    portfolio_id = params.get("portfolio_id", "backtest")
    start = params.get("start", config.START_DATE)
    end = params.get("end", config.END_DATE)
    provider_name = params.get("provider", "fixture")
    strict = bool(params.get("strict", False))
    run_gemini = bool(params.get("run_gemini", False))
    run_mode = _run_mode_from_params(params, default_compare_for_legacy_gemini=True)
    universe = _parse_universe_from_params(params)

    if provider_name == "live":
        from agentic_bl.providers.live import LiveProvider

        provider = LiveProvider()
    else:
        from agentic_bl.providers.fixture import FixtureProvider

        provider = FixtureProvider()

    evq: "queue.Queue" = queue.Queue()
    DONE = object()

    def subscriber(ev):
        evq.put(ev.to_dict())

    def progress(message: str):
        evq.put({"_meta": "progress", "message": message})

    def worker():
        try:
            summary = run_backtest(
                provider,
                universe,
                start,
                end,
                portfolio_id=portfolio_id,
                strict=strict,
                progress=progress,
                subscriber=subscriber,
                run_gemini=run_gemini,
                run_mode=run_mode,
            )
            evq.put({
                "_meta": "done",
                "summary_path": os.path.basename(summary.get("summary_path", "")),
                "run_log_paths": [os.path.basename(p) for p in summary.get("run_log_paths", []) if p],
                "months": len(summary.get("per_month", [])),
                "universe": universe,
                "run_mode": run_mode,
            })
        except Exception as e:  # surface to the client; do not hang
            evq.put({"_meta": "error", "error": f"{type(e).__name__}: {e}"})
        finally:
            evq.put(DONE)

    threading.Thread(target=worker, daemon=True).start()

    loop = asyncio.get_event_loop()
    try:
        while True:
            item = await loop.run_in_executor(None, evq.get)
            if item is DONE:
                break
            await ws.send_json(item)
    except WebSocketDisconnect:
        return
    finally:
        try:
            await ws.close()
        except Exception:
            pass


@app.get("/healthz")
def healthz():
    return {"ok": True}


# static UI (mounted last so /api, /ws and /healthz take precedence)
if os.path.isdir(UI_DIR):
    app.mount("/", StaticFiles(directory=UI_DIR, html=True), name="ui")
