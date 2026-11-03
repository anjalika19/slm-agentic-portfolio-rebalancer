"""Structured, ordered debate event stream (spec section 5).

Every meaningful step of a rebalance emits an :class:`Event`. The emitter keeps a
monotonically increasing ``seq``, fans events out to any number of live
subscribers (the websocket server), and the whole log is persisted to
``runs/<portfolio_id>_<timestamp>.json`` so the UI can replay a recorded run
without ever re-invoking a model.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional


# Event types at minimum required by the spec.
# Replace with:
EVENT_TYPES = {
    "run_started",
    "signal_emitted",
    "single_model_result",
    "single_model_emitted",
    "single_model_pairwise_view",
    "baseline_view",
    "opening_statement",
    "rebuttal",
    "agent_message",
    "tiebreak",
    "resolution",
    "moderator_synthesis",
    "vote_cast",
    "consensus_result",
    "consensus_failure",
    "validator_result",
    "final_view",
    "optimized_portfolio",
    "warning",
    "run_finished",
    # Moderator process events (added with multi-round debate architecture)
    "moderator_framing",    # conflict classification surfaced before rebuttals
    "evidence_ledger",      # Round 2 cross-agent fact summary
    "round_summary",        # end-of-round factual ledger
      "agent_consensus_baseline",
    "agent_resolution_candidate",
    "agent_consensus_candidate",
    "moderator_final_decision",
    "moderator_skipped",
}

def _utc_now_iso(clock: Optional[Callable[[], datetime]]) -> str:
    now = clock() if clock is not None else datetime.now(timezone.utc)
    return now.isoformat()


@dataclass
class Event:
    seq: int
    ts: str
    phase: str
    type: str
    round: Optional[int] = None
    ticker: Optional[str] = None
    agent: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


Subscriber = Callable[[Event], None]


class EventEmitter:
    """Ordered emitter with live fan-out and a persisted JSON run log.

    Thread-safe: the backtest driver and server may emit from worker threads.
    A ``clock`` may be injected for deterministic tests.
    """

    def __init__(
        self,
        portfolio_id: str,
        as_of_date: str,
        runs_dir: str,
        meta: Optional[Dict[str, Any]] = None,
        clock: Optional[Callable[[], datetime]] = None,
        timestamp_label: Optional[str] = None,
    ) -> None:
        self.portfolio_id = portfolio_id
        self.as_of_date = as_of_date
        self.runs_dir = runs_dir
        self.meta = dict(meta or {})
        self._clock = clock
        self._seq = 0
        self._events: List[Event] = []
        self._subscribers: List[Subscriber] = []
        self._lock = threading.RLock()

        label = timestamp_label or _file_timestamp(clock)
        os.makedirs(runs_dir, exist_ok=True)
        fname = f"{_slug(portfolio_id)}_{_slug(as_of_date)}_{label}.json"
        self.path = os.path.join(runs_dir, fname)

    # ---- subscription -------------------------------------------------- #
    def subscribe(self, fn: Subscriber) -> Callable[[], None]:
        with self._lock:
            self._subscribers.append(fn)

        def _unsub() -> None:
            with self._lock:
                if fn in self._subscribers:
                    self._subscribers.remove(fn)

        return _unsub

    # ---- emission ------------------------------------------------------ #
    def emit(
        self,
        type: str,
        phase: str,
        payload: Optional[Dict[str, Any]] = None,
        *,
        round: Optional[int] = None,
        ticker: Optional[str] = None,
        agent: Optional[str] = None,
    ) -> Event:
        if type not in EVENT_TYPES:
            # Never silently swallow an unknown type -- surface it loudly.
            raise ValueError(f"Unknown event type: {type!r}. Known: {sorted(EVENT_TYPES)}")
        with self._lock:
            self._seq += 1
            ev = Event(
                seq=self._seq,
                ts=_utc_now_iso(self._clock),
                phase=phase,
                type=type,
                round=round,
                ticker=ticker,
                agent=agent,
                payload=dict(payload or {}),
            )
            self._events.append(ev)
            subs = list(self._subscribers)
        for fn in subs:
            try:
                fn(ev)
            except Exception:  # a dead subscriber must not break the run
                pass
        return ev

    def warning(self, message: str, phase: str = "system", **payload: Any) -> Event:
        return self.emit("warning", phase, {"message": message, **payload})

    # ---- persistence --------------------------------------------------- #
    @property
    def events(self) -> List[Event]:
        with self._lock:
            return list(self._events)

    def to_run_log(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "portfolio_id": self.portfolio_id,
                "as_of_date": self.as_of_date,
                "meta": self.meta,
                "n_events": len(self._events),
                "events": [e.to_dict() for e in self._events],
            }

    def save(self) -> str:
        log = self.to_run_log()
        tmp = self.path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(log, fh, indent=2, default=_json_default)
        os.replace(tmp, self.path)
        return self.path


def _json_default(obj: Any) -> Any:
    import numpy as np

    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in str(s))


def _file_timestamp(clock: Optional[Callable[[], datetime]]) -> str:
    now = clock() if clock is not None else datetime.now(timezone.utc)
    return now.strftime("%Y%m%dT%H%M%S")


def load_run_log(path: str) -> Dict[str, Any]:
    with open(path) as fh:
        return json.load(fh)
