"""Mutable run state for one rebalance + the context object carried through nodes.

The state is deliberately explicit about the thing the old system got wrong: the
vote must consume each agent's *revised* post-debate position, so we record
``revised_positions[round][ticker][agent]`` and never re-derive from Phase-1
signals except in round 0 (spec 3a).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from events import EventEmitter
from providers.base import InputBundle


@dataclass
class RunState:
    bundle: InputBundle
    universe_order: List[str]

    # Phase 1
    agent_outputs: Dict[str, Dict] = field(default_factory=dict)   # agent -> SpecialistAgentOutput
    used_mock: Dict[str, bool] = field(default_factory=dict)        # agent -> bool

    # Phase 2 / working BL views: ticker -> view dict
    working_views: Dict[str, Dict] = field(default_factory=dict)

    # Phase 3 debate ledgers (keyed by round)
    opening_statements: Dict[int, Dict[str, Dict[str, Dict]]] = field(default_factory=dict)
    rebuttals: Dict[int, Dict[str, Dict[str, Dict]]] = field(default_factory=dict)
    # THE fix (3a): revised positions written by each rebuttal, read by the vote
    revised_positions: Dict[int, Dict[str, Dict[str, Dict]]] = field(default_factory=dict)
    moderator_syntheses: Dict[int, Dict[str, Dict]] = field(default_factory=dict)
    votes: Dict[int, Dict[str, Dict[str, str]]] = field(default_factory=dict)

    consensus: Dict[str, Dict] = field(default_factory=dict)        # ticker -> resolution record
    contested: List[str] = field(default_factory=list)             # tickers still in debate
    debate_round: int = 0

    # covariance / equilibrium prior (computed once, reused by omega + posterior)
    sigma: Any = None                  # np.ndarray, annualized PSD
    cov_cond: Optional[float] = None
    cov_note: str = ""
    pi: Any = None                     # np.ndarray, equilibrium excess returns
    market_weights: Dict[str, float] = field(default_factory=dict)

    # Phase 4 / validator
    validator: Dict[str, Dict] = field(default_factory=dict)        # agent -> result
    final_views: Dict[str, Any] = field(default_factory=dict)
    optimized: Dict[str, Any] = field(default_factory=dict)

    def latest_revised(self, ticker: str, agent: str) -> Optional[Dict]:
        """Latest revised position for (ticker, agent) across completed rounds, or None."""
        for r in range(self.debate_round, 0, -1):
            rec = self.revised_positions.get(r, {}).get(ticker, {}).get(agent)
            if rec is not None:
                return rec
        return None


@dataclass
class RunContext:
    emitter: EventEmitter
    strict: bool
    scale: float
    max_rounds: int
    accept_threshold: int
    model_family: str = "gemma"
    model_name: str = ""
    sham: bool = False  # negative control: scramble peer facts in the debate (spec B4)
    calibrator: Any = None
