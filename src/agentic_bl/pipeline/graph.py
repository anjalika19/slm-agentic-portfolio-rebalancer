"""LangGraph orchestrator."""

from __future__ import annotations
from typing import Any, Dict, TypedDict
from langgraph.graph import END, START, StateGraph
from agentic_bl.pipeline import phases
from agentic_bl.pipeline.state import RunContext, RunState


class GraphState(TypedDict):
    rs: Any
    ctx: Any


def _node_phase1(s: GraphState) -> Dict[str, Any]:
    phases.phase1_signals(s["rs"], s["ctx"])
    return {}


def _node_phase2(s: GraphState) -> Dict[str, Any]:
    phases.phase2_baseline(s["rs"], s["ctx"])
    return {}


def _node_debate(s: GraphState) -> Dict[str, Any]:
    phases.debate_round(s["rs"], s["ctx"])
    return {}


def _node_force_failures(s: GraphState) -> Dict[str, Any]:
    phases.force_consensus_lowconf(s["rs"], s["ctx"])   # BUG 4 FIX: was force_consensus_failures
    return {}


def _node_validate(s: GraphState) -> Dict[str, Any]:
    phases.validate(s["rs"], s["ctx"])
    return {}


def _node_finalize(s: GraphState) -> Dict[str, Any]:
    phases.finalize(s["rs"], s["ctx"])
    return {}


def _route_after_debate(s: GraphState) -> str:
    rs: RunState = s["rs"]
    ctx: RunContext = s["ctx"]
    if rs.contested and rs.debate_round < ctx.max_rounds:
        return "debate"
    return "force_failures"


def _route_after_phase2(s: GraphState) -> str:
    rs: RunState = s["rs"]
    return "debate" if rs.contested else "force_failures"


def build_graph():
    g = StateGraph(GraphState)
    g.add_node("phase1", _node_phase1)
    g.add_node("phase2", _node_phase2)
    g.add_node("debate", _node_debate)
    g.add_node("force_failures", _node_force_failures)
    g.add_node("validate", _node_validate)
    g.add_node("finalize", _node_finalize)

    g.add_edge(START, "phase1")
    g.add_edge("phase1", "phase2")
    g.add_conditional_edges("phase2", _route_after_phase2, {"debate": "debate", "force_failures": "force_failures"})
    g.add_conditional_edges("debate", _route_after_debate, {"debate": "debate", "force_failures": "force_failures"})
    g.add_edge("force_failures", "validate")
    g.add_edge("validate", "finalize")
    g.add_edge("finalize", END)
    return g.compile()