"""The five pipeline phases as plain functions over (RunState, RunContext).

These are wired by the LangGraph orchestrator in ``graph.py`` and are also directly
unit-testable. Each phase emits the structured events the UI replays.

Correctness fixes implemented here:
  3a  the vote reads ``state.latest_revised(...)`` (post-debate position), only
      falling back to the Phase-1 signal in round 0;
  3e  unresolved tickers after MAX rounds are forced to neutral with a logged
      ``consensus_failure`` event -- never the moderator's last guess silently;
  3f  ``used_mock`` is tracked per agent and a warning event is emitted;
  3g  an agent's strongest_counterargument_agent may not be itself.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import numpy as np

from agentic_bl import bl_core
from agentic_bl import config
from agentic_bl import calibration
from agentic_bl.agents import mock
from agentic_bl.agents import payloads
from agentic_bl.agents.llm import structured_call
from agentic_bl.agents.normalize import normalize
from agentic_bl.agents.prompts import (
    SPECIALIST_PROMPTS,
    MODERATOR_SYNTHESIS_PROMPT,
    opening_statement_prompt,
    rebuttal_statement_prompt,
)
from agentic_bl.pipeline.state import RunContext, RunState
from agentic_bl.pipeline.validator import validate_agent, validate_rebuttal
from agentic_bl.schemas import (
    AgentOpeningStatement,
    AgentRebuttalStatement,
    EMISSION_SCHEMAS,
    ModeratorSynthesis,
)


def _direction(sig: float) -> str:
    if sig > 0.20:
        return "long"
    if sig < -0.20:
        return "short"
    return "neutral"


def direction_from_q(q: float):
    """Derive a display direction from q without erasing small calibrated views.

    Earlier we forced q=0 whenever |q| < NEUTRAL_BAND. That made many stock-level
    views disappear before the shared pairwise converter saw them, so Gemma/Gemini
    often collapsed to the equal-weight prior. Keep calibrated stock-level q for
    ranking, while the shared converter filters adjacent spreads that are too
    small to justify an active BL constraint.
    """
    q = _safe(q)
    if abs(q) < config.VIEW_SPREAD_FLOOR:
        return "neutral", q
    return ("short", q) if q < 0 else ("long", q)


def _safe(x, default: float = 0.0) -> float:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _anchored_q(prior_q: float, candidate_q: float, alpha: float) -> float:
    """Apply a bounded debate update instead of replacing the prior outright."""
    alpha = float(np.clip(_safe(alpha, 0.0), 0.0, 1.0))
    return _safe(prior_q) + alpha * (_safe(candidate_q) - _safe(prior_q))


def _reliability_capped_alpha(calibrated_view, configured_alpha: float):
    """Return (effective_alpha, learned_reliability) for a calibrated source.

    Debate is allowed to move the committed Phase-1 view only to the extent that
    the same source family demonstrated historical reliability. Missing or
    under-sampled family calibration receives zero update authority rather than
    silently falling back to a generic model-family prior.
    """
    params = getattr(calibrated_view, "params", None)
    reliability = getattr(params, "reliability", None)
    n_samples = int(getattr(params, "n_samples", 0) or 0)
    if reliability is None or n_samples < config.CALIBRATION_MIN_SAMPLES:
        learned_reliability = 0.0
    else:
        learned_reliability = float(np.clip(_safe(reliability, 0.0), 0.0, 1.0))
    configured_alpha = float(np.clip(_safe(configured_alpha, 0.0), 0.0, 1.0))
    return min(configured_alpha, learned_reliability), learned_reliability


def _eligible_latest_directions(state: RunState, ticker: str) -> Dict[str, str]:
    """Latest valid specialist directions used to decide whether adjudication remains."""
    directions: Dict[str, str] = {}
    for agent in config.SPECIALISTS:
        revised = state.latest_revised(ticker, agent)
        if revised is not None:
            eligible = (
                bool(revised.get("eligible_to_vote", True))
                and revised.get("participation_status") != "invalid_rebuttal"
            )
            if not eligible:
                continue
            direction = revised.get("direction")
            if not direction:
                direction = direction_from_q(_safe(revised.get("q")))[0]
        else:
            signal = _safe(
                state.agent_outputs.get(agent, {})
                .get("signals", {})
                .get(ticker, {})
                .get("signal")
            )
            direction = _direction(signal)
        if direction in {"long", "short", "neutral"}:
            directions[agent] = direction
    return directions


def _has_long_short_conflict(directions: Dict[str, str]) -> bool:
    values = set(directions.values())
    return "long" in values and "short" in values


def _parallel_agents_enabled() -> bool:
    """Whether independent specialist model calls may run concurrently.

    This is deliberately opt-in because some local Ollama setups queue requests
    internally or run out of VRAM if too many generations happen at once.
    """
    return bool(
        getattr(config, "PARALLEL_AGENT_CALLS", False)
        and int(getattr(config, "AGENT_PARALLEL_WORKERS", 1) or 1) > 1
    )


def _map_agents(agents, fn):
    """Run fn(agent) sequentially or concurrently and return {agent: result}.

    Results are returned by agent name so callers can still emit events in the
    canonical config.SPECIALISTS order. Exceptions propagate exactly as before.
    """
    agents = list(agents)
    if not _parallel_agents_enabled() or len(agents) <= 1:
        return {agent: fn(agent) for agent in agents}

    workers = max(1, min(int(config.AGENT_PARALLEL_WORKERS), len(agents)))
    out = {}
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="agent") as ex:
        futures = {ex.submit(fn, agent): agent for agent in agents}
        for fut in as_completed(futures):
            agent = futures[fut]
            out[agent] = fut.result()
    return out


# --------------------------------------------------------------------------- #
# covariance + prior (computed once)
# --------------------------------------------------------------------------- #
def ensure_market_model(state: RunState, ctx: RunContext) -> None:
    if state.sigma is not None:
        return
    cov = bl_core.covariance_from_block(
        state.bundle.covariance, state.universe_order, strict=ctx.strict
    )
    state.sigma = cov.matrix
    state.cov_cond = cov.condition_number
    state.cov_note = cov.note
    caps = state.bundle.market_caps
    n = len(state.universe_order)
    equal_weights = {t: 1.0 / n for t in state.universe_order}
    cap_weights = None
    if caps and all(t in caps for t in state.universe_order):
        raw = np.array([float(caps[t]) for t in state.universe_order], dtype=float)
        if np.all(np.isfinite(raw)) and raw.sum() > 0:
            raw = raw / raw.sum()
            cap_weights = {t: float(raw[i]) for i, t in enumerate(state.universe_order)}
            ctx.emitter.emit(
                "baseline_view", phase="market_model",
                payload={"type": "available_market_cap_weights", "source": "market_cap",
                         "weights": {t: round(cap_weights[t], 4) for t in state.universe_order}},
            )

    # For small custom research baskets, the benchmark shown in the UI is equal
    # weight. Using cap weights as the BL prior can make the optimizer infeasible
    # under MAX_WEIGHT + active-budget constraints, which caused silent equal-weight
    # fallbacks even though the agents had emitted views. Default: equal-weight prior.
    if config.BL_PRIOR_MODE == "market_cap" and cap_weights is not None:
        state.market_weights = cap_weights
        prior_source = "market_cap"
    else:
        state.market_weights = equal_weights
        prior_source = "equal_weight_research_prior"
        if config.BL_PRIOR_MODE == "market_cap" and cap_weights is None:
            ctx.emitter.warning(
                "BL_PRIOR_MODE=market_cap but market caps are unavailable; using equal-weight prior",
                phase="market_model",
            )

    ctx.emitter.emit(
        "baseline_view", phase="market_model",
        payload={"type": "bl_prior_weights", "source": prior_source,
                 "weights": {t: round(state.market_weights[t], 4) for t in state.universe_order}},
    )
    w = np.array([state.market_weights[t] for t in state.universe_order], dtype=float)
    state.pi = bl_core.equilibrium_returns(state.sigma, w)
    ctx.emitter.emit(
        "warning" if cov.ridge_applied else "baseline_view",
        phase="market_model",
        payload={
            "type": "covariance",
            "condition_number": round(cov.condition_number, 3),
            "ridge_applied": cov.ridge_applied,
            "note": cov.note,
        },
    )


# --------------------------------------------------------------------------- #
# Phase 1 -- independent specialist signals
# --------------------------------------------------------------------------- #
def phase1_signals(state: RunState, ctx: RunContext) -> None:
    """Sequential design (spec / user): the Economist runs FIRST and its regime +
    sector views are fed into the other three specialists. Agents emit categorical
    convictions + agent-specific fields (no q/P) -- those appear only at the
    moderator's baseline."""
    # 1) economist first (sector-level macro view)
    _run_specialist(state, ctx, "economist", macro_context=None)
    econ_view = state.agent_outputs["economist"].get("agent_view", {})
    macro_context = {
        "risk_regime": econ_view.get("risk_regime"),
        "rate_environment": econ_view.get("rate_environment"),
        "favored_sectors": econ_view.get("favored_sectors", []),
        "risk_sectors": econ_view.get("risk_sectors", []),
        "sector_views": econ_view.get("sector_views", {}),
    }
    # 2) the other three, each fed the economist's macro context. Keep Phase 1
    # sequential because _run_specialist writes state and streams UI events directly.
    # The heavier debate opening/rebuttal calls below are parallelized safely by
    # collecting results first, then emitting in canonical agent order.
    for agent in ("quant", "fundamental", "sentiment"):
        _run_specialist(state, ctx, agent, macro_context=macro_context)


def _run_specialist(state: RunState, ctx: RunContext, agent: str, macro_context) -> None:
    package = state.bundle.packages[agent]
    # The model gets a COMPACT payload (salient numbers, field names preserved so
    # cited_values still trace); the full package stays in state for validator/UI.
    payload = {
        "portfolio_context": {
            "portfolio_id": state.bundle.portfolio_id,
            "as_of_date": state.bundle.as_of_date,
            "horizon": state.bundle.horizon,
            "portfolio_universe": state.universe_order,
        },
       "agent_input": payloads.compact_package(
        agent, package, as_of_date=state.bundle.as_of_date   # ← pass date
    ),
    }
    if macro_context is not None:
        payload["macro_context_from_economist"] = macro_context

    res = structured_call(
        role=agent,
        system_prompt=SPECIALIST_PROMPTS[agent],
        user_payload=payload,
        schema_model=EMISSION_SCHEMAS[agent],
        mock_fn=lambda a=agent, p=package: mock.mock_specialist_output(a, p),
        strict=ctx.strict,
        model_family=ctx.model_family,
        model_name=ctx.model_name,
        temperature=config.temperature_for_call(agent, phase="phase1"),
    )
    normalized = normalize(agent, res.data, state.universe_order, state.bundle.sectors)

    # A1: enforce FULL-UNIVERSE coverage. If the agent omitted tickers, re-prompt once
    # with the explicit missing list; then fill an explicit abstain for any still
    # missing and emit a warning. A run must never reach the portfolio with an agent
    # silently covering only some names.
    missing = [t for t in state.universe_order if t not in normalized.get("signals", {})]
    if missing and not res.used_mock:
        retry = structured_call(
            role=agent,
            system_prompt=SPECIALIST_PROMPTS[agent]
            + f"\n\nYou MUST return a signal for EVERY ticker. You omitted {missing}. "
              f"Return all {len(state.universe_order)} tickers: {state.universe_order}.",
            user_payload=payload,
            schema_model=EMISSION_SCHEMAS[agent],
            mock_fn=lambda a=agent, p=package: mock.mock_specialist_output(a, p),
            strict=False,  # coverage retry must not raise; we fall back to abstain
            model_family=ctx.model_family,
            model_name=ctx.model_name,
            temperature=config.temperature_for_call(agent, phase="phase1"),
        )
        normalized = normalize(agent, retry.data, state.universe_order, state.bundle.sectors)
        missing = [t for t in state.universe_order if t not in normalized.get("signals", {})]
    for t in missing:
        normalized.setdefault("signals", {})[t] = {
            "signal": 0.0, "signal_label": "neutral", "confidence": 0.05,
            "rationale": "[abstain] agent did not provide a signal for this ticker.",
            "cited_values": [], "extra": {"abstain": True},
        }
        ctx.emitter.warning(
            f"{agent}: no signal for {t} after re-prompt; filled explicit abstain",
            phase="phase1", agent=agent, ticker=t,
        )

    state.agent_outputs[agent] = normalized
    state.used_mock[agent] = res.used_mock
    if res.used_mock:
        ctx.emitter.warning(
            f"{agent}: model unavailable, used deterministic mock signal",
            phase="phase1", agent=agent, detail=res.note,
        )
    agent_view = normalized.get("agent_view", {})
    for t in state.universe_order:
        sig = normalized.get("signals", {}).get(t, {})
        ctx.emitter.emit(
            "signal_emitted",
            phase="phase1",
            ticker=t,
            agent=agent,
            payload={
                "signal": _safe(sig.get("signal")),
                "signal_label": sig.get("signal_label"),
                "confidence": _safe(sig.get("confidence")),
                "direction": _direction(_safe(sig.get("signal"))),
                "rationale": sig.get("rationale", ""),
                "cited_values": sig.get("cited_values", []),
                "extra": sig.get("extra", {}),
                "abstain": bool(sig.get("extra", {}).get("abstain")),
                "agent_view": agent_view if agent in ("economist", "quant") else None,
                "used_mock": res.used_mock,
            },
        )


# --------------------------------------------------------------------------- #
# Phase 2 -- agent-derived baseline views
# --------------------------------------------------------------------------- #
def phase2_baseline(state: RunState, ctx: RunContext) -> None:
    ensure_market_model(state, ctx)
    calibrator = ctx.calibrator or calibration.load_view_calibrator()
    pending = []
    for t in state.universe_order:
        sigs = [
            _safe(state.agent_outputs[a].get("signals", {}).get(t, {}).get("signal"))
            for a in config.SPECIALISTS
        ]
        confs = [
            _safe(state.agent_outputs[a].get("signals", {}).get(t, {}).get("confidence"), 0.5)
            for a in config.SPECIALISTS
        ]
        mean_sig = float(np.mean(sigs))
        raw_q = calibration.ticker_signal_to_raw_q(mean_sig)
        # agreement -> raw confidence: lower when agents disagree (std high)
        disagreement = float(np.std(sigs))
        raw_conf = float(np.clip(np.mean(confs) * (1.0 - min(disagreement, 1.0)), 0.05, 0.95))
        cal = calibrator.calibrate(
            raw_q,
            raw_conf,
            family=f"{ctx.model_family}_phase1_consensus",
        )
        directions = {_direction(s) for s in sigs}
        direction_conflict = ("long" in directions and "short" in directions)
        agent_raw_qs = [calibration.ticker_signal_to_raw_q(s) for s in sigs]
        q_spread = float(max(agent_raw_qs) - min(agent_raw_qs)) if agent_raw_qs else 0.0
        q_only_contested = bool(
            config.ENABLE_Q_ONLY_DEBATE
            and not direction_conflict
            and q_spread >= config.DEBATE_DISAGREEMENT_Q
        )
        is_contested = direction_conflict or q_only_contested
        contest_reason = (
            "direction_conflict" if direction_conflict
            else "q_only_disagreement" if q_only_contested
            else "phase1_agreement"
        )
        pending.append(
            (t, cal, mean_sig, sigs, disagreement, q_spread, is_contested, contest_reason)
        )

    # build omega via Idzorek using the full P stack and calibrated confidences
    P = np.array([bl_core.p_row_absolute(t, state.universe_order) for t in state.universe_order])
    Omega = bl_core.idzorek_omega(P, state.sigma, np.array([x[1].confidence for x in pending]))

    contested: List[str] = []
    for i, (t, cal, mean_sig, sigs, disagreement, q_spread, is_contested, contest_reason) in enumerate(pending):
        direction, q = direction_from_q(cal.q)  # label consistent with calibrated q (A3)
        view = {
            "ticker": t,
            "direction": direction,
            "q": round(q, 6),
            "raw_q": round(cal.raw_q, 6),
            "confidence": round(cal.confidence, 4),
            "raw_confidence": round(cal.raw_confidence, 4),
            "calibration": cal.as_metadata(),
            "omega": round(float(Omega[i, i]), 10),
            "p_row": bl_core.p_row_absolute(t, state.universe_order),
            "source": "agent_phase1_consensus",
            "agent_signals": {a: round(_safe(s), 4) for a, s in zip(config.SPECIALISTS, sigs)},
            "disagreement_std": round(disagreement, 4),
            "raw_q_spread": round(q_spread, 6),
            "contest_reason": contest_reason,
            "resolved": not is_contested,
        }
        state.working_views[t] = view
        if is_contested:
            contested.append(t)
        ctx.emitter.emit(
            "baseline_view",
            phase="phase2",
            ticker=t,
            payload={**view, "contested": is_contested},
        )
        ctx.emitter.emit(
            "agent_consensus_baseline", phase="phase2", round=0, ticker=t,
            payload={
                "kind": "agent_phase1_consensus",
                "direction": view["direction"],
                "q": view["q"],
                "raw_q": view["raw_q"],
                "confidence": view["confidence"],
                "raw_confidence": view["raw_confidence"],
                "calibration": view["calibration"],
                "message": (
                    f"Agent-derived baseline for {t}: {view['direction']} "
                    f"(raw_q={view['raw_q']:+.3f} -> calibrated q={view['q']:+.3f}, "
                    f"raw_conf={view['raw_confidence']:.2f} -> calibrated conf={view['confidence']:.2f})."
                    + (
                        " Agents disagree on direction — opening debate."
                        if contest_reason == "direction_conflict"
                        else " Large q-only disagreement — opening configured ablation debate."
                        if contest_reason == "q_only_disagreement"
                        else " Agents do not straddle long and short; no debate needed."
                    )
                ),
            },
        )
    state.contested = contested
    # tickers never contested are resolved at baseline
    for t in state.universe_order:
        if t not in contested:
            state.consensus[t] = {
                "resolved": True, "round": 0, "via": "baseline_agreement", "accepts": None,
            }

def _build_evidence_ledger(state: RunState, ticker: str, from_round: int) -> Dict:
    """Factual summary of what each agent cited in the previous round.
    No interpretation — just: who cited what number, and is it in the input package."""
    from agentic_bl.pipeline.validator import _collect_numbers, _traces
    
    ledger = {}
    for agent in config.SPECIALISTS:
        opening = state.opening_statements.get(from_round, {}).get(ticker, {}).get(agent, {})
        evidence = opening.get("evidence") or []
        pool = []
        _collect_numbers(state.bundle.packages.get(agent, {}), pool)
        
        cited = []
        for ev in evidence:
            val = ev.get("value")
            try:
                v = float(val)
                traceable = _traces(v, pool, 0.05)
            except (TypeError, ValueError):
                traceable = False
            cited.append({
                "metric": ev.get("name"),
                "value": val,
                "traceable": traceable,
                "note": ev.get("note", ""),
            })
        ledger[agent] = {
            "direction": opening.get("direction"),
            "q": opening.get("q"),
            "cited_evidence": cited,
            "n_traceable": sum(1 for c in cited if c["traceable"]),
        }
    return ledger

# --------------------------------------------------------------------------- #
# Phase 3 -- one debate round over the contested tickers
# --------------------------------------------------------------------------- #
def _agent_current_signal(state: RunState, ticker: str, agent: str, scale: float) -> Dict:
    """The agent's current stance as a signal-dict (latest revised, else Phase-1)."""
    rev = state.latest_revised(ticker, agent)
    phase1 = state.agent_outputs[agent].get("signals", {}).get(ticker, {})
    if rev is not None:
        return {
            "signal": _safe(rev.get("q")) / scale if scale else 0.0,
            "confidence": _safe(rev.get("confidence"), 0.5),
            "rationale": phase1.get("rationale", ""),
            "cited_values": phase1.get("cited_values", []),
        }
    return {
        "signal": _safe(phase1.get("signal")),
        "confidence": _safe(phase1.get("confidence"), 0.5),
        "rationale": phase1.get("rationale", ""),
        "cited_values": phase1.get("cited_values", []),
    }


def _agent_context(state: RunState, agent: str, ticker: str, scale: float) -> Dict:
    cur = _agent_current_signal(state, ticker, agent, scale)
    p1 = state.agent_outputs[agent].get("signals", {}).get(ticker, {})
    return {
        "agent_name": agent,
        "ticker": ticker,
        "phase1_signal": {
            "signal": p1.get("signal"),
            "confidence": p1.get("confidence"),
            "rationale": (p1.get("rationale") or "")[:220],
            "cited_values": (p1.get("cited_values") or [])[:4],
        },
        "current_stance": {
            "direction": _direction(_safe(cur.get("signal"))),
            "q": round(_safe(cur.get("signal")) * scale, 4),
            "confidence": round(_safe(cur.get("confidence"), 0.5), 3),
        },
        "used_mock": state.used_mock.get(agent, False),
    }

def _classify_opening_disagreement(openings: Dict, state: RunState, ticker: str) -> Dict:
    dirs = {a: o.get("direction") for a, o in openings.items()}
    bull = [a for a, d in dirs.items() if d == "long"]
    bear = [a for a, d in dirs.items() if d == "short"]
    
    if not bull or not bear:
        return {"type": "aligned", "resolution_path": "No debate needed; agents agree.", 
                "sides": {}, "disputed_numbers": []}
    
    # Check for same-metric conflicts across bull/bear cited evidence
    disputed = []
    for a in bull:
        for b in bear:
            ev_a = {e["name"]: e["value"] for e in (openings[a].get("evidence") or [])}
            ev_b = {e["name"]: e["value"] for e in (openings[b].get("evidence") or [])}
            for name in set(ev_a) & set(ev_b):
                if abs(float(ev_a[name] or 0) - float(ev_b[name] or 0)) > 0.05:
                    disputed.append({"metric": name, "bull_value": ev_a[name], 
                                     "bear_value": ev_b[name]})
    
    if disputed:
        conflict_type = "factual"
        path = ("One of these numbers is wrong or outdated. Cite the source. "
                "The side whose number is traceable to the input package wins.")
    elif len(bull) >= 2 and len(bear) >= 2:
        conflict_type = "weighting"
        path = ("Both sides have real evidence. Argue about which matters MORE for "
                "next month's return specifically. Context agents (Economist, Sentiment) "
                "can break the tie if their evidence is stock-specific.")
    else:
        conflict_type = "horizon"
        path = ("One side has trailing data; the other has forward catalysts. "
                "Argue about whether the catalyst bites within the next 30 days.")
    staleness_warning = None
    fund_opening = openings.get("fundamental", {})
    fund_evidence = fund_opening.get("evidence") or []
    # The compact payload now includes days_since_earnings in the input
    # If the fundamental agent cited ratios and data is aging/stale, flag it
    fund_pkg = state.bundle.packages.get("fundamental", {})
    fund_stock = (fund_pkg.get("stocks") or {}).get(ticker, {})
    days = fund_stock.get("days_since_earnings")
    freshness = fund_stock.get("data_freshness", "unknown")
    if freshness in ("aging", "stale") and (bull or bear):
        staleness_warning = (
            f"Note: Fundamental data for {ticker} is {days} days old ({freshness}). "
            f"If Fundamental is arguing from ratios, next earnings may change these numbers. "
            f"Sentiment and Quant data is more current."
        )
    
    return {
        "type": conflict_type,
        "resolution_path": path,
        "sides": {"bull": bull, "bear": bear},
        "disputed_numbers": disputed,
        "staleness_warning": staleness_warning,   
    }

def debate_round(state: RunState, ctx: RunContext) -> None:
    state.debate_round += 1
    r = state.debate_round
    scale = ctx.scale
    state.opening_statements[r] = {}
    state.rebuttals[r] = {}
    state.revised_positions[r] = {}
    state.moderator_syntheses[r] = {}
    state.votes[r] = {}

    prev_openings = None  # for the sham control: feed a DIFFERENT ticker's peer facts
    for ticker in list(state.contested):
        issue = {
            "ticker": ticker,
            "current_bl_view": payloads.compact_view(state.working_views[ticker]),
            "round": r,
        }

        # Round 2 openings must start with the Round 1 evidence ledger.
        # Previously this was injected after openings, so openings could not use it.
        if r == 2:
            evidence_ledger = _build_evidence_ledger(state, ticker, r - 1)
            issue["evidence_ledger_from_round1"] = evidence_ledger
            ctx.emitter.emit(
                "evidence_ledger",
                phase="debate",
                round=r,
                ticker=ticker,
                payload=evidence_ledger,
            )

        # ---- openings ------------------------------------------------- #
        openings: Dict[str, Dict] = {}

        def _opening_call(agent: str):
            cur = _agent_current_signal(state, ticker, agent, scale)
            return structured_call(
                role=agent,
                system_prompt=opening_statement_prompt(agent, debate_round=r, max_rounds=ctx.max_rounds),
                user_payload={
                    "round": r,
                    "issue": issue,
                    "your_agent_context": _agent_context(state, agent, ticker, scale),
                },
                schema_model=AgentOpeningStatement,
                mock_fn=lambda a=agent, c=cur: mock.mock_opening_statement(a, ticker, c, scale),
                strict=ctx.strict,
                model_family=ctx.model_family,
                model_name=ctx.model_name,
                temperature=config.temperature_for_call(agent, phase="debate", debate_round=r, max_rounds=ctx.max_rounds),
            )

        opening_results = _map_agents(config.SPECIALISTS, _opening_call)
        for agent in config.SPECIALISTS:
            res = opening_results[agent]
            openings[agent] = res.data
            ctx.emitter.emit(
                "opening_statement", phase="debate", round=r, ticker=ticker, agent=agent,
                payload={**res.data, "used_mock": res.used_mock},
            )
            # roundtable transcript: the opening is the agent's first message this round
            ctx.emitter.emit(
                "agent_message", phase="debate", round=r, ticker=ticker, agent=agent,
                payload={
                    "kind": "opening", "responding_to": None, "stance": None,
                    "direction": res.data.get("direction"), "q": res.data.get("q"),
                    "message": res.data.get("position_rationale", ""),
                },
            )
        state.opening_statements[r][ticker] = openings

        framing = _classify_opening_disagreement(openings, state, ticker)
        ctx.emitter.emit(
            "moderator_framing", phase="debate", round=r, ticker=ticker,
            payload={
                "conflict_type": framing["type"],
                "what_will_resolve_this": framing["resolution_path"],
                "agents_on_each_side": framing["sides"],
                "key_numbers_in_dispute": framing["disputed_numbers"],
                "staleness_warning": framing.get("staleness_warning"),  # ← NEW
            }
        )

        # ---- rebuttals (write revised positions: spec 3a) ------------- #
        rebuttals: Dict[str, Dict] = {}
        state.revised_positions[r][ticker] = {}
        # SHAM negative control (spec B4): feed each agent a DIFFERENT ticker's peer
        # facts (the previous contested ticker's openings), so the "information transfer"
        # carries genuinely wrong information. Real debate must beat this or the
        # deliberation is just noise the aggregation smooths.
        peer_view = openings
        if ctx.sham:
            if prev_openings is not None:
                peer_view = prev_openings                       # wrong-ticker facts
            else:  # first ticker has no predecessor -> sign-flip its facts
                peer_view = {a: {**o, "q": -_safe(o.get("q")),
                                 "direction": {"long": "short", "short": "long"}.get(o.get("direction"), "neutral")}
                             for a, o in openings.items()}

        def _rebuttal_call(agent: str):
            # pick the 1-2 peers whose direction most disagrees with this agent (spec B1)
            peers = _disagreeing_peers(agent, peer_view)
            peer_openings = _peer_openings_block(agent, peer_view)
            rebuttal_whiteboard = _rebuttal_whiteboard(state, agent, ticker, scale, peer_view, peers)
            res = structured_call(
                role=agent,
                system_prompt=rebuttal_statement_prompt(agent, r, _format_peer_openings(peer_openings), max_rounds=ctx.max_rounds),
                user_payload={
                    "round": r,
                    "issue": issue,
                    "your_agent_context": _agent_context(state, agent, ticker, scale),
                    "peer_openings": peer_openings,  # dedicated quoting source; not rationale-compressed
                    "peers_to_address": peers,  # respond to these BY NAME
                    "opening_statements_from_all_agents": {
                        a: payloads.compact_opening(o) for a, o in peer_view.items()
                    },
                    "whiteboard_for_number_tracing": rebuttal_whiteboard,
                },
                schema_model=AgentRebuttalStatement,
                mock_fn=lambda a=agent: mock.mock_rebuttal_statement(
                    a, ticker, openings[a], peer_view, scale,
                    round_no=r, working_q=_safe(state.working_views[ticker]["q"]),
                ),
                strict=ctx.strict,
                model_family=ctx.model_family,
                model_name=ctx.model_name,
                temperature=config.temperature_for_call(agent, phase="debate", debate_round=r, max_rounds=ctx.max_rounds),
            )
            data, res = _validated_rebuttal_or_freeze(
                state, ctx, agent, ticker, r, issue, openings, peer_view, peers, res, scale, rebuttal_whiteboard
            )
            return {"data": data, "res": res}

        rebuttal_results = _map_agents(config.SPECIALISTS, _rebuttal_call)
        for agent in config.SPECIALISTS:
            item = rebuttal_results[agent]
            data, res = item["data"], item["res"]
            rebuttals[agent] = data
            # the revised position the vote will consume
            state.revised_positions[r][ticker][agent] = {
                "direction": data.get("revised_direction"),
                "q": _safe(data.get("revised_q")),
                "confidence": _safe(data.get("revised_confidence"), 0.5),
                "participation_status": data.get("participation_status", "valid"),
                "eligible_to_vote": bool(data.get("eligible_to_vote", True)),
                "adjudication_weight_multiplier": _safe(
                    data.get("adjudication_weight_multiplier"), 1.0
                ),
                "concessions": data.get("concessions", []) or [],
            }
            ctx.emitter.emit(
                "rebuttal", phase="debate", round=r, ticker=ticker, agent=agent,
                payload={**data, "used_mock": res.used_mock},
            )
            # roundtable transcript: each by-name reply is its own ordered message.
            # Some strict/local model outputs pass schema but leave replies empty; still
            # emit a visible rebuttal line so the UI has a real debate transcript.
            replies = data.get("replies", []) or []
            if not replies:
                ctx.emitter.emit(
                    "agent_message", phase="debate", round=r, ticker=ticker, agent=agent,
                    payload={
                        "kind": "rebuttal", "responding_to": data.get("responded_to") or None,
                        "stance": None, "direction": data.get("revised_direction"),
                        "q": data.get("revised_q"), "message": _rebuttal_summary(data),
                    },
                )
            for reply in replies:
                tgt = reply.get("responding_to")
                if not tgt or tgt == agent:  # A4: never address yourself
                    continue
                ctx.emitter.emit(
                    "agent_message", phase="debate", round=r, ticker=ticker, agent=agent,
                    payload={
                        "kind": "reply", "responding_to": tgt,
                        "stance": reply.get("stance"), "quoted_claim": reply.get("quoted_claim", ""),
                        "cited_number": reply.get("cited_number"),
                        "message": reply.get("my_response") or reply.get("message", ""),
                    },
                )
        state.rebuttals[r][ticker] = rebuttals

   
        # ---- agent consensus candidate (no moderator during debate) -----
        # Agents generate the candidate view together through their revised positions.
        # The moderator is reserved only for the post-max-round no-consensus fallback.
        if r == ctx.max_rounds:
            resolution = _extract_agent_bl_views(state, ctx, ticker, openings, rebuttals)
        else:
            resolution = _evidence_gated_resolution(state, ctx, ticker, openings, rebuttals, r)
        resolution = _guard_invalid_neutral_resolution(
            resolution=resolution,
            rebuttals=rebuttals,
            ticker=ticker,
        )
        if resolution:
            ctx.emitter.emit("agent_resolution_candidate", phase="debate", round=r, ticker=ticker, payload=resolution)

        q_agg = _safe(resolution.get("q"))
        conf_agg = _safe(resolution.get("confidence"), 0.3)
        consensus_rec = {
            "ticker": ticker,
            "recommended_q": q_agg,
            "recommended_confidence": conf_agg,
            "reasoning_summary": resolution.get("reason", "agent evidence-gated resolution"),
            # BUG FIX (calibration mismatch): this used to be the hardcoded literal
            # "agent_consensus_candidate", which does NOT match the
            # "agent_consensus_r{n}" prefix that learn_calibration.py's
            # _family_from_source() looks for. Every debate-round sample was
            # therefore silently bucketed into "default" during offline calibration
            # fitting, and at runtime calibrate(..., family=f"gemma_debate_r{r}")
            # could never find a match either -- so Gemma's actual final debate
            # view was calibrated with the untrained "default" params (~0.96x,
            # near-zero shrinkage) instead of a properly fitted per-round factor.
            "source": f"agent_consensus_r{r}",
            "round_job": ROUND_JOBS.get(r, ("debate", ""))[0],
            "tiebreak": resolution,
        }
        state.moderator_syntheses[r][ticker] = consensus_rec  # kept for backward-compatible UI/state key

        swayed = []
        for a in config.SPECIALISTS:
            for inf in rebuttals.get(a, {}).get("influenced_by", []) or []:
                if inf and inf != a and inf not in swayed:
                    swayed.append(inf)
        changed = [a for a in config.SPECIALISTS if rebuttals.get(a, {}).get("concessions")]

        # Keep a copy of the last *committed* view. The candidate is installed
        # temporarily so the existing voter can evaluate it, but it must not
        # become the next round's prior unless the vote actually passes.
        prior_view = dict(state.working_views[ticker])
        prior_q = _safe(prior_view.get("q"))
        prior_conf = _safe(prior_view.get("confidence"), 0.3)

        raw_q = _safe(consensus_rec.get("recommended_q"))
        raw_conf = _safe(consensus_rec.get("recommended_confidence"), 0.5)
        cal = (ctx.calibrator or calibration.load_view_calibrator()).calibrate(
            raw_q,
            raw_conf,
            family=f"{ctx.model_family}_debate_r{r}",
        )
        unanchored_calibrated_q = _safe(cal.q)
        effective_alpha, learned_reliability = _reliability_capped_alpha(
            cal, config.DEBATE_ACCEPTED_UPDATE_ALPHA
        )
        candidate_q = _anchored_q(
            prior_q,
            unanchored_calibrated_q,
            effective_alpha,
        )
        direction, q = direction_from_q(candidate_q)
        p_row = bl_core.p_row_absolute(ticker, state.universe_order)
        unanchored_calibrated_conf = _safe(cal.confidence, prior_conf)
        new_conf = float(np.clip(
            prior_conf + effective_alpha * (unanchored_calibrated_conf - prior_conf),
            config.MIN_VIEW_CONFIDENCE,
            config.MAX_VIEW_CONFIDENCE,
        ))
        omega = float(bl_core.idzorek_omega(np.array([p_row]), state.sigma, np.array([new_conf]))[0, 0])
        state.working_views[ticker].update(
            {
                "direction": direction,
                "q": round(q, 6),
                "raw_q": round(cal.raw_q, 6),
                "confidence": round(new_conf, 4),
                "raw_confidence": round(cal.raw_confidence, 4),
                "calibration": cal.as_metadata(),
                "anchor_prior_q": round(prior_q, 6),
                "anchor_prior_confidence": round(prior_conf, 4),
                "unanchored_calibrated_q": round(unanchored_calibrated_q, 6),
                "unanchored_calibrated_confidence": round(unanchored_calibrated_conf, 4),
                "debate_update_alpha": round(effective_alpha, 6),
                "debate_update_alpha_cap": config.DEBATE_ACCEPTED_UPDATE_ALPHA,
                "debate_source_reliability": round(learned_reliability, 6),
                "omega": round(omega, 10),
                "revision_note": consensus_rec.get("reasoning_summary", ""),
                "who_swayed_the_decision": swayed,
                "agents_that_changed_position": changed,
                "unresolved_objections": resolution.get("unresolved_objections", []),
                "source": f"agent_consensus_r{r}",
                "resolved": True,
            }
        )
        ctx.emitter.emit(
            "agent_consensus_candidate", phase="debate", round=r, ticker=ticker,
            payload={**consensus_rec, "recommended_direction": direction, "recommended_q": round(q, 6),
                     "raw_recommended_q": round(cal.raw_q, 6), "recommended_confidence": round(new_conf, 4),
                     "raw_recommended_confidence": round(cal.raw_confidence, 4), "calibration": cal.as_metadata(),
                     "who_swayed_the_decision": swayed, "agents_that_changed_position": changed,
                     "omega": round(omega, 10),
                     # aliases matching learn_calibration.py's expected field names
                     "raw_q": round(cal.raw_q, 6), "raw_confidence": round(cal.raw_confidence, 4),
                     "anchor_prior_q": round(prior_q, 6),
                     "anchor_prior_confidence": round(prior_conf, 4),
                     "unanchored_calibrated_q": round(unanchored_calibrated_q, 6),
                     "unanchored_calibrated_confidence": round(unanchored_calibrated_conf, 4),
                     "debate_update_alpha": round(effective_alpha, 6),
                     "debate_update_alpha_cap": config.DEBATE_ACCEPTED_UPDATE_ALPHA,
                     "debate_source_reliability": round(learned_reliability, 6)},
        )
        ctx.emitter.emit(
            "agent_message", phase="debate", round=r, ticker=ticker, agent="agent_consensus",
            payload={
                "kind": "consensus_candidate", "responding_to": None, "stance": None,
                "direction": direction, "q": round(q, 6),
                "message": (consensus_rec.get("reasoning_summary", "") +
                            f" Candidate: {direction} q={q:+.3f}" +
                            (f"; influenced by {', '.join(swayed)}." if swayed else ".")),
            },
        )


        # ---- vote (consume REVISED positions: spec 3a) ---------------- #
        accepts, tally, eligible_voters = _vote_on_view(state, ctx, ticker, r)
        resolved = (
            bool(resolution.get("resolved", True))
            and eligible_voters >= ctx.accept_threshold
            and accepts >= ctx.accept_threshold
        )
        state.votes[r][ticker] = tally

        dissent_explanation = _consensus_dissent_explanation(
            resolution=resolution,
            rebuttals=rebuttals,
            tally=tally,
            accepts=accepts,
            threshold=ctx.accept_threshold,
        )

        ctx.emitter.emit(
            "consensus_result", phase="debate", round=r, ticker=ticker,
            payload={
                "accepts": accepts,
                "threshold": ctx.accept_threshold,
                "eligible_voters": eligible_voters,
                "resolved": resolved,
                "tally": tally,
                "final_q": state.working_views[ticker]["q"],
                "final_direction": state.working_views[ticker]["direction"],
                "dissent_explanation": dissent_explanation,
            },
        )

        ctx.emitter.emit(
            "agent_message", phase="debate", round=r, ticker=ticker, agent="agent_consensus",
            payload={
                "kind": "dissent_explanation",
                "responding_to": None,
                "stance": None,
                "direction": state.working_views[ticker]["direction"],
                "q": state.working_views[ticker]["q"],
                "message": dissent_explanation,
            },
        )
        if resolved:
            state.consensus[ticker] = {
                "resolved": True, "round": r, "via": "debate_vote", "accepts": accepts,
            }
            state.contested.remove(ticker)
        else:
            # Critical: an unaccepted candidate is diagnostic output only. Do
            # not let it silently become the prior for the next debate round.
            state.working_views[ticker] = prior_view
            ctx.emitter.warning(
                f"{ticker}: unaccepted debate candidate was not committed; prior view restored",
                phase="debate",
                round=r,
                ticker=ticker,
                kind="candidate_not_committed",
                candidate_q=round(q, 6),
                restored_q=round(prior_q, 6),
                accepts=accepts,
                eligible_voters=eligible_voters,
                threshold=ctx.accept_threshold,
            )
        prev_openings = openings  # remember for the sham control's next ticker
        round_summary = {
            "ticker": ticker,
            "round": r,
            "positions_after_round": {
                a: {
                    "direction": rebuttals.get(a, {}).get("revised_direction"),
                    "q": rebuttals.get(a, {}).get("revised_q"),
                    "confidence": rebuttals.get(a, {}).get("revised_confidence"),
                    "changed_from_opening": (
                        rebuttals.get(a, {}).get("revised_direction") != 
                        openings.get(a, {}).get("direction")
                    ),
                }
                for a in config.SPECIALISTS
            },
            "facts_conceded": {
                a: rebuttals.get(a, {}).get("concessions", [])
                for a in config.SPECIALISTS
            },
            "open_objections": {
                a: rebuttals.get(a, {}).get("still_objects", [])
                for a in config.SPECIALISTS
            },
            "resolution": {
                "classification": resolution.get("classification"),
                "resolved": resolution.get("resolved"),
                "q": resolution.get("q"),
                "confidence": resolution.get("confidence"),
            },
        }
        ctx.emitter.emit("round_summary", phase="debate", round=r, ticker=ticker, 
                        payload=round_summary)


def _rebuttal_summary(data: Dict) -> str:
    parts = [
        f"Revised position: {data.get('revised_direction', '—')} "
        f"q={_safe(data.get('revised_q')):+.3f} "
        f"with confidence {_safe(data.get('revised_confidence'), 0.5):.2f}."
    ]
    influenced = data.get("influenced_by") or []
    concessions = data.get("concessions") or []
    objections = data.get("still_objects") or []
    if influenced:
        parts.append("Influenced by " + ", ".join(map(str, influenced)) + ".")
    if concessions:
        parts.append("Concedes: " + "; ".join(map(str, concessions)) + ".")
    if objections:
        parts.append("Still objects: " + "; ".join(map(str, objections)) + ".")
    if data.get("strongest_counterargument_agent"):
        parts.append(f"Strongest counterargument: {data['strongest_counterargument_agent']}.")
    return " ".join(parts)


def _vote_on_view(state: RunState, ctx: RunContext, ticker: str, r: int):
    """Vote on the current candidate using revised positions.

    Invalid rebuttals are ineligible. A matching direction alone is not enough:
    the candidate q must be within an uncertainty-aware tolerance, or the agent
    must explicitly concede and still be reasonably close.
    """
    view = state.working_views[ticker]
    q = _safe(view["q"])
    direction = view["direction"]
    tally: Dict[str, Dict] = {}
    accepts = 0
    eligible_voters = 0

    for agent in config.SPECIALISTS:
        rev = state.latest_revised(ticker, agent)
        if rev is not None:
            pref_q = _safe(rev.get("q"))
            pref_dir = rev.get("direction") or _direction(pref_q / ctx.scale if ctx.scale else 0)
            confidence = float(np.clip(_safe(rev.get("confidence"), 0.5), 0.05, 0.95))
            eligible = bool(rev.get("eligible_to_vote", True)) and rev.get("participation_status") != "invalid_rebuttal"
            concessions = rev.get("concessions", []) or []
            source = f"revised_r{r}"
        else:
            sig = _safe(state.agent_outputs[agent].get("signals", {}).get(ticker, {}).get("signal"))
            pref_q = sig * ctx.scale
            pref_dir = _direction(sig)
            confidence = 0.5
            eligible = True
            concessions = []
            source = "phase1_signal"

        tolerance = config.BASE_ACCEPTANCE_TOLERANCE + 0.03 * (1.0 - confidence)
        distance = abs(pref_q - q)

        if not eligible:
            vote = "invalid"
        else:
            eligible_voters += 1
            if distance <= tolerance:
                vote = "accept"
            elif concessions and distance <= 2.0 * tolerance:
                vote = "accept"
            elif direction == pref_dir or pref_dir == "neutral":
                vote = "adjust"
            else:
                vote = "reject"

        if vote == "accept":
            accepts += 1

        tally[agent] = {
            "vote": vote,
            "eligible": eligible,
            "preferred_q": round(pref_q, 6),
            "preferred_direction": pref_dir,
            "current_q": round(q, 6),
            "current_direction": direction,
            "acceptance_tolerance": round(tolerance, 6),
            "q_distance": round(distance, 6),
            "position_source": source,
        }
        ctx.emitter.emit(
            "vote_cast", phase="debate", round=r, ticker=ticker, agent=agent,
            payload=tally[agent],
        )

    return accepts, tally, eligible_voters


# Each debate round performs a DISTINCT information operation (spec A3), not the same
# "do you still agree?" three times (that would be the martingale).
ROUND_JOBS = {
    1: ("info_transfer", "Sharing what each only knows"),
    2: ("forward_positioning", "What happens next month?"),
    3: ("classification_tiebreak", "Settling the disagreement"),
}


def _cited(state: RunState, agent: str, ticker: str):
    return state.agent_outputs.get(agent, {}).get("signals", {}).get(ticker, {}).get("cited_values", []) or []


def _agent_qs_by_side(rebuttals: Dict[str, Dict], side: str) -> List[float]:
    return [_safe(r.get("revised_q")) for r in rebuttals.values() if r.get("revised_direction") == side]


def _mean_side_q(rebuttals: Dict[str, Dict], side: str) -> float:
    vals = _agent_qs_by_side(rebuttals, side)
    return float(np.mean(vals)) if vals else 0.0


def _significant_signal(state: RunState, agent: str, ticker: str) -> bool:
    """Heuristic anti-conformity gate from traceable cited values.

    Strong statistical/factor evidence should not be erased by a majority with no
    grounded counter-evidence. Names are deliberately broad because the four agents
    cite different metric names.
    """
    for cv in _cited(state, agent, ticker):
        name = str(cv.get("name", "")).lower()
        val = _safe(cv.get("value"))
        if "t_stat" in name or name.endswith("_t"):
            if abs(val) >= 2.0:
                return True
        if "p_value" in name and 0 < val <= 0.05:
            return True
        if any(k in name for k in ("excess_ret_3m", "net_style_factor_support", "net_factor_support", "alpha", "drawdown")):
            if abs(val) >= 0.08 or val <= -0.08:
                return True
    return False


def _context_evidence_tilt(state: RunState, ticker: str) -> List[Dict]:
    tilts = []
    s = state.agent_outputs.get("sentiment", {}).get("signals", {}).get(ticker, {})
    if s.get("cited_values"):
        sig = _safe(s.get("signal"))
        if abs(sig) > 0.2:
            tilts.append({
                "agent": "sentiment",
                "direction": _direction(sig),
                "reason": s.get("rationale", "")[:220],
                "cited_values": s.get("cited_values", [])[:4],
                "stock_specific": True,
            })
    e = state.agent_outputs.get("economist", {}).get("signals", {}).get(ticker, {})
    if e.get("cited_values"):
        sig = _safe(e.get("signal"))
        if abs(sig) > 0.2:
            tilts.append({
                "agent": "economist",
                "direction": _direction(sig),
                "reason": e.get("rationale", "")[:220],
                "cited_values": e.get("cited_values", [])[:4],
                "stock_specific": False,
                "sector": (e.get("extra") or {}).get("sector"),
            })
    return tilts

def _extract_agent_bl_views(
    state: RunState, ctx: RunContext, ticker: str,
    openings: Dict[str, Dict], rebuttals: Dict[str, Dict]
) -> Dict:
    """Round-3 aggregation using valid revised positions, not frozen openings."""
    agent_views = {}
    weighted = []

    for agent in config.SPECIALISTS:
        revised = rebuttals.get(agent) or {}
        valid = revised.get("participation_status", "valid") != "invalid_rebuttal"
        q_a = _safe(revised.get("revised_q", openings.get(agent, {}).get("q")))
        c_a = float(np.clip(_safe(revised.get("revised_confidence", 0.3)), 0.05, 0.80))
        d_a = revised.get("revised_direction") or openings.get(agent, {}).get("direction", "neutral")
        multiplier = _safe(revised.get("adjudication_weight_multiplier"), 1.0 if valid else config.INVALID_REBUTTAL_WEIGHT)
        weight = c_a * multiplier

        agent_views[agent] = {
            "q": q_a, "confidence": c_a, "direction": d_a,
            "valid_rebuttal": valid, "weight": weight,
        }
        if valid and weight > 0:
            weighted.append((agent, q_a, c_a, weight))

    if not weighted:
        return {
            "classification": "no_valid_rebuttals",
            "resolved": False,
            "direction": state.working_views[ticker].get("direction", "neutral"),
            "q": _safe(state.working_views[ticker].get("q")),
            "confidence": 0.10,
            "reason": "No valid revised positions survived rebuttal validation.",
            "agent_views": agent_views,
        }

    qs = [x[1] for x in weighted]
    weights = [x[3] for x in weighted]
    q_agg = float(np.average(qs, weights=weights))
    q_std = float(np.std(qs)) if len(qs) > 1 else 0.0
    conf_agg = float(np.clip(
        np.average([x[2] for x in weighted], weights=weights)
        * max(0.25, 1.0 - q_std / 0.12),
        0.10, 0.65,
    ))
    direction, q_final = direction_from_q(q_agg)

    return {
        "classification": "agent_bl_views",
        "resolved": True,
        "direction": direction,
        "q": q_final,
        "confidence": conf_agg,
        "reason": (
            "Round 3 valid revised views aggregated: "
            + ", ".join(
                f"{a}=(q={v['q']:+.3f}, conf={v['confidence']:.2f}, valid={v['valid_rebuttal']})"
                for a, v in agent_views.items()
            )
        ),
        "agent_views": agent_views,
        "q_disagreement_std": round(q_std, 4),
    }


def _guard_invalid_neutral_resolution(
    *, resolution: Dict, rebuttals: Dict[str, Dict], ticker: str
) -> Dict:
    """Block fake neutrality caused only by cancellation/disagreement.

    Neutral is allowed when enough valid agents genuinely support a near-zero q.
    Otherwise produce a low-confidence directional adjudication from valid views.
    """
    resolution = dict(resolution or {})
    q_candidate = _safe(resolution.get("q"))
    direction_candidate, _ = direction_from_q(q_candidate)
    if direction_candidate != "neutral":
        return resolution

    valid = []
    for agent, stance in rebuttals.items():
        if stance.get("participation_status", "valid") == "invalid_rebuttal":
            continue
        q = _safe(stance.get("revised_q"))
        conf = float(np.clip(_safe(stance.get("revised_confidence"), 0.3), 0.05, 0.80))
        multiplier = _safe(stance.get("adjudication_weight_multiplier"), 1.0)
        valid.append((agent, q, conf * multiplier))

    neutral_supporters = [
        agent for agent, q, _ in valid
        if abs(q) <= config.NEUTRAL_EVIDENCE_BAND
    ]
    if len(neutral_supporters) >= config.NEUTRAL_SUPPORT_MIN:
        resolution["neutral_supporters"] = neutral_supporters
        return resolution

    if not valid:
        resolution.update({
            "resolved": False,
            "classification": "invalid_neutral_no_valid_views",
            "confidence": 0.10,
            "reason": f"{ticker}: neutral candidate blocked; no valid rebuttals survived.",
        })
        return resolution

    weighted_q = float(np.average(
        [q for _, q, _ in valid],
        weights=[max(w, 1e-6) for _, _, w in valid],
    ))

    if abs(weighted_q) <= config.NEUTRAL_EVIDENCE_BAND:
        long_strength = sum(max(q, 0.0) * w for _, q, w in valid)
        short_strength = sum(max(-q, 0.0) * w for _, q, w in valid)
        chosen_side = "long" if long_strength >= short_strength else "short"
        side = [(q, w) for _, q, w in valid if (q > 0 if chosen_side == "long" else q < 0)]
        if side:
            weighted_q = float(np.average(
                [q for q, _ in side],
                weights=[max(w, 1e-6) for _, w in side],
            )) * 0.5
        else:
            strongest = max(valid, key=lambda x: abs(x[1] * x[2]))
            weighted_q = strongest[1] * 0.5

    direction, q_final = direction_from_q(weighted_q)
    if direction == "neutral":
        q_final = config.NEUTRAL_EVIDENCE_BAND * (1.01 if weighted_q >= 0 else -1.01)
        direction, q_final = direction_from_q(q_final)

    resolution.update({
        "classification": "invalid_neutral_blocked_directional_adjudication",
        "resolved": True,
        "direction": direction,
        "q": q_final,
        "confidence": min(_safe(resolution.get("confidence"), 0.2), 0.30),
        "neutral_supporters": neutral_supporters,
        "reason": (
            f"{ticker}: neutral candidate blocked because only {len(neutral_supporters)} "
            f"valid agent(s) supported near-zero q; low-confidence {direction} "
            f"adjudication q={q_final:+.4f} emitted instead."
        ),
    })
    return resolution


def _evidence_gated_resolution(
    state: RunState,
    ctx: RunContext,
    ticker: str,
    openings: Dict[str, Dict],
    rebuttals: Dict[str, Dict],
    r: int,
) -> Dict:
    """Evidence-gated resolver.

    This avoids pretending that a weak average is a real tiebreak.

    Resolution classes:
      - factual: conflicting traceable facts; do not average.
      - aligned: agents no longer straddle directions.
      - horizon_weighting: context evidence breaks the tie only when the
        conjunction rule is satisfied.
      - protected_dissent_weakened: context evidence is valid, but a lone
        significant dissenter prevents a full-strength override.
      - no_context_weighted_average: no context tiebreak exists, so only a
        small low-conviction average is emitted.
      - irreducible: no valid evidence-gated resolution.
    """
    eligible_rebuttals = {
        a: (rebuttals.get(a) or {})
        for a in config.SPECIALISTS
        if (rebuttals.get(a) or {}).get("participation_status", "valid")
        != "invalid_rebuttal"
        and bool((rebuttals.get(a) or {}).get("eligible_to_vote", True))
    }
    dirs = {
        a: d.get("revised_direction")
        for a, d in eligible_rebuttals.items()
    }

    bull = [a for a, d in dirs.items() if d == "long"]
    bear = [a for a, d in dirs.items() if d == "short"]
    engaged = [a for a, d in eligible_rebuttals.items() if d.get("engaged", True)]

    # Nobody has a directional view.
    if not bull and not bear:
        return {
            "classification": "aligned",
            "resolved": True,
            "direction": "neutral",
            "q": 0.0,
            "confidence": 0.25,
            "reason": "All revised positions are neutral; no directional BL view survives.",
            "engaged_agents": engaged,
        }

    # Only one side remains after rebuttals.
    if not bull or not bear:
        side = "long" if bull else "short"
        side_agents = bull or bear
        q = _mean_side_q(rebuttals, side)
        conf = float(
            np.clip(
                np.mean(
                    [
                        _safe(rebuttals[a].get("revised_confidence"), 0.3)
                        for a in side_agents
                    ]
                ),
                0.1,
                0.9,
            )
        )

        return {
            "classification": "aligned",
            "resolved": True,
            "direction": side,
            "q": q,
            "confidence": conf,
            "reason": f"Agents no longer straddle directions; resolved to {side} from engaged revised positions.",
            "engaged_agents": engaged,
        }

    disputants = sorted(bull + bear)

    # Factual conflict: same cited metric, materially different numbers.
    # Do not average this away.
    for a in bull:
        for b in bear:
            av = {
                str(cv.get("name")): _safe(cv.get("value"))
                for cv in _cited(state, a, ticker)
            }

            for cv in _cited(state, b, ticker):
                nm = str(cv.get("name"))
                if nm in av and abs(av[nm] - _safe(cv.get("value"))) > 0.05 * max(abs(av[nm]), 1e-9):
                    return {
                        "classification": "factual",
                        "resolved": False,
                        "direction": "neutral",
                        "q": 0.0,
                        "confidence": 0.15,
                        "metric": nm,
                        "disputants": [a, b],
                        "reason": f"Factual conflict on {nm}; do not average until the source number is reconciled.",
                        "engaged_agents": engaged,
                    }

    # Protect a lone significant dissenter.
    lone_side = None
    lone_agent = None

    if len(bull) == 1 and len(bear) >= 2:
        lone_side, lone_agent = "long", bull[0]
    elif len(bear) == 1 and len(bull) >= 2:
        lone_side, lone_agent = "short", bear[0]

    protected = bool(lone_agent and _significant_signal(state, lone_agent, ticker))

    # Context evidence can break a tie only if it is one-sided and grounded.
    tilts = _context_evidence_tilt(state, ticker)
    usable = [t for t in tilts if t.get("direction") in ("long", "short")]

    def _agent_has_grounded_side_fact(agent: str) -> bool:
        return bool(
            (openings.get(agent, {}).get("evidence") or [])
            or _cited(state, agent, ticker)
            or _significant_signal(state, agent, ticker)
        )

    if usable:
        dirs_ctx = {t["direction"] for t in usable}

        if len(dirs_ctx) > 1:
            return {
                "classification": "irreducible",
                "resolved": False,
                "direction": "neutral",
                "q": 0.0,
                "confidence": 0.15,
                "reason": "Context agents conflict; no evidence-gated tiebreak.",
                "context_evidence": usable,
                "disputants": disputants,
                "engaged_agents": engaged,
            }

        side = next(iter(dirs_ctx))
        side_agents = bull if side == "long" else bear
        context_agent = next(
            (t["agent"] for t in usable if t["direction"] == side),
            "context",
        )

        # Conjunction rule:
        # Context evidence can break the tie only when it is stock-specific
        # sentiment OR when the favored side has grounded ticker/factor evidence.
        sentiment_stock_specific = any(
            t["agent"] == "sentiment" and t.get("stock_specific")
            for t in usable
        )

        side_has_grounded_fact = any(
            _agent_has_grounded_side_fact(a)
            for a in side_agents
        )

        conjunction_satisfied = sentiment_stock_specific or side_has_grounded_fact

        # Main bug fix:
        # If conjunction is NOT satisfied, do NOT call it horizon weighting.
        if not conjunction_satisfied:
            return {
                "classification": "irreducible",
                "resolved": False,
                "direction": "neutral",
                "q": 0.0,
                "confidence": 0.15,
                "reason": (
                    f"Context evidence ({context_agent}) tilts {side}, but the conjunction "
                    "rule is not satisfied by grounded ticker/factor evidence on that side."
                ),
                "context_evidence": usable,
                "disputants": disputants,
                "engaged_agents": engaged,
            }

        # If there is a protected dissenter on the opposite side, weaken the context view.
        if protected and side != lone_side:
            q_context = _mean_side_q(rebuttals, side)
            q_weakened = float(np.clip(q_context * 0.25, -0.03, 0.03))
            direction_w, q_weakened = direction_from_q(q_weakened)

            return {
                "classification": "protected_dissent_weakened",
                "resolved": True,
                "direction": direction_w,
                "q": q_weakened,
                "confidence": 0.15,
                "dissent_protected": True,
                "protected_dissenter": lone_agent,
                "tiebreaker": context_agent,
                "reason": (
                    f"{lone_agent} holds statistically significant {lone_side} evidence; "
                    f"context tilts {side} with conjunction satisfied, but cannot override fully. "
                    f"Weakened {direction_w} view q={q_weakened:+.4f} emitted at low conviction."
                ),
                "context_evidence": usable,
                "disputants": disputants,
                "engaged_agents": engaged,
            }

        # Valid context tiebreak.
        q_ctx = _mean_side_q(rebuttals, side)
        conf_ctx = float(
            np.clip(
                np.mean(
                    [
                        _safe(rebuttals[a].get("revised_confidence"), 0.3)
                        for a in side_agents
                    ]
                ),
                0.25,
                0.70,
            )
        )

        return {
            "classification": "horizon_weighting",
            "resolved": True,
            "direction": side,
            "q": round(q_ctx, 6),
            "confidence": conf_ctx,
            "dissent_protected": False,
            "tiebreaker": context_agent,
            "reason": (
                f"Context evidence ({context_agent}) tilts {side} and the conjunction "
                "rule is satisfied by grounded ticker/factor evidence."
            ),
            "context_evidence": usable,
            "disputants": disputants,
            "engaged_agents": engaged,
        }

    # No usable context exists.
    # This is allowed, but it must be labeled as a weak average, not as a tiebreak.
    q_agg_fallback, _ = _aggregate_positions(eligible_rebuttals)
    q_agg_fallback = float(np.clip(q_agg_fallback, -0.03, 0.03))
    dir_fallback, q_agg_fallback = direction_from_q(q_agg_fallback)

    return {
        "classification": "no_context_weighted_average",
        "resolved": True,
        "direction": dir_fallback,
        "q": q_agg_fallback,
        "confidence": 0.12,
        "dissent_protected": protected,
        "protected_dissenter": lone_agent if protected else None,
        "reason": "No grounded context tiebreaker; confidence-weighted average of agent positions emitted at low conviction.",
        "disputants": disputants,
        "engaged_agents": engaged,
    }


def _aggregate_positions(rebuttals: Dict[str, Dict]):
    """Confidence-weighted aggregate of the agents' revised positions (thin moderator,
    spec A). Returns (q, confidence). The moderator authors no view of its own."""
    items = [
        r for r in rebuttals.values()
        if r.get("participation_status", "valid") != "invalid_rebuttal"
        and bool(r.get("eligible_to_vote", True))
    ]
    if not items:
        return 0.0, 0.3
    weights = [
        _safe(r.get("revised_confidence"), 0.3)
        * _safe(r.get("adjudication_weight_multiplier"), 1.0)
        for r in items
    ]
    wsum = sum(weights) or 1.0
    q = sum(_safe(r.get("revised_q")) * w for r, w in zip(items, weights)) / wsum
    conf = float(np.clip(np.average(
        [_safe(r.get("revised_confidence"), 0.3) for r in items],
        weights=[max(w, 1e-6) for w in weights],
    ), 0.1, 0.95))
    return q, conf

def _consensus_dissent_explanation(
    resolution: Dict,
    rebuttals: Dict[str, Dict],
    tally: Dict[str, Dict],
    accepts: int,
    threshold: int,
) -> str:
    rejectors = [
        agent for agent, vote in (tally or {}).items()
        if vote.get("vote") != "accept"
    ]

    if not rejectors:
        return "All agents accepted the consensus candidate."

    parts = [
        f"Consensus reached with dissent from {', '.join(rejectors)}."
        if accepts >= threshold
        else f"No consensus: dissent remained from {', '.join(rejectors)}."
    ]

    for agent in rejectors:
        rb = rebuttals.get(agent, {}) or {}
        pref_q = (tally.get(agent, {}) or {}).get("preferred_q")
        pref_dir = (tally.get(agent, {}) or {}).get("preferred_direction")
        objections = rb.get("still_objects") or []
        concessions = rb.get("concessions") or []

        reason_bits = []
        if objections:
            reason_bits.append("unresolved objection: " + "; ".join(map(str, objections[:2])))
        if concessions:
            reason_bits.append("conceded: " + "; ".join(map(str, concessions[:1])))

        if not reason_bits:
            reason_bits.append(
                "dissent did not provide enough resolved next-30-day evidence to control q"
            )

        parts.append(
            f"{agent} preferred {pref_dir} q={pref_q}; "
            + " | ".join(reason_bits)
        )

    final_reason = resolution.get("reason")
    if final_reason:
        parts.append("Final candidate rationale: " + str(final_reason))

    return " ".join(parts)


def _guard_sycophancy(data: Dict, opening: Dict, agent: str, ctx: RunContext, ticker: str, r: int) -> Dict:
    """Anti-conformity (spec): a DIRECTION change requires a new cited fact / corrected
    number -- never 'the majority disagrees with me'. An unevidenced flip is reverted to
    the agent's opening position and flagged as sycophancy (head-count is not evidence)."""
    data = dict(data)
    opening_dir = opening.get("direction")
    revised_dir = data.get("revised_direction")
    if opening_dir and revised_dir and revised_dir != opening_dir:
        evidenced = bool(data.get("new_fact_or_correction")) and any(
            rep.get("cited_number") is not None and (rep.get("my_response") or rep.get("message"))
            for rep in (data.get("replies") or [])
        )
        if not evidenced:
            data["revised_direction"] = opening_dir
            data["revised_q"] = _safe(opening.get("q"))
            data["flagged_sycophancy"] = True
            ctx.emitter.warning(
                f"{agent}: unevidenced direction flip {opening_dir}->{revised_dir} on {ticker} "
                f"reverted (sycophancy: head-count is not evidence)",
                phase="debate", round=r, ticker=ticker, agent=agent, kind="sycophancy",
            )
    return data


def _seed(ticker: str, r: int) -> int:
    h = r
    for ch in ticker:
        h = (h * 131 + ord(ch)) & 0x7FFFFFFF
    return h


def _derange(items: list, seed: int) -> list:
    """A deterministic derangement (no element keeps its index) for the sham control."""
    n = len(items)
    if n < 2:
        return list(items)
    perm = list(range(n))
    # rotate by a seed-dependent non-zero offset -> guaranteed derangement
    off = 1 + (seed % (n - 1))
    perm = [(i + off) % n for i in range(n)]
    return [items[perm[i]] for i in range(n)]


def _opening_text(o: Dict) -> str:
    return (
        o.get("position_rationale")
        or o.get("message")
        or o.get("objection_to_current_view")
        or ""
    )


def _peer_openings_block(agent: str, openings: Dict[str, Dict], limit: int = 200) -> Dict[str, str]:
    """Dedicated, lightly truncated peer openings for rebuttal quoting.

    This is deliberately separate from compact_opening/whiteboard compression so Gemma
    can copy `quoted_claim` from text it actually sees.
    """
    out = {}
    for peer, opening in (openings or {}).items():
        if peer == agent:
            continue
        txt = _opening_text(opening)
        out[peer] = txt[:limit] + ("..." if len(txt) > limit else "")
    return out


def _format_peer_openings(peer_openings: Dict[str, str]) -> str:
    if not peer_openings:
        return "No peer openings available."
    return "\n".join(f'{peer.upper()} said: "{text}"' for peer, text in peer_openings.items())


def _rebuttal_whiteboard(state: RunState, agent: str, ticker: str, scale: float,
                         peer_view: Dict[str, Dict], peers: List[Dict]) -> Dict:
    """Ticker-scoped payload used both for Gemma and number tracing.

    It includes all agents' current ticker evidence plus peer openings; validation
    traces cited_number against this whole structure rather than only the quoted claim.
    """
    return {
        "ticker": ticker,
        "your_agent": agent,
        "your_agent_context": _agent_context(state, agent, ticker, scale),
        "all_agent_contexts": {a: _agent_context(state, a, ticker, scale) for a in config.SPECIALISTS},
        "peer_openings": _peer_openings_block(agent, peer_view),
        "peers_to_address": peers,
        "current_bl_view": payloads.compact_view(state.working_views.get(ticker, {})),
    }


def _opening_best_evidence(o: Dict) -> Dict:
    ev = o.get("evidence") or []
    for e in ev:
        if e.get("value") is not None:
            return {"name": e.get("name"), "value": _safe(e.get("value")), "note": e.get("note", "")}
    # fall back to q so the peer still has a numeric claim to answer
    return {"name": "q", "value": _safe(o.get("q")), "note": "opening preferred annualized q"}


def _disagreeing_peers(agent: str, openings: Dict[str, Dict], k: int = 2):
    """Return 1-2 peers whose opening direction disagrees most with this agent.

    If nobody disagrees directionally, fall back to the largest q-distance peer so the
    model still emits a non-empty reply, but validation will not demand one reply per
    peer. Each claim is copied from the opening statement text that also appears in
    `peer_openings`, making quoted_claim feasible for smaller Gemma models.
    """
    me = openings.get(agent, {})
    my_q = _safe(me.get("q"))
    my_dir = me.get("direction")
    others = {a: o for a, o in openings.items() if a != agent}
    disagree = {a: o for a, o in others.items() if o.get("direction") != my_dir}
    candidates = disagree or others
    ranked = sorted(candidates, key=lambda a: abs(_safe(candidates[a].get("q")) - my_q), reverse=True)
    out = []
    for a in ranked[:k]:
        o = candidates[a]
        ev = _opening_best_evidence(o)
        claim = _opening_text(o)[:200]
        out.append({
            "agent": a,
            "direction": o.get("direction"),
            "q": o.get("q"),
            "claim": claim,
            "quoted_claim": claim[:120],
            "cited_metric": ev.get("name"),
            "cited_number": ev.get("value"),
        })
    return out


def _validated_rebuttal_or_freeze(state: RunState, ctx: RunContext, agent: str, ticker: str, r: int,
                                  issue: Dict, openings: Dict, peer_view: Dict, peers: List[Dict],
                                  res, scale: float, rebuttal_whiteboard: Dict):
    """Validate a rebuttal, repair it up to N times, then mark it invalid.

    A failed rebuttal is not a successful frozen opinion: it becomes ineligible
    to vote and receives sharply reduced adjudication influence.
    """
    def clean(d):
        d = _guard_self_counterargument(d, agent, ctx, ticker, r)
        for rep in d.get("replies") or []:
            if not rep.get("message") and rep.get("my_response"):
                rep["message"] = rep.get("my_response")
        return d

    current_res = res
    data = clean(current_res.data)
    verdict = validate_rebuttal(
        data, agent=agent, ticker=ticker, opening=openings.get(agent, {}),
        peers_to_address=peers, package=state.bundle.packages.get(agent, {}),
        whiteboard_payload=rebuttal_whiteboard,
    )

    attempts = 1
    while (
        not verdict["valid"]
        and not current_res.used_mock
        and attempts <= config.REBUTTAL_MAX_RETRIES
    ):
        current_res = structured_call(
            role=agent,
            system_prompt=(
                rebuttal_statement_prompt(
                    agent, r,
                    _format_peer_openings(_peer_openings_block(agent, peer_view)),
                    max_rounds=ctx.max_rounds,
                )
                + "\n\nVALIDATION FAILURE: " + "; ".join(verdict["issues"])
                + "\nReturn corrected JSON with non-empty replies. "
                + "A direction change requires a cited_number traced to the whiteboard/input. "
                + "A q-only change may instead explicitly identify a concession or reweighting "
                + "of already-traced evidence."
            ),
            user_payload={
                "round": r,
                "issue": issue,
                "your_agent_context": _agent_context(state, agent, ticker, scale),
                "peer_openings": _peer_openings_block(agent, peer_view),
                "peers_to_address": peers,
                "opening_statements_from_all_agents": {
                    a: payloads.compact_opening(o) for a, o in peer_view.items()
                },
                "whiteboard_for_number_tracing": rebuttal_whiteboard,
                "previous_validation_issues": verdict["issues"],
            },
            schema_model=AgentRebuttalStatement,
            mock_fn=lambda a=agent: mock.mock_rebuttal_statement(
                a, ticker, openings[a], peer_view, scale,
                round_no=r, working_q=_safe(state.working_views[ticker]["q"]),
            ),
            strict=ctx.strict,
            model_family=ctx.model_family,
            model_name=ctx.model_name,
            temperature=config.temperature_for_call(
                agent, phase="debate", debate_round=r, max_rounds=ctx.max_rounds
            ),
        )
        attempts += 1
        data = clean(current_res.data)
        verdict = validate_rebuttal(
            data, agent=agent, ticker=ticker, opening=openings.get(agent, {}),
            peers_to_address=peers, package=state.bundle.packages.get(agent, {}),
            whiteboard_payload=rebuttal_whiteboard,
        )

    if not verdict["valid"]:
        data = _freeze_rebuttal(agent, ticker, openings.get(agent, {}), verdict["issues"])
        data["rebuttal_validation"] = {**verdict, "attempts": attempts}
        ctx.emitter.warning(
            f"{agent}: invalid rebuttal on {ticker}; excluded from voting",
            phase="debate", round=r, ticker=ticker, agent=agent,
            kind="invalid_rebuttal_excluded", issues=verdict["issues"], attempts=attempts,
        )
    else:
        data["engaged"] = True
        data["participation_status"] = "valid"
        data["eligible_to_vote"] = True
        data["adjudication_weight_multiplier"] = 1.0
        data = _guard_sycophancy(data, openings.get(agent, {}), agent, ctx, ticker, r)
        data["rebuttal_validation"] = {**verdict, "attempts": attempts}

    return data, current_res

def _freeze_rebuttal(agent: str, ticker: str, opening: Dict, issues: List[str]) -> Dict:
    """Preserve the last known stance for audit only; do not count it as debate."""
    return {
        "agent_name": agent,
        "ticker": ticker,
        "revised_direction": opening.get("direction", "neutral"),
        "revised_q": _safe(opening.get("q")),
        "revised_confidence": min(_safe(opening.get("confidence"), 0.3), 0.35),
        "replies": [],
        "responded_to": "",
        "concessions": [],
        "still_objects": ["invalid rebuttal; excluded from consensus voting"],
        "influenced_by": [],
        "strongest_counterargument_agent": None,
        "engaged": False,
        "new_fact_or_correction": False,
        "participation_status": "invalid_rebuttal",
        "eligible_to_vote": False,
        "adjudication_weight_multiplier": config.INVALID_REBUTTAL_WEIGHT,
        "freeze_reason": "; ".join(issues),
    }


def _guard_self_counterargument(data: Dict, agent: str, ctx: RunContext, ticker: str, r: int) -> Dict:
    """An agent may never cite ITSELF as a peer/counterargument (spec 3g + A4).
    Applies to strongest_counterargument_agent, responded_to, influenced_by AND each
    reply's responding_to. The structured_call already re-prompted once on schema
    failure; this drops the self-reference and warns."""
    data = dict(data)
    if data.get("strongest_counterargument_agent") == agent:
        data["strongest_counterargument_agent"] = None
        ctx.emitter.warning(f"{agent}: named itself as strongest counterargument; dropped",
                             phase="debate", round=r, ticker=ticker, agent=agent)
    if data.get("responded_to") == agent:
        data["responded_to"] = ""
    if agent in (data.get("influenced_by") or []):
        data["influenced_by"] = [x for x in data["influenced_by"] if x != agent]
    replies = data.get("replies") or []
    clean = [rep for rep in replies if rep.get("responding_to") and rep.get("responding_to") != agent]
    if len(clean) != len(replies):
        ctx.emitter.warning(f"{agent}: dropped self-addressed reply", phase="debate",
                            round=r, ticker=ticker, agent=agent)
    data["replies"] = clean
    return data

def force_consensus_lowconf(state: RunState, ctx: RunContext) -> None:
    """Resolve remaining tickers after max rounds.

    The end-only moderator is called only when eligible revised positions still
    contain both long and short. Otherwise the last committed view is retained.
    Any moderator update is capped by its learned source reliability.
    """
    for ticker in list(state.contested):
        last_round = state.debate_round
        openings = state.opening_statements.get(last_round, {}).get(ticker, {})
        rebuttals = state.rebuttals.get(last_round, {}).get(ticker, {})
        agent_positions = {
            a: state.latest_revised(ticker, a) for a in config.SPECIALISTS
        }
        eligible_directions = _eligible_latest_directions(state, ticker)

        # A moderator is a directional adjudicator, not a q averaging service.
        # If valid revised positions no longer straddle long and short, retain the
        # last committed (normally Phase-1) view and avoid another noisy model call.
        if not _has_long_short_conflict(eligible_directions):
            view = state.working_views[ticker]
            view["resolved"] = True
            state.consensus[ticker] = {
                "resolved": True,
                "round": last_round,
                "via": "phase1_retained_no_direction_conflict",
                "accepts": None,
            }
            ctx.emitter.emit(
                "moderator_skipped",
                phase="debate",
                round=last_round,
                ticker=ticker,
                payload={
                    "reason": "eligible revised positions do not contain both long and short",
                    "eligible_directions": eligible_directions,
                    "retained_direction": view.get("direction"),
                    "retained_q": view.get("q"),
                    "retained_confidence": view.get("confidence"),
                    "message": (
                        f"{ticker}: moderator skipped because no eligible long-short "
                        "conflict remained; last committed view retained."
                    ),
                },
            )
            continue

        res = structured_call(
            role="moderator",
            system_prompt=MODERATOR_SYNTHESIS_PROMPT,
            user_payload={
                "ticker": ticker,
                "max_rounds": ctx.max_rounds,
                "last_round": last_round,
                "opening_statements": openings,
                "rebuttals": rebuttals,
                "latest_agent_positions": agent_positions,
                "unresolved_objections": {
                    a: (rebuttals.get(a, {}) or {}).get("still_objects", [])
                    for a in config.SPECIALISTS
                },
                "q_definition": "expected annualized outperformance vs the rest of the portfolio",
            },
            schema_model=ModeratorSynthesis,
            mock_fn=lambda: mock.mock_moderator_synthesis(ticker, openings, rebuttals, ctx.scale),
            strict=ctx.strict,
            model_family=ctx.model_family,
            model_name=ctx.model_name,
            temperature=config.temperature_for_call("moderator"),
        )
        syn = res.data
        guarded = _guard_invalid_neutral_resolution(
            resolution={
                "classification": "moderator_end_only",
                "resolved": True,
                "q": _safe(syn.get("recommended_q")),
                "confidence": _safe(syn.get("recommended_confidence"), 0.2),
                "reason": syn.get("moderator_reasoning_summary", ""),
            },
            rebuttals=rebuttals,
            ticker=ticker,
        )
        prior_view = dict(state.working_views[ticker])
        prior_q = _safe(prior_view.get("q"))
        prior_conf = _safe(prior_view.get("confidence"), 0.3)

        raw_q_final = _safe(guarded.get("q"))
        raw_conf_final = float(np.clip(_safe(guarded.get("confidence"), 0.2), 0.05, 0.55))
        cal = (ctx.calibrator or calibration.load_view_calibrator()).calibrate(
            raw_q_final,
            raw_conf_final,
            family=f"{ctx.model_family}_moderator_end",
        )
        unanchored_calibrated_q = _safe(cal.q)
        effective_moderator_alpha, moderator_reliability = _reliability_capped_alpha(
            cal, config.MODERATOR_UPDATE_ALPHA
        )
        anchored_moderator_q = _anchored_q(
            prior_q,
            unanchored_calibrated_q,
            effective_moderator_alpha,
        )
        direction, q_final = direction_from_q(anchored_moderator_q)
        reliability_anchored_conf = (
            prior_conf
            + effective_moderator_alpha * (_safe(cal.confidence, prior_conf) - prior_conf)
        )
        conf_final = float(np.clip(
            min(reliability_anchored_conf, config.MODERATOR_CONFIDENCE_CAP),
            config.MIN_VIEW_CONFIDENCE,
            config.MAX_VIEW_CONFIDENCE,
        ))
        if guarded.get("classification") == "invalid_neutral_blocked_directional_adjudication":
            syn["moderator_reasoning_summary"] = guarded.get("reason", syn.get("moderator_reasoning_summary", ""))

        view = state.working_views[ticker]
        view.update({
            "direction": direction,
            "q": round(q_final, 6),
            "raw_q": round(cal.raw_q, 6),
            "confidence": round(conf_final, 4),
            "raw_confidence": round(cal.raw_confidence, 4),
            "calibration": cal.as_metadata(),
            "anchor_prior_q": round(prior_q, 6),
            "anchor_prior_confidence": round(prior_conf, 4),
            "unanchored_calibrated_q": round(unanchored_calibrated_q, 6),
            "unanchored_calibrated_confidence": round(_safe(cal.confidence, prior_conf), 4),
            "moderator_update_alpha": round(effective_moderator_alpha, 6),
            "moderator_update_alpha_cap": config.MODERATOR_UPDATE_ALPHA,
            "moderator_source_reliability": round(moderator_reliability, 6),
            "source": "moderator_end_only_no_consensus",
            "revision_note": syn.get("moderator_reasoning_summary", ""),
            "who_swayed_the_decision": syn.get("who_swayed_the_decision", []),
            "agents_that_changed_position": syn.get("agents_that_changed_position", []),
            "unresolved_objections": syn.get("unresolved_objections", []),
            "resolved": True,
        })
        p_row = bl_core.p_row_absolute(ticker, state.universe_order)
        view["omega"] = round(
            float(bl_core.idzorek_omega(
                np.array([p_row]), state.sigma, np.array([conf_final])
            )[0, 0]), 10
        )
        state.consensus[ticker] = {
            "resolved": True,
            "round": last_round,
            "via": "moderator_end_only_no_consensus",
            "accepts": None,
        }
        ctx.emitter.emit(
            "moderator_final_decision", phase="debate", ticker=ticker,
            payload={
                **syn,
                "direction": direction,
                "q": q_final,
                "confidence": conf_final,
                "omega": view["omega"],
                "anchor_prior_q": round(prior_q, 6),
                "anchor_prior_confidence": round(prior_conf, 4),
                "unanchored_calibrated_q": round(unanchored_calibrated_q, 6),
                "unanchored_calibrated_confidence": round(_safe(cal.confidence, prior_conf), 4),
                "moderator_update_alpha": round(effective_moderator_alpha, 6),
                "moderator_update_alpha_cap": config.MODERATOR_UPDATE_ALPHA,
                "moderator_source_reliability": round(moderator_reliability, 6),
                "used_mock": res.used_mock,
                "message": (
                    f"{ticker}: no agent consensus after {last_round} rounds; "
                    f"moderator proposal was reliability-anchored to {direction} "
                    f"q={q_final:+.4f} conf={conf_final:.2f} "
                    f"(alpha={effective_moderator_alpha:.3f})."
                ),
            },
        )
        ctx.emitter.emit(
            "agent_message", phase="debate", round=last_round, ticker=ticker, agent="moderator",
            payload={
                "kind": "final_no_consensus_decision",
                "responding_to": None,
                "stance": None,
                "direction": direction,
                "q": round(q_final, 6),
                "message": syn.get("moderator_reasoning_summary", ""),
            },
        )
    state.contested = []

# --------------------------------------------------------------------------- #
# Validator (section 4)
# --------------------------------------------------------------------------- #
def validate(state: RunState, ctx: RunContext) -> None:
    for agent in config.SPECIALISTS:
        result = validate_agent(
            state.agent_outputs[agent], state.bundle.packages[agent], state.universe_order
        )
        result["used_mock"] = state.used_mock.get(agent, False)
        state.validator[agent] = result
        ctx.emitter.emit(
            "validator_result", phase="validate", agent=agent,
            payload=result,
        )


# --------------------------------------------------------------------------- #
# Phase 4 -- BL posterior + constrained long-only MVO
# --------------------------------------------------------------------------- #
def finalize(state: RunState, ctx: RunContext) -> None:
    ensure_market_model(state, ctx)
    order = state.universe_order
    P_rows, Q_list, omega_list, widen_list, view_labels, view_kinds = [], [], [], [], [], []
    in_view = {t: False for t in order}

    # Shared conversion: either multi-agent model ends as calibrated stock-level
    # views; BL sees pairwise rows produced by the same function as both single controls.
    eligible_views = {}
    for t in order:
        cons = state.consensus.get(t, {})
        v = dict(state.working_views[t])
        if not cons.get("resolved", True):
            v["resolved"] = False
        eligible_views[t] = v

    pairwise_views = calibration.ticker_views_to_pairwise_bl_views(
        eligible_views,
        order,
        max_pairs=config.MAX_PAIRWISE_VIEWS,
        spread_floor=config.VIEW_SPREAD_FLOOR,
        source=f"{ctx.model_family}_debate_stock_views_to_pairwise",
    )

    for idx, pv in enumerate(pairwise_views):
        P_rows.append(pv["p_row"])
        Q_list.append(round(float(pv["q"]), 6))
        omega_list.append(float(pv["confidence"]))
        widen_list.append(1.0)
        winner, loser = pv["long_ticker"], pv["short_ticker"]
        view_labels.append(
            f"{winner} > {loser} (stock-view spread {pv['q']:+.1%}; raw {pv.get('raw_q', 0.0):+.1%})"
        )
        view_kinds.append(pv["view_type"])
        in_view[winner] = True
        in_view[loser] = True
        ctx.emitter.emit(
            "final_view",
            phase="finalize",
            agent=f"{ctx.model_family}_debate",
            payload={
                **pv,
                "view_id": f"debate_pair_{idx+1}",
                "label": f"{winner}_gt_{loser}",
                "feeds_black_litterman": True,
            },
        )

    # The adjacent-chain converter returns at most N-1 independent relative rows.
    # Low-information ties below VIEW_SPREAD_FLOOR are omitted rather than being
    # interpreted by BL as confident equal-return constraints.

    # ── Step 3: No absolute fallback by default ───────────────────────────────
    # If fewer than two valid stock views exist, skip rather than
    # forcing a one-name absolute view. This keeps final BL views cross-sectional.
    # Only spreads that clear VIEW_SPREAD_FLOOR reach BL. Validate the remaining
    # rows for structure and finiteness.
    if P_rows:
        bl_core.validate_view_rows(np.array(P_rows))
    assert all(np.isfinite(float(q)) for q in Q_list), f"a view row has non-finite Q: {Q_list}"
    assert all(float(q) >= -1e-9 for q in Q_list), f"adjacent-chain Q must be non-negative: {Q_list}"

    debate_health_warnings = []
    if not P_rows:
        debate_health_warnings.append("NO_SAME_DIRECTION_PAIRWISE_VIEWS")
        ctx.emitter.warning(
            "no same-direction pairwise BL views formed; posterior will remain near prior",
            phase="finalize", kind="NO_SAME_DIRECTION_PAIRWISE_VIEWS",
        )
    rounded_q = [round(float(q), 6) for q in Q_list if abs(float(q)) >= config.VIEW_SPREAD_FLOOR]
    dup_q = sorted({q for q in rounded_q if rounded_q.count(q) > 1})
    if dup_q:
        debate_health_warnings.append("MULTIPLE_IDENTICAL_Q_VALUES_AVERAGING_ARTIFACT")
        ctx.emitter.warning(
            f"multiple BL views share identical q values {dup_q}; possible averaging artifact",
            phase="finalize", kind="MULTIPLE_IDENTICAL_Q_VALUES_AVERAGING_ARTIFACT",
            duplicate_q=dup_q,
        )

    view_cond = None
    if P_rows:
        P = np.array(P_rows)
        Q = np.array(Q_list)
        # Idzorek Omega with per-row widen (omega_list holds confidences here)
        Omega = bl_core.idzorek_omega(P, state.sigma, np.array(omega_list), widen=np.array(widen_list))
        omega_diag = np.diag(Omega)
        post = bl_core.bl_posterior(state.pi, state.sigma, P, Q, Omega)
        # D6: watch linear dependence -- overlapping rows on the same names make
        # P^T Omega^-1 P ill-conditioned. Log the precision condition number; warn if high.
        precision = np.linalg.pinv(config.BL_TAU * state.sigma) + P.T @ np.linalg.pinv(Omega) @ P
        view_cond = float(np.linalg.cond(precision))
        if view_cond > 1e8:
            ctx.emitter.warning(
                f"ill-conditioned view precision (P^T Omega^-1 P) cond={view_cond:.1e}; "
                f"redundant/overlapping views",
                phase="finalize", condition_number=view_cond,
            )
    else:
        # no views at all -> posterior equals the equilibrium prior (everyone at prior weight)
        P = np.zeros((0, len(order)))
        Q = np.zeros(0)
        omega_diag = np.zeros(0)
        post = bl_core.BLPosterior(mu_bl=np.array(state.pi, dtype=float), cov_bl=state.sigma, pi=state.pi)
    sector_of = state.bundle.sectors
    benchmark_weights = np.array(
        [state.market_weights[t] for t in order],
        dtype=float,
    )
    opt = bl_core.optimize_long_only(
        post.mu_bl,
        state.sigma,
        order,
        sector_of,
        benchmark_weights=benchmark_weights,
    )

    weights = {t: round(float(w), 6) for t, w in zip(order, opt.weights)}
    eq_weights = dict(state.market_weights)

    final_views = {
        "source": "final_bl_view",
        "as_of_date": state.bundle.as_of_date,
        "portfolio_id": state.bundle.portfolio_id,
        "universe_order": order,
        "views": [state.working_views[t] for t in order],
        "pairwise_views": pairwise_views,
        "tau": config.BL_TAU,
    }
    state.final_views = final_views

    for t in order:
        v = state.working_views[t]
        i = order.index(t)
        ctx.emitter.emit(
            "final_view", phase="finalize", ticker=t,
            payload={
                "direction": v["direction"],
                "in_view": in_view[t],                # did this stock earn a relative view?
                "debate_q": v["q"],                   # the debate's CALIBRATED view return (for reference)
                "confidence": v["confidence"],         # calibrated confidence (for reference)
                # BUG FIX (calibration mismatch): the two fields below were missing
                # entirely. learn_calibration.py's harvester fell back to reading
                # "debate_q"/"confidence" as if they were raw, and "source" as if
                # it were the event type "final_view" -- so every sample from this
                # event (regardless of whether it came from phase1 baseline, a
                # debate round, or the moderator fallback) was silently bucketed
                # into the untrained "default" family using ALREADY-CALIBRATED
                # numbers relabeled as raw. That both (a) prevented gemma_debate_r*
                # / gemma_moderator_end from ever being learned and (b) would have
                # compounded shrinkage incorrectly on any future re-fit.
                # working_views[t] already carries the correct raw values and the
                # correct per-path source tag ("agent_phase1_consensus",
                # "agent_consensus_r{n}", or "moderator_end_only_no_consensus") --
                # just wire them through.
                "raw_q": v.get("raw_q", v["q"]),
                "raw_confidence": v.get("raw_confidence", v["confidence"]),
                "source": v.get("source"),
                "pi": round(float(state.pi[i]), 6),
                "mu_bl": round(float(post.mu_bl[i]), 6),
                "resolved": state.consensus.get(t, {}).get("resolved", True),
            },
        )

    # D5 risk readout (annualized -> monthly): portfolio vol + parametric 95% VaR/CVaR.
    w_vec = np.array([weights[t] for t in order])
    mu_p_ann = float(w_vec @ post.mu_bl)
    sigma_p_ann = opt.portfolio_vol
    risk_readout = bl_core.parametric_var_cvar(mu_p_ann / 12.0, sigma_p_ann / np.sqrt(12.0))

    state.optimized = {
        "weights": weights,
        "equilibrium_weights": eq_weights,
        "posterior_expected_returns": {t: round(float(post.mu_bl[i]), 6) for i, t in enumerate(order)},
        "equilibrium_returns": {t: round(float(state.pi[i]), 6) for i, t in enumerate(order)},
        "status": opt.status,
        "used_fallback": opt.used_fallback,
        "portfolio_vol": round(opt.portfolio_vol, 6),
        "risk_readout": risk_readout,           # spec D5: monthly VaR95 / CVaR95
        "note": opt.note,
        "risk_gate": {
            "min_weight": config.MIN_WEIGHT,
            "max_weight": config.MAX_WEIGHT,
            "max_active_deviation": config.MAX_ACTIVE_DEVIATION,
            "active_weight_budget": config.ACTIVE_WEIGHT_BUDGET,
            "sector_cap": config.SECTOR_CAP,
            "vol_cap": config.VOL_CAP,
        },
        "covariance_condition_number": round(_safe(state.cov_cond), 3),
        "view_precision_condition_number": round(view_cond, 2) if view_cond else None,
        "cap_adjusted": opt.cap_adjusted,
        "debate_health_warnings": debate_health_warnings,
        # the actual Black-Litterman matrices, post-debate, for display/audit
        "bl_inputs": {
            "order": order,
            "tau": config.BL_TAU,
            "risk_aversion": config.RISK_AVERSION,
            "P": [[round(x, 4) for x in row] for row in P.tolist()],     # K x N picking matrix
            "Q": [round(float(x), 6) for x in Q],                        # K x 1 view returns
            "omega_diag": [round(float(x), 8) for x in omega_diag],      # diag(Omega), K
            "pi": [round(float(x), 6) for x in state.pi],                # N equilibrium returns
            "mu_bl": [round(float(x), 6) for x in post.mu_bl],           # N posterior returns
            "view_labels": view_labels,                                  # K labels (ticker or "econ: F > A")
            "view_kinds": view_kinds,                                    # K pairwise rows from calibrated stock views
        },
        "pairwise_views": pairwise_views,
        "view_calibration": {
            "path": config.VIEW_CALIBRATION_PATH,
            "default_q_multiplier": config.DEFAULT_Q_MULTIPLIER,
            "default_confidence_center": config.DEFAULT_CONFIDENCE_CENTER,
        },
    }
    if opt.cap_adjusted:
        ctx.emitter.warning(
            f"per-name weight cap adjusted for feasibility: {opt.cap_adjusted['reason']}",
            phase="finalize", **opt.cap_adjusted,
        )
    if opt.used_fallback:
        ctx.emitter.warning(
            f"optimizer used fallback: {opt.note}", phase="finalize",
        )
    ctx.emitter.emit(
        "optimized_portfolio", phase="finalize",
        payload=state.optimized,
    )
