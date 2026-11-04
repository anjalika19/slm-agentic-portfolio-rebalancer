"""Value-tracing and debate validators."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from agentic_bl.config import HALLUCINATION_TOL


_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _collect_numbers(obj: Any, out: List[float]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect_numbers(v, out)


def _traces(value: float, pool: List[float], tol: float) -> bool:
    for p in pool:
        denom = max(abs(p), abs(value), 1e-9)
        if abs(p - value) / denom <= tol:
            return True
    return False


def _has_number_text(text: str) -> bool:
    return bool(_NUM_RE.search(text or ""))


def _to_float(x):
    try:
        if x is None or isinstance(x, bool):
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def validate_agent(
    agent_output: Dict, package: Dict, universe: List[str], tol: float = HALLUCINATION_TOL
) -> Dict:
    """Return shape issues + per-ticker cited-value tracing + a hallucination rate.

    Agent-specific gates are included here so context agents cannot silently abstain
    or emit ungrounded copy-paste text.
    """
    pool: List[float] = []
    _collect_numbers(package, pool)

    issues: List[str] = []
    agent_name = agent_output.get("agent_name", "")
    signals = agent_output.get("signals", {})
    actual = set(signals.keys())
    expected = set(universe)
    if expected - actual:
        issues.append(f"missing tickers: {sorted(expected - actual)}")
    if actual - expected:
        issues.append(f"unexpected tickers: {sorted(actual - expected)}")
    for t, obj in signals.items():
        s = obj.get("signal")
        if not isinstance(s, (int, float)) or isinstance(s, bool):
            issues.append(f"{t}: signal not numeric")
        elif s < -1 or s > 1:
            issues.append(f"{t}: signal outside [-1,1]")

    # Sentiment must not abstain when relevant news or analyst fallback exists.
    if agent_name == "sentiment":
        for t in universe:
            st = (package.get("stocks") or {}).get(t, {})
            sig = signals.get(t, {})
            n_surv = int(st.get("surviving_article_count") or len(st.get("news") or []) or 0)
            n_raw = int(st.get("article_count") or st.get("raw_article_count") or 0)
            analyst_up = _to_float((st.get("analyst") or {}).get("implied_upside"))
            abstain = bool((sig.get("extra") or {}).get("abstain")) or sig.get("signal_label") == "neutral" and not sig.get("cited_values")
            if n_surv > 0 and abstain:
                issues.append(f"{t}: abstained despite {n_surv} relevant articles")
            elif n_raw > 0 and n_surv == 0 and analyst_up is not None and abstain:
                issues.append(f"{t}: abstained despite analyst fallback implied_upside={analyst_up}")

    # Economist must provide per-sector traceable evidence, not one repeated sentence.
    if agent_name == "economist":
        rationales = []
        for t in universe:
            obj = signals.get(t, {})
            cited = obj.get("cited_values", []) or []
            if not cited:
                issues.append(f"{t}: economist cited zero macro numbers")
            rat = (obj.get("extra") or {}).get("sector_rationale") or obj.get("rationale", "")
            rationales.append(rat.strip())
        unique = {r for r in rationales if r}
        if len(universe) > 1 and len(unique) <= 1:
            issues.append("economist reused identical sector rationale across tickers")

    total = 0
    hallucinated = 0
    per_ticker: Dict[str, Dict] = {}
    for t, obj in signals.items():
        cited = obj.get("cited_values", []) or []
        bad: List[Dict] = []
        for cv in cited:
            total += 1
            val = cv.get("value")
            try:
                val = float(val)
            except (TypeError, ValueError):
                hallucinated += 1
                bad.append({"name": cv.get("name"), "value": cv.get("value"), "reason": "non-numeric"})
                continue
            if not _traces(val, pool, tol):
                hallucinated += 1
                bad.append({"name": cv.get("name"), "value": val, "reason": "not_found_in_input"})
        per_ticker[t] = {"n_cited": len(cited), "n_untraced": len(bad), "untraced": bad}

    rate = (hallucinated / total) if total else 0.0
    return {
        "verdict": "accept" if not issues else "flag",
        "shape_issues": issues,
        "n_cited_total": total,
        "n_hallucinated": hallucinated,
        "hallucination_rate": round(rate, 4),
        "tolerance": tol,
        "per_ticker": per_ticker,
    }


def validate_rebuttal(
    data: Dict,
    *,
    agent: str,
    ticker: str,
    opening: Dict,
    peers_to_address: List[Dict],
    package: Dict,
    tol: float = HALLUCINATION_TOL,
    whiteboard_payload: Dict | None = None,
) -> Dict:
    """Validate a rebuttal as a real debate turn.

    The model's schema can be valid while the debate is fake (empty replies, generic
    influence labels). This validator rejects that and lets the caller re-prompt once.
    """
    issues: List[str] = []
    pool: List[float] = []
    _collect_numbers(package, pool)
    if whiteboard_payload is not None:
        _collect_numbers(whiteboard_payload, pool)

    replies = data.get("replies") or []
    agent_direction = opening.get("direction")
    disagreeing_peers = [
        p.get("agent")
        for p in (peers_to_address or [])
        if p.get("agent") and p.get("agent") != agent and p.get("direction") != agent_direction
    ]
    # If peer selection was not direction-filtered for some legacy call, fall back to
    # the provided peer list. The rule is still: at least one real reply, not one per peer.
    candidate_peers = disagreeing_peers or [
        p.get("agent") for p in (peers_to_address or []) if p.get("agent") and p.get("agent") != agent
    ]
    replied_to = {r.get("responding_to") for r in replies if r.get("responding_to")}
    if not replies:
        issues.append("replies empty — must reply to at least one disagreeing peer")
    elif candidate_peers and not replied_to.intersection(candidate_peers):
        issues.append(
            f"must reply to at least one agent who disagrees "
            f"(disagreers: {candidate_peers}, replied to: {sorted(replied_to)})"
        )

    any_trace = False
    for i, rep in enumerate(replies):
        if not rep.get("responding_to") or rep.get("responding_to") == agent:
            issues.append(f"reply[{i}]: invalid responding_to")
        if rep.get("stance") not in {"agree", "disagree", "partial"}:
            issues.append(f"reply[{i}]: invalid stance")
        if not rep.get("quoted_claim"):
            issues.append(f"reply[{i}]: missing quoted_claim")
        response = rep.get("my_response") or ""
        if not response:
            issues.append(f"reply[{i}]: missing my_response")
        if not _has_number_text(response) and rep.get("cited_number") is None:
            issues.append(f"reply[{i}]: my_response cites no number")
        num = _to_float(rep.get("cited_number"))
        if num is None:
            # allow numeric text if cited_number omitted, but still flag no trace.
            continue
        if _traces(num, pool, tol):
            any_trace = True
        else:
            issues.append(f"reply[{i}]: cited_number {num} not traced to whiteboard/input for ticker")

    old_dir = opening.get("direction")
    old_q = _to_float(opening.get("q")) or 0.0
    new_dir = data.get("revised_direction")
    new_q = _to_float(data.get("revised_q")) or 0.0

    # Debate is allowed to reweight facts that were already on the whiteboard.
    # The old rule required every material q update to claim a *new* fact, which
    # incorrectly rejected legitimate Bayesian/evidential reweighting.
    direction_changed = bool(old_dir and new_dir and old_dir != new_dir)
    q_changed_materially = abs(new_q - old_q) > 0.015
    has_new_fact = bool(data.get("new_fact_or_correction"))
    has_explicit_reweighting = bool(
        data.get("concessions")
        or data.get("influenced_by")
        or data.get("strongest_counterargument_agent")
    )

    if direction_changed and not (has_new_fact or any_trace):
        issues.append(
            "direction changed without a traceable counter-evidence number"
        )
    elif q_changed_materially and not (
        has_new_fact or any_trace or has_explicit_reweighting
    ):
        issues.append(
            "q changed without traceable evidence or an explicit reweighting/concession"
        )

    return {"valid": not issues, "issues": issues, "any_traceable_reply_number": any_trace}
