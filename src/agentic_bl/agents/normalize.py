"""Normalize per-agent Phase-1 emissions into the internal numeric form.

Agents emit categorical convictions + agent-specific fields + rationale (NO q/P);
the BL math downstream needs a numeric signal in [-1, 1] and a confidence. This
module is the single bridge: it maps the 5-way label to a value, derives a
confidence from the agent-specific flags, and keeps the raw emission + extra fields
so the UI can show exactly what each agent said. q/P/Ω still appear only later, at
the moderator's baseline.
"""

from __future__ import annotations

from typing import Dict

from agentic_bl.schemas import SIGNAL5_VALUE, STANCE_VALUE


def _clip(x, lo=0.05, hi=0.95):
    return max(lo, min(hi, x))


def _base_conf(value: float) -> float:
    return 0.30 + 0.50 * abs(value)


def _norm_common(emission: Dict, agent: str, universe) -> Dict:
    return {
        "agent_name": emission.get("agent_name", agent),
        "as_of_date": emission.get("as_of_date", ""),
        "horizon": emission.get("horizon", "1M"),
        "portfolio_id": emission.get("portfolio_id", ""),
        "summary": emission.get("summary", ""),
        "raw": emission,            # original per-agent shape (for the UI)
        "signals": {},
    }


def normalize(agent: str, emission: Dict, universe, sector_map: Dict[str, str]) -> Dict:
    if agent == "quant":
        return _norm_quant(emission, universe)
    if agent == "fundamental":
        return _norm_fundamental(emission, universe)
    if agent == "sentiment":
        return _norm_sentiment(emission, universe)
    if agent == "economist":
        return _norm_economist(emission, universe, sector_map)
    raise ValueError(agent)


def _norm_quant(em: Dict, universe) -> Dict:
    out = _norm_common(em, "quant", universe)
    out["agent_view"] = em.get("portfolio_level_view", {})
    for t, s in (em.get("stock_signals") or {}).items():
        val = SIGNAL5_VALUE.get(s.get("signal"), 0.0)
        conf = _base_conf(val)
        align = s.get("factor_alignment")
        conf += 0.10 if align == "supporting" else (-0.10 if align == "adverse" else 0.0)
        out["signals"][t] = {
            "signal": val,
            "signal_label": s.get("signal"),
            "confidence": round(_clip(conf), 3),
            "rationale": s.get("rationale", ""),
            "cited_values": s.get("cited_values", []),
            "extra": {
                "primary_factors": s.get("primary_factors", []),
                "factor_alignment": align,
                "alpha_contribution": s.get("alpha_contribution"),
            },
        }
    return out


def _norm_fundamental(em: Dict, universe) -> Dict:
    out = _norm_common(em, "fundamental", universe)
    for t, s in (em.get("signals") or {}).items():
        val = SIGNAL5_VALUE.get(s.get("signal"), 0.0)
        conf = _base_conf(val)
        q = s.get("quality_tier")
        conf += 0.10 if q == "strong" else (-0.05 if q == "weak" else 0.0)
        out["signals"][t] = {
            "signal": val,
            "signal_label": s.get("signal"),
            "confidence": round(_clip(conf), 3),
            "rationale": s.get("rationale", ""),
            "cited_values": s.get("cited_values", []),
            "extra": {
                "valuation_call": s.get("valuation_call"),
                "quality_tier": q,
                "key_drivers": s.get("key_drivers", []),
            },
        }
    return out


def _norm_sentiment(em: Dict, universe) -> Dict:
    out = _norm_common(em, "sentiment", universe)
    for t, s in (em.get("signals") or {}).items():
        val = SIGNAL5_VALUE.get(s.get("signal"), 0.0)
        conf = _base_conf(val)
        event = bool(s.get("event_driven"))
        # event-driven catalysts can be high-conviction; opinion-only is capped (spec 6)
        conf = conf + 0.15 if event else min(conf, 0.70)
        out["signals"][t] = {
            "signal": val,
            "signal_label": s.get("signal"),
            "confidence": round(_clip(conf), 3),
            "rationale": s.get("rationale", ""),
            "cited_values": s.get("cited_values", []),
            "extra": {
                "primary_catalyst": s.get("primary_catalyst", ""),
                "event_driven": event,
            },
        }
    return out


def _as_list(x) -> list:
    if isinstance(x, list):
        return [str(v) for v in x]
    if isinstance(x, str):
        return [x]
    return []


def _norm_economist(em: Dict, universe, sector_map: Dict[str, str]) -> Dict:
    """Economist is sector-level and never names single stocks; map sector stances
    down to tickers while recovering from common Gemma list/dict mistakes."""
    out = _norm_common(em, "economist", universe)
    valid_sectors = {sector_map.get(t, "") for t in universe}
    raw_sector_views = em.get("sector_views") or {}
    # Gemma sometimes nests favored_sectors/risk_sectors inside sector_views; ignore
    # those pseudo-sectors when mapping to tickers.
    sector_views = {
        k: v for k, v in raw_sector_views.items()
        if k in valid_sectors and isinstance(v, dict)
    }

    favored = set(_as_list(em.get("favored_sectors")))
    risk = set(_as_list(em.get("risk_sectors")))
    recovered_from_sector_views = False
    if (not favored and not risk) and sector_views:
        recovered_from_sector_views = True
        for sector, view in sector_views.items():
            stance = view.get("stance", "neutral") if isinstance(view, dict) else "neutral"
            if stance in ("overweight", "favored", "long"):
                favored.add(sector)
            elif stance in ("underweight", "avoid", "short", "risk"):
                risk.add(sector)

    # Keep lists consistent with sector_views even when the model supplied stale or
    # malformed top-level lists.
    for sector, view in sector_views.items():
        stance = view.get("stance", "neutral") if isinstance(view, dict) else "neutral"
        if stance == "overweight":
            favored.add(sector); risk.discard(sector)
        elif stance == "underweight":
            risk.add(sector); favored.discard(sector)
        elif stance == "neutral":
            favored.discard(sector); risk.discard(sector)

    out["agent_view"] = {
        "rate_environment": em.get("rate_environment"),
        "risk_regime": em.get("risk_regime"),
        "cycle": em.get("cycle"),
        "regime_rationale": em.get("regime_rationale", ""),
        "sector_views": sector_views,
        "favored_sectors": sorted(favored),   # overweight
        "risk_sectors": sorted(risk),         # underweight
        "recovered_sector_lists_from_sector_views": recovered_from_sector_views,
        "ignored_sector_view_keys": sorted(k for k in raw_sector_views.keys() if k not in valid_sectors),
    }
    for t in universe:
        sector = sector_map.get(t, "")
        sv = sector_views.get(sector) or {}
        stance = sv.get("stance", "neutral")
        # If lists were the only available source, use them as a fallback stance.
        if not sv:
            if sector in favored:
                stance = "overweight"
            elif sector in risk:
                stance = "underweight"
        val = STANCE_VALUE.get(stance, 0.0)
        conf = 0.5 if stance != "neutral" else 0.3
        sector_cited = sv.get("cited_values") or em.get("cited_values") or []
        sector_rationale = sv.get("rationale") or f"{sector}: {stance} sector macro view."
        out["signals"][t] = {
            "signal": val,
            "signal_label": _stance_to_label(stance),
            "confidence": round(conf, 3),
            "rationale": em.get("regime_rationale", "") + " " + sector_rationale,
            "cited_values": sector_cited,
            "extra": {
                "sector": sector,
                "stance": stance,
                "sector_rationale": sector_rationale,
                "rate_environment": em.get("rate_environment"),
                "risk_regime": em.get("risk_regime"),
                "cycle": em.get("cycle"),
                "recovered_sector_lists_from_sector_views": recovered_from_sector_views,
            },
        }
    return out


def _stance_to_label(stance: str) -> str:
    return {"overweight": "long", "underweight": "underweight", "neutral": "neutral"}.get(stance, "neutral")
