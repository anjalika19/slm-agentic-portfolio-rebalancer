"""Deterministic mock generators (spec 3f).

These are used ONLY when ``STRICT_MODEL=False`` and a real model call is
unavailable/failed. They are *not* random: they read the actual numbers from each
agent's input package so (a) signals are genuinely differentiated across tickers,
(b) the ``cited_values`` trace back to real inputs so the validator passes, and
(c) the debate mocks actually move positions so the revised-position machinery
(spec 3a) and "debate moves the portfolio" property are demonstrable with no model.

Every consumer must record ``used_mock=True`` and emit a warning when these run.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, x)))


def _f(x, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------- #
# Phase 1 specialist mocks -- emit the NEW per-agent schemas (categorical signal +
# agent-specific fields, no q/P). Read real package numbers so signals differ and
# cited_values trace. The normalizer turns these into the internal numeric form.
# --------------------------------------------------------------------------- #
def _label5(v: float) -> str:
    if v >= 0.7:
        return "strongly_long"
    if v >= 0.2:
        return "long"
    if v <= -0.7:
        return "avoid"
    if v <= -0.2:
        return "underweight"
    return "neutral"


# next-stronger same-direction tier, used to justify "X not Y" in rationales
_STRONGER = {"long": "strongly_long", "underweight": "avoid"}


def _pstr(p) -> str:
    p = _f(p, 1.0)
    return "p<0.01" if p < 0.01 else ("p<0.05" if p < 0.05 else f"p={p:.2f}")


def _pct(x) -> str:
    return f"{_f(x)*100:+.1f}%"


def _quant_rationale(t: str, label: str, sf: Dict, n: int) -> str:
    exps = sf.get("factor_model", {}).get("factor_exposures", []) or []
    by_mag = sorted(exps, key=lambda e: abs(_f(e.get("interaction_3m"))), reverse=True)
    support = [e for e in by_mag if e.get("signal_strength") == "supporting"]
    adverse = [e for e in by_mag if e.get("signal_strength") == "adverse"]
    r = sf.get("returns", {}); risk = sf.get("risk", {}); fm = sf.get("factor_model", {})
    ranks = sf.get("cross_sectional_ranks", {})
    parts = []
    dirword = {"strongly_long": "a high-conviction long", "long": "long", "neutral": "neutral",
               "underweight": "an underweight", "avoid": "an avoid"}[label]
    parts.append(f"{t} is {dirword} on price-statistical evidence.")
    if support:
        e = support[0]
        parts.append(
            f"{e['factor']} loading is {_f(e['beta']):+.2f} (t = {_f(e['t_stat']):.1f}, {_pstr(e['p_value'])}) "
            f"and its 3-month interaction is {_f(e['interaction_3m']):+.3f}, a supportive tailwind"
            + (f"; {support[1]['factor']} beta {_f(support[1]['beta']):+.2f} (t = {_f(support[1]['t_stat']):.1f}) reinforces" if len(support) > 1 else "")
            + "."
        )
    elif adverse:
        e = adverse[0]
        parts.append(
            f"{e['factor']} loading {_f(e['beta']):+.2f} (t = {_f(e['t_stat']):.1f}, {_pstr(e['p_value'])}) "
            f"is a headwind (3m interaction {_f(e['interaction_3m']):+.3f})."
        )
    else:
        parts.append("no FF5 style exposure clears the reliability threshold, so the read is largely price/idiosyncratic.")
    rank_v = ranks.get("rank_vol_21d"); below = (rank_v is not None and rank_v <= n / 2)
    parts.append(
        f"Trailing 3M excess return vs benchmark is {_pct(r.get('excess_ret_3m_vs_benchmark'))} "
        f"(12M total {_pct(r.get('ret_12m'))}); 252d realized vol {_f(risk.get('vol_252d'))*100:.0f}% "
        f"({'below' if below else 'above'} universe median, vol rank {rank_v}/{n}), max drawdown "
        f"{_pct(risk.get('max_drawdown_63d'))}."
    )
    parts.append(f"Estimated annualized alpha contribution {_pct(fm.get('alpha_annualized'))} "
                 f"(net style-factor support {_f(fm.get('net_style_factor_support', fm.get('net_factor_support'))):+.3f}, R²={_f(fm.get('r_squared')):.2f}).")
    # justify the exact tier
    if label in _STRONGER and adverse:
        e = adverse[0]
        parts.append(f"Signal is {label} not {_STRONGER[label]} because {e['factor']} is mildly adverse ({_f(e['beta']):+.2f}).")
    elif label == "strongly_long":
        parts.append("Conviction is high: multiple factors align with no significant adverse loading.")
    elif label == "neutral":
        parts.append("Net read is neutral: supportive and adverse factor pressure roughly offset.")
    return " ".join(parts)


def _fundamental_rationale(t: str, label: str, st: Dict) -> str:
    raw = st.get("raw_ratios", {}); ranks = st.get("sector_percentile_ranks", {})
    cons = st.get("consensus", {})
    dirword = {"strongly_long": "a high-conviction long", "long": "long", "neutral": "neutral",
               "underweight": "an underweight", "avoid": "an avoid"}[label]
    def tier(p):
        p = _f(p)
        return "top-quartile" if p >= 0.75 else ("above-median" if p >= 0.5 else ("below-median" if p >= 0.25 else "bottom-quartile"))
    parts = [f"{t} is {dirword} on sector-relative fundamentals."]
    parts.append(
        f"ROE {_f(raw.get('roe'))*100:.0f}% sits at the {_f(ranks.get('roe'))*100:.0f}th sector percentile ({tier(ranks.get('roe'))}); "
        f"EPS growth TTM {_f(raw.get('eps_growth_ttm'))*100:+.0f}% ({_f(ranks.get('eps_growth_ttm'))*100:.0f}th) and "
        f"3Y CAGR {_f(raw.get('eps_growth_3y_cagr'))*100:+.0f}% corroborate."
    )
    parts.append(
        f"Valuation P/E {_f(raw.get('pe_ratio')):.1f} / P/B {_f(raw.get('pb_ratio')):.1f} is {tier(ranks.get('pe_ratio'))} on price "
        f"(valuation percentile {_f(ranks.get('pe_ratio'))*100:.0f}); leverage D/E {_f(raw.get('debt_to_equity')):.2f}, current ratio {_f(raw.get('current_ratio')):.1f}."
    )
    parts.append(f"Consensus target {cons.get('best_target_price')} implies {_pct(cons.get('implied_upside'))} upside (BEST EPS {cons.get('best_eps')}).")
    if label in _STRONGER:
        weak = min(["roe", "eps_growth_ttm", "pe_ratio"], key=lambda k: _f(ranks.get(k)))
        parts.append(f"Signal is {label} not {_STRONGER[label]} because {weak} only ranks {_f(ranks.get(weak))*100:.0f}th in sector.")
    elif label == "strongly_long":
        parts.append("High conviction: cheap valuation AND top-quartile quality/growth with contained leverage.")
    return " ".join(parts)


def _sentiment_rationale(t: str, label: str, st: Dict) -> str:
    factual = _f(st.get("relevance_weighted_sentiment", st.get("avg_factual_sentiment"))); events = st.get("event_flags", [])
    news = st.get("news", []) or []; an = st.get("analyst", {})
    up = int(_f(st.get("analyst_upgrades"))); down = int(_f(st.get("analyst_downgrades")))
    dirword = {"strongly_long": "a high-conviction long", "long": "long", "neutral": "neutral",
               "underweight": "an underweight", "avoid": "an avoid"}[label]
    parts = [f"{t} is {dirword} on news/event flow."]
    parts.append(f"{st.get('article_count', len(news))} articles this month, average factual sentiment {factual:+.2f}.")
    if news:
        top = max(news, key=lambda n: abs(_f(n.get("sentiment_score"))))
        parts.append(f"Top item: \"{top.get('title')}\" ({top.get('source')}, FinBERT {_f(top.get('sentiment_score')):+.2f}).")
    if events:
        parts.append(f"Hard catalyst present ({', '.join(events)}) → event-driven, outranking opinion drift.")
    else:
        parts.append("No hard event — view rests on sentiment/opinion, so conviction is capped.")
    if an:
        parts.append(f"Analyst consensus {an.get('recommendation_key')} ({an.get('num_analyst_opinions')} analysts), "
                     f"target {an.get('target_mean_price')} = {_pct(an.get('implied_upside'))} upside; "
                     f"{up} upgrades / {down} downgrades.")
    return " ".join(parts)


def _economist_rationale(pkg: Dict, rate: str, risk: str, cycle: str, favored, under) -> str:
    m = pkg.get("macro_snapshot", {})
    vix = _f(m.get("vix"), 18); spread = _f(m.get("yield_curve_10y2y"))
    parts = [f"Regime — rate: {rate}, risk: {risk}, cycle: {cycle}."]
    parts.append(
        f"VIX {vix:.1f} ({risk}), fed funds {_f(m.get('fed_funds_rate')):.2f}% "
        f"with {m.get('fed_statement_tone','neutral')} guidance, 10y-2y {spread:+.2f} ({'inverted' if spread < 0 else 'positive'}), "
        f"CPI {_f(m.get('cpi_yoy')):.1f}% YoY, unemployment {_f(m.get('unemployment_rate')):.1f}%, GDP {_f(m.get('gdp_qoq')):.1f}% QoQ, WTI ${_f(m.get('oil_wti')):.0f}."
    )
    if risk == "stressed" or cycle == "contraction":
        parts.append("Elevated vol / late-cycle signals argue for defensives over cyclicals.")
    elif risk == "calm" and cycle == "expansion":
        parts.append("Contained vol and expansion support cyclicals and risk assets.")
    else:
        parts.append("Mixed signals keep the stance balanced.")
    if favored:
        parts.append(f"Overweight {', '.join(favored)}; underweight {', '.join(under) or 'none'}.")
    return " ".join(parts)


def mock_specialist_output(agent_name: str, package: Dict) -> Dict:
    return {
        "quant": _mock_quant,
        "fundamental": _mock_fundamental,
        "sentiment": _mock_sentiment,
        "economist": _mock_economist,
    }[agent_name](package)


def _meta(package: Dict, agent: str) -> Dict:
    return {
        "agent_name": agent,
        "as_of_date": package.get("as_of_date", ""),
        "horizon": package.get("horizon", "1M"),
        "portfolio_id": package.get("portfolio_id", ""),
    }


def _mock_quant(pkg: Dict) -> Dict:
    signals = {}
    for t in pkg.get("portfolio_universe", []):
        sf = pkg.get("stock_features", {}).get(t, {})
        ex3 = _f(sf.get("returns", {}).get("excess_ret_3m_vs_benchmark"))
        net = _f(sf.get("factor_model", {}).get("net_style_factor_support", sf.get("factor_model", {}).get("net_factor_support")))
        vol = _f(sf.get("risk", {}).get("vol_21d"), 0.2)
        v = _clip(6.0 * ex3 + 3.0 * net - 0.2 * (vol - 0.2))
        exps = sf.get("factor_model", {}).get("factor_exposures", [])
        style_exps = [e for e in exps if e.get("factor") in {"smb", "hml", "rmw", "cma"}]
        sig_exps = sorted(
            [e for e in style_exps if e.get("signal_strength") in {"supporting", "adverse"}],
            key=lambda e: abs(_f(e.get("interaction_3m"))),
            reverse=True,
        )
        prim = [e["factor"] for e in sig_exps[:2]] or [
            e["factor"]
            for e in sorted(
                style_exps,
                key=lambda e: abs(_f(e.get("interaction_3m"))),
                reverse=True,
            )[:2]
        ]
        align = "supporting" if net > 0.01 else ("adverse" if net < -0.01 else "mixed")
        label = _label5(v)
        cited = [
            {"name": "excess_ret_3m_vs_benchmark", "value": ex3},
            {"name": "ret_12m", "value": _f(sf.get("returns", {}).get("ret_12m"))},
            {"name": "vol_252d", "value": _f(sf.get("risk", {}).get("vol_252d"))},
            {"name": "net_style_factor_support", "value": net},
            {"name": "alpha_annualized", "value": _f(sf.get("factor_model", {}).get("alpha_annualized"))},
        ]
        for e in sig_exps[:2]:
            cited.append({"name": f"{e['factor']}_beta", "value": _f(e.get("beta"))})
        signals[t] = {
            "signal": label,
            "primary_factors": prim or ["price_action"],
            "factor_alignment": align,
            "alpha_contribution": round(
                _f(sf.get("factor_model", {}).get("alpha_annualized")), 4
            ) if _f(sf.get("factor_model", {}).get("alpha_p_value"), 1.0) <= 0.10 else 0.0,
            "rationale": _quant_rationale(t, label, sf, len(pkg.get("portfolio_universe", [])) or 6),
            "cited_values": cited,
        }
    return {
        **_meta(pkg, "quant"),
        "stock_signals": signals,
        "portfolio_level_view": {
            "recommended_tilt": "tilt toward positive net-factor-support names",
            "covariance_note": f"regime: {pkg.get('macro_regime')}",
            "whiteboard_instruction": "cap single-name and per-sector exposure per the risk gate",
        },
        "summary": "[mock] quant factor/price evidence.",
    }


def _mock_fundamental(pkg: Dict) -> Dict:
    signals = {}
    for t in pkg.get("portfolio_universe", []):
        st = pkg.get("stocks", {}).get(t, {})
        ranks = st.get("sector_percentile_ranks", {})
        roe = _f(ranks.get("roe"), 0.5)
        epsg = _f(ranks.get("eps_growth_ttm"), 0.5)
        val = _f(ranks.get("pe_ratio"), 0.5)
        upside = _f(st.get("consensus", {}).get("implied_upside"))
        quality = (roe + epsg + val) / 3.0
        v = _clip(2.4 * (quality - 0.5) + 0.8 * upside)
        drivers = sorted(ranks, key=lambda k: ranks[k], reverse=True)[:3]
        label = _label5(v)
        signals[t] = {
            "signal": label,
            "valuation_call": "cheap" if val > 0.6 else ("expensive" if val < 0.4 else "fair"),
            "quality_tier": "strong" if roe > 0.6 else ("weak" if roe < 0.4 else "average"),
            "key_drivers": drivers,
            "rationale": _fundamental_rationale(t, label, st),
            "cited_values": [
                {"name": "roe", "value": roe},
                {"name": "eps_growth_ttm", "value": epsg},
                {"name": "pe_ratio", "value": val},
                {"name": "implied_upside", "value": upside},
            ],
        }
    return {**_meta(pkg, "fundamental"), "signals": signals, "summary": "[mock] fundamental bottom-up."}


def _mock_sentiment(pkg: Dict) -> Dict:
    signals = {}
    for t in pkg.get("portfolio_universe", []):
        st = pkg.get("stocks", {}).get(t, {})
        factual = _f(st.get("relevance_weighted_sentiment", st.get("avg_factual_sentiment")))
        up = _f(st.get("analyst_upgrades"))
        down = _f(st.get("analyst_downgrades"))
        events = st.get("event_flags", [])
        has_event = any(e in events for e in ("earnings_surprise", "regulatory", "M&A"))
        analyst = st.get("analyst", {})
        v = _clip(1.3 * factual + 0.15 * (up - down) + 0.2 * _f(analyst.get("implied_upside")))
        catalyst = (events[0] if events else ("analyst upgrades" if up > down else "sentiment drift"))
        label = _label5(v)
        signals[t] = {
            "signal": label,
            "primary_catalyst": catalyst,
            "event_driven": bool(has_event),
            "rationale": _sentiment_rationale(t, label, st),
            "cited_values": [
                {"name": "relevance_weighted_sentiment", "value": factual},
                {"name": "surviving_article_count", "value": _f(st.get("surviving_article_count", st.get("article_count")))},
                {"name": "analyst_upgrades", "value": up},
                {"name": "analyst_downgrades", "value": down},
                {"name": "implied_upside", "value": _f(st.get("analyst", {}).get("implied_upside"))},
            ],
        }
    return {**_meta(pkg, "sentiment"), "signals": signals, "summary": "[mock] news/event sentiment."}


def _economist_regime(macro: Dict):
    """Derive the three regime axes from the macro snapshot (RATE/RISK/CYCLE)."""
    vix = _f(macro.get("vix"), 18.0)
    gdp = _f(macro.get("gdp_qoq"), 2.0)
    curve = _f(macro.get("yield_curve_10y2y"), 0.5)
    fed6m = _f(macro.get("fed_funds_6m_change"), 0.0)
    tone = macro.get("fed_statement_tone", "neutral")
    risk = "calm" if vix < 16 else ("stressed" if vix > 22 else "normal")
    rate = ("tightening" if (tone == "hawkish" or fed6m > 0.15)
            else ("easing" if (tone == "dovish" or fed6m < -0.15) else "stable"))
    if gdp >= 2.0 and curve > 0:
        cycle = "expansion"
    elif gdp < 0.5 or curve < 0:
        cycle = "contraction"
    else:
        cycle = "mid_cycle"
    return rate, risk, cycle


def _mock_economist(pkg: Dict) -> Dict:
    stances = pkg.get("sector_stances", {})
    macro = pkg.get("macro_snapshot", {})
    rate, risk, cycle = _economist_regime(macro)
    sector_views, favored, under = {}, [], []
    for sec in pkg.get("sectors_in_universe", []) or list(stances.keys()):
        st = stances.get(sec, "neutral")
        stance = st if st in ("overweight", "underweight") else "neutral"
        macro_cited = [
            {"name": "vix", "value": _f(macro.get("vix"))},
            {"name": "yield_curve_10y2y", "value": _f(macro.get("yield_curve_10y2y"))},
            {"name": "fed_funds_rate", "value": _f(macro.get("fed_funds_rate"))},
        ]
        if sec == "Energy":
            macro_cited.append({"name": "oil_wti_yoy", "value": _f(macro.get("oil_wti_yoy"))})
        if sec in ("Industrials", "Materials"):
            macro_cited.append({"name": "manufacturing_production_yoy", "value": _f(macro.get("manufacturing_production_yoy"))})
        if sec in ("Financials", "Real Estate"):
            macro_cited.append({"name": "yield_10y", "value": _f(macro.get("yield_10y"))})
        rationale = f"{sec}: {stance}; cites " + ", ".join(f"{c['name']}={c['value']}" for c in macro_cited[:4])
        sector_views[sec] = {"stance": stance, "rationale": rationale, "cited_values": macro_cited}
        if stance == "overweight":
            favored.append(sec)
        elif stance == "underweight":
            under.append(sec)
    return {
        **_meta(pkg, "economist"),
        "rate_environment": rate,
        "risk_regime": risk,
        "cycle": cycle,
        "regime_rationale": _economist_rationale(pkg, rate, risk, cycle, favored, under),
        "cited_values": [
            {"name": "vix", "value": _f(macro.get("vix"))},
            {"name": "fed_funds_rate", "value": _f(macro.get("fed_funds_rate"))},
            {"name": "yield_curve_10y2y", "value": _f(macro.get("yield_curve_10y2y"))},
            {"name": "cpi_yoy", "value": _f(macro.get("cpi_yoy"))},
            {"name": "unemployment_rate", "value": _f(macro.get("unemployment_rate"))},
            {"name": "gdp_qoq", "value": _f(macro.get("gdp_qoq"))},
        ],
        "sector_views": sector_views,
        "favored_sectors": favored,
        "risk_sectors": under,
        "summary": "[mock] macro regime (rate/risk/cycle) + sector stances.",
    }


# --------------------------------------------------------------------------- #
# Phase 3 debate mocks -- these MOVE positions toward persuasive opponents so the
# revised-position machinery is exercised even with no model running.
# --------------------------------------------------------------------------- #
def mock_opening_statement(agent_name: str, ticker: str, phase1_signal: Dict, scale: float) -> Dict:
    sig = _f(phase1_signal.get("signal"))
    conf = _f(phase1_signal.get("confidence"), 0.5)
    direction = "long" if sig > 0.2 else ("short" if sig < -0.2 else "neutral")
    return {
        "agent_name": agent_name,
        "ticker": ticker,
        "direction": direction,
        "q": round(sig * scale, 4),
        "confidence": round(conf, 4),
        "position_rationale": phase1_signal.get("rationale", "")[:300] or f"{agent_name} view.",
        "objection_to_current_view": "",
        "evidence": [
            {"name": cv["name"], "value": _f(cv.get("value")), "note": ""}
            for cv in phase1_signal.get("cited_values", [])[:3]
        ],
        "what_would_change_my_mind": "Strong opposing factual evidence from another agent.",
    }


def mock_rebuttal_statement(
    agent_name: str,
    ticker: str,
    own_opening: Dict,
    all_openings: Dict[str, Dict],
    scale: float,
    round_no: int = 1,
    working_q: float = None,
) -> Dict:
    """Revised position = own q pulled toward the 'room' (the confidence-weighted view
    of the OTHER agents, anchored by the moderator's current working q). The pull
    GROWS each round (0.45 -> 0.70 -> 0.90), so genuine listening accumulates and the
    panel converges by round 3 unless conviction is extreme (spec 3a)."""
    own_q = _f(own_opening.get("q"))
    own_conf = _f(own_opening.get("confidence"), 0.5)
    others = {a: o for a, o in all_openings.items() if a != agent_name}
    if others:
        wsum = sum(_f(o.get("confidence"), 0.3) for o in others.values()) or 1.0
        cons_q = sum(_f(o.get("q")) * _f(o.get("confidence"), 0.3) for o in others.values()) / wsum
        # which opponent moved me: the most credible one furthest from me
        opponent = max(others, key=lambda a: abs(_f(others[a].get("q")) - own_q) * _f(others[a].get("confidence"), 0.3))
        opp_q = _f(others[opponent].get("q"))
    else:
        cons_q, opponent, opp_q = own_q, None, own_q

    # focal point = the room consensus, nudged toward the moderator's working view
    focal = cons_q if working_q is None else 0.7 * cons_q + 0.3 * _f(working_q)
    base_pull = {1: 0.30, 2: 0.62}.get(round_no, 0.90)
    # very high own-conviction resists a little; low conviction yields more
    pull = _clip(base_pull * (1.25 - 0.5 * own_conf), 0.2, 0.95)
    # ANTI-CONFORMITY: a high-conviction agent holding a cited fact resists the majority
    # pull, so a genuine high-conviction conflict survives to Round 3 (where a context
    # agent tiebreaks it) instead of being smoothed away. A grounded minority must live.
    if own_conf >= 0.75:
        pull = min(pull, 0.30)
    revised_q = own_q + pull * (focal - own_q)
    moved = abs(revised_q - own_q)
    revised_dir = "long" if revised_q > 0.01 else ("short" if revised_q < -0.01 else "neutral")

    concessions, influenced, still = [], [], []
    if moved > 1e-3 and opponent is not None:
        verb = "conceded materially to" if moved > 0.02 else "softened toward"
        concessions.append(
            f"{verb} {opponent} (their q={opp_q:+.3f}): moved my q {own_q:+.3f} → {revised_q:+.3f}."
        )
        influenced.append(opponent)
        # name a second influencer if a third agent also pulled hard
        second = sorted(others, key=lambda a: abs(_f(others[a].get("q")) - own_q), reverse=True)
        if len(second) > 1 and second[1] != opponent and abs(_f(others[second[1]].get("q")) - own_q) > 0.02:
            influenced.append(second[1])
    else:
        still.append(f"Hold my read; opposing evidence (e.g. {opponent}) doesn't overturn my numbers.")
    # an agent that moved toward the group gains a little conviction; a holdout keeps its own
    new_conf = own_conf + (0.06 if moved > 1e-3 else -0.02)

    # roundtable: address the 1-2 most-disagreeing peers BY NAME (spec B)
    ranked = sorted(others, key=lambda a: abs(_f(others[a].get("q")) - own_q), reverse=True)
    replies = []
    for peer in ranked[:2]:
        pq = _f(others[peer].get("q"))
        gap = abs(pq - own_q)
        same_dir = (pq > 0) == (own_q > 0)
        if same_dir:
            stance = "agree"
        elif moved > 0.02:
            stance = "partial"
        else:
            stance = "disagree"
        claim = (others[peer].get("position_rationale") or "")[:90] or f"q={pq:+.3f}"
        msg = _reply_message(agent_name, peer, stance, pq, own_q, revised_q)
        replies.append({
            "responding_to": peer,
            "stance": stance,
            "quoted_claim": claim,
            "my_response": msg,
            "cited_number": round(pq, 4),
            "message": msg,
        })
    return {
        "agent_name": agent_name,
        "ticker": ticker,
        "revised_direction": revised_dir,
        "revised_q": round(revised_q, 4),
        "revised_confidence": round(_clip(new_conf, 0.1, 0.95), 4),
        "replies": replies,
        "responded_to": opponent or "",
        "concessions": concessions,
        "still_objects": still,
        "influenced_by": influenced,
        "strongest_counterargument_agent": opponent,
        "engaged": bool(replies),
        "new_fact_or_correction": bool(replies and moved > 1e-3),
    }


def _reply_message(me: str, peer: str, stance: str, peer_q: float, own_q: float, revised_q: float) -> str:
    pname = peer.capitalize()
    if stance == "agree":
        return (f"{pname}, your {peer_q:+.1%} view lines up with mine ({own_q:+.1%}); "
                f"I'll hold near {revised_q:+.1%} — we're aligned.")
    if stance == "partial":
        return (f"{pname}, your {peer_q:+.1%} read is partly right and it moves me from "
                f"{own_q:+.1%} to {revised_q:+.1%}, but I won't go all the way to your number.")
    return (f"{pname}, I disagree — your {peer_q:+.1%} call cuts against my evidence; "
            f"I'm staying near {own_q:+.1%} until the numbers say otherwise.")


def mock_moderator_synthesis(
    ticker: str, openings: Dict[str, Dict], rebuttals: Dict[str, Dict], scale: float
) -> Dict:
    """Confidence-weighted synthesis of the agents' REVISED positions."""
    items = list(rebuttals.values())
    wsum = sum(_f(r.get("revised_confidence"), 0.3) for r in items) or 1.0
    q = sum(_f(r.get("revised_q")) * _f(r.get("revised_confidence"), 0.3) for r in items) / wsum
    conf = float(np.clip(np.mean([_f(r.get("revised_confidence"), 0.3) for r in items]), 0.1, 0.95))
    direction = "long" if q > 0.01 else ("short" if q < -0.01 else "neutral")
    swayed = sorted(rebuttals, key=lambda a: _f(rebuttals[a].get("revised_confidence")), reverse=True)[:2]
    changed = [a for a, r in rebuttals.items() if r.get("concessions")]
    objections = [o for r in items for o in r.get("still_objects", [])]
    return {
        "ticker": ticker,
        "recommended_direction": direction,
        "recommended_q": round(q, 4),
        "recommended_confidence": round(conf, 4),
        "moderator_reasoning_summary": (
            f"[mock] confidence-weighted revised positions -> q={q:+.3f} ({direction})."
        ),
        "who_swayed_the_decision": swayed,
        "agents_that_changed_position": changed,
        "unresolved_objections": objections[:4],
    }
