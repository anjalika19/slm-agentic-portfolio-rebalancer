"""Compact input-injection payloads for the model (spec allows refactoring the
input-injection format).

The full agent packages are rich (nested factor exposures, ranks, covariance) -- by
design, because the validator traces cited values against them and the UI shows
them. But feeding the entire nested JSON to a local Gemma makes prompts thousands of
tokens long and ruinously slow on CPU. These builders extract just the salient,
decision-relevant numbers per ticker into a small flat structure. Field NAMES are
preserved so the model's cited_values still trace back to the full package.
"""

from __future__ import annotations

from typing import Dict, List


def _round(x, d=4):
    try:
        return round(float(x), d)
    except (TypeError, ValueError):
        return x


def compact_package(agent: str, package: Dict, as_of_date: str = "") -> Dict:
    if agent == "sentiment":
        return _compact_sentiment(package, as_of_date)
    if agent == "fundamental":
        return _compact_fundamental(package, as_of_date)
    return {"quant": _compact_quant, "economist": _compact_economist}[agent](package)


def _compact_quant(p: Dict) -> Dict:
    """Keep every FF5 number the prompt asks the model to cite.

    FF5 has only five factors, so carrying the complete factor-regime block is
    inexpensive and avoids the old prompt/payload mismatch where p-values and
    interactions were requested but never shown to the model.
    """
    out = {
        "macro_regime": p.get("macro_regime"),
        "factor_model_spec": p.get("factor_model_spec", {}),
        "factor_regime": p.get("factor_regime", {}),
        "stocks": {},
    }
    for ticker, stock in p.get("stock_features", {}).items():
        returns = stock.get("returns", {})
        risk = stock.get("risk", {})
        factor_model = stock.get("factor_model", {})
        ranks = stock.get("cross_sectional_ranks", {})

        style_exposures = [
            exposure
            for exposure in factor_model.get("factor_exposures", [])
            if exposure.get("factor") in {"smb", "hml", "rmw", "cma"}
        ]
        style_exposures = sorted(
            style_exposures,
            key=lambda exposure: abs(exposure.get("interaction_3m") or 0.0),
            reverse=True,
        )
        market = next(
            (
                exposure
                for exposure in factor_model.get("factor_exposures", [])
                if exposure.get("factor") == "mkt_rf"
            ),
            None,
        )

        def compact_exposure(exposure):
            if not exposure:
                return None
            return {
                "factor": exposure.get("factor"),
                "beta": exposure.get("beta"),
                "t_stat": exposure.get("t_stat"),
                "p_value": exposure.get("p_value"),
                "factor_return_3m": exposure.get("factor_return_3m"),
                "interaction_3m": exposure.get("interaction_3m"),
                "signal_strength": exposure.get("signal_strength"),
            }

        out["stocks"][ticker] = {
            "returns": {
                "ret_1m": returns.get("ret_1m"),
                "ret_3m": returns.get("ret_3m"),
                "ret_12m": returns.get("ret_12m"),
                "excess_ret_1m_vs_benchmark": returns.get("excess_ret_1m_vs_benchmark"),
                "excess_ret_3m_vs_benchmark": returns.get("excess_ret_3m_vs_benchmark"),
                "excess_ret_6m_vs_benchmark": returns.get("excess_ret_6m_vs_benchmark"),
            },
            "risk": {
                "vol_21d": risk.get("vol_21d"),
                "vol_252d": risk.get("vol_252d"),
                "tracking_vol_63d": risk.get("tracking_vol_63d"),
                "beta_to_benchmark_252d": risk.get("beta_to_benchmark_252d"),
                "max_drawdown_63d": risk.get("max_drawdown_63d"),
            },
            "cross_sectional_ranks": {
                "rank_ret_3m": ranks.get("rank_ret_3m"),
                "rank_ret_12m": ranks.get("rank_ret_12m"),
                "rank_vol_21d": ranks.get("rank_vol_21d"),
                "rank_net_style_factor_support": ranks.get(
                    "rank_net_style_factor_support",
                    ranks.get("rank_net_factor_support"),
                ),
            },
            "factor_model": {
                "n_obs": factor_model.get("n_obs"),
                "alpha_annualized": factor_model.get("alpha_annualized"),
                "alpha_t_stat": factor_model.get("alpha_t_stat"),
                "alpha_p_value": factor_model.get("alpha_p_value"),
                "r_squared": factor_model.get("r_squared"),
                "adj_r_squared": factor_model.get("adj_r_squared"),
                "market_exposure": compact_exposure(market),
                "net_style_factor_support": factor_model.get(
                    "net_style_factor_support",
                    factor_model.get("net_factor_support"),
                ),
                "top_style_factors": [
                    compact_exposure(exposure) for exposure in style_exposures[:4]
                ],
            },
        }
    return out


def _days_since_earnings(earn_ann_dt: str, as_of_date: str) -> int:
    """Days between last earnings announcement and rebalancing date."""
    if not earn_ann_dt or not as_of_date:
        return None
    try:
        ann = datetime.strptime(str(earn_ann_dt)[:10], "%Y-%m-%d")
        ref = datetime.strptime(as_of_date, "%Y-%m-%d")
        return max(0, (ref - ann).days)
    except (ValueError, TypeError):
        return None


def _fundamental_staleness_label(days: int) -> str:
    if days is None:
        return "unknown"
    if days <= 30:
        return "fresh"       # within 30 days — high confidence
    if days <= 60:
        return "recent"      # 30-60 days — normal
    if days <= 89:
        return "aging"       # 60-89 days — end of quarter, next earnings soon
    return "stale"           # 90+ days — earnings overdue, data unreliable


def _compact_fundamental(p: Dict, as_of_date: str = "") -> Dict:
    out = {"stocks": {}}
    for t, st in p.get("stocks", {}).items():
        earn_ann_dt = st.get("earn_ann_dt")
        days = _days_since_earnings(earn_ann_dt, as_of_date)
        out["stocks"][t] = {
            "sector": st.get("sector"),
            "sector_percentile_ranks": st.get("sector_percentile_ranks", {}),
            "implied_upside": st.get("consensus", {}).get("implied_upside"),
            # Staleness metadata — agent uses these to calibrate confidence
            "earn_ann_dt": earn_ann_dt,
            "days_since_earnings": days,
            "data_freshness": _fundamental_staleness_label(days),
            # Key raw ratios for richer debate arguments
            "key_raw_ratios": {
                k: st.get("raw_ratios", {}).get(k)
                for k in ["gross_margin_trend", "eps_growth_ttm", "eps_growth_3y_cagr",
                          "revenue_growth", "roe"]
                if st.get("raw_ratios", {}).get(k) is not None
            },
        }
    return out


from datetime import datetime, timezone

def _sentiment_decay(time_published: str, as_of_date: str, half_life_days: float = 7.0) -> float:
    """Exponential decay weight: 1.0 at as_of_date, 0.5 at half_life_days ago."""
    try:
        # time_published format: "20240628T131253" or "20240628T000000"
        pub = datetime.strptime(time_published[:15], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        ref = datetime.strptime(as_of_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        days_old = max(0.0, (ref - pub).total_seconds() / 86400.0)
        return 2.0 ** (-days_old / half_life_days)
    except (ValueError, TypeError):
        return 0.5  # unknown age → medium weight


def _compact_sentiment(p: Dict, as_of_date: str = "") -> Dict:
    out = {"market_context": p.get("market_context", {}), "stocks": {}}
    for t, st in p.get("stocks", {}).items():
        news_raw = st.get("news") or []
        
        # Sort by |sentiment| descending — keep top 3 positive + top 3 negative
        pos = sorted([n for n in news_raw if (n.get("ticker_sentiment_score") or 0) > 0],
                     key=lambda n: n.get("ticker_sentiment_score", 0), reverse=True)[:3]
        neg = sorted([n for n in news_raw if (n.get("ticker_sentiment_score") or 0) < 0],
                     key=lambda n: n.get("ticker_sentiment_score", 0))[:3]
        news_selected = pos + neg
        
        # Recency-weighted sentiment
        if news_selected and as_of_date:
            weights = [_sentiment_decay(n.get("time_published", ""), as_of_date)
                       for n in news_selected]
            scores = [n.get("ticker_sentiment_score") or n.get("sentiment_score") or 0.0
                      for n in news_selected]
            wsum = sum(weights) or 1.0
            recency_weighted = sum(s * w for s, w in zip(scores, weights)) / wsum
        else:
            recency_weighted = st.get("relevance_weighted_sentiment")
        
        out["stocks"][t] = {
            "avg_factual_sentiment": st.get("avg_factual_sentiment"),
            "relevance_weighted_sentiment": st.get("relevance_weighted_sentiment"),
            "recency_weighted_sentiment": round(recency_weighted, 4) if recency_weighted else None,
            "article_count": st.get("article_count"),
            "surviving_article_count": st.get("surviving_article_count"),
            "event_flags": st.get("event_flags", []),
            "analyst_upgrades": st.get("analyst_upgrades"),
            "analyst_downgrades": st.get("analyst_downgrades"),
            "news": [
                {
                    "title": n.get("title"),
                    "source": n.get("source"),
                    "ticker_sentiment_score": n.get("ticker_sentiment_score"),
                    "relevance_score": n.get("relevance_score", n.get("relevance")),
                    "time_published": n.get("time_published"),
                    # Pre-compute days_ago so the model doesn't have to parse dates
                    "days_ago": _days_ago(n.get("time_published", ""), as_of_date),
                }
                for n in news_selected
            ],
            "analyst": st.get("analyst", {}),
        }
    return out


def _days_ago(time_published: str, as_of_date: str) -> int:
    """Return integer days between article and rebalancing date. None if unparseable."""
    try:
        pub = datetime.strptime(time_published[:8], "%Y%m%d")
        ref = datetime.strptime(as_of_date, "%Y-%m-%d")
        return max(0, (ref - pub).days)
    except (ValueError, TypeError):
        return None


def _compact_economist(p: Dict) -> Dict:
    return {
        "macro_snapshot": p.get("macro_snapshot", {}),
        "macro_context": p.get("macro_context", {}),
        "derived_regime": p.get("derived_regime"),
        "sector_stances": p.get("sector_stances", {}),
        "sectors_in_universe": p.get("sectors_in_universe", []),
        "universe_sector_mapping": p.get("universe_sector_mapping", {}),
    }


def compact_view(view: Dict) -> Dict:
    return {
        "ticker": view.get("ticker"),
        "direction": view.get("direction"),
        "q": _round(view.get("q"), 4),
        "confidence": _round(view.get("confidence"), 3),
    }


def compact_opening(o: Dict) -> Dict:
    return {
        "agent_name": o.get("agent_name"),
        "direction": o.get("direction"),
        "q": _round(o.get("q"), 4),
        "confidence": _round(o.get("confidence"), 3),
        "position_rationale": (o.get("position_rationale") or "")[:220],
        "objection": (o.get("objection_to_current_view") or "")[:160],
        "evidence": (o.get("evidence") or [])[:3],
    }


def compact_rebuttal(r: Dict) -> Dict:
    return {
        "agent_name": r.get("agent_name"),
        "revised_direction": r.get("revised_direction"),
        "revised_q": _round(r.get("revised_q"), 4),
        "revised_confidence": _round(r.get("revised_confidence"), 3),
        "concessions": (r.get("concessions") or [])[:3],
        "still_objects": (r.get("still_objects") or [])[:3],
        "influenced_by": r.get("influenced_by") or [],
    }
