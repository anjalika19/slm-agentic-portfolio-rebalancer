"""FixtureProvider -- deterministic, realistic, point-in-time packages (spec 6).

Builds the four agent packages (matching the live JSON layout from
``05_input_json_assembler.py`` and the folder builders) and a valid PSD
covariance, entirely from the self-consistent synthetic market in
``synthetic_market.py`` plus a few deterministic latent scores. No Bloomberg /
FRED / API keys required, so the whole pipeline + UI run anywhere.

Differentiation is intentional: GOOGL/UNH carry positive alpha + quality, XOM/BAC
negative, so specialists genuinely disagree and the debate has work to do.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from config import (
    ANNUAL_RETURN_SCALE,
    BENCHMARK,
    COV_WINDOW_DAYS,
    SECTOR_BY_TICKER,
)
from factor_model import run_factor_regression, to_monthly_returns
from providers.base import DataProvider, InputBundle
from providers import synthetic_market as mkt
from providers import quant_block


def _stable_rng(*parts) -> np.random.Generator:
    h = 0
    for p in parts:
        for ch in str(p):
            h = (h * 131 + ord(ch)) & 0x7FFFFFFF
    return np.random.default_rng(h)


def _pct_rank(values: Dict[str, float], higher_is_better: bool = True) -> Dict[str, float]:
    """Percentile rank where 1.0 == best (top of the GICS-sector cross-section)."""
    s = pd.Series(values, dtype=float)
    # best value -> highest percentile: ascending matches higher_is_better.
    r = s.rank(ascending=higher_is_better, pct=True)
    return {k: round(float(v), 3) for k, v in r.items()}


def _rank_int(values: Dict[str, float], higher_is_better: bool = True) -> Dict[str, int]:
    s = pd.Series(values, dtype=float)
    r = s.rank(ascending=not higher_is_better, method="min")
    return {k: int(v) for k, v in r.items()}


# latent per-ticker quality used to make fundamentals/sentiment differentiated
_QUALITY = {"GOOGL": 0.8, "UNH": 0.6, "BAC": -0.2, "CAT": 0.25, "PEP": 0.1, "XOM": -0.45}


class FixtureProvider(DataProvider):
    name = "fixture"

    # ---- public API ---------------------------------------------------- #
    def get_input_bundle(
        self, portfolio_id: str, as_of_date: str, universe: List[str], horizon: str = "1M"
    ) -> InputBundle:
        ts = pd.Timestamp(as_of_date)
        px = mkt.prices_through(ts)[universe + [BENCHMARK]]
        warnings: List[str] = []
        if len(px) < 260:
            warnings.append(f"only {len(px)} daily obs through {as_of_date}; factor history thin")

        cov_block = quant_block.covariance_block(px, universe)
        factor_daily = mkt.factor_returns_through(ts)
        macro = self._macro_snapshot(ts)

        sectors = {t: SECTOR_BY_TICKER.get(t, "Unknown") for t in universe}

        packages = {
            "quant": quant_block.build_quant_package(
                portfolio_id, as_of_date, horizon, universe, px, BENCHMARK,
                factor_daily, macro["macro_regime"],
            ),
            "fundamental": self._fundamental_package(
                portfolio_id, as_of_date, horizon, universe, px
            ),
            "sentiment": self._sentiment_package(
                portfolio_id, as_of_date, horizon, universe, px, macro
            ),
            "economist": self._economist_package(
                portfolio_id, as_of_date, horizon, universe, sectors, macro
            ),
        }

        return InputBundle(
            portfolio_id=portfolio_id,
            as_of_date=as_of_date,
            horizon=horizon,
            universe=list(universe),
            sectors=sectors,
            packages=packages,
            covariance=cov_block,
            warnings=warnings,
        )

    def forward_return(self, ticker: str, start_date: str, end_date: str) -> float:
        return mkt.forward_total_return(ticker, start_date, end_date)

    def benchmark_forward_return(self, start_date: str, end_date: str) -> float:
        return mkt.forward_total_return(BENCHMARK, start_date, end_date)

    def risk_free_return(self, start_date: str, end_date: str) -> float:
        rf = mkt.daily_factor_returns()["rf"]
        mask = (rf.index > pd.Timestamp(start_date)) & (rf.index <= pd.Timestamp(end_date))
        holding = rf.loc[mask].dropna()
        if holding.empty:
            return 0.0
        return float((1.0 + holding).prod() - 1.0)

    def baseline_inputs(self, universe: List[str], as_of_date: str):
        """Point-in-time (mu, Sigma) for baselines: trailing 12m mean + rolling
        60-month monthly covariance, both annualized."""
        px = mkt.prices_through(as_of_date)[universe]
        monthly = px.resample("ME").last().pct_change().dropna(how="any")
        if len(monthly) < 24:
            return None
        mu = monthly.tail(12).mean().reindex(universe).to_numpy() * 12.0
        cov = (monthly.tail(60).cov() * 12.0).loc[universe, universe].to_numpy()
        cov = 0.5 * (cov + cov.T)
        return mu, cov

    # ---- shared computed blocks --------------------------------------- #
    def _macro_snapshot(self, ts: pd.Timestamp) -> Dict:
        # deterministic monthly macro that drifts slowly; VIX + tone reused by sentiment
        rng = _stable_rng("macro", ts.strftime("%Y-%m"))
        vix = round(float(14 + 8 * rng.random()), 2)
        fed = round(float(4.5 + 1.0 * (rng.random() - 0.5)), 2)
        regime = "Risk-On" if vix < 18 else ("Defensive" if vix > 22 else "Uncertain")
        tone = "dovish" if fed < 4.3 else ("hawkish" if fed > 4.7 else "neutral")
        return {
            "fed_funds_rate": fed,
            "fed_funds_6m_change": round(float(rng.normal(0, 0.25)), 2),
            "yield_2y": round(fed - 0.2, 2),
            "yield_10y": round(fed + 0.1, 2),
            "yield_curve_10y2y": round(0.3 + rng.normal(0, 0.2), 2),
            "vix": vix,
            "credit_spread_baa10y": round(float(1.6 + 0.6 * rng.random()), 2),
            "cpi_yoy": round(float(2.5 + 1.5 * rng.random()), 1),
            "unemployment_rate": round(float(3.7 + 0.6 * rng.random()), 1),
            "gdp_qoq": round(float(1.5 + 2.0 * (rng.random() - 0.3)), 1),
            "manufacturing_production_yoy": round(float(rng.normal(0.5, 1.5)), 1),
            "oil_wti": round(float(70 + 20 * rng.random()), 1),
            "fed_statement_tone": tone,
            "macro_regime": regime,
        }

    def _fundamental_package(self, pid, as_of, horizon, universe, px) -> Dict:
        raw: Dict[str, Dict[str, float]] = {}
        for t in universe:
            rng = _stable_rng("funda", t, as_of[:7])
            q = _QUALITY.get(t, 0.0)
            raw[t] = {
                "pe_ratio": round(float(22 - 6 * q + rng.normal(0, 2)), 2),
                "pb_ratio": round(float(3.5 - 1.2 * q + rng.normal(0, 0.4)), 2),
                "roe": round(float(0.15 + 0.12 * q + rng.normal(0, 0.02)), 4),
                "debt_to_equity": round(float(1.1 - 0.5 * q + rng.normal(0, 0.15)), 3),
                "eps_growth_ttm": round(float(0.08 + 0.15 * q + rng.normal(0, 0.03)), 4),
                "eps_growth_3y_cagr": round(float(0.07 + 0.12 * q + rng.normal(0, 0.02)), 4),
                "revenue_growth": round(float(0.06 + 0.10 * q + rng.normal(0, 0.02)), 4),
                "gross_margin_trend": round(float(0.01 * q + rng.normal(0, 0.005)), 4),
                "current_ratio": round(float(1.3 + 0.4 * q + rng.normal(0, 0.15)), 3),
                "best_eps": round(float(8 + 6 * q + rng.normal(0, 1)), 2),
                "best_target_price": round(float(px[t].iloc[-1] * (1.0 + 0.12 * q + rng.normal(0, 0.04))), 2),
            }

        # GICS-sector percentile ranks (here: cross-sectional across the universe)
        ranks = {
            "pe_ratio": _pct_rank({t: raw[t]["pe_ratio"] for t in universe}, higher_is_better=False),
            "pb_ratio": _pct_rank({t: raw[t]["pb_ratio"] for t in universe}, higher_is_better=False),
            "roe": _pct_rank({t: raw[t]["roe"] for t in universe}, higher_is_better=True),
            "debt_to_equity": _pct_rank({t: raw[t]["debt_to_equity"] for t in universe}, higher_is_better=False),
            "eps_growth_ttm": _pct_rank({t: raw[t]["eps_growth_ttm"] for t in universe}, higher_is_better=True),
            "eps_growth_3y_cagr": _pct_rank({t: raw[t]["eps_growth_3y_cagr"] for t in universe}, higher_is_better=True),
            "revenue_growth": _pct_rank({t: raw[t]["revenue_growth"] for t in universe}, higher_is_better=True),
            "gross_margin_trend": _pct_rank({t: raw[t]["gross_margin_trend"] for t in universe}, higher_is_better=True),
            "current_ratio": _pct_rank({t: raw[t]["current_ratio"] for t in universe}, higher_is_better=True),
        }

        stocks = {}
        for t in universe:
            last = float(px[t].iloc[-1])
            upside = round(raw[t]["best_target_price"] / last - 1.0, 4)
            stocks[t] = {
                "sector": SECTOR_BY_TICKER.get(t, "Unknown"),
                "raw_ratios": raw[t],
                "sector_percentile_ranks": {k: ranks[k][t] for k in ranks},
                "consensus": {
                    "best_eps": raw[t]["best_eps"],
                    "best_target_price": raw[t]["best_target_price"],
                    "implied_upside": upside,
                },
            }
        return {
            "task": "fundamental_portfolio_signal_generation",
            "as_of_date": as_of,
            "horizon": horizon,
            "portfolio_id": pid,
            "portfolio_universe": list(universe),
            "stocks": stocks,
        }

    # ---- sentiment package -------------------------------------------- #
    def _sentiment_package(self, pid, as_of, horizon, universe, px, macro) -> Dict:
        stocks = {}
        for t in universe:
            rng = _stable_rng("sent", t, as_of[:7])
            q = _QUALITY.get(t, 0.0)
            factual = round(float(np.clip(0.3 * q + rng.normal(0, 0.15), -1, 1)), 3)
            subjective = round(float(np.clip(0.4 * q + rng.normal(0, 0.25), -1, 1)), 3)
            n_articles = int(rng.integers(5, 60))
            up = int(rng.integers(0, 4) + (1 if q > 0.3 else 0))
            down = int(rng.integers(0, 4) + (1 if q < -0.2 else 0))
            events = []
            if rng.random() < 0.25:
                events.append("earnings_surprise")
            if rng.random() < 0.10:
                events.append("regulatory")
            if rng.random() < 0.07:
                events.append("M&A")
            # synthetic AlphaVantage-style news headlines (title, source, FinBERT-style
            # sentiment score, relevance) -- the live path replaces these with real news
            news = self._synth_news(t, rng, q, events)
            # yfinance ticker.info-style analyst block
            last = float(px[t].iloc[-1])
            target = round(last * (1.0 + 0.10 * q + float(rng.normal(0, 0.05))), 2)
            rec = ("buy" if q > 0.3 else ("sell" if q < -0.3 else "hold"))
            stocks[t] = {
                "avg_factual_sentiment": factual,
                "avg_subjective_sentiment": subjective,
                "article_count": n_articles,
                "event_flags": events,
                "analyst_upgrades": up,
                "analyst_downgrades": down,
                "news": news,
                "analyst": {
                    "target_mean_price": target,
                    "current_price": round(last, 2),
                    "implied_upside": round(target / last - 1.0, 4),
                    "recommendation_key": rec,
                    "num_analyst_opinions": int(rng.integers(8, 35)),
                    "source": "yfinance.Ticker.info (synthetic)",
                },
            }
        return {
            "task": "sentiment_portfolio_signal_generation",
            "as_of_date": as_of,
            "horizon": horizon,
            "portfolio_id": pid,
            "portfolio_universe": list(universe),
            "market_context": {
                "vix_level": macro["vix"],
                "fed_statement_tone": macro["fed_statement_tone"],
            },
            "stocks": stocks,
        }

    def _synth_news(self, t, rng, q, events) -> list:
        """A few AlphaVantage NEWS_SENTIMENT-style headlines with FinBERT-style scores."""
        sources = ["Reuters", "Bloomberg", "WSJ", "Barron's", "Seeking Alpha"]
        templates = []
        if "earnings_surprise" in events:
            templates.append((f"{t} tops quarterly earnings estimates", 0.55 + 0.3 * max(q, 0)))
        if "regulatory" in events:
            templates.append((f"{t} faces new regulatory scrutiny", -0.45))
        if "M&A" in events:
            templates.append((f"{t} explores strategic acquisition", 0.25))
        # filler headlines reflecting the stock's quality tilt
        base = [
            (f"Analysts weigh {t} growth outlook", 0.15 * q),
            (f"{t} guidance in focus for investors", 0.10 * q + 0.05),
            (f"Sector rotation puts {t} in spotlight", 0.05 * q),
        ]
        templates += base
        out = []
        for title, score in templates[:4]:
            out.append({
                "title": title,
                "source": sources[int(rng.integers(0, len(sources)))],
                "sentiment_score": round(float(np.clip(score + rng.normal(0, 0.08), -1, 1)), 3),
                "relevance": round(float(rng.uniform(0.4, 1.0)), 2),
            })
        return out

    # ---- economist package -------------------------------------------- #
    def _economist_package(self, pid, as_of, horizon, universe, sectors, macro) -> Dict:
        # three-axis regime taxonomy (rate/risk/cycle) derived from the macro snapshot
        rate, risk, cycle = _regime_axes(macro)
        # sector stances keyed off risk+cycle: a risk-on tilt when EITHER the cycle is
        # expanding or vol is calm; a defensive tilt when contracting or stressed.
        if cycle == "expansion" or risk == "calm":
            base = {"Energy": "overweight", "Financials": "overweight", "Industrials": "overweight",
                    "Consumer Staples": "underweight"}
        elif cycle == "contraction" or risk == "stressed":
            base = {"Consumer Staples": "overweight", "Health Care": "overweight",
                    "Financials": "underweight", "Energy": "underweight", "Industrials": "underweight"}
        else:
            base = {}
        sectors_in = sorted(set(sectors.values()))
        stances = {s: base.get(s, "neutral") for s in sectors_in}
        return {
            "task": "economist_portfolio_signal_generation",
            "as_of_date": as_of,
            "horizon": horizon,
            "portfolio_id": pid,
            "portfolio_universe": list(universe),
            "macro_snapshot": {k: macro[k] for k in macro if k not in ("macro_regime",)},
            "derived_regime": {"rate_environment": rate, "risk_regime": risk, "cycle": cycle},
            "sector_stances": stances,
            "universe_sector_mapping": sectors,
            "sectors_in_universe": sectors_in,
        }


def _regime_axes(macro: Dict):
    """Three-axis regime (rate/risk/cycle) from the macro snapshot -- same rule the
    economist mock uses, so the input package and the agent's output agree."""
    vix = float(macro.get("vix", 18.0)); gdp = float(macro.get("gdp_qoq", 2.0))
    curve = float(macro.get("yield_curve_10y2y", 0.5)); fed6m = float(macro.get("fed_funds_6m_change", 0.0))
    tone = macro.get("fed_statement_tone", "neutral")
    risk = "calm" if vix < 16 else ("stressed" if vix > 22 else "normal")
    rate = ("tightening" if (tone == "hawkish" or fed6m > 0.15)
            else ("easing" if (tone == "dovish" or fed6m < -0.15) else "stable"))
    cycle = "expansion" if (gdp >= 2.0 and curve > 0) else ("contraction" if (gdp < 0.5 or curve < 0) else "mid_cycle")
    return rate, risk, cycle


def _diff(a, b):
    if a is None or b is None:
        return None
    return round(float(a - b), 6)
