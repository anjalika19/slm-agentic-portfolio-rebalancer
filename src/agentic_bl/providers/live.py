"""LiveProvider -- wraps the user's existing fetch pipeline behind DataProvider.

Ports the scripts under ``live_agent_project`` / the drive-download folder
(``quant_common.py`` + ``02..05`` quant builders, ``economist.py``,
``preprocessor.py``, the sentiment builders). The quant leg runs from the bundled
Bloomberg-exported prices/benchmark plus Kenneth French FF5 data
(``price_data_wide.csv``, the daily FF5 ZIP/CSV, ``benchmark_data.csv``) and needs no API keys. The macro leg uses FRED.

Hard rule (spec 6): if a required key or file is missing, FAIL LOUDLY -- never
silently mock. The only escape hatch is the explicit, noisy opt-in
``LIVE_ALLOW_FIXTURE_FALLBACK=1`` for the fundamental/sentiment legs (which need
Bloomberg/news access); when set, those packages are filled from the fixture and a
warning is attached to the bundle so it is impossible to mistake for real data.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pickle
import re
from typing import Dict, List
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from agentic_bl import config
from agentic_bl.config import BENCHMARK, SECTOR_BY_TICKER, UNIVERSE_BBG
from agentic_bl.providers import quant_block
from agentic_bl.providers.base import DataProvider, InputBundle


def _num(x):
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        return float(x)
    except (TypeError, ValueError):
        return None



def _parse_published_at(value):
    """Parse provider-specific publication times into a naive UTC Timestamp."""
    if value is None or value == "":
        return None
    try:
        if isinstance(value, (int, float)):
            ts = pd.to_datetime(value, unit="s", utc=True, errors="coerce")
        else:
            text = str(value).strip()
            if re.fullmatch(r"\d{8}T\d{4,6}", text):
                fmt = "%Y%m%dT%H%M%S" if len(text) == 15 else "%Y%m%dT%H%M"
                ts = pd.to_datetime(text, format=fmt, utc=True, errors="coerce")
            else:
                ts = pd.to_datetime(text, utc=True, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.tz_convert(None) if getattr(ts, "tzinfo", None) is not None else ts
    except Exception:
        return None


def _normalise_title(value) -> str:
    text = str(value or "").lower().strip()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _load_module(name: str, path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"required live script not found: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cache_enabled() -> bool:
    return bool(getattr(config, "LIVE_DATA_CACHE", True))


def _cache_key(payload: Dict) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:24]


def _cache_path(namespace: str, key: str, suffix: str = ".json") -> str:
    base = getattr(config, "LIVE_CACHE_DIR", os.path.join(config.PROJECT_ROOT, ".cache", "live_data"))
    path = os.path.join(base, namespace)
    os.makedirs(path, exist_ok=True)
    return os.path.join(path, key + suffix)


def _read_json_cache(namespace: str, key_payload: Dict):
    if not _cache_enabled():
        return None
    path = _cache_path(namespace, _cache_key(key_payload), ".json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as fh:
            return json.load(fh)
    except Exception:
        return None


def _write_json_cache(namespace: str, key_payload: Dict, value) -> None:
    if not _cache_enabled():
        return
    path = _cache_path(namespace, _cache_key(key_payload), ".json")
    tmp = path + ".tmp"
    try:
        with open(tmp, "w") as fh:
            json.dump(value, fh, default=str)
        os.replace(tmp, path)
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


def _read_pickle_cache(namespace: str, key_payload: Dict):
    if not _cache_enabled():
        return None
    path = _cache_path(namespace, _cache_key(key_payload), ".pkl")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except Exception:
        return None


def _write_pickle_cache(namespace: str, key_payload: Dict, value) -> None:
    if not _cache_enabled():
        return
    path = _cache_path(namespace, _cache_key(key_payload), ".pkl")
    tmp = path + ".tmp"
    try:
        with open(tmp, "wb") as fh:
            pickle.dump(value, fh)
        os.replace(tmp, path)
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


class LiveProvider(DataProvider):
    name = "live"

    def __init__(self, scripts_dir: str | None = None):
        self.dir = scripts_dir or config.LIVE_SCRIPTS_DIR
        self.qdir = os.path.join(self.dir, "quant")
        self.qc = _load_module("quant_common_live", os.path.join(self.qdir, "quant_common.py"))
        self.price_path = os.path.join(self.qdir, "data", "price_data_wide.csv")
        self.factor_path = config.FACTOR_DATA_PATH
        self.bench_path = os.path.join(self.qdir, "data", "benchmark_data.csv")
        self.allow_fixture_fallback = config._env_bool("LIVE_ALLOW_FIXTURE_FALLBACK", False)
        self._prices_wide = None
        self._factors = None
        self._bench = None
        self._fund_ctx = None
        self._fund_mod = None
        self.workbook_path = os.path.join(self.dir, "fundamental", "data", "Book1.xlsx")

    # ---- lazy data loads ---------------------------------------------- #
    def _prices(self) -> pd.DataFrame:
        if self._prices_wide is None:
            df = self.qc.load_prices(self.price_path)
            wide = df.pivot_table(index="date", columns="ticker", values="PX_LAST").sort_index()
            self._prices_wide = wide
        return self._prices_wide

    def _download_benchmark_yfinance(self) -> pd.Series:
        """Download and cache S&P/benchmark prices from yfinance.

        Default is SPY adjusted close, which is a practical dividend-adjusted
        S&P 500 proxy. Configure with BENCHMARK_YF_TICKER.
        """
        try:
            import yfinance as yf
        except Exception as e:
            raise RuntimeError(f"benchmark download requires yfinance: {e}")

        ticker = config.BENCHMARK_YF_TICKER
        start = config.BENCHMARK_START_DATE

        # Pull through tomorrow so today's latest available close is included
        # when yfinance has it.
        end = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d")

        df = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )

        if df is None or df.empty:
            raise RuntimeError(f"yfinance returned no benchmark data for {ticker}")

        # yfinance may return either a normal column index or a MultiIndex.
        if isinstance(df.columns, pd.MultiIndex):
            if ("Close", ticker) in df.columns:
                px = df[("Close", ticker)]
            else:
                px = df.xs("Close", axis=1, level=0).iloc[:, 0]
        else:
            if "Close" not in df.columns:
                raise RuntimeError(
                    f"benchmark download for {ticker} did not include Close column"
                )
            px = df["Close"]

        px = pd.to_numeric(px, errors="coerce").dropna()
        px.index = pd.to_datetime(px.index).tz_localize(None)
        px = px.sort_index()

        if px.empty:
            raise RuntimeError(f"benchmark price series for {ticker} is empty after cleaning")

        # Cache it so future runs do not need to redownload unless the file is deleted.
        os.makedirs(os.path.dirname(self.bench_path), exist_ok=True)
        out = pd.DataFrame({
            "date": px.index,
            "benchmark_price": px.values,
        })
        out.to_csv(self.bench_path, index=False)

        return px.astype(float)

    def _benchmark(self) -> pd.Series:
        if self._bench is not None:
            return self._bench

        # Prefer the bundled/cache file. Only download when it is missing or unusable.
        if os.path.exists(self.bench_path):
            loaded = self.qc.load_benchmark(self.bench_path)
            if loaded is not None and not loaded.empty:
                series = loaded.set_index("date")["benchmark_price"].dropna().sort_index()
                if not series.empty:
                    self._bench = series.astype(float)
                    return self._bench

        self._bench = self._download_benchmark_yfinance()
        return self._bench

    def _factor_daily(self, universe_only: bool = True) -> pd.DataFrame:
        del universe_only  # retained for API compatibility
        if self._factors is None:
            if not os.path.exists(self.factor_path):
                raise FileNotFoundError(
                    "Kenneth French FF5 data not found at "
                    f"{self.factor_path}. Set FACTOR_DATA_PATH to the raw daily ZIP "
                    "or a normalized CSV with date,mkt_rf,smb,hml,rmw,cma,rf."
                )
            factors = self.qc.load_factors(self.factor_path).set_index("date").sort_index()
            required = [*config.FF5_FACTORS, config.RISK_FREE_FACTOR]
            missing = [column for column in required if column not in factors.columns]
            if missing:
                raise ValueError(
                    f"FF5 data at {self.factor_path} is missing columns: {missing}"
                )
            self._factors = factors[required].astype(float)
        return self._factors

    # ---- price panel for one as_of (universe short names + benchmark) -- #
    def _panel(self, universe: List[str], as_of: str) -> pd.DataFrame:
        ts = pd.Timestamp(as_of)
        wide = self._prices()
        wide = wide.loc[wide.index <= ts]
        cols = {}
        for t in universe:
            bbg = UNIVERSE_BBG.get(t, f"{t} US EQUITY")
            
            if bbg not in wide.columns:
                raise KeyError(f"ticker {bbg} not in price data ({self.price_path})")
            cols[t] = wide[bbg]
        panel = pd.DataFrame(cols)
        bench = self._benchmark()
        if bench is not None:
            panel[BENCHMARK] = bench.reindex(panel.index).ffill()
        else:
            # benchmark proxy: equal-weight universe price index (loudly noted later)
            panel[BENCHMARK] = panel[universe].pct_change().mean(axis=1).fillna(0).add(1).cumprod() * 100.0
        return panel.dropna(how="all")

    # ---- DataProvider API --------------------------------------------- #
    def get_input_bundle(
        self, portfolio_id: str, as_of_date: str, universe: List[str], horizon: str = "1M"
    ) -> InputBundle:
        warnings: List[str] = []
        panel = self._panel(universe, as_of_date)
        factor_daily = self._factor_daily().loc[lambda d: d.index <= pd.Timestamp(as_of_date)]
        if self._benchmark() is None:
            warnings.append("benchmark series unavailable; using equal-weight universe proxy")

        cov_block = quant_block.covariance_block(panel, universe)
        quant_pkg = quant_block.build_quant_package(
            portfolio_id, as_of_date, horizon, universe, panel, BENCHMARK, factor_daily,
            macro_regime="Live",
        )

        sectors = {t: SECTOR_BY_TICKER.get(t, "Unknown") for t in universe}
        packages = {"quant": quant_pkg}

        # economist via FRED (loud failure without key)
        packages["economist"] = self._economist_package(portfolio_id, as_of_date, horizon, universe, sectors, warnings)
        # sentiment via point-in-time news providers; analyst signal intentionally omitted
        packages["sentiment"] = self._sentiment_package(
            portfolio_id, as_of_date, horizon, universe, warnings
        )
        # fundamental from the Bloomberg workbook (Book1.xlsx); loud failure if absent
        packages["fundamental"] = self._fundamental_package(
            portfolio_id, as_of_date, horizon, universe, warnings
        )

        market_caps = self._market_cap_weights(universe, as_of_date, warnings)
        return InputBundle(
            portfolio_id=portfolio_id, as_of_date=as_of_date, horizon=horizon,
            universe=list(universe), sectors=sectors, packages=packages,
            covariance=cov_block, market_caps=market_caps, warnings=warnings,
        )

    def _economist_package(self, pid, as_of, horizon, universe, sectors, warnings) -> Dict:
        key = config.FRED_API_KEY or os.getenv("FRED_API_KEY")
        if not key:
            if self.allow_fixture_fallback:
                warnings.append("economist: NO FRED key -> fixture fallback (NOT real macro data)")
                from agentic_bl.providers.fixture import FixtureProvider

                return FixtureProvider().get_input_bundle(pid, as_of, universe, horizon).packages["economist"]
            raise RuntimeError(
                "LiveProvider economist requires a FRED API key (set FRED_API_KEY). "
                "Refusing to mock silently."
            )
        econ = _load_module("economist_live", os.path.join(self.dir, "economist", "economist.py"))
        as_of_ts = pd.Timestamp(as_of) + pd.offsets.MonthEnd(0)
        start = (as_of_ts - pd.DateOffset(months=24)).strftime("%Y-%m-%d")
        end = as_of_ts.strftime("%Y-%m-%d")
        macro_cache_key = {
            "kind": "fred_macro",
            "start": start,
            "end": end,
            "version": 1,
        }
        macro = _read_pickle_cache("fred_macro", macro_cache_key)
        if macro is not None:
            warnings.append(f"economist: using cached FRED macro data for {start} -> {end}")
        else:
            macro = econ.pull_fred_data(api_key=key, start=start, end=end)
            _write_pickle_cache("fred_macro", macro_cache_key, macro)
        valid = macro.index[macro.index <= as_of_ts]
        if len(valid) == 0:
            raise RuntimeError(f"no FRED data through {as_of_ts.date()}")
        row = macro.loc[valid[-1]]
        example = econ.build_example(valid[-1], row, macro["yield_curve_10y2y"].dropna())
        import json as _json

        payload = next(_json.loads(m["content"]) for m in example["messages"] if m["role"] == "user")
        return {
            **payload,
            "task": "economist_portfolio_signal_generation",
            "as_of_date": as_of, "horizon": horizon, "portfolio_id": pid,
            "portfolio_universe": list(universe),
            "universe_sector_mapping": sectors,
            "sectors_in_universe": sorted(set(sectors.values())),
            "macro_regime_label": payload.get("derived_regime", {}),
            "sector_stances": payload.get("sector_stances", {}),
        }

    def _sentiment_package(self, pid, as_of, horizon, universe, warnings) -> Dict:
        """Point-in-time sentiment package with provider reliability + relevance filtering.

        Cascade: AlphaVantage NEWS_SENTIMENT -> Finnhub company-news -> yfinance.news.
        Analyst recommendations are intentionally omitted until a true point-in-time source
        is added. AlphaVantage throttle payloads are HTTP 200 with a Note /
        Information key, so they are explicitly detected and not treated as zero news.
        """
        try:
            import yfinance as yf
        except Exception as e:
            raise RuntimeError(f"sentiment live path needs yfinance: {e}")

        as_of_ts = pd.Timestamp(as_of).normalize() + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        start_ts = as_of_ts.normalize() - pd.Timedelta(days=config.NEWS_LOOKBACK_DAYS)
        relevance_floor = float(config.SENTIMENT_RELEVANCE_FLOOR)
        stocks: Dict[str, Dict] = {}
        for idx, t in enumerate(universe):
            ticker_cache_key = {
                "kind": "sentiment_ticker",
                "ticker": t,
                "start": start_ts.strftime("%Y-%m-%d"),
                "end": as_of_ts.strftime("%Y-%m-%d"),
                "relevance_floor": relevance_floor,
                "max_articles": config.NEWS_MAX_ARTICLES_PER_TICKER,
                "version": 1,
            }
            cached_stock = _read_json_cache("sentiment_ticker", ticker_cache_key)
            if cached_stock is not None:
                stocks[t] = cached_stock
                warnings.append(f"sentiment {t}: using cached news package")
                continue

            raw_articles: List[Dict] = []
            provider = "none"
            provider_status = "no_provider_attempted"

            # --- 1) AlphaVantage NEWS_SENTIMENT, point-in-time ---
            av_key = config.ALPHAVANTAGE_API_KEY
            if av_key:
                raw_articles, provider_status = self._fetch_alphavantage_news(
                    t, av_key, start_ts, as_of_ts, warnings, sleep_first=(idx > 0)
                )
                if raw_articles:
                    provider = "AlphaVantage"

            # --- 2) Finnhub fallback, point-in-time ---
            if not raw_articles:
                fh_key = config.FINNHUB_API_KEY or os.getenv("FINNHUB_API_KEY", "")
                if fh_key:
                    raw_articles, provider_status = self._fetch_finnhub_news(t, fh_key, start_ts, as_of_ts, warnings)
                    if raw_articles:
                        provider = "Finnhub"

            # --- 3) yfinance fallback; use only items timestamped <= as_of when possible ---
            if not raw_articles:
                raw_articles, provider_status = self._fetch_yfinance_news(t, start_ts, as_of_ts, warnings)
                if raw_articles:
                    provider = "yfinance.news"

            filtered, weighted, drop_reasons = self._filter_relevant_articles(
                raw_articles,
                relevance_floor,
                requested_ticker=t,
                start_ts=start_ts,
                as_of_ts=as_of_ts,
            )
            # Provider fallbacks without per-ticker sentiment remain context-only and
            # carry a neutral numeric score rather than an invented analyst signal.
            news = []
            for art in filtered[: config.NEWS_MAX_ARTICLES_PER_TICKER]:
                score = _num(art.get("ticker_sentiment_score"))
                rel = _num(art.get("relevance_score"))
                news.append({
                    "title": art.get("title"),
                    "source": art.get("source"),
                    "sentiment_score": round(score or 0.0, 3),
                    "ticker_sentiment_score": round(score or 0.0, 3),
                    "relevance": round(rel if rel is not None else 1.0, 3),
                    "relevance_score": round(rel if rel is not None else 1.0, 3),
                    "time_published": art.get("time_published"),
                    "provider": art.get("provider", provider),
                })
            surviving = len(filtered)
            factual = round(float(weighted), 3) if surviving else 0.0
            stocks[t] = {
                "avg_factual_sentiment": factual,
                "avg_subjective_sentiment": factual,
                "relevance_weighted_sentiment": factual,
                "article_count": len(raw_articles),
                "raw_article_count": len(raw_articles),
                "surviving_article_count": surviving,
                "dropped_article_count": max(0, len(raw_articles) - surviving),
                "relevance_floor": relevance_floor,
                "news_provider": provider,
                "provider_status": provider_status,
                "event_flags": [],
                "analyst_upgrades": 0,
                "analyst_downgrades": 0,
                "news": news,
                "data_quality": {
                    "point_in_time": True,
                    "requested_ticker": t,
                    "window_start": start_ts.strftime("%Y-%m-%d"),
                    "window_end": as_of_ts.strftime("%Y-%m-%d"),
                    "raw_count": len(raw_articles),
                    "accepted_count": surviving,
                    "drop_reasons": drop_reasons,
                    "has_sentiment_signal": surviving > 0,
                    "analyst_signal_included": False,
                },
            }
            _write_json_cache("sentiment_ticker", ticker_cache_key, stocks[t])
            warnings.append(
                f"sentiment {t}: provider={provider}, status={provider_status}, "
                f"raw_articles={len(raw_articles)}, surviving={surviving}, floor={relevance_floor}"
            )
        return {
            "task": "sentiment_portfolio_signal_generation",
            "as_of_date": as_of, "horizon": horizon, "portfolio_id": pid,
            "portfolio_universe": list(universe),
            "market_context": {
                "source": "AlphaVantage -> Finnhub -> yfinance.news",
                "analyst_signal": "omitted_pending_point_in_time_source",
            },
            "stocks": stocks,
        }

    def _fetch_alphavantage_news(self, ticker: str, api_key: str, start_ts, as_of_ts, warnings, sleep_first=False):
        import time
        import requests

        if sleep_first and config.ALPHAVANTAGE_MIN_SECONDS > 0:
            time.sleep(float(config.ALPHAVANTAGE_MIN_SECONDS))
        time_to = as_of_ts.strftime("%Y%m%dT2359")
        time_from = start_ts.strftime("%Y%m%dT0000")
        last_status = "not_called"
        for attempt in range(2):
            try:
                r = requests.get(
                    "https://www.alphavantage.co/query",
                    params={
                        "function": "NEWS_SENTIMENT", "tickers": ticker, "apikey": api_key,
                        "time_from": time_from, "time_to": time_to, "limit": 50, "sort": "RELEVANCE",
                    },
                    timeout=30,
                )
                payload = r.json() if r.ok else {}
            except Exception as e:  # noqa: BLE001
                last_status = f"alphavantage_error:{e}"
                break
            throttle_msg = payload.get("Note") or payload.get("Information")
            if throttle_msg:
                last_status = "alphavantage_throttled"
                warnings.append(f"sentiment {ticker}: AlphaVantage throttled: {str(throttle_msg)[:120]}")
                if attempt == 0:
                    time.sleep(float(config.ALPHAVANTAGE_MIN_SECONDS))
                    continue
                return [], last_status
            feed = payload.get("feed") or []
            articles = []
            for art in feed:
                ts = art.get("ticker_sentiment") or []
                row = next((x for x in ts if x.get("ticker") == ticker), None)
                if row is None:
                    continue
                score = _num(row.get("ticker_sentiment_score"))
                rel = _num(row.get("relevance_score"))
                if score is None or rel is None:
                    continue
                articles.append({
                    "title": art.get("title"), "source": art.get("source"),
                    "ticker_sentiment_score": score, "relevance_score": rel,
                    "time_published": art.get("time_published"), "provider": "AlphaVantage",
                    "ticker": ticker,
                })
            return articles, "ok" if articles else "ok_zero_feed"
        return [], last_status

    def _fetch_finnhub_news(self, ticker: str, api_key: str, start_ts, as_of_ts, warnings):
        import requests

        try:
            r = requests.get(
                "https://finnhub.io/api/v1/company-news",
                params={
                    "symbol": ticker,
                    "from": start_ts.strftime("%Y-%m-%d"),
                    "to": as_of_ts.strftime("%Y-%m-%d"),
                    "token": api_key,
                },
                timeout=30,
            )
            data = r.json() if r.ok else []
        except Exception as e:  # noqa: BLE001
            warnings.append(f"sentiment {ticker}: Finnhub fetch failed: {e}")
            return [], f"finnhub_error:{e}"
        if isinstance(data, dict) and data.get("error"):
            warnings.append(f"sentiment {ticker}: Finnhub error: {data.get('error')}")
            return [], "finnhub_error"
        articles = []
        for art in data or []:
            # Finnhub lacks per-ticker sentiment in this endpoint; score neutral and
            # retain as fallback context only. relevance=1 because company-news is already ticker scoped.
            articles.append({
                "title": art.get("headline"), "source": art.get("source"),
                "ticker_sentiment_score": 0.0, "relevance_score": 1.0,
                "time_published": art.get("datetime"), "provider": "Finnhub",
                "ticker": ticker,
            })
        return articles, "ok" if articles else "ok_zero_feed"

    def _fetch_yfinance_news(self, ticker: str, start_ts, as_of_ts, warnings):
        try:
            import yfinance as yf
            items = yf.Ticker(ticker).news or []
        except Exception as e:  # noqa: BLE001
            warnings.append(f"sentiment {ticker}: yfinance.news failed: {e}")
            return [], f"yfinance_news_error:{e}"
        articles = []
        for art in items:
            content = art.get("content") if isinstance(art.get("content"), dict) else {}
            ts = (
                art.get("providerPublishTime")
                or art.get("pubDate")
                or content.get("pubDate")
                or content.get("displayTime")
            )
            published = _parse_published_at(ts)
            if published is None or not (start_ts <= published <= as_of_ts):
                continue
            time_published = published.strftime("%Y%m%dT%H%M%S")
            title = art.get("title") or content.get("title")
            source = art.get("publisher") or content.get("provider", {}).get("displayName")
            articles.append({
                "title": title, "source": source,
                "ticker_sentiment_score": 0.0, "relevance_score": 1.0,
                "time_published": time_published, "provider": "yfinance.news",
                "ticker": ticker,
            })
        return articles, "ok" if articles else "ok_zero_feed"

    def _filter_relevant_articles(
        self,
        articles: List[Dict],
        floor: float,
        requested_ticker: str,
        start_ts: pd.Timestamp,
        as_of_ts: pd.Timestamp,
    ):
        """Enforce point-in-time, ticker-specific, relevance, and dedupe rules."""
        filtered = []
        seen_titles = set()
        drop_reasons = {
            "missing_title": 0,
            "missing_or_invalid_timestamp": 0,
            "outside_window": 0,
            "low_relevance": 0,
            "ticker_mismatch": 0,
            "duplicate": 0,
        }
        requested = requested_ticker.upper()

        for art in articles or []:
            title = str(art.get("title") or "").strip()
            title_key = _normalise_title(title)
            if not title_key:
                drop_reasons["missing_title"] += 1
                continue

            published = _parse_published_at(art.get("time_published"))
            if published is None:
                drop_reasons["missing_or_invalid_timestamp"] += 1
                continue
            if not (start_ts <= published <= as_of_ts):
                drop_reasons["outside_window"] += 1
                continue

            rel = _num(art.get("relevance_score"))
            if rel is None or rel < floor:
                drop_reasons["low_relevance"] += 1
                continue

            article_ticker = str(art.get("ticker") or "").upper()
            if article_ticker != requested:
                drop_reasons["ticker_mismatch"] += 1
                continue

            if title_key in seen_titles:
                drop_reasons["duplicate"] += 1
                continue
            seen_titles.add(title_key)

            filtered.append({
                **art,
                "relevance_score": rel,
                "time_published": published.strftime("%Y%m%dT%H%M%S"),
            })

        filtered.sort(
            key=lambda a: (
                _num(a.get("relevance_score")) or 0.0,
                _parse_published_at(a.get("time_published")) or pd.Timestamp.min,
            ),
            reverse=True,
        )
        denom = sum((_num(a.get("relevance_score")) or 0.0) for a in filtered)
        if not filtered or denom <= 0:
            return filtered, 0.0, drop_reasons
        weighted = sum(
            (_num(a.get("ticker_sentiment_score")) or 0.0)
            * (_num(a.get("relevance_score")) or 0.0)
            for a in filtered
        ) / denom
        return filtered, weighted, drop_reasons



    def _market_cap_weights(self, universe: List[str], as_of: str, warnings: List[str]) -> Dict[str, float]:
        """Cap weights for the BL equilibrium prior: shares_outstanding(as_of) x price(as_of),
        normalized to sum to 1.

        PiT discipline: we use HISTORICAL price (already PiT-correct via _prices()) but
        shares outstanding is harder to get historically without a paid data source.
        Fallback chain, each step logged:
          1. yfinance Ticker.get_shares_full(start, end) -- point-in-time shares from
             SEC filing dates, when available.
          2. yfinance Ticker.info['sharesOutstanding'] -- CURRENT shares applied to the
             HISTORICAL price. Shares outstanding is slow-moving for large-caps (buybacks
             change it by low single-digit % per year), so this is a small, documented bias
             -- not a structural one. Logged explicitly per ticker.
          3. Equal weight 1/N -- last resort, only if no shares data at all. Logged as a
             warning so it's visible in the run's provider_warnings.
        """
        cache_key = {
            "kind": "market_caps",
            "universe": list(universe),
            "as_of": as_of,
            "version": 1,
        }
        cached = _read_json_cache("market_caps", cache_key)
        if cached is not None:
            warnings.append(f"market_caps: using cached weights for {as_of}")
            return {str(k): float(v) for k, v in cached.items()}

        try:
            import yfinance as yf
        except Exception as e:
            warnings.append(f"market_caps: yfinance unavailable ({e}); using equal weight 1/N")
            n = len(universe)
            weights = {t: 1.0 / n for t in universe}
            return weights

        prices = self._prices()
        as_of_ts = pd.Timestamp(as_of)
        caps: Dict[str, float] = {}
        for t in universe:
            bb = config.UNIVERSE_BBG.get(t, t) if hasattr(config, "UNIVERSE_BBG") else t
            col = bb if bb in prices.columns else t
            try:
                price_row = prices.loc[:as_of_ts, col].dropna()
                price = float(price_row.iloc[-1]) if len(price_row) else None
            except Exception:
                price = None
            if price is None or price <= 0:
                warnings.append(f"market_caps {t}: no historical price found; excluded from cap calc")
                continue

            shares = None
            source = "none"
            try:
                tk = yf.Ticker(t)
                hist_shares = tk.get_shares_full(start=as_of_ts - pd.Timedelta(days=400), end=as_of)
                if hist_shares is not None and len(hist_shares):
                    s = hist_shares[hist_shares.index <= as_of_ts]
                    if len(s):
                        shares = float(s.iloc[-1])
                        source = "yfinance_shares_full_pit"
            except Exception:
                pass
            if shares is None:
                try:
                    info_shares = tk.info.get("sharesOutstanding")
                    if info_shares:
                        shares = float(info_shares)
                        source = "yfinance_current_shares_applied_to_historical_price"
                except Exception:
                    pass

            if shares is None or shares <= 0:
                warnings.append(f"market_caps {t}: no shares-outstanding data; excluded from cap calc")
                continue
            caps[t] = price * shares
            warnings.append(f"market_caps {t}: source={source}, price={price:.2f}, shares={shares:,.0f}")

        if not caps or len(caps) < len(universe):
            missing = [t for t in universe if t not in caps]
            if missing:
                warnings.append(
                    f"market_caps: missing data for {missing}; falling back to equal weight 1/N for ALL tickers "
                    f"(partial cap weighting would distort the cross-sectional comparison)"
                )
            n = len(universe)
            weights = {t: 1.0 / n for t in universe}
            return weights

        total = sum(caps.values())
        weights = {t: caps[t] / total for t in universe}
        _write_json_cache("market_caps", cache_key, weights)
        return weights

    def _fundamental_context(self):
        if self._fund_ctx is None:
            if not os.path.exists(self.workbook_path):
                raise FileNotFoundError(f"fundamental workbook not found: {self.workbook_path}")
            self._fund_mod = _load_module(
                "fundamental_prep", os.path.join(self.dir, "fundamental", "preprocessor.py")
            )
            self._fund_ctx = self._fund_mod.build_context(self.workbook_path)
        return self._fund_mod, self._fund_ctx

    def _fundamental_package(self, pid, as_of, horizon, universe, warnings) -> Dict:
        """Real fundamentals from the Bloomberg workbook (Book1.xlsx), point-in-time
        (gated on EARN_ANN_DT), with sector-percentile ranks vs the workbook's actual
        sector peers. Loud failure if the workbook is missing."""
        if not os.path.exists(self.workbook_path):
            if self.allow_fixture_fallback:
                warnings.append("fundamental: NO Book1.xlsx -> fixture fallback (NOT real)")
                from agentic_bl.providers.fixture import FixtureProvider

                return FixtureProvider().get_input_bundle(pid, as_of, universe, horizon).packages["fundamental"]
            raise RuntimeError(
                f"LiveProvider fundamental requires the Bloomberg workbook at {self.workbook_path}. "
                "Refusing to mock silently."
            )

        cache_key = {
            "kind": "fundamental_package",
            "as_of": as_of,
            "horizon": horizon,
            "universe": list(universe),
            "workbook_mtime": os.path.getmtime(self.workbook_path),
            "version": 1,
        }
        cached = _read_json_cache("fundamental_package", cache_key)
        if cached is not None:
            cached["portfolio_id"] = pid
            warnings.append(f"fundamental: using cached package for {as_of}")
            return cached

        prep, ctx = self._fundamental_context()
        ts = pd.Timestamp(as_of)
        ticker_sector = ctx["ticker_sector"]
        bbg_for = {t: UNIVERSE_BBG.get(t, f"{t} US EQUITY") for t in universe}

        # build records for the universe tickers + all their real sector peers
        sectors_needed = {ticker_sector.get(bbg_for[t]) for t in universe}
        peers = [bb for bb, sec in ticker_sector.items() if sec in sectors_needed]
        rec = {}
        for bb in peers:
            try:
                rec[bb] = prep.build_stock_quarter_record(ctx, bb, ts)
            except Exception as e:  # a single bad peer must not sink the sector
                warnings.append(f"fundamental: record failed for {bb}: {e}")

        def g(r, *path):
            cur = r
            for k in path:
                cur = (cur or {}).get(k) if isinstance(cur, dict) else None
            return _num(cur)

        def raw_of(bb):
            r = rec.get(bb, {})
            return {
                "pe_ratio": g(r, "valuation", "pe_trailing", "value"),
                "pb_ratio": g(r, "valuation", "pb", "value"),
                "roe": g(r, "profitability", "roe", "value"),
                "debt_to_equity": g(r, "health", "debt_to_equity", "value"),
                "eps_growth_ttm": g(r, "growth", "eps_growth_yoy", "value"),
                "eps_growth_3y_cagr": g(r, "growth", "eps_3y_cagr", "value"),
                "revenue_growth": g(r, "growth", "rev_growth_yoy", "value"),
                "gross_margin_trend": g(r, "profitability", "gross_margin", "trend_yoy_pp"),
                "current_ratio": g(r, "health", "current_ratio", "value"),
                "best_eps": g(r, "estimates", "best_eps_fwd"),
                "best_target_price": g(r, "estimates", "target_price"),
                "implied_upside_pct": g(r, "estimates", "implied_upside_pct"),
                "earn_ann_dt": r.get("earn_ann_dt") or r.get("meta", {}).get("earn_ann_dt"),

            }

        raws = {bb: raw_of(bb) for bb in rec}
        # percentile direction: lower-is-better for valuation/leverage, else higher
        lower_better = {"pe_ratio", "pb_ratio", "debt_to_equity"}
        metrics = ["pe_ratio", "pb_ratio", "roe", "debt_to_equity", "eps_growth_ttm",
                   "eps_growth_3y_cagr", "revenue_growth", "gross_margin_trend", "current_ratio"]

        stocks = {}
        for t in universe:
            bb = bbg_for[t]
            sec = ticker_sector.get(bb, "Unknown")
            peer_bbs = [p for p in raws if ticker_sector.get(p) == sec]
            ranks = {}
            for m in metrics:
                series = pd.Series({p: raws[p][m] for p in peer_bbs}, dtype="float64")
                pr = series.rank(ascending=(m not in lower_better), pct=True)
                ranks[m] = round(float(pr.get(bb)), 3) if pd.notna(pr.get(bb)) else 0.5
            up = raws[bb]["implied_upside_pct"]
            stocks[t] = {
                "sector": sec,
                "raw_ratios": {k: raws[bb][k] for k in metrics + ["best_eps", "best_target_price"]},
                 "earn_ann_dt": raws.get(bb, {}).get("earn_ann_dt"),  
                "sector_percentile_ranks": ranks,
                "consensus": {
                    "best_eps": raws[bb]["best_eps"],
                    "best_target_price": raws[bb]["best_target_price"],
                    "implied_upside": round(up / 100.0, 4) if up is not None else None,
                },
                "data_source": "Bloomberg Book1.xlsx (PiT via EARN_ANN_DT)",
                "n_sector_peers": len(peer_bbs),
            }
        package = {
            "task": "fundamental_portfolio_signal_generation",
            "as_of_date": as_of, "horizon": horizon, "portfolio_id": pid,
            "portfolio_universe": list(universe), "stocks": stocks,
        }
        _write_json_cache("fundamental_package", cache_key, package)
        return package

    def _needs_external(self, agent, pid, as_of, horizon, universe, warnings) -> Dict:
        if self.allow_fixture_fallback:
            warnings.append(f"{agent}: external source unavailable -> fixture fallback (NOT real data)")
            from agentic_bl.providers.fixture import FixtureProvider

            return FixtureProvider().get_input_bundle(pid, as_of, universe, horizon).packages[agent]
        raise RuntimeError(
            f"LiveProvider {agent} requires its data source (Bloomberg workbook / news API). "
            f"Set LIVE_ALLOW_FIXTURE_FALLBACK=1 to fill {agent} from the fixture (clearly flagged), "
            f"or wire the real source. Refusing to mock silently."
        )

    # In LiveProvider:
    def forward_return(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> float:
        prices = self._prices()

        candidates = [
            UNIVERSE_BBG.get(ticker),
            f"{ticker} US EQUITY",
            ticker,
        ]

        column = next(
            (
                candidate
                for candidate in candidates
                if candidate and candidate in prices.columns
            ),
            None,
        )

        if column is None:
            raise ValueError(
                f"forward_return: {ticker} not in price data; "
                f"checked {candidates}. "
                f"Example available columns: {list(prices.columns[:10])}"
            )

        series = prices[column].dropna().sort_index()

        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        start_candidates = series.index[
            (series.index >= start)
            & (series.index <= start + pd.Timedelta(days=7))
        ]

        end_candidates = series.index[
            (series.index <= end)
            & (series.index >= end - pd.Timedelta(days=7))
        ]

        if len(start_candidates) == 0:
            raise ValueError(
                f"forward_return: no {ticker} price within "
                f"7 days after {start_date}"
            )

        if len(end_candidates) == 0:
            raise ValueError(
                f"forward_return: incomplete holding period for {ticker}; "
                f"no price within 7 days before {end_date}. "
                "Extend price data or end the backtest one rebalance earlier."
            )

        start_index = start_candidates[0]
        end_index = end_candidates[-1]

        start_price = float(series.loc[start_index])
        end_price = float(series.loc[end_index])

        if start_price == 0:
            raise ValueError(
                f"forward_return: zero start price for {ticker}"
            )

        return float(end_price / start_price - 1.0)
    def risk_free_return(self, start_date: str, end_date: str) -> float:
        factors = self._factor_daily()
        rf = factors[config.RISK_FREE_FACTOR].dropna().sort_index()
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        if rf.empty or rf.index.max() < end - pd.Timedelta(days=7):
            raise ValueError(
                f"risk_free_return: FF5/RF data does not cover holding end {end_date}; "
                f"latest available date is {rf.index.max() if not rf.empty else 'none'}"
            )
        holding = rf.loc[(rf.index > start) & (rf.index <= end)]
        if holding.empty:
            return 0.0
        return float((1.0 + holding).prod() - 1.0)

    def benchmark_forward_return(self, start_date: str, end_date: str) -> float:
        bench = self._benchmark()

        if bench is None or bench.empty:
            raise ValueError(
                "benchmark_forward_return: benchmark series unavailable. "
                "Could not load cached benchmark_data.csv or download from yfinance."
            )

        s = pd.Timestamp(start_date)
        e = pd.Timestamp(end_date)

        idx = bench.index.sort_values()

        if not any(idx >= s):
            raise ValueError(f"benchmark_forward_return: no benchmark data on/after {start_date}")
        if not any(idx <= e):
            raise ValueError(f"benchmark_forward_return: no benchmark data on/before {end_date}")

        s_idx = idx[idx >= s][0]
        e_idx = idx[idx <= e][-1]

        p_start = bench.loc[s_idx]
        p_end = bench.loc[e_idx]

        if pd.isna(p_start) or pd.isna(p_end) or p_start == 0:
            return 0.0

        return float((p_end - p_start) / p_start)