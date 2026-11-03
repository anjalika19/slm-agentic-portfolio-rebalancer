"""Shared single-model control for Gemma (SLM) and Gemini (LLM)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import random
import numpy as np

import bl_core
import calibration
import config
from agents import mock, payloads
from agents.llm import structured_call
from agents.normalize import normalize
from agents.prompts.gemini import SINGLE_GEMINI_SYSTEM, SINGLE_MODEL_SYSTEM
from events import EventEmitter
from providers.base import DataProvider, InputBundle
from schemas import GeminiSingleModelEmission, SIGNAL5_VALUE, SingleModelEmission


def _direction_from_signal_value(v: float) -> str:
    if abs(v) < 0.20:
        return "neutral"
    return "long" if v > 0 else "short"


def _direction_from_q(q: float) -> str:
    if abs(q) < config.VIEW_SPREAD_FLOOR:
        return "neutral"
    return "long" if q > 0 else "short"


def _label_from_value(v: float) -> str:
    if v <= -0.75:
        return "avoid"
    if v <= -0.20:
        return "underweight"
    if v >= 0.75:
        return "strongly_long"
    if v >= 0.20:
        return "long"
    return "neutral"


def _numeric_cited_values(cited: List[Dict[str, Any]], limit: int = 8) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    seen = set()
    for c in cited or []:
        name = str(c.get("name", "")).strip()
        if not name or name in seen:
            continue
        try:
            val = float(c.get("value"))
        except (TypeError, ValueError):
            continue
        out.append({"name": name, "value": val})
        seen.add(name)
        if len(out) >= limit:
            break
    return out


def _single_tag(model_family: str) -> str:
    return f"single_{config.normalize_model_family(model_family)}"


def build_single_model_payload(bundle: InputBundle, universe: List[str]) -> Dict[str, Any]:
    """Build the consolidated per-stock package for the one-shot analyst."""
    compact = {
        agent: payloads.compact_package(agent, bundle.packages[agent], as_of_date=bundle.as_of_date)
        for agent in config.SPECIALISTS
        if agent in bundle.packages
    }
    econ = compact.get("economist", {})
    stock_packages: Dict[str, Dict[str, Any]] = {}
    for t in universe:
        sector = (econ.get("universe_sector_mapping") or {}).get(t) or bundle.sectors.get(t)
        stock_packages[t] = {
            "fundamental": (compact.get("fundamental", {}).get("stocks") or {}).get(t, {}),
            "quant": (compact.get("quant", {}).get("stocks") or {}).get(t, {}),
            "news_sentiment": (compact.get("sentiment", {}).get("stocks") or {}).get(t, {}),
            "economist_macro": {
                "sector": sector,
                "sector_stance": (econ.get("sector_stances") or {}).get(sector, "neutral"),
                "derived_regime": econ.get("derived_regime"),
                "macro_snapshot": econ.get("macro_snapshot", {}),
                "macro_context": econ.get("macro_context", {}),
            },
        }
    return {
        "portfolio_context": {
            "portfolio_id": bundle.portfolio_id,
            "as_of_date": bundle.as_of_date,
            "horizon": bundle.horizon,
            "portfolio_universe": list(universe),
            "benchmark": config.BENCHMARK,
        },
        "macro_package": {
            "derived_regime": econ.get("derived_regime"),
            "macro_snapshot": econ.get("macro_snapshot", {}),
            "macro_context": econ.get("macro_context", {}),
            "sector_stances": econ.get("sector_stances", {}),
        },
        "stock_packages": stock_packages,
    }


# Backward-compatible public name.
build_single_gemini_payload = build_single_model_payload


def _hallucination_rate(emission: Dict[str, Any], payload: Dict[str, Any], tol: float = None) -> float:
    """Fraction of cited values that do not trace to any numeric input value."""
    tol = config.HALLUCINATION_TOL if tol is None else tol
    flat: List[float] = []

    def _walk(o: Any) -> None:
        if isinstance(o, dict):
            for v in o.values():
                _walk(v)
        elif isinstance(o, list):
            for v in o:
                _walk(v)
        elif isinstance(o, (int, float)) and not isinstance(o, bool):
            flat.append(float(o))

    _walk(payload)
    cited_blocks: List[List[Dict[str, Any]]] = []
    cited_blocks.extend(sig.get("cited_values", []) for sig in emission.get("signals", {}).values())
    cited_blocks.extend(view.get("cited_values", []) for view in emission.get("pairwise_views", []))

    total = miss = 0
    for cited_values in cited_blocks:
        for cv in cited_values:
            val = cv.get("value")
            if val is None:
                continue
            total += 1
            try:
                fval = float(val)
            except (TypeError, ValueError):
                miss += 1
                continue
            if not any(abs(fval - x) <= tol * (abs(x) + 1e-9) for x in flat):
                miss += 1
    return round(miss / total, 4) if total else 0.0


def _collect_mock_scores(bundle: InputBundle, universe: List[str]) -> Tuple[Dict[str, Dict[str, Any]], str]:
    outs = {a: mock.mock_specialist_output(a, bundle.packages[a]) for a in config.SPECIALISTS}
    norm = {a: normalize(a, outs[a], universe, bundle.sectors) for a in config.SPECIALISTS}
    econ_view = norm.get("economist", {}).get("agent_view", {})
    macro_regime = "/".join(
        str(x or "")
        for x in [econ_view.get("rate_environment"), econ_view.get("risk_regime"), econ_view.get("cycle")]
        if x
    ) or "unknown"
    scores: Dict[str, Dict[str, Any]] = {}
    for t in universe:
        vals = [float(norm[a]["signals"].get(t, {}).get("signal", 0.0)) for a in config.SPECIALISTS]
        confs = [float(norm[a]["signals"].get(t, {}).get("confidence", 0.5)) for a in config.SPECIALISTS]
        avg = float(np.mean(vals))
        disagreement = float(np.std(vals))
        conf = round(float(np.clip(np.mean(confs) * (1.0 - min(disagreement, 1.0)), 0.05, 0.9)), 3)
        cited: List[Dict[str, Any]] = []
        for a in config.SPECIALISTS:
            for c in norm[a]["signals"].get(t, {}).get("cited_values", [])[:3]:
                cited.append({"name": f"{a}.{c.get('name')}", "value": c.get("value")})
        scores[t] = {
            "avg": avg,
            "disagreement": disagreement,
            "confidence": conf,
            "cited_values": _numeric_cited_values(cited, limit=10),
        }
    return scores, macro_regime


def mock_single_model_emission(
    bundle: InputBundle,
    universe: List[str],
    *,
    model_family: str = "gemini",
) -> Dict[str, Any]:
    """Deterministic C1 fallback: blend mocks into stock-level views."""
    scores, macro_regime = _collect_mock_scores(bundle, universe)
    signals: Dict[str, Dict[str, Any]] = {}
    for t in universe:
        avg = float(scores[t]["avg"])
        label = _label_from_value(avg)
        signals[t] = {
            "signal": label,
            # Gemini-Single's real schema emits q directly. Keep mock fallback
            # compatible without reintroducing the five-bucket q mapping: the
            # mock uses the underlying continuous blended score.
            **({"q": round(calibration.ticker_signal_to_raw_q(avg), 6)} if config.normalize_model_family(model_family) == "gemini" else {}),
            "confidence": scores[t]["confidence"],
            "rationale": (
                f"[mock C1] Blended one-shot read for {t}: specialist signals average {avg:+.2f} "
                f"with disagreement {scores[t]['disagreement']:.2f}. Pairwise BL rows are derived later "
                "from calibrated stock-level views."
            ),
            "cited_values": scores[t]["cited_values"],
        }
    return {
        "agent_name": _single_tag(model_family),
        "as_of_date": bundle.as_of_date,
        "horizon": bundle.horizon,
        "portfolio_id": bundle.portfolio_id,
        "macro_regime": macro_regime,
        "signals": signals,
        "pairwise_views": [],
        "summary": "[mock] C1 single-model control emitted stock-level views; Python builds pairwise BL rows.",
    }


def _call_single_model(
    bundle: InputBundle,
    universe: List[str],
    *,
    strict: Optional[bool],
    model_family: str,
    model_name: Optional[str] = None,
) -> tuple[Dict[str, Any], bool, str, Dict[str, Any]]:
    model_family = config.normalize_model_family(model_family)
    tag = _single_tag(model_family)
    payload = build_single_model_payload(bundle, universe)
    # Gemini-Single emits its own continuous stock-level q. Gemma-Single keeps
    # the legacy categorical schema so this change does not alter the SLM control.
    schema_model = GeminiSingleModelEmission if model_family == "gemini" else SingleModelEmission
    system_prompt = SINGLE_GEMINI_SYSTEM if model_family == "gemini" else SINGLE_MODEL_SYSTEM
    res = structured_call(
        role=tag,
        system_prompt=system_prompt,
        user_payload=payload,
        schema_model=schema_model,
        temperature=config.temperature_for_call(tag),
        mock_fn=lambda: mock_single_model_emission(
            bundle, universe, model_family=model_family
        ),
        strict=strict,
        model_family=model_family,
        model_name=model_name,
    )
    data = schema_model.model_validate(res.data).model_dump()
    data["agent_name"] = tag
    # Fairness guard: ignore any direct pairwise views the LLM emitted despite the prompt.
    data["pairwise_views"] = []
    return data, res.used_mock, res.note, payload


def _fill_missing_neutrals(
    data: Dict[str, Any],
    universe: List[str],
    emitter: Optional[EventEmitter],
    *,
    tag: str,
    model_family: str,
) -> Dict[str, Any]:
    missing = [t for t in universe if t not in data.get("signals", {})]
    for t in missing:
        data.setdefault("signals", {})[t] = {
            # q=0 is an explicit abstention for the continuous Gemini schema.
            **({"q": 0.0} if config.normalize_model_family(model_family) == "gemini" else {"signal": "neutral"}),
            "confidence": 0.05,
            "rationale": f"[abstain] {tag} did not provide a ticker signal for this name.",
            "cited_values": [],
        }
        if emitter is not None:
            emitter.warning(
                f"{tag}: no ticker signal for {t}; filled explicit neutral abstain",
                phase="single_model",
                agent=tag,
                ticker=t,
            )
    return data


def _ticker_views_from_single_data(
    data: Dict[str, Any],
    universe: List[str],
    view_calibrator: calibration.ViewCalibrator,
    *,
    model_family: str,
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for t in universe:
        s = data.get("signals", {}).get(t, {})
        raw_conf = float(np.clip(float(s.get("confidence", 0.25) or 0.25), 0.05, 0.95))
        if model_family == "gemini":
            # Gemini-Single owns the continuous stock-level q and confidence.
            # Do not run these new semantics through the legacy 2024
            # gemini_stock_signal calibration, which was fitted to the old
            # categorical +/-0.12, +/-0.06, 0 representation. Apply only the
            # common pre-specified q multiplier used by the experiment.
            raw_q = float(s.get("q", 0.0) or 0.0)
            q = float(raw_q * config.DEFAULT_Q_MULTIPLIER)
            confidence = raw_conf
            label = "neutral" if abs(raw_q) < config.VIEW_SPREAD_FLOOR else ("long" if raw_q > 0 else "underweight")
            # Retain a bounded diagnostic signal for legacy event consumers only.
            # It does not feed q or portfolio construction.
            scale = max(abs(float(config.ANNUAL_RETURN_SCALE)), 1e-12)
            signal_value = float(np.clip(raw_q / scale, -1.0, 1.0))
            source = "single_gemini_direct_stock_q_fixed_scaled"
            calibration_meta = {
                "family": "gemini_continuous_stock_q",
                "policy": "direct_model_q_fixed_multiplier",
                "q_multiplier": round(float(config.DEFAULT_Q_MULTIPLIER), 6),
                "confidence_policy": "model_raw_then_shared_pairwise_clip",
                "confidence_floor": round(float(config.MIN_VIEW_CONFIDENCE), 6),
                "confidence_cap": round(float(config.MAX_VIEW_CONFIDENCE), 6),
                "legacy_calibration_ignored": True,
            }
        else:
            label = s.get("signal", "neutral")
            signal_value = float(SIGNAL5_VALUE.get(label, 0.0))
            raw_q = calibration.ticker_signal_to_raw_q(signal_value)
            cal = view_calibrator.calibrate(
                raw_q, raw_conf, family=f"{model_family}_stock_signal"
            )
            q = float(cal.q)
            confidence = float(cal.confidence)
            raw_q = float(cal.raw_q)
            raw_conf = float(cal.raw_confidence)
            source = f"single_{model_family}_stock_signal_calibrated"
            calibration_meta = cal.as_metadata()

        out[t] = {
            "ticker": t,
            "direction": _direction_from_q(q),
            "q": round(float(q), 6),
            "raw_q": round(float(raw_q), 6),
            "confidence": round(float(confidence), 4),
            "raw_confidence": round(float(raw_conf), 4),
            "calibration": calibration_meta,
            "signal": round(signal_value, 4),
            "signal_label": label,
            "rationale": s.get("rationale", ""),
            "cited_values": s.get("cited_values", []),
            "macro_regime": data.get("macro_regime", ""),
            "source": source,
            "resolved": True,
        }
    return out


def _emit_single_model_views(
    emitter: EventEmitter,
    *,
    data: Dict[str, Any],
    universe: List[str],
    ticker_views: Dict[str, Dict[str, Any]],
    pairwise_views: List[Dict[str, Any]],
    used_mock: bool,
    note: str,
    feeds_black_litterman: bool,
    model_family: str,
    model_name: str,
) -> None:
    tag = _single_tag(model_family)
    emitter.emit(
        "single_model_result",
        phase="single_model",
        agent=tag,
        payload={
            "agent_name": data.get("agent_name", tag),
            "as_of_date": data.get("as_of_date"),
            "horizon": data.get("horizon"),
            "portfolio_id": data.get("portfolio_id"),
            "macro_regime": data.get("macro_regime", ""),
            "summary": data.get("summary", ""),
            "model_family": model_family,
            "model": model_name,
            "used_mock": used_mock,
            "note": note,
            "feeds_black_litterman": feeds_black_litterman,
            "view_mode": "stock_level_then_shared_pairwise",
            "n_pairwise_views": len(pairwise_views),
        },
    )
    for t in universe:
        v = ticker_views[t]
        emitter.emit(
            "signal_emitted",
            phase="phase1",
            ticker=t,
            agent=tag,
            payload={
                "signal": v["signal"],
                "signal_label": v["signal_label"],
                "raw_q": v["raw_q"],
                "confidence": v["raw_confidence"],
                "direction": _direction_from_q(v["raw_q"]) if model_family == "gemini" else _direction_from_signal_value(v["signal"]),
                "source": "single_gemini_direct_stock_q" if model_family == "gemini" else f"single_{model_family}_stock_signal",
                "view_mode": "continuous_model_q" if model_family == "gemini" else "categorical_signal",
                "feeds_black_litterman": False,
            },
        )
        emitter.emit(
            "single_model_emitted",
            phase="single_model",
            ticker=t,
            agent=tag,
            payload={
                **v,
                "model_family": model_family,
                "model": model_name,
                "used_mock": used_mock,
                "feeds_black_litterman": False,
            },
        )
    for idx, view in enumerate(pairwise_views):
        emitter.emit(
            "single_model_pairwise_view",
            phase="single_model",
            agent=tag,
            payload={
                **view,
                "view_id": f"c1_pair_{idx+1}",
                "label": f"{view['long_ticker']}_gt_{view['short_ticker']}",
                "model_family": model_family,
                "model": model_name,
                "used_mock": used_mock,
                "source": f"single_{model_family}_stock_views_to_pairwise",
                "feeds_black_litterman": feeds_black_litterman,
            },
        )


def run_single_gemini_reference(bundle: InputBundle, universe: List[str], *, strict: Optional[bool], emitter=None) -> Dict[str, Any]:
    """Expose the Gemini reference inside a normal debate run without optimizing."""
    model_family = "gemini"
    model_name = config.model_for_family(model_family)
    tag = _single_tag(model_family)
    data, used_mock, note, payload = _call_single_model(
        bundle,
        universe,
        strict=strict,
        model_family=model_family,
        model_name=model_name,
    )
    data = _fill_missing_neutrals(data, universe, emitter, tag=tag, model_family=model_family)
    view_calibrator = calibration.load_view_calibrator()
    ticker_views = _ticker_views_from_single_data(
        data, universe, view_calibrator, model_family=model_family
    )
    pairwise_views = calibration.ticker_views_to_pairwise_bl_views(
        ticker_views,
        universe,
        max_pairs=config.MAX_PAIRWISE_VIEWS,
        spread_floor=config.VIEW_SPREAD_FLOOR,
        source="single_gemini_stock_views_to_pairwise",
    )
    if emitter is not None:
        _emit_single_model_views(
            emitter,
            data=data,
            universe=universe,
            ticker_views=ticker_views,
            pairwise_views=pairwise_views,
            used_mock=used_mock,
            note=note,
            feeds_black_litterman=False,
            model_family=model_family,
            model_name=model_name,
        )
    return {"data": data, "used_mock": used_mock, "note": note, "payload": payload, "ticker_views": ticker_views, "pairwise_views": pairwise_views}


def run_single_model_rebalance(
    provider: DataProvider,
    portfolio_id: str,
    as_of_date: str,
    universe: List[str],
    *,
    horizon: str = "1M",
    strict: Optional[bool] = None,
    emitter: Optional[EventEmitter] = None,
    subscriber: Optional[Callable] = None,
    save: bool = True,
    clock: Optional[Callable[[], datetime]] = None,
    bundle: Optional[InputBundle] = None,
    model_family: str = "gemini",
    model_name: Optional[str] = None,
    sham: bool = False,
) -> Dict[str, Any]:
    """Run the matched single-model flow for one date."""
    del sham
    model_family = config.normalize_model_family(model_family)
    model_name = model_name or config.model_for_family(model_family)

    strict = config.STRICT_MODEL if strict is None else strict
    random.seed(config.PYTHON_RANDOM_SEED)
    np.random.seed(config.NUMPY_RANDOM_SEED)

    if bundle is None:
        bundle = provider.get_input_bundle(portfolio_id, as_of_date, universe, horizon)

    tag = _single_tag(model_family)
    if emitter is None:
        emitter = EventEmitter(
            portfolio_id=portfolio_id,
            as_of_date=as_of_date,
            runs_dir=config.RUNS_DIR,
            meta={
                "provider": provider.name,
                "config": config.summary(),
                "strict_model": strict,
                "universe": list(universe),
                "horizon": horizon,
                "configuration": "single_model",
                "architecture": "single_agent",
                "model_family": model_family,
                "model_name": model_name,
                "single_model_bl": True,
                "single_model_view_mode": "stock_level_then_shared_pairwise",
                "seeds": {
                    "python_random_seed": config.PYTHON_RANDOM_SEED,
                    "numpy_seed": config.NUMPY_RANDOM_SEED,
                },
            },
            clock=clock,
        )
    emitter.meta.setdefault("agent_inputs", bundle.packages)
    emitter.meta.setdefault("covariance", bundle.covariance)
    emitter.meta.setdefault("sectors", bundle.sectors)
    if subscriber is not None:
        emitter.subscribe(subscriber)

    emitter.emit(
        "run_started",
        phase="system",
        payload={
            "portfolio_id": portfolio_id,
            "as_of_date": as_of_date,
            "universe": list(universe),
            "provider": provider.name,
            "strict_model": strict,
            "architecture": "single_agent",
            "model_family": model_family,
            "model_name": model_name,
            "single_model": tag,
            "single_model_bl": True,
            "single_model_view_mode": "stock_level_then_shared_pairwise",
            "provider_warnings": bundle.warnings,
        },
    )
    for w in bundle.warnings:
        emitter.warning(w, phase="provider")

    data, used_mock, note, payload = _call_single_model(
        bundle,
        universe,
        strict=strict,
        model_family=model_family,
        model_name=model_name,
    )
    data = _fill_missing_neutrals(data, universe, emitter, tag=tag, model_family=model_family)
    hrate = _hallucination_rate(data, payload)

    order = list(universe)
    cov = bl_core.covariance_from_block(bundle.covariance, order, strict=strict)
    Sigma = cov.matrix
    caps = bundle.market_caps
    n = len(order)
    equal_w = np.full(n, 1.0 / n, dtype=float)
    cap_w = None
    if caps and all(t in caps for t in order):
        raw = np.array([float(caps[t]) for t in order], dtype=float)
        if np.all(np.isfinite(raw)) and raw.sum() > 0:
            cap_w = raw / raw.sum()
            emitter.emit(
                "baseline_view", phase="market_model",
                payload={
                    "type": "available_market_cap_weights",
                    "source": "market_cap",
                    "weights": {t: round(float(cap_w[i]), 4) for i, t in enumerate(order)},
                },
            )

    if config.BL_PRIOR_MODE == "market_cap" and cap_w is not None:
        w_mkt = cap_w
        prior_source = "market_cap"
    else:
        w_mkt = equal_w
        prior_source = "equal_weight_research_prior"
        if config.BL_PRIOR_MODE == "market_cap" and cap_w is None:
            emitter.warning("BL_PRIOR_MODE=market_cap but market caps are unavailable; using equal-weight prior", phase="market_model")

    emitter.emit(
        "baseline_view", phase="market_model",
        payload={
            "type": "bl_prior_weights",
            "source": prior_source,
            "weights": {t: round(float(w_mkt[i]), 4) for i, t in enumerate(order)},
        },
    )
    pi = bl_core.equilibrium_returns(Sigma, w_mkt)

    view_calibrator = calibration.load_view_calibrator()
    ticker_views = _ticker_views_from_single_data(
        data, order, view_calibrator, model_family=model_family
    )
    pairwise_views = calibration.ticker_views_to_pairwise_bl_views(
        ticker_views,
        order,
        max_pairs=config.MAX_PAIRWISE_VIEWS,
        spread_floor=config.VIEW_SPREAD_FLOOR,
        source=f"single_{model_family}_stock_views_to_pairwise",
    )

    _emit_single_model_views(
        emitter,
        data=data,
        universe=order,
        ticker_views=ticker_views,
        pairwise_views=pairwise_views,
        used_mock=used_mock,
        note=note,
        feeds_black_litterman=True,
        model_family=model_family,
        model_name=model_name,
    )
    emitter.emit("validator_result", phase="validator", agent=tag, payload={"hallucination_rate": hrate, "used_mock": used_mock})

    labels: List[str] = []
    final_views: Dict[str, Dict[str, Any]] = {}
    if pairwise_views:
        P = np.array([v["p_row"] for v in pairwise_views], dtype=float)
        bl_core.validate_view_rows(P)
        Q = np.array([float(v["q"]) for v in pairwise_views], dtype=float)
        conf = np.array([float(v["confidence"]) for v in pairwise_views], dtype=float)
        Omega = bl_core.idzorek_omega(P, Sigma, conf)
        post = bl_core.bl_posterior(pi, Sigma, P, Q, Omega)
        omega_diag = np.diag(Omega)
        for idx, view in enumerate(pairwise_views):
            label = f"{view['long_ticker']}_gt_{view['short_ticker']}"
            labels.append(label)
            payload_view = {
                **view,
                "view_id": f"c1_pair_{idx+1}",
                "label": label,
                "direction": "pairwise",
                "macro_regime": data.get("macro_regime", ""),
            }
            final_views[label] = payload_view
            emitter.emit("final_view", phase="phase4", agent=tag, payload=payload_view)
    else:
        P = np.zeros((0, len(order)))
        Q = np.zeros(0)
        omega_diag = np.zeros(0)
        post = bl_core.BLPosterior(mu_bl=np.array(pi, dtype=float), cov_bl=Sigma, pi=pi)
        emitter.warning(
            f"fewer than two valid {model_family} stock views; posterior remains at prior",
            phase="single_model",
        )

    opt = bl_core.optimize_long_only(post.mu_bl, Sigma, order, bundle.sectors, benchmark_weights=w_mkt)
    weights = {t: float(round(opt.weights[i], 6)) for i, t in enumerate(order)}
    eq_weights = {t: float(round(w_mkt[i], 6)) for i, t in enumerate(order)}
    optimized = {
        "weights": weights,
        "equilibrium_weights": eq_weights,
        "status": opt.status,
        "used_fallback": opt.used_fallback,
        "note": opt.note,
        "portfolio_vol": round(float(opt.portfolio_vol), 6),
        "mu_bl": {t: round(float(post.mu_bl[i]), 6) for i, t in enumerate(order)},
        "posterior_expected_returns": {t: round(float(post.mu_bl[i]), 6) for i, t in enumerate(order)},
        "equilibrium_returns": {t: round(float(pi[i]), 6) for i, t in enumerate(order)},
        "P": P.tolist(),
        "Q": {labels[i]: round(float(Q[i]), 6) for i in range(len(labels))},
        "Omega_diag": {labels[i]: round(float(omega_diag[i]), 10) for i in range(len(labels))},
        "view_labels": labels,
        "view_types": [v.get("view_type", "adjacent_chain_from_stock_views") for v in pairwise_views],
        "stock_views": ticker_views,
        "pairwise_views": final_views,
        "cap_adjusted": opt.cap_adjusted,
        "risk_gate": {
            "min_weight": config.MIN_WEIGHT,
            "max_weight": config.MAX_WEIGHT,
            "max_active_deviation": config.MAX_ACTIVE_DEVIATION,
            "active_weight_budget": config.ACTIVE_WEIGHT_BUDGET,
            "sector_cap": config.SECTOR_CAP,
            "vol_cap": config.VOL_CAP,
        },
        "bl_inputs": {
            "order": order,
            "tau": config.BL_TAU,
            "risk_aversion": config.RISK_AVERSION,
            "P": [[round(float(x), 4) for x in row] for row in P.tolist()],
            "Q": [round(float(x), 6) for x in Q],
            "omega_diag": [round(float(x), 8) for x in omega_diag],
            "pi": [round(float(x), 6) for x in pi],
            "mu_bl": [round(float(x), 6) for x in post.mu_bl],
            "view_labels": labels,
            "view_kinds": [v.get("view_type", "adjacent_chain_from_stock_views") for v in pairwise_views],
        },
        "view_calibration": {
            "path": config.VIEW_CALIBRATION_PATH,
            "default_q_multiplier": config.DEFAULT_Q_MULTIPLIER,
            "default_confidence_center": config.DEFAULT_CONFIDENCE_CENTER,
        },
        "source": f"single_{model_family}_stock_views_shared_pairwise_bl_posterior",
    }
    emitter.emit("optimized_portfolio", phase="phase4", payload=optimized)

    result = {
        "portfolio_id": portfolio_id,
        "as_of_date": as_of_date,
        "architecture": "single_agent",
        "model_family": model_family,
        "model_name": model_name,
        "weights": weights,
        "equilibrium_weights": eq_weights,
        "optimized": optimized,
        "final_views": {"stock_views": ticker_views, "pairwise_views": final_views},
        "consensus": {},
        "validator": {tag: {"hallucination_rate": hrate}},
        "used_mock": {tag: used_mock},
        "seeds": {
            "python_random_seed": config.PYTHON_RANDOM_SEED,
            "numpy_seed": config.NUMPY_RANDOM_SEED,
        },
        "n_rounds": 0,
        "consensus_failures": [],
        "single_model": {"data": data, "used_mock": used_mock, "note": note, "payload": payload},
        "emission": data,
    }

    emitter.emit(
        "run_finished",
        phase="system",
        payload={
            "weights": weights,
            "n_rounds": 0,
            "consensus_failures": [],
            "any_mock": used_mock,
            "single_model_bl": True,
            "single_model_view_mode": "stock_level_then_shared_pairwise",
        },
    )
    if save:
        result["run_log_path"] = emitter.save()
    result["emitter"] = emitter
    return result
