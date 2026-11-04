"""Calibration and shared view construction for model-generated stock views.

The research rule is: models produce *raw* stock-level opinions; this module
turns those opinions into statistical Black-Litterman inputs using constants
learned from historical out-of-sample view outcomes.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from agentic_bl import bl_core
from agentic_bl import config
@dataclass
class CalibrationParams:
    """Small set of learned constants used online at rebalance time.

    q_multiplier:
        Shrinks or expands raw q. It is learned from historical raw-q versus
        realized-return magnitude, but the estimate is reliability-weighted and
        shrunk toward a shared/global prior. A family with coin-flip directional
        skill therefore cannot receive an extreme multiplier solely because of
        a noisy magnitude regression.
    confidence_center:
        Baseline BL confidence learned from historical directional hit rate.
    confidence_slope:
        How much to use raw model confidence. If raw confidence was not useful
        historically, this should be close to zero.
    n_samples:
        Number of historical view observations behind the params.
    raw_q_slope:
        Unshrunk robust no-intercept magnitude slope, retained for diagnostics.
    rank_ic:
        Mean within-date Spearman rank correlation between raw q and realized
        stock-vs-rest return. This measures the cross-sectional ranking skill
        that the portfolio construction step actually uses.
    reliability:
        A [0,1] score combining directional and cross-sectional ranking skill.
        It controls how far a family-specific q multiplier can move away from
        the shared/global multiplier.
    """

    q_multiplier: float = config.DEFAULT_Q_MULTIPLIER
    confidence_center: float = config.DEFAULT_CONFIDENCE_CENTER
    confidence_slope: float = config.DEFAULT_CONFIDENCE_SLOPE
    confidence_floor: float = config.MIN_VIEW_CONFIDENCE
    confidence_cap: float = config.MAX_VIEW_CONFIDENCE
    n_samples: int = 0
    rmse: Optional[float] = None
    hit_rate: Optional[float] = None
    raw_q_slope: Optional[float] = None
    rank_ic: Optional[float] = None
    reliability: Optional[float] = None


@dataclass
class CalibratedView:
    q: float
    confidence: float
    raw_q: float
    raw_confidence: float
    params: CalibrationParams
    family: str

    def as_metadata(self) -> Dict[str, Any]:
        p = self.params
        return {
            "family": self.family,
            "q_multiplier": round(float(p.q_multiplier), 6),
            "confidence_center": round(float(p.confidence_center), 6),
            "confidence_slope": round(float(p.confidence_slope), 6),
            "confidence_floor": round(float(p.confidence_floor), 6),
            "confidence_cap": round(float(p.confidence_cap), 6),
            "n_samples": int(p.n_samples),
            "rmse": None if p.rmse is None else round(float(p.rmse), 6),
            "hit_rate": None if p.hit_rate is None else round(float(p.hit_rate), 6),
            "raw_q_slope": None if p.raw_q_slope is None else round(float(p.raw_q_slope), 6),
            "rank_ic": None if p.rank_ic is None else round(float(p.rank_ic), 6),
            "reliability": None if p.reliability is None else round(float(p.reliability), 6),
        }


class ViewCalibrator:
    """Online calibrator used by both Gemma debate and Gemini C1.

    It can load a JSON file learned offline by ``learn_calibration.py``. If no
    file exists, it uses conservative defaults so runs remain reproducible.
    """

    def __init__(self, params_by_family: Optional[Dict[str, CalibrationParams]] = None):
        self.params_by_family = params_by_family or {"default": CalibrationParams()}

    @classmethod
    def default(cls) -> "ViewCalibrator":
        return cls({"default": CalibrationParams()})

    @classmethod
    def from_json_file(cls, path: Optional[str]) -> "ViewCalibrator":
        if not path or not os.path.exists(path):
            return cls.default()
        with open(path, "r") as fh:
            raw = json.load(fh)
        params_by_family = {}
        for family, d in (raw.get("families") or raw).items():
            # Ignore future diagnostic keys rather than making old/new calibration
            # files brittle across small research iterations.
            allowed = CalibrationParams.__dataclass_fields__
            clean = {k: v for k, v in d.items() if k in allowed}
            params_by_family[family] = CalibrationParams(**clean)
        params_by_family.setdefault("default", CalibrationParams())
        return cls(params_by_family)

    def params_for(self, family: str) -> CalibrationParams:
        return self.params_by_family.get(family) or self.params_by_family.get("default") or CalibrationParams()

    def calibrate(self, raw_q: float, raw_confidence: float, *, family: str = "default") -> CalibratedView:
        p = self.params_for(family)
        raw_q = _safe_float(raw_q)
        raw_confidence = _safe_float(raw_confidence, 0.5)

        q = raw_q * p.q_multiplier
        # Interpret raw confidence only as a weak ranking signal around the learned
        # historical center. If confidence_slope=0, all views of this family get
        # the same empirical confidence.
        confidence = p.confidence_center + p.confidence_slope * (raw_confidence - 0.5)

        # With no fitted history, the default center/slope must not erase a real
        # disagreement penalty. Treat raw confidence only as a downward veto:
        # an unlearned model may become less influential when agents disagree,
        # but self-reported confidence cannot lift it above the conservative prior.
        if p.n_samples <= 0:
            confidence = min(confidence, raw_confidence)
        confidence = float(np.clip(confidence, p.confidence_floor, p.confidence_cap))
        return CalibratedView(
            q=float(q),
            confidence=confidence,
            raw_q=raw_q,
            raw_confidence=raw_confidence,
            params=p,
            family=family,
        )

    @classmethod
    def learn(
        cls,
        samples: Iterable[Dict[str, Any]],
        *,
        family: str = "default",
        prior: Optional[CalibrationParams] = None,
    ) -> CalibrationParams:
        """Learn params from historical out-of-sample stock-view samples.

        Each sample must contain ``raw_q``, ``raw_confidence`` and ``realized_y``.
        ``as_of_date`` is optional but strongly recommended because it lets us
        estimate within-date cross-sectional rank skill.

        The q multiplier is intentionally learned in two parts:

        1. a robust magnitude slope between raw q and realized return;
        2. a reliability score from directional hit rate and cross-sectional IC.

        The family estimate is then reliability-weighted and shrunk toward the
        proven portion of ``prior`` (normally the shared calibration). Weak,
        coin-flip, or negatively related families are allowed to abstain.
        """
        rows: List[Tuple[float, float, float, Optional[str]]] = []
        for s in samples:
            raw_q = _safe_float(s.get("raw_q"), None)
            realized_y = _safe_float(s.get("realized_y"), None)
            if raw_q is None or realized_y is None or abs(raw_q) < 1e-12:
                continue
            rows.append(
                (
                    raw_q,
                    _safe_float(s.get("raw_confidence"), 0.5),
                    realized_y,
                    None if s.get("as_of_date") is None else str(s.get("as_of_date")),
                )
            )

        prior_params = prior or CalibrationParams()
        n = len(rows)
        if n < config.CALIBRATION_MIN_SAMPLES:
            prior_reliability = float(np.clip(prior_params.reliability or 0.0, 0.0, 1.0))
            return CalibrationParams(
                q_multiplier=float(prior_params.q_multiplier) * prior_reliability,
                confidence_center=config.MIN_VIEW_CONFIDENCE + prior_reliability * (
                    config.MAX_VIEW_CONFIDENCE - config.MIN_VIEW_CONFIDENCE
                ),
                confidence_slope=float(prior_params.confidence_slope) * prior_reliability,
                confidence_floor=config.MIN_VIEW_CONFIDENCE,
                confidence_cap=config.MAX_VIEW_CONFIDENCE,
                n_samples=n,
                rmse=None,
                hit_rate=None,
                raw_q_slope=None,
                rank_ic=None,
                reliability=0.0,
            )

        x = np.array([r[0] for r in rows], dtype=float)
        c = np.array([r[1] for r in rows], dtype=float)
        y = np.array([r[2] for r in rows], dtype=float)
        dates = [r[3] for r in rows]

        # Monthly stock-vs-rest returns can annualize into very large outliers.
        # Winsorization keeps one unusual month from setting the q scale for an
        # entire model family while retaining every observation in the fit.
        lo, hi = np.quantile(y, [config.CALIBRATION_WINSOR_LOW, config.CALIBRATION_WINSOR_HIGH])
        y_fit = np.clip(y, max(lo, -config.CALIBRATION_REALIZED_Y_CAP), min(hi, config.CALIBRATION_REALIZED_Y_CAP))

        # Ridge-stabilized no-intercept magnitude slope. The ridge is expressed as
        # a fraction of total x energy so it behaves consistently across families.
        x_energy = float(np.dot(x, x))
        ridge = config.CALIBRATION_RIDGE_FRACTION * x_energy
        beta_hat_raw = float(np.dot(x, y_fit) / (x_energy + ridge + 1e-12))
        # A weak or negatively related signal should be allowed to abstain. Do not
        # force every family to retain a positive multiplier merely because it
        # exists in the calibration set.
        beta_hat = float(np.clip(max(beta_hat_raw, 0.0), config.MIN_Q_MULTIPLIER, config.MAX_Q_MULTIPLIER))

        correct = (np.sign(x) == np.sign(y)).astype(float)
        hit_rate = float(np.mean(correct))
        rank_ic = _mean_within_date_rank_ic(x, y, dates)

        hit_skill = float(
            np.clip(
                (hit_rate - 0.5) / max(config.CALIBRATION_FULL_SKILL_HIT_EDGE, 1e-12),
                0.0,
                1.0,
            )
        )
        rank_skill = float(
            np.clip(
                max(rank_ic, 0.0) / max(config.CALIBRATION_FULL_SKILL_RANK_IC, 1e-12),
                0.0,
                1.0,
            )
        )
        reliability = float(
            config.CALIBRATION_HIT_SKILL_WEIGHT * hit_skill
            + (1.0 - config.CALIBRATION_HIT_SKILL_WEIGHT) * rank_skill
        )

        sample_weight = n / (n + config.CALIBRATION_Q_SHRINK_N)
        prior_reliability = float(np.clip(prior_params.reliability or 0.0, 0.0, 1.0))
        prior_effective_beta = float(prior_params.q_multiplier) * prior_reliability
        family_effective_beta = beta_hat * reliability
        beta = prior_effective_beta + sample_weight * (
            family_effective_beta - prior_effective_beta
        )
        beta = float(np.clip(beta, config.MIN_Q_MULTIPLIER, config.MAX_Q_MULTIPLIER))

        pred = beta * x
        rmse = float(np.sqrt(np.mean((y - pred) ** 2)))

        # Idzorek confidence is authority relative to the prior, not directional
        # hit rate itself. A 50% hit rate is no skill and must map near the
        # confidence floor rather than to a powerful 50% BL view.
        target_center = config.MIN_VIEW_CONFIDENCE + reliability * (
            config.MAX_VIEW_CONFIDENCE - config.MIN_VIEW_CONFIDENCE
        )
        prior_center = config.MIN_VIEW_CONFIDENCE + prior_reliability * (
            config.MAX_VIEW_CONFIDENCE - config.MIN_VIEW_CONFIDENCE
        )
        center_weight = n / (n + config.CONFIDENCE_CENTER_SHRINK_N)
        center = prior_center + center_weight * (target_center - prior_center)
        center = float(np.clip(center, config.MIN_VIEW_CONFIDENCE, config.MAX_VIEW_CONFIDENCE))

        z = c - 0.5
        slope_hat = float(np.dot(z, correct - hit_rate) / (np.dot(z, z) + 1e-12))
        slope_weight = n / (n + config.CONFIDENCE_SLOPE_SHRINK_N)
        slope = prior_params.confidence_slope + slope_weight * (
            float(np.clip(slope_hat, 0.0, config.MAX_CONFIDENCE_SLOPE))
            - prior_params.confidence_slope
        )
        slope = float(np.clip(slope, 0.0, config.MAX_CONFIDENCE_SLOPE))

        return CalibrationParams(
            q_multiplier=beta,
            confidence_center=center,
            confidence_slope=slope,
            confidence_floor=config.MIN_VIEW_CONFIDENCE,
            confidence_cap=config.MAX_VIEW_CONFIDENCE,
            n_samples=n,
            rmse=rmse,
            hit_rate=hit_rate,
            raw_q_slope=beta_hat_raw,
            rank_ic=rank_ic,
            reliability=reliability,
        )


def load_view_calibrator() -> ViewCalibrator:
    return ViewCalibrator.from_json_file(config.VIEW_CALIBRATION_PATH)


def ticker_signal_to_raw_q(signal_value: float) -> float:
    """Convert a normalized stock signal [-1,1] to raw annualized q before calibration."""
    return float(_safe_float(signal_value) * config.ANNUAL_RETURN_SCALE)


def ticker_views_to_pairwise_bl_views(
    ticker_views: Dict[str, Dict[str, Any]],
    order: List[str],
    *,
    max_pairs: int = None,
    spread_floor: float = None,
    source: str = "ticker_views_to_pairwise",
) -> List[Dict[str, Any]]:
    """Convert calibrated stock-level views into an independent adjacent chain.

    Valid stock views are sorted by calibrated q. Adjacent pairs whose
    calibrated spread is below ``spread_floor`` are omitted: Q=0 is not a
    neutral BL input, it is an active opinion that the two assets should have
    equal expected returns. For the remaining rows, the converter creates at
    most ``n-1`` adjacent ranking views:

        rank 1 > rank 2, rank 2 > rank 3, ..., rank n-1 > rank n

    This preserves the informative part of the cross-sectional ordering without
    feeding BL many overlapping rows or forcing low-information ties into the
    posterior. When no adjacent spread clears the floor, the posterior remains at
    the prior and the portfolio stays at its benchmark allocation.
    """
    spread_floor = config.VIEW_SPREAD_FLOOR if spread_floor is None else float(spread_floor)

    ranked = []
    order_index = {ticker: i for i, ticker in enumerate(order)}
    for ticker in order:
        view = ticker_views.get(ticker) or {}
        if not _eligible_ticker_view(view):
            continue
        q = _safe_float(view.get("q"), None)
        if q is None:
            continue
        ranked.append((ticker, view, float(q)))

    ranked.sort(key=lambda item: (-item[2], order_index[item[0]]))
    if len(ranked) < 2:
        return []

    natural_count = len(ranked) - 1
    if max_pairs is None or int(max_pairs) <= 0:
        limit = natural_count
    else:
        limit = min(int(max_pairs), natural_count)

    pairwise = []
    for rank_idx, ((long_ticker, long_view, long_q), (short_ticker, short_view, short_q)) in enumerate(
        zip(ranked[:-1], ranked[1:]),
        start=1,
    ):
        if len(pairwise) >= limit:
            break

        spread = max(0.0, float(long_q - short_q))
        if spread < spread_floor:
            continue
        long_raw_q = _safe_float(long_view.get("raw_q"), long_q)
        short_raw_q = _safe_float(short_view.get("raw_q"), short_q)
        raw_spread = max(0.0, float(long_raw_q - short_raw_q))
        long_conf = _safe_float(long_view.get("confidence"), 0.5)
        short_conf = _safe_float(short_view.get("confidence"), 0.5)
        confidence = float(
            np.clip(
                (long_conf + short_conf) / 2.0,
                config.MIN_VIEW_CONFIDENCE,
                config.MAX_VIEW_CONFIDENCE,
            )
        )

        pairwise.append(
            {
                "long_ticker": long_ticker,
                "short_ticker": short_ticker,
                "q": spread,
                "raw_q": raw_spread,
                "confidence": confidence,
                "p_row": bl_core.p_row_relative([long_ticker], [short_ticker], order),
                "view_type": "adjacent_chain_from_stock_views",
                "source": source,
                "rank_long": rank_idx,
                "rank_short": rank_idx + 1,
                "below_spread_floor": bool(spread < spread_floor),
                "long_source": long_view.get("source"),
                "short_source": short_view.get("source"),
                "long_stock_q": long_q,
                "short_stock_q": short_q,
                "long_raw_q": long_raw_q,
                "short_raw_q": short_raw_q,
            }
        )

    return pairwise


def _eligible_ticker_view(view: Dict[str, Any]) -> bool:
    if not view:
        return False
    if view.get("resolved") is False:
        return False
    # A neutral stock score can still establish the ranking order, but adjacent
    # ties are removed later by VIEW_SPREAD_FLOOR before they reach BL.
    return _safe_float(view.get("q"), None) is not None


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks for ties, implemented without scipy."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        average_rank = 0.5 * (i + j - 1)
        ranks[order[i:j]] = average_rank
        i = j
    return ranks


def _mean_within_date_rank_ic(x: np.ndarray, y: np.ndarray, dates: List[Optional[str]]) -> float:
    grouped: Dict[str, List[int]] = {}
    for i, date in enumerate(dates):
        grouped.setdefault(date or "__all__", []).append(i)

    ics = []
    for idxs in grouped.values():
        if len(idxs) < 3:
            continue
        xx = x[idxs]
        yy = y[idxs]
        if np.allclose(xx, xx[0]) or np.allclose(yy, yy[0]):
            continue
        rx = _rankdata(xx)
        ry = _rankdata(yy)
        corr = float(np.corrcoef(rx, ry)[0, 1])
        if np.isfinite(corr):
            ics.append(corr)
    return float(np.mean(ics)) if ics else 0.0


def _safe_float(x: Any, default: Optional[float] = 0.0) -> Optional[float]:
    try:
        if x is None:
            return default
        f = float(x)
        if np.isnan(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def params_to_json(params_by_family: Dict[str, CalibrationParams]) -> Dict[str, Any]:
    return {"families": {k: asdict(v) for k, v in params_by_family.items()}}
