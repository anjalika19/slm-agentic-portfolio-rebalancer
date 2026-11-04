"""Pydantic schemas shared across the framework.

These are enforced on every model call (re-prompt once on failure, spec Phase 1).
The specialist output schema is exactly the one in spec section 1, including the
required ``cited_values`` block the validator (section 4) traces back to inputs.
"""

from __future__ import annotations

import math

from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


Direction = Literal["long", "short", "neutral"]

# 5-way categorical conviction used at Phase-1 emission (NOT numeric q/P -- those
# only appear when the moderator builds the BL views). Map to a numeric tilt for the
# moderator's baseline q.
Signal5 = Literal["strongly_long", "long", "neutral", "underweight", "avoid"]
SIGNAL5_VALUE = {
    "strongly_long": 1.0,
    "long": 0.5,
    "neutral": 0.0,
    "underweight": -0.5,
    "avoid": -1.0,
}
# economist sector stance -> per-ticker tilt
STANCE_VALUE = {"overweight": 0.5, "neutral": 0.0, "underweight": -0.5}

# Economist regime taxonomy (matches the original economist.py schema)
RATE_LABELS = ["tightening", "stable", "easing"]
RISK_LABELS = ["stressed", "normal", "calm"]
CYCLE_LABELS = ["contraction", "mid_cycle", "expansion"]
STANCE_LABELS = ["overweight", "neutral", "underweight"]


class CitedValue(BaseModel):
    name: str
    value: float


class MacroCitedValue(BaseModel):
    name: str
    value: Union[float, str]


# --------------------------------------------------------------------------- #
# Phase 1 -- specialist emissions (per-agent shapes; signal + rationale, NO q/P).
# Each carries cited_values so the value-tracing validator (section 4) still works.
# --------------------------------------------------------------------------- #
class QuantStockSignal(BaseModel):
    signal: Signal5
    primary_factors: List[str] = Field(default_factory=list)
    factor_alignment: Literal["supporting", "mixed", "adverse"] = "mixed"
    alpha_contribution: float = 0.0
    rationale: str = ""
    cited_values: List[CitedValue] = Field(default_factory=list)


class QuantPortfolioView(BaseModel):
    recommended_tilt: str = ""
    covariance_note: str = ""
    whiteboard_instruction: str = ""


class QuantEmission(BaseModel):
    agent_name: str = "quant"
    as_of_date: str = ""
    horizon: str = "1M"
    portfolio_id: str = ""
    stock_signals: Dict[str, QuantStockSignal]
    portfolio_level_view: QuantPortfolioView = Field(default_factory=QuantPortfolioView)
    summary: str = ""


class FundamentalStockSignal(BaseModel):
    signal: Signal5
    valuation_call: Literal["cheap", "fair", "expensive"] = "fair"
    quality_tier: Literal["strong", "average", "weak"] = "average"
    key_drivers: List[str] = Field(default_factory=list)
    rationale: str = ""
    cited_values: List[CitedValue] = Field(default_factory=list)


class FundamentalEmission(BaseModel):
    agent_name: str = "fundamental"
    as_of_date: str = ""
    horizon: str = "1M"
    portfolio_id: str = ""
    signals: Dict[str, FundamentalStockSignal]
    summary: str = ""


class SentimentStockSignal(BaseModel):
    signal: Signal5
    primary_catalyst: str = ""
    event_driven: bool = False
    rationale: str = ""
    cited_values: List[CitedValue] = Field(default_factory=list)


class SentimentEmission(BaseModel):
    agent_name: str = "sentiment"
    as_of_date: str = ""
    horizon: str = "1M"
    portfolio_id: str = ""
    signals: Dict[str, SentimentStockSignal]
    summary: str = ""


class SectorView(BaseModel):
    stance: Literal["overweight", "neutral", "underweight"] = "neutral"
    rationale: str = ""
    cited_values: List[MacroCitedValue] = Field(default_factory=list)


class EconomistEmission(BaseModel):
    agent_name: str = "economist"
    as_of_date: str = ""
    horizon: str = "1M"
    portfolio_id: str = ""
    # three independent regime axes (original economist schema)
    rate_environment: Literal["tightening", "stable", "easing"] = "stable"
    risk_regime: Literal["stressed", "normal", "calm"] = "normal"
    cycle: Literal["contraction", "mid_cycle", "expansion"] = "mid_cycle"
    regime_rationale: str = ""
    cited_values: List[MacroCitedValue] = Field(default_factory=list)
    sector_views: Dict[str, SectorView] = Field(default_factory=dict)
    favored_sectors: List[str] = Field(default_factory=list)   # overweight sectors
    risk_sectors: List[str] = Field(default_factory=list)      # underweight sectors
    summary: str = ""




# --------------------------------------------------------------------------- #
# C1 control -- single Gemini analyst, zero-shot, one consolidated all-data call.
# --------------------------------------------------------------------------- #
class SingleModelPairwiseBLView(BaseModel):
    """Legacy direct pairwise view.

    Kept only so old run logs and older Gemini responses still validate. New
    research runs should leave ``pairwise_views`` empty: Gemini emits stock-level
    views and Python converts them to pairwise BL rows using the same function as
    the Gemma debate path.
    """

    view_type: Literal["pairwise"] = "pairwise"
    long_ticker: str
    short_ticker: str
    q: float = Field(
        ge=0.0, le=1.0,
        description="Annualized expected outperformance spread in decimal units; 0.06 means long_ticker beats short_ticker by 6%.",
    )
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""
    cited_values: List[CitedValue] = Field(default_factory=list)

    @model_validator(mode="after")
    def _valid_pair(self):
        if self.long_ticker == self.short_ticker:
            raise ValueError("pairwise BL view requires two different tickers")
        if self.q <= 0.0:
            raise ValueError("pairwise BL view q must be positive; encode the preferred asset as long_ticker")
        return self


class SingleModelTickerSignal(BaseModel):
    # Legacy/shared single-model control used by Gemma-Single. The categorical
    # label is still converted to q in Python for backward compatibility.
    signal: Signal5
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""
    cited_values: List[CitedValue] = Field(default_factory=list)


class GeminiSingleTickerView(BaseModel):
    """Continuous stock-level view emitted directly by Gemini-Single.

    ``q`` is the model's own annualized expected relative return for the ticker
    versus the average of the other names in the current portfolio. It is not
    quantized through Signal5 and is intentionally left without an artificial
    magnitude bound. Downstream fixed-scaling replay applies the common 0.25
    multiplier.
    """

    q: float = Field(
        description=(
            "Continuous annualized expected relative return in decimal units; "
            "e.g. 0.08 means +8% annualized relative outperformance and -0.05 "
            "means -5% annualized relative underperformance."
        )
    )
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""
    cited_values: List[CitedValue] = Field(default_factory=list)

    @field_validator("q")
    @classmethod
    def _finite_q(cls, v: float) -> float:
        if not math.isfinite(float(v)):
            raise ValueError("q must be finite")
        return float(v)


class SingleModelEmission(BaseModel):
    agent_name: str = "single_gemini"
    as_of_date: str = ""
    horizon: str = "1M"
    portfolio_id: str = ""
    macro_regime: str = ""
    signals: Dict[str, SingleModelTickerSignal]
    pairwise_views: List[SingleModelPairwiseBLView] = Field(default_factory=list)  # legacy/optional
    summary: str = ""

    @field_validator("signals")
    @classmethod
    def _non_empty(cls, v: Dict[str, SingleModelTickerSignal]) -> Dict[str, SingleModelTickerSignal]:
        if not v:
            raise ValueError("signals must contain at least one ticker")
        return v


class GeminiSingleModelEmission(BaseModel):
    """Gemini-Single output with direct continuous per-stock q values."""

    agent_name: str = "single_gemini"
    as_of_date: str = ""
    horizon: str = "1M"
    portfolio_id: str = ""
    macro_regime: str = ""
    signals: Dict[str, GeminiSingleTickerView]
    pairwise_views: List[SingleModelPairwiseBLView] = Field(default_factory=list)
    summary: str = ""

    @field_validator("signals")
    @classmethod
    def _non_empty(cls, v: Dict[str, GeminiSingleTickerView]) -> Dict[str, GeminiSingleTickerView]:
        if not v:
            raise ValueError("signals must contain at least one ticker")
        return v



# Per-agent emission schema registry (economist is sector-level, no per-ticker).
EMISSION_SCHEMAS = {
    "quant": QuantEmission,
    "fundamental": FundamentalEmission,
    "sentiment": SentimentEmission,
    "economist": EconomistEmission,
}


# --------------------------------------------------------------------------- #
# Legacy uniform output -- kept for the model-client tests / generic callers.
# --------------------------------------------------------------------------- #
class TickerSignal(BaseModel):
    signal: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    cited_values: List[CitedValue] = Field(default_factory=list)


class SpecialistAgentOutput(BaseModel):
    agent_name: str
    as_of_date: str
    horizon: str
    portfolio_id: str
    signals: Dict[str, TickerSignal]
    summary: str = ""

    @field_validator("signals")
    @classmethod
    def _non_empty(cls, v: Dict[str, TickerSignal]) -> Dict[str, TickerSignal]:
        if not v:
            raise ValueError("signals must contain at least one ticker")
        return v


# --------------------------------------------------------------------------- #
# Debate statements (Phase 3)
# --------------------------------------------------------------------------- #
class EvidencePoint(BaseModel):
    name: str
    value: Optional[float] = None
    note: str = ""


class AgentOpeningStatement(BaseModel):
    agent_name: str
    ticker: str
    direction: Direction
    q: float                      # agent's preferred annualized excess return view
    confidence: float = Field(ge=0.0, le=1.0)
    position_rationale: str
    objection_to_current_view: str = ""
    evidence: List[EvidencePoint] = Field(default_factory=list)
    what_would_change_my_mind: str = ""


class AgentReply(BaseModel):
    """One direct, by-name reply to a specific peer in the roundtable (spec B)."""
    responding_to: str                                  # peer agent being addressed
    stance: Literal["agree", "disagree", "partial"] = "partial"
    quoted_claim: str = ""                              # snippet of the peer's point
    my_response: str = ""                               # direct, evidence-bearing answer
    cited_number: Optional[float] = None                 # number used in my_response
    # Backward-compatible alias for old UI/mocks; normalizer fills it from my_response.
    message: str = ""


class AgentRebuttalStatement(BaseModel):
    agent_name: str
    ticker: str
    # Revised post-debate position -- THIS is what the vote must consume (spec 3a).
    revised_direction: Direction
    revised_q: float
    revised_confidence: float = Field(ge=0.0, le=1.0)
    # roundtable: one or more direct replies to the most-disagreeing peers (spec B)
    replies: List[AgentReply] = Field(default_factory=list)
    responded_to: str = ""
    concessions: List[str] = Field(default_factory=list)
    still_objects: List[str] = Field(default_factory=list)
    influenced_by: List[str] = Field(default_factory=list)
    strongest_counterargument_agent: Optional[str] = None
    engaged: bool = True
    new_fact_or_correction: bool = False
    freeze_reason: str = ""


class ModeratorSynthesis(BaseModel):
    ticker: str
    recommended_direction: Direction
    recommended_q: float
    recommended_confidence: float = Field(ge=0.0, le=1.0)
    moderator_reasoning_summary: str
    who_swayed_the_decision: List[str] = Field(default_factory=list)
    agents_that_changed_position: List[str] = Field(default_factory=list)
    unresolved_objections: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Black-Litterman views
# --------------------------------------------------------------------------- #
class BLViewItem(BaseModel):
    ticker: str
    direction: Direction
    q: float
    confidence: float = Field(ge=0.0, le=1.0)
    omega: float = Field(gt=0.0)
    p_row: List[float] = Field(default_factory=list)


class BLViewSet(BaseModel):
    source: str
    as_of_date: str
    portfolio_id: str
    universe_order: List[str]
    views: List[BLViewItem]
    summary: str = ""
