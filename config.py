"""Central configuration for the agentic Black-Litterman framework.

Every tunable constant the spec calls out lives here so a run can be reproduced
from one place. Values may be overridden via environment variables (see
``_env_*`` helpers) which is convenient for the server and backtest driver.
"""

from __future__ import annotations

from multiprocessing import process
import os
from typing import Dict, List, Optional


# --------------------------------------------------------------------------- #
# env helpers
# --------------------------------------------------------------------------- #
def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_optional_float(name: str, default: Optional[float] = None) -> Optional[float]:
    """Read an optional float; blank/none/off disables the setting."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"", "none", "null", "off", "false", "disabled"}:
        return None
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


# --------------------------------------------------------------------------- #
# model wiring
# --------------------------------------------------------------------------- #
# NOTE: there is no public "gemma4" tag in the Ollama registry; gemma3 is the
# latest Gemma. Override with GEMMA_MODEL=<your tag> if you have a custom build.
GEMMA_MODEL: str = _env_str("GEMMA_MODEL", "gemma4:latest")
OLLAMA_BASE_URL: str = _env_str("OLLAMA_BASE_URL", "http://localhost:11434")

# When True, a failed model call raises instead of falling back to a mock.
# Default True for real runs (spec 3f). The fixture demo sets STRICT_MODEL=False.
STRICT_MODEL: bool = _env_bool("STRICT_MODEL", True)

# Temperature for all role calls (deterministic by default).
MODEL_TEMPERATURE: float = _env_float("MODEL_TEMPERATURE", 0.2)
MODEL_TIMEOUT_S: int = _env_int("MODEL_TIMEOUT_S", 240)
MODEL_SEED: int = _env_int("MODEL_SEED", 12345)
PYTHON_RANDOM_SEED: int = _env_int("PYTHON_RANDOM_SEED", MODEL_SEED)
NUMPY_RANDOM_SEED: int = _env_int("NUMPY_RANDOM_SEED", MODEL_SEED)
# Dynamic temperatures by phase. Keep debate low-variance but allow Round 2 a little
# more synthesis. Override any value with env vars for ablations.
TEMP_SPECIALIST_PHASE1: float = _env_float("TEMP_SPECIALIST_PHASE1", 0.10)
TEMP_DEBATE_ROUND1: float = _env_float("TEMP_DEBATE_ROUND1", 0.15)
TEMP_DEBATE_ROUND2: float = _env_float("TEMP_DEBATE_ROUND2", 0.20)
TEMP_DEBATE_FINAL: float = _env_float("TEMP_DEBATE_FINAL", 0.05)
TEMP_ONE_ROUND_DEBATE: float = _env_float("TEMP_ONE_ROUND_DEBATE", 0.10)
TEMP_MODERATOR: float = _env_float("TEMP_MODERATOR", 0.00)
TEMP_VALIDATOR: float = _env_float("TEMP_VALIDATOR", 0.00)
TEMP_SINGLE_MODEL: float = _env_float(
    "TEMP_SINGLE_MODEL", _env_float("TEMP_SINGLE_GEMINI", 0.05)
)
# Backward-compatible name used by older code and saved config snapshots.
TEMP_SINGLE_GEMINI: float = TEMP_SINGLE_MODEL
GEMINI_MODEL: str = _env_str("GEMINI_MODEL", "gemini4:latest")

GEMINI_API_KEY: str = _env_str(
    "GEMINI_API_KEY",
    _env_str("GOOGLE_API_KEY", "")
)
RUN_SINGLE_GEMINI_C1: bool = _env_bool("RUN_SINGLE_GEMINI_C1", False)
ROLES: List[str] = [
    "quant", "fundamental", "sentiment", "economist", "moderator",
    "validator", "single_gemini", "single_gemma",
]
SPECIALISTS: List[str] = ["quant", "fundamental", "sentiment", "economist"]
MODEL_FAMILIES = {"gemma", "gemini"}

# Optional speed mode. It does not change prompts, schemas, BL math, or optimizer
# settings. It only allows independent specialist calls to be issued concurrently.
# Keep off by default for maximum stability/reproducibility; enable for batch runs
# after confirming your local Ollama setup can handle parallel requests.
PARALLEL_AGENT_CALLS: bool = _env_bool("PARALLEL_AGENT_CALLS", True)
AGENT_PARALLEL_WORKERS: int = _env_int("AGENT_PARALLEL_WORKERS", 4)


def normalize_model_family(model_family: str) -> str:
    family = str(model_family or "gemma").strip().lower()
    aliases = {"slm": "gemma", "llm": "gemini", "ollama": "gemma"}
    family = aliases.get(family, family)
    if family not in MODEL_FAMILIES:
        raise ValueError(f"unknown model_family={model_family!r}; expected gemma or gemini")
    return family


def provider_for_family(model_family: str) -> str:
    return "gemini" if normalize_model_family(model_family) == "gemini" else "ollama"


def model_for_family(model_family: str) -> str:
    return GEMINI_MODEL if normalize_model_family(model_family) == "gemini" else GEMMA_MODEL


def provider_for_role(role: str, model_family: str = "") -> str:
    """Resolve the runtime provider without coupling it to the architecture."""
    if model_family:
        return provider_for_family(model_family)
    return "gemini" if role == "single_gemini" else "ollama"


def model_for_role(role: str, model_family: str = "") -> str:
    """Resolve the runtime model without coupling it to the architecture."""
    if model_family:
        return model_for_family(model_family)
    return GEMINI_MODEL if role == "single_gemini" else GEMMA_MODEL
 

# --------------------------------------------------------------------------- #
# Black-Litterman / optimizer constants
# --------------------------------------------------------------------------- #
ANNUAL_RETURN_SCALE: float = _env_float("ANNUAL_RETURN_SCALE", 0.12)
BL_TAU: float = _env_float("BL_TAU", 0.05)
RISK_AVERSION: float = _env_float("RISK_AVERSION", 2.5)

# Risk gate (constrained long-only QP).
# Common risk limits for both Gemma and Gemini. In the six-stock research book,
# equal weight is 16.67%; the default per-name active bound therefore keeps weights
# between roughly 6.67% and 26.67%. Feasibility adjustments, if ever required for
# a non-equal benchmark, are explicit in the run log.
MIN_WEIGHT: float = _env_float("MIN_WEIGHT", 0.05)
MAX_WEIGHT: float = _env_float("MAX_WEIGHT", 0.28)
MAX_ACTIVE_DEVIATION: float = _env_float("MAX_ACTIVE_DEVIATION", 0.10)
SECTOR_CAP: float = _env_float("SECTOR_CAP", 0.40)
# Absolute volatility is a diagnostic by default, not a hard feasibility gate.
# Set VOL_CAP explicitly only for a mandate that genuinely requires one.
VOL_CAP: Optional[float] = _env_optional_float("VOL_CAP", None)

# Benchmark-relative portfolio sizing. The hard per-name deviation and total L1
# budget prevent a noisy one-month view from producing 0%/30% corner portfolios;
# the soft penalties still allow the optimizer to express relative convictions.
ACTIVE_WEIGHT_BUDGET: float = _env_float("ACTIVE_WEIGHT_BUDGET", 0.40)
BENCHMARK_PENALTY: float = _env_float("BENCHMARK_PENALTY", 0.05)
CONCENTRATION_PENALTY: float = _env_float("CONCENTRATION_PENALTY", 0.10)

# Research portfolios are user-defined 6-stock baskets whose benchmark in the UI
# is equal-weight. Using market-cap weights as the BL prior can make one name
# dominate the prior (e.g., MSFT >50% of a small basket), which conflicts with
# MAX_WEIGHT/active-budget constraints and makes the optimizer fall back to equal
# weight. Default to an equal-weight prior/benchmark for clean architecture tests.
# Set BL_PRIOR_MODE=market_cap only when you intentionally want a cap-weighted
# miniature market portfolio.
BL_PRIOR_MODE: str = _env_str("BL_PRIOR_MODE", "equal_weight").lower()
# Backward-compatible legacy constant. The active research path uses the explicit
# DEFAULT_Q_MULTIPLIER below and a learned calibration file.
GLOBAL_VIEW_SHRINKAGE: float = _env_float("GLOBAL_VIEW_SHRINKAGE", 0.50)

# View calibration. The model emits raw stock-level views. A learned calibration
# file maps raw q/confidence into BL q/confidence. Without the file, conservative
# defaults are used.
VIEW_CALIBRATION_PATH: str = _env_str("VIEW_CALIBRATION_PATH", "")
CALIBRATION_MIN_SAMPLES: int = _env_int("CALIBRATION_MIN_SAMPLES", 30)
DEFAULT_Q_MULTIPLIER: float = _env_float("DEFAULT_Q_MULTIPLIER", 0.25)
MIN_Q_MULTIPLIER: float = _env_float("MIN_Q_MULTIPLIER", 0.0)
MAX_Q_MULTIPLIER: float = _env_float("MAX_Q_MULTIPLIER", 0.75)
CALIBRATION_Q_SHRINK_N: int = _env_int("CALIBRATION_Q_SHRINK_N", 150)
CALIBRATION_WINSOR_LOW: float = _env_float("CALIBRATION_WINSOR_LOW", 0.025)
CALIBRATION_WINSOR_HIGH: float = _env_float("CALIBRATION_WINSOR_HIGH", 0.975)
CALIBRATION_REALIZED_Y_CAP: float = _env_float("CALIBRATION_REALIZED_Y_CAP", 1.0)
CALIBRATION_RIDGE_FRACTION: float = _env_float("CALIBRATION_RIDGE_FRACTION", 0.10)
CALIBRATION_FULL_SKILL_HIT_EDGE: float = _env_float("CALIBRATION_FULL_SKILL_HIT_EDGE", 0.10)
CALIBRATION_FULL_SKILL_RANK_IC: float = _env_float("CALIBRATION_FULL_SKILL_RANK_IC", 0.20)
CALIBRATION_HIT_SKILL_WEIGHT: float = _env_float("CALIBRATION_HIT_SKILL_WEIGHT", 0.50)
MIN_VIEW_CONFIDENCE: float = _env_float("MIN_VIEW_CONFIDENCE", 0.05)
MAX_VIEW_CONFIDENCE: float = _env_float("MAX_VIEW_CONFIDENCE", 0.80)
DEFAULT_CONFIDENCE_CENTER: float = _env_float("DEFAULT_CONFIDENCE_CENTER", 0.55)
DEFAULT_CONFIDENCE_SLOPE: float = _env_float("DEFAULT_CONFIDENCE_SLOPE", 0.10)
CONFIDENCE_CENTER_SHRINK_N: int = _env_int("CONFIDENCE_CENTER_SHRINK_N", 100)
MAX_CONFIDENCE_SLOPE: float = _env_float("MAX_CONFIDENCE_SLOPE", 0.50)
CONFIDENCE_SLOPE_SHRINK_N: int = _env_int("CONFIDENCE_SLOPE_SHRINK_N", 100)
VIEW_SPREAD_FLOOR: float = _env_float("VIEW_SPREAD_FLOOR", 0.0025)
MAX_PAIRWISE_VIEWS: int = _env_int("MAX_PAIRWISE_VIEWS", 0)  # 0 = automatic n-1 adjacent chain

# Neutral band on the BL view return q: |q| < band => the view is neutral and q is
# forced to 0, so a view's direction label can never disagree with its q (spec A3).
NEUTRAL_BAND: float = _env_float("NEUTRAL_BAND", 0.02)

# Idzorek omega floor so a view can never have zero uncertainty.
MIN_OMEGA: float = _env_float("MIN_OMEGA", 1e-8)

# --------------------------------------------------------------------------- #
# debate constants
# --------------------------------------------------------------------------- #
CONSENSUS_ACCEPT_THRESHOLD: int = _env_int("CONSENSUS_ACCEPT_THRESHOLD", 3)  # >=3 of 4
MAX_DEBATE_ROUNDS: int = _env_int("MAX_DEBATE_ROUNDS", 3)

# Rebuttal/consensus integrity controls. Invalid model output must not masquerade
# as a valid frozen opinion, and disagreement must not automatically become q=0.
REBUTTAL_MAX_RETRIES: int = _env_int("REBUTTAL_MAX_RETRIES", 2)
INVALID_REBUTTAL_WEIGHT: float = _env_float("INVALID_REBUTTAL_WEIGHT", 0.25)
NEUTRAL_SUPPORT_MIN: int = _env_int("NEUTRAL_SUPPORT_MIN", 2)
NEUTRAL_EVIDENCE_BAND: float = _env_float("NEUTRAL_EVIDENCE_BAND", 0.01)
BASE_ACCEPTANCE_TOLERANCE: float = _env_float("BASE_ACCEPTANCE_TOLERANCE", 0.01)
# By default, debate is selective: it opens only when eligible specialist views
# straddle long and short. Same-direction magnitude disagreement is handled by
# Phase-1 averaging + calibration instead of three expensive debate rounds.
# Set ENABLE_Q_ONLY_DEBATE=1 for an explicit ablation; the spread threshold is in
# annualized-return q units.
ENABLE_Q_ONLY_DEBATE: bool = _env_bool("ENABLE_Q_ONLY_DEBATE", False)
DEBATE_DISAGREEMENT_Q: float = _env_float("DEBATE_DISAGREEMENT_Q", 0.10)

# Until source-specific debate calibration is available, a debate candidate is
# a bounded update to the already-grounded Phase-1 view rather than a wholesale
# replacement. Accepted agent consensus receives more weight than an end-only
# moderator fallback that exists precisely because the agents did not converge.
DEBATE_ACCEPTED_UPDATE_ALPHA: float = _env_float(
    "DEBATE_ACCEPTED_UPDATE_ALPHA", 0.50
)
MODERATOR_UPDATE_ALPHA: float = _env_float("MODERATOR_UPDATE_ALPHA", 0.25)
MODERATOR_CONFIDENCE_CAP: float = _env_float("MODERATOR_CONFIDENCE_CAP", 0.35)

# --------------------------------------------------------------------------- #
# news / sentiment plumbing
# --------------------------------------------------------------------------- #
SENTIMENT_RELEVANCE_FLOOR: float = _env_float("SENTIMENT_RELEVANCE_FLOOR", 0.80)
NEWS_LOOKBACK_DAYS: int = _env_int("NEWS_LOOKBACK_DAYS", 30)
NEWS_MAX_ARTICLES_PER_TICKER: int = _env_int("NEWS_MAX_ARTICLES_PER_TICKER", 12)
ALPHAVANTAGE_MIN_SECONDS: float = _env_float("ALPHAVANTAGE_MIN_SECONDS", 12.0)
FINNHUB_API_KEY: str = _env_str("FINNHUB_API_KEY", "")

# --------------------------------------------------------------------------- #
# validator
# --------------------------------------------------------------------------- #
HALLUCINATION_TOL: float = _env_float("HALLUCINATION_TOL", 0.05)  # 5% relative

# --------------------------------------------------------------------------- #
# factor model -- Kenneth French 5 factors (daily source, monthly regression)
# --------------------------------------------------------------------------- #
# Five regressors + intercept over 60 months gives materially more stable
# estimates than the previous 15-factor/36-month specification.
FACTOR_LOOKBACK_MONTHS: int = _env_int("FACTOR_LOOKBACK_MONTHS", 60)
FACTOR_MIN_OBS: int = _env_int("FACTOR_MIN_OBS", 36)
FACTOR_SIGNAL_P_MAX: float = _env_float("FACTOR_SIGNAL_P_MAX", 0.10)
FACTOR_HAC_LAGS: int = _env_int("FACTOR_HAC_LAGS", 3)
COV_WINDOW_DAYS: int = _env_int("COV_WINDOW_DAYS", 63)

# RF is required to construct stock excess returns, but it is not a regressor.
FF5_FACTORS: List[str] = ["mkt_rf", "smb", "hml", "rmw", "cma"]
FF5_STYLE_FACTORS: List[str] = ["smb", "hml", "rmw", "cma"]
RISK_FREE_FACTOR: str = "rf"

# --------------------------------------------------------------------------- #
# observability / Phoenix
# --------------------------------------------------------------------------- #
PHOENIX_ENABLED: bool = _env_bool("PHOENIX_ENABLED", False)
PHOENIX_PROJECT_NAME: str = _env_str("PHOENIX_PROJECT_NAME", "agentic-slm-research")
PHOENIX_COLLECTOR_ENDPOINT: str = _env_str("PHOENIX_COLLECTOR_ENDPOINT", "")
PHOENIX_OTEL_PROTOCOL: str = _env_str("PHOENIX_OTEL_PROTOCOL", "http/protobuf")
PHOENIX_TRACE_INPUTS: bool = _env_bool("PHOENIX_TRACE_INPUTS", True)
PHOENIX_TRACE_OUTPUTS: bool = _env_bool("PHOENIX_TRACE_OUTPUTS", True)

# --------------------------------------------------------------------------- #
# backtest
# --------------------------------------------------------------------------- #
TXN_COST_BPS: float = _env_float("TXN_COST_BPS", 10.0)  # round-trip bps on turnover
BENCHMARK: str = _env_str("BENCHMARK", "SPY")
BENCHMARK_YF_TICKER: str = _env_str("BENCHMARK_YF_TICKER", "SPY")
BENCHMARK_START_DATE: str = _env_str("BENCHMARK_START_DATE", "2000-01-01")
START_DATE: str = _env_str("START_DATE", "2025-01-31")
END_DATE: str = _env_str("END_DATE", "2026-03-31")

# --------------------------------------------------------------------------- #
# universe  (6 stocks, one per sector; Bloomberg ticker -> GICS sector)
# --------------------------------------------------------------------------- #
# Keys are display tickers; UNIVERSE_BBG maps them to the Bloomberg identifiers
# the LiveProvider expects.
UNIVERSE: List[str] = ["MSFT", "META", "V", "UNH", "CVX", "CAT"]

# Complete point-in-time research universe. ``UNIVERSE`` above remains the default
# six-stock basket, while the maps below must cover every ticker that can be passed
# dynamically by the UI or batch runner. Keeping only the current six names here
# caused all other batch portfolios to receive sector="Unknown" and made the
# sector-cap constraints infeasible.
UNIVERSE_BBG: Dict[str, str] = {
    ticker: f"{ticker} US EQUITY"
    for ticker in [
        "GOOGL", "NFLX", "META", "DIS", "TMUS",
        "AMZN", "TSLA", "HD", "MCD", "LOW",
        "PG", "KO", "PEP", "COST", "WMT",
        "XOM", "CVX", "COP", "SLB",
        "JPM", "BAC", "GS", "MS", "V", "MA",
        "UNH", "JNJ", "MRK", "LLY", "ABBV", "TMO",
        "CAT", "GE", "BA", "RTX", "HON", "UPS",
        "AAPL", "MSFT", "NVDA", "AVGO", "AMD", "ORCL", "CRM",
        "LIN", "SHW", "APD", "FCX",
        "PLD", "AMT", "EQIX",
        "NEE", "DUK", "SO",
    ]
}

SECTOR_BY_TICKER: Dict[str, str] = {
    # Communication Services
    "GOOGL": "Communication Services",
    "NFLX": "Communication Services",
    "META": "Communication Services",
    "DIS": "Communication Services",
    "TMUS": "Communication Services",

    # Consumer Discretionary
    "AMZN": "Consumer Discretionary",
    "TSLA": "Consumer Discretionary",
    "HD": "Consumer Discretionary",
    "MCD": "Consumer Discretionary",
    "LOW": "Consumer Discretionary",

    # Consumer Staples
    "PG": "Consumer Staples",
    "KO": "Consumer Staples",
    "PEP": "Consumer Staples",
    "COST": "Consumer Staples",
    "WMT": "Consumer Staples",

    # Energy
    "XOM": "Energy",
    "CVX": "Energy",
    "COP": "Energy",
    "SLB": "Energy",

    # Financials
    "JPM": "Financials",
    "BAC": "Financials",
    "GS": "Financials",
    "MS": "Financials",
    "V": "Financials",
    "MA": "Financials",

    # Health Care
    "UNH": "Health Care",
    "JNJ": "Health Care",
    "MRK": "Health Care",
    "LLY": "Health Care",
    "ABBV": "Health Care",
    "TMO": "Health Care",

    # Industrials
    "CAT": "Industrials",
    "GE": "Industrials",
    "BA": "Industrials",
    "RTX": "Industrials",
    "HON": "Industrials",
    "UPS": "Industrials",

    # Information Technology
    "AAPL": "Information Technology",
    "MSFT": "Information Technology",
    "NVDA": "Information Technology",
    "AVGO": "Information Technology",
    "AMD": "Information Technology",
    "ORCL": "Information Technology",
    "CRM": "Information Technology",

    # Materials
    "LIN": "Materials",
    "SHW": "Materials",
    "APD": "Materials",
    "FCX": "Materials",

    # Real Estate
    "PLD": "Real Estate",
    "AMT": "Real Estate",
    "EQIX": "Real Estate",

    # Utilities
    "NEE": "Utilities",
    "DUK": "Utilities",
    "SO": "Utilities",
}
DEFAULT_PORTFOLIO_ID: str = _env_str("PORTFOLIO_ID", "demo6")
DEFAULT_HORIZON: str = _env_str("HORIZON", "1M")
DEFAULT_AS_OF: str = _env_str("AS_OF_DATE", "2024-09-30")

# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")

# Live data cache. This caches only point-in-time input data pulls, not model
# outputs and not portfolio weights. Delete this directory any time you want a
# completely fresh data pull.
LIVE_DATA_CACHE: bool = _env_bool("LIVE_DATA_CACHE", True)
LIVE_CACHE_DIR: str = _env_str("LIVE_CACHE_DIR", os.path.join(PROJECT_ROOT, ".cache", "live_data"))

# Where the LiveProvider finds the ported fetch scripts (the drive-download dir).
LIVE_SCRIPTS_DIR: str = _env_str(
    "LIVE_SCRIPTS_DIR",
    os.path.normpath(os.path.join(PROJECT_ROOT, "live_data")),
)
# Raw Kenneth French daily FF5 ZIP or a normalized CSV with
# date,mkt_rf,smb,hml,rmw,cma,rf. Override for local/Colab layouts.
FACTOR_DATA_PATH: str = _env_str(
    "FACTOR_DATA_PATH",
    os.path.join(
        LIVE_SCRIPTS_DIR,
        "quant",
        "data",
        "F-F_Research_Data_5_Factors_2x3_daily.csv",
    ),
)
FRED_API_KEY: str = _env_str("FRED_API_KEY", "")

def sector_of(ticker: str) -> str:
    return SECTOR_BY_TICKER.get(ticker, "Unknown")

def temperature_for_call(role: str, *, phase: str = "", debate_round: int = 0, max_rounds: int = 3) -> float:
    """Return deterministic, role/phase-aware temperature.

    Keep this centralized so backtests are reproducible and ablations can override
    individual stages without changing orchestration code.
    """
    if role == "validator":
        return TEMP_VALIDATOR
    if role == "moderator":
        return TEMP_MODERATOR
    if role in {"single_gemini", "single_gemma", "single_model"}:
        return TEMP_SINGLE_MODEL
    if phase == "phase1":
        return TEMP_SPECIALIST_PHASE1
    if phase == "debate":
        if max_rounds <= 1:
            return TEMP_ONE_ROUND_DEBATE
        if debate_round >= max_rounds:
            return TEMP_DEBATE_FINAL
        if debate_round == 2:
            return TEMP_DEBATE_ROUND2
        return TEMP_DEBATE_ROUND1
    return MODEL_TEMPERATURE

def summary() -> Dict[str, object]:
    """Snapshot of the active config -- embedded in every run JSON for provenance."""
    return {
        "GEMMA_MODEL": GEMMA_MODEL,
        "GEMINI_MODEL": GEMINI_MODEL,
        "RUN_SINGLE_GEMINI_C1": RUN_SINGLE_GEMINI_C1,
        "STRICT_MODEL": STRICT_MODEL,
        "MODEL_SEED": MODEL_SEED,
        "PYTHON_RANDOM_SEED": PYTHON_RANDOM_SEED,
        "NUMPY_RANDOM_SEED": NUMPY_RANDOM_SEED,
        "ANNUAL_RETURN_SCALE": ANNUAL_RETURN_SCALE,
        "BL_TAU": BL_TAU,
        "RISK_AVERSION": RISK_AVERSION,
        "MIN_WEIGHT": MIN_WEIGHT,
        "MAX_WEIGHT": MAX_WEIGHT,
        "MAX_ACTIVE_DEVIATION": MAX_ACTIVE_DEVIATION,
        "ACTIVE_WEIGHT_BUDGET": ACTIVE_WEIGHT_BUDGET,
        "BENCHMARK_PENALTY": BENCHMARK_PENALTY,
        "BL_PRIOR_MODE": BL_PRIOR_MODE,
        "CONCENTRATION_PENALTY": CONCENTRATION_PENALTY,
        "GLOBAL_VIEW_SHRINKAGE": GLOBAL_VIEW_SHRINKAGE,
        "VIEW_CALIBRATION_PATH": VIEW_CALIBRATION_PATH,
        "CALIBRATION_MIN_SAMPLES": CALIBRATION_MIN_SAMPLES,
        "DEFAULT_Q_MULTIPLIER": DEFAULT_Q_MULTIPLIER,
        "MIN_Q_MULTIPLIER": MIN_Q_MULTIPLIER,
        "MAX_Q_MULTIPLIER": MAX_Q_MULTIPLIER,
        "CALIBRATION_Q_SHRINK_N": CALIBRATION_Q_SHRINK_N,
        "CALIBRATION_WINSOR_LOW": CALIBRATION_WINSOR_LOW,
        "CALIBRATION_WINSOR_HIGH": CALIBRATION_WINSOR_HIGH,
        "CALIBRATION_REALIZED_Y_CAP": CALIBRATION_REALIZED_Y_CAP,
        "CALIBRATION_RIDGE_FRACTION": CALIBRATION_RIDGE_FRACTION,
        "CALIBRATION_FULL_SKILL_HIT_EDGE": CALIBRATION_FULL_SKILL_HIT_EDGE,
        "CALIBRATION_FULL_SKILL_RANK_IC": CALIBRATION_FULL_SKILL_RANK_IC,
        "CALIBRATION_HIT_SKILL_WEIGHT": CALIBRATION_HIT_SKILL_WEIGHT,
        "MIN_VIEW_CONFIDENCE": MIN_VIEW_CONFIDENCE,
        "MAX_VIEW_CONFIDENCE": MAX_VIEW_CONFIDENCE,
        "DEFAULT_CONFIDENCE_CENTER": DEFAULT_CONFIDENCE_CENTER,
        "DEFAULT_CONFIDENCE_SLOPE": DEFAULT_CONFIDENCE_SLOPE,
        "CONFIDENCE_CENTER_SHRINK_N": CONFIDENCE_CENTER_SHRINK_N,
        "MAX_CONFIDENCE_SLOPE": MAX_CONFIDENCE_SLOPE,
        "VIEW_SPREAD_FLOOR": VIEW_SPREAD_FLOOR,
        "MAX_PAIRWISE_VIEWS": MAX_PAIRWISE_VIEWS,
        "SECTOR_CAP": SECTOR_CAP,
        "VOL_CAP": VOL_CAP,
        "CONSENSUS_ACCEPT_THRESHOLD": CONSENSUS_ACCEPT_THRESHOLD,
        "MAX_DEBATE_ROUNDS": MAX_DEBATE_ROUNDS,
        "ENABLE_Q_ONLY_DEBATE": ENABLE_Q_ONLY_DEBATE,
        "DEBATE_DISAGREEMENT_Q": DEBATE_DISAGREEMENT_Q,
        "DEBATE_ACCEPTED_UPDATE_ALPHA": DEBATE_ACCEPTED_UPDATE_ALPHA,
        "MODERATOR_UPDATE_ALPHA": MODERATOR_UPDATE_ALPHA,
        "MODERATOR_CONFIDENCE_CAP": MODERATOR_CONFIDENCE_CAP,
        "HALLUCINATION_TOL": HALLUCINATION_TOL,
        "SENTIMENT_RELEVANCE_FLOOR": SENTIMENT_RELEVANCE_FLOOR,
        "NEWS_LOOKBACK_DAYS": NEWS_LOOKBACK_DAYS,
        "NEWS_MAX_ARTICLES_PER_TICKER": NEWS_MAX_ARTICLES_PER_TICKER,
        "ALPHAVANTAGE_MIN_SECONDS": ALPHAVANTAGE_MIN_SECONDS,
        "FACTOR_LOOKBACK_MONTHS": FACTOR_LOOKBACK_MONTHS,
        "FACTOR_MIN_OBS": FACTOR_MIN_OBS,
        "FACTOR_SIGNAL_P_MAX": FACTOR_SIGNAL_P_MAX,
        "FACTOR_HAC_LAGS": FACTOR_HAC_LAGS,
        "FF5_FACTORS": FF5_FACTORS,
        "FACTOR_DATA_PATH": FACTOR_DATA_PATH,
        "COV_WINDOW_DAYS": COV_WINDOW_DAYS,
        "TXN_COST_BPS": TXN_COST_BPS,
        "BENCHMARK": BENCHMARK,
        "UNIVERSE": UNIVERSE,
    }
