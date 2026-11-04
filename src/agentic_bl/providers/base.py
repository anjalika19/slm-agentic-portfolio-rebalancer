"""DataProvider interface (spec section 6).

Given ``(portfolio_id, as_of_date, universe)`` a provider returns, point-in-time,
one input package per agent plus the asset covariance matrix. The backtest also
asks the provider for *forward* realized returns -- those are kept strictly
separate from the agent packages so the point-in-time wall is never breached.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class InputBundle:
    """Everything the pipeline needs for one rebalance date."""

    portfolio_id: str
    as_of_date: str
    horizon: str
    universe: List[str]
    sectors: Dict[str, str]
    # one input package per specialist agent, matching the live JSON layout
    packages: Dict[str, Dict] = field(default_factory=dict)
    # covariance block: {"order": [...], "matrix": [[...]], "window_days": int,
    #                    "annualized": True}
    covariance: Dict = field(default_factory=dict)
    # market-cap weights for the BL equilibrium prior pi = delta * Sigma * w_cap.
    # ticker -> normalized weight (sums to 1.0). Empty dict -> caller falls back
    # to 1/N with an explicit warning (see ensure_market_model in phases.py).
    market_caps: Dict[str, float] = field(default_factory=dict)
    # provenance / warnings surfaced to the event stream
    warnings: List[str] = field(default_factory=list)

    def covariance_matrix(self) -> np.ndarray:
        return np.asarray(self.covariance["matrix"], dtype=float)


class DataProvider(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def get_input_bundle(
        self, portfolio_id: str, as_of_date: str, universe: List[str], horizon: str = "1M"
    ) -> InputBundle:
        """Build the point-in-time input packages + covariance for one date."""

    @abc.abstractmethod
    def forward_return(self, ticker: str, start_date: str, end_date: str) -> float:
        """Realized return over the holding period. NEVER exposed to agents."""
    
    def benchmark_forward_return(self, start_date: str, end_date: str) -> float:
        """Realized benchmark/S&P return over the holding period.

        This is used only for backtest comparison and is NEVER exposed to agents.
        Providers should override this when they have a real benchmark series.
        """
        raise NotImplementedError(
            f"{self.name} provider does not implement benchmark_forward_return()"
        )

    def risk_free_return(self, start_date: str, end_date: str) -> float:
        """Realized risk-free return over the holding period.

        Used only for performance metrics. Providers without RF data retain the
        historical zero-rate fallback.
        """
        return 0.0

    def market_weights(self, universe: List[str], as_of_date: str) -> Dict[str, float]:
        """Neutral market portfolio used as the BL equilibrium anchor (default 1/N)."""
        n = len(universe)
        return {t: 1.0 / n for t in universe}

    def baseline_inputs(self, universe: List[str], as_of_date: str):
        """(mu_annual, Sigma_annual) for the baseline portfolios, point-in-time.

        Optional. Providers that cannot supply a returns history return ``None`` and
        the backtest will skip MVO/Risk-Parity for that date (and say so).
        """
        return None
