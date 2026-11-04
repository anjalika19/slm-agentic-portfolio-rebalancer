# Design notes

This document holds the detail behind the [README](../README.md): agent output shapes, the debate protocol, how Black-Litterman views are built, and the correctness fixes the project was rebuilt to make. File references are relative to `src/agentic_bl/`.

## 1. Agent outputs (Phase 1)

Each specialist emits a **categorical** conviction, agent-specific fields and a rationale. No agent emits `q`, `P` or `Ω`; those first appear when Python maps the consensus to BL views. Schemas are in `schemas.py`, and the mapping to numbers is in `agents/normalize.py`.

| Agent | Emission |
|---|---|
| **Economist** (sector-level, never single stocks) | `risk_regime`, `rate_environment`, `regime_rationale`, `sector_views{sector: stance}`, `favored_sectors`, `risk_sectors` |
| **Quant** | `stock_signals{ticker: {signal, primary_factors, factor_alignment, alpha_contribution, rationale, cited_values}}` + `portfolio_level_view` |
| **Fundamental** | `{signal, valuation_call, quality_tier, key_drivers, rationale, cited_values}` |
| **Sentiment** | `{signal, primary_catalyst, event_driven, rationale, cited_values}` |

- **Signal scale.** The signal is 5-way: `strongly_long | long | neutral | underweight | avoid`. The normaliser maps it to the numeric tilt the BL math needs and derives confidence from each agent's flags. The raw emission is kept for the UI.
- **Sentiment input.** In live mode Sentiment reads AlphaVantage `NEWS_SENTIMENT` headlines plus a yfinance analyst block; in fixture mode it reads synthetic equivalents. Its `event_driven` flag separates hard catalysts from drifting opinion, and opinion-only views get a confidence cap.
- **Quant factor model.** Quant runs a rolling 60-month Fama-French five-factor regression with HAC errors (`factor_model.py`).
  - The dependent variable is each stock's monthly return minus RF. The regressors are Mkt-RF, SMB, HML, RMW and CMA.
  - Mkt-RF counts as systematic risk; only SMB, HML, RMW and CMA count as style support.
  - The prompt tells the agent to treat recent factor interactions as evidence about the current regime, not as a promise they will continue next month.

## 2. Debate protocol

Tickers where the Phase-1 directions conflict are debated for up to `MAX_DEBATE_ROUNDS = 3` rounds (`pipeline/phases.py`). Each round is a different information operation:

| Round | Job | What agents must do |
|---|---|---|
| R1 | Information transfer | Say which peer facts are new to them and update on those |
| R2 | Forward positioning | Take a next-month position and name the peers they agree or disagree with, grounded in cited facts |
| R3 | Classification + tiebreak | Classify the remaining conflict as *factual*, *horizon/weighting* or *irreducible* |

- **Tiebreak** (`_evidence_gated_resolution`). A weighting conflict can be settled by a context agent (the Economist's regime or the Sentiment catalyst), subject to three rules:
  - **Conjunction rule.** The regime may tilt the result only when it lines up with a matching stock-specific fact.
  - **Conflict of interest.** A context agent that is itself one of the disputants can't adjudicate.
  - **No fake tiebreaks.** If no grounded tiebreaker exists, the result is labelled a low-conviction weighted average, not passed off as a tiebreak.
- **Anti-conformity.** High-conviction agents push back on the majority, so genuine conflicts reach R3 instead of being smoothed away. Changing direction requires citing a new fact; otherwise the change is reverted as sycophancy (`_guard_sycophancy`).
- **Roundtable transcript.** In rebuttals each agent answers its one or two most-disagreeing peers *by name* (`replies: [{responding_to, stance, quoted_claim, message}]`). Every reply is emitted as an ordered `agent_message` event, and the UI replays these as a chronological council transcript.
- **Thin moderator.** The moderator aggregates the agents' positions weighted by confidence and authors no `q` of its own. It is only called at the end (`force_consensus_lowconf`), and only if the remaining revised positions still contain both a long and a short.

## 3. Black-Litterman views

- **From stock views to BL rows.** Consensus stock views are calibrated into expected-return levels, sorted, and turned into an *independent adjacent chain* of pairwise rows: rank 1 > rank 2, rank 2 > rank 3, and so on (`calibration.ticker_views_to_pairwise_bl_views`).
- **Small spreads are dropped.** Adjacent pairs whose spread falls below `VIEW_SPREAD_FLOOR` are left out, because a `Q = 0` row isn't neutral: it actively asserts that the two returns are equal.
- **No basket rows.** Rows of the form "X beats the whole basket" are rejected (`bl_core.validate_view_rows`). They would imply every name beats every other, which is inconsistent.
- **The posterior** (`bl_core.bl_posterior`):

  ```
  E[R] = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹ [(τΣ)⁻¹Π + PᵀΩ⁻¹Q],   Π = δΣw,   τ = 0.05, δ = 2.5
  ```

- **Diagnostics.** After the posterior, the code logs a parametric VaR₉₅/CVaR₉₅ (`bl_core.parametric_var_cvar`) and the condition number of the view precision. Neither changes the weights.

## 4. Correctness fixes

These are the reasons this is a clean rebuild of the original notebook rather than a port.

| Fix | Where |
|---|---|
| The vote uses each agent's **revised post-debate position**, falling back to the Phase-1 signal only in round 0 | `pipeline/phases.py::_vote_on_view` |
| Ω is tied to the prior the **Idzorek** way, `Ω_ii = (1/c − 1)·(Pᵢ τΣ Pᵢᵀ)` with a floor, instead of an arbitrary `(1−c)²` | `bl_core.py::idzorek_omega` |
| A real constrained QP in cvxpy: maximise `μᵀw − (δ/2)wᵀΣw` subject to min/max weight, active-weight and sector limits; `VOL_CAP` is optional | `bl_core.py::optimize_long_only` |
| Covariance is checked to be symmetric and positive semi-definite, with a ridge only as a last resort; it is **never** silently replaced by the identity, and its condition number is logged | `bl_core.py::build_covariance`, `covariance_from_block` |
| Tickers still unresolved after the last round are settled by the gated moderator or keep their last view, and each case is logged | `pipeline/phases.py::force_consensus_lowconf` |
| No silent mocking: `STRICT_MODEL=1` raises; `0` records `used_mock=true` and emits a warning event | `agents/llm.py::structured_call` |
| An agent can't name itself as its strongest counterargument (it is re-prompted once, then the field is dropped) | `pipeline/phases.py::_guard_self_counterargument` |
| Every `cited_values` number is traced back to the agent's input package (within 5%), giving a per-agent hallucination rate | `pipeline/validator.py` |
| Point-in-time: forward prices never reach the agent packages | `providers/` |

### Hardening from live runs

- **Full-universe coverage.** Every agent must return a signal for every ticker. A missing one triggers one re-prompt, then an explicit `abstain` plus a warning event.
- **Feasible weight cap.** A configured `MAX_WEIGHT` that can't be satisfied for the universe size is raised to a feasible level. The change is reported as an event and in the UI, never applied silently.
- **Label and q agree.** A view's direction is derived from `q` against `NEUTRAL_BAND`, so the label can't contradict the number.
- **Influence consistency.** The moderator's `who_swayed` comes from the same `influenced_by` ledger the UI's influence matrix uses.

## 5. UI

The viewer is `ui/`, served by `server/app.py`. Each agent has a consistent colour: quant blue, fundamental green, sentiment amber, economist purple.

- **Debate.** An office scene, the "debate floor":
  - Each specialist has an illustrated avatar at their own desk, with a monitor matching their speciality, and the moderator stands at a whiteboard showing the ticker and round.
  - The current speaker glows and gets a speech bubble with an evidence chip. The agent being answered gets a dashed ring in the reply's stance colour, and a context-agent tiebreak glows gold.
  - Each desk's nameplate has a meter that tracks that agent's latest q as the debate moves.
  - Round chips and a per-message dot timeline jump anywhere in the debate. The right column shows the latest vote and the full chronological transcript, and the stance cards and influence matrix are in a collapsible panel.
- **Agent inputs.** The exact package each agent saw.
- **Outcome & Portfolio.**
  - Final views alongside π (equilibrium) and μ_BL (posterior).
  - The full P/Q/Ω matrices.
  - Weights vs 1/N.
  - The validator table: hallucination rate per agent, and `used_mock` flags.
- **Backtest.**
  - Equity curve vs the baselines, net of costs.
  - Weight evolution over time.
  - Monthly returns.
  - Performance table: cumulative and annualised return, Sharpe, Sortino, max drawdown, Calmar, turnover.
  - Any month's debate can be opened in the Debate tab.

Each run is saved to `runs/<id>_<date>_<ts>.json`. The server lists these alongside the bundled `examples/`, so any saved run can be replayed without calling a model.

## 6. Configuration

Every constant is in `config.py` and can be overridden with an environment variable. The most useful ones:

- **Models:** `GEMMA_MODEL`, `GEMINI_MODEL`, `STRICT_MODEL`
- **Black-Litterman and optimiser:** `ANNUAL_RETURN_SCALE`, `BL_TAU`, `RISK_AVERSION`, `MAX_WEIGHT`, `SECTOR_CAP`, `VOL_CAP`, `VIEW_SPREAD_FLOOR`
- **Debate and validation:** `CONSENSUS_ACCEPT_THRESHOLD`, `MAX_DEBATE_ROUNDS`, `HALLUCINATION_TOL`
- **Factor model:** `FACTOR_LOOKBACK_MONTHS`, `FACTOR_MIN_OBS`, `FACTOR_SIGNAL_P_MAX`, `FACTOR_HAC_LAGS`, `FACTOR_DATA_PATH`
- **Backtest:** `COV_WINDOW_DAYS`, `TXN_COST_BPS`, `BENCHMARK`, `START_DATE`, `END_DATE`
- **Calibration:** `VIEW_CALIBRATION_PATH`, which points at a file fitted by `scripts/research/learn_calibration.py`
