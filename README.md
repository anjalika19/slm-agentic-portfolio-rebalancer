# Agentic Black-Litterman — multi-agent debate portfolio rebalancer

A from-scratch, multi-agent **Black-Litterman (BL)** monthly rebalancing framework
plus a web UI that makes the agent debate legible: who said what, who influenced
whom, and how the consensus and final portfolio were reached.

The **Economist runs first** (macro/sector level) and feeds its regime + sector
views to the other three specialists (Quant, Fundamental, News-Sentiment). Each
specialist then emits a **categorical** conviction + agent-specific fields +
rationale — **no q/P** at this stage. The moderator maps those to baseline BL views
(this is where q/P/Ω first appear); contested tickers go through an up-to-3-round
debate; the BL posterior + a constrained long-only MVO produce the weights. The
whole flow runs once per month-end in a backtest loop.

Phase-1 emission shapes (categorical, no q/P — see `schemas.py` / `agents/normalize.py`):
- **Economist** (sector-level, never single stocks): `risk_regime`, `rate_environment`,
  `regime_rationale`, `sector_views{sector:{stance}}`, `favored_sectors`, `risk_sectors`.
- **Quant**: `stock_signals{ticker:{signal(5-way), primary_factors, factor_alignment,
  alpha_contribution, rationale, cited_values}}` + `portfolio_level_view`.
- **Fundamental**: `{signal(5-way), valuation_call, quality_tier, key_drivers, rationale, cited_values}`.
- **Sentiment**: `{signal(5-way), primary_catalyst, event_driven, rationale, cited_values}` —
  input includes AlphaVantage NEWS_SENTIMENT headlines + yfinance analyst block
  (live) or realistic synthetic equivalents (fixture). `event_driven` separates hard
  catalysts from opinion drift (opinion-only views are confidence-capped).

The 5-way signal is `strongly_long | long | neutral | underweight | avoid`. A
normalizer maps it to the numeric tilt the BL math needs and derives confidence from
the agent-specific flags, keeping the raw emission for the UI.

### Three distinct debate rounds + context-agent tiebreak (spec A3)

Each round is a *different* information operation (not "do you still agree?" ×3 — that's
the martingale): **R1 information transfer** (which peer facts are new to me?), **R2 forward
positioning + grounded named (dis)agreement**, **R3 disagreement classification + tiebreak**.
At R3 a remaining conflict is classed **factual** / **horizon-weighting** / **irreducible**;
a type-(b) weighting conflict is settled by a **context agent** — the Economist's regime or
the Sentiment catalyst — under a **conjunction rule** (regime tilts only with the aligned
stock-specific fact) and a **conflict-of-interest** guard (a context agent that is itself a
disputant can't adjudicate). The tiebreak shifts q only. A grounded minority survives:
high-conviction agents resist the majority pull (anti-conformity), so genuine conflicts reach
R3 instead of being smoothed away. (`pipeline/phases.py::classify_and_tiebreak`, tested in
`tests/test_tiebreak.py`.)

### Ablations (spec B4) — `ablations.py`

Decomposition controls, all scored by the same forward-accuracy harness:
`full_debate` vs `phase1_only` (does debate add anything?), `single_gemma_all_data`
(specialization vs deliberation), and `sham_debate` (negative control: each agent is fed a
*different* ticker's peer facts — real debate must beat this).
```bash
STRICT_MODEL=0 python ablations.py --start 2024-01-31 --end 2024-09-30
```

### Debate evaluation harness (does the debate add information?)

Outcome Sharpe can't tell you whether the *debate* did anything (a good Sharpe can
come from Phase-1 alone), so `evaluate.py` scores the **views** directly against the
realized next-month return (point-in-time; agents never see it):

- **B1** forward view accuracy, Phase-1 (pre) vs post-debate (post): hit-rate + Brier.
- **B2** correction (wrong→right) vs subversion (right→wrong) + sycophancy rate.
- **B3** per-round hit-rate → a martingale verdict (flat ⇒ cut later rounds).

```bash
STRICT_MODEL=0 python evaluate.py --as-of 2024-06-30      # one month (noisy)
STRICT_MODEL=0 python backtest.py --start ... --end ...   # aggregate B1-B3 across months
```
The backtest prints and persists `debate_evaluation`. Anti-conformity is enforced
upstream: a direction flip needs a new cited fact or it's reverted as sycophancy
(`pipeline/phases.py::_guard_sycophancy`); the moderator is **thin** (code
confidence-weighted aggregation, authors no q of its own).

### Black-Litterman views (absolute + pairwise; basket rows banned)

Only stocks that reach a **directional consensus** in the debate get a view row;
neutral/unresolved stocks are absent from `P` and stay at the prior. Two shapes only:
- **ABSOLUTE** `+1` on the stock, `Q` = expected return level, **wider Ω** (×3) so a
  standalone conviction informs but can't dominate the prior;
- **PAIRWISE** `+1`/`−1` (single names; this universe is one stock per sector, so the
  economist's "Energy > Consumer Staples" is exactly `+1 XOM, −1 PEP`).

"X beats the whole basket" rows are **banned** (`bl_core.validate_view_rows`) — they'd
make every name outperform every other, an inconsistent `Q`. After the posterior, a
**VaR₉₅/CVaR₉₅** risk readout (`bl_core.parametric_var_cvar`, spec D5) is reported (it
does not change the weights). The view-precision condition number is logged (D6).

### Black-Litterman views (legacy note)

The single-stock specialists produce **absolute** views (unit P-row `[0..1..0]`, Q =
the asset's expected return). The Economist's favored-vs-avoided sectors become
**relative** (zero-sum) views: P-row `+1/k` on the favored basket, `-1/m` on the
avoided basket, Q = the spread. Both stack into the **same** posterior, which is the
canonical form (Ω is inverted):

```
E[R] = [(τΣ)^-1 + PᵀΩ^-1 P]^-1 [(τΣ)^-1 Π + PᵀΩ^-1 Q],   Π = δΣw,   τ=0.05, δ=2.5
```

(See `bl_core.py::p_row_relative` and `bl_posterior`. The UI's Outcome tab shows the
full P/Q/Ω matrices, labelling each view absolute or relative.)

### Hardening from live runs (each with a test in `tests/test_live_fixes.py`)

- **Full-universe coverage**: every agent must return a signal for every ticker; a
  missing one triggers one re-prompt, then an explicit `abstain` + a warning event.
- **Feasible weight cap**: `MAX_WEIGHT` defaults to 0.30 (≥ 1/N for the 6-stock book);
  an infeasible configured cap is raised to `max(cfg, 1.5/N)` and reported via a
  visible event + the UI — never silently relaxed.
- **Label↔q consistency**: a view's direction is derived from q against `NEUTRAL_BAND`
  (|q|<band ⇒ neutral and q forced to 0), so the label can't disagree with q.
- **Self-reference guard** on every peer field (strongest_counterargument, responded_to,
  influenced_by, each reply's responding_to).
- **Influence consistency**: the moderator's `who_swayed` is derived from the same
  `influenced_by` ledger the matrix uses.

### Roundtable transcript

The rebuttal step gives each agent the 1-2 most-disagreeing peers and requires direct,
by-name replies (`replies: [{responding_to, stance, quoted_claim, message}]`). Each
reply — plus the moderator's baseline and synthesis — is emitted as an ordered
`agent_message` event, so the UI's Debate tab renders a **chronological council
transcript** per ticker (moderator opens → agents state positions → they answer each
other by name with agree/disagree/partial tint → moderator synthesis → vote tally).
The play/scrubber steps message-by-message. The four stance cards + influence matrix
remain as a secondary collapsible panel.

### Matched 2x2 model--architecture evaluation

The same inference and portfolio code now supports all four experimental cells:

| | Single agent | Multi agent |
|---|---|---|
| **Gemma / SLM** | `gemma_single` | `gemma_debate` |
| **Gemini / LLM** | `gemini_c1` | `gemini_debate` |

`factorial` runs all four configurations on the exact same point-in-time input
bundle for every month. The saved summary includes `factorial_effects`, which
reports the single- and multi-agent scale gaps, each model's architecture gain,
the interaction effect, and absolute/relative gap closure for gross and net
portfolio metrics. `compare` remains available for the legacy Gemma-multi versus
Gemini-single headline comparison.

---

## What's correct here (the fixes that were the point)

These are the things the old Colab notebook got wrong; they are the reason this is
a clean rebuild, not a port of its debate/vote logic.

| Spec | Fix | Where | Test |
|------|-----|-------|------|
| **3a** | The vote consumes each agent's **revised post-debate position** (`revised_positions[round][ticker][agent]`), only falling back to the Phase-1 signal in round 0. | `pipeline/phases.py::_vote_on_view` | `test_debate.py::test_vote_uses_revised_positions_not_phase1` |
| **3b** | Ω is tied to the prior the **Idzorek** way: `Ω_ii = (1/c − 1)·(Pᵢ·τΣ·Pᵢᵀ)`, floored — not `(1−c)²` on an arbitrary scale. | `bl_core.py::idzorek_omega` | `test_bl_core.py::test_high_confidence_view_shifts_weight` |
| **3c** | Real constrained QP (cvxpy): maximise `μᵀw − (δ/2)wᵀΣw` with min/max, active-weight, and sector controls. Absolute volatility is logged; `VOL_CAP` is optional and disabled by default. | `bl_core.py::optimize_long_only` | `tests/test_research_fixes.py` |
| **3d** | Covariance is verified symmetric + PSD (ridge only as last resort), **never** silently identity; condition number logged every run. | `bl_core.py::build_covariance` / `covariance_from_block` | `test_bl_core.py::test_covariance_*` |
| **3e** | Unresolved tickers after MAX rounds are forced to neutral with a logged `consensus_failure` event. | `pipeline/phases.py::force_consensus_failures` | `test_debate.py::test_consensus_failure_forced_and_logged` |
| **3f** | No silent mocking: `STRICT_MODEL=True` raises; `False` records `used_mock=true` + emits a warning. | `agents/llm.py::structured_call` | `test_llm.py` |
| **3g** | An agent's strongest-counterargument-agent may not be itself (re-prompt once, then drop). | `pipeline/phases.py::_guard_self_counterargument` | `test_debate.py::test_self_counterargument_guard` |
| **§4** | Validator traces every `cited_values` number back to the agent's input package within 5% → per-agent hallucination rate. | `pipeline/validator.py` | surfaced in UI |
| **PiT** | Forward prices never reach the agent packages. | providers | `test_pit.py` |

---

## Install

Requires Python 3.11+ (developed on 3.12). A venv is included at `.venv`; to rebuild:

```bash
cd agentic_bl
python3.12 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### Ollama + Gemma (for real runs)

```bash
brew install ollama            # or https://ollama.com/download
ollama serve                   # leave running
ollama pull gemma3             # latest Gemma (no public "gemma4" tag exists)
ollama list                    # confirm the exact tag
export GEMMA_MODEL=gemma4:latest   # or your tag
```

The fixture demo runs **without** Ollama (mock fallback); real runs need it.

---

## Run it

All commands from `agentic_bl/` with the venv active.

```bash
# 1) single-date fixture debate (mock-friendly, no Ollama needed)
STRICT_MODEL=0 python run_single.py

# 1b) single-date with the real model (requires Ollama + Gemma)
STRICT_MODEL=1 GEMMA_MODEL=gemma3:latest python run_single.py

# 2) short fixture backtest (monthly loop, 9 months) + baselines
STRICT_MODEL=0 python backtest.py --start 2024-01-31 --end 2024-09-30

# 2b) complete 2x2 evaluation on identical monthly input bundles
STRICT_MODEL=0 python backtest.py --mode factorial --start 2024-01-31 --end 2024-09-30

# Individual missing controls
STRICT_MODEL=0 python backtest.py --mode gemma_single --start 2024-01-31 --end 2024-09-30
STRICT_MODEL=0 python backtest.py --mode gemini_debate --start 2024-01-31 --end 2024-09-30

# Real Gemini runs require a key;
export GEMINI_API_KEY=...

# 3) the UI (replay recorded runs + live streaming)
uvicorn server.app:app --port 8000
#   open http://localhost:8000

# 4) run tests
python -m pytest -q
```

### Switching to the LiveProvider

```bash
# quant leg runs on the bundled real Bloomberg-exported CSVs (no keys needed);
# economist needs a FRED key; fundamental/sentiment need Bloomberg/news access.
export FRED_API_KEY=...                 # for the macro leg
python run_single.py --provider live --portfolio-id live6
# to fill fundamental/sentiment from the fixture (clearly flagged, NOT real data):
LIVE_ALLOW_FIXTURE_FALLBACK=1 python run_single.py --provider live
```
Without keys/sources the LiveProvider **fails loudly** — it never silently mocks.

---

## The UI (the central deliverable)

`http://localhost:8000` — color-coded per agent everywhere
(quant=blue, fundamental=green, sentiment=amber, economist=purple).

* **Debate tab** — pick a ticker; rounds 1→3 unfold with the four agents side by
  side (direction/q/confidence), evidence, concessions, still-objects and who
  influenced whom; the moderator synthesis; the vote tally (accept/adjust/reject
  with each vote's position source) and whether consensus was reached. Plus an
  **influence matrix** (who swayed whom) and a **convergence strip** (q by round).
* **Outcome tab** — final BL views (P/Q/Ω/direction) with π (equilibrium) next to
  μ_BL (posterior), final weights vs the 1/N equilibrium prior, and the validator
  table (per-agent hallucination rate, `used_mock` flags).
* **Backtest tab** — equity curve vs the three baselines, weight-evolution stacked
  area, and a per-month turnover / consensus-failure strip + a performance table
  (cum/ann return, Sharpe, Sortino, MaxDD, Calmar, turnover).
* **Replay & live** — every run is persisted to `runs/<id>_<date>_<ts>.json`; load
  any saved run and scrub/step/play through it without invoking a model. The
  "▶ Live run" button streams a fresh run over a WebSocket.

**Council view (primary).** The Debate tab is a council canvas: the four specialists +
moderator are characters around a table; the current speaker's avatar lights up with a
one-line speech bubble (direction glyph ▲▼●, one evidence chip), an arrow from the peer
being answered tinted green/red/amber (agree/disagree/partial), a per-round header stating
that round's job, and a **gold glow** when a context agent breaks a tie. The play/scrubber
steps one speech bubble at a time and the council follows whoever is speaking. A convergence
ribbon per ticker shows the direction trajectory (e.g. ▼→●→▲ → resolved). The full
transcript, stance cards and influence matrix are collapsible secondary panels.

Build the UI against replay first; it works with zero models running.

---

## Layout

```
agentic_bl/
  config.py            # all constants (GEMMA_MODEL, scales, caps, STRICT_MODEL, universe...)
  events.py            # event schema + emitter + JSON run-log writer
  schemas.py           # pydantic: specialist output, debate statements, BL views
  bl_core.py           # Omega (3b), BL posterior, cvxpy optimizer (3c), covariance (3d)
  factor_model.py      # FF5 rolling-60m excess-return OLS + HAC factor state
  baselines.py         # 1/N, MVO max-Sharpe, Risk Parity
  metrics.py           # Sharpe/Sortino/MaxDD/Calmar/turnover
  backtest.py          # monthly rebalancing driver + reporting
  agents/
    prompts/           # specialist + debate prompts (ported verbatim; see PORTING NOTEs)
    llm.py             # Ollama/Gemma client, schema enforcement, strict/mock indirection
    mock.py            # deterministic mocks (read real package numbers)
  pipeline/
    phases.py          # phase 1-4 logic (the correctness fixes live here)
    graph.py           # LangGraph orchestrator (debate loop via conditional edge)
    validator.py       # value-tracing validator (§4)
    state.py runner.py
  providers/
    base.py            # DataProvider interface + InputBundle
    fixture.py         # deterministic PSD-cov packages (default, runs anywhere)
    live.py            # real data: quant CSVs + Bloomberg Book1.xlsx fundamentals + FRED + AlphaVantage; fails loudly w/o keys/files
    quant_block.py     # shared quant-package builder (matches 05_input_json_assembler.py)
    synthetic_market.py# self-consistent synthetic market behind the fixture
  server/app.py        # FastAPI: replay endpoints + WebSocket live stream
  ui/                  # index.html + app.js (vanilla, no build step)
  runs/                # saved run JSON logs (one per rebalance date) + backtests
  tests/               # bl_core, factor_model, debate(3a/3e/3g), pit, llm
```

## Quant factor model

The Quant specialist uses the Kenneth French daily five-factor data set. The live
loader accepts the original ZIP or a normalized CSV with
`date,mkt_rf,smb,hml,rmw,cma,rf`. Regressions use monthly stock return minus RF as
the dependent variable and Mkt-RF, SMB, HML, RMW, and CMA as regressors. Mkt-RF is
reported as systematic risk; only SMB/HML/RMW/CMA contribute to style-factor
support. The Quant prompt explicitly treats recent factor interactions as regime
evidence rather than guaranteed next-month continuation.

## Notable config knobs (env-overridable)

`GEMMA_MODEL, STRICT_MODEL, ANNUAL_RETURN_SCALE, BL_TAU, RISK_AVERSION, MAX_WEIGHT,
SECTOR_CAP, VOL_CAP=None, VIEW_SPREAD_FLOOR, CONSENSUS_ACCEPT_THRESHOLD=3, MAX_DEBATE_ROUNDS=3,
HALLUCINATION_TOL=0.05, FACTOR_LOOKBACK_MONTHS=60, FACTOR_MIN_OBS=36,
FACTOR_SIGNAL_P_MAX=0.10, FACTOR_HAC_LAGS=3, FACTOR_DATA_PATH, COV_WINDOW_DAYS=63, TXN_COST_BPS=10, BENCHMARK=SPXT, START_DATE, END_DATE, UNIVERSE`.

> **Note on `MAX_WEIGHT` with 6 names:** the spec's 15% per-name cap is infeasible
> for a 6-stock book (6×0.15 < 1) and 1/6 would force exact equal weight. The
> optimizer detects this and relaxes the per-name cap to a feasible level that
> still binds (`2/n`), logging the relaxation in the optimizer note. With a larger
> universe the 15% cap applies unchanged.
