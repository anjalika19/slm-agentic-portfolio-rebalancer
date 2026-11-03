"""
Two coordinated fixes — apply once, then restart the Python process.

  FIX A — providers/live.py + pipeline/state.py + pipeline/phases.py:
          source genuine market-cap weights instead of hardcoded 1/N.

          InputBundle gains a `market_caps: Dict[str, float]` field. LiveProvider
          populates it from (point-in-time shares outstanding x historical price),
          falling back to current shares outstanding x historical price, falling
          back to equal weight with an explicit warning. ensure_market_model() in
          phases.py uses bundle.market_caps when present instead of 1/N.

          Why this matters: with 1/N, pi (the equilibrium prior) is driven ENTIRELY
          by the covariance structure -- a high-vol stock gets a high prior return
          regardless of its actual market size. A true cap-weighted prior reflects
          what the AGGREGATE MARKET believes (Black-Litterman's foundational
          assumption), which is what the debate views are meant to perturb AWAY FROM.
          Without it, "equilibrium" is really "internal equal-weight consensus" --
          a defensible toy but not genuine BL.

  FIX B — agents/prompts/debate.py + agents/prompts/quant.py + fundamental.py +
          sentiment.py + economist.py: reframe what `q` MEANS.

          Agents are currently told q = "expected annualized return for this stock"
          (an absolute-return framing). But finalize() now builds PAIRWISE rows
          using q_long - q_short as the spread. A literal reading of "GOOGL returns
          4.5%" does not mean "GOOGL beats PEP by 10.5%" -- those are different
          claims that happen to combine arithmetically, but the agent never
          reasoned about the SPREAD it is implicitly asserting.

          Fix: reframe q in every relevant prompt as "expected outperformance vs.
          the universe's equal-weight complement" -- i.e. exactly what
          p_row_vs_complement encodes. This makes the agent's mental model match
          the math: when two agents land on opposite sides, the system computes
          their spread, which is now a quantity the agents were ALREADY estimating
          (excess return vs. the rest of the book), not a quantity invented by the
          pairing logic after the fact.
"""

import sys
import pathlib

ROOT = pathlib.Path(__file__).parent


def _patch(path: str, old: str, new: str, label: str) -> bool:
    full = ROOT / path
    src = full.read_text()
    if old not in src:
        print(f"  ⚠️  {label}: pattern not found — skipping (may already be applied)")
        return False
    full.write_text(src.replace(old, new, 1))
    print(f"  ✅ {label}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# FIX A1 — providers/base.py: add market_caps field to InputBundle
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/6] providers/base.py")

_patch(
    "providers/base.py",
    '''    # covariance block: {"order": [...], "matrix": [[...]], "window_days": int,
    #                    "annualized": True}
    covariance: Dict = field(default_factory=dict)
    # provenance / warnings surfaced to the event stream
    warnings: List[str] = field(default_factory=list)''',
    '''    # covariance block: {"order": [...], "matrix": [[...]], "window_days": int,
    #                    "annualized": True}
    covariance: Dict = field(default_factory=dict)
    # market-cap weights for the BL equilibrium prior pi = delta * Sigma * w_cap.
    # ticker -> normalized weight (sums to 1.0). Empty dict -> caller falls back
    # to 1/N with an explicit warning (see ensure_market_model in phases.py).
    market_caps: Dict[str, float] = field(default_factory=dict)
    # provenance / warnings surfaced to the event stream
    warnings: List[str] = field(default_factory=list)''',
    "Add market_caps field to InputBundle",
)

# ─────────────────────────────────────────────────────────────────────────────
# FIX A2 — providers/live.py: fetch market caps, point-in-time where possible
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/6] providers/live.py")

market_cap_method = '''
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
        try:
            import yfinance as yf
        except Exception as e:
            warnings.append(f"market_caps: yfinance unavailable ({e}); using equal weight 1/N")
            n = len(universe)
            return {t: 1.0 / n for t in universe}

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
            return {t: 1.0 / n for t in universe}

        total = sum(caps.values())
        return {t: caps[t] / total for t in universe}

'''

_patch(
    "providers/live.py",
    "    def _fundamental_context(self):",
    market_cap_method + "    def _fundamental_context(self):",
    "Add _market_cap_weights method",
)

# Wire it into get_input_bundle's return
_patch(
    "providers/live.py",
    '''        return InputBundle(
            portfolio_id=portfolio_id, as_of_date=as_of_date, horizon=horizon,
            universe=list(universe), sectors=sectors, packages=packages,
            covariance=cov_block, warnings=warnings,
        )''',
    '''        market_caps = self._market_cap_weights(universe, as_of_date, warnings)
        return InputBundle(
            portfolio_id=portfolio_id, as_of_date=as_of_date, horizon=horizon,
            universe=list(universe), sectors=sectors, packages=packages,
            covariance=cov_block, market_caps=market_caps, warnings=warnings,
        )''',
    "Wire market_caps into InputBundle construction",
)

# ─────────────────────────────────────────────────────────────────────────────
# FIX A3 — pipeline/phases.py: ensure_market_model uses bundle.market_caps
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/6] pipeline/phases.py")

_patch(
    "pipeline/phases.py",
    '''    mw = state.bundle  # market weights provider default = 1/N
    n = len(state.universe_order)
    state.market_weights = {t: 1.0 / n for t in state.universe_order}
    w = np.array([state.market_weights[t] for t in state.universe_order])
    state.pi = bl_core.equilibrium_returns(state.sigma, w)''',
    '''    caps = state.bundle.market_caps
    n = len(state.universe_order)
    if caps and all(t in caps for t in state.universe_order):
        state.market_weights = {t: caps[t] for t in state.universe_order}
        ctx.emitter.emit(
            "baseline_view", phase="market_model",
            payload={"type": "market_weights", "source": "market_cap",
                     "weights": {t: round(state.market_weights[t], 4) for t in state.universe_order}},
        )
    else:
        state.market_weights = {t: 1.0 / n for t in state.universe_order}
        ctx.emitter.warning(
            "market_caps unavailable or incomplete; equilibrium prior pi uses 1/N "
            "(this is an internal equal-weight consensus prior, not a genuine market-cap "
            "BL prior -- see docs)",
            phase="market_model",
        )
    w = np.array([state.market_weights[t] for t in state.universe_order])
    state.pi = bl_core.equilibrium_returns(state.sigma, w)''',
    "Use bundle.market_caps for the equilibrium prior; fall back to 1/N with explicit warning",
)

# ─────────────────────────────────────────────────────────────────────────────
# FIX B1 — agents/prompts/debate.py: reframe q in opening + rebuttal prompts
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/6] agents/prompts/debate.py")

_patch(
    "agents/prompts/debate.py",
    '''Output strict JSON only. Schema:
{{
  "agent_name": "{agent_name}",
  "ticker": "<ticker>",
  "direction": "long" | "neutral" | "short",
  "q": <float: your preferred annualized excess return>,
  "confidence": <float 0.0-1.0>,
  "position_rationale": "<your evidence and reasoning for this round>",
  "objection_to_current_view": "<why the current BL view is wrong, if you disagree>",
  "evidence": [
    {{"name": "<metric>", "value": <float>, "note": "<why this matters>"}}
  ],
  "what_would_change_my_mind": "<what peer evidence would shift your view>"
}}
Do not produce portfolio weights. Output strict JSON only.
"""''',
    '''IMPORTANT — WHAT q MEANS:
q is NOT this stock's expected absolute return. q is the stock's expected
OUTPERFORMANCE versus the equal-weight average of the other names in the
portfolio over the next month, expressed as an ANNUALIZED rate.

  q = +0.06  ->  "I expect this stock to beat the rest of the book by 6%/yr"
  q = -0.04  ->  "I expect this stock to LAG the rest of the book by 4%/yr"
  q = 0.0    ->  "I expect this stock to perform in line with the rest"

This matters because when you and another agent disagree -- one long, one
short, on DIFFERENT tickers -- the system pairs your views and computes the
SPREAD between them. If you are estimating an absolute return number instead
of a relative one, that spread will not mean what you intended. Always anchor
your q to "better or worse than the rest of this specific 6-stock book," not
to some external absolute benchmark.

Output strict JSON only. Schema:
{{
  "agent_name": "{agent_name}",
  "ticker": "<ticker>",
  "direction": "long" | "neutral" | "short",
  "q": <float: expected annualized OUTPERFORMANCE vs the rest of the portfolio>,
  "confidence": <float 0.0-1.0>,
  "position_rationale": "<your evidence and reasoning for this round>",
  "objection_to_current_view": "<why the current BL view is wrong, if you disagree>",
  "evidence": [
    {{"name": "<metric>", "value": <float>, "note": "<why this matters>"}}
  ],
  "what_would_change_my_mind": "<what peer evidence would shift your view>"
}}
Do not produce portfolio weights. Output strict JSON only.
"""''',
    "Reframe q as relative outperformance in opening_statement_prompt",
)

_patch(
    "agents/prompts/debate.py",
    '''Your output must be valid JSON matching this schema exactly:

{{
  "agent_name": "<your role>",
  "ticker": "<ticker>",
  "revised_direction": "long" | "neutral" | "short",
  "revised_q": <float, your updated expected return>,
  "revised_confidence": <float 0.0-1.0>,''',
    '''Reminder: revised_q is expected OUTPERFORMANCE vs the rest of the portfolio's
equal-weight average (not an absolute return). +0.06 means "beats the book by
6%/yr"; -0.04 means "lags the book by 4%/yr."

Your output must be valid JSON matching this schema exactly:

{{
  "agent_name": "<your role>",
  "ticker": "<ticker>",
  "revised_direction": "long" | "neutral" | "short",
  "revised_q": <float, expected OUTPERFORMANCE vs the rest of the portfolio>,
  "revised_confidence": <float 0.0-1.0>,''',
    "Reframe revised_q as relative outperformance in rebuttal_statement_prompt",
)

# ─────────────────────────────────────────────────────────────────────────────
# FIX B2 — moderator synthesis prompt: same reframing for consistency
# ─────────────────────────────────────────────────────────────────────────────
_patch(
    "agents/prompts/debate.py",
    '''Important:
- Do not average away disagreement.
- Classify the conflict as factual, horizon/weighting, or irreducible.''',
    '''Important:
- q values from every agent represent expected OUTPERFORMANCE vs the rest of the
  portfolio's equal-weight average, not absolute returns. Your recommended_q must
  use the same framing.
- Do not average away disagreement.
- Classify the conflict as factual, horizon/weighting, or irreducible.''',
    "Reframe q in MODERATOR_SYNTHESIS_PROMPT",
)

# ─────────────────────────────────────────────────────────────────────────────
# FIX B3 — specialist Phase-1 prompts: add the same q framing note
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/6] agents/prompts/{quant,fundamental,sentiment,economist}.py")

Q_FRAMING_NOTE = '''
NOTE ON `signal`: think of your signal as relative conviction within THIS universe
-- "stronger than the other 5 names" (positive) or "weaker than the other 5"
(negative), not an absolute forecast against the broader market. This keeps your
Phase-1 signal consistent with how debate-stage q values are interpreted later in
the pipeline (relative outperformance vs the rest of the portfolio).
'''

for fname in ["quant.py", "fundamental.py", "sentiment.py", "economist.py"]:
    path = f"agents/prompts/{fname}"
    full = ROOT / path
    src = full.read_text()
    if "NOTE ON `signal`" in src:
        print(f"  ⚠️  {fname}: q framing note already present — skipping")
        continue
    # Insert right after SYSTEM_PROMPT's closing of _SCHEMA_BACKBONE, i.e. just
    # before the final closing triple-quote of the assembled SYSTEM_PROMPT. We
    # do this generically by inserting before the LAST occurrence of '"""' in
    # the file (the end of the SYSTEM_PROMPT string).
    idx = src.rfind('"""')
    if idx == -1:
        print(f"  ⚠️  {fname}: could not find SYSTEM_PROMPT terminator — skipping")
        continue
    new_src = src[:idx] + Q_FRAMING_NOTE + src[idx:]
    full.write_text(new_src)
    print(f"  ✅ {fname}: added q framing note")

# ─────────────────────────────────────────────────────────────────────────────
# Clear module cache
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/6] clearing module cache")
for mod in list(sys.modules.keys()):
    if any(x in mod for x in ["bl_core", "pipeline", "agents", "providers", "config", "events"]):
        del sys.modules[mod]
print("  ✅ cleared")

print("""
══════════════════════════════════════════════════════
DONE. Summary:

  FIX A — genuine market-cap BL prior
    · InputBundle gains market_caps: Dict[str, float]
    · LiveProvider._market_cap_weights(): PiT shares x historical price,
      falling back to current shares x historical price, falling back to
      1/N with an explicit warning at every fallback step
    · ensure_market_model() now uses bundle.market_caps when complete;
      falls back to 1/N with a loud warning otherwise (never silent)

  FIX B — q reframed as relative outperformance everywhere
    · opening_statement_prompt / rebuttal_statement_prompt / 
      MODERATOR_SYNTHESIS_PROMPT all now define q = outperformance vs the
      rest of the portfolio's equal-weight average
    · All 4 specialist Phase-1 prompts get a consistency note tying their
      `signal` field to the same relative framing

Why both fixes together matter:
  Before: pi was an "internal consensus" (1/N x Sigma) and agents reasoned
  in absolute-return space while the math used relative spreads. Two
  separate sources of mismatch between what the system computes and what
  it claims to represent.
  After: pi is a genuine market-cap prior (or explicitly flagged as 1/N
  when caps are unavailable), and every agent's q is the SAME quantity the
  pairwise/vs-complement rows already encode. The framework's math and its
  prompts are now describing the same model.

Caveat to flag in the paper:
  Shares outstanding for historical dates is approximate (yfinance's
  get_shares_full when available; current shares as fallback). This is a
  much smaller bias than 1/N's structural mismatch, but it is not a paid-
  data-grade point-in-time market cap. Document this in the limitations
  section alongside the SFT teacher-bias caveat already there.
══════════════════════════════════════════════════════
""")