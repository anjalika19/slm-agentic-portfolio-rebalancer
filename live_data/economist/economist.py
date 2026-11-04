"""
Economist Agent Data Builder
=============================
Pulls all macro data from FRED, derives regime labels,
constructs manually-labeled context fields, and emits
one JSON training example per month-end date.

Output: economist_sft_corpus.jsonl
  - Each line: one complete economist training example
  - model turn is null (filled by Qwen teacher generation)

FRED series used:
  FEDFUNDS    - Fed Funds effective rate (monthly)
  DGS2        - 2Y Treasury yield (daily -> month-end)
  DGS10       - 10Y Treasury yield (daily -> month-end)
  T10Y2Y      - 10Y-2Y spread (daily -> month-end)
  VIXCLS      - VIX (daily -> month-end)
  BAA10Y      - Moody's Baa corporate spread vs 10Y Treasury (daily -> month-end)
  IPMAN       - Industrial Production: Manufacturing (monthly, compute YoY)
  CPIAUCSL    - CPI (monthly, compute YoY)
  UNRATE      - Unemployment rate (monthly)
  DCOILWTICO  - WTI crude spot, USD/bbl (daily -> month-end; level + YoY)
  A191RL1Q225SBEA - Real GDP QoQ annualized (quarterly)

You need a free FRED API key: https://fred.stlouisfed.org/docs/api/api_key.html
Set it as FRED_API_KEY environment variable or pass directly.

--------------------------------------------------------------------------------
CHANGES IN THIS REVISION (see review notes):

1. FACTUAL FIXES to MANUAL_CONTEXT, verified against Federal Reserve primary
   sources (FOMC statements):
     - 2019-08: there was NO August FOMC meeting; the old note invented a cut.
     - 2019-09: cut was to 1.75-2.00% (2nd cut), not "1.5-1.75% / 3 cuts done".
     - 2019-10: cut was to 1.50-1.75% (3rd cut), not "1.25-1.5%".
     - 2014-01: taper stepped to $65B at the Jan 28-29 FOMC (was "$75B").
     - 2014-02: no Feb FOMC; Yellen's first meeting as chair was March, not Feb.

2. GDP LOOK-AHEAD BUG FIXED. A191RL1Q225SBEA is stamped at the QUARTER START,
   so the old `.resample("ME").last().shift(1)` made each quarter's value visible
   ~2 months before its advance estimate was released. We now map each quarterly
   value to the month-end at which it first becomes public (quarter_start +
   GDP_RELEASE_LAG_MONTHS), then forward-fill. No look-ahead.

3. CYCLE RULE IMPROVED. The old rule fired "contraction" on EITHER a single
   manufacturing-IP soft patch OR a sustained curve inversion, which collapsed
   the entire Sep-2022 -> Dec-2024 stretch into "contraction" despite 3%+ GDP.
   v2 tightens this further: contraction now requires an OUTRIGHT real-output
   decline (gdp_qoq < 0). A curve inversion predicts a FUTURE recession and no
   longer forces a present-tense "contraction", which removes the residual
   false positives at 2023-10 and 2024-07 (they become "mid_cycle"). See
   derive_regime().

4. OIL ADDED. DCOILWTICO (WTI spot) is pulled with a 12-month lookback and
   exposed as oil_wti (month-end level) and oil_wti_yoy. Many geopolitical and
   inflation notes reference oil shocks; the teacher can now anchor them in a
   number and inform the Energy stance instead of asserting qualitatively.

5. GDP VERIFICATION DIAGNOSTIC. pull_fred_data() now logs, for the most recent
   quarters, quarter_start -> value -> first-visible month. Cross-check against
   BEA (Q1-2024 = 1.6%, Q2-2024 = 3.0%); if FRED's latest vintage differs from
   the released prints, decide whether to keep the revised vintage before
   spending teacher tokens.

KNOWN LIMITATIONS (disclose in the paper, not bugs):
  - FRED returns the LATEST VINTAGE, not real-time vintages. CPI/IP/UNRATE/GDP
    carry later revisions, so "strict point-in-time" is approximate for the macro
    agent. ALFRED vintage data would be needed for a hard PiT claim.
  - Monthly indicators (CPI/IP/UNRATE) use the calendar-month value, which is not
    released until the following month. This embeds ~2-6 weeks of hindsight for
    those three series. Acceptable at monthly rebalancing frequency, but noted.
  - metadata.regime stores the DERIVED label (pre-teacher). The system prompt lets
    the teacher refine it, so after generation, re-extract the regime from the
    model JSON before scoring "regime label accuracy" in the paper.
--------------------------------------------------------------------------------
"""

import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from fredapi import Fred
import ssl
import urllib.request

# fredapi uses urllib. Some macOS Python builds ship without a CA bundle, so point
# urllib at certifi's bundle when it's available. Certificate verification stays on.
try:
    import certifi

    urllib.request.install_opener(
        urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=ssl.create_default_context(cafile=certifi.where()))
        )
    )
except ImportError:  # pragma: no cover - fall back to the system CA store
    pass

log = logging.getLogger(__name__)

# ── CONFIG ────────────────────────────────────────────────────────────────────

# Read the key from the environment; never commit a real key to source.
FRED_API_KEY  = os.getenv("FRED_API_KEY", "")
TRAIN_START   = "2013-01-31"   # SFT training window start
TRAIN_END     = "2024-12-31"   # SFT training window end
OUT_FILE      = Path(os.getenv("ECON_OUT_FILE", "economist_sft_corpus.jsonl"))

UNIVERSE_SECTORS = [
    "Communication Services", "Consumer Discretionary", "Consumer Staples",
    "Energy", "Financials", "Health Care", "Industrials",
    "Information Technology", "Materials", "Real Estate", "Utilities",
]

# Regime derivation thresholds
RATE_TIGHTENING_THRESH  =  0.25   # 6M Fed Funds change > +0.25 -> tightening
RATE_EASING_THRESH      = -0.25   # 6M Fed Funds change < -0.25 -> easing
VIX_STRESSED_THRESH     = 25.0
VIX_CALM_THRESH         = 16.0
CREDIT_SPREAD_STRESSED_THRESH = 3.0   # calibrated for BAA10Y (Baa-10Y spread)
CREDIT_SPREAD_CALM_THRESH     = 1.8   # calibrated for BAA10Y
MFG_CONTRACTION_THRESH  = -1.0
MFG_EXPANSION_THRESH    =  2.0
CURVE_INVERSION_MONTHS  =  3      # consecutive months inverted -> recession signal

# Cycle thresholds on real GDP QoQ (annualized, %)
GDP_CONTRACTION_THRESH  =  0.0    # gdp_qoq < 0  -> outright output decline
GDP_EXPANSION_THRESH    =  2.5    # gdp_qoq > 2.5 -> solid growth

# GDP release timing. A191RL1Q225SBEA is stamped at the quarter START.
# The advance estimate releases ~1 month after the quarter ENDS:
#   Q1 (stamped Jan 1, ends Mar 31) -> advance ~end of April.
# quarter_start + 4 months, snapped to month-end, lands Q1 on May 31 -> safely
# after the advance release. Set to 3 for the more timely (Apr-30) variant.
GDP_RELEASE_LAG_MONTHS  =  4

# ── SYSTEM PROMPT ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a senior macro economist and portfolio strategist. \
You will be given a snapshot of macroeconomic conditions at a specific month-end date. \
Your task is to:
1. Confirm or refine the derived regime labels (rate_environment, risk_regime, cycle).
2. Write a macro_rationale of 4-5 sentences that synthesizes the macro picture. \
   You MUST cite at least 6 specific numeric values from the macro_snapshot. \
   Explain the mechanism, not just the state.
3. Assign a sector stance (overweight / neutral / underweight) for each of the 11 GICS sectors. \
   Every stance must follow logically from the macro_rationale.
4. State exactly 2 macro_risks that are specific and falsifiable, each traceable to \
   a specific field in the input.

Rules:
- Never use generic phrases like "geopolitical uncertainty is elevated" without tying \
  them to a specific causal chain from the input data.
- Regime labels must be internally consistent with the rationale. You cannot write \
  "no recession signal" and output cycle: contraction.
- Cite fed_statement_tone and geopolitical_risk explicitly in the rationale when they \
  are not neutral/normal.
- When a geopolitical_note references an oil or energy shock, anchor it to oil_wti / \
  oil_wti_yoy rather than asserting it qualitatively, and let it inform the Energy stance.
- Output valid JSON only. No preamble, no markdown, no explanation outside the JSON.

Output schema:
{
  "as_of_date": "YYYY-MM-DD",
  "regime": {
    "rate_environment": "tightening|stable|easing",
    "risk_regime": "stressed|normal|calm",
    "cycle": "contraction|mid_cycle|expansion"
  },
  "macro_rationale": "string",
  "sector_stances": {
    "Communication Services": "overweight|neutral|underweight",
    "Consumer Discretionary": "overweight|neutral|underweight",
    "Consumer Staples": "overweight|neutral|underweight",
    "Energy": "overweight|neutral|underweight",
    "Financials": "overweight|neutral|underweight",
    "Health Care": "overweight|neutral|underweight",
    "Industrials": "overweight|neutral|underweight",
    "Information Technology": "overweight|neutral|underweight",
    "Materials": "overweight|neutral|underweight",
    "Real Estate": "overweight|neutral|underweight",
    "Utilities": "overweight|neutral|underweight"
  },
  "macro_risks": ["string", "string"]
}"""

# ── MANUAL CONTEXT LABELS ─────────────────────────────────────────────────────
# Fed statement tone and geopolitical risk for each month-end in training window.
# Filled from historical knowledge - no data pull needed.
# Format: "YYYY-MM" -> {fed_tone, fed_note, geo_risk, geo_note}
#
# Entries marked  # FIXED  were corrected against Federal Reserve primary sources.

MANUAL_CONTEXT = {
    # 2013 - post-crisis recovery, QE3 ongoing, taper talk begins mid-year
    "2013-01": {"fed_tone": "dovish",   "fed_note": "QE3 ongoing at $85B/month, ZIRP maintained, forward guidance tied to 6.5% unemployment threshold",                         "geo_risk": "normal",   "geo_note": "No major supply disruption; fiscal cliff resolved, sequester risk limited"},
    "2013-02": {"fed_tone": "dovish",   "fed_note": "Fed reaffirming QE3, debate about pace beginning internally but not signaled publicly",                                      "geo_risk": "normal",   "geo_note": "No material geopolitical shock"},
    "2013-03": {"fed_tone": "dovish",   "fed_note": "QE3 continuing, Bernanke emphasizing data-dependence, no tapering signal yet",                                              "geo_risk": "normal",   "geo_note": "Cyprus banking crisis minor, contained to eurozone periphery"},
    "2013-04": {"fed_tone": "dovish",   "fed_note": "Fed maintaining accommodation, some internal debate on tapering pace emerging",                                              "geo_risk": "normal",   "geo_note": "North Korea tensions elevated briefly but no escalation"},
    "2013-05": {"fed_tone": "neutral",  "fed_note": "Bernanke May 22 testimony hints at tapering possibility in coming meetings if data improves, markets react sharply",         "geo_risk": "normal",   "geo_note": "No major commodity supply disruption"},
    "2013-06": {"fed_tone": "neutral",  "fed_note": "Fed signals tapering could begin later in 2013 if economy performs as expected, taper tantrum in bond markets",             "geo_risk": "normal",   "geo_note": "Syria chemical weapons concerns rising but no market impact yet"},
    "2013-07": {"fed_tone": "neutral",  "fed_note": "Bernanke emphasizes tapering not tightening, rate hikes still far away, markets stabilizing after taper tantrum",           "geo_risk": "normal",   "geo_note": "Egypt political crisis contained, no commodity shock"},
    "2013-08": {"fed_tone": "neutral",  "fed_note": "Fed preparing markets for tapering, debate on timing September vs December",                                                "geo_risk": "elevated", "geo_note": "Syria chemical weapons attack, US military strike threatened, oil prices spike briefly"},
    "2013-09": {"fed_tone": "neutral",  "fed_note": "Surprising no-taper decision in September, Fed cites fiscal uncertainty and weak data, markets rally",                      "geo_risk": "normal",   "geo_note": "Syria tensions de-escalated via diplomatic agreement"},
    "2013-10": {"fed_tone": "neutral",  "fed_note": "Government shutdown and debt ceiling fight delay tapering, Yellen nominated as next Fed chair",                             "geo_risk": "normal",   "geo_note": "US government shutdown creating domestic fiscal uncertainty"},
    "2013-11": {"fed_tone": "neutral",  "fed_note": "Tapering discussion resuming as shutdown resolved, December meeting in focus",                                              "geo_risk": "normal",   "geo_note": "No major geopolitical shock"},
    "2013-12": {"fed_tone": "neutral",  "fed_note": "Taper begins: QE reduced to $75B/month from $85B, Bernanke signals gradual pace, rates still near zero",                   "geo_risk": "normal",   "geo_note": "No major supply disruption"},

    # 2014 - gradual taper, geopolitical events mid-year
    "2014-01": {"fed_tone": "neutral",  "fed_note": "Taper steps to $65B/month at Bernanke's final FOMC (Jan 28-29); emerging-market currency stress from dollar strength but Fed stays the course",  "geo_risk": "normal",   "geo_note": "EM currency stress (Turkey, Argentina) but contained"},   # FIXED: was "$75B/month"
    "2014-02": {"fed_tone": "neutral",  "fed_note": "No FOMC meeting in February; taper pace holds at $65B/month, Yellen sworn in as chair Feb 3 with her first meeting due in March",                "geo_risk": "normal",   "geo_note": "Ukraine political crisis beginning, Yanukovych ousted"},   # FIXED: Yellen's first meeting was March, not Feb
    "2014-03": {"fed_tone": "neutral",  "fed_note": "Taper to $55B/month, Yellen removes 6.5% unemployment threshold guidance, forward guidance more qualitative",              "geo_risk": "elevated", "geo_note": "Russia annexes Crimea, Western sanctions begin, energy security concerns in Europe"},
    "2014-04": {"fed_tone": "neutral",  "fed_note": "Taper continues to $45B/month, labor market improving but participation rate declining",                                    "geo_risk": "elevated", "geo_note": "Ukraine conflict escalating in eastern regions, Russia-West tensions elevated"},
    "2014-05": {"fed_tone": "neutral",  "fed_note": "Taper to $35B/month, Yellen flags housing and labor market slack, rates to stay low well after taper ends",                "geo_risk": "elevated", "geo_note": "Ukraine civil conflict ongoing, European energy supply risk"},
    "2014-06": {"fed_tone": "neutral",  "fed_note": "Taper to $25B/month, economic outlook improving, first rate hike still distant",                                            "geo_risk": "elevated", "geo_note": "ISIS rapid advance in Iraq, oil supply disruption risk in Kurdistan region"},
    "2014-07": {"fed_tone": "neutral",  "fed_note": "Taper to $25B/month, Yellen flags potential earlier rate hike if labor market improves faster than expected",              "geo_risk": "elevated", "geo_note": "MH17 shot down over Ukraine, Gaza conflict, ISIS expanding, multiple simultaneous geopolitical risks"},
    "2014-08": {"fed_tone": "neutral",  "fed_note": "Taper continues, Jackson Hole speech Yellen discusses labor market slack in detail",                                        "geo_risk": "elevated", "geo_note": "Ukraine conflict, Iraq/ISIS, sanctions on Russia intensifying"},
    "2014-09": {"fed_tone": "neutral",  "fed_note": "Taper to $15B/month, Fed removes considerable time language on rates, hiking cycle getting closer",                         "geo_risk": "elevated", "geo_note": "Scotland independence referendum, Hong Kong protests, ISIS ongoing"},
    "2014-10": {"fed_tone": "neutral",  "fed_note": "QE ends, last taper, Fed signals patient approach to rate hikes, global growth concerns emerge",                            "geo_risk": "elevated", "geo_note": "Ebola outbreak causing market fear, global growth slowdown concerns from Europe and China"},
    "2014-11": {"fed_tone": "neutral",  "fed_note": "QE ended, Fed patient on rates, dollar strengthening weighing on inflation, ECB easing expectations",                       "geo_risk": "normal",   "geo_note": "Geopolitical risks easing slightly, oil price decline beginning"},
    "2014-12": {"fed_tone": "neutral",  "fed_note": "Fed signals patience on rates, removes considerable time language, oil price crash accelerating, Russia ruble crisis",       "geo_risk": "elevated", "geo_note": "Oil price crash (-50% from peak) stressing energy producers and Russia, ruble in freefall"},

    # 2015 - first rate hike cycle begins, China devaluation
    "2015-01": {"fed_tone": "neutral",  "fed_note": "Fed patient on rates, watching international developments, ECB launching QE, dollar surge",                                 "geo_risk": "normal",   "geo_note": "Charlie Hebdo attacks in France, oil price stabilizing at lower level"},
    "2015-02": {"fed_tone": "neutral",  "fed_note": "Fed patient, labor market improving but inflation below target, watching Greece negotiations",                               "geo_risk": "normal",   "geo_note": "Greece debt negotiations creating eurozone uncertainty"},
    "2015-03": {"fed_tone": "neutral",  "fed_note": "Fed drops patient language, signaling rate hike possible at any meeting from June, dollar surging",                         "geo_risk": "normal",   "geo_note": "Greece debt crisis ongoing, Ukraine ceasefire fragile"},
    "2015-04": {"fed_tone": "neutral",  "fed_note": "Fed in wait-and-see mode, weak Q1 GDP blamed on weather and port strike, rate hike timeline pushed back",                   "geo_risk": "normal",   "geo_note": "Yemen conflict beginning, Saudi Arabia-Iran proxy tensions"},
    "2015-05": {"fed_tone": "neutral",  "fed_note": "Fed cautious, watching Greece and China, global growth concerns, dollar strength weighing on US exports",                    "geo_risk": "normal",   "geo_note": "No major new geopolitical shock, Greece negotiations continuing"},
    "2015-06": {"fed_tone": "neutral",  "fed_note": "Fed delays hike amid Greece crisis and China equity volatility, signals September as likely first hike",                     "geo_risk": "elevated", "geo_note": "Greece capital controls imposed, Grexit risk peaks, Chinese equity market crash beginning"},
    "2015-07": {"fed_tone": "neutral",  "fed_note": "Fed monitoring China and Greece, September hike still on table if data holds",                                              "geo_risk": "elevated", "geo_note": "China equity circuit breaker interventions, yuan devaluation fears, Greece deal reached"},
    "2015-08": {"fed_tone": "neutral",  "fed_note": "China yuan devaluation and equity crash forces Fed to delay September hike, global volatility spike",                       "geo_risk": "elevated", "geo_note": "China yuan devaluation shock, global EM currency stress, commodity price collapse accelerating"},
    "2015-09": {"fed_tone": "neutral",  "fed_note": "Fed holds rates, citing international developments and low inflation, December hike now expected",                           "geo_risk": "elevated", "geo_note": "China slowdown concerns, VW emissions scandal, European refugee crisis escalating"},
    "2015-10": {"fed_tone": "neutral",  "fed_note": "Fed signals December hike likely if labor market continues improving, economy on solid footing domestically",                "geo_risk": "elevated", "geo_note": "Russia intervention in Syria, China/EM stress, and commodity weakness keep geopolitical risk elevated"},
    "2015-11": {"fed_tone": "hawkish",  "fed_note": "Fed clearly signaling December rate hike, labor market meeting criteria, liftoff imminent after 7 years at zero",           "geo_risk": "elevated", "geo_note": "Paris attacks, Russia-Turkey jet incident, ISIS territorial expansion"},
    "2015-12": {"fed_tone": "hawkish",  "fed_note": "First rate hike: fed funds to 0.25-0.50%, liftoff after 7 years, gradual path signaled, one hike per quarter expected",    "geo_risk": "normal",   "geo_note": "Geopolitical tensions elevated but priced, oil price decline continuing"},

    # 2016 - slow hiking cycle, Brexit, election
    "2016-01": {"fed_tone": "neutral",  "fed_note": "Fed pausing after December hike, China turmoil and oil crash forcing reconsideration of pace",                              "geo_risk": "elevated", "geo_note": "China circuit breakers triggered, oil below $30, global recession fears spiking"},
    "2016-02": {"fed_tone": "neutral",  "fed_note": "Fed on hold, financial market stress from China and oil dampening rate hike expectations significantly",                     "geo_risk": "elevated", "geo_note": "Oil near cycle lows, bank stress in Europe, China capital outflows accelerating"},
    "2016-03": {"fed_tone": "neutral",  "fed_note": "Fed holds, only 2 hikes now expected in 2016 down from 4, Yellen citing global risks",                                     "geo_risk": "normal",   "geo_note": "Brussels terrorist attacks, oil stabilizing, China fears moderating"},
    "2016-04": {"fed_tone": "neutral",  "fed_note": "Fed patient, economy improving but Brexit uncertainty and election year caution",                                            "geo_risk": "normal",   "geo_note": "Brexit referendum approaching, Panama Papers, oil recovering toward $45"},
    "2016-05": {"fed_tone": "neutral",  "fed_note": "Fed signals June hike possible, minutes more hawkish than expected, markets reprice",                                       "geo_risk": "normal",   "geo_note": "Brexit polls tightening, creating uncertainty for June Fed decision"},
    "2016-06": {"fed_tone": "neutral",  "fed_note": "Fed holds ahead of Brexit referendum, Brexit shock causes immediate hold, all hike plans suspended",                         "geo_risk": "elevated", "geo_note": "Brexit vote shocks markets, pound crashes, global risk-off, EU political uncertainty surges"},
    "2016-07": {"fed_tone": "neutral",  "fed_note": "Fed on hold post-Brexit, assessing economic fallout, strong July jobs report restores some confidence",                      "geo_risk": "elevated", "geo_note": "Brexit aftermath, Turkey coup attempt, Nice truck attack, political uncertainty elevated"},
    "2016-08": {"fed_tone": "neutral",  "fed_note": "Jackson Hole Yellen signals rate hike case strengthening but no specific timing, data-dependent",                           "geo_risk": "normal",   "geo_note": "Geopolitical risks moderating, Brexit uncertainty persisting but acute phase passed"},
    "2016-09": {"fed_tone": "neutral",  "fed_note": "Fed holds with 3 dissents, signals December hike likely, waiting for more progress on dual mandate",                        "geo_risk": "normal",   "geo_note": "Syria conflict, US election uncertainty, OPEC production cut discussions beginning"},
    "2016-10": {"fed_tone": "neutral",  "fed_note": "Fed on hold pre-election, December hike broadly expected, election outcome key risk",                                       "geo_risk": "normal",   "geo_note": "US election uncertainty, OPEC cut agreement being negotiated"},
    "2016-11": {"fed_tone": "neutral",  "fed_note": "Trump election victory surprises markets, reflation trade begins, rate hike expectations accelerate",                        "geo_risk": "normal",   "geo_note": "Trump election triggers reflation trade, dollar surges, EM stress, policy uncertainty high"},
    "2016-12": {"fed_tone": "hawkish",  "fed_note": "Fed hikes to 0.50-0.75%, signals 3 hikes in 2017 vs 2 previously, dot plot shift hawkish on fiscal stimulus expectations", "geo_risk": "normal",   "geo_note": "Berlin Christmas market attack, OPEC cut finalizing, dollar at multi-year highs"},

    # 2017 - steady hikes, tax reform, low volatility
    "2017-01": {"fed_tone": "neutral",  "fed_note": "Fed on hold, assessing Trump fiscal stimulus impact, uncertainty about policy timing",                                       "geo_risk": "normal",   "geo_note": "Trump inauguration, executive orders on immigration creating uncertainty, no commodity shock"},
    "2017-02": {"fed_tone": "neutral",  "fed_note": "Fed signaling March hike possible, economy on solid footing, inflation approaching target",                                 "geo_risk": "normal",   "geo_note": "French election uncertainty, North Korea missile tests, no major commodity disruption"},
    "2017-03": {"fed_tone": "hawkish",  "fed_note": "Fed hikes to 0.75-1.0%, signals two more hikes in 2017, confident in economic outlook",                                    "geo_risk": "normal",   "geo_note": "US strikes Syria, North Korea tensions rising, French election risk"},
    "2017-04": {"fed_tone": "neutral",  "fed_note": "Fed on hold, balance sheet normalization discussion beginning, economy solid",                                               "geo_risk": "normal",   "geo_note": "North Korea tensions peak briefly, French election first round approaching"},
    "2017-05": {"fed_tone": "neutral",  "fed_note": "Fed signals June hike and balance sheet runoff plan, Macron wins French election relieving European risk",                   "geo_risk": "normal",   "geo_note": "French election resolved favorably, North Korea tensions moderating, political risk premium declining"},
    "2017-06": {"fed_tone": "hawkish",  "fed_note": "Fed hikes to 1.0-1.25%, announces balance sheet normalization plan to begin later in 2017",                                "geo_risk": "normal",   "geo_note": "Qatar diplomatic crisis, no major market impact, global growth synchronized upswing"},
    "2017-07": {"fed_tone": "neutral",  "fed_note": "Fed on hold, balance sheet plan confirmed for September, economy performing as expected",                                    "geo_risk": "normal",   "geo_note": "North Korea ICBM test, Venezuela crisis, no commodity shock"},
    "2017-08": {"fed_tone": "neutral",  "fed_note": "Jackson Hole Yellen and Draghi both signal gradual normalization, no specific timing",                                      "geo_risk": "normal",   "geo_note": "North Korea nuclear test, Charlottesville, Harvey hurricane, no supply shock"},
    "2017-09": {"fed_tone": "hawkish",  "fed_note": "Fed begins balance sheet runoff, signals December hike, inflation below target but transitory",                             "geo_risk": "normal",   "geo_note": "North Korea H-bomb test, Irma and Maria hurricanes causing domestic disruption"},
    "2017-10": {"fed_tone": "neutral",  "fed_note": "Powell nominated as next Fed chair, December hike expected, tax reform gaining momentum",                                   "geo_risk": "normal",   "geo_note": "Las Vegas shooting, Catalonia independence referendum, no commodity shock"},
    "2017-11": {"fed_tone": "neutral",  "fed_note": "Tax reform advancing in Congress, Fed on track for December hike, Powell confirmed as chair",                               "geo_risk": "normal",   "geo_note": "Saudi Arabia corruption purge, Lebanon crisis briefly, oil rising on geopolitical premium"},
    "2017-12": {"fed_tone": "hawkish",  "fed_note": "Fed hikes to 1.25-1.50%, tax cuts signed into law, 3 hikes expected in 2018, balance sheet runoff continuing",            "geo_risk": "normal",   "geo_note": "Tax reform signed, Bitcoin mania, no geopolitical supply shock, global synchronized growth"},

    # 2018 - tightening cycle, trade war, Q4 crash
    "2018-01": {"fed_tone": "hawkish",  "fed_note": "Tax stimulus boosting growth expectations, wage inflation data in January triggers equity selloff, 4 hikes now priced",     "geo_risk": "normal",   "geo_note": "US government shutdown briefly, no commodity shock, synchronized global growth continuing"},
    "2018-02": {"fed_tone": "hawkish",  "fed_note": "Powell first meeting as chair, wage data sparks VIX spike and equity correction, but Powell signals steady path",           "geo_risk": "normal",   "geo_note": "VIX products implosion, correction orderly, no geopolitical trigger"},
    "2018-03": {"fed_tone": "hawkish",  "fed_note": "Fed hikes to 1.5-1.75%, Powell signals gradual pace, tariff announcements on steel and aluminum create trade war fears",   "geo_risk": "elevated", "geo_note": "Trump steel and aluminum tariffs announced, trade war with China escalating, European tariff threat"},
    "2018-04": {"fed_tone": "hawkish",  "fed_note": "Fed on hold, economy strong, trade war concerns rising, Syria strikes, dollar strengthening",                               "geo_risk": "elevated", "geo_note": "US strikes Syria, China-US trade war escalating, EM currencies under pressure from dollar surge"},
    "2018-05": {"fed_tone": "hawkish",  "fed_note": "Fed on hold, Italian political crisis and EM stress, but domestic economy strong, June hike on track",                      "geo_risk": "elevated", "geo_note": "Italian political crisis, Turkey lira stress, Argentina IMF bailout, EM contagion fears"},
    "2018-06": {"fed_tone": "hawkish",  "fed_note": "Fed hikes to 1.75-2.0%, raises 2018 forecast to 4 hikes, trade war with China formally launched with tariffs",             "geo_risk": "elevated", "geo_note": "US-China trade war formally begins with $34B tariffs, retaliation announced, escalation risk high"},
    "2018-07": {"fed_tone": "hawkish",  "fed_note": "Fed on hold, economy very strong, tariff effects being monitored, September hike expected",                                 "geo_risk": "elevated", "geo_note": "Trade war tariffs expanding, China yuan weakening, US-EU trade tensions"},
    "2018-08": {"fed_tone": "hawkish",  "fed_note": "Jackson Hole Powell signals continued gradual rate hikes, strong economy, trade war risk acknowledged",                      "geo_risk": "elevated", "geo_note": "Turkey lira crisis, EM contagion spreading, US-China trade war continuing, Argentina peso collapse"},
    "2018-09": {"fed_tone": "hawkish",  "fed_note": "Fed hikes to 2.0-2.25%, removes accommodative language, signals above-neutral rates, another hike in December",             "geo_risk": "elevated", "geo_note": "US-China trade war escalating to $200B tariffs, EM stress continuing, Italy budget crisis"},
    "2018-10": {"fed_tone": "hawkish",  "fed_note": "Powell neutral rate comments spook markets, equity selloff begins, Fed signals December hike still on track",               "geo_risk": "elevated", "geo_note": "Khashoggi murder causing Saudi Arabia risk premium, equity market correction accelerating"},
    "2018-11": {"fed_tone": "hawkish",  "fed_note": "Fed on hold, equity selloff continuing, oil crash (-30%) raising recession fears, December hike still expected",             "geo_risk": "normal",   "geo_note": "Oil price crash from $85 to $55, recession fears rising, US-China trade war truce hopes"},
    "2018-12": {"fed_tone": "hawkish",  "fed_note": "Fed hikes despite market selloff, dot plot shifts to 2 hikes in 2019, Powell later walks back hawkishness in January",     "geo_risk": "elevated", "geo_note": "US government shutdown, equity bear market, trade war, global growth slowdown fears"},

    # 2019 - pivot, three cuts
    "2019-01": {"fed_tone": "dovish",   "fed_note": "Powell dramatic pivot: patient and data-dependent, rate hikes on pause, balance sheet runoff being reconsidered",            "geo_risk": "normal",   "geo_note": "US-China trade war truce, government shutdown, global growth concerns but no acute shock"},
    "2019-02": {"fed_tone": "dovish",   "fed_note": "Fed patient, watching global growth slowdown, China trade deal progress, balance sheet runoff to end sooner",               "geo_risk": "normal",   "geo_note": "US-China trade talks progressing, India-Pakistan tensions briefly, Venezuela crisis"},
    "2019-03": {"fed_tone": "dovish",   "fed_note": "Fed signals no hikes in 2019, dot plot collapses to zero hikes, balance sheet runoff to end September",                    "geo_risk": "normal",   "geo_note": "Brexit extension, US-China trade talks ongoing, no major commodity shock"},
    "2019-04": {"fed_tone": "dovish",   "fed_note": "Fed patient, economy solid but global risks elevated, rate hikes off table for 2019",                                       "geo_risk": "normal",   "geo_note": "Sri Lanka attacks, Libya conflict, oil rising on OPEC cuts and Iran sanctions"},
    "2019-05": {"fed_tone": "dovish",   "fed_note": "Fed patient, but trade war escalation increases cut expectations, markets pricing insurance cut",                             "geo_risk": "elevated", "geo_note": "US-China trade war re-escalates sharply, Huawei sanctions, tariffs to 25%, markets reprice"},
    "2019-06": {"fed_tone": "dovish",   "fed_note": "Fed signals rate cuts coming, drops patient language, G20 trade truce temporarily relieves pressure",                        "geo_risk": "elevated", "geo_note": "Iran sanctions, oil tanker attacks in Gulf, US-China G20 meeting providing temporary relief"},
    "2019-07": {"fed_tone": "dovish",   "fed_note": "Fed cuts 25bps to 2.0-2.25%, insurance cut against trade war and global slowdown, but Powell says not a cycle",            "geo_risk": "elevated", "geo_note": "Trade war ongoing, Hong Kong protests beginning, Iran tensions, global PMI weakness"},
    "2019-08": {"fed_tone": "dovish",   "fed_note": "No FOMC meeting in August; markets price further cuts as the China trade war re-escalates (Aug 1 tariff move) and the 10Y-2Y curve briefly inverts Aug 14, stoking recession fears", "geo_risk": "elevated", "geo_note": "US-China trade war escalates with new tariffs, yuan breaks 7.0, curve inverts, EM stress"},   # FIXED: no Aug meeting; old note invented a cut
    "2019-09": {"fed_tone": "dovish",   "fed_note": "Fed cuts 25bps to 1.75-2.00% (second cut of the cycle); mid-September repo-market stress forces emergency overnight repo operations from Sept 17",  "geo_risk": "elevated", "geo_note": "Saudi Aramco drone attack disrupts 5% of global oil supply briefly, repo market stress"},   # FIXED: was "1.5-1.75% / 3 cuts done"
    "2019-10": {"fed_tone": "dovish",   "fed_note": "Fed cuts 25bps to 1.50-1.75% (third cut); signals a pause, frames the three cuts as a mid-cycle adjustment, and begins T-bill purchases to rebuild reserves", "geo_risk": "elevated", "geo_note": "US-China Phase 1 deal progressing, Turkey Syria invasion, Brexit extension again"},   # FIXED: was "1.25-1.5%"
    "2019-11": {"fed_tone": "neutral",  "fed_note": "Fed on hold, signals pause after 3 cuts, economy resilient, watching trade negotiations",                                   "geo_risk": "normal",   "geo_note": "US-China Phase 1 trade deal close to signing, Hong Kong protests, Chile social unrest"},
    "2019-12": {"fed_tone": "neutral",  "fed_note": "Fed on hold, US-China Phase 1 agreed but not yet signed, economy solid, rates on hold for foreseeable future, balance sheet growing via T-bills", "geo_risk": "normal",   "geo_note": "US-China Phase 1 deal agreed but not yet signed; Brexit uncertainty eased after UK election"},

    # 2020 - COVID
    "2020-01": {"fed_tone": "neutral",  "fed_note": "Fed on hold, economy solid, Phase 1 signed, watching COVID reports from China with concern",                                "geo_risk": "elevated", "geo_note": "US-Iran tensions after Soleimani killing, COVID-19 emerging in Wuhan, early containment uncertainty"},
    "2020-02": {"fed_tone": "neutral",  "fed_note": "Fed monitoring COVID closely, markets begin pricing pandemic risk, emergency cut being discussed",                            "geo_risk": "elevated", "geo_note": "COVID-19 spreading globally, Italy outbreak declared, market selloff accelerating"},
    "2020-03": {"fed_tone": "dovish",   "fed_note": "Emergency cuts: Fed cuts 150bps to zero in two unscheduled meetings, launches unlimited QE, full crisis response",           "geo_risk": "elevated", "geo_note": "COVID-19 global pandemic declared, lockdowns worldwide, oil price war Russia-Saudi, deepest shock since 2008"},
    "2020-04": {"fed_tone": "dovish",   "fed_note": "Fed deploying all tools: Main Street lending, municipal liquidity, corporate credit, unlimited QE",                          "geo_risk": "elevated", "geo_note": "Global lockdowns, oil futures go negative, 30M US unemployment claims, pandemic peak uncertainty"},
    "2020-05": {"fed_tone": "dovish",   "fed_note": "Fed maintaining all facilities, Powell warns of long-term economic scarring, Congress passing stimulus",                      "geo_risk": "elevated", "geo_note": "Reopening beginning but second wave fears, George Floyd protests, US-China tensions rising"},
    "2020-06": {"fed_tone": "dovish",   "fed_note": "Fed dots show no hikes through 2022, yield curve control being considered, economy recovering faster than expected",         "geo_risk": "elevated", "geo_note": "Second wave fears in US sunbelt, US-China tensions, Hong Kong security law, social unrest"},
    "2020-07": {"fed_tone": "dovish",   "fed_note": "Fed maintaining accommodation, recovery uneven, watching fiscal stimulus expiration, extending emergency facilities",          "geo_risk": "elevated", "geo_note": "US COVID surge in sunbelt states, US-China consulate closures, election uncertainty building"},
    "2020-08": {"fed_tone": "dovish",   "fed_note": "Jackson Hole: Powell introduces Average Inflation Targeting, will overshoot 2% before hiking, major regime shift",           "geo_risk": "normal",   "geo_note": "Vaccine trial progress raising hopes, US-China tension stabilizing, Belarus protests"},
    "2020-09": {"fed_tone": "dovish",   "fed_note": "Fed formalizes AIT framework, signals no hikes until inflation at 2% for some time and labor fully recovered",               "geo_risk": "normal",   "geo_note": "Election uncertainty rising, vaccine Phase 3 trials ongoing, stimulus negotiations stalled"},
    "2020-10": {"fed_tone": "dovish",   "fed_note": "Fed on hold, urging more fiscal stimulus, election uncertainty, second wave in Europe",                                      "geo_risk": "normal",   "geo_note": "US election approaching, European second wave lockdowns, no vaccine yet, stimulus stalled"},
    "2020-11": {"fed_tone": "dovish",   "fed_note": "Pfizer vaccine 90% efficacy announced, risk-on surge, Fed maintaining full accommodation as economy still weak",             "geo_risk": "normal",   "geo_note": "Vaccine breakthrough changes macro outlook, Biden election victory, transition uncertainty fading"},
    "2020-12": {"fed_tone": "dovish",   "fed_note": "Fed extends QE guidance until substantial progress, vaccine rollout beginning, $900B stimulus passed",                       "geo_risk": "normal",   "geo_note": "COVID vaccines rolling out, UK-EU Brexit deal finalized, stimulus signed, political risk declining"},

    # 2021 - recovery, inflation surprise
    "2021-01": {"fed_tone": "dovish",   "fed_note": "Fed fully accommodative, vaccine rollout slow, economy still well below potential, fiscal stimulus expected",                 "geo_risk": "normal",   "geo_note": "Capitol riot January 6, Biden inauguration, vaccine rollout challenges, no commodity shock"},
    "2021-02": {"fed_tone": "dovish",   "fed_note": "Fed maintaining accommodation, long rates rising on reflation trade, Powell pushes back on taper talk",                     "geo_risk": "normal",   "geo_note": "Texas freeze disrupts energy supply briefly, bond market selloff on reflation trade"},
    "2021-03": {"fed_tone": "dovish",   "fed_note": "$1.9T American Rescue Plan signed, Fed maintains stance despite rising rates and inflation concerns, transitory framing",   "geo_risk": "normal",   "geo_note": "Suez Canal blocked briefly by Ever Given, vaccine rollout accelerating globally"},
    "2021-04": {"fed_tone": "dovish",   "fed_note": "Fed fully accommodative, economy recovering strongly, inflation rising but deemed transitory, no taper discussion yet",      "geo_risk": "normal",   "geo_note": "India COVID second wave severe, supply chain stress building globally, no major commodity shock"},
    "2021-05": {"fed_tone": "dovish",   "fed_note": "Fed on hold, CPI 5% surprises to upside, Fed sticks to transitory narrative, some dissent growing",                         "geo_risk": "normal",   "geo_note": "Colonial Pipeline ransomware attack, Israel-Gaza conflict, supply chain stress accelerating"},
    "2021-06": {"fed_tone": "neutral",  "fed_note": "Dot plot shifts: 2 hikes now expected in 2023, tapering discussion beginning, transitory vs persistent inflation debate",   "geo_risk": "normal",   "geo_note": "G7 summit, Delta variant emerging, supply chain disruptions worsening, lumber prices spiking"},
    "2021-07": {"fed_tone": "neutral",  "fed_note": "Fed tapering discussion more concrete, Delta variant causing uncertainty, Powell at Jackson Hole will be key",               "geo_risk": "normal",   "geo_note": "Delta variant spreading globally, Afghanistan Taliban takeover, supply chain stress"},
    "2021-08": {"fed_tone": "neutral",  "fed_note": "Jackson Hole Powell signals tapering appropriate soon but no specific timing, separates taper from rate hikes clearly",      "geo_risk": "normal",   "geo_note": "Afghanistan collapse, Hurricane Ida, Delta wave peaking, supply chain stress peak"},
    "2021-09": {"fed_tone": "neutral",  "fed_note": "Fed signals November taper announcement, Evergrande collapse spooks markets briefly but Fed stays course",                   "geo_risk": "normal",   "geo_note": "Evergrande debt crisis in China causes brief contagion fear, energy price spike in Europe"},
    "2021-10": {"fed_tone": "neutral",  "fed_note": "November taper announcement confirmed, $15B/month reduction, inflation 6.2% forcing faster pace consideration",              "geo_risk": "normal",   "geo_note": "Natural gas crisis in Europe, supply chain stress peak, US debt ceiling drama resolved"},
    "2021-11": {"fed_tone": "neutral",  "fed_note": "Taper begins, Omicron variant discovered and spooks markets, Powell drops transitory language completely",                   "geo_risk": "elevated", "geo_note": "Omicron variant discovered in South Africa, immediate global risk-off, travel bans reimposed"},
    "2021-12": {"fed_tone": "hawkish",  "fed_note": "Taper accelerated to $30B/month, dot plot shows 3 hikes in 2022, CPI at 7% highest since 1982, pivot to fighting inflation", "geo_risk": "elevated", "geo_note": "Omicron wave severe but milder outcomes, Russia massing troops on Ukraine border, energy crisis Europe"},

    # 2022 - aggressive tightening, Ukraine
    "2022-01": {"fed_tone": "hawkish",  "fed_note": "Fed signals March hike, taper ending, balance sheet runoff to follow quickly, CPI 7.5%, most hawkish pivot in decades",    "geo_risk": "elevated", "geo_note": "Russia-Ukraine invasion risk rising, troops massed at border, energy supply security concerns in Europe"},
    "2022-02": {"fed_tone": "hawkish",  "fed_note": "February 24 Russia invades Ukraine, emergency conditions, oil spikes, Fed must balance inflation fighting with geopolitical shock", "geo_risk": "elevated", "geo_note": "Russia invades Ukraine, sanctions imposed, oil/gas/wheat prices surge, European energy crisis begins"},
    "2022-03": {"fed_tone": "hawkish",  "fed_note": "Fed hikes 25bps to 0.25-0.50%, first hike since 2018, signals aggressive path, 7 hikes in 2022 now expected",             "geo_risk": "elevated", "geo_note": "Ukraine war ongoing, commodity price shock persisting, supply chain stress, China lockdowns"},
    "2022-04": {"fed_tone": "hawkish",  "fed_note": "Fed signals 50bps hikes coming, balance sheet runoff to begin May, most aggressive tightening path since 1980s",            "geo_risk": "elevated", "geo_note": "Ukraine war escalating in Donbas, China Shanghai lockdown disrupting global supply chains"},
    "2022-05": {"fed_tone": "hawkish",  "fed_note": "Fed hikes 50bps to 0.75-1.0%, balance sheet runoff begins, signals more 50bps hikes, recession risk rising",               "geo_risk": "elevated", "geo_note": "Ukraine war, China lockdowns, food security crisis in developing world, Sri Lanka default"},
    "2022-06": {"fed_tone": "hawkish",  "fed_note": "Fed hikes 75bps largest since 1994, CPI 9.1%, signals more large hikes, recession risk increasing significantly",           "geo_risk": "elevated", "geo_note": "Ukraine war, energy crisis Europe, food price shock globally, EM debt stress accelerating"},
    "2022-07": {"fed_tone": "hawkish",  "fed_note": "Fed hikes 75bps to 2.25-2.50%, signals data-dependent pace, CPI peak hopes, technical recession Q1+Q2",                   "geo_risk": "elevated", "geo_note": "Ukraine war, Russia Nordstream gas cuts to Europe, energy rationing risk, Italy political crisis"},
    "2022-08": {"fed_tone": "hawkish",  "fed_note": "Jackson Hole Powell very hawkish: will keep hiking until job done, historical lesson on stopping too early, markets sell off", "geo_risk": "elevated", "geo_note": "Ukraine war, Nordstream leaks, Europe energy crisis deepening, Taiwan Pelosi visit tensions"},
    "2022-09": {"fed_tone": "hawkish",  "fed_note": "Fed hikes 75bps to 3.0-3.25%, dot plot shows terminal rate 4.6%, recession accepted as necessary cost to fight inflation",  "geo_risk": "elevated", "geo_note": "UK mini-budget crisis LDI pension fund stress, Ukraine mobilization, Nordstream sabotage"},
    "2022-10": {"fed_tone": "hawkish",  "fed_note": "Fed signals 75bps in November, terminal rate expectations rising to 5%+, global dollar strength stressing EM",             "geo_risk": "elevated", "geo_note": "Ukraine war, global dollar stress, EM debt crisis fears, China zero-COVID, UK financial instability"},
    "2022-11": {"fed_tone": "hawkish",  "fed_note": "Fed hikes 75bps to 3.75-4.0%, signals smaller hikes coming but higher terminal rate, CPI softening slightly",              "geo_risk": "elevated", "geo_note": "FTX crypto collapse, Ukraine war, China COVID protests beginning, midterm elections"},
    "2022-12": {"fed_tone": "hawkish",  "fed_note": "Fed hikes 50bps to 4.25-4.50%, dot plot shows 5.1% terminal rate, signals higher for longer despite slowing hike pace",   "geo_risk": "elevated", "geo_note": "Ukraine war ongoing, China zero-COVID abandoned suddenly, reopening uncertainty, energy prices moderating"},

    # 2023 - peak rates, banking stress, soft landing
    "2023-01": {"fed_tone": "hawkish",  "fed_note": "Fed signals more hikes needed, 25bps pace now, terminal rate above 5%, inflation still too high despite declining",         "geo_risk": "normal",   "geo_note": "China reopening begins, Ukraine war ongoing, US debt ceiling approaching, balloon incident"},
    "2023-02": {"fed_tone": "hawkish",  "fed_note": "Fed hikes 25bps to 4.5-4.75%, signals ongoing hikes, strong jobs report revives higher-for-longer narrative",               "geo_risk": "normal",   "geo_note": "Turkey-Syria earthquake, Ukraine war, no new major supply disruption"},
    "2023-03": {"fed_tone": "hawkish",  "fed_note": "SVB and Signature Bank collapse, Fed hikes 25bps despite banking stress, signals near terminal rate",                       "geo_risk": "elevated", "geo_note": "SVB Silicon Valley Bank collapse triggers banking stress, Credit Suisse rescued, FDIC emergency actions"},
    "2023-04": {"fed_tone": "neutral",  "fed_note": "Fed signals possible pause after next hike, banking stress monitoring, CPI decelerating, terminal rate near",               "geo_risk": "normal",   "geo_note": "Banking stress fading, Sudan conflict, Ukraine war ongoing, debt ceiling negotiation"},
    "2023-05": {"fed_tone": "neutral",  "fed_note": "Fed hikes 25bps to 5.0-5.25%, signals conditional pause, most aggressive cycle in 40 years possibly complete",             "geo_risk": "normal",   "geo_note": "US debt ceiling crisis approaching, first regional bank failures post-SVB (First Republic), Ukraine war"},
    "2023-06": {"fed_tone": "neutral",  "fed_note": "Fed pauses after 10 hikes, dot plot still shows 2 more hikes in 2023, data-dependent, soft landing hopes rising",          "geo_risk": "normal",   "geo_note": "Debt ceiling resolved, Ukraine counteroffensive, Wagner mutiny in Russia briefly"},
    "2023-07": {"fed_tone": "neutral",  "fed_note": "Fed hikes 25bps to 5.25-5.50%, signals data-dependent, could be final hike, soft landing increasingly plausible",          "geo_risk": "normal",   "geo_note": "Ukraine war ongoing, Niger coup, no major commodity shock, China slowdown concerns"},
    "2023-08": {"fed_tone": "neutral",  "fed_note": "Jackson Hole Powell signals higher for longer, not confident inflation beaten, keeps option of more hikes open",             "geo_risk": "normal",   "geo_note": "China property crisis (Country Garden), Maui wildfires, no energy supply shock"},
    "2023-09": {"fed_tone": "neutral",  "fed_note": "Fed holds, higher-for-longer message, long rates surging (10Y hits 4.8%), tightening happening through market rates",       "geo_risk": "elevated", "geo_note": "Oil rising on Saudi/Russia supply cuts and long-rate volatility; Middle East risk not yet acute"},
    "2023-10": {"fed_tone": "neutral",  "fed_note": "Fed holds, Waller and others signal market doing tightening for them via long rates, pivot discussion beginning",            "geo_risk": "elevated", "geo_note": "Hamas attack on Israel October 7, Israel-Gaza war begins, Middle East escalation risk, oil spike"},
    "2023-11": {"fed_tone": "neutral",  "fed_note": "Fed holds, pivot expectations building rapidly, inflation data softening, Powell less hawkish than expected",                "geo_risk": "elevated", "geo_note": "Israel-Gaza war ongoing, Houthi attacks on Red Sea shipping beginning, no broad supply shock yet"},
    "2023-12": {"fed_tone": "dovish",   "fed_note": "Fed holds, dot plot shifts to 3 cuts in 2024, Powell acknowledges inflation progress, pivot confirmed, markets rally",       "geo_risk": "elevated", "geo_note": "Israel-Gaza war, Red Sea shipping attacks by Houthis, Argentina Milei elected, China deflation"},

    # 2024 - cuts begin, higher for longer then pivot
    "2024-01": {"fed_tone": "neutral",  "fed_note": "Fed pushes back on March cut expectations, strong economic data delaying pivot, higher for longer narrative reasserts",      "geo_risk": "elevated", "geo_note": "Red Sea shipping crisis from Houthi attacks, Israel-Gaza continuing, Taiwan elections"},
    "2024-02": {"fed_tone": "neutral",  "fed_note": "Fed on hold, inflation re-acceleration in January forces delay, March cut off table, June now base case",                   "geo_risk": "elevated", "geo_note": "Red Sea crisis persisting, Ukraine Russia gaining territory, US aid package delayed"},
    "2024-03": {"fed_tone": "neutral",  "fed_note": "Fed holds, dot plot still shows 3 cuts but inflation stickier, soft landing confidence building, June cut expected",        "geo_risk": "elevated", "geo_note": "Baltimore bridge collapse disrupts logistics; Red Sea attacks continue; Iran-Israel risk rising but no direct exchange yet"},
    "2024-04": {"fed_tone": "neutral",  "fed_note": "Fed delays cuts further, CPI surprise forces higher for longer return, June cut now in doubt, September more likely",        "geo_risk": "elevated", "geo_note": "Iran direct attack on Israel with drones and missiles, retaliation cycle fear, oil spike"},
    "2024-05": {"fed_tone": "neutral",  "fed_note": "Fed holds rates and pushes back on near-term cuts; inflation confidence not sufficient",                                     "geo_risk": "normal",   "geo_note": "Iran-Israel tensions de-escalating, Red Sea crisis continuing, Rafah offensive"},
    "2024-06": {"fed_tone": "neutral",  "fed_note": "Fed holds, dot plot reduces to one cut in 2024, but June CPI surprise to downside opens door to September cut",             "geo_risk": "normal",   "geo_note": "French snap elections causing EU uncertainty, Israel-Gaza ceasefire talks, G7 summit"},
    "2024-07": {"fed_tone": "neutral",  "fed_note": "Fed on hold, Powell signals September cut likely if data continues improving, soft landing increasingly consensus",          "geo_risk": "normal",   "geo_note": "Trump assassination attempt, Biden drops out of election, carry trade unwind triggered by BOJ"},
    "2024-08": {"fed_tone": "dovish",   "fed_note": "Jackson Hole Powell says time has come to cut, September cut confirmed, labor market data weakening, pivot imminent",        "geo_risk": "normal",   "geo_note": "BOJ rate hike triggers yen carry trade unwind, brief global selloff, Harris-Trump election race"},
    "2024-09": {"fed_tone": "dovish",   "fed_note": "Fed cuts 50bps to 4.75-5.0%, larger than expected cut, signals gradual further cuts, labor market priority now",            "geo_risk": "normal",   "geo_note": "Israel-Lebanon escalation, Hezbollah pager attack, Middle East risk elevated but oil contained"},
    "2024-10": {"fed_tone": "dovish",   "fed_note": "Fed on track for more cuts, economy resilient, inflation near target, soft landing narrative dominant",                       "geo_risk": "normal",   "geo_note": "Israel-Iran strikes exchange, election uncertainty final stretch, no commodity shock"},
    "2024-11": {"fed_tone": "dovish",   "fed_note": "Fed cuts 25bps to 4.5-4.75%, Trump election victory signals tariffs and fiscal expansion, rate cut path becoming uncertain", "geo_risk": "normal",   "geo_note": "Trump wins election, tariff risk returns, dollar surges, EM stress, deregulation expectations"},
    "2024-12": {"fed_tone": "neutral",  "fed_note": "Fed cuts 25bps but signals pause, dot plot reduces to 2 cuts in 2025, inflation stickier than expected, tariff uncertainty", "geo_risk": "normal",   "geo_note": "Syria government falls, Trump tariff threats escalating, Fed more cautious on further cuts"},
}


# ── FRED DATA PULL ────────────────────────────────────────────────────────────

def pull_fred_data(api_key: str, start: str, end: str) -> pd.DataFrame:
    """Pull all macro series from FRED and align to month-end dates."""
    fred = Fred(api_key=api_key)
    log.info("Pulling FRED data...")

    def get_fred_series(name: str, series_id: str,
                        observation_start: str, observation_end: str) -> pd.Series:
        """Pull one FRED series and make failures identify the exact series."""
        try:
            print(f"Pulling FRED series: {name} = {series_id}")
            s = fred.get_series(
                series_id,
                observation_start=observation_start,
                observation_end=observation_end,
            )
            s.index = pd.to_datetime(s.index)
            return s
        except Exception as e:
            print(e)
            raise RuntimeError(
                f"Bad FRED series: {name} = {series_id}"
            ) from e

    # Monthly series — pull directly
    monthly_series = {
        "fed_funds":    "FEDFUNDS",
        "manufacturing_production": "IPMAN",
        "unemployment": "UNRATE",
    }

    # Daily series — pull daily, resample to month-end
    daily_series = {
        "yield_2y":  "DGS2",
        "yield_10y": "DGS10",
        "spread":    "T10Y2Y",
        "vix":       "VIXCLS",
        "credit_spread_baa10y": "BAA10Y",
    }

    frames = {}

    # Pull 12 months of lookback so YoY fields and 6M changes exist at TRAIN_START
    macro_start = (
        pd.to_datetime(start)
        .to_period("M")
        .to_timestamp()
        - pd.DateOffset(months=12)
    ).strftime("%Y-%m-%d")

    cpi_raw = get_fred_series("cpi", "CPIAUCSL", macro_start, end)

    # Oil — WTI spot (daily). Pulled with the same 12-month lookback as CPI so a
    # YoY value exists at TRAIN_START. Many geopolitical/inflation notes lean on
    # oil shocks; this gives the teacher a number to anchor Energy-sector calls.
    oil_raw = get_fred_series("oil_wti", "DCOILWTICO", macro_start, end)

    # GDP — quarterly. Pull extra lookback so the PiT availability shift still
    # leaves a valid (forward-filled) value at TRAIN_START.
    gdp_raw = get_fred_series("gdp_qoq", "A191RL1Q225SBEA", "2011-01-01", end)

    # Pull monthly series
    for name, series_id in monthly_series.items():
        s = get_fred_series(name, series_id, macro_start, end)
        frames[name] = s.resample("ME").last()

    # Pull daily series — resample to month-end
    for name, series_id in daily_series.items():
        s = get_fred_series(name, series_id, start, end)
        frames[name] = s.resample("ME").last()

    # CPI YoY
    cpi_monthly = cpi_raw.resample("ME").last()
    cpi_yoy = cpi_monthly.pct_change(12) * 100
    frames["cpi_yoy"] = cpi_yoy.loc[start:end]

    # Oil: month-end WTI level + YoY. NOTE: the 2020-04 negative WTI print was an
    # intraday futures event; the month-END level (~$18) is positive, so the
    # month-end series is well-behaved. As with manufacturing IP, the 2021 YoY
    # readings are large base effects off the 2020 crash — directionally real but
    # mechanically inflated; the teacher should read them as base effects.
    oil_monthly = oil_raw.resample("ME").last()
    oil_yoy = oil_monthly.pct_change(12) * 100
    frames["oil_wti"]     = oil_monthly.loc[start:end]
    frames["oil_wti_yoy"] = oil_yoy.loc[start:end]

    # ── GDP: point-in-time availability mapping (FIX) ──────────────────────────
    # A191RL1Q225SBEA is stamped at the QUARTER START (Q1 -> Jan 1). Its advance
    # estimate is released ~1 month after the quarter ENDS. Map each quarterly
    # value to the month-end at which it first becomes public, then ffill.
    #   Q1 (Jan 1) + GDP_RELEASE_LAG_MONTHS(=4) -> May 31  (safely post-release)
    # The OLD code did `.resample("ME").last().shift(1)`, which surfaced each
    # quarter ~2 months early -> look-ahead. This replaces it.
    gdp_raw = gdp_raw.copy()
    gdp_raw.index = pd.to_datetime(gdp_raw.index)
    gdp_avail_index = (
        gdp_raw.index + pd.DateOffset(months=GDP_RELEASE_LAG_MONTHS)
    ) + pd.offsets.MonthEnd(0)
    gdp_avail = pd.Series(gdp_raw.values, index=gdp_avail_index)
    gdp_avail = gdp_avail[~gdp_avail.index.duplicated(keep="last")].sort_index()
    gdp_monthly = gdp_avail.reindex(frames["fed_funds"].index).ffill()
    frames["gdp_qoq"] = gdp_monthly.loc[start:end]

    # DIAGNOSTIC (change 3): print quarter -> value -> first-visible month for the
    # most recent quarters so the GDP cells can be eyeballed against BEA prints.
    # Reference: BEA has Q1-2024 = 1.6% and Q2-2024 = 3.0%. If the rows below show
    # materially different numbers, the FRED vintage differs from the released
    # prints — confirm before generating, since reviewers will know these values.
    log.info("GDP availability map (quarter_start -> value -> first visible):")
    for q_start, avail_m, val in zip(
        gdp_raw.index[-10:], gdp_avail_index[-10:], gdp_raw.values[-10:]
    ):
        v = "  n/a" if (val is None or np.isnan(val)) else f"{val:5.1f}%"
        log.info(f"    {pd.Timestamp(q_start).date()}  ->  {v}  ->  "
                 f"visible {pd.Timestamp(avail_m).date()}")

    macro = pd.DataFrame(frames)
    macro.index = pd.to_datetime(macro.index)

    # Manufacturing production YoY growth, used as a cycle activity signal
    macro["manufacturing_production_yoy"] = (
        macro["manufacturing_production"].pct_change(12) * 100
    )

    # Derived fields
    macro["fed_funds_6m_change"] = macro["fed_funds"].diff(6).round(4)
    macro["yield_curve_10y2y"] = (macro["yield_10y"] - macro["yield_2y"]).round(4)

    log.info(f"Macro data: {len(macro)} months, {macro.columns.tolist()}")
    return macro

# ── REGIME DERIVATION ─────────────────────────────────────────────────────────

def _is_num(x) -> bool:
    return x is not None and not (isinstance(x, float) and np.isnan(x))


def derive_regime(row: pd.Series, spread_series: pd.Series,
                  as_of_date: pd.Timestamp) -> dict:
    """Derive three-axis regime label from macro snapshot."""

    # ── Rate environment (unchanged) ──
    chg = row.get("fed_funds_6m_change", None)
    if not _is_num(chg):
        rate_env = None
    elif chg > RATE_TIGHTENING_THRESH:
        rate_env = "tightening"
    elif chg < RATE_EASING_THRESH:
        rate_env = "easing"
    else:
        rate_env = "stable"

    # ── Risk regime (unchanged) ──
    vix = row.get("vix", None)
    credit_spread = row.get("credit_spread_baa10y", None)
    if not _is_num(vix) or not _is_num(credit_spread):
        risk = None
    elif vix > VIX_STRESSED_THRESH or credit_spread > CREDIT_SPREAD_STRESSED_THRESH:
        risk = "stressed"
    elif vix < VIX_CALM_THRESH and credit_spread < CREDIT_SPREAD_CALM_THRESH:
        risk = "calm"
    else:
        risk = "normal"

    # ── Cycle (REVISED v2) ──
    # v1 fixed the Sep-2022 -> Dec-2024 over-contraction but still fired
    # "contraction" on (weak mfg + inverted curve) when the LAGGED GDP value
    # wasn't high enough to veto, producing two false positives (2023-10,
    # 2024-07) in stretches that were actually expanding.
    # v2: contraction requires an OUTRIGHT decline in real output (gdp_qoq < 0).
    # An inverted curve PREDICTS a future recession; it does not make the present
    # quarter a contraction, so it no longer forces the label. A manufacturing
    # recession with an inverted curve and non-negative GDP now falls through to
    # "mid_cycle" (a slowdown), which is the correct present-tense reading.
    gdp     = row.get("gdp_qoq", None)
    mfg_yoy = row.get("manufacturing_production_yoy", None)

    spread_window = spread_series.loc[:as_of_date].tail(CURVE_INVERSION_MONTHS)
    inverted = (len(spread_window) == CURVE_INVERSION_MONTHS and
                (spread_window < 0).all())

    gdp_decline = _is_num(gdp) and gdp < GDP_CONTRACTION_THRESH
    gdp_strong  = _is_num(gdp) and gdp > GDP_EXPANSION_THRESH
    mfg_weak    = _is_num(mfg_yoy) and mfg_yoy <= MFG_CONTRACTION_THRESH
    mfg_strong  = _is_num(mfg_yoy) and mfg_yoy > MFG_EXPANSION_THRESH

    if not _is_num(mfg_yoy) and not _is_num(gdp):
        cycle = None
    elif gdp_decline:
        # Outright real-output decline is the ONLY trigger for contraction.
        cycle = "contraction"
    elif gdp_strong and not mfg_weak:
        cycle = "expansion"
    elif mfg_strong and not inverted:
        cycle = "expansion"
    else:
        # Includes mfg recession + inverted curve with non-negative GDP: a
        # slowdown / late-cycle read, not a present-tense contraction.
        cycle = "mid_cycle"

    return {
        "rate_environment": rate_env,
        "risk_regime":      risk,
        "cycle":            cycle,
    }


# ── SAFE ROUND ────────────────────────────────────────────────────────────────

def _r(val, dp=2):
    """Round a value safely, return None if NaN."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return round(float(val), dp)


# ── BUILD ONE EXAMPLE ─────────────────────────────────────────────────────────

def build_example(date: pd.Timestamp, row: pd.Series,
                  spread_series: pd.Series) -> dict:
    """Build one complete economist training example for a given month-end."""

    as_of_str = date.strftime("%Y-%m-%d")
    month_key  = date.strftime("%Y-%m")

    # Get manual context labels
    ctx = MANUAL_CONTEXT.get(month_key, {
        "fed_tone": "neutral",
        "fed_note": "No FOMC meeting this month, policy unchanged",
        "geo_risk": "normal",
        "geo_note":  "No major geopolitical supply disruption",
    })

    # Derive regime
    regime = derive_regime(row, spread_series, date)

    # Build macro snapshot
    macro_snapshot = {
        "fed_funds_rate":      _r(row.get("fed_funds"),        2),
        "fed_funds_6m_change": _r(row.get("fed_funds_6m_change"), 2),
        "yield_2y":            _r(row.get("yield_2y"),         2),
        "yield_10y":           _r(row.get("yield_10y"),        2),
        "yield_curve_10y2y":   _r(row.get("yield_curve_10y2y"),2),
        "vix":                 _r(row.get("vix"),              2),
        "credit_spread_baa10y": _r(row.get("credit_spread_baa10y"), 2),
        "manufacturing_production_yoy": _r(row.get("manufacturing_production_yoy"), 1),
        "cpi_yoy":             _r(row.get("cpi_yoy"),          1),
        "unemployment_rate":   _r(row.get("unemployment"),     1),
        "gdp_qoq":             _r(row.get("gdp_qoq"),          1),
        "oil_wti":             _r(row.get("oil_wti"),          1),
        "oil_wti_yoy":         _r(row.get("oil_wti_yoy"),      1),
    }

    macro_context = {
        "fed_statement_tone": ctx["fed_tone"],
        "fed_statement_note": ctx["fed_note"],
        "geopolitical_risk":  ctx["geo_risk"],
        "geopolitical_note":  ctx["geo_note"],
    }

    user_payload = {
        "task":             "economist_regime_and_sector_signal",
        "as_of_date":       as_of_str,
        "horizon":          "1M",
        "macro_snapshot":   macro_snapshot,
        "macro_context":    macro_context,
        "derived_regime":   regime,
        "universe_sectors": UNIVERSE_SECTORS,
    }

    return {
        "id": f"{as_of_str}_ECONOMIST",
        "messages": [
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": json.dumps(user_payload,
                                                       separators=(",", ":"),
                                                       allow_nan=False)},
            {"role": "model",   "content": None},
        ],
        "metadata": {
            "as_of_date":       as_of_str,
            "regime":           regime,   # NOTE: derived (pre-teacher) label
            "fed_tone":         ctx["fed_tone"],
            "geopolitical_risk":ctx["geo_risk"],
        },
    }


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    if not FRED_API_KEY:
        raise ValueError(
            "Set your FRED API key: export FRED_API_KEY=your_key_here\n"
            "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
        )

    macro = pull_fred_data(FRED_API_KEY, TRAIN_START, TRAIN_END)

    # Filter to training window month-ends
    dates = macro.loc[TRAIN_START:TRAIN_END].index
    log.info(f"Building examples for {len(dates)} month-ends "
             f"({TRAIN_START} to {TRAIN_END})")

    spread_series = macro["yield_curve_10y2y"].dropna()

    examples  = []
    skipped   = []

    for date in dates:
        row = macro.loc[date]

        # Skip if critical fields are missing
        missing = [f for f in ["fed_funds", "vix", "credit_spread_baa10y", "manufacturing_production_yoy", "cpi_yoy"]
                   if pd.isna(row.get(f))]
        if missing:
            log.warning(f"Skipping {date.date()} — missing: {missing}")
            skipped.append({"date": str(date.date()), "missing": missing})
            continue

        examples.append(build_example(date, row, spread_series))

    # Write JSONL
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, allow_nan=False) + "\n")

    log.info(f"Wrote {len(examples)} examples to {OUT_FILE}")
    if skipped:
        log.warning(f"Skipped {len(skipped)} dates: {[s['date'] for s in skipped]}")

    # Quick sanity check
    print("\n── SANITY CHECK ──────────────────────────────")
    print(f"Total examples:  {len(examples)}")
    print(f"Skipped:         {len(skipped)}")

    # Regime distribution
    from collections import Counter
    rate_dist  = Counter(e["metadata"]["regime"]["rate_environment"] for e in examples)
    risk_dist  = Counter(e["metadata"]["regime"]["risk_regime"]      for e in examples)
    cycle_dist = Counter(e["metadata"]["regime"]["cycle"]            for e in examples)
    tone_dist  = Counter(e["metadata"]["fed_tone"]                   for e in examples)
    geo_dist   = Counter(e["metadata"]["geopolitical_risk"]          for e in examples)

    print(f"\nRate env:  {dict(rate_dist)}")
    print(f"Risk:      {dict(risk_dist)}")
    print(f"Cycle:     {dict(cycle_dist)}")
    print(f"Fed tone:  {dict(tone_dist)}")
    print(f"Geo risk:  {dict(geo_dist)}")

    # Show one example
    ex = examples[min(100, len(examples) - 1)]
    payload = json.loads(ex["messages"][1]["content"])
    print(f"\n── SAMPLE EXAMPLE ({ex['id']}) ──")
    print(json.dumps(payload["macro_snapshot"], indent=2))
    print(json.dumps(payload["derived_regime"],  indent=2))
    print(json.dumps(payload["macro_context"],   indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()