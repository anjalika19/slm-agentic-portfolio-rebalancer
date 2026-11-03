"""Prompt registry. All specialist + debate prompts, ported from the old notebook
and folder builders (see per-file PORTING NOTEs)."""

from agents.prompts import quant, fundamental, sentiment, economist
from agents.prompts.debate import (
    opening_statement_prompt,
    rebuttal_statement_prompt,
    MODERATOR_SYNTHESIS_PROMPT,
)

SPECIALIST_PROMPTS = {
    "quant": quant.SYSTEM_PROMPT,
    "fundamental": fundamental.SYSTEM_PROMPT,
    "sentiment": sentiment.SYSTEM_PROMPT,
    "economist": economist.SYSTEM_PROMPT,
}

__all__ = [
    "SPECIALIST_PROMPTS",
    "opening_statement_prompt",
    "rebuttal_statement_prompt",
    "MODERATOR_SYNTHESIS_PROMPT",
]
