"""Shared structured-output client for Gemma/Ollama and Gemini.

Every specialist, moderator, validator, and single-model control routes through
``structured_call``. ``model_family`` selects the backend independently of the
agent architecture, which makes the four factorial configurations directly
comparable without duplicating pipeline logic.

Behaviour on failure (spec 3f):
  * one re-prompt on a schema/JSON validation failure;
  * if the model is still unusable and ``STRICT_MODEL=True`` -> raise (real runs);
  * if ``STRICT_MODEL=False`` -> call the supplied deterministic ``mock_fn``, and
    the result carries ``used_mock=True`` so the caller emits a loud warning.

Nothing here is ever silently mocked.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Type

from pydantic import BaseModel

from agentic_bl import config
try:
    import ollama
except Exception:  # pragma: no cover
    ollama = None


try:
    from google import genai
    from google.genai import types as genai_types
except Exception:  # pragma: no cover
    genai = None
    genai_types = None



@dataclass
class CallResult:
    data: Dict[str, Any]
    used_mock: bool
    note: str = ""


_client = None
_gemini_client = None
_thread_local = threading.local()


def _get_client():
    """Return an Ollama client.

    In sequential mode this behaves like the old singleton. In optional parallel
    mode, each worker thread gets its own client object so concurrent HTTP calls
    do not share mutable client/session state.
    """
    global _client
    if ollama is None:
        raise RuntimeError("the 'ollama' package is not installed")
    if getattr(config, "PARALLEL_AGENT_CALLS", False):
        client = getattr(_thread_local, "ollama_client", None)
        if client is None:
            client = ollama.Client(host=config.OLLAMA_BASE_URL)
            _thread_local.ollama_client = client
        return client
    if _client is None:
        _client = ollama.Client(host=config.OLLAMA_BASE_URL)
    return _client

def get_gemini_client():
    global _gemini_client
    if genai is None:
        raise RuntimeError("the 'google-genai' package is not installed")
    if not config.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is required for Gemini experiments")
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _gemini_client

def ollama_available() -> bool:
    if ollama is None:
        return False
    try:
        _get_client().list()
        return True
    except Exception:
        return False

def gemini_available() -> bool:
    if genai is None or not config.GEMINI_API_KEY:
        return False
    try:
        get_gemini_client()
        return True
    except Exception:
        return False


def _ollama_structured(
    model: str, system_prompt: str, user_payload: Dict[str, Any], schema_model: Type[BaseModel], temperature: float
) -> Dict[str, Any]:
    """One structured call to Ollama, validated against ``schema_model``.

    Re-prompts ONCE on JSON/schema failure (spec Phase 1). Raises on hard failure.
    """
    client = _get_client()
    schema = schema_model.model_json_schema()
    last_err: Optional[Exception] = None

    for attempt in range(2):
        repair = ""
        if attempt > 0:
            repair = (
                "\n\nYour previous response failed JSON/schema validation. "
                "Return corrected strict JSON only. No markdown."
            )
        messages = [
            {"role": "system", "content": system_prompt + repair},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, default=str)},
        ]
        try:
            resp = client.chat(
                model=model,
                messages=messages,
                format=schema,
                options={"temperature": temperature, "seed": config.MODEL_SEED},
            )
            content = resp["message"]["content"]
            parsed = _extract_json(content)
            validated = schema_model.model_validate(parsed)
            return validated.model_dump()
        except Exception as e:  # noqa: BLE001 - retry then surface
            last_err = e

    raise RuntimeError(f"Ollama structured call failed after retry: {last_err}")

def _gemini_response_text(resp: Any) -> str:
    text = getattr(resp, "text", None)
    if text:
        return text
    try:
        return resp.candidates[0].content.parts[0].text
    except Exception:
        return str(resp)


def _inline_ref(ref: str, root: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve a local Pydantic JSON-schema ref such as #/$defs/Foo."""
    if not ref.startswith("#/$defs/"):
        return {}
    name = ref.rsplit("/", 1)[-1]
    target = (root.get("$defs") or {}).get(name, {})
    return json.loads(json.dumps(target))


def _gemini_dynamic_keys(user_payload: Dict[str, Any]) -> Dict[str, list[str]]:
    """Known dictionary keys for Gemini's restricted structured-output schema."""
    portfolio_context = user_payload.get("portfolio_context") or {}
    universe = portfolio_context.get("portfolio_universe") or user_payload.get("universe") or []
    tickers = [str(t).strip().upper() for t in universe if str(t).strip()]

    agent_input = user_payload.get("agent_input") or {}
    mapping = agent_input.get("universe_sector_mapping") or {}
    sectors = sorted({str(v) for v in mapping.values() if str(v).strip()})
    return {
        "signals": tickers,
        "stock_signals": tickers,
        "sector_views": sectors,
    }


def _gemini_clean_schema(
    schema: Any,
    root: Dict[str, Any],
    *,
    dynamic_keys: Optional[Dict[str, list[str]]] = None,
    field_name: str = "",
) -> Any:
    """Convert Pydantic JSON Schema into the smaller Gemini Developer API subset.

    The Developer API structured-output path is stricter than full JSON Schema.
    In particular, Pydantic emits ``additionalProperties`` for ``Dict[...]``
    fields, which Gemini can reject.  We strip unsupported/control-only keys and
    inline local refs so the API receives a simple object/array/string schema.
    Local Pydantic validation still enforces the full schema after generation.
    """
    if isinstance(schema, list):
        return [
            _gemini_clean_schema(x, root, dynamic_keys=dynamic_keys, field_name=field_name)
            for x in schema
        ]
    if not isinstance(schema, dict):
        return schema

    if "$ref" in schema:
        return _gemini_clean_schema(
            _inline_ref(str(schema["$ref"]), root),
            root,
            dynamic_keys=dynamic_keys,
            field_name=field_name,
        )

    out: Dict[str, Any] = {}
    for key, value in schema.items():
        if key in {"$defs", "$schema", "additionalProperties", "default", "title"}:
            continue
        # Gemini response_schema does not need const; enum is more portable.
        if key == "const":
            out["enum"] = [value]
            continue
        out[key] = _gemini_clean_schema(
            value,
            root,
            dynamic_keys=dynamic_keys,
            field_name=key if key not in {"properties", "items"} else field_name,
        )

    # Gemini Developer API rejects Pydantic's additionalProperties form. For the
    # model-generated dictionaries in this project, the keys are known from the
    # point-in-time payload, so expose them as explicit properties instead.
    additional = schema.get("additionalProperties")
    keys = (dynamic_keys or {}).get(field_name, [])
    if isinstance(additional, dict) and keys:
        item_schema = _gemini_clean_schema(
            additional,
            root,
            dynamic_keys=dynamic_keys,
            field_name=field_name,
        )
        out["type"] = "object"
        out["properties"] = {
            key: json.loads(json.dumps(item_schema))
            for key in keys
        }
        if field_name in {"signals", "stock_signals"}:
            out["required"] = list(keys)
    return out


def _gemini_response_schema(schema_model: Type[BaseModel], user_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Build a Gemini-safe schema for single- and multi-agent calls."""
    raw = schema_model.model_json_schema()
    dynamic_keys = _gemini_dynamic_keys(user_payload)
    schema = _gemini_clean_schema(raw, raw, dynamic_keys=dynamic_keys)

    # Pairwise views are deliberately derived in Python for the single-model
    # controls, so do not force that legacy field to be generated.
    if schema_model.__name__ == "SingleModelEmission":
        required = list(dict.fromkeys(list(schema.get("required", [])) + ["signals"]))
        schema["required"] = [name for name in required if name != "pairwise_views"]

    return schema


def _gemini_structured(
    model: str, system_prompt: str, user_payload: Dict[str, Any], schema_model: Type[BaseModel], temperature:float
) -> Dict[str, Any]:
    """One structured Gemini call, validated against ``schema_model``.

    The Google Gen AI SDK supports JSON/structured output with Pydantic schemas.
    We still locally validate and retry once so its behavior mirrors the Ollama path.
    """
    client = get_gemini_client()
    last_err: Optional[Exception] = None
    contents = json.dumps(user_payload, ensure_ascii=False, default=str)

    for attempt in range(2):
        repair = ""
        if attempt > 0:
            repair = (
                "\n\nYour previous response failed JSON/schema validation. "
                "Return corrected strict JSON only. No markdown."
            )
        try:
            if genai_types is not None:
                cfg = genai_types.GenerateContentConfig(
                    system_instruction=system_prompt + repair,
                    temperature=temperature,
                    response_mime_type="application/json",
                    response_schema=_gemini_response_schema(schema_model, user_payload),
                )
            else:  # defensive fallback for unusual SDK installs
                cfg = {
                    "system_instruction": system_prompt + repair,
                    "temperature": temperature,
                    "response_mime_type": "application/json",
                    "response_schema": _gemini_response_schema(schema_model, user_payload),
                }
            resp = client.models.generate_content(model=model, contents=contents, config=cfg)
            parsed = _extract_json(_gemini_response_text(resp))
            validated = schema_model.model_validate(parsed)
            return validated.model_dump()
        except Exception as e:  # noqa: BLE001 - retry then surface
            last_err = e

    raise RuntimeError(f"Gemini structured call failed after retry: {last_err}")



def structured_call(
    role: str,
    system_prompt: str,
    user_payload: Dict[str, Any],
    schema_model: Type[BaseModel],
    temperature: float,
    mock_fn: Callable[[], Dict[str, Any]],
    strict: Optional[bool] = None,
    model_family: Optional[str] = None,
    model_name: Optional[str] = None,
) -> CallResult:
    """Single entry point for every role call.

    ``strict`` overrides the global default so a single run (CLI/server/backtest)
    can choose strict-vs-mock without mutating module state.
    """
    if strict is None:
        strict = config.STRICT_MODEL
    model = model_name or config.model_for_role(role, model_family or "")
    provider = config.provider_for_role(role, model_family or "")
    if temperature is None:
        temperature = config.temperature_for_call(role)
    try:
        if provider == "gemini":
            data = _gemini_structured(model, system_prompt, user_payload, schema_model, temperature)
        else:
            data = _ollama_structured(model, system_prompt, user_payload, schema_model, temperature)
        return CallResult(data=data, used_mock=False, note=f"{provider}:{model}")
    except Exception as e:  # noqa: BLE001
        if strict:
            raise
        data = mock_fn()
        # validate the mock too so a bad mock surfaces immediately
        schema_model.model_validate(data)
        return CallResult(data=data, used_mock=True, note=f"mock fallback: {e}")


def _extract_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1] if "```" in text else text
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError(f"No JSON object in model response: {text[:300]}")
        return json.loads(text[start : end + 1])
