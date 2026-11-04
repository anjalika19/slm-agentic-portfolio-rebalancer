from __future__ import annotations

import json
from contextlib import ExitStack, contextmanager
from typing import Any, Dict, Optional

from opentelemetry.trace import Status, StatusCode

from agentic_bl import config
_tracer_provider = None
_tracer = None
_init_error: Optional[str] = None


def _jsonable(value: Any, max_chars: int = 12000) -> Any:
    """Make payloads safe-ish for trace display.

    Phoenix can store large inputs/outputs, but avoid sending huge full data
    packages by truncating serialized values.
    """
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        text = str(value)

    if len(text) > max_chars:
        text = text[:max_chars] + "...[truncated]"

    try:
        return json.loads(text)
    except Exception:
        return text


def _clean_attrs(attrs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """OTel attributes should be primitives or primitive lists."""
    clean: Dict[str, Any] = {}
    for k, v in (attrs or {}).items():
        if v is None:
            continue
        if isinstance(v, (str, bool, int, float)):
            clean[k] = v
        elif isinstance(v, (list, tuple)) and all(isinstance(x, (str, bool, int, float)) for x in v):
            clean[k] = list(v)
        else:
            clean[k] = json.dumps(v, ensure_ascii=False, default=str)[:2000]
    return clean


def get_tracer():
    global _tracer_provider, _tracer, _init_error

    if not config.PHOENIX_ENABLED:
        return None

    if _tracer is not None:
        return _tracer

    try:
        from phoenix.otel import register

        kwargs = {
            "project_name": config.PHOENIX_PROJECT_NAME,
            "protocol": config.PHOENIX_OTEL_PROTOCOL,
            # Auto instrumentation is useful if you later add OpenAI/LangChain clients.
            # Your current direct ollama.Client calls still need manual spans below.
            "auto_instrument": True,
        }

        if config.PHOENIX_COLLECTOR_ENDPOINT:
            kwargs["endpoint"] = config.PHOENIX_COLLECTOR_ENDPOINT

        _tracer_provider = register(**kwargs)
        _tracer = _tracer_provider.get_tracer("agentic_slm_research")
        return _tracer

    except Exception as exc:
        _init_error = str(exc)
        return None


@contextmanager
def trace_context(session_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, tags=None):
    """Attach run/session metadata to all child spans."""
    if not config.PHOENIX_ENABLED:
        yield
        return

    try:
        from phoenix.otel import using_metadata, using_session, using_tags

        with ExitStack() as stack:
            if session_id:
                stack.enter_context(using_session(session_id))
            if metadata:
                stack.enter_context(using_metadata(_jsonable(metadata, max_chars=4000)))
            if tags:
                stack.enter_context(using_tags(list(tags)))
            yield
    except Exception:
        # Observability should never break portfolio execution.
        yield


@contextmanager
def start_span(
    name: str,
    kind: str = "chain",
    input_value: Any = None,
    attributes: Optional[Dict[str, Any]] = None,
):
    tracer = get_tracer()

    if tracer is None:
        yield None
        return

    with tracer.start_as_current_span(
        name,
        openinference_span_kind=kind,
        attributes=_clean_attrs(attributes),
    ) as span:
        try:
            if input_value is not None and config.PHOENIX_TRACE_INPUTS:
                span.set_input(_jsonable(input_value))

            yield span

        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            raise

        else:
            span.set_status(Status(StatusCode.OK))


def set_span_output(span, output_value: Any) -> None:
    if span is None or not config.PHOENIX_TRACE_OUTPUTS:
        return
    try:
        span.set_output(_jsonable(output_value))
    except Exception:
        pass


def set_span_attrs(span, attrs: Dict[str, Any]) -> None:
    if span is None:
        return
    try:
        span.set_attributes(_clean_attrs(attrs))
    except Exception:
        pass