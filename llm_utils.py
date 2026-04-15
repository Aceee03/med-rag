from __future__ import annotations

import os
from typing import Any

DEFAULT_LLM_PROVIDER = "ollama"
DEFAULT_OLLAMA_MODEL = "gpt-oss:20b"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _thinking_env(name: str) -> bool | str | None:
    raw = os.getenv(name)
    if not raw:
        return None
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    if value in {"low", "medium", "high"}:
        return value
    return None


def _build_ollama_llm(prefix: str, default_model: str, default_max_tokens: int) -> Any:
    model = os.getenv(f"{prefix}_MODEL") or os.getenv("OLLAMA_MODEL") or default_model or DEFAULT_OLLAMA_MODEL
    request_timeout = _float_env(
        f"{prefix}_TIMEOUT",
        _float_env("OLLAMA_REQUEST_TIMEOUT", 300.0),
    )
    thinking = _thinking_env(f"{prefix}_THINKING")
    if thinking is None:
        thinking = _thinking_env("OLLAMA_THINKING")

    from llama_index.llms.ollama import Ollama

    kwargs: dict[str, Any] = {
        "model": model,
        "base_url": os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
        "temperature": _float_env(f"{prefix}_TEMPERATURE", 0.0),
        "request_timeout": request_timeout,
    }

    max_tokens = _int_env(f"{prefix}_MAX_TOKENS", default_max_tokens)
    if max_tokens > 0:
        kwargs["num_output"] = max_tokens

    context_window = _int_env(f"{prefix}_CONTEXT_WINDOW", 0)
    if context_window > 0:
        kwargs["context_window"] = context_window

    if thinking is not None:
        kwargs["thinking"] = thinking

    return Ollama(**kwargs)


def _build_openai_llm(prefix: str, default_model: str, default_max_tokens: int) -> Any:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    model = os.getenv(f"{prefix}_MODEL", default_model)
    max_tokens = _int_env(f"{prefix}_MAX_TOKENS", default_max_tokens)
    reasoning_effort = os.getenv(f"{prefix}_REASONING_EFFORT")

    from llama_index.llms.openai import OpenAI

    kwargs = {
        "model": model,
        "api_key": api_key,
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort

    try:
        return OpenAI(**kwargs)
    except TypeError:
        kwargs.pop("reasoning_effort", None)
        return OpenAI(**kwargs)


def _maybe_build_openai_llm(prefix: str, default_model: str, default_max_tokens: int) -> Any:
    provider = (os.getenv("LLM_PROVIDER") or DEFAULT_LLM_PROVIDER).strip().lower()

    if provider == "ollama":
        return _build_ollama_llm(prefix, default_model=DEFAULT_OLLAMA_MODEL, default_max_tokens=default_max_tokens)

    if provider == "openai":
        return _build_openai_llm(prefix, default_model, default_max_tokens)

    if provider == "auto":
        try:
            return _build_ollama_llm(prefix, default_model=DEFAULT_OLLAMA_MODEL, default_max_tokens=default_max_tokens)
        except Exception:
            return _build_openai_llm(prefix, default_model, default_max_tokens)

    raise ValueError(f"Unsupported LLM_PROVIDER={provider!r}. Use 'ollama', 'openai', or 'auto'.")
