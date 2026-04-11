from __future__ import annotations

import os
from typing import Any


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _maybe_build_openai_llm(prefix: str, default_model: str, default_max_tokens: int) -> Any:
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
