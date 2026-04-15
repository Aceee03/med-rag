from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

#DEFAULT_OLLAMA_MODEL = "qwen35-27b-local:latest"
DEFAULT_OLLAMA_MODEL = "gpt-oss:20b"

DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
STARTUP_TEST_PROMPT = "Reply with exactly OK."
AVAILABLE_MODELS: list[str] = []
# Options: "resume" or "scratch". This only applies when you run the file with no CLI args.
DEFAULT_RUN_MODE = "scratch"
# Edit these values directly if you want low/medium thinking without using env vars.
DEFAULT_STAGE_THINKING: dict[str, bool | str] = {
    "PIPELINE_ENRICH": "low",
    "PIPELINE_SUMMARY": "low",
    "PIPELINE_GRAPH": "low",
    "PIPELINE_REPAIR": "low",
    "PIPELINE_ANSWER": "low",
}


def _status(message: str) -> None:
    print(f"[pipeline_ollama] {message}", flush=True)


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


def _thinking_env(name: str, default: bool | str) -> bool | str:
    raw = os.getenv(name)
    if not raw:
        return default

    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    if value in {"low", "medium", "high"}:
        return value
    return default


def _default_thinking_for_prefix(prefix: str) -> bool | str:
    return DEFAULT_STAGE_THINKING.get(prefix, True)


def _default_timeout_for_prefix(prefix: str) -> float:
    if prefix == "PIPELINE_ENRICH":
        return 120.0
    if prefix in {"PIPELINE_GRAPH", "PIPELINE_REPAIR"}:
        return 900.0
    if prefix in {"PIPELINE_SUMMARY", "PIPELINE_ANSWER"}:
        return 300.0
    return 300.0


def _base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).rstrip("/")


def _apply_default_run_mode(argv: list[str]) -> list[str]:
    if argv:
        return argv

    mode = DEFAULT_RUN_MODE.strip().lower()
    if mode == "resume":
        _status("Default run mode: resume from latest checkpoints")
        return argv

    if mode == "scratch":
        scratch_args = [
            "--force-rebuild-nodes",
            "--force-rebuild-enrichment",
            "--force-rebuild-graph",
            "--force-rebuild-clean-graph",
            "--force-rebuild-repair-graph",
            "--force-rebuild-clinical-repair-graph",
            "--force-rebuild-communities",
            "--no-resume-graph",
        ]
        _status("Default run mode: rebuild from scratch")
        return [*argv, *scratch_args]

    raise RuntimeError(
        f"Unsupported DEFAULT_RUN_MODE={DEFAULT_RUN_MODE!r}. Use 'resume' or 'scratch'."
    )


def _preferred_default_model(prefix: str) -> str:
    _ = prefix
    return DEFAULT_OLLAMA_MODEL


def _model_for_prefix(prefix: str, default_model: str) -> str:
    _ = prefix
    _ = default_model
    return DEFAULT_OLLAMA_MODEL


def _load_dotenv_if_available() -> None:
    env_path = Path(".env")
    if not env_path.exists():
        _status("No .env file found. Using process environment and launcher defaults.")
        return

    try:
        from dotenv import load_dotenv
    except ImportError:
        _status("python-dotenv is not installed. Skipping .env load.")
        return

    load_dotenv(env_path, override=False)
    _status(f"Loaded environment from {env_path.resolve()}")


def _require_ollama_adapter() -> None:
    try:
        importlib.import_module("llama_index.llms.ollama")
    except ImportError as exc:
        raise RuntimeError(
            "Missing Ollama adapter. Install it with: python -m pip install llama-index-llms-ollama"
        ) from exc


def _http_json(method: str, url: str, payload: dict[str, Any] | None = None, timeout: float = 15.0) -> dict[str, Any]:
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

    request = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama request failed with HTTP {exc.code}: {details}") from exc
    except URLError as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {url}. Start Ollama and confirm the base URL."
        ) from exc

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Ollama returned non-JSON output from {url}: {raw[:200]!r}") from exc


def _fetch_available_models(base_url: str) -> list[str]:
    payload = _http_json("GET", f"{base_url}/api/tags", timeout=10.0)
    models = payload.get("models", [])
    return [str(item.get("name", "")).strip() for item in models if str(item.get("name", "")).strip()]


def _resolve_model_name(model: str, available_models: list[str]) -> str:
    if model in available_models:
        return model
    if ":" not in model and f"{model}:latest" in available_models:
        return f"{model}:latest"
    raise RuntimeError(
        "Configured Ollama model is not installed.\n"
        f"Requested: {model}\n"
        f"Available: {', '.join(available_models) if available_models else '(none)'}"
    )


def _test_model(base_url: str, model: str) -> str:
    payload = {
        "model": model,
        "prompt": STARTUP_TEST_PROMPT,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0,
            "num_predict": 16,
        },
    }
    response = _http_json("POST", f"{base_url}/api/generate", payload=payload, timeout=90.0)
    text = str(response.get("response", "")).strip()
    if not text:
        message = response.get("message") or {}
        text = str(message.get("content", "")).strip()
    if not text:
        text = str(response.get("thinking", "")).strip()
    if not text:
        raise RuntimeError(f"Ollama model {model} returned an empty response during the startup test.")
    return text


def _build_ollama_llm(prefix: str, default_model: str, default_max_tokens: int) -> Any:
    model = _model_for_prefix(prefix, default_model)
    thinking = _thinking_env(
        f"{prefix}_THINKING",
        _thinking_env("OLLAMA_THINKING", _default_thinking_for_prefix(prefix)),
    )
    request_timeout = _float_env(
        f"{prefix}_TIMEOUT",
        _float_env("OLLAMA_REQUEST_TIMEOUT", _default_timeout_for_prefix(prefix)),
    )

    try:
        from llama_index.llms.ollama import Ollama
    except ImportError as exc:
        raise RuntimeError(
            "Missing Ollama adapter. Install it with: python -m pip install llama-index-llms-ollama"
        ) from exc

    resolved_model = _resolve_model_name(model, AVAILABLE_MODELS or _fetch_available_models(_base_url()))

    ollama_kwargs: dict[str, Any] = {
        "model": resolved_model,
        "base_url": _base_url(),
        "request_timeout": request_timeout,
        "temperature": _float_env(
            f"{prefix}_TEMPERATURE",
            _float_env("OLLAMA_TEMPERATURE", 0.0),
        ),
        "thinking": thinking,
    }

    context_window = _int_env(
        f"{prefix}_CONTEXT_WINDOW",
        _int_env("OLLAMA_CONTEXT_WINDOW", 0),
    )
    if context_window > 0:
        ollama_kwargs["context_window"] = context_window

    max_tokens = _int_env(f"{prefix}_MAX_TOKENS", default_max_tokens)
    if max_tokens > 0:
        ollama_kwargs["num_output"] = max_tokens

    _status(
        "Creating Ollama client "
        f"for {prefix} with model={resolved_model}, base_url={ollama_kwargs['base_url']}, "
        f"max_tokens={max_tokens}, timeout={request_timeout}, thinking={thinking}"
    )
    return Ollama(**ollama_kwargs)


def _patch_pipeline_module() -> Any:
    try:
        pipeline = importlib.import_module("pipeline")
    except ImportError as exc:
        raise RuntimeError(
            "Could not import pipeline.py. Install project dependencies with: python -m pip install -r requirements.txt"
        ) from exc

    pipeline._maybe_build_openai_llm = _build_ollama_llm
    return pipeline


def _startup_check() -> tuple[str, str]:
    global AVAILABLE_MODELS

    _load_dotenv_if_available()
    _require_ollama_adapter()

    base_url = _base_url()
    default_model = _model_for_prefix("PIPELINE_GRAPH", DEFAULT_OLLAMA_MODEL)
    _status(f"Checking Ollama server at {base_url}")
    AVAILABLE_MODELS = _fetch_available_models(base_url)
    _status(f"Detected {len(AVAILABLE_MODELS)} local model(s)")
    _status(
        f"Default run mode: {DEFAULT_RUN_MODE}. "
        + "Thinking defaults: "
        + ", ".join(
            f"{stage}={value}" for stage, value in DEFAULT_STAGE_THINKING.items()
        )
    )

    resolved_model = _resolve_model_name(default_model, AVAILABLE_MODELS)
    _status(f"Using startup test model: {resolved_model}")
    _status("Running a short LLM smoke test (thinking disabled for probe)")
    test_response = _test_model(base_url, resolved_model)
    _status(f"LLM test succeeded. Sample response: {test_response[:120]!r}")
    return base_url, resolved_model


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    _status("Launcher started")
    _status(f"Python executable: {sys.executable}")

    try:
        argv = _apply_default_run_mode(argv)
        base_url, model = _startup_check()
        _status(f"Ollama is available at {base_url}")
        _status(f"Primary model ready: {model}")
        pipeline = _patch_pipeline_module()
        _status("Pipeline module loaded and LLM builder patched to Ollama")
        _status(f"Delegating to pipeline.py with args: {argv if argv else '(no CLI args)'}")
        exit_code = int(pipeline.main(argv))
        _status(f"pipeline.py finished with exit code {exit_code}")
        return exit_code
    except Exception as exc:
        _status(f"Startup failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
