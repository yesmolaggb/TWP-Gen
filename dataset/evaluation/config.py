"""Shared configuration and LLM client for the TWP-Gen evaluation code.

Every credential, endpoint and model name is resolved from the repository-wide
configuration (``<repo>/twpgen_settings.py``, values in ``<repo>/twpgen_config.json``
or ``<repo>/.env``). Nothing is hard-coded here, so a single entry in the root
configuration serves the whole pipeline:

    export OPENAI_API_KEY=...            # or put llm.api_key in twpgen_config.json

Resolution order for the model settings:

    explicit argument -> environment variable -> twpgen_settings.llm -> built-in default
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

_HERE = Path(__file__).resolve()
EVALUATION_DIR = _HERE.parent
DATASET_DIR = _HERE.parents[1]
REPO_ROOT = _HERE.parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:  # central configuration
    import twpgen_settings as repo_config
except Exception:  # pragma: no cover - keep the module usable stand-alone
    repo_config = None  # type: ignore[assignment]


DEFAULT_MODEL = repo_config.llm.model if repo_config else "qwen3-32b"
DEFAULT_BASE_URL = (
    repo_config.llm.base_url
    if repo_config
    else "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
DEFAULT_TIMEOUT = repo_config.llm.timeout if repo_config else 360.0

ENV_FILE = REPO_ROOT / ".env"


def load_env_file(path: Path | None = None) -> Path | None:
    """Load ``KEY=VALUE`` pairs into ``os.environ``; already-set variables win."""
    target = Path(path) if path else ENV_FILE
    if not target.is_file():
        return None
    for raw in target.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
    return target


@dataclass(frozen=True)
class EvalSettings:
    """Resolved connection settings for every evaluation call."""

    api_key: str
    base_url: str
    model: str = DEFAULT_MODEL
    timeout: float = DEFAULT_TIMEOUT
    temperature: float = 0.1
    max_output_tokens: int = 4096

    def ensure_api_key(self) -> None:
        if not self.api_key:
            raise RuntimeError(
                "No API key configured. Set OPENAI_API_KEY (or ARTICLE_LLM_API_KEY) "
                f"in {ENV_FILE}, or put it under 'llm.api_key' in twpgen_config.json."
            )

    @property
    def label(self) -> str:
        return self.model


def _first_non_empty(*values: str | None) -> str:
    for value in values:
        if value:
            return value
    return ""


def resolve_settings(
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float | None = None,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
    env_file: str | Path | None = None,
) -> EvalSettings:
    """Resolve the evaluator settings once; every scoring script calls this."""
    load_env_file(env_file)

    config_llm = getattr(repo_config, "llm", None) if repo_config else None

    resolved_key = _first_non_empty(
        api_key,
        os.environ.get("ARTICLE_LLM_API_KEY"),
        os.environ.get("OPENAI_API_KEY"),
        getattr(config_llm, "api_key", None),
    )
    resolved_url = _first_non_empty(
        base_url,
        os.environ.get("ARTICLE_LLM_BASE_URL"),
        os.environ.get("OPENAI_BASE_URL"),
        os.environ.get("OPENAI_API_BASE"),
        getattr(config_llm, "base_url", None),
        DEFAULT_BASE_URL,
    )
    resolved_model = _first_non_empty(
        model,
        os.environ.get("ARTICLE_LLM_MODEL"),
        os.environ.get("TWPGEN_LLM_MODEL"),
        getattr(config_llm, "model", None),
        DEFAULT_MODEL,
    )

    env_timeout = os.environ.get("TWPGEN_LLM_TIMEOUT")
    resolved_timeout = float(
        timeout
        if timeout is not None
        else (env_timeout or getattr(config_llm, "timeout", None) or DEFAULT_TIMEOUT)
    )
    env_temperature = os.environ.get("TWPGEN_EVAL_TEMPERATURE")
    resolved_temperature = float(
        temperature if temperature is not None else (env_temperature or 0.1)
    )
    env_max_tokens = os.environ.get("TWPGEN_EVAL_MAX_TOKENS")
    resolved_max_tokens = int(
        max_output_tokens if max_output_tokens is not None else (env_max_tokens or 4096)
    )

    return EvalSettings(
        api_key=resolved_key,
        base_url=resolved_url,
        model=resolved_model,
        timeout=resolved_timeout,
        temperature=resolved_temperature,
        max_output_tokens=resolved_max_tokens,
    )


_CLIENT_CACHE: dict[tuple[str, str, float], object] = {}


def build_client(settings: EvalSettings):
    """Return a cached OpenAI-compatible client for the given settings."""
    settings.ensure_api_key()
    key = (settings.base_url, settings.api_key, settings.timeout)
    client = _CLIENT_CACHE.get(key)
    if client is None:
        from openai import OpenAI

        client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=settings.timeout,
        )
        _CLIENT_CACHE[key] = client
    return client


def dataset_dir() -> Path:
    """Folder holding the task list and this evaluation code."""
    return DATASET_DIR


def topics_file() -> Path:
    """Path of the 60-task JSON list, from the central configuration when available."""
    if repo_config is not None:
        candidate = Path(getattr(repo_config, "topic_dataset_file", ""))
        if candidate.is_file():
            return candidate
    return DATASET_DIR / "whitepaper_topics.json"


def default_results_dir() -> Path:
    """Where evaluation result files are written when no output path is given."""
    if repo_config is not None:
        output_dir = Path(getattr(repo_config, "output_dir", ""))
        if output_dir:
            return output_dir / "evaluation"
    return REPO_ROOT / "output" / "evaluation"


def load_topics(path: str | Path | None = None) -> list[dict]:
    """Load the task list: ``id``, ``title`` and ``domain`` per task."""
    import json

    target = Path(path) if path else topics_file()
    payload = json.loads(target.read_text(encoding="utf-8"))
    return list(payload.get("topics", []))
