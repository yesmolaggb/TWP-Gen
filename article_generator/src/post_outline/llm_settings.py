"""Single source of truth for model / API settings in this module.

Everything that talks to an LLM (generation, editorial pass, citation rebuild)
resolves its credentials through :func:`resolve_settings` and creates its client
through :func:`build_client`, so a key only ever has to be configured once.

Resolution order (first hit wins)
---------------------------------
1. explicit arguments (``--model`` / ``--base-url`` / ``--api-key``)
2. environment variables, including the values loaded from the project ``.env``

   * key:      ``ARTICLE_LLM_API_KEY`` → ``OPENAI_API_KEY``
   * endpoint: ``ARTICLE_LLM_BASE_URL`` → ``OPENAI_BASE_URL`` → ``OPENAI_API_BASE``
   * model:    ``ARTICLE_LLM_MODEL`` → ``TWPGEN_LLM_MODEL``

3. the repository's own config module ``article_generator/src/config/llms_config.py``
   (which reads ``article_generator/src/config/llms.toml``)
4. built-in defaults (see ``DEFAULT_*`` below)

The ``.env`` file is looked up at ``<repo root>/.env`` unless a path is given
explicitly (``--env-file``).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "qwen3-32b"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_TIMEOUT = 360.0
ENV_FILE_NAME = ".env"


def project_root() -> Path:
    """Repository root, derived from this file's location."""
    return Path(__file__).resolve().parents[3]


def load_env(env_file: Path | str | None = None) -> Path | None:
    """Load ``KEY=VALUE`` pairs into ``os.environ`` (already-set vars win).

    Returns the file that was used, or ``None`` when nothing was found.
    """
    candidates: list[Path] = []
    if env_file:
        candidates.append(Path(env_file))
    candidates.append(project_root() / ENV_FILE_NAME)
    for path in candidates:
        if not path.is_file():
            continue
        for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            name = name.strip()
            value = value.strip().strip('"').strip("'")
            if name and name not in os.environ:
                os.environ[name] = value
        return path
    return None


def _repo_llm_configs() -> dict[str, Any]:
    """Reuse the repository's LLM config module when it is importable."""
    try:
        article_root = project_root() / "article_generator"
        if str(article_root) not in sys.path:
            sys.path.insert(0, str(article_root))
        from src.config.llms_config import llm_configs  # type: ignore

        return dict(llm_configs)
    except Exception:
        return {}


@dataclass(frozen=True)
class LLMSettings:
    """Resolved connection settings for every LLM call in this module."""

    api_key: str
    base_url: str
    model: str = DEFAULT_MODEL
    timeout: float = DEFAULT_TIMEOUT
    enable_thinking: bool = True

    def ensure_api_key(self) -> None:
        if not self.api_key:
            raise RuntimeError(
                "No API key configured. Put OPENAI_API_KEY (or ARTICLE_LLM_API_KEY) "
                f"into {project_root() / ENV_FILE_NAME}, or pass --env-file/--api-key."
            )


def _first(*values: str | None) -> str:
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
    enable_thinking: bool | None = None,
    env_file: Path | str | None = None,
) -> LLMSettings:
    """Resolve settings once; every entry point calls this."""
    load_env(env_file)
    configs = _repo_llm_configs()
    basic = configs.get("basic")

    resolved_key = _first(
        api_key,
        os.environ.get("ARTICLE_LLM_API_KEY"),
        os.environ.get("OPENAI_API_KEY"),
        getattr(basic, "api_key", None),
    )
    resolved_base = _first(
        base_url,
        os.environ.get("ARTICLE_LLM_BASE_URL"),
        os.environ.get("OPENAI_BASE_URL"),
        os.environ.get("OPENAI_API_BASE"),
        getattr(basic, "endpoint", None) if basic else None,
        DEFAULT_BASE_URL,
    )
    resolved_model = _first(
        model,
        os.environ.get("ARTICLE_LLM_MODEL"),
        os.environ.get("TWPGEN_LLM_MODEL"),
        getattr(basic, "model", None) if basic else None,
        DEFAULT_MODEL,
    )
    resolved_timeout = timeout or DEFAULT_TIMEOUT
    if enable_thinking is None:
        raw = os.environ.get("ARTICLE_ENABLE_THINKING", "").strip().lower()
        enable_thinking = raw in {"1", "true", "yes", "on"}

    return LLMSettings(
        api_key=resolved_key,
        base_url=resolved_base,
        model=resolved_model,
        timeout=resolved_timeout,
        enable_thinking=bool(enable_thinking),
    )


_CLIENTS: dict[tuple[str, str, float], Any] = {}


def build_client(settings: LLMSettings):
    """Create (and cache) an OpenAI-compatible client for the given settings."""
    settings.ensure_api_key()
    cache_key = (settings.base_url, settings.api_key, settings.timeout)
    client = _CLIENTS.get(cache_key)
    if client is None:
        from openai import OpenAI

        client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=settings.timeout,
        )
        _CLIENTS[cache_key] = client
    return client
