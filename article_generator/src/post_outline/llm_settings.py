"""Adapter between this module and the repository-wide configuration.

All credentials, paths, the topic list and the verb-sense dictionary location are
defined once in ``<repo>/twpgen_settings.py`` (+ ``twpgen_config.json``). This
file only:

* imports that module (adding the repository root to ``sys.path``),
* exposes :func:`resolve_settings` / :func:`build_client` for the entry points,
* provides convenient accessors for the paths this module needs.

Priority for the model settings stays: explicit arguments → repository config
(``twpgen_settings.llm``) → this module's fallback defaults. So a single entry in
the repository ``.env`` or ``twpgen_config.json`` is enough for every stage.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:  # repository-wide configuration (single source of truth)
    import twpgen_settings as repo_config
except Exception:  # pragma: no cover - keeps the module usable stand-alone
    repo_config = None  # type: ignore[assignment]

DEFAULT_MODEL = repo_config.llm.model if repo_config else "qwen3-32b"
DEFAULT_BASE_URL = (
    repo_config.llm.base_url
    if repo_config
    else "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
DEFAULT_TIMEOUT = repo_config.llm.timeout if repo_config else 360.0
ENV_FILE_NAME = ".env"


def project_root() -> Path:
    """Repository root, derived from this file's location."""
    return _REPO_ROOT


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


def _article_llm_configs() -> dict[str, Any]:
    """Reuse article_generator's own llms.toml config when it is importable."""
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
    configs = _article_llm_configs()
    basic = configs.get("basic")
    repo_llm = repo_config.llm if repo_config else None

    resolved_key = _first(
        api_key,
        os.environ.get("ARTICLE_LLM_API_KEY"),
        os.environ.get("OPENAI_API_KEY"),
        getattr(repo_llm, "api_key", None),
        getattr(basic, "api_key", None),
    )
    resolved_base = _first(
        base_url,
        os.environ.get("ARTICLE_LLM_BASE_URL"),
        os.environ.get("OPENAI_BASE_URL"),
        os.environ.get("OPENAI_API_BASE"),
        getattr(repo_llm, "base_url", None),
        getattr(basic, "endpoint", None) if basic else None,
        DEFAULT_BASE_URL,
    )
    resolved_model = _first(
        model,
        os.environ.get("ARTICLE_LLM_MODEL"),
        os.environ.get("TWPGEN_LLM_MODEL"),
        getattr(repo_llm, "model", None),
        getattr(basic, "model", None) if basic else None,
        DEFAULT_MODEL,
    )
    resolved_timeout = timeout or getattr(repo_llm, "timeout", None) or DEFAULT_TIMEOUT
    if enable_thinking is None:
        raw = os.environ.get("ARTICLE_ENABLE_THINKING", "").strip().lower()
        if raw:
            enable_thinking = raw in {"1", "true", "yes", "on"}
        elif repo_llm is not None:
            enable_thinking = bool(repo_llm.enable_thinking)
        else:
            enable_thinking = False

    return LLMSettings(
        api_key=resolved_key,
        base_url=resolved_base,
        model=resolved_model,
        timeout=resolved_timeout,
        enable_thinking=bool(enable_thinking),
    )


# --------------------------------------------------------------------------- #
# paths / topics / resources — all defined in the repository config
# --------------------------------------------------------------------------- #
def default_outline_root() -> Path | None:
    return Path(repo_config.outline_dir) if repo_config else None


def repo_output_dir() -> str:
    """Configured output root (``output_dir`` in the repository config)."""
    return str(repo_config.output_dir) if repo_config else "output"


def default_source_root() -> Path | None:
    return Path(repo_config.article_source_dir) if repo_config else None


def default_topic_file() -> Path | None:
    return Path(repo_config.topic_file) if repo_config else None


def verb_sense_dict() -> Path | None:
    """Path of the verb-sense dictionary used by the outline stage."""
    return Path(repo_config.dict_file) if repo_config else None


def topics_from_config() -> list[str]:
    return list(repo_config.topics) if repo_config else []


def source_dir_aliases() -> dict[str, str]:
    """Topic → source directory alias mapping, from the repository config."""
    return dict(getattr(repo_config, "source_dir_aliases", {}) or {}) if repo_config else {}


def load_topics(path: Path | str | None = None) -> list[str]:
    if repo_config is not None:
        return repo_config.load_topics(path)
    if path:
        target = Path(path)
        if target.is_file():
            return [
                line.strip()
                for line in target.read_text(encoding="utf-8", errors="ignore").splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
    return []


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
