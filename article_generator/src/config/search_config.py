import os
import re
import toml
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Type, TypeVar, List


T = TypeVar("T", bound="SearchConfig")

TAVILY_KEYS_FILE = Path(os.environ.get("TWPGEN_TAVILY_KEY_FILE", "")) if os.environ.get("TWPGEN_TAVILY_KEY_FILE") else None


def _load_tavily_keys() -> List[str]:
    """从 TWPGEN_TAVILY_KEY_FILE 加载所有 API key"""
    if TAVILY_KEYS_FILE is None or not TAVILY_KEYS_FILE.exists():
        return []
    keys = []
    with open(TAVILY_KEYS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            key = line.strip()
            if key and not key.startswith('#'):
                keys.append(key)
    return keys


@dataclass(kw_only=True)
class SearchConfig:
    engine: str
    jina_api_key: str
    tavily_api_key: str
    tavily_api_keys: List[str]
    timeout: int = 30

    @classmethod
    def from_dict(cls: Type[T], config_dict: Dict[str, str]) -> T:
        required_fields = ["engine"]
        for field in required_fields:
            if field not in config_dict:
                raise ValueError(f"Configuration missing required field: {field}")

        timeout = config_dict.get("timeout", 30)
        try:
            timeout = int(timeout)
            if timeout < 1 or timeout > 300:
                raise ValueError("Timeout must be between 1 and 300 seconds")
        except (ValueError, TypeError):
            raise ValueError("Timeout must be a valid integer")

        # 优先从文件加载 key 池，fallback 到 toml 里配置的第一个 key
        file_keys = _load_tavily_keys()
        toml_key = config_dict.get("tavily_api_key", "").strip()
        env_keys = re.split(r"[\n,;]+", os.environ.get("TAVILY_API_KEYS", ""))
        env_keys = [k.strip() for k in env_keys if k.strip()]
        env_single = os.environ.get("TAVILY_API_KEY", "").strip()

        all_keys = env_keys + ([env_single] if env_single else []) + file_keys + ([toml_key] if toml_key else [])

        primary_key = all_keys[0] if all_keys else ""

        return cls(
            engine=config_dict["engine"],
            jina_api_key=os.environ.get("JINA_API_KEY", config_dict.get("jina_api_key", "")),
            tavily_api_key=primary_key,
            tavily_api_keys=all_keys,
            timeout=timeout,
        )


def load_search_config(config_path: Path = None) -> SearchConfig:
    config_path = config_path or Path(__file__).parent / "search.toml"

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    raw_config = toml.load(config_path)
    if not isinstance(raw_config, dict) or "search" not in raw_config:
        raise ValueError(
            "Invalid configuration file format. Expected [search] section."
        )

    return SearchConfig.from_dict(raw_config["search"])


search_config = load_search_config()


if __name__ == "__main__":
    try:
        config = load_search_config()
        print("Loaded search configuration:")
        print(f"Engine: {config.engine}")
        print(f"Jina API Key: {config.jina_api_key[:4]}...{config.jina_api_key[-4:]}")
        print(f"Tavily API Keys: {len(config.tavily_api_keys)} loaded")
        for k in config.tavily_api_keys:
            print(f"  {k[:12]}...")
        print(f"Timeout: {config.timeout}s")
    except Exception as e:
        print(f"Error loading configuration: {e}")
