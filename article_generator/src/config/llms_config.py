import os
import toml
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Type, TypeVar, Literal, Optional




def _load_env_file(path: Path):
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("' ")
        if key and key not in os.environ:
            os.environ[key] = value


_PROJECT_ROOT = Path(os.environ.get("TWPGEN_ROOT", Path(__file__).resolve().parents[3]))
_load_env_file(_PROJECT_ROOT / ".env")


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

T = TypeVar('T', bound='BaseLLMConfig')


@dataclass(kw_only=True)
class BaseLLMConfig:
    """Base LLM configuration shared by all model roles."""
    base_url: Optional[str]
    api_base: Optional[str]
    model: str
    api_key: str
    max_tokens: int = 32769
    context_window: int = 32769
    temperature: float = 0.6
    enable_thinking: bool = False

    @classmethod
    def from_dict(cls: Type[T], config_dict: Dict[str, str]) -> T:
        try:
            return cls(
                base_url=os.environ.get('ARTICLE_LLM_BASE_URL') or os.environ.get('OPENAI_BASE_URL') or config_dict.get('base_url'),
                api_base=os.environ.get('ARTICLE_LLM_BASE_URL') or os.environ.get('OPENAI_BASE_URL') or os.environ.get('OPENAI_API_BASE') or config_dict.get('api_base'),
                model=os.environ.get('ARTICLE_LLM_MODEL') or os.environ.get('TWPGEN_LLM_MODEL') or config_dict['model'],
                api_key=os.environ.get('ARTICLE_LLM_API_KEY') or os.environ.get('OPENAI_API_KEY') or config_dict.get('api_key', ''),
                max_tokens=int(config_dict.get('max_tokens', 32769)),
                context_window=int(config_dict.get('context_window', 32769)),
                temperature=float(config_dict.get('temperature', 0.6)),
                enable_thinking=_env_bool('ARTICLE_ENABLE_THINKING', bool(config_dict.get('enable_thinking', False))),
            )
        except KeyError as e:
            raise ValueError(f"Configuration missing required field: {e}") from e

    @property
    def endpoint(self) -> str:
        return self.base_url or self.api_base


def load_llm_configs(config_path: Path = None) -> Dict[str, BaseLLMConfig]:
    config_path = config_path or Path(__file__).parent / "llms.toml"

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    raw_config = toml.load(config_path)
    if not isinstance(raw_config, dict):
        raise ValueError("Invalid configuration file format. Expected a dictionary structure.")

    configs = {}
    for config_name, config_data in raw_config.items():
        configs[config_name] = BaseLLMConfig.from_dict(config_data)

    return configs


llm_configs = load_llm_configs()

LLMType = Literal["basic", "clarify", "planner", "query_generation", "evaluate", "report"]

basic_llm = llm_configs['basic']
clarify_llm = llm_configs['clarify']
planner_llm = llm_configs['planner']
query_generation_llm = llm_configs['query_generation']
evaluate_llm = llm_configs['evaluate']
report_llm = llm_configs['report']
