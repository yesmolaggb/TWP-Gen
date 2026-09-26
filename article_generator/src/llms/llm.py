import time
from typing import Any, Callable, Dict, Generator, List, Union

from openai import OpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

from src.config.llms_config import LLMType, llm_configs

_client_cache: Dict[LLMType, OpenAI] = {}
RESERVED_OUTPUT_MARGIN = 256
MAX_RETRIES = 3


def _message_role(message: Union[HumanMessage, AIMessage, SystemMessage, dict]) -> str:
    if isinstance(message, dict):
        return message.get("role", "user")

    message_type = getattr(message, "type", "")
    if message_type == "human":
        return "user"
    if message_type == "ai":
        return "assistant"
    if message_type == "system":
        return "system"
    return "user"


def _message_content(message: Union[HumanMessage, AIMessage, SystemMessage, dict]) -> str:
    if isinstance(message, dict):
        return str(message.get("content", ""))
    return str(getattr(message, "content", ""))


def _to_openai_messages(messages: List[Union[HumanMessage, AIMessage, SystemMessage, dict]]) -> List[dict]:
    return [{"role": _message_role(message), "content": _message_content(message)} for message in messages]


def _estimate_prompt_tokens(messages: List[Union[HumanMessage, AIMessage, SystemMessage, dict]]) -> int:
    # Dependency-free conservative estimate. It intentionally overestimates ASCII-heavy prompts.
    return sum(len(_message_content(message)) + 4 for message in messages)


def _completion_max_tokens(llm_type: LLMType, messages: List[Union[HumanMessage, AIMessage, SystemMessage, dict]]) -> int:
    llm_config = llm_configs[llm_type]
    remaining = llm_config.context_window - _estimate_prompt_tokens(messages) - RESERVED_OUTPUT_MARGIN
    return max(1, min(llm_config.max_tokens, remaining))


def _get_field(obj: Any, name: str, default: str = "") -> str:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default) or default
    value = getattr(obj, name, None)
    if value is not None:
        return value
    model_extra = getattr(obj, "model_extra", None)
    if isinstance(model_extra, dict):
        return model_extra.get(name, default) or default
    return default


def _get_client(llm_type: LLMType) -> OpenAI:
    if llm_type in _client_cache:
        return _client_cache[llm_type]

    try:
        llm_config = llm_configs[llm_type]
    except KeyError as e:
        raise KeyError(f"LLM configuration for '{llm_type}' not found") from e

    # Use the configured OpenAI-compatible endpoint directly.
    client = OpenAI(base_url=llm_config.endpoint, api_key=llm_config.api_key)
    _client_cache[llm_type] = client
    return client


def _extra_body(llm_type: LLMType) -> dict:
    return {
        "chat_template_kwargs": {
            "enable_thinking": bool(llm_configs[llm_type].enable_thinking)
        }
    }


def _with_retries(call: Callable[[], Any], llm_type: LLMType, max_tokens: int) -> Any:
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return call()
        except Exception as e:
            last_error = e
            print(
                f"call llm api error ({llm_type}, attempt {attempt}/{MAX_RETRIES}, "
                f"max_tokens={max_tokens}): {type(e).__name__}: {e}"
            )
            if attempt < MAX_RETRIES:
                time.sleep(2 * attempt)
    raise last_error


def llm(
    llm_type: LLMType,
    messages: List[Union[HumanMessage, AIMessage, SystemMessage, dict]],
    stream: bool = False,
) -> Union[Generator[str, None, None], str]:
    if stream:
        return _stream_llm_response(llm_type, messages)
    return _non_stream_llm_response(llm_type, messages)


def _stream_llm_response(
    llm_type: LLMType, messages: List[Union[HumanMessage, AIMessage, SystemMessage, dict]]
) -> Generator[str, None, None]:
    max_tokens = _completion_max_tokens(llm_type, messages)
    try:
        llm_config = llm_configs[llm_type]
        stream = _with_retries(
            lambda: _get_client(llm_type).chat.completions.create(
                model=llm_config.model,
                messages=_to_openai_messages(messages),
                temperature=llm_config.temperature,
                max_tokens=max_tokens,
                stream=True,
                extra_body=_extra_body(llm_type),
            ),
            llm_type,
            max_tokens,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            reasoning_content = _get_field(delta, "reasoning_content")
            content = _get_field(delta, "content")
            if reasoning_content or content:
                yield reasoning_content, content
    except Exception as e:
        print(f"call llm api failed ({llm_type}, max_tokens={max_tokens}): {type(e).__name__}: {e}")


def _non_stream_llm_response(
    llm_type: LLMType, messages: List[Union[HumanMessage, AIMessage, SystemMessage, dict]]
) -> str:
    max_tokens = _completion_max_tokens(llm_type, messages)
    try:
        llm_config = llm_configs[llm_type]
        response = _with_retries(
            lambda: _get_client(llm_type).chat.completions.create(
                model=llm_config.model,
                messages=_to_openai_messages(messages),
                temperature=llm_config.temperature,
                max_tokens=max_tokens,
                extra_body=_extra_body(llm_type),
            ),
            llm_type,
            max_tokens,
        )
    except Exception as e:
        print(f"call llm api failed ({llm_type}, max_tokens={max_tokens}): {type(e).__name__}: {e}")
        return ""

    msg = response.choices[0].message
    reasoning_content = _get_field(msg, "reasoning_content")
    content = _get_field(msg, "content")
    return (
        f"<thinking>{reasoning_content}</thinking>\n{content}"
        if reasoning_content
        else f"{content}"
    )


if __name__ == "__main__":
    print(llm("basic", [{"role": "user", "content": "?????????????"}], stream=False))
