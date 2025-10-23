import asyncio
import json
import logging
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import yaml
import tiktoken
from typing import Any, Optional, Union, Sequence, Dict, Callable, Awaitable, get_args, get_origin
from langchain_core.messages import AIMessage, BaseMessage, ChatMessage, FunctionMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, model_validator
from langmem.short_term import RunningSummary

def _load_prompt(key: str = None, filename: str = "prompt.yaml", base_dir: str = None) -> str:
    if base_dir is None:
        base_dir = Path(__file__).parent
    prompt_path = Path(base_dir) / filename
    with open(prompt_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        prompt_data = yaml.safe_load(content)
        if key and isinstance(prompt_data, dict) and key in prompt_data:
            return prompt_data[key]
        elif isinstance(prompt_data, str):
            return prompt_data
        return ""

def validate_tools(tools: Sequence[Union[Dict[str, Any], type, Callable, BaseTool]]) -> None:
    for tool in tools:
        if hasattr(tool, 'run') or hasattr(tool, 'arun') or isinstance(tool, BaseTool):
            continue
        elif callable(tool):
            continue
        elif isinstance(tool, dict):
            continue
        else:
            raise ValueError(f"Invalid tool: {tool}. Tools must be callable, BaseTool instances, or dictionaries.")

def build_memory_config(config: Optional[RunnableConfig]) -> dict[str, Any]:
    if not config:
        return {"configurable": {}}
    configurable = config.get("configurable", {})
    if configurable is None:
        configurable = {}
    return {"configurable": dict(configurable)}


def coerce_running_summary(value: Any) -> Optional[RunningSummary]:
    if isinstance(value, RunningSummary):
        return value
    if isinstance(value, dict):
        summary_text = value.get("summary")
        if not summary_text:
            return None
        summarized_ids = value.get("summarized_message_ids") or []
        if isinstance(summarized_ids, set):
            normalized_ids = {str(item) for item in summarized_ids}
        else:
            normalized_ids = {str(item) for item in summarized_ids}
        last_id = value.get("last_summarized_message_id")
        return RunningSummary(
            summary=summary_text,
            summarized_message_ids=normalized_ids,
            last_summarized_message_id=last_id,
        )
    return None

def coerce_profile_content(payload: Any) -> Optional[dict[str, Any]]:
    data = _coerce_mapping(payload)
    if not isinstance(data, dict):
        return None
    if "content" in data:
        content = _coerce_mapping(data["content"])
        if not isinstance(content, dict):
            return None
        data = content
    if "Address" in data:
        address = _normalize_profile_address(data.get("Address"))
        if address is None:
            data.pop("Address", None)
        else:
            data["Address"] = address
    return data

def coerce_procedural_content(payload: Any) -> Optional[dict[str, Any]]:
    content = coerce_profile_content(payload)
    if isinstance(content, dict):
        return {k: v for k, v in content.items()}
    return None


def _coerce_mapping(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if hasattr(value, "dict"):
        return value.dict(exclude_none=True)
    if isinstance(value, dict):
        return {k: v for k, v in value.items()}
    return value


def _normalize_profile_address(value: Any) -> Optional[dict[str, Any]]:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return {"primary": value.model_dump(exclude_none=True)}
    if hasattr(value, "dict"):
        return {"primary": value.dict(exclude_none=True)}
    if isinstance(value, dict):
        if not value:
            return None
        core_fields = {"Street", "City", "State", "ZipCode", "Country"}
        if any(key in core_fields for key in value.keys()):
            return {"primary": {k: v for k, v in value.items() if v is not None}}
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            coerced = _coerce_mapping(item)
            if isinstance(coerced, dict):
                filtered = {k: v for k, v in coerced.items() if v is not None}
                if filtered:
                    normalized[str(key)] = filtered
        return normalized or None
    return None

def format_profile_overview(data: Any) -> Optional[str]:
    if data is None:
        return None
    if hasattr(data, "model_dump"):
        data = data.model_dump(exclude_none=True)
    elif hasattr(data, "dict"):
        data = data.dict(exclude_none=True)
    elif isinstance(data, dict):
        data = {k: v for k, v in data.items() if v is not None}
    else:
        return None
    if not data:
        return None
    ordered_keys = [
        ("PreferredName", "Preferred Name"),
        ("FormalName", "Formal Name"),
        ("Pronouns", "Pronouns"),
        ("Locale", "Locale"),
        ("Timezone", "Timezone"),
        ("Greeting", "Greeting"),
        ("Summary", "Summary"),
        ("Email", "Email"),
        ("Phone", "Phone"),
        ("Address", "Address"),
        ("Preferences", "Preferences"),
        ("Communication", "Communication"),
        ("WorkContext", "Work"),
        ("Contacts", "Contacts"),
        ("Notes", "Notes"),
    ]
    lines = ["<User Details>"]
    for key, label in ordered_keys:
        if key not in data:
            continue
        value = data.get(key)
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            items = [str(item) for item in value if item is not None]
            rendered = ", ".join(item.strip() for item in items if item.strip())
        elif isinstance(value, dict):
            rendered = ", ".join(
                f"{k}: {v}" for k, v in value.items() if v is not None
            )
        else:
            rendered = str(value).strip()
        if rendered:
            lines.append(f"{label}: {rendered}")
    lines.append("</User Details>")
    return "\n".join(lines)

def format_procedural_overview(data: Any) -> Optional[str]:
    if data is None:
        return None
    if hasattr(data, "model_dump"):
        data = data.model_dump(exclude_none=True)
    elif hasattr(data, "dict"):
        data = data.dict(exclude_none=True)
    elif isinstance(data, dict):
        data = {k: v for k, v in data.items() if v is not None}
    else:
        return None
    if not data:
        return None
    ordered_keys = [
        ("CoreDirectives", "Core Directives"),
        ("ResponseGuidelines", "Response Guidelines"),
        ("ToolingGuidelines", "Tooling Guidelines"),
        ("EscalationPolicy", "Escalation Policy"),
    ]
    lines = ["<Procedural Instructions>"]
    for key, label in ordered_keys:
        if key not in data:
            continue
        value = data.get(key)
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            items = [str(item) for item in value if item is not None]
            rendered = ", ".join(item.strip() for item in items if item.strip())
        elif isinstance(value, dict):
            rendered = ", ".join(f"{k}: {v}" for k, v in value.items() if v is not None)
        else:
            rendered = str(value).strip()
        if rendered:
            lines.append(f"{label}: {rendered}")
    metadata = data.get("Metadata") if isinstance(data, dict) else None
    if isinstance(metadata, dict) and metadata:
        rendered = ", ".join(f"{k}: {v}" for k, v in metadata.items() if v is not None)
        if rendered:
            lines.append(f"Metadata: {rendered}")
    lines.append("</Procedural Instructions>")
    return "\n".join(lines)

def _extract_profile_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (int, float)):
        return [str(value)]
    if isinstance(value, (list, tuple, set)):
        items: list[str] = []
        for item in value:
            items.extend(_extract_profile_strings(item))
        return items
    if isinstance(value, dict):
        items: list[str] = []
        for item in value.values():
            items.extend(_extract_profile_strings(item))
        return items
    return []

PROFILE_PRECONTEXT_FIELDS: tuple[str, ...] = (
    "PreferredName",
    "FormalName",
    "Pronouns",
    "Locale",
    "Timezone",
    "Greeting",
    "Summary",
    "Communication",
    "Preferences",
    "WorkContext",
)

def _profile_value_is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value)
    if isinstance(value, (list, tuple, set)):
        return any(_profile_value_is_present(item) for item in value)
    if isinstance(value, dict):
        return any(_profile_value_is_present(item) for item in value.values())
    return True

def normalize_profile_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        value = value.model_dump(exclude_none=True)
    elif hasattr(value, "dict"):
        value = value.dict(exclude_none=True)
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, inner in value.items():
            normalized_inner = normalize_profile_value(inner)
            if _profile_value_is_present(normalized_inner):
                normalized[str(key)] = normalized_inner
        return normalized
    if isinstance(value, (list, tuple, set)):
        normalized_items = [normalize_profile_value(item) for item in value]
        return [item for item in normalized_items if _profile_value_is_present(item)]
    if isinstance(value, str):
        text = value.strip()
        return text if text else None
    return value

def prepare_profile_precontext_payload(content: Any) -> dict[str, Any]:
    if hasattr(content, "model_dump"):
        content = content.model_dump(exclude_none=True)
    elif hasattr(content, "dict"):
        content = content.dict(exclude_none=True)
    if not isinstance(content, dict):
        return {}
    payload: dict[str, Any] = {}
    for key in PROFILE_PRECONTEXT_FIELDS:
        if key not in content:
            continue
        normalized = normalize_profile_value(content[key])
        if not _profile_value_is_present(normalized):
            continue
        payload[key] = normalized
    return payload

def build_profile_directives(content: Any) -> Optional[str]:
    if not isinstance(content, dict):
        return None
    directives: list[str] = []
    name = content.get("PreferredName") or content.get("FormalName")
    if isinstance(name, str) and name.strip():
        directives.append(f"Address the user as {name.strip()}.")
    timezone = content.get("Timezone")
    if isinstance(timezone, str) and timezone.strip():
        directives.append(f"Consider the user's timezone: {timezone.strip()}.")
    preference_values: list[str] = []
    preference_values.extend(_extract_profile_strings(content.get("Communication")))
    preference_values.extend(_extract_profile_strings(content.get("Preferences")))
    unique_preferences: list[str] = []
    seen: set[str] = set()
    for value in preference_values:
        lowered = value.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        unique_preferences.append(value)
    if unique_preferences:
        directives.append("Apply these profile cues:")
        directives.extend(f"- {item}" for item in unique_preferences)
    if not directives:
        return None
    return "Use the user's profile preferences below when responding.\n" + "\n".join(directives)

def build_procedural_directives(content: Any) -> Optional[str]:
    if not isinstance(content, dict):
        return None
    sections: list[str] = []
    core = content.get("CoreDirectives")
    if isinstance(core, str) and core.strip():
        sections.append("Core directives:\n" + core.strip())
    response = content.get("ResponseGuidelines")
    if isinstance(response, str) and response.strip():
        sections.append("Response guidelines:\n" + response.strip())
    tooling = content.get("ToolingGuidelines")
    if isinstance(tooling, str) and tooling.strip():
        sections.append("Tool usage expectations:\n" + tooling.strip())
    escalation = content.get("EscalationPolicy")
    if isinstance(escalation, str) and escalation.strip():
        sections.append("Escalation policy:\n" + escalation.strip())
    if not sections:
        return None
    return "Apply these procedural rules when generating responses.\n" + "\n\n".join(sections)


def _coerce_manage_tool_content(payload: Any, *, tool_name: str) -> Any:
    if payload is None:
        return None
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(exclude_none=True)
    elif hasattr(payload, "dict"):
        payload = payload.dict(exclude_none=True)
    elif isinstance(payload, str):
        text = payload.strip()
        if not text:
            return {}
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{tool_name} content must be valid JSON") from exc
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(exclude_none=True)
    elif hasattr(payload, "dict"):
        payload = payload.dict(exclude_none=True)
    return payload


def wrap_manage_memory_tool_json(tool: BaseTool | None, *, tool_name: str) -> BaseTool | None:
    if not tool:
        return tool
    if hasattr(tool, "args_schema"):
        tool.args_schema = _wrap_manage_args_schema(tool.args_schema, tool_name=tool_name)
    original_func = getattr(tool, "func", None)
    original_coroutine = getattr(tool, "coroutine", None)

    def _prepare(call_args: tuple[Any, ...], call_kwargs: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
        args_list = list(call_args)
        if args_list:
            args_list[0] = _coerce_manage_tool_content(args_list[0], tool_name=tool_name)
        if "content" in call_kwargs:
            call_kwargs["content"] = _coerce_manage_tool_content(call_kwargs["content"], tool_name=tool_name)
        return tuple(args_list), call_kwargs

    if original_func:

        def _wrapped_func(*args: Any, **kwargs: Any) -> Any:
            prepared_args, prepared_kwargs = _prepare(args, kwargs)
            return original_func(*prepared_args, **prepared_kwargs)

        tool.func = _wrapped_func

    if original_coroutine:

        async def _wrapped_coroutine(*args: Any, **kwargs: Any) -> Any:
            prepared_args, prepared_kwargs = _prepare(args, kwargs)
            return await original_coroutine(*prepared_args, **prepared_kwargs)

        tool.coroutine = _wrapped_coroutine

    return tool


def _wrap_manage_args_schema(schema: Any, *, tool_name: str) -> Any:
    if not isinstance(schema, type) or not issubclass(schema, BaseModel):
        return schema
    if getattr(schema, "__manage_tool_wrapped__", False):
        return schema

    class ManageArgsSchema(schema):
        __manage_tool_wrapped__ = True
        __content_target__ = _extract_manage_content_target(schema)

        @model_validator(mode="before")
        @classmethod
        def _normalize_manage_content(cls, value: Any) -> Any:
            if isinstance(value, dict) and "content" in value:
                coerced = _coerce_manage_tool_content(value["content"], tool_name=tool_name)
                value["content"] = _coerce_to_target(coerced, cls.__content_target__)
            return value

    ManageArgsSchema.__name__ = f"{schema.__name__}ManagePatched"
    return ManageArgsSchema


def _extract_manage_content_target(schema: type[BaseModel]) -> Any:
    field = schema.model_fields.get("content") if hasattr(schema, "model_fields") else None
    if not field:
        return None
    annotation = getattr(field, "annotation", None)
    if annotation is None:
        return None
    origin = get_origin(annotation)
    options = get_args(annotation) if origin is Union else (annotation,)
    preferred = None
    for candidate in options:
        if candidate is None or candidate is type(None):
            continue
        if candidate in (str, bytes):
            return None
        if candidate is dict or get_origin(candidate) is dict:
            preferred = dict
            break
        if hasattr(candidate, "model_validate"):
            return candidate
    return preferred


def _coerce_to_target(value: Any, target: Any) -> Any:
    if target is None:
        return value
    if target is dict:
        return value if isinstance(value, dict) else value
    if hasattr(target, "model_validate"):
        if isinstance(value, target):
            return value
        if isinstance(value, dict):
            return target.model_validate(value)
    return value


PrecontextProvider = Callable[[Any, RunnableConfig], Awaitable[Any] | Any]


def register_precontext_provider(registry: dict[str, PrecontextProvider], name: str, provider: PrecontextProvider) -> None:
    registry[name] = provider


def unregister_precontext_provider(registry: dict[str, PrecontextProvider], name: str) -> None:
    registry.pop(name, None)


def _message_to_token_payload(message: BaseMessage) -> dict[str, Any]:
    if isinstance(message, HumanMessage):
        return {"role": "user", "content": message.content}
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": message.content}
    if isinstance(message, AIMessage):
        payload: dict[str, Any] = {"role": "assistant", "content": message.content}
        if message.tool_calls:
            payload["tool_calls"] = message.tool_calls
        if "function_call" in message.additional_kwargs:
            payload["function_call"] = message.additional_kwargs["function_call"]
        if "reasoning_content" in message.additional_kwargs:
            payload["reasoning_content"] = message.additional_kwargs["reasoning_content"]
        return payload
    if isinstance(message, ToolMessage):
        return {"role": "tool", "content": message.content, "tool_call_id": message.tool_call_id}
    if isinstance(message, FunctionMessage):
        return {"role": "function", "content": message.content, "name": message.name}
    if isinstance(message, ChatMessage):
        return {"role": message.role, "content": message.content}
    return {"role": "assistant", "content": getattr(message, "content", "")}


@lru_cache(maxsize=32)
def _encoding_for_model(name: str) -> tiktoken.Encoding:
    candidates: list[str] = []
    cleaned = (name or "").strip()
    if cleaned:
        candidates.append(cleaned)
        if "/" in cleaned:
            candidates.extend(part for part in cleaned.split("/") if part)
        if ":" in cleaned:
            candidates.extend(part for part in cleaned.split(":") if part)
    seen: set[str] = set()
    ordered: list[str] = []
    for candidate in candidates:
        key = candidate.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    for candidate in ordered:
        try:
            return tiktoken.encoding_for_model(candidate)
        except Exception:
            continue
    return tiktoken.get_encoding("cl100k_base")


def _count_with_encoding(messages: Sequence[dict[str, Any]], encoding: tiktoken.Encoding) -> int:
    def _encode_content(message: dict[str, Any]) -> int:
        content = message.get("content", "")
        if isinstance(content, str):
            return len(encoding.encode(content))
        else:
            serialized = json.dumps(content, separators=(",", ":"), ensure_ascii=False, sort_keys=True)
            return len(encoding.encode(serialized))
    
    if len(messages) < 10:
        return sum(_encode_content(msg) for msg in messages)
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        return sum(executor.map(_encode_content, messages))


def build_message_token_counter(model_name: Optional[str]) -> Callable[[Sequence[Any]], int]:
    resolved_name = model_name or ""
    encoding = _encoding_for_model(resolved_name)

    def _counter(messages: Sequence[Any]) -> int:
        payload: list[dict[str, Any]] = []
        for message in messages:
            if isinstance(message, RemoveMessage):
                continue
            if isinstance(message, BaseMessage):
                payload.append(_message_to_token_payload(message))
            elif isinstance(message, dict) and "role" in message:
                payload.append(message)
        if not payload:
            return 0
        return _count_with_encoding(payload, encoding)

    return _counter


def count_tokens(messages: Sequence[Any], counter: Optional[Callable[[Sequence[Any]], int]], *, logger: Optional[logging.Logger] = None) -> int:
    if counter is None:
        raise ValueError("Token counter is not initialized")
    return counter(messages)


async def build_system_precontext(
    prompt: str,
    providers: dict[str, PrecontextProvider],
    state: Any,
    config: Optional[RunnableConfig],
    *,
    fallback_config: RunnableConfig,
    logger: logging.Logger,
) -> list[str]:
    active_config = config or fallback_config
    context_parts: list[str] = [prompt.strip()] if prompt else []
    for name, provider in providers.items():
        try:
            result = provider(state, active_config)
            if asyncio.iscoroutine(result):
                result = await result
        except Exception:
            logger.exception("Pre-context provider '%s' failed", name)
            continue
        context_parts.extend(_normalize_precontext_result(result))
    return [part for part in context_parts if part]


def make_profile_precontext_provider(
    get_profile_context: Callable[[RunnableConfig], Awaitable[tuple[Optional[str], Optional[dict[str, Any]]]]],
    build_profile_directives_fn: Callable[[Any], Optional[str]],
    get_profile_memory_key: Callable[[], Optional[str]],
) -> PrecontextProvider:

    async def _provider(state: Any, config: RunnableConfig) -> Optional[list[str]]:
        overview_text, profile_content = await get_profile_context(config)
        if not overview_text and not profile_content:
            return None
        sections: list[str] = []
        if overview_text:
            sections.append(overview_text)
        profile_directives = build_profile_directives_fn(profile_content)
        if profile_directives:
            sections.append(profile_directives)
        memory_key = get_profile_memory_key()
        if memory_key:
            sections.append(
                "Use manage_profile_memory with action='update' and id='{}' when the user requests profile changes. If no existsing profile is found, do not create a new one. We will create a new one in the background.".format(memory_key)
            )
        return sections

    return _provider


def make_procedural_precontext_provider(
    get_procedural_context: Callable[[RunnableConfig], Awaitable[tuple[Optional[str], Optional[dict[str, Any]]]]],
    build_procedural_directives_fn: Callable[[Any], Optional[str]],
    get_procedural_memory_key: Callable[[], Optional[str]],
) -> PrecontextProvider:

    async def _provider(state: Any, config: RunnableConfig) -> Optional[list[str]]:
        overview_text, procedural_content = await get_procedural_context(config)
        if not overview_text and not procedural_content:
            return None
        sections: list[str] = []
        if overview_text:
            sections.append(overview_text)
        procedural_directives = build_procedural_directives_fn(procedural_content)
        if procedural_directives:
            sections.append(procedural_directives)
        memory_key = get_procedural_memory_key()
        if memory_key:
            sections.append(
                "Use manage_procedural_memory with action='update' and id='{}' when adjusting system directives. Do not create additional procedural records.".format(memory_key)
            )
        return sections

    return _provider


def count_words_in_messages(messages: Sequence[Any]) -> int:
    total = 0
    for message in messages:
        content = getattr(message, "content", message)
        total += count_words_in_content(content)
    return total


def count_words_in_content(content: Any) -> int:
    if content is None:
        return 0
    if isinstance(content, str):
        return len([part for part in content.split() if part])
    if isinstance(content, list):
        total = 0
        for item in content:
            total += count_words_in_content(item)
        return total
    if isinstance(content, dict):
        total = 0
        for value in content.values():
            total += count_words_in_content(value)
        return total
    if hasattr(content, "model_dump"):
        return count_words_in_content(content.model_dump())
    if hasattr(content, "dict"):
        return count_words_in_content(content.dict())
    return len([part for part in str(content).split() if part])

def make_utc_datetime_precontext_provider() -> PrecontextProvider:

    def _provider(state: Any, config: RunnableConfig) -> str:
        now = datetime.now(timezone.utc)
        return now.strftime("Current UTC date: %Y-%m-%d | Current UTC time: %H:%M:%S")

    return _provider


def _normalize_precontext_result(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, Iterable):
        items: list[str] = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                items.append(text)
        return items
    text = str(value).strip()
    return [text] if text else []
