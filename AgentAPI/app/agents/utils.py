import asyncio
import json
import logging
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

import yaml
from typing import Any, Optional, Union, Sequence, Dict, Callable, Awaitable, get_args, get_origin
from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, model_validator

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

def coerce_profile_content(payload: Any) -> Optional[dict[str, Any]]:
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(exclude_none=True)
    elif hasattr(payload, "dict"):
        payload = payload.dict(exclude_none=True)
    elif isinstance(payload, dict):
        payload = {k: v for k, v in payload.items()}
    else:
        return None
    if isinstance(payload, dict) and "content" in payload:
        content = payload["content"]
        if hasattr(content, "model_dump"):
            return content.model_dump(exclude_none=True)
        if hasattr(content, "dict"):
            return content.dict(exclude_none=True)
        if isinstance(content, dict):
            return {k: v for k, v in content.items()}
        return None
    return payload if isinstance(payload, dict) else None

def coerce_procedural_content(payload: Any) -> Optional[dict[str, Any]]:
    content = coerce_profile_content(payload)
    if isinstance(content, dict):
        return {k: v for k, v in content.items()}
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
