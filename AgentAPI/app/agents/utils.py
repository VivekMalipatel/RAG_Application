from pathlib import Path
import json
import yaml
from typing import Any, Optional, Union, Sequence, Dict, Callable, get_args, get_origin
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
