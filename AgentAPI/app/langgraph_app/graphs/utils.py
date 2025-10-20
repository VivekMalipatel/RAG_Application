import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence, TypeVar

from pydantic import BaseModel, ValidationError


TDecision = TypeVar("TDecision", bound=BaseModel)


def flatten_content(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                if "text" in item:
                    text = flatten_content(item.get("text"))
                    if text:
                        parts.append(text)
                elif "content" in item:
                    text = flatten_content(item.get("content"))
                    if text:
                        parts.append(text)
            elif isinstance(item, str):
                text = item.strip()
                if text:
                    parts.append(text)
        return " ".join(parts)
    if value is None:
        return ""
    return str(value).strip()


def messages_to_transcript(messages: Sequence[Any], limit: int = 6) -> str:
    window = list(messages)[-limit:]
    lines: list[str] = []
    for message in window:
        role = getattr(message, "type", None)
        if not role:
            role = message.__class__.__name__
        content = getattr(message, "content", "")
        text = flatten_content(content)
        if not text:
            continue
        lines.append(f"{role.upper()}: {text}")
    return "\n".join(lines)


def normalize_queries(entries: Iterable[str], limit: int = 3) -> list[str]:
    normalized: list[str] = []
    for entry in entries:
        if not entry:
            continue
        text = entry.strip()
        if not text:
            continue
        normalized.append(text)
        if len(normalized) == limit:
            break
    return normalized


def build_partitioned_config(config: Mapping[str, Any], scope: str) -> dict[str, Any]:
    configurable = config.get("configurable", {})
    hashed_suffix = hashlib.sha256(scope.encode()).hexdigest()
    org_id = configurable.get("org_id", "")
    if "$" in org_id:
        org_token = f"{org_id}${hashed_suffix}"
    else:
        org_token = org_id
    thread_id = configurable.get("thread_id", "")
    user_id = configurable.get("user_id", "")
    thread_token = f"{thread_id}${hashed_suffix}" if thread_id else hashed_suffix
    user_token = f"{user_id}${hashed_suffix}" if user_id else hashed_suffix
    return {
        "configurable": {
            "thread_id": thread_token,
            "user_id": user_token,
            "org_id": org_token,
        }
    }


def parse_planner_output(value: str, model: type[TDecision]) -> TDecision:
    if not value:
        return model(action="respond", reason=None, queries=[])
    text = value.strip()
    if not text:
        return model(action="respond", reason=None, queries=[])
    start = text.find("{")
    end = text.rfind("}")
    candidate = text[start : end + 1] if start != -1 and end != -1 and end >= start else text
    parsed = None
    try:
        loaded = json.loads(candidate)
    except json.JSONDecodeError:
        loaded = None
    if isinstance(loaded, dict):
        normalized: dict[str, object] = {}
        for key, item in loaded.items():
            normalized[key.lower()] = item
        if "action" in normalized and isinstance(normalized["action"], str):
            normalized["action"] = normalized["action"].lower()
        if "queries" in normalized and isinstance(normalized["queries"], str):
            normalized["queries"] = [normalized["queries"]]
        if "queries" in normalized and normalized["queries"] is None:
            normalized["queries"] = []
        try:
            parsed = model.model_validate(normalized)
        except ValidationError:
            parsed = None
    if parsed is None:
        return model(action="respond", reason=None, queries=[])
    return parsed
