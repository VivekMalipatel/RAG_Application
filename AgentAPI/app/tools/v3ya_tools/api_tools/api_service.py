from pathlib import Path
from typing import Any, Dict, Optional, Type
from typing import Literal

import yaml
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from config import config
from ..utils.api import ConfigurationService, PartService, ProfileService, QuoteService


def _load_descriptions() -> Dict[str, str]:
    yaml_path = Path(__file__).parent / "description.yaml"
    if not yaml_path.exists():
        return {}
    with open(yaml_path, "r", encoding="utf-8") as yaml_file:
        return yaml.safe_load(yaml_file) or {}


_TOOL_DESCRIPTIONS = _load_descriptions()


class QuoteTableRequest(BaseModel):
    action: Literal["get_by_id", "list_by_user"] = Field(description="Action to perform on the quotes table")
    quote_id: Optional[str] = Field(default=None, description="Quote identifier required for the get_by_id action")
    user_id: Optional[str] = Field(default=None, description="User identifier required for the list_by_user action")
    page: int = Field(default=1, ge=1, description="Page number when listing quotes")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum records per page when listing quotes")


class PartTableRequest(BaseModel):
    action: Literal["get_by_id"] = Field(description="Action to perform on the parts table")
    part_id: Optional[str] = Field(default=None, description="Part identifier required for the get_by_id action")


class ConfigurationTableRequest(BaseModel):
    action: Literal[
        "create",
        "get_all",
        "get_by_id",
        "update",
        "delete",
        "update_by_part_id",
    ] = Field(description="Action to perform on the configurations table")
    config_id: Optional[str] = Field(default=None, description="Configuration identifier for get_by_id, update, or delete actions")
    part_id: Optional[str] = Field(default=None, description="Part identifier used for filtering or updating by part")
    technology_id: Optional[str] = Field(default=None, description="Technology identifier used for filtering configuration records")
    material_id: Optional[str] = Field(default=None, description="Material identifier used for filtering configuration records")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Payload for create or update actions")
    page: int = Field(default=1, ge=1, description="Page number when retrieving configuration lists")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum records per page when retrieving configuration lists")


class ProfileTableRequest(BaseModel):
    action: Literal["list_by_user"] = Field(description="Action to perform on the profiles table")
    user_id: Optional[str] = Field(default=None, description="User identifier required for the list_by_user action")


BASE_URL = config.V3YA_API_BASE_URL
if not BASE_URL.endswith("/"):
    BASE_URL = f"{BASE_URL}/"

def _build_service(service_cls):
    if config.V3YA_CLIENT_ID and config.V3YA_CLIENT_SECRET:
        return service_cls(BASE_URL, client_id=config.V3YA_CLIENT_ID, client_secret=config.V3YA_CLIENT_SECRET)
    return service_cls(BASE_URL)


part_service = _build_service(PartService)
quote_service = _build_service(QuoteService)
configuration_service = _build_service(ConfigurationService)
profile_service = _build_service(ProfileService)


def _normalize_response(data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return data if isinstance(data, dict) else {}


def _build_response(data: Optional[Dict[str, Any]], **metadata: Any) -> Dict[str, Any]:
    payload = dict(metadata)
    payload["data"] = _normalize_response(data)
    return payload


def _register_tool(name: str, args_schema: Type[BaseModel], default_description: str):
    description = _TOOL_DESCRIPTIONS.get(name, default_description)
    return tool(
        name_or_callable=name,
        description=description,
        args_schema=args_schema,
        parse_docstring=False,
        infer_schema=True,
    )


@_register_tool(
    "quotes_table_tool",
    QuoteTableRequest,
    "Interact with quotes table endpoints",
)
async def quotes_table_tool(
    action: str,
    quote_id: Optional[str] = None,
    user_id: Optional[str] = None,
    page: int = 1,
    limit: int = 10,
) -> Dict[str, Any]:
    if action == "get_by_id":
        if not quote_id:
            return _build_response(None, table="quotes", action=action, error="quote_id is required")
        data = await quote_service.get_by_id(quote_id)
        return _build_response(data, table="quotes", action=action, quote_id=quote_id)
    if action == "list_by_user":
        if not user_id:
            return _build_response(None, table="quotes", action=action, error="user_id is required")
        data = await quote_service.list_by_user(user_id, page=page, limit=limit)
        return _build_response(
            data,
            table="quotes",
            action=action,
            user_id=user_id,
            page=page,
            limit=limit,
        )
    return _build_response(None, table="quotes", action=action, error="Unsupported action")


@_register_tool(
    "parts_table_tool",
    PartTableRequest,
    "Interact with parts table endpoints",
)
async def parts_table_tool(action: str, part_id: Optional[str] = None) -> Dict[str, Any]:
    if action == "get_by_id":
        if not part_id:
            return _build_response(None, table="parts", action=action, error="part_id is required")
        data = await part_service.get_by_id(part_id)
        return _build_response(data, table="parts", action=action, part_id=part_id)
    return _build_response(None, table="parts", action=action, error="Unsupported action")


@_register_tool(
    "configurations_table_tool",
    ConfigurationTableRequest,
    "Interact with configurations table endpoints",
)
async def configurations_table_tool(
    action: str,
    config_id: Optional[str] = None,
    part_id: Optional[str] = None,
    technology_id: Optional[str] = None,
    material_id: Optional[str] = None,
    data: Optional[Dict[str, Any]] = None,
    page: int = 1,
    limit: int = 10,
) -> Dict[str, Any]:
    if action == "create":
        if not isinstance(data, dict):
            return _build_response(None, table="configurations", action=action, error="data payload is required")
        created = await configuration_service.create(data)
        return _build_response(created, table="configurations", action=action)
    if action == "get_all":
        records = await configuration_service.get_all(
            page=page,
            limit=limit,
            part_id=part_id,
            technology_id=technology_id,
            material_id=material_id,
        )
        return _build_response(
            records,
            table="configurations",
            action=action,
            part_id=part_id,
            technology_id=technology_id,
            material_id=material_id,
            page=page,
            limit=limit,
        )
    if action == "get_by_id":
        if not config_id:
            return _build_response(None, table="configurations", action=action, error="config_id is required")
        record = await configuration_service.get_by_id(config_id)
        return _build_response(record, table="configurations", action=action, config_id=config_id)
    if action == "update":
        if not config_id or not isinstance(data, dict):
            return _build_response(None, table="configurations", action=action, error="config_id and data are required")
        success = await configuration_service.update(config_id, data)
        return _build_response({"success": success}, table="configurations", action=action, config_id=config_id)
    if action == "delete":
        if not config_id:
            return _build_response(None, table="configurations", action=action, error="config_id is required")
        success = await configuration_service.delete(config_id)
        return _build_response({"success": success}, table="configurations", action=action, config_id=config_id)
    if action == "update_by_part_id":
        if not part_id or not isinstance(data, dict):
            return _build_response(None, table="configurations", action=action, error="part_id and data are required")
        success = await configuration_service.update_by_part_id(part_id, data)
        return _build_response({"success": success}, table="configurations", action=action, part_id=part_id)
    return _build_response(None, table="configurations", action=action, error="Unsupported action")


@_register_tool(
    "profiles_table_tool",
    ProfileTableRequest,
    "Interact with profiles table endpoints",
)
async def profiles_table_tool(action: str, user_id: Optional[str] = None) -> Dict[str, Any]:
    if action == "list_by_user":
        if not user_id:
            return _build_response(None, table="profiles", action=action, error="user_id is required")
        data = await profile_service.list_by_user(user_id)
        return _build_response(data, table="profiles", action=action, user_id=user_id)
    return _build_response(None, table="profiles", action=action, error="Unsupported action")


__all__ = [
    "quotes_table_tool",
    "parts_table_tool",
    "configurations_table_tool",
    "profiles_table_tool",
]

