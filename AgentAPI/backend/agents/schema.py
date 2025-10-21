from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CapabilityToggle(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    name: str
    label: str
    description: str | None = None
    enabled_by_default: bool = False


class AgentDefinition(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    agent_id: str
    display_name: str
    description: str | None = None
    category: str | None = None
    capabilities: list[CapabilityToggle] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)