from typing import Iterable

from .schema import AgentDefinition, CapabilityToggle

_ICON_NAME_CHOICES: set[str] = {
    "globe",
    "web",
    "knowledge",
    "research",
    "academic",
    "social",
    "finance",
    "analysis",
    "workflow",
    "code",
    "data",
    "search",
    "user",
    "tasks",
    "launch",
    "security",
    "experiment",
    "default",
}

AGENT_CATALOG: list[AgentDefinition] = [
    AgentDefinition(
        agent_id="chat_agent",
        display_name="Chat Agent",
        description="General purpose assistant with optional knowledge retrieval.",
        category="primary",
        capabilities=[
            CapabilityToggle(
                name="knowledge_search",
                label="Knowledge Search",
                description="Run knowledge graph search before responding.",
                enabled_by_default=False,
            )
        ],
        metadata={
            "graph": "chat",
            "util_agents": ["knowledge_search"],
            "icon": "social",
            "capability_icons": {"knowledge_search": "knowledge"},
        },
    )
]


def get_agent_catalog() -> list[AgentDefinition]:
    return AGENT_CATALOG


def _validate_catalog(catalog: Iterable[AgentDefinition]) -> None:
    for agent in catalog:
        icon = agent.metadata.get("icon") if agent.metadata else None
        if not isinstance(icon, str) or not icon.strip():
            raise ValueError(f"Agent '{agent.agent_id}' must define a string icon metadata value.")
        if icon not in _ICON_NAME_CHOICES:
            raise ValueError(
                f"Agent '{agent.agent_id}' uses icon '{icon}', which is not in the approved icon list."
            )

        if not agent.capabilities:
            continue

        capability_icons = agent.metadata.get("capability_icons") if agent.metadata else None
        if not isinstance(capability_icons, dict):
            raise ValueError(
                f"Agent '{agent.agent_id}' capabilities require a capability_icons metadata dict."
            )
        for capability in agent.capabilities:
            icon_name = capability_icons.get(capability.name)
            if not isinstance(icon_name, str) or not icon_name.strip():
                raise ValueError(
                    f"Capability '{capability.name}' for agent '{agent.agent_id}' must define an icon override."
                )
            if icon_name not in _ICON_NAME_CHOICES:
                raise ValueError(
                    f"Capability '{capability.name}' for agent '{agent.agent_id}' uses icon '{icon_name}', which is not supported."
                )


_validate_catalog(AGENT_CATALOG)
