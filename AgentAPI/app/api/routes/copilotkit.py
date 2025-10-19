import asyncio
from typing import Any, AsyncIterator, Literal, Sequence

from dotenv import load_dotenv
from fastapi import APIRouter, Request
from langchain_core.messages.base import messages_to_dict
from langchain_core.runnables import RunnableConfig
from langgraph.types import All
from copilotkit import CopilotKitRemoteEndpoint, LangGraphAgent
from copilotkit.integrations.fastapi import add_fastapi_endpoint

from agents.v3ya_agent.v3ya_agent import V3yaAgent

load_dotenv()

router = APIRouter()


def _extract_properties(context: object) -> dict[str, Any]:
    if isinstance(context, dict):
        properties = context.get("properties")
        if isinstance(properties, dict):
            return dict(properties)
    return {}


def _build_agent_config(properties: dict[str, Any]) -> dict[str, Any]:
    configurable: dict[str, Any] = {}

    user_id = properties.get("user_id") or "copilotkit-user"
    org_id = properties.get("org_id") or "copilotkit-org"
    thread_id = properties.get("thread_id") or f"{user_id}-thread"

    configurable["user_id"] = user_id
    configurable["org_id"] = org_id
    configurable["thread_id"] = thread_id

    checkpoint_id = properties.get("checkpoint_id")
    if checkpoint_id:
        configurable["checkpoint_id"] = checkpoint_id

    return {"configurable": configurable}


class V3yaGraphWrapper:
    def __init__(self, properties: dict[str, Any]):
        self._properties = properties
        self._agent: V3yaAgent | None = None
        self._graph = None
        self._lock = asyncio.Lock()
        self._base_config = _build_agent_config(self._properties)

    async def _ensure_graph(self):
        if self._graph is None:
            async with self._lock:
                if self._graph is None:
                    agent = V3yaAgent(config=self._base_config, properties=self._properties)
                    await agent.compile(name="copilotkit_v3ya_agent")
                    self._agent = agent
                    self._graph = agent.graph
        return self._graph

    async def stream(self, initial_state: dict[str, Any], stream_mode: str = "updates"):
        graph = await self._ensure_graph()
        config = self._agent.config if self._agent else None
        async for event in graph.stream(initial_state, config=config, stream_mode=stream_mode):
            yield event

    def astream_events(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        *,
        version: Literal["v1", "v2"] = "v2",
        include_names: Sequence[All] | None = None,
        include_types: Sequence[All] | None = None,
        include_tags: Sequence[All] | None = None,
        exclude_names: Sequence[All] | None = None,
        exclude_types: Sequence[All] | None = None,
        exclude_tags: Sequence[All] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        async def _runner():
            graph = await self._ensure_graph()
            async for event in graph.astream_events(
                input,
                config,
                version=version,
                include_names=include_names,
                include_types=include_types,
                include_tags=include_tags,
                exclude_names=exclude_names,
                exclude_types=exclude_types,
                exclude_tags=exclude_tags,
                **kwargs,
            ):
                yield event
        return _runner()

    async def aupdate_state(self, *args, **kwargs):
        graph = await self._ensure_graph()
        return await graph.aupdate_state(*args, **kwargs)

    async def aget_state(self, *args, **kwargs):
        graph = await self._ensure_graph()
        return await graph.aget_state(*args, **kwargs)

    def aget_state_history(
        self,
        config: RunnableConfig,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[Any]:
        async def _history_runner():
            graph = await self._ensure_graph()
            async for snapshot in graph.aget_state_history(
                config,
                filter=filter,
                before=before,
                limit=limit,
            ):
                yield snapshot
        return _history_runner()

    def get_input_jsonschema(self, *args, **kwargs):
        if self._graph is None:
            raise RuntimeError("Graph not initialized")
        return self._graph.get_input_jsonschema(*args, **kwargs)

    def get_output_jsonschema(self, *args, **kwargs):
        if self._graph is None:
            raise RuntimeError("Graph not initialized")
        return self._graph.get_output_jsonschema(*args, **kwargs)

    def config_schema(self, *args, **kwargs):
        if self._graph is None:
            raise RuntimeError("Graph not initialized")
        return self._graph.config_schema(*args, **kwargs)

    def context_schema(self, *args, **kwargs):
        if self._graph is None:
            raise RuntimeError("Graph not initialized")
        return self._graph.context_schema(*args, **kwargs)

    @property
    def nodes(self):
        if self._graph is None:
            return {}
        return self._graph.nodes

    @property
    def config(self):
        if self._graph is None:
            return self._base_config
        return self._graph.config

    async def aclose(self):
        if self._agent is not None:
            await self._agent.aclose()
            self._agent = None
            self._graph = None


def _serialize_event(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _serialize_event(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize_event(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_serialize_event(item) for item in value)
    if hasattr(value, "type") and hasattr(value, "content"):
        return messages_to_dict([value])[0]
    return value


def configure_agents(context: object) -> list[LangGraphAgent]:
    properties = _extract_properties(context)
    langgraph_config = {"configurable": {"properties": properties}}
    graph_wrapper = V3yaGraphWrapper(properties)
    return [
        LangGraphAgent(
            name="agent3ya",
            description="Agentic chat flow with dynamic tool use.",
            graph=graph_wrapper,
            langgraph_config=langgraph_config,
        )
    ]


sdk = CopilotKitRemoteEndpoint(agents=configure_agents)

add_fastapi_endpoint(router, sdk, "/copilotkit")
add_fastapi_endpoint(router, sdk, "/copilotkit/")


@router.get("/copilotkit/info")
async def copilotkit_info():
    return await sdk.handle_request(
        Request(
            scope={
                "type": "http",
                "method": "POST",
                "headers": [],
                "path": "/copilotkit/info",
                "query_string": b"",
            },
            receive=lambda: {"type": "http.request", "body": b"{}"},
        )
    )

@router.post("/copilotkit")
async def copilotkit_no_slash(request: Request):
    return await sdk.handle_request(request)


