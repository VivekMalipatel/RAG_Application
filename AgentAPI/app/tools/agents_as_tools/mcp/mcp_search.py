import os
import yaml
import uuid
from pathlib import Path
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage
from langgraph.config import get_stream_writer
from pydantic import BaseModel, Field
from agents.util_agents.mcp_agent.mcp_agent import MCPAgent

TOOL_NAME = "mcp_agent"
STREAMING_ENABLED = os.getenv("ENABLE_MCP_STREAM_UPDATES", "0") not in {"", "0", "false", "False"}

class MCPRequest(BaseModel):
    prompt: str = Field(
        description="Clear task description (with full details and context) as a prompt to the MCP agent."
    )

def get_tool_description(tool_name: str, yaml_filename: str = "description.yaml") -> str:
    yaml_path = Path(__file__).parent / yaml_filename
    with open(yaml_path, 'r') as f:
        descriptions = yaml.safe_load(f)
    return descriptions.get(tool_name, "")

@tool(
    name_or_callable=TOOL_NAME,
    description=get_tool_description(TOOL_NAME),
    args_schema=MCPRequest,
    parse_docstring=False,
    infer_schema=True,
    response_format="content"
)
async def mcp_agent(prompt: str, config: RunnableConfig):
    def emit_message(msg: str):
        if not STREAMING_ENABLED:
            return
        try:
            writer = get_stream_writer()
            writer(msg)
        except (RuntimeError, Exception):
            pass
    
    emit_message(f"MCP Agent invoked with '{prompt}' as instruction")

    parent_thread_id = config.get("configurable").get("thread_id")
    parent_user_id = config.get("configurable").get("user_id")
    parent_org_id = config.get("configurable").get("org_id")
    
    subgraph_thread_id = str(uuid.uuid5(uuid.UUID(parent_thread_id), TOOL_NAME))

    modified_config = dict(config)
    if "configurable" not in modified_config:
        modified_config["configurable"] = {}
    else:
        modified_config["configurable"] = dict(modified_config["configurable"])
    
    modified_config["configurable"]["thread_id"] = subgraph_thread_id
    modified_config["configurable"]["user_id"] = parent_user_id
    modified_config["configurable"]["org_id"] = parent_org_id

    agent = MCPAgent(config=modified_config)
    compiled_agent = await agent.compile(name=TOOL_NAME)

    input_data = {
        "messages": [HumanMessage(content=prompt)],
        "user_id": parent_user_id,
        "org_id": parent_org_id
    }

    result = await compiled_agent.ainvoke(input_data, config=modified_config)
    return {
        "messages": result.get("messages"),
        "user_id": input_data["user_id"],
        "org_id": input_data["org_id"]
    }
