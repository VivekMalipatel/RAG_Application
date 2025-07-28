
import yaml
from pathlib import Path
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage
from langgraph.config import get_stream_writer
from pydantic import BaseModel, Field
from agents.util_agents.mcp_agent.mcp_agent import MCPAgent

TOOL_NAME = "mcp_agent"

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
    writer = get_stream_writer()
    writer(f"MCP Agent invoked with '{prompt}' as instruction")

    # Instantiate the MCPAgent with the provided config
    agent = MCPAgent(config=config)
    compiled_agent = await agent.compile(name=TOOL_NAME)

    input_data = {
        "messages": [HumanMessage(content=prompt)],
        "user_id": config.get("configurable", {}).get("user_id"),
        "org_id": config.get("configurable", {}).get("org_id")
    }

    result = await compiled_agent.ainvoke(input_data, config=config)
    return {
        "messages": result.get("messages"),
        "user_id": input_data["user_id"],
        "org_id": input_data["org_id"]
    }
