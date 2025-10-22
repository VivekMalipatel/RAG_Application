import os
import yaml
import hashlib
import uuid
from pathlib import Path
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage
from langgraph.config import get_stream_writer

from pydantic import BaseModel, Field
from agents.base_agents.base_agent import BaseAgent
from agents.util_agents.web_agent.web_agent import WebAgent

TOOL_NAME = "web_search_scrape_agent"
STREAMING_ENABLED = os.getenv("ENABLE_WEB_SEARCH_STREAM_UPDATES", "0") not in {"", "0", "false", "False"}

class WebSearchScrapeRequest(BaseModel):
    prompt: str = Field(
        description="Clear task description (with full details and context) as a prompt to the web search scrape agent."
    )

def get_tool_description(tool_name: str, yaml_filename: str = "description.yaml") -> str:
    yaml_path = Path(__file__).parent / yaml_filename
    with open(yaml_path, 'r') as f:
        descriptions = yaml.safe_load(f)
    return descriptions.get(tool_name, "")



@tool(
    name_or_callable=TOOL_NAME,
    description=get_tool_description(TOOL_NAME),
    args_schema=WebSearchScrapeRequest,
    parse_docstring=False,
    infer_schema=True,
    response_format="content"
)
async def web_search_scrape_agent(prompt : str, config: RunnableConfig) :
    def emit_message(msg: str):
        if not STREAMING_ENABLED:
            return
        try:
            writer = get_stream_writer()
            writer(msg)
        except (RuntimeError, Exception):
            pass
    
    emit_message(f"🔍 Starting comprehensive web research: '{prompt}'")
    emit_message(f"📋 Phase 1: Discovering relevant sources...")

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

    web_search_scrape_agent = WebAgent(
        config=modified_config,
        model_kwargs={},
        vlm_kwargs={},
        node_kwargs={},
        debug=False
    )
    emit_message(f"WebAgent instance created and configured.")

    compiled_agent: BaseAgent = await web_search_scrape_agent.compile(name=TOOL_NAME)
    emit_message(f"WebAgent compiled for tool: {TOOL_NAME}")

    input={
        "messages": [
            HumanMessage(content=prompt)
        ],
        "user_id": config["configurable"]["user_id"],
        "org_id": config["configurable"]["org_id"]
    }
    emit_message(f"🎯 Phase 2: Extracting detailed information...")
    result = await compiled_agent.ainvoke(input, config=config)
    emit_message(f"✅ Research complete - returning comprehensive results")
    return {"messages": result["messages"], "user_id": config["configurable"]["user_id"], "org_id": config["configurable"]["org_id"]}
