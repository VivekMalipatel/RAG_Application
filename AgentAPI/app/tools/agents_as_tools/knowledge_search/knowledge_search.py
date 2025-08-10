import yaml
import hashlib
from pathlib import Path
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage
from langgraph.config import get_stream_writer
from typing import Optional, Dict, Any, List

from agents.base_agents.base_state import BaseState
from agents.base_agents.base_agent import BaseAgent
from pydantic import BaseModel, Field
from agents.util_agents.knowledge_search_agent.knowledge_search_agent import KnowledgeSearchAgent

TOOL_NAME = "knowledge_search_agent"

class KnowledgeSearchRequest(BaseModel):
    prompt: str = Field(
        description="Clear task description (with full details and context) as a prompt to the knowledge search agent."
    )

def get_tool_description(tool_name: str, yaml_filename: str = "description.yaml") -> str:
    yaml_path = Path(__file__).parent / yaml_filename
    with open(yaml_path, 'r') as f:
        descriptions = yaml.safe_load(f)
    return descriptions.get(tool_name, "")

@tool(
    name_or_callable=TOOL_NAME,
    description=get_tool_description(TOOL_NAME),
    args_schema=KnowledgeSearchRequest,
    parse_docstring=False,
    infer_schema=True,
    response_format="content"
)
async def knowledge_search_agent(prompt : str, config: RunnableConfig) -> List[Dict[str, Any]] :

    org_id : str = config.get("configurable").get("org_id")

    if len(org_id.split("$")) > 1:
        org_id = f"{config.get('configurable').get('org_id')}${hashlib.sha256(TOOL_NAME.encode()).hexdigest()}"

    config = {
        "configurable": {
            "thread_id": f"{config.get('configurable').get('thread_id')}${hashlib.sha256(TOOL_NAME.encode()).hexdigest()}",
            "user_id": f"{config.get('configurable').get('user_id')}${hashlib.sha256(TOOL_NAME.encode()).hexdigest()}",
            "org_id": org_id
        }
    }

    writer = get_stream_writer()
    writer(f"Knowledge Search Agent invoked with '{prompt}' as instruction\n\n")
    knowledge_search_agent = KnowledgeSearchAgent(
                                config=config,
                                model_kwargs={},
                                vlm_kwargs={},
                                node_kwargs={},
                                debug=False
                            )
    
    compiled_agent: BaseAgent = await knowledge_search_agent.compile(name=TOOL_NAME)

    input={
            "messages": [
                HumanMessage(content=prompt)
            ],
            "user_id": config["configurable"]["user_id"],
            "org_id": config["configurable"]["org_id"]
        }

    result = await compiled_agent.ainvoke(input, config=config)
    return [{"type": "text", "text": result["messages"][-1].content}]
