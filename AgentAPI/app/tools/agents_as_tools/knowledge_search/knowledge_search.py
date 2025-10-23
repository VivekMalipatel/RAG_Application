import yaml
from pathlib import Path
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage
from typing import Optional, Dict, Any, List

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

    incoming_config: Dict[str, Any] = dict(config or {})
    configurable: Dict[str, Any] = dict(incoming_config.get("configurable") or {})
    incoming_config["configurable"] = configurable

    user_id = configurable.get("user_id")
    org_id = configurable.get("org_id")

    if not user_id or not org_id:
        raise ValueError("Knowledge search agent requires user_id and org_id in configurable config")

    knowledge_search_agent = KnowledgeSearchAgent(
                                config=incoming_config,
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
            "user_id": user_id,
            "org_id": org_id
        }

    result = await compiled_agent.ainvoke(input, config=incoming_config)
    return [{"type": "text", "text": result["messages"][-1].content}]
