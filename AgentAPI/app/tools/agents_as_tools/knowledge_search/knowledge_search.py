import yaml
from pathlib import Path
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from agents.base_agents.base_state import BaseState
from agents.base_agents.base_agent import BaseAgent
from agents.knowledge_search_agent.knowledge_search_agent import KnowledgeSearchAgent

TOOL_NAME = "knowledge_search_agent"

def get_tool_description(tool_name: str, yaml_filename: str = "description.yaml") -> str:
    yaml_path = Path(__file__).parent / yaml_filename
    with open(yaml_path, 'r') as f:
        descriptions = yaml.safe_load(f)
    return descriptions.get(tool_name, "")

@tool(
    name=TOOL_NAME,
    description=get_tool_description(TOOL_NAME),
    parse_docstring=False,
    infer_schema=True,
    response_format="content"
)
async def knowledge_search_agent(state : BaseState, config: RunnableConfig) :

    knowledge_search_agent = KnowledgeSearchAgent(
                                model_kwargs={},
                                vlm_kwargs={},
                                node_kwargs={},
                                debug=False
                            )
    
    compiled_agent: BaseAgent = await knowledge_search_agent.compile(name=TOOL_NAME)
