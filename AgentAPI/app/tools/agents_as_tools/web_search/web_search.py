import yaml
import hashlib
from pathlib import Path
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from agents.base_agents.base_agent import BaseAgent
from agents.util_agents.web_search_agent.web_search_agent import WebAgent

TOOL_NAME = "web_search_scrape_agent"

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

    web_search_scrape_agent = WebAgent(
                                model_kwargs={},
                                vlm_kwargs={},
                                node_kwargs={},
                                debug=False
                            )

    compiled_agent: BaseAgent = await web_search_scrape_agent.compile(name=TOOL_NAME)

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

    input={
            "messages": [
                HumanMessage(content=prompt)
            ],
            "user_id": config["configurable"]["user_id"],
            "org_id": config["configurable"]["org_id"]
        }

    result = await compiled_agent.ainvoke(input, config=config)
    return {"messages": result["messages"], "user_id": config["configurable"]["user_id"], "org_id": config["configurable"]["org_id"]}
