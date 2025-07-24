from pathlib import Path
import yaml
from typing import Optional, Dict, Any
from langchain_core.runnables import RunnableConfig
from agents.base_agents.base_agent import BaseAgent
from tools.core_tools.web_search.web_search_tool import web_search_tool
from tools.core_tools.web_scrape.web_scrape_tool import web_scrape_tool

class WebAgent(BaseAgent):
    def __init__(self,
                 prompt: Optional[str] = None,
                 *,
                 config: RunnableConfig,
                 model_kwargs: Optional[Dict[str, Any]] = None,
                 vlm_kwargs: Optional[Dict[str, Any]] = None,
                 node_kwargs: Optional[Dict[str, Any]] = None,
                 recursion_limit: Optional[int] = 25,
                 debug: bool = False):
        if prompt is None:
            prompt = self._load_prompt()
        super().__init__(
            prompt=prompt,
            config=config,
            model_kwargs=model_kwargs,
            vlm_kwargs=vlm_kwargs,
            node_kwargs=node_kwargs,
            recursion_limit=recursion_limit,
            debug=debug
        )
        
        self.bind_tools([web_search_tool, web_scrape_tool])

    def _load_prompt(self) -> str:
        prompt_path = Path(__file__).parent / "prompt.yaml"
        with open(prompt_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            prompt_data = yaml.safe_load(content)
            if isinstance(prompt_data, dict) and 'WebSearchAgent' in prompt_data:
                return prompt_data['WebSearchAgent']
            elif isinstance(prompt_data, str):
                return prompt_data

if __name__ == "__main__":
    import asyncio
    from langchain_core.messages import HumanMessage

    async def test_web_agent():
        config = {
            "configurable": {
                "thread_id": "32",
                "user_id": "32",
                "org_id": "32"
            }
        }
        agent = WebAgent(
            config=config,
            model_kwargs={},
            vlm_kwargs={},
            node_kwargs={},
            debug=True
        )
        compiled_agent = await agent.compile(name="WebAgent")
        
        print("=== Testing Web Search & Scrape Agent ===")

        # Test web search
        search_input = {
            "type": "search",
            "messages": [HumanMessage(content="Bytecrafts services site:bytecrafts.in")],
            "num_results": 10,
            "language": "en",
            "engines": ["google"],
            "categories": ["general"],
            "user_id": config["configurable"]["user_id"],
            "org_id": config["configurable"]["org_id"]
        }
        print("\n--- Test Query: Web Search ---")
        result_search = await compiled_agent.ainvoke(search_input, config=config)
        print(f"Result: {result_search}")

        # Test web scrape
        scrape_input = {
            "type": "scrape",
            "messages": [HumanMessage(content="List all services provided by Bytecrafts.")],
            "url": "https://www.bytecrafts.in/",
            "extraction_prompt": "List all services provided by Bytecrafts.",
            "user_id": config["configurable"]["user_id"],
            "org_id": config["configurable"]["org_id"]
        }
        print("\n--- Test Query: Web Scrape ---")
        result_scrape = await compiled_agent.ainvoke(scrape_input, config=config)
        print(f"Result: {result_scrape}")

        print("\n=== Web Search & Scrape Agent Test Complete ===")

    asyncio.run(test_web_agent())
