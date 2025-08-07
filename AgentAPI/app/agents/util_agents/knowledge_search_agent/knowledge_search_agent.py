from pathlib import Path
import yaml
from typing import Optional, Dict, Any
from langchain_core.runnables import RunnableConfig

from agents.base_agents.base_agent import BaseAgent
from tools.core_tools.knowledge_search.knowledge_search_tool import knowledge_search_tool
from agents.utils import _load_prompt

class KnowledgeSearchAgent(BaseAgent):
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
            prompt = ""
        
        knowledge_search_prompt = _load_prompt("KnowledgeSearchAgent", base_dir=Path(__file__).parent)
        final_prompt = prompt + knowledge_search_prompt

        super().__init__(
            prompt=final_prompt,
            config=config,
            model_kwargs=model_kwargs,
            vlm_kwargs=vlm_kwargs,
            node_kwargs=node_kwargs,
            recursion_limit=recursion_limit,
            debug=debug
        )
        
        self.bind_tools([knowledge_search_tool])
    

if __name__ == "__main__":
    import asyncio
    from langchain_core.messages import HumanMessage
    
    async def test_knowledge_search_agent():

        config = {
            "configurable": {
                "thread_id": "test_thread",
                "user_id": "36cdf650-0eb6-4dfa-a51d-5377cf704700",
                "org_id": "1"
            }
        }

        agent = KnowledgeSearchAgent(
            model_kwargs={},
            vlm_kwargs={},
            node_kwargs={},
            debug=True,
            config=config
        )
        
        compiled_agent = await agent.compile(name="KnowledgeSearchAgent")
        
        print("=== Testing Knowledge Search Agent ===")
        
        test_input = {
            "messages": [
                HumanMessage(content="Search for all documents in the knowledge graph and show me their metadata and summarise their content.")
            ],
            "user_id": config["configurable"]["user_id"],
            "org_id": config["configurable"]["org_id"]
        }
        
        print("\n--- Test Query: Document Search ---")
        result = await compiled_agent.ainvoke(test_input, config=config)
        print(f"Result: {result}")
        
        follow_up_input = {
            "messages": [
                HumanMessage(content="Now search for some of the pages and describe them.")
            ]
        }
        
        print("\n--- Test Query: Entity and Relationship Search ---")
        result2 = await compiled_agent.ainvoke(follow_up_input, config=config)
        print(f"Result: {result2}")
        
        print("\n=== Knowledge Search Agent Test Complete ===")
    
    asyncio.run(test_knowledge_search_agent())
