from pathlib import Path
import yaml
from typing import Optional, Dict, Any

from agents.base_agents.base_agent import BaseAgent
from tools.knowledge_search.knowledge_search_tool import knowledge_search_tool


class KnowledgeSearchAgent(BaseAgent):
    def __init__(self,
                 prompt: Optional[str] = None,
                 *,
                 model_kwargs: Optional[Dict[str, Any]] = None,
                 vlm_kwargs: Optional[Dict[str, Any]] = None,
                 node_kwargs: Optional[Dict[str, Any]] = None,
                 recursion_limit: Optional[int] = 25,
                 debug: bool = False):
        
        if prompt is None:
            prompt = self._load_prompt()
        
        super().__init__(
            prompt=prompt,
            model_kwargs=model_kwargs,
            vlm_kwargs=vlm_kwargs,
            node_kwargs=node_kwargs,
            recursion_limit=recursion_limit,
            debug=debug
        )
        
        self.bind_tools([knowledge_search_tool])
    
    def _load_prompt(self) -> str:
        prompt_path = Path(__file__).parent / "prompt.yaml"
        
        with open(prompt_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
            prompt_data = yaml.safe_load(content)
            if isinstance(prompt_data, dict) and 'KnowledgeSearchAgent' in prompt_data:
                return prompt_data['KnowledgeSearchAgent']
            elif isinstance(prompt_data, str):
                return prompt_data

if __name__ == "__main__":
    import asyncio
    from langchain_core.messages import HumanMessage
    
    async def test_knowledge_search_agent():
        agent = KnowledgeSearchAgent(
            model_kwargs={},
            vlm_kwargs={},
            node_kwargs={},
            debug=True
        )
        
        compiled_agent = await agent.compile(name="KnowledgeSearchAgent")
        
        config = {
            "configurable": {
                "thread_id": "test_thread",
                "user_id": "test_user",
                "org_id": "test_org"
            }
        }
        
        print("=== Testing Knowledge Search Agent ===")
        
        test_input = {
            "messages": [
                HumanMessage(content="Search for all documents in the knowledge graph and show me their metadata.")
            ],
            "user_id": config["configurable"]["user_id"],
            "org_id": config["configurable"]["org_id"]
        }
        
        print("\n--- Test Query: Document Search ---")
        result = await compiled_agent.ainvoke(test_input, config=config)
        print(f"Result: {result}")
        
        follow_up_input = {
            "messages": [
                HumanMessage(content="Now search for some of the entities and their relationships in the knowledge graph.")
            ]
        }
        
        print("\n--- Test Query: Entity and Relationship Search ---")
        result2 = await compiled_agent.ainvoke(follow_up_input, config=config)
        print(f"Result: {result2}")
        
        print("\n=== Knowledge Search Agent Test Complete ===")
    
    asyncio.run(test_knowledge_search_agent())
