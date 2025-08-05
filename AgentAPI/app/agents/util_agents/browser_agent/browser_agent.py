from pathlib import Path
import yaml
from typing import Optional, Dict, Any
from langchain_core.runnables import RunnableConfig

from agents.base_agents.base_agent import BaseAgent
from agents.utils import _load_prompt
from tools.core_tools.browser_use.browser_use_tool import browser_use_tool, mb_use_browser_tool

class BrowserAgent(BaseAgent):
    def __init__(self,
                 prompt: Optional[str] = None,
                 *,
                 config: RunnableConfig,
                 model_kwargs: Optional[Dict[str, Any]] = None,
                 vlm_kwargs: Optional[Dict[str, Any]] = None,
                 node_kwargs: Optional[Dict[str, Any]] = None,
                 recursion_limit: Optional[int] = 25,
                 debug: bool = False,
                 use_marinabox: bool = True):
        if prompt is None:
            prompt = ""
        
        browser_agent_prompt = _load_prompt("BrowserAgent", base_dir=Path(__file__).parent)
        final_prompt = prompt + browser_agent_prompt
        
        super().__init__(
            prompt=final_prompt,
            config=config,
            model_kwargs=model_kwargs,
            vlm_kwargs=vlm_kwargs,
            node_kwargs=node_kwargs,
            recursion_limit=recursion_limit,
            debug=debug
        )
        
        # Choose tools based on marinabox availability
        if use_marinabox:
            self.bind_tools([mb_use_browser_tool, browser_use_tool])
        else:
            self.bind_tools([browser_use_tool])


if __name__ == "__main__":
    import asyncio
    from langchain_core.messages import HumanMessage
    from tools.core_tools.browser_use.browser_use_tool import mb_start_browser, mb_stop_browser

    async def test_browser_agent():
        agent = BrowserAgent(
            model_kwargs={},
            vlm_kwargs={},
            node_kwargs={},
            debug=True,
            use_marinabox=True
        )
        compiled_agent = await agent.compile(name="BrowserAgent")
        config = {
            "configurable": {
                "thread_id": "browser_test",
                "user_id": "test_user",
                "org_id": "test_org"
            }
        }
        print("=== Testing Browser Agent with Marinabox ===")

        # Test browser automation with marinabox tool
        browser_input = {
            "messages": [HumanMessage(content="Use mb_use_browser_tool to open bytecrafts.in and find their services section. Take a screenshot.")],
            "user_id": config["configurable"]["user_id"],
            "org_id": config["configurable"]["org_id"]
        }
        print("\n--- Test Query: Browser Automation ---")
        result = await compiled_agent.ainvoke(browser_input, config=config)
        print(f"Result: {result}")

        print("\n=== Browser Agent Test Complete ===")

    # Test LangGraph workflow functions
    def test_workflow_functions():
        print("\n=== Testing Workflow Functions ===")
        
        # Test start browser
        initial_state = {"messages": []}
        browser_state = mb_start_browser(initial_state)
        print(f"Start browser result: {browser_state}")
        
        # Test stop browser
        final_state = mb_stop_browser(browser_state)
        print(f"Stop browser result: {final_state}")

    # Run tests
    test_workflow_functions()
    asyncio.run(test_browser_agent())
