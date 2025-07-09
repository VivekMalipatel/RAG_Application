from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore, IndexConfig
from langgraph.types import interrupt
from typing import Any, Optional, Literal
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import OpenAIEmbeddings
from dataclasses import dataclass
from pydantic import BaseModel, Field
import uuid
import asyncio

from agents.base_agents.base_agent import BaseAgent 
from agents.base_agents.base_state import BaseState
from agents.base_agents.memory.base_checkpointer import BaseMemorySaver
from agents.base_agents.memory.base_store import BaseMemoryStore
from db.redis import redis
from config import config as envconfig
from prompts import get_research_prompt


@dataclass(frozen=True)
class DeepResearchConfig:
    max_subqueries: int = 5
    max_research_rounds: int = 3


class SubqueryList(BaseModel):
    subqueries: list[str] = Field(description="List of specific research subqueries based on identified gaps")


@tool  
def human_research_guidance_tool(gaps_found: str, current_research: str = "") -> str:
    """    Provides human clarification on intent analysis and research gaps."""
    human_response = interrupt({
        "type": "research_guidance", 
        "gaps_found": gaps_found,
        "current_research": current_research,
        "instructions": "Please provide guidance on how to address these research gaps."
    })
    return human_response.get("data", "Continue with current research approach")

    
class DeepResearchAgent(BaseAgent):

    def __init__(self,
                 prompt: Optional[str] = "You are a specialized research agent.",
                 *,
                 model_kwargs: Optional[dict[str, Any]] = None,
                 vlm_kwargs: Optional[dict[str, Any]] = None,
                 node_kwargs: Optional[dict[str, Any]] = None,
                 debug: bool = False,
                 max_subqueries: int = 5,
                 max_research_rounds: int = 3):
        
        super().__init__(
            prompt=prompt,
            model_kwargs=model_kwargs,
            vlm_kwargs=vlm_kwargs,
            node_kwargs=node_kwargs,
            debug=debug
        )
        
        self._research_config = DeepResearchConfig(
            max_subqueries=max_subqueries,
            max_research_rounds=max_research_rounds
        )
        
        self._init_research_agents()

    def _init_research_agents(self):
        self.gather_background_knowledge_agent = BaseAgent(
            prompt=get_research_prompt('gather_background_knowledge'),
            model_kwargs=self._config.model_kwargs,
            vlm_kwargs=self._config.vlm_kwargs,
            node_kwargs=self._config.node_kwargs,
            debug=self._config.debug
        )

        self.user_intent_analysis_agent = BaseAgent(
            prompt=get_research_prompt('user_intent_analysis'),
            model_kwargs=self._config.model_kwargs,
            vlm_kwargs=self._config.vlm_kwargs,
            node_kwargs=self._config.node_kwargs,
            debug=self._config.debug
        )

        self.query_intent_analysis_agent = BaseAgent(
            prompt=get_research_prompt('query_intent_analysis'),
            model_kwargs=self._config.model_kwargs,
            vlm_kwargs=self._config.vlm_kwargs,
            node_kwargs=self._config.node_kwargs,
            debug=self._config.debug
        )

        self.gap_analysis_agent = BaseAgent(
            prompt=get_research_prompt('gap_analysis'),
            model_kwargs=self._config.model_kwargs,
            vlm_kwargs=self._config.vlm_kwargs,
            node_kwargs=self._config.node_kwargs,
            debug=self._config.debug
        )

        self.generate_report_agent = BaseAgent(
            prompt=get_research_prompt('generate_report'),
            model_kwargs=self._config.model_kwargs,
            vlm_kwargs=self._config.vlm_kwargs,
            node_kwargs=self._config.node_kwargs,
            debug=self._config.debug
        )

        self.gaps_to_subquery_agent = BaseAgent(
            prompt=get_research_prompt('gaps_to_subquery'),
            model_kwargs=self._config.model_kwargs,
            vlm_kwargs=self._config.vlm_kwargs,
            node_kwargs=self._config.node_kwargs,
            debug=self._config.debug
        ).with_structured_output(SubqueryList)

        self.subquery_processor_agent = BaseAgent(
            prompt=get_research_prompt('subquery_processor'),
            model_kwargs=self._config.model_kwargs,
            vlm_kwargs=self._config.vlm_kwargs,
            node_kwargs=self._config.node_kwargs,
            debug=self._config.debug
        )
        
        self._node_agents = {
            "gather_background_knowledge": self.gather_background_knowledge_agent,
            "user_intent_analysis": self.user_intent_analysis_agent,
            "query_intent_analysis": self.query_intent_analysis_agent,
            "gap_analysis": self.gap_analysis_agent,
            "generate_report": self.generate_report_agent,
            "gaps_to_subquery": self.gaps_to_subquery_agent,
            "subquery_processor": self.subquery_processor_agent
        }

    def _create_node_config(self, node_name: str, original_config: RunnableConfig) -> RunnableConfig:
        original_configurable = original_config.get("configurable", {})
        original_thread_id = original_configurable.get("thread_id", "default")
        original_user_id = original_configurable.get("user_id", "default")
        original_org_id = original_configurable.get("org_id", "default")
        
        node_config = original_config.copy()
        node_config["configurable"] = {
            "thread_id": f"{original_thread_id}_{str(uuid.uuid4())}",
            "user_id": f"{original_user_id}_{node_name}",
            "org_id": f"{original_org_id}_{original_user_id}"
        }
        
        return node_config

    async def _compile_node_agents(self,
                      checkpointer: Optional[BaseMemorySaver] = None,
                      *,
                      store: Optional[BaseStore] = None,
                      interrupt_before: list[str] | Literal['*'] | None = None,
                      interrupt_after: list[str] | Literal['*'] | None = None,
                      debug: bool = False,
                      name: str | None = None) -> 'BaseAgent':
        for node_name, agent in self._node_agents.items():
            await agent.compile(
                checkpointer=checkpointer,
                store=store,
                debug=debug,
                name=f"research_{node_name}"
            )

    async def gather_background_knowledge(self, state: BaseState, config: RunnableConfig) -> dict:
        node_config = self._create_node_config("gather_background_knowledge", config)
        result = await self.gather_background_knowledge_agent.ainvoke(state, node_config)
        return result

    async def user_intent_analysis(self, state: BaseState, config: RunnableConfig) -> dict:
        node_config = self._create_node_config("user_intent_analysis", config)
        result = await self.user_intent_analysis_agent.ainvoke(state, node_config)
        return result
     
    async def human_clarification_node(self, state: BaseState, config: RunnableConfig) -> dict:
        last_message = state["messages"][-1] if state["messages"] else ""
        context = str(last_message)
        if hasattr(last_message, "content"):
            unclear_content = last_message.content
        else:
            unclear_content = str(last_message)
            
        human_response = interrupt({
            "type": "human_clarification",
            "query": unclear_content,
            "context": context,
            "instructions": "Please provide clarification to help focus the research. What specific aspects would you like me to research?"
        })
        clarification = human_response.get("data", "No clarification provided")
        
        return {"messages": [HumanMessage(content=clarification)]}

    async def query_intent_analysis(self, state: BaseState, config: RunnableConfig) -> dict:
        node_config = self._create_node_config("query_intent_analysis", config)
        result = await self.query_intent_analysis_agent.ainvoke(state, node_config)
        return result
    
    async def gap_analysis(self, state: BaseState, config: RunnableConfig) -> dict:
        node_config = self._create_node_config("gap_analysis", config)
        result = await self.gap_analysis_agent.ainvoke(state, node_config)
        return result

    async def generate_report(self, state: BaseState, config: RunnableConfig) -> dict:
        node_config = self._create_node_config("generate_report", config)
        result = await self.generate_report_agent.ainvoke(state, node_config)
        return result
  
    async def gaps_to_subquery(self, state: BaseState, config: RunnableConfig) -> dict:
        node_config = self._create_node_config("gaps_to_subquery", config)
        structured_result: SubqueryList = await self.gaps_to_subquery_agent.ainvoke(state, node_config)
        
        subqueries = structured_result.subqueries[:self._research_config.max_subqueries]
        
        tasks = []
        for subquery in subqueries:
            subquery_state = {"messages": [HumanMessage(content=subquery)]}
            subquery_config = self._create_node_config("subquery_processor", config)
            task = self.subquery_processor_agent.ainvoke(subquery_state, subquery_config)
            tasks.append(task)
        
        subquery_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_messages = state["messages"].copy()
        for i, result in enumerate(subquery_results):
            if isinstance(result, Exception):
                self._logger.error(f"Error processing subquery {i}: {result}")
                continue
            if isinstance(result, dict) and "messages" in result:
                all_messages.extend(result["messages"])
        
        return {"messages": all_messages}

    def should_request_clarification(self, state: BaseState) -> str:
        last_message = state["messages"][-1] if state["messages"] else ""

        if hasattr(last_message, "content"):
            content = last_message.content.upper()
        elif isinstance(last_message, dict) and "content" in last_message:
            content = last_message["content"].upper()
        else:
            content = ""
        
        if "UNCLEAR" in content:
            return "human_clarification"
        else:
            return "query_intent_analysis"

    def should_generate_report(self, state: BaseState) -> str:
        last_message = state["messages"][-1] if state["messages"] else ""

        if hasattr(last_message, "content"):
                content = last_message.content.upper()
        elif isinstance(last_message, dict) and "content" in last_message:
            content = last_message["content"].upper()
        else:
            content = ""
        
        if "NO_GAPS" in content:
            return "generate_report"
        else:
            return "gaps_to_subquery"


    def _compile_graph(self, has_tools: bool, **compile_kwargs) -> CompiledStateGraph:
        graph_builder = StateGraph(BaseState)
        
        graph_builder.add_node("gather_background_knowledge", self.gather_background_knowledge)
        graph_builder.add_node("user_intent_analysis", self.user_intent_analysis)
        graph_builder.add_node("human_clarification", self.human_clarification_node)
        graph_builder.add_node("query_intent_analysis", self.query_intent_analysis)
        graph_builder.add_node("gap_analysis", self.gap_analysis)
        graph_builder.add_node("generate_report", self.generate_report)
        graph_builder.add_node("gaps_to_subquery", self.gaps_to_subquery)
        
        graph_builder.add_edge(START, "gather_background_knowledge")
        graph_builder.add_edge("gather_background_knowledge", "user_intent_analysis")
        
        graph_builder.add_conditional_edges(
            "user_intent_analysis",
            self.should_request_clarification,
            {
                "human_clarification": "human_clarification",
                "query_intent_analysis": "query_intent_analysis"
            }
        )
        graph_builder.add_edge("human_clarification", "gather_background_knowledge")
        graph_builder.add_edge("query_intent_analysis", "gap_analysis")
        
        graph_builder.add_conditional_edges(
            "gap_analysis",
            self.should_generate_report,
            {
                "generate_report": "generate_report",
                "gaps_to_subquery": "gaps_to_subquery"
            }
        )
        graph_builder.add_edge("gaps_to_subquery", "gap_analysis")
        graph_builder.add_edge("generate_report", END)

        return graph_builder.compile(**compile_kwargs)

    async def compile(self,
                      checkpointer: Optional[BaseMemorySaver] = None,
                      *,
                      store: Optional[BaseStore] = None,
                      interrupt_before: list[str] | Literal['*'] | None = None,
                      interrupt_after: list[str] | Literal['*'] | None = None,
                      debug: bool = False,
                      name: str | None = None) -> 'DeepResearchAgent':

        self._checkpointer = checkpointer
        if checkpointer is None:
            checkpointer = BaseMemorySaver(redis_client=redis.get_session())
            self._checkpointer = checkpointer
            await self._checkpointer.asetup()

        self._store = store
        if store is None:
            index_config: IndexConfig = {
                "dims": envconfig.MULTIMODEL_EMBEDDING_MODEL_DIMS,
                "embed": OpenAIEmbeddings(
                    model=envconfig.MULTIMODEL_EMBEDDING_MODEL,
                    base_url=envconfig.OPENAI_BASE_URL, 
                    api_key=envconfig.OPENAI_API_KEY
                ),
                "ann_index_config": {"vector_type": "vector"},
                "distance_type": "cosine",
            }

            store = BaseMemoryStore(
                redis_client=redis.get_session(),
                index=index_config,
            )
            self._store = store
            await self._store.setup()

        self._interrupt_before = interrupt_before
        self._interrupt_after = interrupt_after
        self._name = name

        compile_kwargs = {
            "checkpointer": checkpointer,
            "store": store,
            "interrupt_before": interrupt_before,
            "interrupt_after": interrupt_after,
            "debug": debug if debug is not None else self._config.debug,
            "name": name
        }

        await super().compile(**compile_kwargs)
        await self._compile_node_agents(checkpointer=checkpointer, store=store, debug=debug)
        return self


if __name__ == "__main__":
    import asyncio
    from langchain_core.messages import HumanMessage
    
    async def test_deep_research_agent():
        print("=== Testing DeepResearchAgent with Hierarchical Memory ===")
        
        try:
            agent = DeepResearchAgent(
                model_kwargs={},
                vlm_kwargs={},
                node_kwargs={},
                debug=True,
                max_subqueries=3,
                max_research_rounds=2
            )
            
            compiled_agent = await agent.compile(name="test_research_agent")
            
            config = {
                "configurable": {
                    "thread_id": "main_research_thread", 
                    "user_id": "researcher_001", 
                    "org_id": "research_org"
                }
            }
            
            print(f"\n--- Main DeepResearch Config ---")
            print(f"Thread ID: {config['configurable']['thread_id']}")
            print(f"User ID: {config['configurable']['user_id']}")
            print(f"Org ID: {config['configurable']['org_id']}")
            
            print(f"\n--- Node Memory Hierarchy Examples ---")
            for node_name in ["gather_background_knowledge", "user_intent_analysis", "gap_analysis"]:
                node_config = compiled_agent._create_node_config(node_name, config)
                print(f"\n{node_name} node:")
                print(f"  Thread ID: {node_config['configurable']['thread_id']}")
                print(f"  User ID: {node_config['configurable']['user_id']}")
                print(f"  Org ID: {node_config['configurable']['org_id']}")
            
            print(f"\n--- Test: Research Query ---")
            research_input = {
                "messages": [HumanMessage(content="What are the latest developments in quantum computing?")],
                "user_id": "researcher_001",
                "org_id": "research_org"
            }
            
            result = await compiled_agent.ainvoke(research_input, config=config)
            print(f"Research Result: {result}")
            
            print(f"\n=== Hierarchical Memory Test Completed ===")
            
        except Exception as e:
            print(f"Test failed with error: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")

    asyncio.run(test_deep_research_agent())
