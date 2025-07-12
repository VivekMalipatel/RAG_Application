from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore, IndexConfig
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt
from typing import Any, Optional, Literal, List
from typing_extensions import TypedDict
from langchain_core.tools import tool, BaseTool
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage, ToolMessage
from langchain_openai import OpenAIEmbeddings
from dataclasses import dataclass
from pydantic import Field
import uuid
import asyncio
import json

from agents.base_agents.base_agent import BaseAgent 
from agents.base_agents.base_state import BaseState
from agents.base_agents.memory.base_checkpointer import BaseMemorySaver
from agents.base_agents.memory.base_store import BaseMemoryStore
from db.redis import redis
from llm.llm import LLM
from config import config as envconfig
from prompts import get_research_prompt


@dataclass(frozen=True)
class DeepResearchConfig:
    max_subqueries: int = 5
    recursionlimit: int = 100

class SubqueryList(TypedDict):
    subqueries: list[str] = Field(description="List of specific research subqueries based on identified gaps")

class GapExistence(TypedDict):
    exists: bool = Field(description="Indicates if a gap exists between the current research and the research contract. If True, it means there are gaps that need to be addressed. If False, it means the research is complete and no gaps were found.")
    message: str = Field(description="Message providing details about the gap analysis")


@tool
def human_clarification_tool(query: str) -> str:
    """Requests human clarification on the research query. After Gathering background knowledge Or During Query Intent Analysis, if the intent is unclear or any user inputs will be helpful, this tool will be invoked. 

    parameters
    ----------
    query: str
        The query for which clarification is needed. This should have set of questions to the user to help get a better understanding of the intent of the user.

    """
    
    human_response = interrupt({
        "type": "human_clarification", 
        "query": query,
        "instructions": "Please clarify the intent behind this query by answering these questions."
    })
    
    if human_response and hasattr(human_response, 'content'):
        return human_response.content
    elif isinstance(human_response, dict) and "response" in human_response:
        return human_response["response"]
    else:
        return "No clarification provided by the user. Continue with current research approach."

def create_redis_search_tool(redis_store: BaseMemoryStore, namespace: str):
    @tool
    async def redis_vector_search(query: str, limit: int = 5) -> str:
        """Search the So far researched knowledge in the knowledge store using vector similarity.
        You will be using this tool to try to retrieve the the information that satisfies the research contract with k=5. You will be calling this tool once for every contract item. 
        If the contract Item cannot be answered with the retrieved information, you can consider it as a research gap. 
        """
        search_results = await redis_store.asearch(namespace, query=query, limit=limit)
        if not search_results:
            return f"No relevant information found for query: {query}"
        
        formatted_results = []
        for result in search_results:
            content = result.value["content"]
            score = result.score
            formatted_results.append(f"Score: {score:.3f} - {content}")
        
        return "\n".join(formatted_results)

    return redis_vector_search

    
class DeepResearchAgent(BaseAgent):

    def __init__(self,
                 prompt: Optional[str] = "You are a specialized research agent.",
                 *,
                 model_kwargs: Optional[dict[str, Any]] = None,
                 vlm_kwargs: Optional[dict[str, Any]] = None,
                 node_kwargs: Optional[dict[str, Any]] = None,
                 debug: bool = False,
                 max_subqueries: int = 10,
                 max_research_rounds: int = 200):
        
        super().__init__(
            prompt=prompt,
            model_kwargs=model_kwargs,
            vlm_kwargs=vlm_kwargs,
            node_kwargs=node_kwargs,
            debug=debug
        )

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
        
        self._research_config = DeepResearchConfig(
            max_subqueries=max_subqueries,
            recursionlimit=max_research_rounds
        )
        
        self._gap_analysis_redis_store = BaseMemoryStore(redis_client=redis.get_session(), index=index_config)
        self._gap_analysis_state = None
        self._gap_analysis_llm : LLM = None
        self._gap_analysis_search_system_prompt = None
        self._gap_analysis_search_graph_compiled: CompiledStateGraph = None
        self._gap_analysis_redis_search_tool : BaseTool = None
        
        self._init_research_agents()

    def _init_research_agents(self):
        self.gather_background_knowledge_agent = BaseAgent(
            prompt=get_research_prompt('gather_background_knowledge'),
            model_kwargs=self._config.model_kwargs,
            vlm_kwargs=self._config.vlm_kwargs,
            node_kwargs=self._config.node_kwargs,
            debug=self._config.debug
        ).bind_tools([human_clarification_tool])

        self.query_intent_analysis_agent = BaseAgent(
            prompt=get_research_prompt('query_intent_analysis'),
            model_kwargs=self._config.model_kwargs,
            vlm_kwargs=self._config.vlm_kwargs,
            node_kwargs=self._config.node_kwargs,
            debug=self._config.debug
        ).bind_tools([human_clarification_tool])

        self.gap_analysis_agent = BaseAgent(
            prompt=get_research_prompt('gap_analysis'),
            model_kwargs=self._config.model_kwargs,
            vlm_kwargs=self._config.vlm_kwargs,
            node_kwargs=self._config.node_kwargs,
            debug=self._config.debug
        ).with_structured_output(GapExistence)

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
        node_config["recursion_limit"] = self._research_config.recursionlimit
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
            if agent is not None:
                await agent.compile(
                    checkpointer=checkpointer,
                    store=store,
                    debug=debug,
                    name=f"research_{node_name}"
                )

    async def gather_background_knowledge(self, state: BaseState, config: RunnableConfig) -> dict:
        node_config = self._create_node_config("gather_background_knowledge", config)
        result = await self.gather_background_knowledge_agent.ainvoke(state, node_config)
        return {"messages": result["messages"], "user_id": config["configurable"]["user_id"], "org_id": config["configurable"]["org_id"]}

    async def query_intent_analysis(self, state: BaseState, config: RunnableConfig) -> dict:
        node_config = self._create_node_config("query_intent_analysis", config)
        result = await self.query_intent_analysis_agent.ainvoke(state, node_config)
        return {"messages": result["messages"], "user_id": config["configurable"]["user_id"], "org_id": config["configurable"]["org_id"]}
    
    async def gap_analysis(self, state: BaseState, config: RunnableConfig) -> dict:

        node_config = self._create_node_config("gap_analysis", config)

        if not self._gap_analysis_state:
            self._gap_analysis_state = state.copy()

        async def _gap_search(message: BaseMessage) -> str:

            search_state = BaseState(messages=[message])
            if not self._gap_analysis_redis_search_tool:
                self._gap_analysis_redis_search_tool = create_redis_search_tool(self._gap_analysis_redis_store, ("memories",f"{config['configurable']['org_id']}_{config['configurable']['user_id']}_gap_analysis_node"))

            if not self._gap_analysis_llm:
                self._gap_analysis_llm = LLM().bind_tools([self._gap_analysis_redis_search_tool])
            
            if not self._gap_analysis_search_system_prompt:
                self._gap_analysis_search_system_prompt = "You are a research gap analyzer. Use the vector search tool to find relevant information and identify gaps in the current research."
            
            async def search_node(state: BaseState):
                messages = [SystemMessage(content=self._gap_analysis_search_system_prompt)] + state["messages"]
                result = await self._gap_analysis_llm.ainvoke(messages)
                return {"messages":[result]}

            if not self._gap_analysis_search_graph_compiled:

                graph_builder = StateGraph(BaseState)
                graph_builder.add_node("search_node", search_node)
                graph_builder.add_node("redis_search", ToolNode([self._gap_analysis_redis_search_tool]))

                graph_builder.add_edge(START, "search_node")
                graph_builder.add_conditional_edges("search_node",tools_condition,{"tools":"redis_search", "__end__": END})
                self._gap_analysis_search_graph_compiled = graph_builder.compile()

            return await self._gap_analysis_search_graph_compiled.ainvoke(search_state, {"configurable": {"recursion_limit": self._research_config.recursionlimit}}) 
        
        gaps = await _gap_search(self._gap_analysis_state["messages"][-1])
        messages =  state["messages"] + gaps["messages"]
        result = await self.gap_analysis_agent.ainvoke({"messages": messages}, config=node_config)
        return {"messages":result["messages"][-1], "user_id": config["configurable"]["user_id"], "org_id": config["configurable"]["org_id"]}

    async def generate_report(self, state: BaseState, config: RunnableConfig) -> dict:
        node_config = self._create_node_config("generate_report", config)
        result = await self.generate_report_agent.ainvoke(state, node_config)
        return result
  
    async def gaps_to_subquery(self, state: BaseState, config: RunnableConfig) -> dict:
        node_config = self._create_node_config("gaps_to_subquery", config)
        structured_result = await self.gaps_to_subquery_agent.ainvoke(state, node_config)
        structured_result: AIMessage = structured_result["messages"][-1]
        structured_result : SubqueryList = json.loads(structured_result.content)
        
        subqueries = structured_result["subqueries"][:self._research_config.max_subqueries]
        
        tasks = []
        for subquery in subqueries:
            subquery_state = {"messages": [HumanMessage(content=subquery)]}
            subquery_config = self._create_node_config("subquery_processor", config)
            task = self.subquery_processor_agent.ainvoke(subquery_state, subquery_config)
            tasks.append(task)
        
        subquery_results = await asyncio.gather(*tasks, return_exceptions=True)

        configurable = config.get("configurable", {})
        org_id = configurable.get("org_id", "default")
        user_id = configurable.get("user_id", "default")
        gap_analysis_namespace = ("memories",f"{org_id}_{user_id}_gap_analysis_node")
        
        async def store_message(i, result, subquery):
            if isinstance(result, Exception):
                print(f"Error processing subquery {i}: {result}")
                return
            if isinstance(result, dict) and "messages" in result:
                store_tasks = []
                for message in result["messages"]:
                    if isinstance(message, (AIMessage, ToolMessage)) and hasattr(message, "content") and message.content:
                        store_tasks.append(
                            self._gap_analysis_redis_store.aput(
                            namespace=gap_analysis_namespace,
                            key=f"subquery_{i}_{subquery}",
                            value={"content": message.content, "type": "subquery_result"}
                            )
                        )
            
            if store_tasks:
                await asyncio.gather(*store_tasks, return_exceptions=True)

        storage_tasks = [
            store_message(i, result, subqueries[i]) 
            for i, result in enumerate(subquery_results)
        ]
        
        await asyncio.gather(*storage_tasks, return_exceptions=True)
        
        return {
            "messages": state["messages"],
            "user_id": config["configurable"]["user_id"], 
            "org_id": config["configurable"]["org_id"],
        }

    def should_generate_report(self, state: BaseState) -> str:
        messages : AIMessage = state["messages"][-1]
        content = messages.content
        gap_existence: GapExistence = json.loads(content)

        if gap_existence["exists"]:
            return "gaps_to_subquery"
        else:
            return "generate_report"

    def _compile_graph(self, has_tools: bool, **compile_kwargs) -> CompiledStateGraph:
        graph_builder = StateGraph(BaseState)
        
        graph_builder.add_node("gather_background_knowledge", self.gather_background_knowledge)
        graph_builder.add_node("human_clarification_gather_gather_background_knowledge", ToolNode([human_clarification_tool]))
        graph_builder.add_node("query_intent_analysis", self.query_intent_analysis)
        graph_builder.add_node("human_clarification_query_intent_analysis", ToolNode([human_clarification_tool]))
        graph_builder.add_node("gap_analysis", self.gap_analysis)
        graph_builder.add_node("generate_report", self.generate_report)
        graph_builder.add_node("gaps_to_subquery", self.gaps_to_subquery)
        
        graph_builder.add_edge(START, "gather_background_knowledge")
        graph_builder.add_conditional_edges(
            "gather_background_knowledge",
            tools_condition,
            {"tools":"human_clarification_gather_gather_background_knowledge", "__end__": "query_intent_analysis"}
        )
        graph_builder.add_edge("human_clarification_gather_gather_background_knowledge","gather_background_knowledge")
        graph_builder.add_conditional_edges(
            "query_intent_analysis",
            tools_condition,
            {"tools":"human_clarification_query_intent_analysis", "__end__": "gap_analysis"}
        )
        # graph_builder.add_edge("gather_background_knowledge", "query_intent_analysis")
        graph_builder.add_edge("human_clarification_query_intent_analysis", "query_intent_analysis")
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
                max_subqueries=5,
                max_research_rounds=200
            )
            
            compiled_agent = await agent.compile(name="test_research_agent")
            
            config = {
                "recursion_limit": 200,
                "configurable": {
                    "thread_id": "main_research_thread", 
                    "user_id": "researcher_001", 
                    "org_id": "research_org",
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
                "org_id": "research_org",
                "research_rounds": 0
            }
            
            result = await compiled_agent.ainvoke(research_input, config=config)
            
            while True:
                try:
                    if result and "type" in str(result) and "human_clarification" in str(result):
                        print(f"\n--- Clarification Needed ---")
                        last_message = result["messages"][-1]
                        if hasattr(last_message, 'additional_kwargs') and 'tool_calls' in last_message.additional_kwargs:
                            tool_call = last_message.additional_kwargs["tool_calls"][0]
                            query = json.loads(tool_call["function"]["arguments"])["query"]
                        else:
                            query = "Please provide clarification for the research query."
                        print(f"Query: {query}")
                        user_input = input("Enter your clarification: ")
                        
                        clarification_input = {
                            "messages": [HumanMessage(content=user_input)],
                            "user_id": "researcher_001", 
                            "org_id": "research_org",
                            "research_rounds": 0
                        }
                        result = await compiled_agent.ainvoke(clarification_input, config=config)
                    else:
                        break
                except Exception as interrupt_error:
                    print(f"Interrupt handling error: {interrupt_error}")
                    break
            
            print(f"Research Result: {result}")
            
            print(f"\n=== Hierarchical Memory Test Completed ===")
            
        except Exception as e:
            print(f"Test failed with error: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")

    asyncio.run(test_deep_research_agent())
