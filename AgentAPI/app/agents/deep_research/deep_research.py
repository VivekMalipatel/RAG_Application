from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore
from langgraph.types import All, StreamMode, interrupt
import typing
import asyncio
import json
import logging
from typing import Any, Sequence, Union, Optional, Callable, AsyncIterator, Literal
from langchain_core.tools import BaseTool, tool
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from dataclasses import dataclass, field
from functools import wraps
from app.agents.base_agents.base_agent import BaseAgent 
from app.agents.base_agents.base_state import BaseState
from app.agents.base_agents.base_checkpointer import MemorySaver
from prompts import get_research_prompt


@dataclass(frozen=True)
class DeepResearchConfig:
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    vlm_kwargs: dict[str, Any] = field(default_factory=dict)
    node_kwargs: dict[str, Any] = field(default_factory=dict)
    debug: bool = False
    max_subqueries: int = 5
    max_research_rounds: int = 3


def requires_compile(fn):
    @wraps(fn)
    async def wrapper(self, *args, **kwargs):
        if not self._compiled_graph:
            raise ValueError("DeepResearchAgent not compiled. Call compile() first.")
        return await fn(self, *args, **kwargs)
    return wrapper


@tool  
def human_research_guidance_tool(gaps_found: str, current_research: str = "") -> str:
    """Request human guidance on research direction when gaps are identified."""
    human_response = interrupt({
        "type": "research_guidance", 
        "gaps_found": gaps_found,
        "current_research": current_research,
        "instructions": "Please provide guidance on how to address these research gaps."
    })
    return human_response.get("data", "Continue with current research approach")

    
class DeepResearchAgent:
    """
    A comprehensive research agent that follows human-in-the-loop workflow.
    Each research node is powered by a BaseAgent for consistent behavior.
    """

    def __init__(self,
                 *,
                 model_kwargs: Optional[dict[str, Any]] = None,
                 vlm_kwargs: Optional[dict[str, Any]] = None,
                 node_kwargs: Optional[dict[str, Any]] = None,
                 debug: bool = False,
                 max_subqueries: int = 5,
                 max_research_rounds: int = 3):
        
        self._config = DeepResearchConfig(
            model_kwargs=model_kwargs or {},
            vlm_kwargs=vlm_kwargs or {},
            node_kwargs=node_kwargs or {},
            debug=debug,
            max_subqueries=max_subqueries,
            max_research_rounds=max_research_rounds
        )
        self._compiled_graph: Optional[CompiledStateGraph] = None
        self._lock = asyncio.Lock()
        self._tools: Sequence[Union[typing.Dict[str, Any], type, Callable, BaseTool]] = []
        self._logger = logging.getLogger(__name__)
        
        # Initialize BaseAgent instances for each research node
        self._init_base_agents()
        
        self._checkpointer = None
        self._store = None
        self._interrupt_before = None
        self._interrupt_after = None
        self._name = None

    def _init_base_agents(self):
        """Initialize BaseAgent instances for each research node"""
        # Create BaseAgent instances without set_prompt (which doesn't exist)
        self.gather_background_knowledge_agent = BaseAgent(
            model_kwargs=self._config.model_kwargs,
            vlm_kwargs=self._config.vlm_kwargs,
            node_kwargs=self._config.node_kwargs,
            debug=self._config.debug
        ).compile()
        
        self.user_intent_analysis_agent = BaseAgent(
            model_kwargs=self._config.model_kwargs,
            vlm_kwargs=self._config.vlm_kwargs,
            node_kwargs=self._config.node_kwargs,
            debug=self._config.debug
        ).compile()
        
        self.human_clarification_agent = BaseAgent(
            model_kwargs=self._config.model_kwargs,
            vlm_kwargs=self._config.vlm_kwargs,
            node_kwargs=self._config.node_kwargs,
            debug=self._config.debug
        ).compile()
        
        self.query_intent_analysis_agent = BaseAgent(
            model_kwargs=self._config.model_kwargs,
            vlm_kwargs=self._config.vlm_kwargs,
            node_kwargs=self._config.node_kwargs,
            debug=self._config.debug
        ).compile()
        
        self.gap_analysis_agent = BaseAgent(
            model_kwargs=self._config.model_kwargs,
            vlm_kwargs=self._config.vlm_kwargs,
            node_kwargs=self._config.node_kwargs,
            debug=self._config.debug
        ).compile()
        
        self.generate_report_agent = BaseAgent(
            model_kwargs=self._config.model_kwargs,
            vlm_kwargs=self._config.vlm_kwargs,
            node_kwargs=self._config.node_kwargs,
            debug=self._config.debug
        ).compile()
        self.gaps_to_subquery_agent = BaseAgent(
            model_kwargs=self._config.model_kwargs,
            vlm_kwargs=self._config.vlm_kwargs,
            node_kwargs=self._config.node_kwargs,
            debug=self._config.debug
        ).compile()
        self.subquery_processor_agent = BaseAgent(
            model_kwargs=self._config.model_kwargs,
            vlm_kwargs=self._config.vlm_kwargs,
            node_kwargs=self._config.node_kwargs,
            debug=self._config.debug
        ).compile()

    ###########RESEARCH NODE IMPLEMENTATIONS###########
    async def gather_background_knowledge(self, state: State) -> dict:
        """Gather background knowledge using BaseAgent"""
        try:
            system_prompt = get_research_prompt('gather_background_knowledge')
            # Add system message to the state
            messages = state["messages"].copy()
            messages.insert(0, SystemMessage(content=system_prompt))
            modified_state = {"messages": messages}
            
            result = await self.gather_background_knowledge_agent.ainvoke(modified_state)
            return result
        except Exception as e:
            self._logger.error(f"Error in gather_background_knowledge: {e}")
            return {"messages": state["messages"]}

    async def user_intent_analysis(self, state: State) -> dict:
        """Analyze user intent using BaseAgent"""
        try:
            result = await self.user_intent_analysis_agent.ainvoke(state)
            return result
        except Exception as e:
            self._logger.error(f"Error in user_intent_analysis: {e}")
            return {"messages": state["messages"]}
     
    async def human_clarification_node(self, state: State) -> dict:
        """Handle human clarification using direct interrupt"""
        try:
            # Get the last message to understand what needs clarification
            last_message = state["messages"][-1] if state["messages"] else ""
            context = str(last_message)
            # Extract the unclear query/intent
            if hasattr(last_message, "content"):
                unclear_content = last_message.content
            else:
                unclear_content = "User's research intent is unclear"
            # Direct interrupt for human clarification
            human_response = interrupt({
                "type": "human_clarification",
                "query": unclear_content,
                "context": context,
                "instructions": "Please provide clarification to help focus the research. What specific aspects would you like me to research?"
            })
            # Process human response
            clarification = human_response.get("data", "No clarification provided")
            # Create response message
            response_message = HumanMessage(content=f"Human clarification received: {clarification}")
            return {"messages": [response_message]}
        except Exception as e:
            self._logger.error(f"Error in human_clarification_node: {e}")
            return {"messages": state["messages"]}
    
    async def query_intent_analysis(self, state: State) -> dict:
        """Analyze query intent using BaseAgent"""
        try:
            result = await self.query_intent_analysis_agent.ainvoke(state)
            return result
        except Exception as e:
            self._logger.error(f"Error in query_intent_analysis: {e}")
            return {"messages": state["messages"]}
    
    async def gap_analysis(self, state: State) -> dict:
        """Perform gap analysis using BaseAgent"""
        try:
            result = await self.gap_analysis_agent.ainvoke(state)
            return result
        except Exception as e:
            self._logger.error(f"Error in gap_analysis: {e}")
            return {"messages": state["messages"]}

    async def generate_report(self, state: State) -> dict:
        """Generate final report using BaseAgent"""
        try:
            result = await self.generate_report_agent.ainvoke(state)
            return result
        except Exception as e:
            self._logger.error(f"Error in generate_report: {e}")
            return {"messages": state["messages"]}
  
    async def gaps_to_subquery(self, state: State) -> dict:
        """Convert gaps to subqueries using BaseAgent with additional processing"""
        try:
            result = await self.gaps_to_subquery_agent.ainvoke(state)
            
            # Extract subqueries from the response
            if result.get("messages"):
                last_message = result["messages"][-1]
                subqueries = self._extract_subqueries(last_message)
                result.update({
                    "subqueries": subqueries,
                    "pending_subqueries": subqueries.copy(),
                    "completed_subqueries": [],
                    "current_subquery_index": 0
                })
            
            return result
        except Exception as e:
            self._logger.error(f"Error in gaps_to_subquery: {e}")
            return {"messages": state["messages"]}

    async def subquery_processor(self, state: State) -> dict:
        """Process all subqueries using BaseAgent instances"""
        try:
            subqueries = state.get("subqueries", [])
            
            if not subqueries:
                return {"messages": state["messages"]}
            
            # Limit number of subqueries
            subqueries = subqueries[:self._config.max_subqueries]
            
            # Process all subqueries using the subquery processor agent
            all_responses = []
            updated_messages = state["messages"].copy()
            
            for i, subquery in enumerate(subqueries):
                # Create a temporary state with the specific subquery
                subquery_state = {
                    "messages": [HumanMessage(content=f"Research this specific question: {subquery}")]
                }
                
                # Process using the BaseAgent
                result = await self.subquery_processor_agent.ainvoke(subquery_state)
                
                if result.get("messages"):
                    response = result["messages"][-1]
                    all_responses.append(response)
                    
                    # Add to conversation history
                    updated_messages.append({"role": "user", "content": f"Subquery {i+1}: {subquery}"})
                    updated_messages.append(response)
            
            return {
                "messages": updated_messages,
                "subqueries": [],  # Clear subqueries after processing
                "pending_subqueries": [],
                "completed_subqueries": subqueries,
                "subquery_results": all_responses
            }
            
        except Exception as e:
            self._logger.error(f"Error in subquery_processor: {e}")
            return {"messages": state["messages"]}

    ###########HELPER METHODS###########
    def _extract_subqueries(self, response) -> list[str]:
        """Extract subqueries from the gaps_to_subquery response"""
        try:
            if hasattr(response, "content"):
                content = response.content
            else:
                content = str(response)
            
            # Try to extract JSON from the response
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = content[start_idx:end_idx]
                parsed = json.loads(json_str)
                return parsed.get("subqueries", [])
        except Exception as e:
            self._logger.debug(f"JSON extraction failed: {e}")
        
        # Fallback: simple line-based extraction
        lines = content.split('\n')
        subqueries = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('{') and not line.startswith('}'):
                if line.startswith('"') and line.endswith('"'):
                    subqueries.append(line[1:-1])
                elif line.startswith('-') or line.startswith('*'):
                    subqueries.append(line[1:].strip())
        
        return subqueries[:self._config.max_subqueries]

    ###########CONDITIONAL FUNCTIONS###########
    def should_request_clarification(self, state: State) -> str:
        """Determine if we need human clarification or can proceed"""
        try:
            last_message = state["messages"][-1]
            
            # Check if the last response contains "UNCLEAR"
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
        except Exception as e:
            self._logger.error(f"Error in should_request_clarification: {e}")
            return "query_intent_analysis"  # Default to proceed

    def should_generate_report(self, state: State) -> str:
        """Determine if we should generate report or process gaps"""
        try:
            last_message = state["messages"][-1]
            
            # Check if the last response contains "NO_GAPS"
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
        except Exception as e:
            self._logger.error(f"Error in should_generate_report: {e}")
            return "generate_report"  # Default to end

    ###########GRAPH COMPILATION###########
    def _compile_graph(self, **compile_kwargs) -> CompiledStateGraph:
        """Compile the research graph with BaseAgent nodes"""
        graph_builder = StateGraph(State)
        
        # Add all research nodes (each powered by BaseAgent)
        graph_builder.add_node("gather_background_knowledge", self.gather_background_knowledge)
        graph_builder.add_node("user_intent_analysis", self.user_intent_analysis)
        graph_builder.add_node("human_clarification", self.human_clarification_node)
        graph_builder.add_node("query_intent_analysis", self.query_intent_analysis)
        graph_builder.add_node("gap_analysis", self.gap_analysis)
        graph_builder.add_node("generate_report", self.generate_report)
        graph_builder.add_node("gaps_to_subquery", self.gaps_to_subquery)
        graph_builder.add_node("subquery_processor", self.subquery_processor)
        
        # Sequential flow
        graph_builder.add_edge(START, "gather_background_knowledge")
        graph_builder.add_edge("gather_background_knowledge", "user_intent_analysis")
        
        # Intent analysis flow
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
        
        # Gap analysis and research iteration
        graph_builder.add_conditional_edges(
            "gap_analysis",
            self.should_generate_report,
            {
                "generate_report": "generate_report",
                "gaps_to_subquery": "gaps_to_subquery"
            }
        )
        graph_builder.add_edge("gaps_to_subquery", "subquery_processor")
        graph_builder.add_edge("subquery_processor", "gap_analysis")
        graph_builder.add_edge("generate_report", END)

        return graph_builder.compile(**compile_kwargs)

    def compile(self,
                checkpointer: Optional[BaseCheckpointSaver] = None,
                *,
                store: Optional[BaseStore] = None,
                interrupt_before: Optional[Union[All, list[str]]] = None,
                interrupt_after: Optional[Union[All, list[str]]] = None,
                debug: Optional[bool] = None,
                name: Optional[str] = None) -> 'DeepResearchAgent':
        """Compile the research agent and all BaseAgent instances"""

        self._checkpointer = checkpointer
        self._store = store
        self._interrupt_before = interrupt_before or ["human_clarification"]
        self._interrupt_after = interrupt_after
        self._name = name

        compile_kwargs = {
            "checkpointer": checkpointer,
            "store": store,
            "interrupt_before": self._interrupt_before,
            "interrupt_after": interrupt_after,
            "debug": debug if debug is not None else self._config.debug,
            "name": name
        }

        self._compiled_graph = self._compile_graph(**compile_kwargs)
        return self

    ###########EXECUTION METHODS###########
    @requires_compile
    async def ainvoke(self,
                      input: dict[str, Any] | Any,
                      config: RunnableConfig | None = None,
                      **kwargs: Any) -> dict[str, Any] | Any:
        
        return await self._compiled_graph.ainvoke(input, config=config, **kwargs)

    @requires_compile
    async def astream(self,
                      input: dict[str, Any] | Any,
                      config: RunnableConfig | None = None,
                      **kwargs: Any) -> AsyncIterator[dict[str, Any] | Any]:
        
        async for chunk in self._compiled_graph.astream(input, config=config, **kwargs):
            yield chunk


if __name__ == "__main__":
    async def test_deep_research_agent():
        """Test the deep research agent with BaseAgent nodes"""
        print("=== Testing DeepResearchAgent with BaseAgent nodes ===")
        
        try:
            checkpointer = MemorySaver()
            
            # Create and compile agent
            agent = DeepResearchAgent(
                model_kwargs={},
                vlm_kwargs={},
                node_kwargs={},
                debug=True,
                max_subqueries=3,
                max_research_rounds=2
            ).compile(
                checkpointer=checkpointer,
                name="test_research_agent"
            )
            
            config = {
                "configurable": {"thread_id": "research_test"},
                "recursion_limit": 100  # Increase from default 25 to 100
            }
            
            # Test research query
            print("\n--- Test: Research Query with BaseAgent nodes ---")
            research_input = {
                "messages": [HumanMessage(content="What are the latest developments in quantum computing?")]
            }
            
            result = await agent.ainvoke(research_input, config=config)
            print(f"Research Result: {result}")
            
            print("\n=== DeepResearchAgent test completed successfully ===")
            
        except Exception as e:
            print(f"Test failed with error: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")

    asyncio.run(test_deep_research_agent())