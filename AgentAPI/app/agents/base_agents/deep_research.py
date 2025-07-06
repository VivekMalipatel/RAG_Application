from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
import typing
from typing import Any, Sequence, Union, Optional, Callable
from langchain_core.tools import BaseTool
from agents.base_agents.single_shot import SingleShotAgent
from agents.base_states.simple_state import State
from agents.base_checkpointers.simple_checkpointer import MemorySaver
from prompts import get_research_prompt

graph_builder = StateGraph(State)

class DeepResearchAgent:

    def __init__(self, prompt=None):

        self.gather_background_knowledge = SingleShotAgent()
        self.user_intent_analysis = SingleShotAgent()
        self.human_clarification = SingleShotAgent()
        self.query_intent_analysis = SingleShotAgent()
        self.gap_analysis = SingleShotAgent()
        self.generate_report = SingleShotAgent()
        self.gaps_to_subquery = SingleShotAgent()
        self.subquery_processor = SingleShotAgent()
        self.graph_builder = StateGraph(State)
        self.checkpointer = MemorySaver()
        self.compiled_graph : CompiledStateGraph = None
        self.prompt = prompt
        self.tools: Sequence[Union[typing.Dict[str, Any], type, Callable, BaseTool]] = []

    def bind_tools(self,
        tools: Sequence[
            Union[typing.Dict[str, Any], type, Callable, BaseTool]
        ],
        *,
        tool_choice: Optional[Union[str]] = None,
        **kwargs: Any,
        ):
        self.tools = tools
        self.reasoning_llm_with_tools = self.llm.bind_tools(
            tools, 
            tool_choice=tool_choice, 
            **kwargs
        )
        self.compiled_graph = self.build_graph_with_tools() if tools else self.compiled_graph
        return self

    ###########BASE NODE###########
    async def _base_research_node(self, state: State, prompt_key: str, fallback_prompt: str = None, 
                                 additional_processing: callable = None) -> dict:
        """Base node function for all research steps"""
        system_prompt = get_research_prompt(prompt_key)
        if not system_prompt and fallback_prompt:
            system_prompt = fallback_prompt
        elif not system_prompt:
            system_prompt = f"You are a helpful AI assistant specialized in {prompt_key.replace('_', ' ')}."
        
        messages = state["messages"].copy()
        messages.insert(0, {"role": "system", "content": system_prompt})
        
        # Use tool-enabled LLM if tools are available
        if hasattr(self, 'reasoning_llm_with_tools') and self.reasoning_llm_with_tools and self.tools:
            response = await self.reasoning_llm_with_tools.ainvoke(messages)
        else:
            # Use the appropriate SingleShotAgent
            agent_name = prompt_key
            if hasattr(self, agent_name):
                agent = getattr(self, agent_name)
                if self.tools:
                    agent.bind_tools(self.tools)
                result = await agent.compiled_graph.ainvoke({"messages": messages[1:]})  # Skip system message as agent handles it
                response = result["messages"][-1] if result.get("messages") else None
            else:
                # Fallback to basic LLM
                response = await self.llm.ainvoke(messages)
        
        result = {"messages": [response] if response else []}
        
        # Apply additional processing if provided
        if additional_processing:
            additional_data = additional_processing(response, state)
            result.update(additional_data)
        
        return result

    ###########SPECIALIZED NODES###########
    async def gather_background_knowledge(self, state: State):
        """Gather background knowledge using the structured prompt system"""
        return await self._base_research_node(state, 'background_knowledge_gatherer')

    async def user_intent_analysis(self, state: State):
        """Analyze if the user intent is clear enough to proceed"""
        return await self._base_research_node(state, 'user_intent_analysis')
    
    async def human_clarification_node(self, state: State):
        """Request human clarification if the intent is unclear"""
        return await self._base_research_node(state, 'human_clarification')
    
    async def query_intent_analysis(self, state: State):
        """Analyze the user's query intent to determine next steps"""
        return await self._base_research_node(state, 'query_intent_analysis')
    
    async def gap_analysis(self, state: State):
        """Analyze if there are gaps in the research that need to be filled"""
        return await self._base_research_node(state, 'gap_analysis')

    async def generate_report(self, state: State):
        """Generate the final research report"""
        return await self._base_research_node(state, 'generate_report')
  
    async def gaps_to_subquery(self, state: State):
        """Convert identified gaps into specific research subqueries"""
        def extract_subqueries_processing(response, state):
            subqueries = self._extract_subqueries(response)
            return {
                "subqueries": subqueries,
                "pending_subqueries": subqueries.copy(),
                "completed_subqueries": [],
                "current_subquery_index": 0
            }
        return await self._base_research_node(
            state, 
            'gaps_to_subquery',
            additional_processing=extract_subqueries_processing
        )

    async def subquery_processor(self, state: State):
        """Process all subqueries sequentially or in parallel"""
        subqueries = state.get("subqueries", [])
        
        if not subqueries:
            return {"messages": state["messages"]}
        
        # Process all subqueries
        all_responses = []
        for i, subquery in enumerate(subqueries):
            system_prompt = get_research_prompt('subquery_llm')
            if not system_prompt:
                system_prompt = """You are a Subquery Research Specialist. Your job is to thoroughly research and answer a specific research subquery.

                Use available tools to gather comprehensive information about the given subquery.
                Provide detailed, well-researched answers with supporting evidence.
                
                Focus on:
                1. Factual accuracy
                2. Multiple perspectives
                3. Supporting evidence and sources
                4. Comprehensive coverage of the topic"""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Research this specific question: {subquery}"}
            ]
            
            # Use tool-enabled LLM if tools are available
            if hasattr(self, 'reasoning_llm_with_tools') and self.reasoning_llm_with_tools:
                response = await self.reasoning_llm_with_tools.ainvoke(messages)
            else:
                response = await self.llm.ainvoke(messages)
            
            all_responses.append(response)
        
        # Combine all responses into the message history
        updated_messages = state["messages"].copy()
        for i, (subquery, response) in enumerate(zip(subqueries, all_responses)):
            updated_messages.append({"role": "user", "content": f"Subquery {i+1}: {subquery}"})
            updated_messages.append(response)
        
        return {
            "messages": updated_messages,
            "subqueries": [],  # Clear subqueries after processing
            "pending_subqueries": [],
            "completed_subqueries": subqueries,
            "subquery_results": all_responses
        }

    def build_graph(self):
        self.graph_builder.add_node("gather_background_knowledge", self.gather_background_knowledge)
        self.graph_builder.add_node("user_intent_analysis", self.user_intent_analysis)
        self.graph_builder.add_node("human_clarification", self.human_clarification_node)
        self.graph_builder.add_node("query_intent_analysis", self.query_intent_analysis)
        self.graph_builder.add_node("gap_analysis", self.gap_analysis)
        self.graph_builder.add_node("generate_report", self.generate_report)
        self.graph_builder.add_node("gaps_to_subquery", self.gaps_to_subquery)
        self.graph_builder.add_node("subquery_processor", self.subquery_processor)
        
        # Sequential flow
        self.graph_builder.add_edge(START, "gather_background_knowledge")
        self.graph_builder.add_edge("gather_background_knowledge", "user_intent_analysis")
        self.graph_builder.add_conditional_edges(
            "user_intent_analysis",
            self.should_request_clarification,
            {
                "human_clarification": "human_clarification",
                "query_intent_analysis": "query_intent_analysis"
            }
        )
        self.graph_builder.add_edge("human_clarification", "gather_background_knowledge")
        self.graph_builder.add_edge("query_intent_analysis", "gap_analysis")
        self.graph_builder.add_conditional_edges(
            "gap_analysis",
            self.should_generate_report,
            {
                "generate_report": "generate_report",
                "gaps_to_subquery": "gaps_to_subquery"
            }
        )
        self.graph_builder.add_edge("gaps_to_subquery", "subquery_processor")
        self.graph_builder.add_edge("subquery_processor", "gap_analysis")
        self.graph_builder.add_edge("generate_report", END)

        # Compile with interrupt for human input
        self.compiled_graph = self.graph_builder.compile(
            checkpointer=self.checkpointer,
            interrupt_before=["human_clarification"]  # This will pause for human input
        )
        return self

    ###########HELPER METHODS###########
    def _extract_subqueries(self, response):
        """Extract subqueries from the gaps_to_subquery response"""
        import json
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
        except:
            pass
        
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
        
        return subqueries[:5]  # Limit to 5 subqueries max

    ###########CONDITIONAL FUNCTIONS###########
    def should_request_clarification(self, state: State) -> str:
        """Determine if we need human clarification or can proceed"""
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

    def should_generate_report(self, state: State) -> str:
        """Determine if we should generate report or process gaps"""
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



