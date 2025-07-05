from typing import Dict, Any, Annotated
from openai import AsyncOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from app.agents.base_agent import BaseAgent
from app.core.config import settings
from app.tools import tool_registry
import json
import re

# ===== State Definition =====
class State(TypedDict):
    messages: Annotated[list, add_messages]

# ===== Main Agent Class =====
class SingleShotAgent(BaseAgent):
    def __init__(self, prompt: str = "", tools: list = []):        
        # Core LLM setup
        self.client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_API_BASE
        )
        self.model = settings.DEFAULT_MODEL
        self.checkpointer = InMemorySaver()
        
        # Agent configuration
        self.available_tools = tools
        
        # Build the LangGraph
        self.graph = self._build_graph()

    # ===== Graph Construction =====
    def _build_graph(self):
        """Build the single shot graph with conditional tool usage"""
        graph_builder = StateGraph(State)
        
        # Add nodes
        graph_builder.add_node("llm_response", self._llm_response)
        graph_builder.add_node("analyze_response", self._analyze_response)
        graph_builder.add_node("execute_tools", self._execute_tools)
        
        # Add edges
        graph_builder.add_edge(START, "llm_response")
        graph_builder.add_edge("llm_response", "analyze_response")
        
        # Conditional edges based on tool analysis
        graph_builder.add_conditional_edges(
            "analyze_response",
            self._should_use_tools,
            {
                "use_tools": "execute_tools",
                "end": END
            }
        )
        
        # After tools, loop back to LLM for final response
        graph_builder.add_edge("execute_tools", "llm_response")
        
        return graph_builder.compile(checkpointer=self.checkpointer)

    def answer(state: State):
        system_prompt = """You are an answer agent. Your job is to answer the user's query based on the planning steps provided. You will be given a query and a list of planning steps, and you should return the final answer to the query."""
        messages = [{"role": "system", "content": system_prompt}] + state["messages"]

        messages.append({"role": "user", "content": f"Based on the planning steps, Answer/solve the user's query."})

        response = llm.invoke(messages)

        return {"messages": [response]}

    def _analyze_response(self, state: State):
        """Analyze if tools are needed based on the last response"""
        # Get the last assistant message
        last_response = self._get_last_assistant_message(state)
        
        # Reset tool state
        self.current_tool_calls = []
        self.tools_needed = False
        
        # Only analyze if tools are available and enabled
        if not self.available_tools or not self.use_tools_enabled:
            return {"messages": []}
        
        # Check if tools are needed and extract calls
        if self._response_needs_tools(last_response):
            self.current_tool_calls = self._extract_tool_calls(last_response)
            self.tools_needed = len(self.current_tool_calls) > 0
        
        return {"messages": []}

    def _should_use_tools(self, state: SingleShotState) -> str:
        """Determine if tools should be used based on current state"""
        has_tools = self.tools_needed
        within_limits = self.iteration_count < self.max_iterations
        tools_enabled = self.use_tools_enabled
        
        if has_tools and within_limits and tools_enabled:
            return "use_tools"
        else:
            return "end"

    def _execute_tools(self, state: SingleShotState):
        """Execute the detected tools and add results to messages"""
        if not self.current_tool_calls:
            return {"messages": []}
        
        # Execute tools and collect results
        tool_results = []
        for tool_call in self.current_tool_calls:
            result = self._execute_single_tool(tool_call)
            tool_results.append(result)
        
        # Format results and add to messages
        results_message = self._format_tool_results(tool_results)
        
        # Update iteration count and reset tool state
        self.iteration_count += 1
        self.current_tool_calls = []
        self.tools_needed = False
        
        return {
            "messages": [{
                "role": "system",
                "content": f"Tool Results:\n{results_message}"
            }]
        }

    # ===== System Prompt & Message Handling =====
    def _get_system_prompt(self) -> str:
        """Build system prompt with available tools"""
        base_prompt = self.get_system_prompt()
        if not base_prompt.strip():
            base_prompt = "You are an answer agent. Your job is to answer the user's query directly and comprehensively."
        
        # Add tools information if available
        if self.available_tools:
            tools_info = f"\n\nAvailable tools: {', '.join(self.available_tools)}"
            tools_info += "\nMention specific tools if you need to use them for better results."
            base_prompt += tools_info
        
        return base_prompt

    def _get_last_assistant_message(self, state: SingleShotState) -> str:
        """Extract the last assistant message content"""
        if not state["messages"]:
            return ""
        
        last_message = state["messages"][-1]
        if isinstance(last_message, dict) and last_message.get("role") == "assistant":
            return last_message.get("content", "")
        
        return ""

    # ===== Tool Analysis & Detection =====
    def _response_needs_tools(self, response: str) -> bool:
        """Analyze if response indicates tools are needed"""
        response_lower = response.lower()
        
        # Tool usage indicators
        tool_indicators = [
            "search", "find", "look up", "get information", "check online",
            "scrape", "fetch", "retrieve", "download", "browse",
            "need to search", "let me search", "i'll look for", "i need to find"
        ]
        
        # Check if response mentions available tools or indicators
        mentions_tools = any(tool.lower() in response_lower for tool in self.available_tools)
        mentions_indicators = any(indicator in response_lower for indicator in tool_indicators)
        
        return mentions_tools or mentions_indicators

    def _extract_tool_calls(self, response: str) -> list:
        """Extract specific tool calls from response"""
        tool_calls = []
        
        for tool_name in self.available_tools:
            if tool_name.lower() in response.lower():
                if tool_name == "web_search":
                    query = self._extract_search_query(response)
                    if query:
                        tool_calls.append({
                            "tool": tool_name,
                            "parameters": {"query": query, "num_results": 5}
                        })
                
                elif tool_name == "web_scraping":
                    url = self._extract_url(response)
                    if url:
                        tool_calls.append({
                            "tool": tool_name,
                            "parameters": {"url": url, "extract_text": True}
                        })
        
        return tool_calls

    # ===== Parameter Extraction =====
    def _extract_search_query(self, response: str) -> str:
        """Extract search query from response"""
        # Look for quoted strings
        quoted_matches = re.findall(r'"([^"]+)"', response)
        if quoted_matches:
            return quoted_matches[0]
        
        # Look for search patterns
        patterns = [
            r"search for (.*?)(?:\.|$|,)",
            r"look up (.*?)(?:\.|$|,)",
            r"find (.*?)(?:\.|$|,)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ""

    def _extract_url(self, response: str) -> str:
        """Extract URL from response"""
        url_pattern = r'https?://[^\s]+'
        urls = re.findall(url_pattern, response)
        return urls[0] if urls else ""

    # ===== Tool Execution =====
    def _execute_single_tool(self, tool_call: dict) -> dict:
        """Execute a single tool call"""
        try:
            tool_name = tool_call["tool"]
            parameters = tool_call["parameters"]
            
            # Get tool from registry
            tool = tool_registry.get_tool(tool_name)
            if not tool:
                return {"success": False, "error": f"Tool '{tool_name}' not found"}
            
            # Execute tool (this would be async in real implementation)
            # For now, return a mock result
            return {
                "success": True,
                "result": f"Mock result for {tool_name} with params {parameters}",
                "tool": tool_name
            }
            
        except Exception as e:
            return {"success": False, "error": str(e), "tool": tool_call.get("tool", "unknown")}

    def _format_tool_results(self, tool_results: list) -> str:
        """Format tool execution results"""
        if not tool_results:
            return "No tool results available."
        
        formatted = []
        for i, result in enumerate(tool_results, 1):
            tool_name = result.get("tool", f"Tool {i}")
            if result.get("success"):
                formatted.append(f"{tool_name}: {result.get('result', 'Success')}")
            else:
                formatted.append(f"{tool_name}: Error - {result.get('error', 'Unknown error')}")
        
        return "\n".join(formatted)

    # ===== Public Interface =====
    async def execute(self, query: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute single shot response with conditional tool usage"""
        try:
            # Setup configuration
            thread_id = config.get("thread_id", "default")
            self.available_tools = config.get("tools", [])
            self.use_tools_enabled = config.get("use_tools", True)
            self.iteration_count = 0
            
            graph_config = {"configurable": {"thread_id": thread_id}}
            
            # Initial state with only messages
            initial_state = {
                "messages": [{"role": "user", "content": query}]
            }
            
            # Execute graph
            result = await self.graph.ainvoke(initial_state, graph_config)
            
            # Extract final response
            final_message = result["messages"][-1]
            response_text = final_message["content"] if isinstance(final_message, dict) else final_message.content
            
            # Build result data
            result_data = {
                "response": response_text,
                "agent_type": "single_shot",
                "thread_id": thread_id,
                "status": "completed",
                "iterations": self.iteration_count,
                "tools_available": self.available_tools,
                "tools_used": len([msg for msg in result["messages"] if msg.get("role") == "system" and "Tool Results" in msg.get("content", "")])
            }
            
            self.log_execution(query, response_text, config)
            return result_data
            
        except Exception as e:
            self.logger.error(f"Error in SingleShotAgent execution: {str(e)}")
            return {
                "response": f"Error: {str(e)}",
                "agent_type": "single_shot",
                "thread_id": config.get("thread_id"),
                "status": "error"
            }
