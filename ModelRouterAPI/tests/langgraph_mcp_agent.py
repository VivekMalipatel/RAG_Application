import os
import json
import asyncio
from typing import Annotated, Sequence, TypedDict, Dict, Any, List
import operator

# Import LangGraph components
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# Import LangChain components
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# Import MCP adapter - using MultiServerMCPClient instead of SingleServerMCPClient
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.adapters import adapter_for_messages

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BASE_URL = os.getenv("MODEL_ROUTER_BASE_URL", "http://localhost:8000/v1")
API_KEY = os.getenv("MODEL_ROUTER_API_KEY", "test-key")
MODEL_NAME = os.getenv("MODEL_ROUTER_MODEL", "meta-llama_Llama-3.1-8B-Instruct_Q8_0")

print(f"Using ModelRouterAPI at: {BASE_URL}")
print(f"Using Model: {MODEL_NAME}")

# -----------------------------------------------------------------------------
# Agent State Definition
# -----------------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]  # For chat history
    context: Dict[str, Any]  # For additional context

# -----------------------------------------------------------------------------
# Tool Definitions
# -----------------------------------------------------------------------------
# Define the MCP-compatible tools
MCP_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Calculate the result of a mathematical expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]

# Define tool implementations
tool_implementations = {
    "web_search": lambda query: f"Here are search results for '{query}': [Simulated search results about {query}]",
    "calculate": lambda expression: f"The result of {expression} is {eval(expression)}"
}

# -----------------------------------------------------------------------------
# MCP Client Setup
# -----------------------------------------------------------------------------
def setup_mcp_client():
    """Setup the MCP client to connect to the ModelRouterAPI."""
    # Configure the MultiServerMCPClient with a single server named "modelrouter"
    mcp_client = MultiServerMCPClient({
        "modelrouter": {
            "url": BASE_URL,
            "transport": "http",  # Using HTTP transport for REST API
            "headers": {"Authorization": f"Bearer {API_KEY}"},
        }
    })
    
    # Register the custom tools with the client
    for tool in MCP_TOOLS:
        tool_name = tool["function"]["name"]
        mcp_client.register_tool("modelrouter", tool)
        
        # Register tool implementation if available
        if tool_name in tool_implementations:
            mcp_client.register_tool_implementation(tool_name, tool_implementations[tool_name])
    
    return mcp_client

# -----------------------------------------------------------------------------
# Agent Creation
# -----------------------------------------------------------------------------
async def create_mcp_agent():
    """Create an agent that uses MCP for tools."""
    # Setup the MCP client
    mcp_client = setup_mcp_client()
    
    # Create LLM adapter with our ModelRouterAPI
    llm = ChatOpenAI(
        model=MODEL_NAME,
        openai_api_base=BASE_URL,
        openai_api_key=API_KEY,
        temperature=0
    )
    
    # Custom agent prompt
    agent_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant with access to tools. 
Use the tools when necessary to provide accurate information.
Think step by step and reason about when to use tools."""),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Get tools from MCP client
    async with mcp_client:
        tools = await mcp_client.aget_tools()
    
    # Create the React agent using the tools
    agent = create_react_agent(llm, tools, agent_prompt)
    
    return agent, mcp_client

# -----------------------------------------------------------------------------
# LangGraph Setup
# -----------------------------------------------------------------------------
async def agent_node(state: AgentState):
    """Node that processes user input and decides actions."""
    # Get agent instance (in real application, would be better to create once and reuse)
    agent, mcp_client = await create_mcp_agent()
    
    # Get the messages from the state
    messages = state["messages"]
    
    # Invoke the agent
    try:
        async with mcp_client:
            response = await agent.ainvoke({"messages": messages})
        
        # Add the agent's response to the messages
        return {"messages": [response]}
    except Exception as e:
        error_msg = f"Error in agent: {str(e)}"
        print(error_msg)
        return {"messages": [AIMessage(content=f"I encountered an error: {error_msg}")]}

# -----------------------------------------------------------------------------
# Graph Construction
# -----------------------------------------------------------------------------
def create_agent_graph():
    """Creates and returns the agent graph."""
    # Initialize graph
    workflow = StateGraph(AgentState)
    
    # Add nodes - with MCP, we only need the agent node as tool execution
    # is handled internally by the React agent pattern
    workflow.add_node("agent", agent_node)
    
    # Add edges
    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", END)
    
    # Create checkpointer for state persistence
    memory = MemorySaver()
    
    # Compile graph
    return workflow.compile(checkpointer=memory)

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
async def main():
    # Create graph
    print("Creating MCP Agent Graph...")
    graph = create_agent_graph()
    
    # Initialize state
    initial_state = {
        "messages": [HumanMessage(content="I need to know the capital of France, and then calculate 15 * 24.")],
        "context": {"session_id": "test-session-1"}
    }
    
    # Set up config with thread ID for persistence
    config = {"configurable": {"thread_id": "mcp_test_thread_1"}}
    
    print("\n=== Starting MCP Agent Execution ===")
    
    try:
        # Stream execution events
        for event in graph.stream(initial_state, config):
            for node_name, node_output in event.items():
                print(f"\n--- Output from {node_name} ---")
                if "messages" in node_output:
                    for msg in node_output["messages"]:
                        print(f"{msg.type}: {msg.content}")
        
        print("\n=== Final State ===")
        final_state = graph.get_state(config)
        print("Conversation History:")
        for msg in final_state.values.get("messages", []):
            print(f"{msg.type}: {msg.content}")
            
    except Exception as e:
        print(f"Error during execution: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())