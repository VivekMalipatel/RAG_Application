import os
from typing import Annotated, Sequence, TypedDict, Literal, Dict, Any
import operator
import json
from datetime import datetime

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

# --- Configuration ---
BASE_URL = os.getenv("MODEL_ROUTER_BASE_URL", "http://localhost:8000/v1")
API_KEY = os.getenv("MODEL_ROUTER_API_KEY", "test-key")
MODEL_NAME = os.getenv("MODEL_ROUTER_MODEL", "meta-llama_Llama-3.1-8B-Instruct_Q8_0")

print(f"Using ModelRouterAPI at: {BASE_URL}")
print(f"Using Model: {MODEL_NAME}")

# --- Custom State Definition ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]  # For chat history
    current_task: str  # Track current task
    task_status: str  # Track task status
    research_findings: Dict[str, Any]  # Store research data
    last_human_feedback: str  # Store last human feedback

# --- Tools Definition ---
@tool
def web_search(query: str) -> str:
    """Simulated web search tool."""
    print(f"Performing web search for: {query}")
    return f"Here are the search results for '{query}': [Simulated web search results]"

@tool
def save_research(topic: str, findings: str, tool_call_id: str) -> str:
    """Save research findings to state."""
    print(f"Saving research on topic: {topic}")
    # We'll use Command to update the state
    state_update = {
        "research_findings": {topic: findings},
        "messages": [ToolMessage(content=f"Saved research on {topic}", tool_call_id=tool_call_id)]
    }
    return Command(update=state_update)

@tool
def request_human_feedback(query: str, context: str, tool_call_id: str) -> str:
    """Request feedback from a human."""
    print(f"\nRequesting human feedback for: {query}")
    print(f"Context: {context}")
    
    # Interrupt execution to get human input
    human_response = interrupt({
        "query": query,
        "context": context,
        "instructions": "Please provide feedback or guidance."
    })
    
    # Update state with human feedback
    state_update = {
        "last_human_feedback": human_response.get("feedback", "No feedback provided"),
        "messages": [ToolMessage(
            content=f"Human feedback received: {human_response.get('feedback', 'No feedback provided')}", 
            tool_call_id=tool_call_id
        )]
    }
    return Command(update=state_update)

# --- Agent Nodes ---
def agent_node(state: AgentState):
    """Main agent node that processes input and decides actions."""
    messages = state["messages"]
    last_message = messages[-1] if messages else None
    
    # Initialize LLM with tools
    llm = ChatOpenAI(
        model=MODEL_NAME,
        openai_api_base=BASE_URL,
        openai_api_key=API_KEY,
        temperature=0
    )
    llm_with_tools = llm.bind_tools([web_search, save_research, request_human_feedback])
    
    # Process with LLM
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def tool_processing_node(state: AgentState):
    """Handles tool execution."""
    print("--- Tool Processing Node ---")
    last_message = state["messages"][-1]
    
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        print("No tool calls found.")
        return state
        
    tool_node = ToolNode(tools=[web_search, save_research, request_human_feedback])
    return tool_node(state)

def human_review_node(state: AgentState):
    """Optional human review of agent's work."""
    print("--- Human Review Node ---")
    last_message = state["messages"][-1]
    
    # Ask for human review
    human_response = interrupt({
        "query": "Would you like to review this response?",
        "response": last_message.content if hasattr(last_message, "content") else str(last_message),
        "instructions": "Type 'approve' to continue or provide feedback"
    })
    
    if human_response.get("feedback", "").lower() != "approve":
        return {
            "messages": [HumanMessage(content=f"Human feedback: {human_response.get('feedback')}")],
            "last_human_feedback": human_response.get("feedback")
        }
    return state

# --- Edge Conditions ---
def should_use_tool(state: AgentState) -> Literal["tool_node", "human_review", END]:
    """Determines if we should use a tool."""
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_node"
    return "human_review"

def should_continue(state: AgentState) -> Literal["agent", "human_review", END]:
    """Determines if we should continue processing."""
    last_message = state["messages"][-1]
    if isinstance(last_message, HumanMessage):
        return "agent"
    return END

# --- Graph Construction ---
def create_agent_graph():
    """Creates and returns the agent graph."""
    # Initialize graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tool_node", tool_processing_node)
    workflow.add_node("human_review", human_review_node)
    
    # Add edges
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        should_use_tool,
        {
            "tool_node": "tool_node",
            "human_review": "human_review",
            END: END
        }
    )
    workflow.add_edge("tool_node", "agent")
    workflow.add_conditional_edges(
        "human_review",
        should_continue,
        {
            "agent": "agent",
            END: END
        }
    )
    
    # Create checkpointer for state persistence
    memory = MemorySaver()
    
    # Compile graph
    return workflow.compile(checkpointer=memory)

# --- Main Execution ---
if __name__ == "__main__":
    # Create graph
    graph = create_agent_graph()
    
    # Initialize state
    initial_state = {
        "messages": [HumanMessage(content="Research the latest developments in AI and get human feedback on your findings.")],
        "current_task": "AI research",
        "task_status": "starting",
        "research_findings": {},
        "last_human_feedback": ""
    }
    
    # Set up config with thread ID for persistence
    config = {"configurable": {"thread_id": "research_thread_1"}}
    
    print("\n=== Starting Agent Execution ===")
    
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
        print("Research Findings:", final_state.values.get("research_findings", {}))
        print("Last Human Feedback:", final_state.values.get("last_human_feedback", ""))
        
        print("\n=== Time Travel Demo ===")
        print("Available checkpoints:")
        for state in graph.get_state_history(config):
            print(f"Checkpoint ID: {state.config['configurable'].get('checkpoint_id')}")
            print(f"Number of messages: {len(state.values['messages'])}")
            print("-" * 50)
            
    except Exception as e:
        print(f"Error during execution: {str(e)}")