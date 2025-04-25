import os
import operator
from typing import TypedDict, Annotated, Sequence, Literal

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.tools import tool, ToolInvocation
from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.tools.base import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# --- Configuration ---
BASE_URL = os.getenv("MODEL_ROUTER_BASE_URL", "http://localhost:8000/v1")
API_KEY = os.getenv("MODEL_ROUTER_API_KEY", "test-key")
MODEL_NAME = os.getenv("MODEL_ROUTER_MODEL", "meta-llama_Llama-3.1-8B-Instruct_Q8_0") # Or your preferred model

print(f"Using ModelRouterAPI at: {BASE_URL}")
print(f"Using Model: {MODEL_NAME}")

# --- Tools ---
@tool
def get_weather(city: str):
    """Gets the current weather for a specified city."""
    print(f"--- Calling get_weather tool for city: {city} ---")
    # In a real scenario, this would call a weather API
    if "san francisco" in city.lower():
        return "The weather in San Francisco is 65 degrees and sunny."
    elif "new york" in city.lower():
        return "The weather in New York is 75 degrees and cloudy."
    else:
        return f"Sorry, I don't have weather information for {city}."

# Tool executor
tools = [get_weather]

# --- LLM Configuration ---
# Use the local ModelRouterAPI endpoint
llm = ChatOpenAI(
    model=MODEL_NAME,
    openai_api_base=BASE_URL,
    openai_api_key=API_KEY,
    temperature=0,
    streaming=False # Streaming can be complex with tool use in basic examples
)

# Bind tools to the LLM for Agent 1
llm_with_tools = llm.bind_tools(tools)

# --- Agent State ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

# --- Agent Nodes ---

def agent_1_node(state: AgentState):
    """Agent 1: Calls LLM with tools, decides next step."""
    print("--- Agent 1 Turn ---")
    response = llm_with_tools.invoke(state['messages'])
    print(f"Agent 1 Raw Response: {response}")
    # If tool call, do not add to messages yet, tool node will handle it
    if not response.tool_calls:
        print("Agent 1 Responding Directly.")
    else:
        print("Agent 1 Requesting Tool Call.")
    # Response (AIMessage with or without tool_calls) is returned to be added to state
    return {"messages": [response]}

def agent_2_node(state: AgentState):
    """Agent 2: Responds to the previous message."""
    print("--- Agent 2 Turn ---")
    # Get the last message (which should be Agent 1's final response)
    last_message = state['messages'][-1]
    print(f"Agent 2 Received: {last_message.pretty_repr()}")
    
    # Create a new context for Agent 2
    agent_2_context = [
        HumanMessage(content=f"Respond to this statement: {last_message.content}")
    ]
    
    response = llm.invoke(agent_2_context)
    print(f"Agent 2 Response: {response.pretty_repr()}")
    return {"messages": [response]}

def tool_node(state: AgentState):
    """Executes tools called by Agent 1."""
    print("--- Tool Execution Turn ---")
    last_message = state['messages'][-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        print("No tool call found.")
        return # Should not happen in expected flow

    tool_messages = []
    for tool_call in last_message.tool_calls:
        # Direct tool execution without ToolExecutor
        tool_result = get_weather(tool_call["args"])
        tool_messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"]))
        print(f"Tool Result ({tool_call['name']}): {tool_result}")

    return {"messages": tool_messages}

# --- Conditional Logic ---

def should_continue(state: AgentState) -> Literal["agent_1", "agent_2", "tool_node", "__end__"]:
    """Determines the next node to run."""
    last_message = state['messages'][-1]
    
    if isinstance(last_message, AIMessage):
        if last_message.tool_calls:
            print("Decision: Route to Tool Node")
            return "tool_node" # Agent 1 called a tool
        else:
            # Agent 1 responded directly, or Agent 2 responded
            # Simple logic: If last message is AI and not tool call, assume it's Agent 1's final response -> Agent 2
            # Or if it's Agent 2's response -> End
            # Let's refine: Check sender if we add that to state, or count turns.
            # For simplicity: If AI message without tool call, assume Agent 1 finished, go to Agent 2.
            # If the *second to last* message was a ToolMessage, it means Agent 1 just processed the tool result, go to Agent 2.
            if len(state['messages']) > 1 and isinstance(state['messages'][-2], ToolMessage):
                 print("Decision: Route to Agent 2 (after tool use)")
                 return "agent_2"
            # If the last message is AI, no tool call, and previous wasn't ToolMessage, assume Agent 1 responded directly.
            elif len(state['messages']) == 2: # Initial Human + First AI response
                 print("Decision: Route to Agent 2 (direct response)")
                 return "agent_2"
            else: # Assume Agent 2 just responded
                 print("Decision: End Graph")
                 return "__end__"
                 
    elif isinstance(last_message, ToolMessage):
        print("Decision: Route back to Agent 1 (after tool execution)")
        return "agent_1" # Tool executed, return to Agent 1 to process result
        
    else: # HumanMessage (initial input)
        print("Decision: Route to Agent 1 (start)")
        return "agent_1"


# --- Build Graph ---
workflow = StateGraph(AgentState)

workflow.add_node("agent_1", agent_1_node)
workflow.add_node("agent_2", agent_2_node)
workflow.add_node("tool_node", tool_node)

workflow.set_entry_point("agent_1")

workflow.add_conditional_edges(
    "agent_1",
    # Logic based on whether the AI Message has tool calls
    lambda state: "tool_node" if isinstance(state['messages'][-1], AIMessage) and state['messages'][-1].tool_calls else "agent_2",
    {
        "tool_node": "tool_node",
        "agent_2": "agent_2",
    }
)

workflow.add_edge("tool_node", "agent_1") # After tool execution, go back to agent 1
workflow.add_edge("agent_2", END) # Agent 2 is the last step

# Compile the graph
app = workflow.compile()

# --- Run the Agent ---
if __name__ == "__main__":
    print("\n--- Running Multi-Agent Graph ---")
    initial_input = {"messages": [HumanMessage(content="Hi Agent 1, what's the weather like in San Francisco?")]}
    
    print(f"Initial Input: {initial_input}")
    
    # Stream events to see the flow
    for event in app.stream(initial_input, {"recursion_limit": 10}):
        for node, output in event.items():
            print(f"--- Output from node: {node} ---")
            # print(output) # Can be verbose
            if 'messages' in output:
                 print(f"Messages: {[m.pretty_repr() for m in output['messages']]}")
        print("\n---\n")

    print("\n--- Final State ---")
    final_state = app.invoke(initial_input, {"recursion_limit": 10})
    if final_state and 'messages' in final_state:
        print("Final Messages:")
        for message in final_state['messages']:
            print(f"  - {message.pretty_repr()}")
    else:
        print("No final state or messages found.")

