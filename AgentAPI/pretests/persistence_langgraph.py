from typing import Annotated, Dict, List, Optional, TypedDict, Union, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from operator import add
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
import uuid

# Define state with appropriate reducers
class ChatState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    context: Dict[str, str]
    user_preferences: Dict[str, str]
    current_step: str

# Define nodes for our graph
def greeter_node(state: ChatState) -> ChatState:
    """Initial node that processes user's first message"""
    user_input = state["messages"][-1].content if state["messages"] else ""
    
    response = f"Hello! I've received your message: '{user_input}'"
    
    return {
        "messages": [AIMessage(content=response)],
        "current_step": "greeting_complete"
    }

def preference_extraction_node(state: ChatState, config: Dict) -> ChatState:
    """Extract user preferences from the conversation history"""
    # Access the store if available in config
    store = config.get("store", None)
    user_id = config.get("configurable", {}).get("user_id", "default_user")
    thread_id = config.get("configurable", {}).get("thread_id", "unknown_thread")
    
    messages = state.get("messages", [])
    user_input = messages[-1].content if len(messages) > 0 and isinstance(messages[-1], HumanMessage) else ""
    
    # Simple preference extraction (in real life, use an LLM)
    preferences = {}
    if "like" in user_input.lower():
        # Extract what user likes (simple heuristic)
        liked_item = user_input.lower().split("like")[1].strip().split()[0]
        preferences["likes"] = liked_item
    
    # If we have a store and extracted preferences, save them
    if store and preferences:
        memory_id = str(uuid.uuid4())
        namespace = (user_id, "preferences")
        store.put(namespace, memory_id, {"preferences": preferences})
        
    response = "I've noted your preferences."
    
    return {
        "messages": [AIMessage(content=response)],
        "user_preferences": preferences,
        "current_step": "preferences_extracted"
    }

def memory_lookup_node(state: ChatState, config: Dict) -> ChatState:
    """Look up previous user preferences from memory store"""
    store = config.get("store", None)
    user_id = config.get("configurable", {}).get("user_id", "default_user")
    
    context = {}
    
    if store:
        namespace = (user_id, "preferences")
        try:
            # Get all preferences for this user
            memories = store.search(namespace)
            if memories:
                # Combine all preferences
                all_preferences = {}
                for memory in memories:
                    all_preferences.update(memory.value.get("preferences", {}))
                
                if all_preferences:
                    context["remembered_preferences"] = all_preferences
                    response = f"I remember that you have these preferences: {all_preferences}"
                else:
                    response = "I don't have any preferences stored for you yet."
            else:
                response = "I don't have any preferences stored for you yet."
        except Exception as e:
            response = f"I tried to look up your preferences but encountered an error: {e}"
    else:
        response = "No memory store available to look up your preferences."
    
    return {
        "messages": [AIMessage(content=response)],
        "context": context,
        "current_step": "memory_lookup_complete"
    }

def response_node(state: ChatState) -> ChatState:
    """Generate final response based on all available context"""
    messages = state.get("messages", [])
    user_input = messages[-1].content if len(messages) > 0 and isinstance(messages[-1], HumanMessage) else ""
    preferences = state.get("user_preferences", {})
    context = state.get("context", {})
    
    remembered_preferences = context.get("remembered_preferences", {})
    
    response_parts = []
    response_parts.append(f"In response to: '{user_input}'")
    
    if preferences:
        response_parts.append(f"Based on your current preferences: {preferences}")
    
    if remembered_preferences:
        response_parts.append(f"And your past preferences: {remembered_preferences}")
    
    response_parts.append("Here's my final response!")
    
    response = " ".join(response_parts)
    
    return {
        "messages": [AIMessage(content=response)],
        "current_step": "response_complete"
    }

# Build the graph
def build_graph():
    builder = StateGraph(ChatState)
    
    builder.add_node("greeter", greeter_node)
    builder.add_node("extract_preferences", preference_extraction_node)
    builder.add_node("lookup_memory", memory_lookup_node)
    builder.add_node("respond", response_node)
    
    # Define edges
    builder.add_edge(START, "greeter")
    builder.add_edge("greeter", "extract_preferences")
    builder.add_edge("extract_preferences", "lookup_memory")
    builder.add_edge("lookup_memory", "respond")
    builder.add_edge("respond", END)
    
    # Setup checkpointing and store
    checkpointer = InMemorySaver()
    store = InMemoryStore()
    
    return builder.compile(checkpointer=checkpointer, store=store)

def demonstrate_checkpointing():
    """Show how to use checkpointing and access state after graph execution"""
    graph = build_graph()
    
    # First execution with a thread ID
    user_id = "user_123"
    thread_id = "thread_1"
    config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
    
    print("\n=== First Execution ===")
    result = graph.invoke(
        {"messages": [HumanMessage(content="Hello, I like chocolate")]},
        config=config
    )
    
    # Display the result
    print("\nFinal state of first execution:")
    for msg in result["messages"]:
        print(f"  {msg.type}: {msg.content}")
    
    # Access state history
    print("\nState history checkpoints:")
    history = list(graph.get_state_history(config))
    for i, checkpoint in enumerate(history):
        next_nodes = checkpoint.next if checkpoint.next else "None (END)"
        print(f"  Checkpoint {i}: Next nodes = {next_nodes}")
        print(f"    Current step: {checkpoint.values.get('current_step', 'initial')}")
    
    # Second execution with the same thread ID - continues the conversation
    print("\n=== Second Execution (same thread) ===")
    result2 = graph.invoke(
        {"messages": [HumanMessage(content="What do I like?")]},
        config=config
    )
    
    # Display the result
    print("\nFinal state of second execution:")
    for msg in result2["messages"]:
        print(f"  {msg.type}: {msg.content}")
    
    # Third execution with a different thread but same user
    print("\n=== Third Execution (different thread, same user) ===")
    thread_id_2 = "thread_2"
    config2 = {"configurable": {"thread_id": thread_id_2, "user_id": user_id}}
    
    result3 = graph.invoke(
        {"messages": [HumanMessage(content="Do you know what I like?")]},
        config=config2
    )
    
    # Display the result
    print("\nFinal state of third execution (should remember preferences):")
    for msg in result3["messages"]:
        print(f"  {msg.type}: {msg.content}")
    
    # Demonstrate updating state
    print("\n=== Updating State and Forking ===")
    
    # Update the first thread's state
    graph.update_state(
        config, 
        {"user_preferences": {"likes": "vanilla"}},
        as_node="extract_preferences"
    )
    
    # Get the updated state
    updated_state = graph.get_state(config)
    print(f"\nUpdated state preferences: {updated_state.values.get('user_preferences', {})}")
    
    # Fourth execution - replay from a specific checkpoint
    print("\n=== Replaying from Specific Checkpoint ===")
    
    # Get a checkpoint ID from history (second checkpoint)
    checkpoint_id = history[1].config["configurable"]["checkpoint_id"]
    replay_config = {"configurable": {"thread_id": thread_id, "checkpoint_id": checkpoint_id}}
    
    result4 = graph.invoke(None, config=replay_config)
    
    print(f"\nReplay result (from checkpoint {checkpoint_id}):")
    for msg in result4["messages"]:
        print(f"  {msg.type}: {msg.content}")

if __name__ == "__main__":
    demonstrate_checkpointing()