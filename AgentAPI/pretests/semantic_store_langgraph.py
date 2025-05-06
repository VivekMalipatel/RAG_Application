from typing import Annotated, Dict, List, Optional, TypedDict, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from operator import add
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
import uuid
import os

# Using a mock embedding function for demonstration purposes
# In a real application, you would use a proper embedding model
def mock_embedding_function(texts):
    """Simple mock embedding function that returns random vectors"""
    import numpy as np
    return [np.random.rand(384) for _ in texts]

class SemanticMemoryState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    facts: List[Dict[str, str]]
    retrieved_memories: List[Dict]
    response: str

def extract_facts_node(state: SemanticMemoryState) -> SemanticMemoryState:
    """Extract facts from user message"""
    user_input = state["messages"][-1].content if state["messages"] else ""
    
    # In a real application, you would use an LLM to extract facts
    facts = []
    
    # Simple rule-based fact extraction for demonstration
    if "my name is" in user_input.lower():
        name = user_input.lower().split("my name is")[1].strip().split()[0].capitalize()
        facts.append({"type": "personal_info", "attribute": "name", "value": name})
        
    if "i live in" in user_input.lower():
        location = user_input.lower().split("i live in")[1].strip().split()[0].capitalize()
        facts.append({"type": "personal_info", "attribute": "location", "value": location})
        
    if "i like" in user_input.lower():
        preference = user_input.lower().split("i like")[1].strip().split()[0]
        facts.append({"type": "preference", "attribute": "likes", "value": preference})
        
    if "i hate" in user_input.lower() or "i don't like" in user_input.lower():
        phrase = "i hate" if "i hate" in user_input.lower() else "i don't like"
        dislike = user_input.lower().split(phrase)[1].strip().split()[0]
        facts.append({"type": "preference", "attribute": "dislikes", "value": dislike})
    
    return {"facts": facts}

def store_memories_node(state: SemanticMemoryState, config: Dict) -> SemanticMemoryState:
    """Store extracted facts in the memory store with semantic indexing"""
    store = config.get("store", None)
    user_id = config.get("configurable", {}).get("user_id", "default_user")
    
    facts = state.get("facts", [])
    if not store or not facts:
        return {}
    
    namespace = (user_id, "semantic_memories")
    
    # Store each fact as a separate memory with semantic indexing
    for fact in facts:
        memory_id = str(uuid.uuid4())
        fact_text = f"{fact['attribute']}: {fact['value']}"
        
        # Store the memory with the fact text indexed for semantic search
        store.put(
            namespace, 
            memory_id, 
            {
                "fact_type": fact["type"],
                "attribute": fact["attribute"],
                "value": fact["value"],
                "fact_text": fact_text
            }
        )
    
    response = f"I've stored {len(facts)} memories." if facts else "No new information to store."
    
    return {
        "messages": [AIMessage(content=response)]
    }

def retrieve_memories_node(state: SemanticMemoryState, config: Dict) -> SemanticMemoryState:
    """Retrieve relevant memories based on user query using semantic search"""
    store = config.get("store", None)
    user_id = config.get("configurable", {}).get("user_id", "default_user")
    
    user_input = state["messages"][-1].content if state["messages"] else ""
    
    if not store:
        return {"retrieved_memories": [], "messages": [AIMessage(content="No memory store available.")]}
    
    namespace = (user_id, "semantic_memories")
    retrieved_memories = []
    
    try:
        # Use the store's search function to find semantically similar memories
        # In a real application, this would search based on embeddings
        memories = store.search(namespace, query=user_input, limit=5)
        
        if memories:
            retrieved_memories = [memory.value for memory in memories]
            
            # For demonstration, show what was retrieved
            memory_texts = [f"{memory.value['attribute']}: {memory.value['value']}" for memory in memories]
            response = f"I found these memories related to your query: {', '.join(memory_texts)}"
        else:
            response = "I don't have any memories related to your query."
    except Exception as e:
        response = f"Error retrieving memories: {str(e)}"
    
    return {
        "retrieved_memories": retrieved_memories,
        "messages": [AIMessage(content=response)]
    }

def generate_response_node(state: SemanticMemoryState) -> SemanticMemoryState:
    """Generate a response using retrieved memories"""
    user_input = state["messages"][-1].content if state["messages"] else ""
    retrieved_memories = state.get("retrieved_memories", [])
    
    if not retrieved_memories:
        response = f"I don't have any relevant information about that. Can you tell me more?"
    else:
        # Group memories by type
        personal_info = [m for m in retrieved_memories if m.get("fact_type") == "personal_info"]
        preferences = [m for m in retrieved_memories if m.get("fact_type") == "preference"]
        
        response_parts = []
        
        if "what do you know about me" in user_input.lower():
            if personal_info:
                info_parts = [f"Your {m['attribute']} is {m['value']}" for m in personal_info]
                response_parts.append(f"Here's what I know about you: {'. '.join(info_parts)}.")
            else:
                response_parts.append("I don't know much about you yet.")
                
            if preferences:
                like_items = [m['value'] for m in preferences if m['attribute'] == 'likes']
                dislike_items = [m['value'] for m in preferences if m['attribute'] == 'dislikes']
                
                if like_items:
                    response_parts.append(f"You like: {', '.join(like_items)}.")
                if dislike_items:
                    response_parts.append(f"You don't like: {', '.join(dislike_items)}.")
        else:
            # Respond based on the most relevant retrieved memory
            if retrieved_memories:
                memory = retrieved_memories[0]
                response_parts.append(f"Based on what I know about your {memory['attribute']} ({memory['value']}), I can help with that.")
    
        response = " ".join(response_parts) if response_parts else "I've considered what I know about you in my response."
    
    return {
        "response": response,
        "messages": [AIMessage(content=response)]
    }

def build_semantic_memory_graph():
    """Build the semantic memory graph with checkpointing and memory store"""
    builder = StateGraph(SemanticMemoryState)
    
    # Add nodes
    builder.add_node("extract_facts", extract_facts_node)
    builder.add_node("store_memories", store_memories_node)
    builder.add_node("retrieve_memories", retrieve_memories_node)
    builder.add_node("generate_response", generate_response_node)
    
    # Add edges
    builder.add_edge(START, "extract_facts")
    builder.add_edge("extract_facts", "store_memories")
    builder.add_edge("store_memories", "retrieve_memories")
    builder.add_edge("retrieve_memories", "generate_response")
    builder.add_edge("generate_response", END)
    
    # Create checkpointer and memory store
    checkpointer = InMemorySaver()
    
    # Create a memory store with mock embeddings for semantic search
    # In a real application, you would use a real embedding model
    # Example with OpenAI embeddings (commented out since we're using a mock):
    # from langchain_openai import OpenAIEmbeddings
    # store = InMemoryStore(
    #     index={
    #         "embed": OpenAIEmbeddings(),
    #         "dims": 1536,
    #         "fields": ["fact_text", "$"]
    #     }
    # )
    
    # Using mock embeddings for demonstration
    store = InMemoryStore(
        index={
            "embed": mock_embedding_function,
            "dims": 384,
            "fields": ["fact_text", "$"]  # Index fact_text field and fallback to all content
        }
    )
    
    return builder.compile(checkpointer=checkpointer, store=store)

def demonstrate_semantic_memory():
    """Show how semantic memory works with LangGraph"""
    graph = build_semantic_memory_graph()
    
    user_id = "user_456"
    thread_id = "semantic_thread_1"
    config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
    
    print("\n=== First Interaction - Storing Facts ===")
    messages = [HumanMessage(content="Hi, my name is Alex. I live in Boston and I like hiking.")]
    result1 = graph.invoke({"messages": messages}, config=config)
    
    print("\nResult after storing facts:")
    for msg in result1.get("messages", []):
        if hasattr(msg, "content"):
            print(f"  {msg.type}: {msg.content}")
    
    print("\n=== Second Interaction - Querying Memory ===")
    messages = [HumanMessage(content="What do you know about me?")]
    result2 = graph.invoke({"messages": messages}, config=config)
    
    print("\nResult of memory query:")
    for msg in result2.get("messages", []):
        if hasattr(msg, "content"):
            print(f"  {msg.type}: {msg.content}")
    
    print("\n=== Third Interaction - Adding More Facts ===")
    messages = [HumanMessage(content="I don't like cold weather and I like coffee.")]
    result3 = graph.invoke({"messages": messages}, config=config)
    
    print("\nResult after adding more facts:")
    for msg in result3.get("messages", []):
        if hasattr(msg, "content"):
            print(f"  {msg.type}: {msg.content}")
    
    print("\n=== Fourth Interaction - Complete Memory Query ===")
    messages = [HumanMessage(content="Tell me everything you know about me.")]
    result4 = graph.invoke({"messages": messages}, config=config)
    
    print("\nFinal memory summary:")
    for msg in result4.get("messages", []):
        if hasattr(msg, "content"):
            print(f"  {msg.type}: {msg.content}")
    
    # New thread, same user - memories should persist across threads
    print("\n=== New Thread, Same User ===")
    new_thread_id = "semantic_thread_2"
    new_config = {"configurable": {"thread_id": new_thread_id, "user_id": user_id}}
    
    messages = [HumanMessage(content="Do you remember who I am?")]
    result5 = graph.invoke({"messages": messages}, config=new_config)
    
    print("\nMemory persistence across threads:")
    for msg in result5.get("messages", []):
        if hasattr(msg, "content"):
            print(f"  {msg.type}: {msg.content}")

if __name__ == "__main__":
    demonstrate_semantic_memory()