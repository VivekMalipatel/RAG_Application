import os
from typing import TypedDict, Annotated, Sequence
import operator

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

# --- Configuration ---
# Replace with the actual base URL of your running ModelRouterAPI
BASE_URL = os.getenv("MODEL_ROUTER_BASE_URL", "http://localhost:8000/v1")
# Replace with your actual API key if needed, or use a test key
API_KEY = os.getenv("MODEL_ROUTER_API_KEY", "test-key")
# Specify the model you want to use (must be available in your ModelRouterAPI)
MODEL_NAME = "meta-llama_Llama-3.1-8B-Instruct_Q8_0" # Or another model like "gpt-3.5-turbo"

print(f"Using ModelRouterAPI at: {BASE_URL}")
print(f"Using Model: {MODEL_NAME}")

# --- LangChain/LangGraph Setup ---

# 1. Define the State for the graph
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

# 2. Define the node(s) for the graph
def call_model(state: AgentState):
    """Calls the LLM using the local API and streams the response."""
    messages = state['messages']
    print(f"\n--- Calling Model ({MODEL_NAME}) ---")
    print(f"Input Messages: {[m.pretty_repr() for m in messages]}")
    
    # Configure ChatOpenAI to use the local API
    llm = ChatOpenAI(
        model=MODEL_NAME,
        openai_api_base=BASE_URL,
        openai_api_key=API_KEY,
        temperature=0.7 # Optional: Adjust temperature
    )
    
    try:
        print("\n--- Streaming Response --- ")
        full_response_content = ""
        final_response_metadata = {}
        # Use the stream method
        for chunk in llm.stream(messages):
            # Print the content chunk
            print(chunk.content, end="", flush=True)
            # Accumulate content
            full_response_content += chunk.content
            # Keep track of response metadata (usually available in the last chunk)
            if chunk.response_metadata:
                 final_response_metadata = chunk.response_metadata

        print("\n--- End of Stream ---")
        
        # Create the final AIMessage with accumulated content
        final_ai_message = AIMessage(
            content=full_response_content,
            response_metadata=final_response_metadata
        )
        
        print(f"\nAggregated Model Response: {final_ai_message.pretty_repr()}")
        # Append the complete response to the messages list for the state
        return {"messages": [final_ai_message]}
    except Exception as e:
        print(f"\nError calling model: {e}")
        # Handle error appropriately
        return state 

# 3. Define the Graph
workflow = StateGraph(AgentState)

# Add the single node for calling the model
workflow.add_node("agent", call_model)

# Set the entry point
workflow.set_entry_point("agent")

# Add a finish edge
workflow.add_edge("agent", END)

# 4. Compile the graph
app = workflow.compile()

# --- Run the Agent ---
if __name__ == "__main__":
    print("\n--- Running LangGraph Agent ---")
    # Use a slightly longer prompt to better see streaming
    initial_input = {"messages": [HumanMessage(content="Explain the concept of RAG in large language models in detail, step by step.")]}
    
    print(f"Initial Input: {initial_input}")
    
    final_state = app.invoke(initial_input)
    
    print("\n--- Final State ---")
    if final_state and 'messages' in final_state:
        print("Messages:")
        for message in final_state['messages']:
            print(f"  - {message.pretty_repr()}")
    else:
        print("No final state or messages found.")
