from typing import Annotated, Dict, List, Literal, Optional, TypedDict, Union
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from operator import add
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage

# Subgraph State Schema
class ResearchState(TypedDict):
    query: str
    facts: List[str]

# Main Graph State Schema
class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    query: str
    facts: List[str]
    needs_approval: bool
    approved: bool
    response: str

# Input State Schema
class InputState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]

# Output State Schema
class OutputState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    response: str
    
# Private State Schema
class PrivateState(TypedDict):
    thinking_process: str

# Subgraph node
def research_node(state: ResearchState) -> ResearchState:
    query = state["query"]
    facts = [
        f"LangGraph fact 1 about {query}",
        f"LangGraph fact 2 about {query}",
        f"LangGraph fact 3 about {query}"
    ]
    return {"facts": facts}

# Build research subgraph
def build_research_subgraph():
    research_builder = StateGraph(ResearchState)
    research_builder.add_node("research", research_node)
    research_builder.add_edge(START, "research")
    research_builder.add_edge("research", END)
    return research_builder.compile()

# Main graph nodes
def extract_query_node(state: AgentState) -> PrivateState:
    user_input = state["messages"][-1].content if state["messages"] else ""
    thinking = f"Extracting query from: '{user_input}'"
    return {"thinking_process": thinking}

def call_research_subgraph(state: AgentState) -> AgentState:
    user_input = state["messages"][-1].content if state["messages"] else ""
    # Create subgraph input
    subgraph_input = {"query": user_input}
    
    # Call research subgraph
    research_subgraph = build_research_subgraph()
    result = research_subgraph.invoke(subgraph_input)
    
    return {"query": user_input, "facts": result["facts"]}

def prepare_response_node(state: AgentState) -> AgentState:
    query = state.get("query", "")
    facts = state.get("facts", [])
    response = f"Here's what I found about '{query}':\n" + "\n- ".join([""] + facts)
    
    return {"response": response, "needs_approval": True}

def human_approval_node(state: AgentState) -> AgentState:
    if not state.get("needs_approval", False):
        return {"approved": True}
    
    response = state.get("response", "")
    
    # Interrupt the graph to ask for human approval
    user_feedback = interrupt(
        {"message": "Please approve this response", "response": response}
    )
    
    # This will only execute after the graph is resumed with the user feedback
    approved = user_feedback.get("approved", False)
    return {"approved": approved}

def final_response_node(state: AgentState) -> AgentState:
    response = state.get("response", "")
    approved = state.get("approved", False)
    
    if approved:
        final_response = response
    else:
        final_response = "I was not able to get approval for my response."
    
    return {
        "response": final_response,
        "messages": [AIMessage(content=final_response)]
    }

# Build main graph
def build_graph():
    builder = StateGraph(AgentState, input=InputState, output=OutputState)
    
    builder.add_node("extract_query", extract_query_node)
    builder.add_node("research", call_research_subgraph)
    builder.add_node("prepare_response", prepare_response_node)
    builder.add_node("human_approval", human_approval_node)
    builder.add_node("final_response", final_response_node)
    
    builder.add_edge(START, "extract_query")
    builder.add_edge("extract_query", "research")
    builder.add_edge("research", "prepare_response")
    builder.add_edge("prepare_response", "human_approval")
    builder.add_edge("human_approval", "final_response")
    builder.add_edge("final_response", END)
    
    return builder.compile()

def run_example_without_interruption():
    graph = build_graph()
    
    # In a real scenario, with human-in-the-loop, we would use threading or async
    # to handle the interruption and resumption. For demonstration, we're
    # creating a simple example where we can examine the graph state
    # at the point of interruption.
    
    # Initialize with a message
    initial_state = {"messages": [HumanMessage(content="Tell me about LangGraph")]}
    
    # This will run until it hits the interrupt
    try:
        result = graph.invoke(initial_state)
        print("Complete result:", result)
    except Exception as e:
        print(f"Graph execution interrupted: {e}")
        print("You would normally gather user input here and resume the graph")
        
        # In a real application, you would get user feedback and resume:
        # result = graph.invoke(
        #     initial_state,  # Same state
        #     {"messages": [{'resume': {'approved': True}}]}  # Resume info with approval
        # )
        # print("Final result:", result)

if __name__ == "__main__":
    run_example_without_interruption()