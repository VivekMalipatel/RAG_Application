from typing import Annotated, Literal, TypedDict, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from operator import add
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage

class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    thought: str
    needs_lookup: bool
    facts: List[str]
    response: str

def thinking_node(state: AgentState) -> Command[Literal["lookup_facts", "generate_response"]]:
    user_input = state["messages"][-1].content if state["messages"] else ""
    thought = f"Analyzing user input: '{user_input}'"
    
    needs_lookup = "who" in user_input.lower() or "what" in user_input.lower()
    
    return Command(
        update={"thought": thought, "needs_lookup": needs_lookup},
        goto="lookup_facts" if needs_lookup else "generate_response"
    )

def lookup_facts_node(state: AgentState) -> AgentState:
    user_input = state["messages"][-1].content if state["messages"] else ""
    facts = ["Fact 1 related to " + user_input, 
             "Fact 2 related to " + user_input]
    
    return {"facts": facts}

def generate_response_node(state: AgentState) -> AgentState:
    user_input = state["messages"][-1].content if state["messages"] else ""
    facts = state.get("facts", [])
    
    if facts:
        response = f"Based on my research about '{user_input}', I found: {', '.join(facts)}"
    else:
        response = f"I understand you're asking about '{user_input}'. Here's my response without needing to look up facts."
    
    return {
        "response": response,
        "messages": [AIMessage(content=response)]
    }

def build_graph():
    builder = StateGraph(AgentState)
    
    builder.add_node("thinking", thinking_node)
    builder.add_node("lookup_facts", lookup_facts_node)
    builder.add_node("generate_response", generate_response_node)
    
    builder.add_edge(START, "thinking")
    builder.add_edge("lookup_facts", "generate_response")
    builder.add_edge("generate_response", END)
    
    return builder.compile()

def run_example():
    graph = build_graph()
    
    result = graph.invoke({"messages": [HumanMessage(content="What is LangGraph?")]})
    print("RESULT 1:", result["response"])
    print("MESSAGES 1:", [msg.content for msg in result["messages"]])
    
    result = graph.invoke({"messages": [HumanMessage(content="Hello")]})
    print("RESULT 2:", result["response"])
    print("MESSAGES 2:", [msg.content for msg in result["messages"]])

if __name__ == "__main__":
    run_example()