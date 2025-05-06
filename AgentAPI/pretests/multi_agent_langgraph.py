from typing import Annotated, Dict, List, Literal, TypedDict
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from operator import add
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage

# Define the state for our multi-agent workflow
class MultiAgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    current_agent: str
    research_results: List[str]
    code_snippets: List[str]
    final_answer: str

def router_node(state: MultiAgentState) -> Command[Literal["researcher", "coder", "writer", "end"]]:
    user_input = state["messages"][-1].content if state["messages"] else ""
    
    if not "current_agent" in state:
        # First time through, start with researcher
        return Command(update={"current_agent": "researcher"}, goto="researcher")
    
    current_agent = state["current_agent"]
    
    if current_agent == "researcher" and state.get("research_results"):
        # Research done, go to coder
        return Command(update={"current_agent": "coder"}, goto="coder")
    
    if current_agent == "coder" and state.get("code_snippets"):
        # Code done, go to writer
        return Command(update={"current_agent": "writer"}, goto="writer")
    
    if current_agent == "writer" and state.get("final_answer"):
        # Everything done, end the workflow
        return Command(update={}, goto="end")
    
    # Stay with current agent if their work isn't complete
    return Command(update={}, goto=current_agent)

def researcher_agent(state: MultiAgentState) -> MultiAgentState:
    user_input = state["messages"][-1].content if state["messages"] else ""
    
    # In a real scenario, this would use an LLM to generate research
    research_message = f"Researcher agent here. I've researched '{user_input}' and found:"
    research_results = [
        f"Research finding 1 about {user_input}",
        f"Research finding 2 about {user_input}",
        f"Research finding 3 about {user_input}"
    ]
    
    agent_message = AIMessage(
        content=research_message + "\n- " + "\n- ".join(research_results),
        name="researcher"
    )
    
    return {
        "research_results": research_results,
        "messages": [agent_message]
    }

def coder_agent(state: MultiAgentState) -> MultiAgentState:
    user_input = state["messages"][-1].content if state["messages"] else ""
    research_results = state.get("research_results", [])
    
    # In a real scenario, this would use an LLM to generate code
    code_message = "Coder agent here. Based on the research, I've created these code snippets:"
    code_snippets = [
        "```python\n# Code snippet 1\ndef example_function():\n    return 'Hello world'\n```",
        "```python\n# Code snippet 2\nclass ExampleClass:\n    def __init__(self):\n        self.value = 42\n```"
    ]
    
    agent_message = AIMessage(
        content=code_message + "\n\n" + "\n\n".join(code_snippets),
        name="coder"
    )
    
    return {
        "code_snippets": code_snippets,
        "messages": [agent_message]
    }

def writer_agent(state: MultiAgentState) -> MultiAgentState:
    user_input = state["messages"][-1].content if state["messages"] else ""
    research_results = state.get("research_results", [])
    code_snippets = state.get("code_snippets", [])
    
    # In a real scenario, this would use an LLM to generate the final answer
    final_message = f"Writer agent here. I've synthesized everything about '{user_input}':"
    final_answer = f"Based on our research and code development, here's a comprehensive response to your query about {user_input}..."
    
    agent_message = AIMessage(
        content=final_message + "\n\n" + final_answer,
        name="writer"
    )
    
    return {
        "final_answer": final_answer,
        "messages": [agent_message]
    }

def end_node(state: MultiAgentState) -> MultiAgentState:
    # This node just passes the final answer through
    return {}

def build_multi_agent_graph():
    builder = StateGraph(MultiAgentState)
    
    # Add nodes
    builder.add_node("router", router_node)
    builder.add_node("researcher", researcher_agent)
    builder.add_node("coder", coder_agent)
    builder.add_node("writer", writer_agent)
    builder.add_node("end", end_node)
    
    # Add edges
    builder.add_edge(START, "router")
    builder.add_edge("router", "researcher")
    builder.add_edge("router", "coder")
    builder.add_edge("router", "writer")
    builder.add_edge("router", "end")
    builder.add_edge("researcher", "router")
    builder.add_edge("coder", "router")
    builder.add_edge("writer", "router")
    builder.add_edge("end", END)
    
    return builder.compile()

def run_multi_agent_example():
    graph = build_multi_agent_graph()
    
    initial_state = {
        "messages": [
            SystemMessage(content="You are a helpful assistant that coordinates multiple specialized agents."),
            HumanMessage(content="Explain how to implement a binary search tree in Python")
        ]
    }
    
    result = graph.invoke(initial_state)
    
    print("FINAL STATE:")
    print(f"Number of messages: {len(result['messages'])}")
    for i, message in enumerate(result["messages"]):
        print(f"\nMessage {i+1}:")
        print(f"Role: {message.type}")
        if hasattr(message, 'name') and message.name:
            print(f"Agent: {message.name}")
        print(f"Content: {message.content[:100]}...")

if __name__ == "__main__":
    run_multi_agent_example()