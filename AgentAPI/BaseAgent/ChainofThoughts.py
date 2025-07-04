from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from typing import Annotated
import json

class ThoughtStep(TypedDict):
    thought: str
    reasoning: str

class ChainOfThoughts(TypedDict):
    thoughts: list[ThoughtStep]

checkpointer = InMemorySaver()

llm = init_chat_model("openai:qwen2.5vl:7b-q8_0")

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

def think_and_reason(state: State):
    system_prompt = """You are a reasoning agent that thinks step by step. When given a query, you should:
    1. Break down your thinking into logical steps
    2. For each step, provide your thought and reasoning
    3. Build upon previous thoughts
    
    Structure your response as:
    {
        "thoughts": [
            {
                "thought": "What I'm thinking about this aspect",
                "reasoning": "Why this thought is important and how it connects"
            }
        ]
    }
    
    Think through the problem systematically and provide detailed reasoning for each thought."""
    
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    response = llm.with_structured_output(ChainOfThoughts).invoke(messages)
    
    return {
        "messages": [{
            "role": "assistant",
            "content": json.dumps(response),
        }]
    }

def answer(state: State):
    system_prompt = """You are an answer agent. Your job is to answer the user's query based on the thoughts and reasoning."""
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]

    messages.append({"role": "user", "content": f"Based on the thoughts and reasoning, Answer/solve the user's query."})

    response = llm.invoke(messages)

    return {"messages": [response]}

graph_builder.add_node("reasoning", think_and_reason)
graph_builder.add_node("answer", answer)
graph_builder.add_edge(START, "reasoning")
graph_builder.add_edge("reasoning", "answer")
graph_builder.add_edge("answer", END)

graph = graph_builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "1"}}
events = graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "what is x when the equation 6x+49=-432",
            },
        ]
    },
    config,
    stream_mode="values",
)

for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()