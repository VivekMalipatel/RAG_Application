from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel
from json import JSONDecodeError
import json
import sys

import json

checkpointer = InMemorySaver()

llm = init_chat_model("openai:qwen2.5vl:7b-q8_0",max_tokens=10000)

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

def answer(state: State):
    system_prompt = """You are an answer agent. Your job is to answer the user's query based on the planning steps provided. You will be given a query and a list of planning steps, and you should return the final answer to the query."""
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]

    messages.append({"role": "user", "content": f"Based on the planning steps, Answer/solve the user's query."})

    response = llm.invoke(messages)

    return {"messages": [response]}

graph_builder.add_node("answer", answer)

graph_builder.add_edge(START, "answer")
graph_builder.add_edge("answer", END)

graph = graph_builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "1"}}
events = graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "What is Langgraph?",
            },
        ]
    },
    config,
    stream_mode="messages",
)

current_node = None
first_token = True

for event, metadata in events:
    node_name = metadata['langgraph_node']
    content = event.content if hasattr(event, 'content') else str(event)
    
    if node_name != current_node:
        if current_node is not None:
            print()
        if node_name == "answer":
            print("\033[1;32mBOT:\033[0m ", end="", flush=True)
        else:
            print("\033[1;34mUSER:\033[0m ", end="", flush=True)
        current_node = node_name
        first_token = True

    if first_token and content.startswith(" "):
        content = content.lstrip()
    print(content, end="", flush=True)
    first_token = False

print()



