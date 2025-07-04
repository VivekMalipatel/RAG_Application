from typing import Annotated
from langchain_tavily import TavilySearch
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.chat_models import init_chat_model
import json

########################LLM Setup######################

llm = init_chat_model("qwen2.5:7b-instruct-q8_0", model_provider="openai", base_url="http://192.5.87.119:8000/v1")
llm_without_tools = init_chat_model("qwen2.5vl:7b-q8_0", model_provider="openai", base_url="http://192.5.87.98:8000/v1")

######################Tool Setup#####################

tools = []

tools.append(TavilySearch(max_results=10))

llm_with_tools = llm.bind_tools(tools)

######################Graph Setup######################

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

checkpointer = MemorySaver()

#######################Base Node Class######################

class ThoughtStep(TypedDict):
    thought: str
    reasoning: str

class ChainOfThoughts(TypedDict):
    thoughts: list[ThoughtStep]

class BaseLLMNode:

    def __init__(self, prompt=None):
        self.prompt = prompt
        self.base_llm_graph_builder = None

    def get_reasoning_prompt(self):

        template = """
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

        if self.prompt:
            return self.prompt + "\n\n" + template
        return template
    
    def get_answer_prompt(self):

        template = """Use the thoughts and reasoning and try to do your job."""

        if self.prompt:
            return self.prompt + "\n\n" + template
        return template

    def think_and_reason(self, state: State):
        system_prompt = self.get_reasoning_prompt()
        
        messages = [{"role": "system", "content": system_prompt}] + state["messages"]

        response = llm_without_tools.with_structured_output(ChainOfThoughts).invoke(messages)

        print("\nThoughts and Reasoning:", response)
        
        return {
            "messages": [{
                "role": "assistant",
                "content": json.dumps(response),
            }]
        }

    def answer_with_tools(self, state: State):
        system_prompt = self.get_answer_prompt()
        messages = [{"role": "system", "content": system_prompt}] + state["messages"]

        messages.append({"role": "user", "content": f"Based on the thoughts and reasoning, do your job."})

        response = llm_with_tools.invoke(messages)

        print("\nAnswer with Tools:", response)

        return {"messages": [response]}
    
    def answer_without_tools(self, state: State):
        system_prompt = self.get_answer_prompt()
        messages = [{"role": "system", "content": system_prompt}] + state["messages"]

        messages.append({"role": "user", "content": f"Based on the thoughts and reasoning, do your job."})

        response = llm_without_tools.invoke(messages)

        print("\nAnswer without Tools:", response)

        return {"messages": [response]}
    
    def build_base_graph(self, answer_with_tools=True):
        self.base_llm_graph_builder = StateGraph(State)
        self.base_llm_graph_builder.add_node("reasoning", self.think_and_reason)
        if answer_with_tools:
            self.base_llm_graph_builder.add_node("answer_with_tools", self.answer_with_tools)
            self.base_llm_graph_builder.add_node("tools", ToolNode(tools))
        else:
            self.base_llm_graph_builder.add_node("answer_without_tools", self.answer_without_tools)

        self.base_llm_graph_builder.add_edge(START, "reasoning")
        if answer_with_tools:
            self.base_llm_graph_builder.add_edge("reasoning", "answer_with_tools")
            self.base_llm_graph_builder.add_conditional_edges(
                "answer_with_tools",
                tools_condition,
                {"tools": "tools", "__end__": END}
            )
            self.base_llm_graph_builder.add_edge("tools", "reasoning")
        else:
            self.base_llm_graph_builder.add_edge("reasoning", "answer_without_tools")
            self.base_llm_graph_builder.add_edge("answer_without_tools", END)

        return self.base_llm_graph_builder.compile()
    
    def execute_node(self, state: State, tools_enabled: bool = True):

        graph = self.build_base_graph(answer_with_tools=tools_enabled)
        event = graph.invoke(state,{"configurable": {"thread_id": "1", "recursion_limit": 50}})
        return {"messages": [event["messages"][-1]]}

########################Graph Nodes########################

def gather_background_knowledge(state: State):

    system_prompt = """You are Background Knowledge Gatherer, the very first stage of a deep-research pipeline.  Your ONLY job is to build a quick but solid *foundation* for downstream agents—nothing more.

    **Scope**
    1. Read the user’s research topic/query and any provided context.  
    2. Retrieve—or recall—high-level background facts: definitions, key entities, timelines, seminal papers, must-know jargon.  
    3. DO NOT attempt to solve the research question itself, draw conclusions, or give opinions.  
    4. Keep it *broad but shallow*: “What do I need to know before I even start researching this in depth?”

    Answer in proper Markdown format, with clear headings and bullet points.
    """
    
    base_node = BaseLLMNode(system_prompt)
    output = base_node.execute_node(state, tools_enabled=True)
    return output

def query_intent_analysis(state: State):

    system_prompt = """You are IntentAnalyzer GPT, the strategist of a deep-research pipeline.  
    Your mission is **not** to answer the research question itself, but to read:

    1. **User query & any contextual notes**  
    2. **Background foundation JSON** produced by *background_knowledge_node*

    …and translate them into a clear, machine-readable **research contract** that downstream agents will follow.

    Tasks
    1. **Clarify goals** – rewrite the user’s ultimate objective in one crisp sentence.  
    2. **Decompose** – extract the explicit or implicit sub-questions that must be answered for the goal to be satisfied.  
    3. **Set deliverable spec** – infer preferred output style (e.g., executive summary, technical deep-dive, comparative table) and target audience (e.g., C-suite, researchers, lay readers).  
    4. **Define success criteria** – quality bars, freshness thresholds, depth-of-analysis, or any stated constraints (time/budget/scope).  
    5. **Prioritize** – assign a *relative weight* (1–5) to each sub-question based on its importance to the overall goal.  
    6. **Flag exclusions** – note any topics or sources the user expressly wants omitted.

    Answer in proper Markdown format, with clear headings and bullet points.
    """
    base_node = BaseLLMNode(system_prompt)
    return base_node.execute_node(state, tools_enabled=False)

def gap_assessment_node(state: State):

    system_prompt = """You are Research GAP Analyser, the “checkpoint” node of a deep-research pipeline.

        Context you will receive
        1. **User query** – original question.  
        2. **Background summary** – Markdown from *background_knowledge_node*.  
        3. **Intent contract** – Markdown from *intent_analysis_node* that includes  *`key_questions`*, weights, and success criteria.  

        Your objectives
        1. **Audit progress** – For every `key_question`, decide whether it is **UNANSWERED**, **PARTIALLY ANSWERED**, or **FULFILLS CRITERIA**.  
        2. **Produce “knowledge gaps”** – all questions in the first two categories, along with *why* they still need work.  
        3. **Set loop directive** – Tell downstream nodes whether to keep iterating or hand off to synthesis.  
        
        Answer in proper Markdown format, with clear headings and bullet points.
        """

    base_node = BaseLLMNode(system_prompt)
    return base_node.execute_node(state, tools_enabled=False)


############################ Graph Compilation ############################

graph_builder.add_node("gather_background_knowledge", gather_background_knowledge)
graph_builder.add_node("query_intent_analysis", query_intent_analysis)
graph_builder.add_node("gap_assessment", gap_assessment_node)


graph_builder.add_edge(START, "gather_background_knowledge")
graph_builder.add_edge("gather_background_knowledge", "query_intent_analysis")
graph_builder.add_edge("query_intent_analysis", "gap_assessment")
graph_builder.add_edge("gap_assessment", END)

graph = graph_builder.compile(checkpointer=checkpointer)

event = graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "I'm learning LangGraph. "
                    "Could you do some research on it for me?"
                ),
            },
        ],
    },
    {"configurable": {"thread_id": "1"}, "recursion_limit": 50},
    stream_mode="values",
)

for e in event:
    if "messages" in e:
        e["messages"][-1].pretty_print()