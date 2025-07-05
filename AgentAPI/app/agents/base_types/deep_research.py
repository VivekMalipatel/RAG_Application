from typing import Dict, Any, List, Annotated
from openai import OpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
import requests
from app.agents.base_agent import BaseAgent
from app.core.config import settings
from app.tools import tool_registry
from app.tools.tool_executor import ToolExecutorMixin
import json
import logging

logger = logging.getLogger(__name__)

class DeepResearchState(TypedDict):
    messages: Annotated[list, add_messages]

class ThoughtStep(TypedDict):
    thought: str
    reasoning: str

class ChainOfThoughts(TypedDict):
    thoughts: list[ThoughtStep]

class BaseLLMNode:
    def __init__(self, prompt=None, client=None, model=None):
        self.prompt = prompt
        self.client = client
        self.model = model

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

    async def think_and_reason(self, state: DeepResearchState):
        system_prompt = self.get_reasoning_prompt()
        
        messages = [{"role": "system", "content": system_prompt}] + state["messages"]

        # For structured output, we'll use a special prompt to get JSON
        structured_prompt = system_prompt + "\n\nPlease respond with valid JSON in the exact format specified above."
        messages = [{"role": "system", "content": structured_prompt}] + state["messages"]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=2000,
            temperature=0.7
        )

        response_content = response.choices[0].message.content
        
        return {
            "messages": [{
                "role": "assistant",
                "content": response_content,
            }]
        }

    async def answer_with_tools(self, state: DeepResearchState, search_enabled=False):
        system_prompt = self.get_answer_prompt()
        messages = [{"role": "system", "content": system_prompt}] + state["messages"]

        messages.append({"role": "user", "content": "Based on the thoughts and reasoning, do your job."})

        # If search is enabled, use the web search tool
        search_results = ""
        if search_enabled:
            # Extract query from the last user message
            last_user_msg = None
            for msg in reversed(state["messages"]):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    last_user_msg = msg.get("content", "")
                    break
            
            if last_user_msg and hasattr(self, '_search_with_tools'):
                search_results = await self._search_with_tools(last_user_msg)
                if search_results:
                    messages.append({"role": "system", "content": f"Search Results:\n{search_results}"})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=3000,
            temperature=0.7
        )

        response_content = response.choices[0].message.content

        return {"messages": [{"role": "assistant", "content": response_content}]}
    
    async def answer_without_tools(self, state: DeepResearchState):
        system_prompt = self.get_answer_prompt()
        messages = [{"role": "system", "content": system_prompt}] + state["messages"]

        messages.append({"role": "user", "content": "Based on the thoughts and reasoning, do your job."})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=3000,
            temperature=0.7
        )

        response_content = response.choices[0].message.content

        return {"messages": [{"role": "assistant", "content": response_content}]}

class DeepResearchAgent(BaseAgent, ToolExecutorMixin):
    def __init__(self, role: str = "Research Agent", instructions: str = "", capabilities: list = None):
        super().__init__(role, instructions, capabilities)
        self.client = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_API_BASE
        )
        self.model = getattr(settings, 'DEFAULT_MODEL', 'qwen2.5:7b-instruct-q8_0')
        self.search_model = getattr(settings, 'SEARCH_MODEL', 'qwen2.5vl:7b-q8_0')
        self.checkpointer = InMemorySaver()
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build the deep research graph"""
        graph_builder = StateGraph(DeepResearchState)
        
        graph_builder.add_node("gather_background_knowledge", self._gather_background_knowledge)
        graph_builder.add_node("query_intent_analysis", self._query_intent_analysis)
        graph_builder.add_node("gap_assessment", self._gap_assessment_node)
        
        graph_builder.add_edge(START, "gather_background_knowledge")
        graph_builder.add_edge("gather_background_knowledge", "query_intent_analysis")
        graph_builder.add_edge("query_intent_analysis", "gap_assessment")
        graph_builder.add_edge("gap_assessment", END)
        
        return graph_builder.compile(checkpointer=self.checkpointer)

    async def _search_with_tools(self, query: str) -> str:
        """Perform web search using the tool registry"""
        try:
            # Use the general tool execution approach
            result = await tool_registry.execute_tool("web_search", {
                "query": query,
                "num_results": 5
            })
            
            if result["success"]:
                return json.dumps(result["result"], indent=2)
            else:
                return f"Search error: {result['error']}"
                
        except Exception as e:
            logger.error(f"Web search error: {str(e)}")
            return f"Search error: {str(e)}"

    def _search_web(self, query: str) -> str:
        """Legacy method - redirect to async search"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self._search_with_tools(query))

    async def _gather_background_knowledge(self, state: DeepResearchState):
        system_prompt = """You are Background Knowledge Gatherer, the very first stage of a deep-research pipeline. Your ONLY job is to build a quick but solid foundation for downstream agents—nothing more.

        **Scope**
        1. Read the user's research topic/query and any provided context.  
        2. Retrieve—or recall—high-level background facts: definitions, key entities, timelines, seminal papers, must-know jargon.  
        3. DO NOT attempt to solve the research question itself, draw conclusions, or give opinions.  
        4. Keep it *broad but shallow*: "What do I need to know before I even start researching this in depth?"

        Answer in proper Markdown format, with clear headings and bullet points.
        """
        
        base_node = BaseLLMNode(system_prompt, self.client, self.search_model)
        base_node._search_with_tools = self._search_with_tools  # Add search capability
        return await base_node.answer_with_tools(state, search_enabled=True)

    async def _query_intent_analysis(self, state: DeepResearchState):
        system_prompt = """You are IntentAnalyzer GPT, the strategist of a deep-research pipeline.  
        Your mission is **not** to answer the research question itself, but to read:

        1. **User query & any contextual notes**  
        2. **Background foundation** produced by *background_knowledge_node*

        …and translate them into a clear, machine-readable **research contract** that downstream agents will follow.

        Tasks
        1. **Clarify goals** – rewrite the user's ultimate objective in one crisp sentence.  
        2. **Decompose** – extract the explicit or implicit sub-questions that must be answered for the goal to be satisfied.  
        3. **Set deliverable spec** – infer preferred output style (e.g., executive summary, technical deep-dive, comparative table) and target audience (e.g., C-suite, researchers, lay readers).  
        4. **Define success criteria** – quality bars, freshness thresholds, depth-of-analysis, or any stated constraints (time/budget/scope).  
        5. **Prioritize** – assign a *relative weight* (1–5) to each sub-question based on its importance to the overall goal.  
        6. **Flag exclusions** – note any topics or sources the user expressly wants omitted.

        Answer in proper Markdown format, with clear headings and bullet points.
        """
        base_node = BaseLLMNode(system_prompt, self.client, self.model)
        return await base_node.answer_without_tools(state)

    async def _gap_assessment_node(self, state: DeepResearchState):
        system_prompt = """You are Research GAP Analyser, the "checkpoint" node of a deep-research pipeline.

        Context you will receive
        1. **User query** – original question.  
        2. **Background summary** – Markdown from *background_knowledge_node*.  
        3. **Intent contract** – Markdown from *intent_analysis_node* that includes  *`key_questions`*, weights, and success criteria.  

        Your objectives
        1. **Audit progress** – For every `key_question`, decide whether it is **UNANSWERED**, **PARTIALLY ANSWERED**, or **FULFILLS CRITERIA**.  
        2. **Produce "knowledge gaps"** – all questions in the first two categories, along with *why* they still need work.  
        3. **Set loop directive** – Tell downstream nodes whether to keep iterating or hand off to synthesis.  
        
        Answer in proper Markdown format, with clear headings and bullet points.
        """

        base_node = BaseLLMNode(system_prompt, self.client, self.model)
        return await base_node.answer_without_tools(state)

    async def execute(self, query: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the deep research agent"""
        try:
            thread_id = config.get("thread_id", "default")
            graph_config = {
                "configurable": {"thread_id": thread_id}, 
                "recursion_limit": 50
            }
            
            initial_state = {
                "messages": [{"role": "user", "content": query}]
            }
            
            # Stream through the graph execution
            events = []
            async for event in self.graph.astream(initial_state, graph_config, stream_mode="values"):
                if "messages" in event:
                    events.append(event)
            
            if events:
                final_event = events[-1]
                final_message = final_event["messages"][-1]
                response_text = final_message["content"] if isinstance(final_message, dict) else final_message.content
                
                # Collect intermediate results
                background = ""
                intent = ""
                gaps = ""
                
                for event in events:
                    messages = event.get("messages", [])
                    for msg in messages:
                        content = msg["content"] if isinstance(msg, dict) else (msg.content if hasattr(msg, 'content') else str(msg))
                        if "background" in content.lower() and not background:
                            background = content
                        elif "intent" in content.lower() and not intent:
                            intent = content
                        elif "gap" in content.lower() and not gaps:
                            gaps = content
                
                result_data = {
                    "response": response_text,
                    "background_knowledge": background,
                    "intent_analysis": intent,
                    "gap_assessment": gaps,
                    "agent_type": "deep_research",
                    "thread_id": thread_id,
                    "status": "completed"
                }
            else:
                result_data = {
                    "response": "No response generated",
                    "agent_type": "deep_research",
                    "thread_id": thread_id,
                    "status": "error"
                }
            
            self.log_execution(query, result_data["response"], config)
            return result_data
            
        except Exception as e:
            self.logger.error(f"Error in DeepResearchAgent execution: {str(e)}")
            return {
                "response": f"Error: {str(e)}",
                "agent_type": "deep_research",
                "thread_id": config.get("thread_id"),
                "status": "error"
            }
