from openai import AsyncOpenAI
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from sqlalchemy.orm import Session
from app.db.db import SessionLocal
from app.agent.memory import ThreadMemoryHandler, AgentMemoryHandler, ClientMemoryHandler
from fastapi import HTTPException
from app.config import settings
import logging
import asyncio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class AgentConfig(BaseModel):
    name: str = Field(..., min_length=3)
    role: str = Field(..., min_length=5)
    instructions: str = Field(..., min_length=10)
    capabilities: List[str] = Field(default_factory=list)
    model_name: str = "qwen2.5vl:7b-q8_0"
    llm_config: Dict[str, Any] = Field(default_factory=dict)

class Plan(BaseModel):
    description: str
    last_step: bool = False

class Message(BaseModel):
    role: str
    content: Any

class ReasoningOutput(BaseModel):
    thought: str

class AgentState(BaseModel):
    messages: List[Message] = Field(default_factory=list)
    action_plan: List[Plan] = Field(default_factory=list)
    reasoning: List[ReasoningOutput] = Field(default_factory=list)
    thread_id: Optional[str] = None
    error: Optional[str] = Field(default=None)

class BaseAgent:
    def __init__(self, config: AgentConfig, db: Session = SessionLocal()):
        self.config = config
        self.db = db
        self._initialize_llm()
        self.workflow_builder = StateGraph(AgentState)
        self._setup_memory_handlers()
        self._build_core_workflow()
        self.workflow = None

    def _initialize_llm(self):
        api_key = settings.OPENAI_API_KEY
        base_url = settings.OPENAI_API_BASE

        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in settings.")
        if not base_url:
            raise ValueError("OPENAI_API_BASE not found in settings.")

        self.llm = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            **self.config.llm_config
        )
        self.model_name = self.config.model_name
        logger.info(f"Initialized LLM with model: {self.model_name}")

    def _setup_memory_handlers(self):
        self.memory = {
            'thread': ThreadMemoryHandler(self.db),
            'agent': AgentMemoryHandler(self.db),
            'client': ClientMemoryHandler(self.db)
        }

    def _build_core_workflow(self):
        self.workflow_builder.add_node("perceive", self._perceive)
        self.workflow_builder.add_node("plan", self._plan)
        self.workflow_builder.add_node("reason", self._reason)
        self.workflow_builder.add_node("act", self._act)
        self.workflow_builder.add_node("evaluate", self._evaluate)
        self.workflow_builder.add_node("learn", self._learn)

        self.workflow_builder.set_entry_point("perceive")
        self.workflow_builder.add_edge("perceive", "plan")
        self.workflow_builder.add_edge("plan", "reason")
        
        self.workflow_builder.add_conditional_edges(
            "reason",
            self._should_continue_planning_or_act,
            {
                "continue_planning": "plan",
                "act": "act"
            }
        )
        
        self.workflow_builder.add_conditional_edges(
            "act",
            self._should_continue,
            {"continue": "evaluate", "end": END}
        )
        self.workflow_builder.add_edge("evaluate", "learn")
        self.workflow_builder.add_edge("learn", END)

    def _perceive(self, state: AgentState) -> dict:
        logger.info(f"Perceive: Processing for thread {state.thread_id}")
        
        loaded_data: Optional[Dict[str, Any]] = None
        if state.thread_id:
            try:
                persisted_content = self.memory['thread'].load(state.thread_id)
                loaded_data = persisted_content

                if loaded_data:
                    logger.info(f"Perceive: Loaded historical state for thread {state.thread_id}")
                else:
                    logger.info(f"Perceive: No historical state found for thread {state.thread_id}")
            except Exception as e:
                logger.error(f"Perceive: Error loading state for thread {state.thread_id}: {e}")
        else:
            logger.warning("Perceive: No thread_id in state, cannot load history.")

        current_input_messages = state.messages

        if loaded_data:
            try:
                historical_agent_state = AgentState(**loaded_data)
                all_messages = historical_agent_state.messages + current_input_messages
                
                return {
                    "messages": all_messages,
                    "action_plan": historical_agent_state.action_plan,
                    "reasoning": historical_agent_state.reasoning,
                    "thread_id": state.thread_id,
                    "error": None
                }
            except Exception as e:
                logger.error(f"Perceive: Error parsing loaded state for thread {state.thread_id}: {e}. Proceeding with fresh state.")

        return {
            "messages": current_input_messages,
            "action_plan": [],
            "reasoning": [],
            "thread_id": state.thread_id,
            "error": None
        }

    async def _plan(self, state: AgentState) -> dict:
        logger.info("Plan: Generating next plan step")
        
        user_message = None
        messages = state.messages
        for message in messages:
            if message.role == "user":
                user_message = message
                break
        
        if not user_message:
            logger.warning("No user message found in state for planning")
            return {"action_plan": state.action_plan, "error": "No user message found"} 
            
        query = user_message.content
        if not query:
            logger.warning("User message has no content for planning")
            return {"action_plan": state.action_plan, "error": "User message has no content"}

        system_prompt_plan = f"""You are {self.config.name}, an advanced reasoning agent. Your designated role is: {self.config.role}.
            Your primary instruction set is: {self.config.instructions}

            Your current task is to meticulously plan the solution to a given problem or query by breaking it down into a sequence of actionable steps. You will generate these steps one at a time, iteratively.

            **Iterative Planning Process:**
            1.  You will receive the main task/problem and potentially a list of previously planned steps.
            2.  If no previous steps exist, your goal is to define the very first step required to address the task.
            3.  If previous steps are provided, your goal is to define the immediate next logical step that follows the last planned step.
            4.  Continue this process until you determine that the plan is complete and no further steps are necessary to solve the problem.

            **Output Requirements:**
            Your response for each step MUST be a JSON object. This JSON object must strictly adhere to the following structure:
            {{
                "description": "A clear, concise, and detailed explanation of the single, specific action to be taken in this step of the plan. This description should be self-contained and understandable.",
                "last_step": <boolean> // Set this to `true` ONLY if this current step is the absolute final step in the entire plan. Otherwise, set it to `false`.
            }}

            **Critical Instructions for `last_step` field:**
            -   Set `last_step: true` if, and only if, you are confident that the current step you are defining is the concluding action needed to fully address the original task/problem.
            -   If more actions are needed after the current step, you MUST set `last_step: false`.

            Example of a valid JSON output:
            {{
                "description": "Analyze the user\'s request to identify key entities and intents.",
                "last_step": false
            }}

            Example of a valid JSON output for the final step:
            {{
                "description": "Compile all gathered information into a final report and present it to the user.",
                "last_step": true
            }}

            Focus solely on generating the next single step in the plan. Do not attempt to solve the entire problem in one go.
            """

        current_plan_list = state.action_plan
        plan_list_str = "\n".join([f"Step {i+1}: {step.description}" for i, step in enumerate(current_plan_list)])
        prompt_query_for_llm = f"This is the task/problem : {query}\n\nPlan as of now : \n{plan_list_str}"

        try:
            response = await self.llm.beta.chat.completions.parse(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt_plan},
                    {"role": "user", "content": prompt_query_for_llm}
                ],
                response_format=Plan, 
            )
            
            new_plan_step_parsed = response.choices[0].message.parsed
            
            updated_plan_list = current_plan_list + [new_plan_step_parsed]
            logger.info(f"Generated plan step: {new_plan_step_parsed.description}")
            
            return {
                "action_plan": updated_plan_list
            }
        except Exception as e:
            logger.error(f"Error during planning LLM call: {e}")
            return {"action_plan": current_plan_list, "error": str(e)}

    async def _reason(self, state: AgentState) -> dict:
        logger.info("Reason: Generating reasoning for the latest plan step")

        action_plan = state.action_plan
        if not action_plan:
            logger.warning("No action plan found to reason about.")
            return {"reasoning": state.reasoning, "error": "No action plan to reason about"}

        current_step_to_reason = action_plan[-1]
        
        user_message = None
        messages = state.messages
        for message in messages:
            if message.role == "user":
                user_message = message
                break
        query = user_message.content if user_message else ""

        system_prompt_reason = f"""You are {self.config.name}, a reasoning agent with the role: {self.config.role}.
            {self.config.instructions}

            You will be given a specific step in a plan and the original task. Your job is to provide detailed reasoning or execution for this specific step.
            Focus only on the current step, not the entire task. Be thorough but specific.
            
            Your output MUST be a JSON object strictly adhering to the following structure:
            {{
                "thought": "Your detailed reasoning or solution for the given step."
            }}
            Example:
            {{
                "thought": "Based on the user's request to understand LLMs, the first step is to define what an LLM is. An LLM, or Large Language Model, is a type of artificial intelligence..."
            }}
            """
            
        reasoning_prompt_for_llm = f"""Original task: {query}
            
            Current plan step to reason about: {current_step_to_reason.description}
                        
            Provide your detailed thought process and reasoning for executing this specific step:"""

        try:
            response = await self.llm.beta.chat.completions.parse(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt_reason},
                    {"role": "user", "content": reasoning_prompt_for_llm}
                ],
                response_format=ReasoningOutput
            )
            reasoning_output_parsed = response.choices[0].message.parsed
            logger.info(f"Generated reasoning for step: {current_step_to_reason.description}")

            current_reasoning_list = state.reasoning
            updated_reasoning_list = current_reasoning_list + [reasoning_output_parsed]
            
            return {"reasoning": updated_reasoning_list}
        except Exception as e:
            logger.error(f"Error during reasoning LLM call: {e}")
            return {"reasoning": state.reasoning, "error": str(e)}

    def _should_continue_planning_or_act(self, state: AgentState) -> str:
        if state.error: 
            logger.warning(f"Error detected in state: {state.error}, proceeding to act for error handling or finalization.")
            return "act" 

        if not state.action_plan:
            logger.warning("No action plan available. Cannot determine if planning is complete. Proceeding to act.")
            return "act"

        is_last_step = state.action_plan[-1].last_step
        if is_last_step:
            logger.info("Planning complete. Proceeding to act.")
            return "act"
        else:
            logger.info("Planning not yet complete. Continuing to next plan step.")
            return "continue_planning"
    
    async def _act(self, state: AgentState) -> dict:
        logger.info("Act: Generating comprehensive response based on planning and reasoning")
        
        plan_steps = state.action_plan
        reasoning_list = state.reasoning
        user_query = ""
        for message in reversed(state.messages):
            if message.role == "user":
                user_query = message.content
                break

        if state.error:
            error_message = f"An error occurred: {state.error}"
            updated_messages = state.messages + [Message(role="assistant", content=error_message)]
            logger.error(f"Act: Finalizing with error: {state.error}")
            return {"messages": updated_messages, "error": state.error}

        if not plan_steps or not reasoning_list:
            logger.warning("Act: Plan or reasoning is incomplete. Generating a summary based on available information.")
            fallback_content = "Could not generate a full response due to incomplete planning or reasoning."
            if user_query: fallback_content += f" User query was: {user_query}"
            updated_messages = state.messages + [Message(role="assistant", content=fallback_content)]
            return {"messages": updated_messages}

        system_prompt_act = f"""You are {self.config.name}, a {self.config.role}.
        {self.config.instructions}

        You have at your disposal a detailed, step-by-step plan and comprehensive reasoning for each step in response to the user's query. Your objective is to synthesize this information into a deep, research-quality answer that leverages all aspects of the plan and reasoning. Specifically:
        1. Integrate every plan step and its reasoning into a coherent narrative.
        2. Provide nuanced explanations and justify conclusions with evidence drawn from the reasoning steps.
        3. Structure the response in Markdown:
           - **Introduction**: Summarize the problem and your overall approach.
           - **Sections**: Use clear headings for major themes or steps.
           - **In-Depth Analysis**: Dive deep into each section with detailed insights.
           - **Conclusion**: Highlight key findings and their significance.
        Use lists, emphasis, and references to reasoning steps to ensure completeness. Adopt a scholarly tone reflecting thorough research and expertise.
        """

        plan_and_reasoning_details = f"User Query: {user_query}\n\nAction Plan and Reasoning:\n"
        for i, step in enumerate(plan_steps):
            plan_and_reasoning_details += f"\nStep {i+1}: {step.description}\n"
            if i < len(reasoning_list):
                plan_and_reasoning_details += f"Reasoning for Step {i+1}: {reasoning_list[i].thought}\n"
            else:
                plan_and_reasoning_details += "Reasoning for Step {i+1}: Not available.\n"

        try:
            response = await self.llm.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt_act},
                    {"role": "user", "content": plan_and_reasoning_details}
                ],
            )
            final_content = response.choices[0].message.content
            logger.info("Act: Successfully generated final response from LLM.")
        except Exception as e:
            logger.error(f"Error during final response generation LLM call: {e}")
            final_content = f"Error generating final response: {str(e)}\n\nBased on available information:\n{plan_and_reasoning_details}"

        updated_messages = state.messages + [Message(role="assistant", content=final_content)]
        return {"messages": updated_messages}

    async def _evaluate(self, state: AgentState) -> dict:
        #TODO: Implement proper evaluation
        logger.info("Evaluate: Placeholder implementation")
        return {}

    def _learn(self, state: AgentState) -> dict:
        logger.info(f"Learn: Processing for thread {state.thread_id}")
        if state.thread_id:
            try:
                state_to_save = state.model_dump()
                success = self.memory['thread'].save(state.thread_id, state_to_save)
                if success:
                    logger.info(f"Learn: Successfully saved state for thread {state.thread_id}")
                else:
                    logger.error(f"Learn: Failed to save state for thread {state.thread_id}")
            except Exception as e:
                logger.error(f"Learn: Error saving state for thread {state.thread_id}: {e}")
        else:
            logger.warning("Learn: No thread_id in state, cannot save state.")
        return {}

    def _should_continue(self, state: AgentState) -> str:
        logger.info("Should continue: Routing to 'evaluate'")
        return "evaluate"

    async def invoke(self, input_message: Dict[str, Any], thread_id: str):
        logger.info(f"Invoking agent {self.config.name} for thread {thread_id}")
        
        initial_state = AgentState(
            messages=[Message(**input_message)],
            thread_id=thread_id
        )

        if self.workflow is None:
            logger.info("Compiling workflow")
            self.workflow = self.workflow_builder.compile()

        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 100
        }
        
        final_state_obj = None
        try:
            final_state_obj = await self.workflow.ainvoke(initial_state, config=config)
            logger.info("Workflow complete")

            if final_state_obj:
                if isinstance(final_state_obj, AgentState):
                    return final_state_obj.model_dump()
                elif isinstance(final_state_obj, dict):
                    return final_state_obj
            return None

        except Exception as e:
            logger.error(f"Error during agent execution: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Agent execution failed for thread {thread_id}: {str(e)}"
            )

if __name__ == "__main__":
    print("Running agent test...")
    
    async def test_agent():
        config = AgentConfig(
            name="ReasoningAssistant",
            role="Expert planning assistant that helps with problem solving",
            instructions="Analyze problems carefully and provide detailed plan before giving answers."
        )
        
        agent = BaseAgent(config)
        
        test_message = {
            "role": "user",
            "content": "Explain me like a kid how an LLM works. I want to know how it can answer my questions."
        }
        
        result = await agent.invoke(test_message, "test-thread-1")

        print("Result:", result)
        if not result:
            print("Test failed or returned incomplete results:", result)
            return

        if "messages" in result:
            print("\n=== MESSAGES ===")
            for msg_obj in result["messages"]:
                if isinstance(msg_obj, dict):
                    role = msg_obj.get('role', 'Unknown')
                    content = msg_obj.get('content', '')
                elif hasattr(msg_obj, 'role') and hasattr(msg_obj, 'content'):
                    role = msg_obj.role
                    content = msg_obj.content
                else:
                    role = 'Unknown'
                    content = str(msg_obj)
                print(f"[{role.upper()}]: {content}")

        if "reasoning" in result and result["reasoning"]:
            print("\n=== REASONING ===")
            for idx, reasoning_item in enumerate(result["reasoning"], start=1):
                thought = "N/A"
                if isinstance(reasoning_item, ReasoningOutput):
                    thought = reasoning_item.thought
                elif isinstance(reasoning_item, dict):
                    thought = reasoning_item.get('thought', 'N/A')
                else:
                    thought = str(reasoning_item)
                print(f"Step {idx} reasoning: {thought}")

        if "action_plan" in result and result["action_plan"]:
            print("\n=== ACTION PLAN ===")
            for idx, plan_item in enumerate(result["action_plan"], start=1):
                desc = "N/A"
                last_step = "N/A"
                if isinstance(plan_item, Plan):
                    desc = plan_item.description
                    last_step = plan_item.last_step
                elif isinstance(plan_item, dict):
                    desc = plan_item.get('description', 'N/A')
                    last_step = plan_item.get('last_step', 'N/A')
                else:
                    desc = str(plan_item)
                print(f"Step {idx}: {desc} (Last Step: {last_step})")

    asyncio.run(test_agent())
