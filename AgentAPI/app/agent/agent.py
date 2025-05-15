import os
from openai import AsyncOpenAI
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from sqlalchemy.orm import Session
from app.db.db import SessionLocal
from app.agent.memory import ThreadMemoryHandler, AgentMemoryHandler, ClientMemoryHandler
from fastapi import HTTPException
from app.config import settings
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

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
    model_name: str = "gemma3:12b-it-q8_0"
    llm_config: Dict[str, Any] = Field(default_factory=dict)

class AgentState(BaseModel):
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    reasoning: Dict[str, Any] = Field(default_factory=dict)
    action_plan: List[Dict[str, Any]] = Field(default_factory=list)
    thread_id: Optional[str] = None
    answer: Optional[str] = None

class Plan(BaseModel):
    description: str
    last_step: bool = False

class Reason(BaseModel):
    reasoning: str

class PlanandReason(BaseModel):
    plan: List[Plan] = Field(default_factory=list)
    reasoning: List[Dict[str, Any]] = Field(default_factory=list)
    reasoning_queue: asyncio.Queue = Field(default_factory=lambda: asyncio.Queue(), exclude=True)
    reasoning_tasks: List[asyncio.Task] = Field(default_factory=list, exclude=True)
    
    model_config = {
        "arbitrary_types_allowed": True
    }
    
    def add_step(self, plan_step: Plan):
        self.plan.append(plan_step)
        self.reasoning_queue.put_nowait(len(self.plan) - 1)
        
    def get_last_step(self) -> Optional[Plan]:
        if self.plan:
            return self.plan[-1]
        return None
    
    def get_plan(self) -> List[Plan]:
        return self.plan
    
    def add_reasoning(self, index: int, reasoning: Dict[str, Any]):
        while len(self.reasoning) <= index:
            self.reasoning.append({})
        self.reasoning[index] = reasoning
    
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
        self.workflow_builder.add_node("planandreason", self._planandreason)
        self.workflow_builder.add_node("act", self._act)
        self.workflow_builder.add_node("evaluate", self._evaluate)
        self.workflow_builder.add_node("learn", self._learn)

        self.workflow_builder.set_entry_point("perceive")
        self.workflow_builder.add_edge("perceive", "planandreason")
        self.workflow_builder.add_edge("planandreason", "act")
        self.workflow_builder.add_conditional_edges(
            "act",
            self._should_continue,
            {"continue": "evaluate", "end": END}
        )
        self.workflow_builder.add_edge("evaluate", "learn")
        self.workflow_builder.add_edge("learn", END)

    def _perceive(self, state: AgentState) -> dict:
        # #TODO: Implement perception in the future
        logger.info("Perceive: Placeholder implementation")
        return {}

    async def _planandreason(self, state: AgentState) -> dict:
        logger.info("Plan: Implementing iterative reasoning")
        
        user_message = None
        for message in state.messages:
            if message.get("role") == "user":
                user_message = message
                break
        
        if not user_message:
            logger.warning("No user message found in state")
            return {"reasoning": {"error": "No user message found"}}
        
        query = user_message.get("content", "")
        if not query:
            logger.warning("User message has no content")
            return {"reasoning": {"error": "User message has no content"}}
        
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
                "description": "Analyze the user's request to identify key entities and intents.",
                "last_step": false
            }}

            Example of a valid JSON output for the final step:
            {{
                "description": "Compile all gathered information into a final report and present it to the user.",
                "last_step": true
            }}

            Focus solely on generating the next single step in the plan. Do not attempt to solve the entire problem in one go.
            """
        
        system_prompt_reason = f"""You are {self.config.name}, a reasoning agent with the role: {self.config.role}.
            {self.config.instructions}

            You will be given a specific step in a plan and the original task. Your job is to provide detailed reasoning or execution for this specific step.
            Provide a comprehensive answer or solution for the specific step you've been given.
            Focus only on the current step, not the entire task. Be thorough but specific.
            
            Your output should be a detailed reasoning or solution for the given step.
            """
        
        plan_and_reason = PlanandReason(plan=[], reasoning=[])
        
        reasoning_worker_task = asyncio.create_task(
            self._reasoning_worker(query, system_prompt_reason, plan_and_reason)
        )
        
        while plan_and_reason.get_last_step() is None or not plan_and_reason.get_last_step().last_step:
            logger.info("Planning step...")
            
            plan_list_str = "\n".join([f"Step {i+1}: {step.description}" for i, step in enumerate(plan_and_reason.get_plan())])
            prompt_query = f"This is the task/problem : {query}\n\nPlan as of now : \n{plan_list_str}"
            
            # call LLM to get next plan step and parse JSON into Plan
            response = await self.llm.beta.chat.completions.parse(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt_plan},
                    {"role": "user", "content": prompt_query}
                ],
                response_format=Plan,
            )

            plan_and_reason.add_step(response.choices[0].message.parsed)
            logger.info(f"Generated step: {response.choices[0].message.parsed.description}")
            
            await asyncio.sleep(0.1)
        
        logger.info("Waiting for reasoning queue to be fully processed...")
        await plan_and_reason.reasoning_queue.join()
        logger.info("All reasoning tasks processed.")

        reasoning_worker_task.cancel()
        try:
            await reasoning_worker_task
        except asyncio.CancelledError:
            logger.info("Reasoning worker task successfully cancelled after completion.")
        except Exception as e:
            logger.error(f"Error awaiting cancelled reasoning worker task: {e}")

        # Convert plan and reasoning into lists for state
        plan_list = [step.dict() for step in plan_and_reason.plan]
        reasoning_list = plan_and_reason.reasoning
        return {"action_plan": plan_list, "reasoning": reasoning_list}

    async def _reasoning_worker(self, query: str, system_prompt: str, plan_and_reason: PlanandReason):
        try:
            while True:
                try:
                    # Try to get an item from the queue with a short timeout
                    try:
                        step_index = await asyncio.wait_for(plan_and_reason.reasoning_queue.get(), 0.5)
                    except asyncio.TimeoutError:
                        # Check if planning is complete and queue is empty
                        planning_complete = (plan_and_reason.get_last_step() is not None and 
                                           plan_and_reason.get_last_step().last_step)
                        if planning_complete and plan_and_reason.reasoning_queue.empty():
                            logger.info("Reasoning worker: Planning complete and queue empty. Exiting.")
                            break
                        continue
                    
                    # Safety check in case the index is out of bounds
                    if step_index >= len(plan_and_reason.plan):
                        logger.warning(f"Reasoning worker: step_index {step_index} out of bounds for plan length {len(plan_and_reason.plan)}. Waiting...")
                        # Re-queue the item for later processing when plan is updated
                        await asyncio.sleep(0.2)
                        plan_and_reason.reasoning_queue.put_nowait(step_index)
                        continue

                    step = plan_and_reason.plan[step_index]
                    
                    logger.info(f"Generating reasoning for step {step_index + 1}: {step.description}")
                    
                    prompt = f"""Original task: {query}
                    
                    Step to reason about: {step.description}
                    
                    Provide detailed reasoning or execution for this specific step:"""
                    
                    response = await self.llm.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ]
                    )
                    
                    reasoning_text = response.choices[0].message.content
                    
                    plan_and_reason.add_reasoning(step_index, {"reasoning": reasoning_text})
                    
                    logger.info(f"Completed reasoning for step {step_index + 1}")
                    
                    # Mark task as done
                    plan_and_reason.reasoning_queue.task_done()
                
                except asyncio.CancelledError:
                    logger.info("Reasoning worker cancelled during processing")
                    raise
                except Exception as e:
                    logger.error(f"Error processing reasoning step: {e}")
                    try:
                        # Make sure to mark as done even if processing fails
                        plan_and_reason.reasoning_queue.task_done()
                    except Exception:
                        pass

        except asyncio.CancelledError:
            logger.info("Reasoning worker cancelled")
            # Clean up the queue before exiting
            while not plan_and_reason.reasoning_queue.empty():
                try:
                    plan_and_reason.reasoning_queue.get_nowait()
                    plan_and_reason.reasoning_queue.task_done()
                except Exception:
                    pass
            raise

    async def _act(self, state: AgentState) -> dict:
        logger.info("Act: Generating comprehensive response based on planning and reasoning")
        
        # Retrieve action plan and reasoning lists from state
        plan_steps = state.action_plan
        reasoning_list = state.reasoning

        user_query = ""
        for message in state.messages:
            if message.get("role") == "user":
                user_query = message.get("content", "")
                break
        
        system_prompt = f"""You are {self.config.name}, a {self.config.role}.
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
        
        # Build content from action plan and reasoning
        steps_content = []
        for i, step in enumerate(plan_steps):
            desc = step.get("description", "")
            step_content = f"Step {i+1}: {desc}\n"
            if i < len(reasoning_list) and reasoning_list[i].get("reasoning"):
                step_content += f"Reasoning: {reasoning_list[i]['reasoning']}\n"
            steps_content.append(step_content)
        
        all_steps = "\n".join(steps_content)
        
        user_content = f"""Original query: {user_query}
        
        Here is the plan and reasoning I've developed to answer this query:
        
        {all_steps}
        
        Please provide a comprehensive answer to the original query based on this information. Format your response in Markdown.
        """
        
        try:
            response = await self.llm.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ]
            )
            
            answer = response.choices[0].message.content
            
            logger.info("Generated comprehensive answer based on plan and reasoning")
            
            return {
                "messages": state.messages + [
                    {"role": "assistant", "content": answer}
                ],
                "answer": answer
            }
            
        except Exception as e:
            logger.error(f"Error generating response in _act: {e}")

    async def _evaluate(self, state: AgentState) -> dict:
        # #TODO: Implement proper evaluation
        logger.info("Evaluate: Placeholder implementation")
        return {}

    def _learn(self, state: AgentState) -> dict:
        # #TODO: Implement learning from interactions
        logger.info("Learn: Placeholder implementation")
        return {}

    def _should_continue(self, state: AgentState) -> str:
        logger.info("Should continue: Returning 'end'")
        return "end"

    async def invoke(self, input_message: Dict[str, Any], thread_id: str):
        logger.info(f"Invoking agent {self.config.name} for thread {thread_id}")
        
        initial_state = AgentState(
            messages=[input_message],
            thread_id=thread_id
        )

        if self.workflow is None:
            logger.info("Compiling workflow")
            self.workflow = self.workflow_builder.compile()

        config = {"configurable": {"thread_id": thread_id}}
        final_state_obj = None

        try:
            async for step_output in self.workflow.astream(initial_state, config=config):
                node_name = list(step_output.keys())[0]
                final_state_obj = step_output[node_name]
                logger.info(f"Executed node: {node_name}")

            logger.info("Workflow complete")
            if final_state_obj:
                if isinstance(final_state_obj, dict):
                    return final_state_obj
                elif hasattr(final_state_obj, "dict"):
                    return final_state_obj.dict()
            return None

        except Exception as e:
            logger.error(f"Error during agent execution: {e}")
            self.db.rollback()
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
            "content": "Explain the process of photosynthesis and why it's important for life on Earth."
        }
        
        result = await agent.invoke(test_message, "test-thread-1")

        # Display result details
        print("Result:", result)
        if not result:
            print("Test failed or returned incomplete results:", result)
            return

        # Print messages if present
        if "messages" in result:
            print("\n=== MESSAGES ===")
            for msg in result["messages"]:
                print(f"[{msg['role'].upper()}]: {msg['content']}")

        # Print final answer
        if "answer" in result:
            print("\n=== ANSWER ===")
            print(result["answer"])

        # Print reasoning steps if present
        if "reasoning" in result:
            print("\n=== REASONING ===")
            for idx, reasoning in enumerate(result["reasoning"], start=1):
                print(f"Step {idx} reasoning: {reasoning}")

        # Print action plan if present
        if "action_plan" in result:
            print("\n=== ACTION PLAN ===")
            for idx, step in enumerate(result["action_plan"], start=1):
                desc = step.get('description', '')
                print(f"Step {idx}: {desc}")

    asyncio.run(test_agent())
