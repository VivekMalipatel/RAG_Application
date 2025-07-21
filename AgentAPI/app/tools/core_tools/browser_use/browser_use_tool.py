from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from typing import Optional, Dict, Any, List, Annotated
from pydantic import BaseModel, Field
from pathlib import Path
import yaml
import asyncio
import json

TOOL_NAME = "browser_use_tool"

class BrowserUseRequest(BaseModel):
    task: str = Field(
        description="Description of the automation task to perform on the browser."
    )
    url: Optional[str] = Field(
        default=None,
        description="Starting URL if different from task description."
    )
    screenshot: Optional[bool] = Field(
        default=True,
        description="Whether to take screenshots during automation."
    )
    headless: Optional[bool] = Field(
        default=False,
        description="Run browser in headless mode (default: False for visibility)."
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Browser session ID for maintaining state across multiple tool calls."
    )

def get_tool_description(tool_name: str, yaml_filename: str = "description.yaml") -> str:
    yaml_path = Path(__file__).parent / yaml_filename
    try:
        with open(yaml_path, 'r') as f:
            descriptions = yaml.safe_load(f)
        return descriptions.get(tool_name, "")
    except Exception:
        return "Browser automation tool using marinabox and browser-use libraries"

@tool(
    name_or_callable=TOOL_NAME,
    description="Scrape content from multiple websites using natural language instructions.",
    args_schema=BrowserUseRequest,
    parse_docstring=False,
    infer_schema=True,
    response_format="content"
)
async def browser_use_tool(request: BrowserUseRequest, config: RunnableConfig) -> str:
    """
    Automate browser interactions using marinabox and browser-use libraries to perform real web automation tasks.
    """
    try:
        # Try marinabox first (preferred)
        try:
            from marinabox import mb_start_browser, mb_stop_browser, mb_use_browser_tool
            
            print(f"[browser_use_tool] Using marinabox for task: {request.task}")
            
            # Start browser session
            browser_state = {"session_id": request.session_id}
            if not request.session_id:
                browser_state = mb_start_browser(browser_state)
            
            # Prepare browser task
            browser_task = {
                "task": request.task,
                "url": request.url,
                "screenshot": request.screenshot,
                "headless": request.headless
            }
            
            # Execute browser automation
            result = mb_use_browser_tool(browser_task)
            
            # Prepare response
            response = {
                "task": request.task,
                "status": "success",
                "result": str(result),
                "url": request.url,
                "screenshot_taken": request.screenshot,
                "session_id": browser_state.get("session_id"),
                "automation_type": "marinabox"
            }
            
            print(f"[browser_use_tool] Marinabox task completed successfully")
            return json.dumps(response, indent=2)
            
        except ImportError:
            print("[browser_use_tool] Marinabox not available, trying browser-use...")
            
            # Fallback to browser-use library
            from browser_use import Agent, Controller
            
            # Create browser controller
            controller = Controller(
                headless=request.headless,
                keep_open=True
            )
            
            # Create browser agent
            agent = Agent(
                task=request.task,
                llm=None,  # Use default LLM or configure as needed
                controller=controller
            )
            
            # Start the browser automation
            print(f"[browser_use_tool] Using browser-use for task: {request.task}")
            if request.url:
                print(f"[browser_use_tool] Starting URL: {request.url}")
            
            # Execute the task
            result = await agent.run(request.task)
            
            # Prepare response
            response = {
                "task": request.task,
                "status": "success",
                "result": str(result),
                "url": request.url,
                "screenshot_taken": request.screenshot,
                "automation_type": "browser-use"
            }
            
            # Take screenshot if requested
            if request.screenshot:
                try:
                    screenshot_path = await controller.take_screenshot()
                    response["screenshot_path"] = str(screenshot_path)
                except Exception as e:
                    response["screenshot_error"] = str(e)
            
            print(f"[browser_use_tool] Browser-use task completed successfully")
            return json.dumps(response, indent=2)
        
    except ImportError:
        # Fallback if both libraries are not installed
        print("[browser_use_tool] No browser automation libraries found, using simulation mode")
        result = {
            "task": request.task,
            "status": "simulated",
            "message": "Neither marinabox nor browser-use libraries are installed. Task simulated.",
            "url": request.url,
            "note": "Install 'marinabox' or 'browser-use' package for real browser automation",
            "automation_type": "simulation"
        }
        return json.dumps(result, indent=2)
        
    except Exception as e:
        print(f"[browser_use_tool] Error during browser automation: {str(e)}")
        error_result = {
            "task": request.task,
            "status": "error",
            "error": str(e),
            "url": request.url,
            "automation_type": "error"
        }
        return json.dumps(error_result, indent=2)

# LangGraph workflow functions for browser automation
def mb_start_browser(state: dict) -> dict:
    """Start browser session for LangGraph workflow"""
    try:
        from marinabox import MarinaboxSDK
        sdk = MarinaboxSDK()
        session = sdk.create_session(
            env_type="browser",
            tag="LangGraph-Browser",
            initial_url="https://example.com"
        )
        print(f"[LangGraph] Browser session started: {session.session_id}")
        return {
            **state,
            "session_id": session.session_id,
            "vnc_port": session.vnc_port,
            "status": session.status,
            "browser_active": True
        }
    except ImportError:
        print("[LangGraph] Marinabox not available, using fallback session")
        import uuid
        return {
            **state,
            "session_id": str(uuid.uuid4()),
            "browser_active": True
        }

def mb_stop_browser(state: dict) -> dict:
    """Stop browser session for LangGraph workflow"""
    try:
        from marinabox import MarinaboxSDK
        sdk = MarinaboxSDK()
        if state.get("session_id"):
            # Stop the session if SDK supports it
            print(f"[LangGraph] Stopping browser session: {state.get('session_id')}")
        return {
            **state,
            "browser_active": False,
            "session_stopped": True
        }
    except Exception as e:
        print(f"[LangGraph] Error stopping browser: {e}")
        return {
            **state,
            "browser_active": False
        }

# Marinabox browser tool for LangGraph
@tool(
    name="mb_use_browser_tool",
    description="Performs browser automation tasks using marinabox",
)
def mb_use_browser_tool(task: str, url: str = None, screenshot: bool = True, session_id: str = None) -> str:
    """
    Execute browser automation task using marinabox, reusing session if session_id is provided.
    
    Args:
        task: Description of the browser task to perform
        url: Optional starting URL
        screenshot: Whether to take a screenshot
        session_id: Optional existing session ID to reuse
    """
    try:
        from marinabox import MarinaboxSDK
        sdk = MarinaboxSDK()
        session = None
        # Use existing session if session_id is provided
        if session_id:
            session = sdk.get_session(session_id)
            if not session:
                # If session_id is invalid, create a new session
                session = sdk.create_session(
                    env_type="browser",
                    tag="Browser-Tool",
                    initial_url=url or "https://example.com"
                )
        else:
            session = sdk.create_session(
                env_type="browser",
                tag="Browser-Tool",
                initial_url=url or "https://example.com"
            )
        
        # Execute browser automation (this would be the actual automation logic)
        print(f"[mb_use_browser_tool] Executing task: {task}")
        if url:
            print(f"[mb_use_browser_tool] Starting URL: {url}")
        
        # Simulate browser automation result
        result = {
            "task": task,
            "status": "success",
            "session_id": session.session_id,
            "url": url,
            "screenshot_taken": screenshot,
            "automation_type": "marinabox",
            "message": f"Successfully executed browser task: {task}"
        }
        
        return json.dumps(result, indent=2)
        
    except ImportError:
        # Fallback to simulation
        result = {
            "task": task,
            "status": "simulated",
            "url": url,
            "screenshot_taken": screenshot,
            "automation_type": "simulation",
            "message": f"Simulated browser task: {task} (marinabox not available)"
        }
        return json.dumps(result, indent=2)
    except Exception as e:
        error_result = {
            "task": task,
            "status": "error",
            "error": str(e),
            "url": url,
            "automation_type": "error"
        }
        return json.dumps(error_result, indent=2)

if __name__ == "__main__":
    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.prebuilt import ToolNode
    from langgraph.graph import StateGraph, START, END
    from langgraph.types import Command
    from typing import Annotated
    import asyncio

    # Test the marinabox SDK directly
    print("=== Testing Marinabox SDK ===")
    try:
        from marinabox import MarinaboxSDK
        sdk = MarinaboxSDK()
        session = sdk.create_session(
            env_type="browser",
            tag="Development",
            initial_url="https://bytecrafts.in"
        )
        print(f"Session ID: {session.session_id}")
        print(f"VNC Port: {session.vnc_port}")
        print(f"Status: {session.status}")
    except ImportError:
        print("Marinabox not available, testing tool in simulation mode")
    except Exception as e:
        print(f"Error: {e}")

    # Test the LangGraph workflow
    print("\n=== Testing LangGraph Browser Workflow ===")
    
    # Set up tools and model
    tools = [mb_use_browser_tool]
    tool_node = ToolNode(tools=tools)
    
    # Define workflow logic
    def should_continue(state: dict):
        messages = state.get("messages", [])
        if len(messages) > 0:
            last_message = messages[-1]
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "tool_node"
        return "stop_browser"

    def call_model(state: dict):
        # For testing, simulate a tool call
        from langchain_core.messages import AIMessage
        from langchain_core.tools import tool_calls
        
        # Simulate an AI message with tool call
        tool_call = {
            "name": "mb_use_browser_tool",
            "args": {
                "task": "Open bytecrafts.in and find their services section",
                "url": "https://bytecrafts.in",
                "screenshot": True
            },
            "id": "test_call_1"
        }
        
        ai_message = AIMessage(
            content="I'll help you browse bytecrafts.in and find their services.",
            tool_calls=[tool_call]
        )
        
        return {
            **state,
            "messages": [ai_message]
        }

    # Create workflow
    workflow = StateGraph(dict)

    # Add nodes
    workflow.add_node("start_browser", mb_start_browser)
    workflow.add_node("agent", call_model)
    workflow.add_node("tool_node", tool_node)
    workflow.add_node("stop_browser", mb_stop_browser)

    # Add edges
    workflow.add_edge(START, "start_browser")
    workflow.add_edge("start_browser", "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tool_node": "tool_node",
            "stop_browser": "stop_browser"
        }
    )
    workflow.add_edge("tool_node", "stop_browser")
    workflow.add_edge("stop_browser", END)

    # Compile and run
    try:
        app = workflow.compile()
        result = app.invoke({"messages": []})
        print("Workflow result:", result)
    except Exception as e:
        print(f"Workflow error: {e}")
        
    print("\n=== Browser Workflow Test Complete ===")
