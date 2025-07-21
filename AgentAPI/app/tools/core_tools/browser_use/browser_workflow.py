from marinabox import mb_start_browser, mb_stop_browser, mb_use_browser_tool
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langgraph.types import Command

# Set up tools and model
SESSION_ID = "f4038c420cbb"
def mb_use_browser_tool_with_session(*args, **kwargs):
    """Browser automation tool using a fixed session ID for persistent sessions."""
    kwargs["session_id"] = SESSION_ID
    return mb_use_browser_tool(*args, **kwargs)
tools = [mb_use_browser_tool_with_session]
tool_node = ToolNode(tools=tools)
model_with_tools = ChatOpenAI(
    model="o3",  # or your preferred OpenAI model
    temperature=0,
    base_url="https://llm.gauravshivaprasad.com/v1",  # <-- set your custom OpenAI URL here
    api_key="sk-372c69b72fb14a90a2e1b0b17884d9b4",
).bind_tools(tools)

# Define workflow logic
def should_continue(state: dict):
    messages = state["messages"]
    print("[DEBUG] should_continue, messages:", messages)
    # Only allow one tool call for demo, then stop
    if state.get("_tool_called"):
        print("[DEBUG] Tool already called, stopping browser.")
        return Command(goto="stop_browser")
    if len(messages) > 0:
        last_message = messages[-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            print("[DEBUG] Tool call detected, going to tool_node.")
            state["_tool_called"] = True
            return Command(goto="tool_node")
    print("[DEBUG] No tool call, stopping browser.")
    return Command(goto="stop_browser")

def call_model(state: dict):
    # Automatically instruct the agent to open bytecrafts.in and list all products
    messages = [
        HumanMessage(content="Open https://bytecrafts.in and list all the products on the site.")
    ]
    response = model_with_tools.invoke(messages)
    return {
        "messages": [response],
        "session_id": state.get("session_id")
    }

# Create workflow
workflow = StateGraph(dict)

# Add nodes
workflow.add_node("agent", call_model)
workflow.add_node("tool_node", tool_node)
workflow.add_node("stop_browser", mb_stop_browser)
workflow.add_node("should_continue", should_continue)

# Add edges
workflow.add_edge(START, "agent")
workflow.add_edge("tool_node", "agent")
workflow.add_edge("agent", "should_continue")
workflow.add_edge("stop_browser", END)

# Compile and run

if __name__ == "__main__":
    # Print VNC port for the session
    try:
        from marinabox import MarinaboxSDK
        sdk = MarinaboxSDK()
        session = sdk.get_session(SESSION_ID)
        print(f"[INFO] VNC port for session {SESSION_ID}: {getattr(session, 'vnc_port', 'unknown')}")
    except Exception as e:
        print(f"[WARN] Could not fetch VNC port: {e}")

    app = workflow.compile()
    print("Starting browser workflow with session_id:", SESSION_ID)
    result = app.invoke({"messages": [], "session_id": SESSION_ID})
    print("Workflow result:", result)