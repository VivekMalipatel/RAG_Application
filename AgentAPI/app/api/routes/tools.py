from fastapi import APIRouter
from tools.agents_as_tools import AVAILABLE_TOOLS
import json

router = APIRouter()

def get_tools():
    """Return the list of available tools in OpenAI-compatible structure."""
    return AVAILABLE_TOOLS

@router.get("/v1/tools")
def list_tools():
    tool_calls = []
    for tool in AVAILABLE_TOOLS:
        func = tool["function"].copy()
        args = {}
        for k in func["parameters"]["properties"]:
            args[k] = "example_value"
        func["arguments"] = json.dumps(args)
        tool_calls.append({
            "id": f"{func['name']}",
            "type": "function",
            "function": func
        })
    return {"data": tool_calls}
