"""
Shared utilities for MCP tool and agent.
"""
import json
import yaml
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)

# === YAML Loader ===
def load_yaml(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# === Prompt/Description Loader ===
def load_prompt_or_description(path: Path, key: Optional[str] = None) -> str:
    data = load_yaml(path)
    if key and isinstance(data, dict):
        return data.get(key, "")
    if isinstance(data, str):
        return data
    return ""

# === MCP Config Loader ===
def load_mcp_config(mcp_config_path: str) -> Dict[str, Any]:
    try:
        with open(mcp_config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        logger.warning(f"⚠️  MCP config file not found: {mcp_config_path}")
        return {}
    except json.JSONDecodeError as e:
        logger.warning(f"⚠️  Invalid JSON in MCP config: {e}")
        return {}

# === Enabled Servers ===
def get_enabled_servers(mcp_config: Dict[str, Any]) -> List[str]:
    if not mcp_config:
        return []
    servers = mcp_config.get('mcp_config', {}).get('servers', {})
    return [name for name, conf in servers.items() if conf.get('enabled', False)]

# === Security Context Injection ===
def inject_security_context(arguments: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    user_id = config.get("configurable", {}).get("user_id")
    org_id = config.get("configurable", {}).get("org_id")
    if not user_id or not org_id:
        return arguments
    enhanced_args = arguments.copy()
    if "user_id" not in enhanced_args:
        enhanced_args["user_id"] = user_id
    if "org_id" not in enhanced_args:
        enhanced_args["org_id"] = org_id
    return enhanced_args

# === Tool Names from Objects ===
def get_tool_names(tool_objs: List[Any]) -> List[str]:
    return [getattr(t, 'name', str(t)) for t in tool_objs]

# === Tool Discovery (Async) ===
async def fetch_tools(client, enabled_configs, connections):
    try:
        tools = await client.get_tools()
        logger.info(f"✅ Fetched {len(tools)} tool objects from MCP servers.")
        return tools
    except Exception as e:
        logger.error(f"⚠️  Error fetching tool objects: {e}")
        return []
