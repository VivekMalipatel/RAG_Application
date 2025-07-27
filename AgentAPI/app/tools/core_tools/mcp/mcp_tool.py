import sys
import asyncio
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List,  Literal

from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_mcp_adapters.client import MultiServerMCPClient
from config import config as app_config

from tools.core_tools.mcp.mcp_utils import (
    load_prompt_or_description, load_mcp_config, get_enabled_servers,
    inject_security_context, get_tool_names, fetch_tools
)

sys.path.append(str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)
TOOL_NAME = "multi_server_mcp_wrapper"

class MCPServerConfig(BaseModel):
    name: str = Field(description="Unique name for the MCP server")
    transport: Literal["stdio", "streamable_http"] = Field(
        description="Transport protocol for the MCP server (stdio or streamable_http only)"
    )
    command: Optional[str] = Field(
        default=None,
        description="Command to run for stdio transport"
    )
    args: Optional[List[str]] = Field(
        default=None,
        description="Arguments for the command (stdio transport)"
    )
    url: Optional[str] = Field(
        default=None,
        description="URL for HTTP-based transports"
    )
    env: Optional[Dict[str, str]] = Field(
        default=None,
        description="Environment variables for the server"
    )
    headers: Optional[Dict[str, str]] = Field(
        default=None,
        description="HTTP headers for HTTP-based transports"
    )
    enabled: bool = Field(default=True, description="Whether the server is enabled")
    priority: Optional[int] = Field(default=1, description="Server priority")
    namespace: Optional[str] = Field(default=None, description="Server namespace")
    timeout: Optional[int] = Field(default=30, description="Connection timeout")
    max_retries: Optional[int] = Field(default=3, description="Maximum retry attempts")

def load_server_configs_from_json(json_path: Optional[str] = None) -> List[MCPServerConfig]:
    from config import config as app_config
    if json_path is None:
        json_path = app_config.MCP_JSON_PATH
    
    try:
        with open(json_path, "r") as f:
            config_data = json.load(f)
        
        servers_config = config_data.get("mcp_config", {}).get("servers", {})
        
        configs = []
        for server_name, server_data in servers_config.items():
            # Only process enabled servers
            if not server_data.get("enabled", False):
                logger.info(f"Skipping disabled server: {server_name}")
                continue
            
            # Only support stdio and streamable_http transports
            transport = server_data.get("transport")
            if transport not in ["stdio", "streamable_http"]:
                logger.warning(f"Unsupported transport '{transport}' for server {server_name}. Skipping.")
                continue
            
            try:
                # Create config based on transport type
                config_dict = {
                    "name": server_name,
                    "transport": transport,
                    "enabled": server_data.get("enabled", True),
                    "priority": server_data.get("priority", 1),
                    "namespace": server_data.get("namespace"),
                    "timeout": server_data.get("timeout", 30),
                    "max_retries": server_data.get("max_retries", 3)
                }
                
                if transport == "stdio":
                    # For stdio, we need command and optionally args
                    command = server_data.get("command")
                    if not command:
                        logger.error(f"No command specified for stdio server {server_name}. Skipping.")
                        continue
                    
                    config_dict.update({
                        "command": command,
                        "args": server_data.get("args", []),
                        "env": server_data.get("env", {})
                    })
                
                elif transport == "streamable_http":
                    # For HTTP, we need URL
                    url = server_data.get("url")
                    if not url or url == "stdio":  # Skip if URL is invalid
                        logger.error(f"No valid URL specified for HTTP server {server_name}. Skipping.")
                        continue
                    
                    config_dict.update({
                        "url": url,
                        "headers": server_data.get("headers", {})
                    })
                
                # Validate and create the config
                server_config = MCPServerConfig(**config_dict)
                configs.append(server_config)
                logger.info(f"Loaded server config: {server_name} ({transport})")
                
            except Exception as e:
                logger.error(f"Failed to create config for server {server_name}: {e}")
                continue
        
        logger.info(f"Successfully loaded {len(configs)} MCP server configurations")
        return configs
        
    except FileNotFoundError:
        logger.error(f"Config file not found: {json_path}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file {json_path}: {e}")
        return []
    except Exception as e:
        logger.error(f"Failed to load MCP server configs from {json_path}: {e}")
        return []

class MCPToolRequest(BaseModel):
    tool_name: str = Field(description="Name of the MCP tool to execute")
    arguments: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments to pass to the MCP tool"
    )
    server_name: Optional[str] = Field(
        default=None,
        description="Specific server to use (if not specified, uses first available)"
    )

class MCPBatchRequest(BaseModel):
    requests: List[MCPToolRequest] = Field(
        description="List of MCP tool requests to execute (max 10 requests)"
    )
    server_configs: Optional[List[MCPServerConfig]] = Field(
        default=None,
        description="MCP server configurations to connect to"
    )

def get_tool_description(tool_name: str, yaml_filename: str = "mcp_descriptions.yaml") -> str:
    yaml_path = Path(__file__).parent / yaml_filename
    try:
        return load_prompt_or_description(yaml_path, key=tool_name) or "Multi-server MCP wrapper for executing tools across multiple MCP servers"
    except Exception:
        return "Multi-server MCP wrapper for executing tools across multiple MCP servers"

def build_server_connections(server_configs: List[MCPServerConfig]) -> Dict[str, Dict[str, Any]]:
    """Build server connections dictionary for MultiServerMCPClient"""
    connections = {}
    
    for config in server_configs:
        connection = {"transport": config.transport}
        
        if config.transport == "stdio":
            if not config.command:
                raise ValueError(f"Command required for stdio transport in server {config.name}")
            connection["command"] = config.command
            if config.args:
                connection["args"] = config.args
            if config.env:
                connection["env"] = config.env
                
        elif config.transport == "streamable_http":
            if not config.url:
                raise ValueError(f"URL required for streamable_http transport in server {config.name}")
            connection["url"] = config.url
            if config.headers:
                connection["headers"] = config.headers
            # Only include 'timeout' for streamable_http
            if config.timeout:
                connection["timeout"] = config.timeout

        connections[config.name] = connection
        logger.debug(f"Built connection for {config.name}: {connection}")
    
    return connections


def format_mcp_response(response_data: Any, tool_name: str = "", server_name: str = "") -> Dict[str, Any]:
    """Format MCP response data for consistent output"""
    formatted_response = {"content": []}
    
    if isinstance(response_data, dict):
        # Handle standard MCP response formats
        if "content" in response_data:
            content = response_data["content"]
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and "type" in item:
                        formatted_response["content"].append(item)
                    else:
                        formatted_response["content"].append({"type": "text", "text": str(item)})
            else:
                formatted_response["content"].append({"type": "text", "text": str(content)})
        
        elif "result" in response_data:
            result = response_data["result"]
            formatted_response["content"].append({
                "type": "text", 
                "text": json.dumps(result, indent=2) if isinstance(result, (dict, list)) else str(result)
            })
        
        else:
            # Handle raw response
            formatted_response["content"].append({"type": "text", "text": json.dumps(response_data, indent=2)})
    
    elif isinstance(response_data, list):
        formatted_response["content"].append({"type": "text", "text": json.dumps(response_data, indent=2)})
    
    else:
        formatted_response["content"].append({"type": "text", "text": str(response_data)})
    
    # Add metadata
    metadata_parts = []
    if tool_name:
        metadata_parts.append(f"tool: {tool_name}")
    if server_name:
        metadata_parts.append(f"server: {server_name}")
    
    if metadata_parts:
        formatted_response["content"].append({
            "type": "text", 
            "text": f"[METADATA] {' | '.join(metadata_parts)}"
        })
    
    return formatted_response

    # Use inject_security_context from mcp_utils

@tool(
    name_or_callable=TOOL_NAME,
    description=get_tool_description(TOOL_NAME),
    args_schema=MCPBatchRequest,
    parse_docstring=False,
    infer_schema=True,
    response_format="content"
)
async def multi_server_mcp_wrapper(
    requests: List[MCPToolRequest], 
    config: RunnableConfig, 
    server_configs: Optional[List[MCPServerConfig]] = None
) -> List[str]:
    user_id = config.get("configurable", {}).get("user_id")
    org_id = config.get("configurable", {}).get("org_id")
    
    if not user_id or not org_id:
        return ["Error: user_id and org_id are required in config for security"]
    
    # Use provided configs or defaults
    configs = server_configs or load_server_configs_from_json()
    if not configs:
        return ["Error: No MCP server configurations provided or all servers are disabled"]
    
    # Filter only enabled servers with supported transports
    enabled_configs = [
        config for config in configs 
        if config.enabled and config.transport in ["stdio", "streamable_http"]
    ]
    
    if not enabled_configs:
        return ["Error: No enabled MCP servers with supported transports (stdio, streamable_http) found"]
    
    logger.info(f"Using {len(enabled_configs)} enabled MCP servers")
    
    # Build connection dictionary for MultiServerMCPClient
    connections = build_server_connections(enabled_configs)
    logger.info(f"Built connections: {list(connections.keys())}")
    
    try:
        client = MultiServerMCPClient(connections)
        
        async def execute_tool_request(request: MCPToolRequest) -> str:
            """Execute a single MCP tool request with retry logic"""
            # Find server config for max_retries
            server_cfg = None
            if request.server_name:
                server_cfg = next((cfg for cfg in enabled_configs if cfg.name == request.server_name), None)
                if not server_cfg:
                    return f"Error: Server '{request.server_name}' not found in enabled configs"
            else:
                # Use first available server if no specific server requested
                server_cfg = enabled_configs[0] if enabled_configs else None
            
            max_retries = getattr(server_cfg, 'max_retries', 3) if server_cfg else 3
            server_name = request.server_name or (server_cfg.name if server_cfg else "unknown")
            
            for attempt in range(max_retries):
                try:
                    # Inject security context
                    enhanced_arguments = inject_security_context(request.arguments, config)
                    
                    # Get tools from specific server or all servers
                    if request.server_name and request.server_name in connections:
                        # Use specific server session
                        async with client.session(request.server_name) as session:
                            all_tools = await fetch_tools(client, enabled_configs, connections)
                    else:
                        # Use all available tools
                        all_tools = await fetch_tools(client, enabled_configs, connections)
                    
                    # Find the target tool
                    target_tool = None
                    for tool_obj in all_tools:
                        if hasattr(tool_obj, 'name') and tool_obj.name == request.tool_name:
                            target_tool = tool_obj
                            break
                    
                    if not target_tool:
                        available_tools = get_tool_names(all_tools)
                        return f"Tool '{request.tool_name}' not found. Available tools: {available_tools}"
                    
                    # Execute the tool
                    result = await target_tool.ainvoke(enhanced_arguments)
                    
                    # Format the response
                    formatted_result = format_mcp_response(
                        result, 
                        tool_name=request.tool_name,
                        server_name=server_name
                    )
                    return json.dumps(formatted_result, indent=2)
                    
                except Exception as e:
                    logger.error(f"Attempt {attempt + 1} error executing MCP tool {request.tool_name}: {str(e)}")
                    if attempt >= max_retries - 1:
                        return f"Error executing tool {request.tool_name} after {max_retries} attempts: {str(e)}"
                    
                    # Wait before retry
                    await asyncio.sleep(1)
        
        # Execute all requests
        tasks = [execute_tool_request(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        formatted_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                formatted_results.append(f"Error in request {i}: {str(result)}")
            else:
                formatted_results.append(result)
        
        return formatted_results
            
    except Exception as e:
        logger.error(f"Error in multi_server_mcp_wrapper: {str(e)}")
        return [f"Wrapper error: {str(e)}"]

class MCPRequestBuilder:
    """Helper class for building MCP requests"""
    
    @staticmethod
    def create_browser_request(action: str, **kwargs) -> MCPToolRequest:
        """Create a browser automation request for Docker Gateway"""
        return MCPToolRequest(
            tool_name=action,  # "screenshot", "scrape", "navigate", etc.
            arguments=kwargs,
            server_name="docker-gateway"
        )
    
    @staticmethod
    def create_screenshot_request(url: str) -> MCPToolRequest:
        """Create a screenshot request"""
        return MCPToolRequest(
            tool_name="screenshot",
            arguments={"url": url},
            server_name="docker-gateway"
        )
    
    @staticmethod
    def create_scrape_request(url: str) -> MCPToolRequest:
        """Create a web scraping request"""
        return MCPToolRequest(
            tool_name="scrape",
            arguments={"url": url},
            server_name="docker-gateway"
        )

async def get_available_tools_from_servers() -> List[Dict[str, Any]]:
    tools_list = []
    
    try:
        configs = load_server_configs_from_json()
        enabled_configs = [cfg for cfg in configs if cfg.enabled and cfg.transport in ["stdio", "streamable_http"]]
        
        if not enabled_configs:
            logger.warning("No enabled MCP server configs found")
            return tools_list
            
        connections = build_server_connections(enabled_configs)
        client = MultiServerMCPClient(connections)
        
        tools = await client.get_tools()
        
        for tool in tools:
            tool_info = {
                "name": getattr(tool, 'name', str(tool)),
                "description": getattr(tool, 'description', 'No description available'),
                "server": enabled_configs[0].name if enabled_configs else "unknown",  # For now, assign to first server
                "transport": enabled_configs[0].transport if enabled_configs else "unknown"
            }
            tools_list.append(tool_info)
            
        logger.info(f"Successfully retrieved {len(tools_list)} tools from MCP servers")
        
    except Exception as e:
        logger.error(f"Error getting tools from MCP servers: {e}")
        
    return tools_list

async def test_enabled_servers():
    """Test function to check which servers are enabled"""
    configs = load_server_configs_from_json()
    enabled_configs = [cfg for cfg in configs if cfg.enabled and cfg.transport in ["stdio", "streamable_http"]]
    print(f"Found {len(enabled_configs)} enabled servers:")
    for config in enabled_configs:
        print(f"- {config.name}: {config.transport} @ {getattr(config, 'url', getattr(config, 'command', 'N/A'))}")
    if enabled_configs:
        connections = build_server_connections(enabled_configs)
        client = MultiServerMCPClient(connections)
        for config in enabled_configs:
            try:
                print(f"\nTools for server '{config.name}':")
                async with client.session(config.name) as session:
                    tools = await fetch_tools(client, [config], {config.name: connections[config.name]})
                    if tools:
                        for tool in tools:
                            print(f"  - {getattr(tool, 'name', str(tool))}")
                    else:
                        print("  (No tools found)")
            except Exception as e:
                print(f"  Error fetching tools for {config.name}: {e}")

if __name__ == "__main__":
    print("Testing server configurations...")
    asyncio.run(test_enabled_servers())
    

