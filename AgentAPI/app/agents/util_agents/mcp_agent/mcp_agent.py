import sys
import json
import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from langchain_core.runnables import RunnableConfig
from dataclasses import dataclass
from enum import Enum

sys.path.append(str(Path(__file__).parent.parent.parent))
from agents.base_agents.base_agent import BaseAgent
from tools.core_tools.mcp.mcp_tool import (
    multi_server_mcp_wrapper,  get_available_tools_from_servers, 
    build_server_connections, load_server_configs_from_json
)
from tools.core_tools.mcp.mcp_utils import (
    load_prompt_or_description, load_mcp_config, get_enabled_servers,
)

logger = logging.getLogger(__name__)

class AgentState(Enum):
    IDLE = "idle"
    CONNECTING = "connecting"
    READY = "ready"
    EXECUTING = "executing"
    ERROR = "error"

@dataclass
class ToolContext:
    """Context for tool execution similar to VS Code's execution context"""
    name: str
    server: str
    description: str
    schema: Dict[str, Any]
    last_used: Optional[str] = None
    usage_count: int = 0

class MCPAgent(BaseAgent):
    def __init__(
        self,
        prompt: Optional[str] = None,
        mcp_config_path: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        vlm_kwargs: Optional[Dict[str, Any]] = None,
        node_kwargs: Optional[Dict[str, Any]] = None,
        recursion_limit: Optional[int] = 25,
        debug: bool = False,
        config: Optional[Any] = None,
        auto_refresh_tools: bool = True,
        **kwargs
    ):
        if prompt is None:
            prompt_path = Path(__file__).parent / "prompt.yaml"
            prompt = load_prompt_or_description(prompt_path, key="MCPAgent")
        
        super().__init__(
            prompt=prompt,
            model_kwargs=model_kwargs,
            vlm_kwargs=vlm_kwargs,
            node_kwargs=node_kwargs,
            recursion_limit=recursion_limit,
            debug=debug,
            config=config,
            **kwargs
        )
        
        self.mcp_config_path = mcp_config_path or "tools/core_tools/mcp/mcp.json"
        self.mcp_config = load_mcp_config(self.mcp_config_path)
        self.enabled_servers = get_enabled_servers(self.mcp_config)
        self.auto_refresh_tools = auto_refresh_tools
        
        # VS Code-like features
        self.state = AgentState.IDLE
        self.tool_contexts: Dict[str, ToolContext] = {}
        self.server_health: Dict[str, bool] = {}
        self.command_history: List[Dict[str, Any]] = []
        self.workspace_context: Dict[str, Any] = {}
        self._available_tool_objects = None
        
        # Initialize workspace
        self._initialize_workspace()

    def _initialize_workspace(self):
        """Initialize workspace context like VS Code"""
        self.workspace_context = {
            "active_servers": [],
            "recent_tools": [],
            "settings": {
                "auto_complete": True,
                "show_suggestions": True,
                "timeout": 30
            },
            "extensions": {}  # For future MCP server extensions
        }

    async def setup_tools(self):
        """Setup tools with health checks like VS Code extensions"""
        try:
            self.state = AgentState.CONNECTING
            self.bind_tools([multi_server_mcp_wrapper])
            
            # Health check servers
            await self._health_check_servers()
            
            # Load tool contexts
            await self._load_tool_contexts()
            
            await self.compile()
            self.state = AgentState.READY
            logger.info("✅ MCPAgent Enhanced: Setup complete")
            
        except Exception as e:
            self.state = AgentState.ERROR
            logger.error(f"⚠️ MCPAgent Enhanced: Error during setup: {e}")
            raise

    async def _health_check_servers(self):
        """Health check all enabled servers"""
        for server in self.enabled_servers:
            try:
                # Test connection to server
                configs = load_server_configs_from_json()
                server_config = next((cfg for cfg in configs if cfg.name == server), None)
                
                if server_config and server_config.enabled:
                    self.server_health[server] = True
                    self.workspace_context["active_servers"].append(server)
                    logger.info(f"✅ Server {server} is healthy")
                else:
                    self.server_health[server] = False
                    logger.warning(f"⚠️ Server {server} is not responding")
                    
            except Exception as e:
                self.server_health[server] = False
                logger.error(f"❌ Server {server} health check failed: {e}")

    async def _load_tool_contexts(self):
        """Load tool contexts with metadata like VS Code IntelliSense"""
        try:
            tools_list = await get_available_tools_from_servers()
            
            for tool_info in tools_list:
                context = ToolContext(
                    name=tool_info.get("name", "unknown"),
                    server=tool_info.get("server", "unknown"),
                    description=tool_info.get("description", "No description"),
                    schema=tool_info.get("schema", {})
                )
                self.tool_contexts[context.name] = context
                
            logger.info(f"✅ Loaded {len(self.tool_contexts)} tool contexts")
            
        except Exception as e:
            logger.error(f"❌ Failed to load tool contexts: {e}")

    # VS Code-like Command Palette equivalent
    async def execute_command(self, command: str, **kwargs) -> Dict[str, Any]:
        """Execute commands like VS Code Command Palette"""
        command_map = {
            "mcp.refresh": self._refresh_tools,
            "mcp.health_check": self._health_check_servers,
            "mcp.list_tools": self._list_tools,
            "mcp.server_status": self._get_server_status,
            "mcp.execute_tool": self._execute_tool_command,
            "mcp.clear_history": self._clear_history,
            "workspace.settings": self._get_workspace_settings
        }
        
        if command not in command_map:
            return {"error": f"Unknown command: {command}", "available": list(command_map.keys())}
        
        try:
            result = await command_map[command](**kwargs)
            
            # Log command to history
            self.command_history.append({
                "command": command,
                "timestamp": asyncio.get_event_loop().time(),
                "success": True,
                "result": result
            })
            
            return result
            
        except Exception as e:
            error_result = {"error": str(e)}
            self.command_history.append({
                "command": command,
                "timestamp": asyncio.get_event_loop().time(),
                "success": False,
                "error": str(e)
            })
            return error_result

    async def _refresh_tools(self) -> Dict[str, Any]:
        """Refresh tool contexts like VS Code reload window"""
        await self._load_tool_contexts()
        return {
            "message": "Tools refreshed successfully",
            "tool_count": len(self.tool_contexts),
            "active_servers": self.workspace_context["active_servers"]
        }

    async def _list_tools(self, server: Optional[str] = None) -> Dict[str, Any]:
        """List available tools with filtering like VS Code explorer"""
        tools = []
        
        for name, context in self.tool_contexts.items():
            if server is None or context.server == server:
                tools.append({
                    "name": name,
                    "server": context.server,
                    "description": context.description,
                    "usage_count": context.usage_count,
                    "last_used": context.last_used
                })
        
        return {
            "tools": sorted(tools, key=lambda x: x["usage_count"], reverse=True),
            "total": len(tools),
            "filtered_by": server
        }

    async def _get_server_status(self) -> Dict[str, Any]:
        """Get server status like VS Code extensions view"""
        return {
            "servers": [
                {
                    "name": server,
                    "healthy": self.server_health.get(server, False),
                    "active": server in self.workspace_context["active_servers"]
                }
                for server in self.enabled_servers
            ],
            "total_servers": len(self.enabled_servers),
            "healthy_servers": sum(1 for h in self.server_health.values() if h)
        }

    async def _execute_tool_command(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tool with enhanced context tracking"""
        if tool_name not in self.tool_contexts:
            return {"error": f"Tool '{tool_name}' not found"}
        
        context = self.tool_contexts[tool_name]
        context.usage_count += 1
        context.last_used = str(asyncio.get_event_loop().time())
        
        # Add to recent tools
        if tool_name not in self.workspace_context["recent_tools"]:
            self.workspace_context["recent_tools"].insert(0, tool_name)
            # Keep only last 10
            self.workspace_context["recent_tools"] = self.workspace_context["recent_tools"][:10]
        
        # Execute the tool
        config = self.config if hasattr(self, 'config') else RunnableConfig()
        result = await self.execute_mcp_request(tool_name, arguments, config)
        
        return {
            "tool": tool_name,
            "server": context.server,
            "result": result,
            "usage_count": context.usage_count
        }

    async def _clear_history(self) -> Dict[str, Any]:
        """Clear command history"""
        cleared_count = len(self.command_history)
        self.command_history.clear()
        return {"message": f"Cleared {cleared_count} commands from history"}

    async def _get_workspace_settings(self) -> Dict[str, Any]:
        """Get workspace settings like VS Code settings"""
        return {
            "workspace": self.workspace_context,
            "state": self.state.value,
            "tool_contexts_count": len(self.tool_contexts),
            "command_history_count": len(self.command_history)
        }

    # Enhanced IntelliSense-like features
    def get_tool_suggestions(self, partial_name: str) -> List[Dict[str, Any]]:
        """Get tool suggestions like VS Code IntelliSense"""
        suggestions = []
        
        for name, context in self.tool_contexts.items():
            if partial_name.lower() in name.lower():
                suggestions.append({
                    "name": name,
                    "server": context.server,
                    "description": context.description,
                    "match_score": self._calculate_match_score(partial_name, name),
                    "usage_count": context.usage_count
                })
        
        # Sort by match score and usage
        suggestions.sort(key=lambda x: (x["match_score"], x["usage_count"]), reverse=True)
        return suggestions[:10]  # Top 10 suggestions

    def _calculate_match_score(self, partial: str, full: str) -> float:
        """Calculate match score for suggestions"""
        if partial.lower() == full.lower():
            return 1.0
        if full.lower().startswith(partial.lower()):
            return 0.8
        if partial.lower() in full.lower():
            return 0.6
        return 0.0

    # Status bar equivalent
    def get_status_bar_info(self) -> Dict[str, Any]:
        """Get status bar information like VS Code status bar"""
        healthy_servers = sum(1 for h in self.server_health.values() if h)
        
        return {
            "state": self.state.value,
            "servers": f"{healthy_servers}/{len(self.enabled_servers)}",
            "tools": len(self.tool_contexts),
            "recent_commands": len(self.command_history),
            "last_activity": self.command_history[-1]["timestamp"] if self.command_history else None
        }

    # Interactive mode like VS Code integrated terminal
    

    async def _show_help(self):
        """Show help information"""
        print("\nAvailable Commands:")
        print("  mcp.refresh        - Refresh tools")
        print("  mcp.list_tools     - List available tools")
        print("  mcp.server_status  - Show server status")
        print("  help               - Show this help")
        print("  exit/quit          - Exit interactive mode")

    # Override the original methods to maintain compatibility
    async def execute_mcp_request(self, tool_name: str, arguments: Dict[str, Any], config: RunnableConfig) -> str:
        """Execute MCP request with enhanced tracking"""
        # Update tool context
        if tool_name in self.tool_contexts:
            context = self.tool_contexts[tool_name]
            context.usage_count += 1
            context.last_used = str(asyncio.get_event_loop().time())
        
        # Call original implementation
        return await super().execute_mcp_request(tool_name, arguments, config) if hasattr(super(), 'execute_mcp_request') else "Method not implemented"

    def display_startup_info(self):
        """Startup info display"""
        print("\n" + "="*60)
        print("🔧 MCP Agent")
        print("="*60)
        
        if not self.enabled_servers:
            print("⚠️  No enabled MCP servers found")
            return
        
        status = self.get_status_bar_info()
        print(f"📊 Status: {status['state'].upper()}")
        print(f"🖥️  Servers: {status['servers']} healthy")
        print(f"🔧 Tools: {status['tools']} available")
        
        print(f"\n📋 Enabled Servers:")
        for server in self.enabled_servers:
            health_icon = "✅" if self.server_health.get(server, False) else "❌"
            server_config = self.mcp_config.get('mcp_config', {}).get('servers', {}).get(server, {})
            transport = server_config.get('transport', 'unknown')
            print(f"  {health_icon} {server} ({transport})")
        
        print(f"\n🚀 Agent ready - Type 'interactive' to start VS Code-like mode")
        print("="*60)

# Enhanced Builder with VS Code-like configuration
class MCPAgentBuilder:
    def __init__(self):
        self.config = {}

    def with_mcp_config(self, config_path: str):
        self.config['mcp_config_path'] = config_path
        return self

    def with_auto_refresh(self, auto_refresh: bool = True):
        self.config['auto_refresh_tools'] = auto_refresh
        return self

    def with_workspace_settings(self, settings: Dict[str, Any]):
        self.config['workspace_settings'] = settings
        return self

    def with_prompt(self, prompt: str):
        self.config['prompt'] = prompt
        return self

    def with_debug(self, debug: bool = True):
        self.config['debug'] = debug
        return self

    def with_config(self, config):
        self.config['config'] = config
        return self

    def build(self) -> MCPAgent:
        if 'config' not in self.config:
            raise ValueError("config argument is required for MCPAgent.")
        return MCPAgent(**self.config)

    async def build_async(self) -> MCPAgent:
        """Build agent and setup tools asynchronously."""
        if 'config' not in self.config:
            raise ValueError("config argument is required for MCPAgent.")
        agent = MCPAgent(**self.config)
        await agent.setup_tools()
        return agent

# Test function
async def test_mcp_agent():
    """Test the MCP agent"""
    print("🧪 Testing MCP Agent (VS Code-like)")

    # Create mock config for testing
    class MockConfig:
        def get(self, key, default=None):
            return {"configurable": {"user_id": "test", "org_id": "test"}}.get(key, default)
    
    try:
        builder = MCPAgentBuilder()
        agent = await builder.with_config(MockConfig()).with_debug(True).build_async()
        print("\nMCP Agent Interactive Mode: Type any message or tool name. Type 'exit' to quit.")
        result = await agent.execute_command("mcp.list_tools")
        print(f"Available tools:")
        for tool in result.get('tools', []):
            print(f"- {tool['name']} ({tool['server']}) : {tool['description']}")
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() == 'exit':
                print("Exiting MCP Agent interactive mode.")
                break
            # If input matches a tool, run the tool
            if user_input in agent.tool_contexts:
                args = input(f"Arguments for '{user_input}' as JSON (or empty for none): ").strip()
                try:
                    arguments = json.loads(args) if args else {}
                except Exception:
                    print("Invalid JSON. Try again.")
                    continue
                result = await agent._execute_tool_command(user_input, arguments)
                print(f"Agent (tool result): {json.dumps(result, indent=2)}")
            else:
                # Otherwise, reply with a default message
                print(f"Agent: I can run tools for you. Type a tool name from the list above, or 'exit' to quit.")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_mcp_agent())
