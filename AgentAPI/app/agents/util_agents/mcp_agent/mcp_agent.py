
import sys
import json
import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from langchain_core.runnables import RunnableConfig

sys.path.append(str(Path(__file__).parent.parent.parent))
from agents.base_agents.base_agent import BaseAgent
from tools.core_tools.mcp.mcp_tool import multi_server_mcp_wrapper, MCPToolRequest, get_default_server_configs,get_available_tools_from_servers, build_server_connections
from tools.core_tools.mcp.mcp_utils import (
    load_prompt_or_description, load_mcp_config, get_enabled_servers,
    inject_security_context, get_tool_names, fetch_tools
)
logger = logging.getLogger(__name__)
from tools.core_tools.mcp.mcp_tool import test_enabled_servers, test_docker_gateway_connection

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
        self.mcp_config_path = mcp_config_path or "/Users/gauravs/Documents/RAG_Application/AgentAPI/app/mcp_local/mcp.json"
        self.mcp_config = load_mcp_config(self.mcp_config_path)
        self.enabled_servers = get_enabled_servers(self.mcp_config)
        self._available_tool_objects = None


    async def setup_tools(self):
        try:
            self.bind_tools([multi_server_mcp_wrapper])
            await self.compile()
            logger.info("✅ MCPAgent: MCP wrapper tool bound and compiled successfully")
        except Exception as e:
            logger.error(f"⚠️ MCPAgent: Error binding MCP wrapper tool: {e}")


    async def fetch_and_cache_tool_objects(self, force_refresh: bool = False):
        if self._available_tool_objects is not None and not force_refresh:
            return self._available_tool_objects
        from langchain_mcp_adapters.client import MultiServerMCPClient
        configs = get_default_server_configs()
        enabled_configs = [cfg for cfg in configs if cfg.enabled and cfg.transport in ["stdio", "streamable_http"]]
        connections = build_server_connections(enabled_configs)
        client = MultiServerMCPClient(connections)
        self._available_tool_objects = await fetch_tools(client, enabled_configs, connections)
        return self._available_tool_objects


    def get_cached_tool_names(self):
        if self._available_tool_objects is None:
            return []
        return get_tool_names(self._available_tool_objects)


    async def get_available_tools_info(self) -> Dict[str, Any]:
        tools_info = {
            "enabled_servers": self.enabled_servers,
            "server_count": len(self.enabled_servers),
            "tools": []
        }
        try:
            tools_list = await get_available_tools_from_servers()
            tools_info["tools"] = tools_list
            logger.info(f"✅ Successfully fetched {len(tools_list)} tools via mcp_tool function")
        except Exception as e:
            logger.error(f"⚠️  Could not fetch tools via mcp_tool function: {e}")
        return tools_info
    
    def display_startup_info(self):
        """Display MCP agent startup information following industry standards."""
        print("\n" + "="*60)
        print("🔧 MCP Agent - Industry Standard Implementation")
        print("="*60)
        if not self.enabled_servers:
            print("⚠️  No enabled MCP servers found")
            return
        print(f"📊 Enabled MCP Servers: {len(self.enabled_servers)}")
        for server in self.enabled_servers:
            server_config = self.mcp_config.get('mcp_config', {}).get('servers', {}).get(server, {})
            transport = server_config.get('transport', 'unknown')
            print(f"  • {server} ({transport})")
        print(f"\n🔧 MCP wrapper bound as core tool")
        print(f"🚀 Agent ready for MCP operations")

    async def debug_connection(self):
        """Debug the agent's connection to MCP servers"""
        print("\n🔍 DEBUGGING MCP CONNECTION")
        print("="*50)
        print("1. Configuration:")
        print(f"   Config path: {self.mcp_config_path}")
        print(f"   Enabled servers: {self.enabled_servers}")
        configs = get_default_server_configs()
        print(f"   Loaded configs: {len(configs)}")
        for cfg in configs:
            print(f"     - {cfg.name}: {cfg.transport} @ {cfg.url}")
        enabled_configs = [cfg for cfg in configs if cfg.enabled and cfg.transport in ["stdio", "streamable_http"]]
        connections = build_server_connections(enabled_configs)
        print(f"   Built connections: {connections}")
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
            client = MultiServerMCPClient(connections)
            print("   Client created successfully")
            tools = await asyncio.wait_for(client.get_tools(), timeout=30)
            print(f"   ✅ Found {len(tools)} tools via agent method:")
            for tool in tools[:3]:
                print(f"     - {getattr(tool, 'name', str(tool))}")
        except asyncio.TimeoutError:
            print("   ❌ Timeout when fetching tools via agent method")
        except Exception as e:
            print(f"   ❌ Error fetching tools via agent method: {e}")
        print("="*50)
    

    async def execute_mcp_request(self, tool_name: str, arguments: Dict[str, Any], config: RunnableConfig) -> str:
        """
        Execute a single MCP tool request using cached tool objects if available. Always refresh cache before invocation.
        """
        await self.fetch_and_cache_tool_objects(force_refresh=True)
        target_tool = None
        for tool_obj in self._available_tool_objects:
            if hasattr(tool_obj, 'name') and tool_obj.name == tool_name:
                target_tool = tool_obj
                break
        if not target_tool:
            available = self.get_cached_tool_names()
            logger.error(f"❌ MCPAgent: Tool '{tool_name}' not found. Available tools: {available}")
            return f"Tool '{tool_name}' not found. Available tools: {available}"
        enhanced_arguments = inject_security_context(arguments, config)
        try:
            logger.info(f"🔗 MCPAgent: Invoking tool '{tool_name}' with arguments: {enhanced_arguments}")
            result = await target_tool.ainvoke(enhanced_arguments)
            logger.info(f"✅ MCPAgent: Tool '{tool_name}' invocation result: {result}")
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error(f"❌ MCPAgent: Error executing MCP request for tool '{tool_name}': {str(e)}")
            return f"Error executing MCP request: {str(e)}"

    async def execute_batch_mcp_requests(self, requests: List[MCPToolRequest], config: RunnableConfig) -> List[str]:
        """
        Execute multiple MCP tool requests in batch.
        Industry standard batch processing method.
        """
        try:
            results = await multi_server_mcp_wrapper.ainvoke({"requests": requests}, config=config)
            return results
        except Exception as e:
            logger.error(f"Error executing batch MCP requests: {str(e)}")
            return [f"Error executing batch MCP requests: {str(e)}"]



class MCPAgentBuilder:
    def __init__(self):
        self.config = {}

    def with_mcp_config(self, config_path: str):
        self.config['mcp_config_path'] = config_path
        return self

    def with_prompt(self, prompt: str):
        self.config['prompt'] = prompt
        return self

    def with_model_kwargs(self, **kwargs):
        self.config['model_kwargs'] = kwargs
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
        agent = MCPAgent(**self.config)
        return agent
    
    async def build_async(self) -> MCPAgent:
        """Build agent and setup tools asynchronously."""
        if 'config' not in self.config:
            raise ValueError("config argument is required for MCPAgent.")
        agent = MCPAgent(**self.config)
        await agent.setup_tools()
        return agent


async def test_mcp_agent():
    print("🧪 Testing MCP Agent Implementation")
    print("\n--- MCP Tool: Enabled Servers Test ---")
    await test_enabled_servers()
    print("\n--- MCP Tool: Docker Gateway Connection Test (agent config) ---")
    await test_docker_gateway_connection(use_agent_config=True)


def create_mcp_agent(config_path: Optional[str] = None, **kwargs) -> MCPAgent:
    return MCPAgent(mcp_config_path=config_path, **kwargs)


async def create_mcp_agent_async(config_path: Optional[str] = None, **kwargs) -> MCPAgent:
    agent = MCPAgent(mcp_config_path=config_path, **kwargs)
    await agent.setup_tools()
    return agent


if __name__ == "__main__":
    asyncio.run(test_mcp_agent())
