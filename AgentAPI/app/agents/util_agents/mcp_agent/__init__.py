"""
MCP Agent Module

Industry-standard MCP (Model Context Protocol) agent implementation that extends BaseAgent.
Provides seamless integration with multiple MCP servers through bound tools.
"""

from .mcp_agent import MCPAgent, MCPAgentBuilder, create_mcp_agent

__all__ = [
    'MCPAgent',
    'MCPAgentBuilder', 
    'create_mcp_agent'
]
