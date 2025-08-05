from abc import ABC, abstractmethod
from mcp.server.fastmcp import FastMCP
import logging
import os
from datetime import datetime
from core.base_server.src.middleware import MCPAuthMiddleware

class BaseMCPServer(ABC):
    def __init__(self, server_name: str, port: int = 8080, enable_auth: bool = True):
        self.server_name = server_name
        self.port = port
        self.enable_auth = enable_auth
        
        if self.enable_auth:
            self.auth_middleware = MCPAuthMiddleware(
                secret_key=os.getenv("MCP_SECRET_KEY"),
                enable_api_keys=os.getenv("MCP_ENABLE_API_KEYS", "true").lower() == "true",
                enable_jwt=os.getenv("MCP_ENABLE_JWT", "true").lower() == "true"
            )

        self.app = FastMCP(server_name, port=self.port)
        self.setup_logging()
        self.setup_common_features()
        self.register_tools()
    
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format=f'%(asctime)s - {self.server_name} - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(self.server_name)
    
    def require_auth(self, permission: str = "read"):
        if not self.enable_auth:
            def no_auth_decorator(func):
                return func
            return no_auth_decorator
        
        return self.auth_middleware.create_auth_decorator(permission)
    
    def setup_common_features(self):        
        @self.app.tool()
        async def health_check() -> dict:
            """Check server health status - No authentication required"""
            return {
                "status": "healthy",
                "server": self.server_name,
                "timestamp": datetime.now().isoformat(),
                "uptime": "running",
                "port": self.port,
                "auth_enabled": self.enable_auth
            }
        
        @self.app.tool()
        @self.require_auth("read")
        async def server_info() -> dict:
            """Get detailed server information - Requires read permission"""
            return {
                "name": self.server_name,
                "version": "1.0.0",
                "capabilities": self.get_capabilities(),
                "port": self.port,
                "transport": "streamable-http",
                "protocol": "streamable-http",
                "status": "active",
                "auth_enabled": self.enable_auth
            }
        
        @self.app.tool()
        @self.require_auth("read")
        async def list_tools() -> dict:
            """List all available tools on this server - Requires read permission"""
            return {
                "server": self.server_name,
                "tools": "Use MCP client to discover available tools",
                "capabilities": self.get_capabilities()
            }
        
        # Authentication management tools
        if self.enable_auth:
            @self.app.tool()
            @self.require_auth("admin")
            async def generate_token(user_id: str, permissions: str = "read") -> dict:
                """Generate a new JWT token - Requires admin permission
                
                Args:
                    user_id: Unique identifier for the user
                    permissions: Comma-separated permissions (read,write,admin)
                """
                try:
                    perm_list = [p.strip() for p in permissions.split(",")]
                    token = self.auth_middleware.generate_jwt_token(user_id, perm_list)
                    
                    return {
                        "success": True,
                        "token": token,
                        "user_id": user_id,
                        "permissions": perm_list,
                        "expires_in_hours": self.auth_middleware.token_expiry_hours
                    }
                except Exception as e:
                    return {"success": False, "error": str(e)}
            
            @self.app.tool()
            @self.require_auth("admin") 
            async def list_api_keys() -> dict:
                """List all configured API keys - Requires admin permission"""
                return {
                    "api_keys": {
                        key[:8] + "..." if len(key) > 8 else key: info 
                        for key, info in self.auth_middleware.api_keys.items()
                    }
                }
    
    @abstractmethod
    def register_tools(self):
        """Override this to register server-specific tools"""
        pass
    
    def get_capabilities(self) -> list:
        """Override to specify server capabilities"""
        base_caps = ["basic", "health-check", "server-info"]
        if self.enable_auth:
            base_caps.extend(["authentication", "authorization", "rate-limiting"])
        return base_caps
    
    def run(self, port: int = None):
        """Start the MCP server, optionally on a specified port"""
        run_port = port if port is not None else self.port
        self.logger.info(f"Starting {self.server_name} MCP Server on port {run_port}")
        self.logger.info(f"Authentication: {'Enabled' if self.enable_auth else 'Disabled'}")
        self.logger.info(f"Capabilities: {', '.join(self.get_capabilities())}")
        
        if self.enable_auth:
            self.logger.info("🔐 Authentication Methods:")
            if self.auth_middleware.enable_jwt:
                self.logger.info("  ✓ JWT Bearer tokens")
            if self.auth_middleware.enable_api_keys:
                self.logger.info(f"  ✓ API Keys ({len(self.auth_middleware.api_keys)} configured)")
        
        try:
            self.app.run(transport="streamable-http")
        except Exception as e:
            self.logger.error(f"Failed to start server: {e}")
            raise