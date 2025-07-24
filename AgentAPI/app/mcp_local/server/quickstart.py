from mcp.server.fastmcp import FastMCP
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio

# Configure logging properly
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("fastmcp.demo")

# Create FastMCP server with proper configuration
mcp = FastMCP("Demo", version="1.0.0")

# Register tools with proper error handling
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers together"""
    logger.debug(f"add({a}, {b}) called")
    return a + b

@mcp.tool()
def subtract(a: int, b: int) -> int:
    """Subtract second number from first number"""
    logger.debug(f"subtract({a}, {b}) called")
    return a - b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    logger.debug(f"multiply({a}, {b}) called")
    return a * b

@mcp.tool()
def divide(a: float, b: float) -> float:
    """Divide first number by second number"""
    logger.debug(f"divide({a}, {b}) called")
    if b == 0:
        raise ValueError("Division by zero is not allowed")
    return a / b

@mcp.tool()
def echo(message: str) -> str:
    """Echo back the input message"""
    logger.debug(f"echo('{message}') called")
    if not message:
        raise ValueError("Message cannot be empty")
    return message

@mcp.tool()
def uppercase(text: str) -> str:
    """Convert text to uppercase"""
    logger.debug(f"uppercase('{text}') called")
    if not isinstance(text, str):
        raise ValueError("Input must be a string")
    return text.upper()

@mcp.tool()
def lowercase(text: str) -> str:
    """Convert text to lowercase"""
    logger.debug(f"lowercase('{text}') called")
    if not isinstance(text, str):
        raise ValueError("Input must be a string")
    return text.lower()

@mcp.tool()
def length(text: str) -> int:
    """Get the length of text"""
    logger.debug(f"length('{text}') called")
    if not isinstance(text, str):
        raise ValueError("Input must be a string")
    return len(text)

# Add resources with proper validation
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting for a person"""
    logger.debug(f"get_greeting('{name}') called")
    if not name or not name.strip():
        raise ValueError("Name cannot be empty")
    return f"Hello, {name.strip()}! Welcome to our demo MCP server."

# Add prompts with proper templates
@mcp.prompt()
def greet_user(name: str, style: str = "friendly") -> str:
    """Generate a greeting prompt with specified style"""
    logger.debug(f"greet_user('{name}', style='{style}') called")
    
    if not name or not name.strip():
        raise ValueError("Name is required")
    
    styles = {
        "friendly": "Please write a warm, friendly greeting",
        "formal": "Please write a formal, professional greeting", 
        "casual": "Please write a casual, relaxed greeting",
        "enthusiastic": "Please write an enthusiastic, energetic greeting"
    }
    
    if style not in styles:
        raise ValueError(f"Style must be one of: {', '.join(styles.keys())}")
    
    return f"{styles[style]} for someone named {name.strip()}."


from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    logger.info("=== Demo MCP Server Starting ===")
    logger.info(f"Tools registered: {len(mcp._tools) if hasattr(mcp, '_tools') else 0}")
    logger.info(f"Resources registered: {len(mcp._resources) if hasattr(mcp, '_resources') else 0}")
    logger.info(f"Prompts registered: {len(mcp._prompts) if hasattr(mcp, '_prompts') else 0}")
    logger.info("=== Server Ready ===")
    yield
    logger.info("Demo MCP Server shutting down...")

app = FastAPI(
    title="Demo MCP Server",
    description="A demonstration MCP server with math and text operations",
    version="1.0.0",
    docs_url="/docs",
    lifespan=lifespan
)

# Add CORS middleware for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


# Health check endpoint (industry standard)
@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "service": "Demo MCP Server",
        "version": "1.0.0",
        "mcp_endpoint": "/mcp"
    }

# Readiness check (Kubernetes standard)
@app.get("/ready", tags=["Health"])
def readiness_check():
    """Readiness check for deployment orchestration"""
    return {
        "status": "ready",
        "tools_count": len(mcp._tools) if hasattr(mcp, '_tools') else 0
    }

# Debug endpoints (development only)
@app.get("/debug/info", tags=["Debug"])
def debug_info():
    """Debug information about the MCP server"""
    info = {
        "server_name": "Demo",
        "mcp_version": getattr(mcp, 'version', 'unknown'),
        "tools_registered": len(mcp._tools) if hasattr(mcp, '_tools') else 0,
        "resources_registered": len(mcp._resources) if hasattr(mcp, '_resources') else 0,
        "prompts_registered": len(mcp._prompts) if hasattr(mcp, '_prompts') else 0
    }
    return info

@app.get("/debug/tools", tags=["Debug"])
async def debug_tools():
    """Show server metadata (development only)"""
    try:
        return {
            "server_name": getattr(mcp, 'name', 'Unknown'),
            "server_type": str(type(mcp).__name__),
            "available_attributes": [attr for attr in dir(mcp) if not attr.startswith('__')],
            "tools": "Use MCP protocol to list tools properly",
            "note": "This endpoint shows server metadata. Use MCP client to list actual tools.",
            "mcp_endpoint": "/mcp for proper tool access"
        }
    except Exception as e:
        logger.error(f"Error in debug_tools: {e}")
        return {"error": str(e), "suggestion": "Check server logs for details"}

@app.get("/debug/resources", tags=["Debug"])
def debug_resources():
    """List all registered resources (development only)"""
    try:
        resources_info = []
        if hasattr(mcp, '_resources'):
            for pattern, resource_func in mcp._resources.items():
                resources_info.append({
                    "pattern": pattern,
                    "description": getattr(resource_func, '__doc__', 'No description')
                })
        
        return {
            "resources": resources_info,
            "count": len(resources_info)
        }
    except Exception as e:
        logger.error(f"Error in debug_resources: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers (industry standard)
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"error": "Bad Request", "message": str(exc)}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "message": "An unexpected error occurred"}
    )


app.mount("/mcp", mcp.streamable_http_app(), name="mcp")

if __name__ == "__main__":
    logger.info("Starting Demo FastMCP Server...")
    logger.info("Endpoints:")
    logger.info("  - MCP Protocol: http://0.0.0.0:8000/mcp")
    logger.info("  - Health Check: http://0.0.0.0:8000/health")
    logger.info("  - API Docs: http://0.0.0.0:8000/docs")
    logger.info("  - Debug Tools: http://0.0.0.0:8000/debug/tools")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )
