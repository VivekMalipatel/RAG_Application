from typing import Any, Dict, List, Optional, Union
import json
import os
import asyncio
import httpx
import traceback
import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass

import dotenv
from mcp.server.fastmcp import Context, FastMCP

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
dotenv.load_dotenv()

# Defaults
DEFAULT_LIMIT = 10
DEFAULT_URL = os.getenv("QDRANT_MCP_URL", "http://localhost:8000")
# Ensure URL has proper http:// or https:// prefix
if DEFAULT_URL and not DEFAULT_URL.startswith(('http://', 'https://')):
    DEFAULT_URL = f"http://{DEFAULT_URL}"
QDRANT_API_ENDPOINT = f"{DEFAULT_URL}/api/v1/mcp"

@dataclass
class AppContext:
    """Application context holding initialized resources."""
    client: httpx.AsyncClient
    base_url: str

@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage API client lifecycle with type-safe context."""
    # Initialize HTTP client on startup
    logger.info("Initializing Qdrant MCP client...")
    client = httpx.AsyncClient(timeout=300.0)  # Increased from 30.0 to 300.0 seconds (5 minutes)
    base_url = QDRANT_API_ENDPOINT
    
    try:
        # Test connection to ensure endpoint is available
        logger.info(f"Testing connection to {base_url}")
        try:
            async with client.stream("GET", f"{base_url}/collections") as response:
                if response.status_code != 200:
                    logger.error(f"Failed to connect to Qdrant MCP endpoint: {response.status_code}")
                    raise ConnectionError(f"Qdrant MCP endpoint returned status {response.status_code}")
            logger.info("Qdrant MCP client initialized successfully")
        except httpx.ConnectError as e:
            logger.error(f"Connection error to {base_url}: {str(e)}")
            raise ConnectionError(f"Could not connect to Qdrant MCP endpoint: {str(e)}")
        except httpx.RequestError as e:
            logger.error(f"Request error to {base_url}: {str(e)}")
            raise ConnectionError(f"Error making request to Qdrant MCP endpoint: {str(e)}")
        
        yield AppContext(client=client, base_url=base_url)
    finally:
        # Cleanup on shutdown
        logger.info("Closing Qdrant MCP client...")
        await client.aclose()
        logger.info("Qdrant MCP client closed")

# Initialize FastMCP server with lifespan
mcp = FastMCP("Qdrant MCP", 
              description="MCP server for Qdrant vector database operations",
              lifespan=app_lifespan)

# ===== Resources =====

@mcp.resource("qdrant://collections")
async def get_collections() -> str:
    """Get a list of all collections in the Qdrant database."""
    logger.info("Resource accessed: get_collections()")
    context = mcp.get_request_context()
    client = context.lifespan_context.client
    base_url = context.lifespan_context.base_url
    
    try:
        response = await client.get(f"{base_url}/collections")
        response.raise_for_status()
        collections = response.json()
        logger.info(f"Retrieved {collections.get('total', 0)} collections from Qdrant")
        return json.dumps(collections, indent=2)
    except Exception as e:
        logger.error(f"Error retrieving collections: {str(e)}")
        return json.dumps({"error": str(e), "details": traceback.format_exc()})

@mcp.resource("qdrant://collections/{user_id}/count")
async def get_collection_count(user_id: str) -> str:
    """Get the number of chunks in a specific user's collection.
    
    Args:
        user_id: User ID / Collection name
    """
    logger.info(f"Resource accessed: get_collection_count(user_id='{user_id}')")
    context = mcp.get_request_context()
    client = context.lifespan_context.client
    base_url = context.lifespan_context.base_url
    
    try:
        response = await client.get(f"{base_url}/collections/{user_id}/count")
        response.raise_for_status()
        count_data = response.json()
        logger.info(f"Retrieved chunk count for user '{user_id}': {count_data.get('count', 0)}")
        return json.dumps(count_data, indent=2)
    except Exception as e:
        logger.error(f"Error retrieving collection count for user '{user_id}': {str(e)}")
        return json.dumps({"error": str(e), "details": traceback.format_exc()})

# ===== Tools =====

@mcp.tool(description="Search for documents using hybrid search with dense and sparse embeddings. This uses qdran't hierarchial search using all the parameters")
async def hybrid_search(
    ctx: Context, 
    query: str, 
    user_id: str,
    top_k: int = 5,
    matryoshka_64_limit: int = 100,
    matryoshka_128_limit: int  = 80,
    matryoshka_256_limit: int = 60,
    dense_limit: int = 40,
    sparse_limit: int = 50,
    final_limit: int= 30,
    hnsw_ef: int= 128,
    # must: Optional[Dict] = None,
    # must_not: Optional[Dict] = None,
    # should: Optional[Dict] = None
) -> str:
    """Perform hybrid search across dense and sparse embeddings.
    
    Args:
        query: The search query text
        user_id: User ID / Collection name to search in
        top_k: Number of results to return
        matryoshka_64_limit: Limit for 64-dim matryoshka embeddings
        matryoshka_128_limit: Limit for 128-dim matryoshka embeddings
        matryoshka_256_limit: Limit for 256-dim matryoshka embeddings
        dense_limit: Limit for dense embeddings
        sparse_limit: Limit for sparse embeddings
        final_limit: Final number of results to consider
        hnsw_ef: HNSW search parameter
        must: Fields that must match in the filter
        must_not: Fields that must not match in the filter
        should: Fields that should match in the filter
    """
    logger.info(f"Tool called: hybrid_search(query='{query}', user_id='{user_id}', top_k={top_k})")
    client = ctx.request_context.lifespan_context.client
    base_url = ctx.request_context.lifespan_context.base_url
    
    try:
        # Construct search parameters
        search_params = {
            "matryoshka_64_limit": matryoshka_64_limit,
            "matryoshka_128_limit": matryoshka_128_limit,
            "matryoshka_256_limit": matryoshka_256_limit,
            "dense_limit": dense_limit,
            "sparse_limit": sparse_limit,
            "final_limit": final_limit,
            "hnsw_ef": hnsw_ef
        }
        
        # Construct filters
        filters = {}
        # if must:
        #     filters["must"] = must
        # if must_not:
        #     filters["must_not"] = must_not
        # if should:
        #     filters["should"] = should
            
        # Prepare request payload
        payload = {
            "query": query,
            "user_id": user_id,
            "top_k": top_k,
            "search_params": search_params
        }
        
        # Only include filters if at least one filter is specified
        # if filters:
        #     payload["filters"] = filters
            
        # Make request to the search endpoint
        response = await client.post(f"{base_url}/search", json=payload)
        response.raise_for_status()
        search_results = response.json()
        
        logger.info(f"Search completed: found {search_results.get('total_found', 0)} results")
        return json.dumps(search_results, indent=2)
    except Exception as e:
        logger.error(f"Error performing hybrid search: {str(e)}")
        return json.dumps({"error": str(e), "details": traceback.format_exc()})

@mcp.tool(description="Get the count of chunks in a collection with optional filtering.")
async def get_filtered_collection_count(
    ctx: Context,
    user_id: str,
    # must: Optional[Dict] = None,
    # must_not: Optional[Dict] = None,
    # should: Optional[Dict] = None
) -> str:
    """Get the number of chunks in a collection with optional filtering.
    
    Args:
        user_id: User ID / Collection name
        must: Fields that must match in the filter
        must_not: Fields that must not match in the filter
        should: Fields that should match in the filter
    """
    logger.info(f"Tool called: get_filtered_collection_count(user_id='{user_id}')")
    client = ctx.request_context.lifespan_context.client
    base_url = ctx.request_context.lifespan_context.base_url
    
    try:
        # Construct query parameters
        params = {}
        # if must:
        #     params["must"] = json.dumps(must)
        # if must_not:
        #     params["must_not"] = json.dumps(must_not)
        # if should:
        #     params["should"] = json.dumps(should)
            
        # Make request to the count endpoint with filters
        response = await client.get(
            f"{base_url}/collections/{user_id}/count", 
            # params=params
        )
        response.raise_for_status()
        count_data = response.json()
        
        logger.info(f"Retrieved filtered chunk count for user '{user_id}': {count_data.get('count', 0)}")
        return json.dumps(count_data, indent=2)
    except Exception as e:
        logger.error(f"Error retrieving filtered collection count: {str(e)}")
        return json.dumps({"error": str(e), "details": traceback.format_exc()})

@mcp.tool(description="List all available collections in the Qdrant database.")
async def list_collections(ctx: Context) -> str:
    """List all collections in the Qdrant database.
    """
    logger.info("Tool called: list_collections()")
    client = ctx.request_context.lifespan_context.client
    base_url = ctx.request_context.lifespan_context.base_url
    
    try:
        # Make request to the collections endpoint
        response = await client.get(f"{base_url}/collections")
        response.raise_for_status()
        collections = response.json()
        
        logger.info(f"Listed {collections.get('total', 0)} collections")
        return json.dumps(collections, indent=2)
    except Exception as e:
        logger.error(f"Error listing collections: {str(e)}")
        return json.dumps({"error": str(e), "details": traceback.format_exc()})

if __name__ == "__main__":
    # Run the server
    logger.info("Starting Qdrant MCP server")
    mcp.run(transport='stdio')
