from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import logging
import json
import importlib
import os

logger = logging.getLogger(__name__)

class ToolInput(BaseModel):
    """Base input model for tools"""
    pass

class ToolOutput(BaseModel):
    """Base output model for tools"""
    success: bool = True
    result: Any = None
    error: Optional[str] = None

class BaseTool(ABC):
    """Base class for all tools"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.logger = logging.getLogger(f"tools.{name}")
    
    @abstractmethod
    async def execute(self, input_data: ToolInput) -> ToolOutput:
        """Execute the tool with given input"""
        pass
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the tool schema for LLM integration"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.get_parameters_schema()
        }
    
    @abstractmethod
    def get_parameters_schema(self) -> Dict[str, Any]:
        """Get the parameters schema for this tool"""
        pass

class ToolRegistry:
    """Registry for managing all available tools"""
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._config_path = os.path.join(os.path.dirname(__file__), "tools_config.json")
    
    def register(self, tool: BaseTool):
        """Register a tool"""
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    def load_tools_from_config(self):
        """Load and register tools from JSON configuration"""
        try:
            if not os.path.exists(self._config_path):
                logger.warning(f"Tools config not found at {self._config_path}")
                return
            
            with open(self._config_path, 'r') as f:
                config = json.load(f)
            
            tools_config = config.get("tools", {})
            
            for tool_name, tool_info in tools_config.items():
                try:
                    # Import the module and class
                    module_name = tool_info["module"]
                    class_name = tool_info["class"]
                    
                    module = importlib.import_module(module_name)
                    tool_class = getattr(module, class_name)
                    
                    # Instantiate the tool
                    tool_instance = tool_class()
                    
                    # Register the tool
                    self.register(tool_instance)
                    
                except Exception as e:
                    logger.error(f"Failed to load tool {tool_name}: {e}")
            
            logger.info(f"Loaded {len(self._tools)} tools from configuration")
            
        except Exception as e:
            logger.error(f"Failed to load tools configuration: {e}")
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name"""
        return self._tools.get(name)
    
    def get_all_tools(self) -> Dict[str, BaseTool]:
        """Get all registered tools"""
        return self._tools.copy()
    
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get schemas for all tools for LLM integration"""
        return [tool.get_schema() for tool in self._tools.values()]
    
    def list_tools(self) -> List[str]:
        """List all tool names"""
        return list(self._tools.keys())
    
    async def execute_tool(self, tool_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool by name with input data"""
        try:
            tool = self.get_tool(tool_name)
            if not tool:
                return {
                    "success": False,
                    "error": f"Tool '{tool_name}' not found. Available tools: {', '.join(self.list_tools())}"
                }
            
            # Convert input_data to the tool's expected input class
            # We need to dynamically determine the input class for each tool
            tool_input = self._create_tool_input(tool, input_data)
            
            # Execute the tool
            result = await tool.execute(tool_input)
            
            return {
                "success": result.success,
                "result": result.result,
                "error": result.error,
                "tool_used": tool_name
            }
            
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return {
                "success": False,
                "error": f"Tool execution error: {str(e)}",
                "tool_used": tool_name
            }
    
    def _create_tool_input(self, tool: BaseTool, input_data: Dict[str, Any]):
        """Create the appropriate input object for a tool"""
        # This is a dynamic approach to handle different tool input types
        # We'll look for the tool's input class by convention
        
        # Import the tool's module to get the input class
        tool_module = tool.__class__.__module__
        
        if "web_tools" in tool_module:
            if tool.name == "web_search":
                from app.tools.web_tools import WebSearchInput
                return WebSearchInput(**input_data)
            elif tool.name == "web_scraping":
                from app.tools.web_tools import WebScrapingInput
                return WebScrapingInput(**input_data)
        
        # For other tools, try to create a generic ToolInput
        # This assumes the tool can handle dict input directly
        return input_data
    
    def get_tools_for_llm(self) -> List[Dict[str, Any]]:
        """Get simple tool list for LLM function calling"""
        tools = []
        for tool_name, tool in self._tools.items():
            tool_def = {
                "name": tool_name,
                "description": tool.description
            }
            tools.append(tool_def)
        return tools

# Global tool registry
tool_registry = ToolRegistry()

# Auto-load tools from configuration
tool_registry.load_tools_from_config()
