"""Tool execution utilities for agents"""

import json
import re
from typing import Dict, Any, List, Optional
from app.tools import tool_registry
import logging

logger = logging.getLogger(__name__)

class ToolExecutorMixin:
    """Mixin class to add tool execution capabilities to agents"""
    
    def detect_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """Detect tool calls in LLM response text"""
        tool_calls = []
        
        # Look for function call patterns like: tool_name(param1=value1, param2=value2)
        # or JSON-like patterns: {"tool": "tool_name", "parameters": {...}}
        
        # Pattern 1: function call syntax
        func_pattern = r'(\w+)\((.*?)\)'
        func_matches = re.finditer(func_pattern, text)
        
        for match in func_matches:
            tool_name = match.group(1)
            params_str = match.group(2)
            
            # Check if this is a valid tool
            if tool_registry.get_tool(tool_name):
                try:
                    # Parse parameters (simplified parsing)
                    params = self._parse_function_params(params_str)
                    tool_calls.append({
                        "tool": tool_name,
                        "parameters": params,
                        "raw_match": match.group(0)
                    })
                except Exception as e:
                    logger.warning(f"Failed to parse tool call parameters: {e}")
        
        # Pattern 2: JSON syntax
        json_pattern = r'\{[^{}]*"tool"\s*:\s*"([^"]+)"[^{}]*\}'
        json_matches = re.finditer(json_pattern, text)
        
        for match in json_matches:
            try:
                json_obj = json.loads(match.group(0))
                tool_name = json_obj.get("tool")
                
                if tool_name and tool_registry.get_tool(tool_name):
                    tool_calls.append({
                        "tool": tool_name,
                        "parameters": json_obj.get("parameters", {}),
                        "raw_match": match.group(0)
                    })
            except json.JSONDecodeError:
                continue
        
        return tool_calls
    
    def _parse_function_params(self, params_str: str) -> Dict[str, Any]:
        """Parse function parameters from string"""
        params = {}
        
        # Simple parameter parsing (key=value, key="value")
        param_pattern = r'(\w+)\s*=\s*([^,]+)'
        matches = re.finditer(param_pattern, params_str)
        
        for match in matches:
            key = match.group(1).strip()
            value = match.group(2).strip()
            
            # Remove quotes if present
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            elif value.startswith("'") and value.endswith("'"):
                value = value[1:-1]
            else:
                # Try to convert to int/float/bool
                try:
                    if value.lower() in ['true', 'false']:
                        value = value.lower() == 'true'
                    elif '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    pass  # Keep as string
            
            params[key] = value
        
        return params
    
    async def execute_detected_tools(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute all detected tool calls"""
        results = []
        
        for tool_call in tool_calls:
            tool_name = tool_call["tool"]
            parameters = tool_call["parameters"]
            
            logger.info(f"Executing tool: {tool_name} with params: {parameters}")
            
            result = await tool_registry.execute_tool(tool_name, parameters)
            result["tool_call"] = tool_call
            results.append(result)
        
        return results
    
    def format_tool_results(self, tool_results: List[Dict[str, Any]]) -> str:
        """Format tool execution results for inclusion in response"""
        if not tool_results:
            return ""
        
        formatted = "\n\n--- Tool Execution Results ---\n"
        
        for i, result in enumerate(tool_results, 1):
            tool_name = result.get("tool_used", "unknown")
            success = result.get("success", False)
            
            formatted += f"\n{i}. {tool_name}: "
            
            if success:
                formatted += "✅ Success\n"
                if result.get("result"):
                    formatted += f"Result: {json.dumps(result['result'], indent=2)}\n"
            else:
                formatted += "❌ Failed\n"
                if result.get("error"):
                    formatted += f"Error: {result['error']}\n"
        
        formatted += "\n--- End Tool Results ---\n"
        return formatted
    
    def get_available_tools_prompt(self) -> str:
        """Get a simple prompt section describing available tools"""
        tools = tool_registry.get_tools_for_llm()
        
        if not tools:
            return ""
        
        prompt = "\n\nAvailable Tools:\n"
        for tool in tools:
            prompt += f"- {tool['name']}: {tool['description']}\n"
        
        prompt += "\nTo use a tool, mention its name in your response.\n\n"
        
        return prompt
