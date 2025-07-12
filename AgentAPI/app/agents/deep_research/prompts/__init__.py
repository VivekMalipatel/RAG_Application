from .manager import PromptsManager
from typing import Optional

# Global instance for easy access
prompts_manager = PromptsManager()

# Convenience functions for common use cases
def get_research_prompt(prompt_name: str) -> Optional[str]:
    """Get a research agent prompt"""
    return prompts_manager.get_prompt('research_agents', prompt_name)

def get_prompt(category: str, prompt_name: str) -> Optional[str]:
    """General function to get any prompt"""
    return prompts_manager.get_prompt(category, prompt_name)

def get_prompt_with_template(category: str, prompt_name: str, **kwargs) -> Optional[str]:
    """Get a prompt with template variable substitution"""
    return prompts_manager.get_prompt_template(category, prompt_name, **kwargs)

def list_all_prompts():
    """List all available prompts organized by category"""
    result = {}
    for category in prompts_manager.list_categories():
        result[category] = prompts_manager.list_prompts(category)
    return result

def search_prompts(search_term: str):
    """Search for prompts containing a specific term"""
    return prompts_manager.search_prompts(search_term)

def reload_prompts():
    """Reload all prompts from files"""
    prompts_manager.reload_prompts()

