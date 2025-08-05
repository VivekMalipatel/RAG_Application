"""
Restaurant prompts using PromptsManager
"""

from .manager import PromptsManager
from pathlib import Path

# Initialize prompts manager with the current directory
prompts_manager = PromptsManager(prompts_dir=Path(__file__).parent)

def get_restaurant_prompt(prompt_type: str) -> str:
    """Get restaurant service prompts for different agent nodes using PromptsManager"""
    
    # Map prompt types to category and prompt name
    prompt_mapping = {
        'customer_greeting': ('restaurant_service', 'customer_greeting'),
        'service_intent_analysis': ('restaurant_service', 'service_intent_analysis'),
        'service_gap_analysis': ('restaurant_service', 'service_gap_analysis'),
        'complete_service': ('service_completion', 'complete_service'),
        'gaps_to_additional_services': ('service_completion', 'gaps_to_additional_services'),
        'additional_service_processor': ('service_execution', 'additional_service_processor'),
    }
    
    if prompt_type not in prompt_mapping:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
    
    category, prompt_name = prompt_mapping[prompt_type]
    prompt = prompts_manager.get_prompt(category, prompt_name)
    
    if prompt is None:
        raise ValueError(f"Prompt not found: {category}.{prompt_name}")
    
    return prompt

def get_restaurant_prompt_info(prompt_type: str) -> dict:
    """Get complete prompt information including name and role"""
    prompt_mapping = {
        'customer_greeting': ('restaurant_service', 'customer_greeting'),
        'service_intent_analysis': ('restaurant_service', 'service_intent_analysis'),
        'service_gap_analysis': ('restaurant_service', 'service_gap_analysis'),
        'complete_service': ('service_completion', 'complete_service'),
        'gaps_to_additional_services': ('service_completion', 'gaps_to_additional_services'),
        'additional_service_processor': ('service_execution', 'additional_service_processor'),
    }
    
    if prompt_type not in prompt_mapping:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
    
    category, prompt_name = prompt_mapping[prompt_type]
    return prompts_manager.get_prompt_info(category, prompt_name)

def list_available_prompts() -> dict:
    """List all available restaurant prompts by category"""
    return {
        category: prompts_manager.list_prompts(category)
        for category in prompts_manager.list_categories()
    }

def reload_restaurant_prompts():
    """Reload all restaurant prompts from YAML files"""
    prompts_manager.reload_prompts()

def validate_restaurant_prompts() -> dict:
    """Validate all restaurant prompts"""
    return prompts_manager.validate_prompts()

def search_restaurant_prompts(search_term: str) -> dict:
    """Search restaurant prompts"""
    return prompts_manager.search_prompts(search_term)
