"""
Prompts Manager - Core prompt management functionality

This module contains the PromptsManager class that handles loading and managing
prompts from YAML files.
"""

import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path

class PromptsManager:
    """Manages loading and accessing prompts from YAML files"""
    
    def __init__(self, prompts_dir: Optional[str] = None):
        """
        Initialize the prompts manager
        
        Args:
            prompts_dir: Path to the prompts directory. If None, uses default location.
        """
        if prompts_dir is None:
            # Default to prompts directory relative to this file
            current_dir = Path(__file__).parent
            self.prompts_dir = current_dir
        else:
            self.prompts_dir = Path(prompts_dir)
        
        self.prompts_cache: Dict[str, Dict[str, Any]] = {}
        self._load_all_prompts()
    
    def _load_all_prompts(self):
        """Load all YAML prompt files into cache"""
        if not self.prompts_dir.exists():
            print(f"Warning: Prompts directory not found: {self.prompts_dir}")
            return
        
        for yaml_file in self.prompts_dir.glob("*.yaml"):
            try:
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    prompts_data = yaml.safe_load(f)
                    if prompts_data:
                        self.prompts_cache.update(prompts_data)
            except Exception as e:
                print(f"Error loading prompts from {yaml_file}: {e}")
    
    def get_prompt(self, category: str, prompt_name: str) -> Optional[str]:
        try:
            return self.prompts_cache[category][prompt_name]['prompt']
        except KeyError:
            print(f"Prompt not found: {category}.{prompt_name}")
            return None
    
    def get_prompt_info(self, category: str, prompt_name: str) -> Optional[Dict[str, Any]]:
        """
        Get complete prompt information including name, role, and prompt
        
        Args:
            category: The prompt category
            prompt_name: The specific prompt name
            
        Returns:
            Dictionary with prompt information if found, None otherwise
        """
        try:
            return self.prompts_cache[category][prompt_name]
        except KeyError:
            print(f"Prompt info not found: {category}.{prompt_name}")
            return None
    
    def list_categories(self) -> list:
        """List all available prompt categories"""
        return list(self.prompts_cache.keys())
    
    def list_prompts(self, category: str) -> list:
        """List all prompts in a specific category"""
        if category in self.prompts_cache:
            return list(self.prompts_cache[category].keys())
        return []
    
    def reload_prompts(self):
        """Reload all prompts from files"""
        self.prompts_cache.clear()
        self._load_all_prompts()
    
    def add_custom_prompt(self, category: str, prompt_name: str, 
                         name: str, role: str, prompt: str):
        """
        Add a custom prompt programmatically
        
        Args:
            category: The prompt category
            prompt_name: The prompt identifier
            name: Display name for the prompt
            role: Role description
            prompt: The actual prompt text
        """
        if category not in self.prompts_cache:
            self.prompts_cache[category] = {}
        
        self.prompts_cache[category][prompt_name] = {
            'name': name,
            'role': role,
            'prompt': prompt
        }
    
    def get_prompt_template(self, category: str, prompt_name: str, **kwargs) -> Optional[str]:
        """
        Get a prompt with template variable substitution
        
        Args:
            category: The prompt category
            prompt_name: The prompt name
            **kwargs: Template variables to substitute
            
        Returns:
            The formatted prompt string if found, None otherwise
        """
        prompt = self.get_prompt(category, prompt_name)
        if prompt and kwargs:
            try:
                return prompt.format(**kwargs)
            except KeyError as e:
                print(f"Missing template variable: {e}")
                return prompt
        return prompt
    
    def search_prompts(self, search_term: str) -> Dict[str, list]:
        """
        Search for prompts containing a specific term
        
        Args:
            search_term: Term to search for in prompt names, roles, or content
            
        Returns:
            Dictionary with categories as keys and matching prompt names as values
        """
        results = {}
        search_lower = search_term.lower()
        
        for category, prompts in self.prompts_cache.items():
            matches = []
            for prompt_name, prompt_info in prompts.items():
                # Search in name, role, and prompt content
                searchable_text = (
                    f"{prompt_name} {prompt_info.get('name', '')} "
                    f"{prompt_info.get('role', '')} {prompt_info.get('prompt', '')}"
                ).lower()
                
                if search_lower in searchable_text:
                    matches.append(prompt_name)
            
            if matches:
                results[category] = matches
        
        return results
    
    def export_prompts(self, output_file: str, categories: Optional[list] = None):
        """
        Export prompts to a YAML file
        
        Args:
            output_file: Path to output file
            categories: List of categories to export. If None, exports all.
        """
        export_data = {}
        
        if categories is None:
            export_data = self.prompts_cache
        else:
            for category in categories:
                if category in self.prompts_cache:
                    export_data[category] = self.prompts_cache[category]
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                yaml.dump(export_data, f, default_flow_style=False, allow_unicode=True)
            print(f"Prompts exported to {output_file}")
        except Exception as e:
            print(f"Error exporting prompts: {e}")
    
    def validate_prompts(self) -> Dict[str, list]:
        """
        Validate all loaded prompts for required fields
        
        Returns:
            Dictionary with validation errors by category
        """
        errors = {}
        required_fields = ['name', 'role', 'prompt']
        
        for category, prompts in self.prompts_cache.items():
            category_errors = []
            for prompt_name, prompt_info in prompts.items():
                for field in required_fields:
                    if field not in prompt_info or not prompt_info[field]:
                        category_errors.append(f"{prompt_name}: missing or empty '{field}'")
                
                # Check if prompt is too short (likely incomplete)
                if len(prompt_info.get('prompt', '')) < 50:
                    category_errors.append(f"{prompt_name}: prompt seems too short")
            
            if category_errors:
                errors[category] = category_errors
        
        return errors
