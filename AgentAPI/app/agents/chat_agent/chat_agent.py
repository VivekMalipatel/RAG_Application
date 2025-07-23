from pathlib import Path
import yaml
from typing import Optional, Dict, Any
from langchain_core.runnables import RunnableConfig

from agents.base_agents.base_agent import BaseAgent


class ChatAgent(BaseAgent):
    def __init__(self,
                 prompt: Optional[str] = None,
                 *,
                 config: RunnableConfig,
                 model_kwargs: Optional[Dict[str, Any]] = None,
                 vlm_kwargs: Optional[Dict[str, Any]] = None,
                 node_kwargs: Optional[Dict[str, Any]] = None,
                 recursion_limit: Optional[int] = 25,
                 debug: bool = False):
        
        if prompt is None:
            prompt = self._load_prompt()
        
        super().__init__(
            prompt=prompt,
            config=config,
            model_kwargs=model_kwargs,
            vlm_kwargs=vlm_kwargs,
            node_kwargs=node_kwargs,
            recursion_limit=recursion_limit,
            debug=debug
        )
            
    def _load_prompt(self) -> str:
        prompt_path = Path(__file__).parent / "prompt.yaml"
        
        with open(prompt_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
            prompt_data = yaml.safe_load(content)
            if isinstance(prompt_data, dict) and 'ChatAgent' in prompt_data:
                return prompt_data['ChatAgent']
            elif isinstance(prompt_data, str):
                return prompt_data

