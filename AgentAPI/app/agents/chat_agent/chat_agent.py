from pathlib import Path
import yaml
from typing import Optional, Dict, Any
from langchain_core.runnables import RunnableConfig
from agents.utils import _load_prompt
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
                
        yaml_prompt = _load_prompt("ChatAgent", base_dir=Path(__file__).parent)
        if prompt is None:
            self.prompt = yaml_prompt
        else:
            self.prompt = prompt + ("\n" + yaml_prompt if yaml_prompt else "")
        super().__init__(
            prompt=self.prompt,
            config=config,
            model_kwargs=model_kwargs,
            vlm_kwargs=vlm_kwargs,
            node_kwargs=node_kwargs,
            recursion_limit=recursion_limit,
            debug=debug
        )

