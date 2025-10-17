from pathlib import Path
from typing import Optional, Dict, Any
from langchain_core.runnables import RunnableConfig
from agents.utils import _load_prompt
from agents.base_agents.base_agent import BaseAgent
from tools.v3ya_tools.api_tools.api_service import (
    configurations_table_tool,
    parts_table_tool,
    profiles_table_tool,
    quotes_table_tool,
)

_PROMPT_DIR = Path(__file__).parent
_BASE_PROMPT_VALUE = _load_prompt("V3yaAgent", base_dir=_PROMPT_DIR)
if not isinstance(_BASE_PROMPT_VALUE, str):
    _BASE_PROMPT_VALUE = ""
_STAGE_PROMPTS_VALUE = _load_prompt("V3yaAgentStagePrompts", base_dir=_PROMPT_DIR)
if not isinstance(_STAGE_PROMPTS_VALUE, dict):
    _STAGE_PROMPTS_VALUE = {}


class V3yaAgent(BaseAgent):
    def __init__(self,
                 prompt: Optional[str] = None,
                 *,
                 config: RunnableConfig,
                 properties: Optional[Dict[str, Any]] = None,
                 model_kwargs: Optional[Dict[str, Any]] = None,
                 vlm_kwargs: Optional[Dict[str, Any]] = None,
                 node_kwargs: Optional[Dict[str, Any]] = None,
                 recursion_limit: Optional[int] = 200,
                 debug: bool = False):
                
        if prompt is None:
            prompt = ""

        properties = dict(properties or {})
        if not isinstance(config, dict):
            raise ValueError("Config must be a dict")
        configurable = config.get("configurable")
        if not isinstance(configurable, dict):
            configurable = {}
            config["configurable"] = configurable
        configurable["properties"] = properties

        stage = properties.get("stage") if isinstance(properties, dict) else None
        stage_key = stage.upper() if isinstance(stage, str) else None
        stage_prompt = _STAGE_PROMPTS_VALUE.get(stage) or _STAGE_PROMPTS_VALUE.get("DEFAULT", "")
        prompt_parts = [stage_prompt.strip(), prompt.strip()]
        prompt_parts = [part for part in prompt_parts if part]
        stage_prefixed_prompt = "\n\n".join(prompt_parts) if prompt_parts else ""

        combined_parts = [stage_prefixed_prompt.strip(), _BASE_PROMPT_VALUE.strip()]
        final_prompt = "\n\n".join(part for part in combined_parts if part)

        super().__init__(
            prompt=final_prompt,
            config=config,
            model_kwargs=model_kwargs,
            vlm_kwargs=vlm_kwargs,
            node_kwargs=node_kwargs,
            recursion_limit=recursion_limit,
            debug=debug,
            enable_profile_memory=True
        )

        all_tools = [
            quotes_table_tool,
            parts_table_tool,
            configurations_table_tool,
            profiles_table_tool,
        ]
        stage_tool_map = {
            "QUOTE_LIST": [
                quotes_table_tool,
                profiles_table_tool
            ],
            "FILE_UPLOAD": [profiles_table_tool,],
            "CONFIGURATION": [
                quotes_table_tool,
                parts_table_tool,
                configurations_table_tool,
                profiles_table_tool,
            ],
            "PREVIEW": [
                quotes_table_tool,
                parts_table_tool,
                configurations_table_tool,
                profiles_table_tool,
            ],
            "CHECKOUT": [
                quotes_table_tool,
                profiles_table_tool,
            ],
            "ORDER_LIST": [
                quotes_table_tool,
            ]
        }
        tools_to_bind = stage_tool_map.get(stage_key, all_tools)
        if tools_to_bind:
            self.bind_tools(tools_to_bind)

