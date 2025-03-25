import logging
import json
import asyncio
from pydantic import BaseModel
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from app.core.models.model_handler import ModelRouter
from app.core.models.model_type import ModelType
from app.core.models.model_provider import Provider
from app.config import settings


class BaseAgent(ABC):
    """
    Abstract Base Class for OmniRAG Agents.
    Provides a common interface for defining AI-driven agents with modular execution.
    """

    def __init__(
        self,
        agent_name: str,
        provider: Provider = Provider.OLLAMA,
        model_name: str = settings.TEXT_LLM_MODEL_NAME,
        model_quantization: str = settings.TEXT_LLM_QUANTIZATION,
        system_prompt: Optional[str] = None,
        temperature: float = settings.TEXT_LLM_TEMPERATURE,
        top_p: float = settings.TEXT_LLM_TOP_P,
    ):
        """
        Initializes the agent with LLM and execution settings.

        Args:
            agent_name (str): Name of the agent.
            provider (Provider): LLM Provider.
            model_name (str): LLM model name.
            model_quantization (str): Quantization level for the model.
            system_prompt (str, optional): System prompt guiding agent behavior.
            temperature (float): LLM temperature setting.
            top_p (float): LLM top-p setting.
        """
        self.agent_name = agent_name
        self.model = ModelRouter(
            provider=provider,
            model_name=model_name,
            model_quantization=model_quantization,
            model_type=ModelType.TEXT_GENERATION,
            system_prompt=system_prompt if system_prompt else f"{agent_name} is an AI assistant.",
            temperature=temperature,
            top_p=top_p,
        )

    @abstractmethod
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Core execution function. Each agent must implement this method.

        Args:
            inputs (Dict[str, Any]): The input data for the agent.

        Returns:
            Dict[str, Any]: Processed output.
        """
        pass

    async def generate_response(self, prompt: str) -> str:
        """
        Uses the LLM to generate a response.

        Args:
            prompt (str): Input prompt.

        Returns:
            str: LLM-generated response.
        """
        try:
            response = await self.model.generate_text(prompt = prompt)
            return response
        except Exception as e:
            logging.error(f"[{self.agent_name}] LLM error: {str(e)}")
            return "An error occurred during response generation."
    
    async def generate_structured_response(self, prompt: str, schema: BaseModel):
        """
        Uses the LLM to generate a structured response.

        Args:
            prompt (str): Input prompt.
            schema (BaseModel): Pydantic schema for the structured response.

        Returns:
            Dict: LLM-generated structured response.
        """
        response = await self.model.client.generate_structured_output(prompt = prompt, schema = schema)
        try:
            return response
        except Exception as e:
            logging.error(f"[{self.agent_name}] Structured response parsing error: {str(e)}")
            return None

    async def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the agent's logic and manages retries.

        Args:
            inputs (Dict[str, Any]): The input data for the agent.

        Returns:
            Dict[str, Any]: Processed output.
        """
        retries = settings.AGENT_MAX_RETRIES
        for attempt in range(retries):
            try:
                result = await self.execute(inputs)
                return result
            except Exception as e:
                logging.error(f"[{self.agent_name}] Execution failed on attempt {attempt + 1}: {str(e)}")
                await asyncio.sleep(settings.AGENT_RETRY_DELAY)
        return {"error": "Agent execution failed after multiple attempts."}