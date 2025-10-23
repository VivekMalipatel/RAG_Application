from typing import Any, Dict, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from config import config as app_config


def normalize_provider(provider: Optional[str]) -> str:
    if not provider:
        return "openai"
    value = provider.strip().lower()
    if value in {"azure", "azure_ai", "azure-openai", "azure-openai-deployment"}:
        return "azure_ai"
    if value in {"aws", "bedrock", "aws_bedrock", "amazon", "amazon_bedrock"}:
        return "bedrock"
    if value in {"google", "google_genai", "google-genai", "gemini"}:
        return "google"
    return value


def _clean_kwargs(values: Dict[str, Any]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for key, value in values.items():
        if value is None:
            continue
        if isinstance(value, str) and value == "":
            continue
        cleaned[key] = value
    return cleaned


def build_chat_model(
    provider: str,
    model: str,
    *,
    client_kwargs: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> BaseChatModel:
    normalized = normalize_provider(provider)
    client_kwargs = _clean_kwargs(client_kwargs or {})
    model_kwargs = _clean_kwargs(model_kwargs or {})

    if normalized == "openai":
        from langchain_openai import ChatOpenAI

        params: Dict[str, Any] = {**client_kwargs, **model_kwargs}
        params["model"] = model
        return ChatOpenAI(**params)

    if normalized == "azure_ai":
        from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel

        credential = client_kwargs.pop("credential", None)
        if credential is None:
            credential = client_kwargs.pop("azure_credential", None)
        if isinstance(credential, str):
            from azure.core.credentials import AzureKeyCredential

            credential = AzureKeyCredential(credential)
        if credential is None:
            try:
                from azure.identity import DefaultAzureCredential

                credential = DefaultAzureCredential()
            except ImportError as exc:
                raise ValueError("Azure AI credential is not configured") from exc
        endpoint = client_kwargs.pop("endpoint", None)
        if endpoint is None:
            endpoint = client_kwargs.pop("azure_endpoint", None)
        if endpoint is None:
            endpoint = client_kwargs.pop("project_endpoint", None)
        if endpoint is None:
            raise ValueError("Azure AI endpoint is not configured")
        params = _clean_kwargs({**client_kwargs, **model_kwargs})
        params.update({
            "credential": credential,
            "endpoint": endpoint,
            "model_name": model,
        })
        return AzureAIChatCompletionsModel(**params)

    if normalized == "bedrock":
        from langchain_aws import ChatBedrock

        params: Dict[str, Any] = dict(client_kwargs)
        if not params.get("region_name") and app_config.AWS_REGION_NAME:
            params["region_name"] = app_config.AWS_REGION_NAME
        if not params.get("aws_access_key_id") and app_config.AWS_ACCESS_KEY_ID:
            params["aws_access_key_id"] = app_config.AWS_ACCESS_KEY_ID
        if not params.get("aws_secret_access_key") and app_config.AWS_SECRET_ACCESS_KEY:
            params["aws_secret_access_key"] = app_config.AWS_SECRET_ACCESS_KEY
        if not params.get("aws_session_token") and app_config.AWS_SESSION_TOKEN:
            params["aws_session_token"] = app_config.AWS_SESSION_TOKEN
        if model_kwargs:
            params.setdefault("model_kwargs", {})
            params["model_kwargs"].update(model_kwargs)
        params["model_id"] = model
        return ChatBedrock(**params)

    if normalized == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        params: Dict[str, Any] = {**client_kwargs, **model_kwargs}
        params["model"] = model
        return ChatGoogleGenerativeAI(**params)

    raise ValueError(f"Unsupported llm provider: {provider}")
