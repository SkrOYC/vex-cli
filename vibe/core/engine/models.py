"""Model factory for creating LangChain models from Vibe configuration."""

import os

from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel

from vibe.core.config import VibeConfig, Backend


def create_model_from_config(config: VibeConfig) -> BaseChatModel:
    """Create a LangChain model from Vibe configuration."""
    model_config = config.get_active_model()
    provider_config = config.get_provider_for_model(model_config)
    
    # Get the actual API key from environment variable
    api_key = os.getenv(provider_config.api_key_env_var) if provider_config.api_key_env_var else None
    
    kwargs = {
        "model": model_config.name,
        "temperature": model_config.temperature,
        "base_url": provider_config.api_base,
        "max_tokens": 16384,  # Consistent max_tokens for both backends
    }
    if api_key:
        kwargs["api_key"] = api_key

    if provider_config.backend == Backend.MISTRAL:
        return ChatMistralAI(**kwargs)
    elif provider_config.backend == Backend.GENERIC:
        return ChatOpenAI(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {provider_config.backend}")
