"""Model factory for creating LangChain models from Vibe configuration."""

import os
from typing import Optional

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
    
    if provider_config.backend == Backend.MISTRAL:
        # For ChatMistralAI, we need to pass the API key as a string
        kwargs = {
            "model": model_config.name,
            "temperature": model_config.temperature,
            "max_tokens": 16384,
            "base_url": provider_config.api_base,
        }
        if api_key:
            kwargs["api_key"] = api_key
        
        return ChatMistralAI(**kwargs)
    elif provider_config.backend == Backend.GENERIC:
        # For ChatOpenAI, we need to pass the API key as a string
        kwargs = {
            "model": model_config.name,
            "temperature": model_config.temperature,
            "base_url": provider_config.api_base,
        }
        # Only add api_key if it exists, as some local providers don't need it
        if api_key:
            kwargs["api_key"] = api_key
        
        return ChatOpenAI(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {provider_config.backend}")