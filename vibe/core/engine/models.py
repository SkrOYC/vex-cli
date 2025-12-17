"""Model factory for creating LangChain models from Vibe configuration."""

import os

from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from vibe.core.config import VibeConfig, Backend


def create_model_from_config(config: VibeConfig) -> BaseChatModel:
    """Create a LangChain model from Vibe configuration."""
    model_config = config.get_active_model()
    provider_config = config.get_provider_for_model(model_config)

    # Get the actual API key from environment variable
    api_key = (
        os.getenv(provider_config.api_key_env_var)
        if provider_config.api_key_env_var
        else None
    )

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


def estimate_tokens_for_messages(
    model: BaseChatModel, messages: list[BaseMessage]
) -> int:
    """Estimate token count for messages using the model's tokenizer when available."""
    # If the model has a built-in token counter, use it
    if hasattr(model, "get_num_tokens_from_messages"):
        try:
            return model.get_num_tokens_from_messages(messages)
        except Exception:
            pass  # Fall back to estimation if the method fails

    # If the model has a get_num_tokens method, use it for individual messages
    if hasattr(model, "get_num_tokens"):
        try:
            total_tokens = 0
            for msg in messages:
                content = getattr(msg, "content", str(msg))
                if isinstance(content, list):
                    # Handle list of content parts (e.g., multimodal)
                    for part in content:
                        if isinstance(part, str):
                            total_tokens += model.get_num_tokens(part)
                        elif isinstance(part, dict) and "text" in part:
                            total_tokens += model.get_num_tokens(
                                str(part.get("text", ""))
                            )
                        else:
                            total_tokens += model.get_num_tokens(str(part))
                elif isinstance(content, str):
                    total_tokens += model.get_num_tokens(content)
                else:
                    total_tokens += model.get_num_tokens(str(content))
            return total_tokens
        except Exception:
            pass  # Fall back to estimation if the method fails

    # Fallback: rough estimation (4 characters per token)
    total_chars = 0
    for msg in messages:
        content = getattr(msg, "content", str(msg))
        if isinstance(content, list):
            for part in content:
                if isinstance(part, str):
                    total_chars += len(part)
                elif isinstance(part, dict) and "text" in part:
                    total_chars += len(str(part.get("text", "")))
                else:
                    total_chars += len(str(part))
        elif isinstance(content, str):
            total_chars += len(content)
        else:
            total_chars += len(str(content))

    return max(1, total_chars // 4)  # At least 1 token
