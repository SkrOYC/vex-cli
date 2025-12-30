"""Tests for create_model_from_config function."""

from __future__ import annotations

import os
from unittest.mock import patch

from vibe.core.config import Backend, ModelConfig, ProviderConfig, VibeConfig
from vibe.core.engine.models import create_model_from_config


class TestCreateModelFromConfig:
    """Test the create_model_from_config function."""

    def test_create_model_with_mistral_backend(self):
        """Test creating a model with Mistral backend."""
        # Set up a mock API key
        os.environ["TEST_MISTRAL_KEY"] = "test-key"

        try:
            config = VibeConfig(
                active_model="test-model",
                providers=[
                    ProviderConfig(
                        name="test-mistral",
                        api_base="https://api.mistral.ai/v1",  # Use valid Mistral API base
                        api_key_env_var="TEST_MISTRAL_KEY",
                        backend=Backend.MISTRAL,
                    )
                ],
                models=[
                    ModelConfig(
                        name="test-model-name",
                        provider="test-mistral",
                        alias="test-model",
                        temperature=0.7,
                    )
                ],
            )

            # Mock the actual model creation to avoid external dependencies
            with patch("vibe.core.engine.models.ChatMistralAI") as mock_mistral:
                create_model_from_config(config)

                # Verify the mock was called with correct parameters
                mock_mistral.assert_called_once()
                call_args = mock_mistral.call_args
                assert call_args[1]["model"] == "test-model-name"
                assert call_args[1]["temperature"] == 0.7
                assert call_args[1]["base_url"] == "https://api.mistral.ai/v1"
                assert call_args[1]["max_tokens"] == 16384
                assert call_args[1]["api_key"] == "test-key"

        finally:
            # Clean up environment variable
            if "TEST_MISTRAL_KEY" in os.environ:
                del os.environ["TEST_MISTRAL_KEY"]

    def test_create_model_with_generic_backend(self):
        """Test creating a model with Generic backend (OpenAI-compatible)."""
        # Set up a mock API key
        os.environ["TEST_OPENAI_KEY"] = "test-key"

        try:
            config = VibeConfig(
                active_model="test-model",
                providers=[
                    ProviderConfig(
                        name="test-openai",
                        api_base="https://api.test-openai.com/v1",
                        api_key_env_var="TEST_OPENAI_KEY",
                        backend=Backend.GENERIC,
                    )
                ],
                models=[
                    ModelConfig(
                        name="test-model-name",
                        provider="test-openai",
                        alias="test-model",
                        temperature=0.5,
                    )
                ],
            )

            # Mock the actual model creation to avoid external dependencies
            with patch("vibe.core.engine.models.ChatOpenAI") as mock_openai:
                create_model_from_config(config)

                # Verify the mock was called with correct parameters
                mock_openai.assert_called_once()
                call_args = mock_openai.call_args
                assert call_args[1]["model"] == "test-model-name"
                assert call_args[1]["temperature"] == 0.5
                assert call_args[1]["base_url"] == "https://api.test-openai.com/v1"
                assert call_args[1]["max_tokens"] == 16384  # Should be consistent now
                assert call_args[1]["api_key"] == "test-key"

        finally:
            # Clean up environment variable
            if "TEST_OPENAI_KEY" in os.environ:
                del os.environ["TEST_OPENAI_KEY"]

    def test_create_model_with_generic_backend_no_api_key(self):
        """Test creating a model with Generic backend when no API key is needed."""
        config = VibeConfig(
            active_model="local-model",
            providers=[
                ProviderConfig(
                    name="local-provider",
                    api_base="http://localhost:8080/v1",
                    api_key_env_var="",  # No API key needed
                    backend=Backend.GENERIC,
                )
            ],
            models=[
                ModelConfig(
                    name="local-model-name",
                    provider="local-provider",
                    alias="local-model",
                    temperature=0.3,
                )
            ],
        )

        # Mock the actual model creation to avoid external dependencies
        with patch("vibe.core.engine.models.ChatOpenAI") as mock_openai:
            create_model_from_config(config)

            # Verify the mock was called with correct parameters
            mock_openai.assert_called_once()
            call_args = mock_openai.call_args
            assert call_args[1]["model"] == "local-model-name"
            assert call_args[1]["temperature"] == 0.3
            assert call_args[1]["base_url"] == "http://localhost:8080/v1"
            assert call_args[1]["max_tokens"] == 16384
            # api_key should not be in kwargs when it's None
            assert "api_key" not in call_args[1] or call_args[1]["api_key"] is None

    def test_create_model_unknown_backend(self):
        """Test creating a model with an unknown backend raises an error."""
        # Since we can't easily create an invalid backend, let's test the error path
        # by patching the function that gets the provider to return one with an
        # invalid backend value
        from unittest.mock import MagicMock

        # Create a config with a regular setup
        config = VibeConfig(
            active_model="test-model",
            providers=[
                ProviderConfig(
                    name="test-provider",
                    api_base="https://api.test.com/v1",
                    api_key_env_var="",
                    backend=Backend.GENERIC,
                )
            ],
            models=[
                ModelConfig(
                    name="test-model-name", provider="test-provider", alias="test-model"
                )
            ],
        )

        # Mock the entire create_model_from_config function to test the error handling
        # by patching the get_provider_for_model method on the class
        with patch.object(VibeConfig, "get_provider_for_model") as mock_get_provider:
            # Create a mock provider with an invalid backend
            mock_provider = MagicMock()
            mock_provider.backend = "UNKNOWN"  # Not in Backend enum
            mock_provider.api_key_env_var = ""
            mock_provider.api_base = "https://api.test.com/v1"

            mock_get_provider.return_value = mock_provider

            # Patch the os.getenv to return None (no API key)
            with patch("vibe.core.engine.models.os.getenv", return_value=None):
                # This should raise a ValueError
                try:
                    create_model_from_config(config)
                    raise AssertionError("Expected ValueError for unknown backend")
                except ValueError as e:
                    assert "Unknown backend" in str(e)
