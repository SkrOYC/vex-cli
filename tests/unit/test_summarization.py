"""Tests for SummarizationMiddleware configuration."""

from __future__ import annotations

import warnings

import pytest

from vibe.core.config import VibeConfig


def test_summarization_config_creation() -> None:
    """Test that summarization config fields can be set and retrieved."""
    config = VibeConfig(
        enable_summarization=True,
        summarization_trigger_tokens=180000,
        summarization_keep_messages=8,
    )

    assert config.enable_summarization is True
    assert config.summarization_trigger_tokens == 180000
    assert config.summarization_keep_messages == 8


def test_summarization_disabled_by_default() -> None:
    """Test that summarization is disabled by default."""
    config = VibeConfig()

    assert config.enable_summarization is False
    assert config.summarization_trigger_tokens == 170000  # Default value
    assert config.summarization_keep_messages == 6  # Default value


def test_summarization_config_validation_trigger_too_low() -> None:
    """Test that validation catches trigger tokens below minimum."""
    with pytest.raises(
        ValueError, match="Summarization trigger must be >= 50000 tokens"
    ):
        VibeConfig(enable_summarization=True, summarization_trigger_tokens=10000)


def test_summarization_config_validation_keep_too_low() -> None:
    """Test that validation catches keep messages below minimum."""
    with pytest.raises(ValueError, match="Must keep at least 2 messages"):
        VibeConfig(enable_summarization=True, summarization_keep_messages=1)


def test_summarization_config_validation_trigger_warning() -> None:
    """Test that warning is issued when trigger < compact threshold."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        VibeConfig(
            enable_summarization=True,
            summarization_trigger_tokens=150000,  # Less than default auto_compact_threshold (200000)
            auto_compact_threshold=200000,
        )

        # Should issue warning
        assert len(w) == 1
        assert "Summarization trigger (150,000 tokens) is less than" in str(
            w[0].message
        )
        assert "Consider adjusting values" in str(w[0].message)


def test_summarization_config_no_validation_when_disabled() -> None:
    """Test that validation doesn't run when summarization is disabled."""
    # Should not raise even with invalid values
    config = VibeConfig(
        enable_summarization=False,
        summarization_trigger_tokens=10000,  # Invalid but ignored
        summarization_keep_messages=1,  # Invalid but ignored
    )

    assert config.enable_summarization is False


def test_summarization_config_edge_cases() -> None:
    """Test that validation accepts minimum valid values."""
    # Should not raise for minimum valid values
    config = VibeConfig(
        enable_summarization=True,
        summarization_trigger_tokens=50000,  # Minimum valid
        summarization_keep_messages=2,  # Minimum valid
    )

    assert config.summarization_trigger_tokens == 50000
    assert config.summarization_keep_messages == 2


def test_summarization_config_from_toml() -> None:
    """Test that configuration can be loaded from TOML-style dict."""
    config_dict = {
        "enable_summarization": True,
        "summarization_trigger_tokens": 160000,
        "summarization_keep_messages": 10,
    }

    config = VibeConfig(**config_dict)

    assert config.enable_summarization is True
    assert config.summarization_trigger_tokens == 160000
    assert config.summarization_keep_messages == 10
