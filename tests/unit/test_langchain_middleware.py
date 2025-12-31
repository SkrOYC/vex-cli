"""Unit tests for langchain_middleware.py - LangChain 1.2.0 native middleware."""

from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime
import pytest

from vibe.core.engine.langchain_middleware import (
    ContextWarningMiddleware,
    LoggerMiddleware,
    PriceLimitMiddleware,
)


class TestContextWarningMiddleware:
    """Test ContextWarningMiddleware functionality."""

    def test_no_warning_when_below_threshold(self):
        """Test that no warning is injected when below threshold."""
        middleware = ContextWarningMiddleware(threshold_percent=0.5, max_context=1000)

        # Create a mock state with messages but low token count
        state = {"messages": ["short message"] * 10}  # ~40 tokens, below 500 threshold

        result = middleware.before_model(state, cast(Runtime, None))
        assert result is None

    def test_warning_when_above_threshold(self):
        """Test that warning is injected when above threshold."""
        from langchain_core.messages import AIMessage

        middleware = ContextWarningMiddleware(threshold_percent=0.5, max_context=1000)

        # Create a message with usage metadata showing 800 tokens (above 500 threshold)
        ai_message = AIMessage(
            content="response",
            usage_metadata={
                "input_tokens": 400,
                "output_tokens": 400,
                "total_tokens": 800,
            },
        )
        state = {"messages": ["long message content"] * 100 + [ai_message]}

        result = middleware.before_model(state, cast(Runtime, None))
        assert result is not None
        assert "warning" in result
        assert "80%" in result["warning"]  # 800/1000 = 80%

    def test_no_warning_after_already_warned(self):
        """Test that warning is only shown once."""
        middleware = ContextWarningMiddleware(threshold_percent=0.5, max_context=1000)
        state = {"messages": ["long message content"] * 100}

        # First call should warn (will fall back to estimation)
        result1 = middleware.before_model(state, cast(Runtime, None))
        assert result1 is not None

        # Second call should not warn (already warned)
        result2 = middleware.before_model(state, cast(Runtime, None))
        assert result2 is None

    def test_no_warning_when_max_context_none(self):
        """Test that no warning is injected when max_context is None."""
        middleware = ContextWarningMiddleware(threshold_percent=0.5, max_context=None)
        state = {"messages": ["long message content"] * 1000}

        result = middleware.before_model(state, cast(Runtime, None))
        assert result is None

    def test_uses_usage_metadata(self):
        """Test that usage_metadata is preferred over estimation."""
        from langchain_core.messages import AIMessage

        middleware = ContextWarningMiddleware(threshold_percent=0.5, max_context=1000)

        # Create a message with usage metadata showing 900 tokens
        ai_message = AIMessage(
            content="short",
            usage_metadata={
                "input_tokens": 450,
                "output_tokens": 450,
                "total_tokens": 900,
            },
        )
        # Add lots of content that would estimate high, but usage_metadata should be used
        state = {"messages": ["very long content " * 1000] * 100 + [ai_message]}

        result = middleware.before_model(state, cast(Runtime, None))
        assert result is not None
        assert "warning" in result
        assert "90%" in result["warning"]  # 900/1000 = 90%

    def test_falls_back_to_estimation(self):
        """Test that estimation is used when usage_metadata is not available."""
        middleware = ContextWarningMiddleware(threshold_percent=0.5, max_context=1000)

        # Create messages without usage_metadata
        # "short message" is 13 chars, * 2000 = 26k chars, /4 = 6500 tokens
        state = {"messages": ["short message"] * 2000}

        result = middleware.before_model(state, cast(Runtime, None))
        assert result is not None
        assert "warning" in result
        assert "650%" in result["warning"]  # 6500/1000 = 650%

    def test_warning_message_format(self):
        """Test warning message is properly formatted."""
        from langchain_core.messages import AIMessage

        middleware = ContextWarningMiddleware(threshold_percent=0.75, max_context=10000)

        ai_message = AIMessage(
            content="response",
            usage_metadata={
                "input_tokens": 4000,
                "output_tokens": 4000,
                "total_tokens": 8000,
            },
        )
        state = {"messages": [ai_message]}

        result = middleware.before_model(state, cast(Runtime, None))
        assert result is not None
        warning = result["warning"]
        assert "80%" in warning  # 8000/10000 = 80%
        assert "8,000" in warning
        assert "10,000" in warning


class TestPriceLimitMiddleware:
    """Test PriceLimitMiddleware functionality."""

    def test_no_error_when_below_limit(self):
        """Test that no error is raised when below price limit."""
        from langchain_core.messages import AIMessage

        pricing = {"test-model": (0.0001, 0.0002)}  # $0.10 per 1k tokens
        middleware = PriceLimitMiddleware(
            max_price=1.0, model_name="test-model", pricing=pricing
        )

        # Create AI message with usage metadata
        ai_message = AIMessage(
            content="response",
            usage_metadata={
                "input_tokens": 1000,
                "output_tokens": 500,
                "total_tokens": 1500,
            },
        )
        state = {"model_name": "test-model", "messages": [ai_message]}

        # Should not raise
        result = middleware.after_model(state, cast(Runtime, None))
        assert result is None

    def test_error_when_above_limit(self):
        """Test that RuntimeError is raised when above price limit."""
        from langchain_core.messages import AIMessage

        pricing = {"test-model": (0.001, 0.002)}  # $1.00 per 1k tokens
        middleware = PriceLimitMiddleware(
            max_price=1.0, model_name="test-model", pricing=pricing
        )

        # Create AI message with usage metadata (1000 input + 500 output = 1500 tokens = $1.50)
        ai_message = AIMessage(
            content="response",
            usage_metadata={
                "input_tokens": 1000,
                "output_tokens": 500,
                "total_tokens": 1500,
            },
        )
        state = {"model_name": "test-model", "messages": [ai_message]}

        with pytest.raises(RuntimeError, match="Price limit exceeded"):
            middleware.after_model(state, cast(Runtime, None))

    def test_accumulates_cost(self):
        """Test that cost accumulates across calls."""
        from langchain_core.messages import AIMessage

        pricing = {"test-model": (0.00005, 0.00005)}  # $0.05 per 1k tokens
        middleware = PriceLimitMiddleware(
            max_price=0.15, model_name="test-model", pricing=pricing
        )

        # First call: 1000 input + 500 output = 1500 tokens = $0.075
        ai_message1 = AIMessage(
            content="response1",
            usage_metadata={
                "input_tokens": 1000,
                "output_tokens": 500,
                "total_tokens": 1500,
            },
        )
        state1 = {"model_name": "test-model", "messages": [ai_message1]}
        middleware.after_model(state1, cast(Runtime, None))

        # Second call: another 1000 input = 1000 tokens = $0.05, total $0.125
        ai_message2 = AIMessage(
            content="response2",
            usage_metadata={
                "input_tokens": 1000,
                "output_tokens": 0,
                "total_tokens": 1000,
            },
        )
        state2 = {"model_name": "test-model", "messages": [ai_message1, ai_message2]}
        middleware.after_model(state2, cast(Runtime, None))

        # Third call: another 1000 input = 1000 tokens = $0.05, total $0.175 > $0.15
        ai_message3 = AIMessage(
            content="response3",
            usage_metadata={
                "input_tokens": 1000,
                "output_tokens": 0,
                "total_tokens": 1000,
            },
        )
        state3 = {
            "model_name": "test-model",
            "messages": [ai_message1, ai_message2, ai_message3],
        }
        with pytest.raises(RuntimeError, match="Price limit exceeded"):
            middleware.after_model(state3, cast(Runtime, None))

    def test_uses_default_pricing_when_model_not_found(self):
        """Test that default pricing is used when model not in pricing dict."""
        from langchain_core.messages import AIMessage

        pricing = {"other-model": (0.001, 0.002)}
        middleware = PriceLimitMiddleware(
            max_price=1.0, model_name="test-model", pricing=pricing
        )

        # Create AI message with unknown model
        ai_message = AIMessage(
            content="response",
            usage_metadata={
                "input_tokens": 1000,
                "output_tokens": 500,
                "total_tokens": 1500,
            },
        )
        state = {"model_name": "unknown-model", "messages": [ai_message]}

        # Should not raise because default pricing (0.0, 0.0) means free
        result = middleware.after_model(state, cast(Runtime, None))
        assert result is None

    def test_no_error_without_usage_metadata(self):
        """Test that no error is raised when usage_metadata is not present."""
        middleware = PriceLimitMiddleware(
            max_price=0.01, model_name="test-model", pricing={}
        )

        # Create AI message without usage_metadata
        ai_message = AIMessage(content="response")
        state = {"model_name": "test-model", "messages": [ai_message]}

        # Should not raise
        result = middleware.after_model(state, cast(Runtime, None))
        assert result is None

    def test_before_model_returns_none(self):
        """Test that before_model always returns None."""
        middleware = PriceLimitMiddleware(
            max_price=1.0, model_name="test-model", pricing={}
        )
        state = {"messages": []}

        result = middleware.before_model(state, cast(Runtime, None))
        assert result is None


class TestLoggerMiddleware:
    """Test LoggerMiddleware functionality."""

    def test_before_agent_logs_session_start(self):
        """Test that before_agent logs session start."""
        middleware = LoggerMiddleware(enabled=True)

        # Mock runtime with session_id
        runtime = type("Runtime", (), {"session_id": "test-session"})()

        with patch("vibe.core.engine.langchain_middleware.logger") as mock_logger:
            result = middleware.before_agent({}, runtime)
            assert result is None
            mock_logger.info.assert_called_once_with(
                "[SESSION START] Agent session test-session initiated"
            )

    def test_after_agent_logs_session_end(self):
        """Test that after_agent logs session end."""
        middleware = LoggerMiddleware(enabled=True)
        runtime = type("Runtime", (), {"session_id": "test-session"})()

        with patch("vibe.core.engine.langchain_middleware.logger") as mock_logger:
            result = middleware.after_agent({}, runtime)
            assert result is None
            mock_logger.info.assert_called_once_with(
                "[SESSION END] Agent session test-session completed"
            )

    def test_before_agent_disabled(self):
        """Test that logging is disabled when enabled=False."""
        from unittest.mock import patch

        middleware = LoggerMiddleware(enabled=False)
        runtime = type("Runtime", (), {"session_id": "test-session"})()

        with patch("vibe.core.engine.langchain_middleware.logger") as mock_logger:
            result = middleware.before_agent({}, runtime)
            assert result is None
            mock_logger.info.assert_not_called()

    def test_wrap_model_call_logs_request_and_response(self):
        """Test wrap_model_call logs request and response."""
        from unittest.mock import patch

        from langchain_core.messages import AIMessage

        middleware = LoggerMiddleware(enabled=True)

        # Mock model request
        mock_model = MagicMock()
        mock_model.model_name = "test-model"
        request = MagicMock()
        request.model = mock_model
        request.messages = ["msg1", "msg2", "msg3"]
        request.tools = [{"name": "tool1"}, {"name": "tool2"}]

        # Mock response
        mock_response = MagicMock()
        mock_response.result = [
            AIMessage(
                content="test",
                usage_metadata={
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "total_tokens": 150,
                },
            )
        ]
        mock_response.structured_response = None

        def mock_handler(req):
            return mock_response

        with patch("vibe.core.engine.langchain_middleware.logger") as mock_logger:
            result = middleware.wrap_model_call(request, mock_handler)
            assert result == mock_response

            # Check request logging
            request_call = mock_logger.info.call_args_list[0]
            assert (
                "[MODEL REQUEST] Model: test-model, Messages: 3, Tools: 2"
                in request_call[0][0]
            )

            # Check response logging
            response_call = mock_logger.info.call_args_list[1]
            assert "input=100, output=50, total=150" in response_call[0][0]

    def test_wrap_model_call_with_structured_response(self):
        """Test wrap_model_call logs structured response."""
        from unittest.mock import patch

        from langchain_core.messages import AIMessage

        middleware = LoggerMiddleware(enabled=True)

        mock_model = MagicMock()
        mock_model.model_name = "test-model"
        request = MagicMock()
        request.model = mock_model
        request.messages = ["msg"]
        request.tools = []

        mock_response = MagicMock()
        mock_response.result = [AIMessage(content="test")]
        mock_response.structured_response = {"key": "value"}

        def mock_handler(req):
            return mock_response

        with patch("vibe.core.engine.langchain_middleware.logger") as mock_logger:
            middleware.wrap_model_call(request, mock_handler)

            # Check structured response logging
            found_log = any(
                "Structured response" in call.args[0]
                for call in mock_logger.info.call_args_list
            )
            assert found_log, "Structured response log message not found."

    def test_wrap_model_call_no_usage_metadata(self):
        """Test wrap_model_call when no usage metadata."""
        from unittest.mock import patch

        from langchain_core.messages import AIMessage

        middleware = LoggerMiddleware(enabled=True)

        mock_model = MagicMock()
        mock_model.model_name = "test-model"
        request = MagicMock()
        request.model = mock_model
        request.messages = ["msg"]
        request.tools = []

        mock_response = MagicMock()
        mock_response.result = [AIMessage(content="test")]
        mock_response.structured_response = None

        def mock_handler(req):
            return mock_response

        with patch("vibe.core.engine.langchain_middleware.logger") as mock_logger:
            middleware.wrap_model_call(request, mock_handler)

            # Check no usage message
            found_log = any(
                "No usage metadata available" in call.args[0]
                for call in mock_logger.info.call_args_list
            )
            assert found_log, "No usage metadata log message not found."

    def test_wrap_model_call_disabled(self):
        """Test wrap_model_call when disabled."""
        from unittest.mock import patch

        middleware = LoggerMiddleware(enabled=False)
        request = MagicMock()
        response = MagicMock()

        def mock_handler(req):
            return response

        with patch("vibe.core.engine.langchain_middleware.logger") as mock_logger:
            result = middleware.wrap_model_call(request, mock_handler)
            assert result == response
            mock_logger.info.assert_not_called()

    def test_wrap_tool_call_logs_execution(self):
        """Test wrap_tool_call logs tool execution."""
        from unittest.mock import patch

        middleware = LoggerMiddleware(enabled=True)

        request = MagicMock()
        request.tool_call = {"name": "test_tool", "arguments": {"param": "value"}}

        mock_result = MagicMock()
        mock_result.content = "tool result"

        def mock_handler(req):
            return mock_result

        with (
            patch("vibe.core.engine.langchain_middleware.logger") as mock_logger,
            patch(
                "vibe.core.engine.langchain_middleware.isinstance", return_value=True
            ) as mock_isinstance,
        ):
            result = middleware.wrap_tool_call(request, mock_handler)
            assert result == mock_result

            # Check tool call logging
            call_log = mock_logger.info.call_args_list[0]
            assert "[TOOL CALL] test_tool" in call_log[0][0]
            assert "param" in call_log[0][0]

            # Check result logging
            result_log = mock_logger.info.call_args_list[1]
            assert "[TOOL RESULT] test_tool: tool result" in result_log[0][0]

    def test_wrap_tool_call_truncates_large_result(self):
        """Test wrap_tool_call truncates large results."""
        from unittest.mock import patch

        middleware = LoggerMiddleware(enabled=True)

        request = MagicMock()
        request.tool_call = {"name": "test_tool", "arguments": {}}

        mock_result = MagicMock()
        mock_result.content = "x" * 300  # Large content

        def mock_handler(req):
            return mock_result

        with (
            patch("vibe.core.engine.langchain_middleware.logger") as mock_logger,
            patch(
                "vibe.core.engine.langchain_middleware.isinstance", return_value=True
            ),
        ):
            middleware.wrap_tool_call(request, mock_handler)

            result_log = mock_logger.info.call_args_list[1]
            logged_content = result_log[0][0]
            assert len(logged_content) <= 250  # Allow for prefix
            assert logged_content.endswith("...")

    def test_wrap_tool_call_no_content_attribute(self):
        """Test wrap_tool_call handles results without content attribute."""
        from unittest.mock import patch

        middleware = LoggerMiddleware(enabled=True)

        request = MagicMock()
        request.tool_call = {"name": "test_tool", "arguments": {}}

        mock_result = "plain string result"  # No content attribute

        def mock_handler(req):
            return mock_result

        with patch("vibe.core.engine.langchain_middleware.logger") as mock_logger:
            result = middleware.wrap_tool_call(request, mock_handler)
            assert result == mock_result

            result_log = mock_logger.info.call_args_list[1]
            assert "plain string result" in result_log[0][0]

    def test_model_call_error_handling(self):
        """Test model call error handling."""
        from unittest.mock import patch

        middleware = LoggerMiddleware(enabled=True)
        request = MagicMock()
        request.model = MagicMock()
        request.model.model_name = "test-model"
        request.messages = []
        request.tools = []

        def failing_handler(req):
            raise ValueError("Test error")

        with patch("vibe.core.engine.langchain_middleware.logger") as mock_logger:
            with pytest.raises(ValueError, match="Test error"):
                middleware.wrap_model_call(request, failing_handler)

            error_log = mock_logger.error.call_args_list[0]
            assert "Model call failed" in error_log[0][0]

    def test_tool_call_error_handling(self):
        """Test tool call error handling."""
        from unittest.mock import patch

        middleware = LoggerMiddleware(enabled=True)
        request = MagicMock()
        request.tool_call = {"name": "test_tool", "arguments": {}}

        def failing_handler(req):
            raise RuntimeError("Tool failed")

        with patch("vibe.core.engine.langchain_middleware.logger") as mock_logger:
            with pytest.raises(RuntimeError, match="Tool failed"):
                middleware.wrap_tool_call(request, failing_handler)

            error_log = mock_logger.error.call_args_list[0]
            assert "test_tool failed" in error_log[0][0]

    def test_truncate_result(self):
        """Test _truncate_result method."""
        middleware = LoggerMiddleware()

        # Short result
        assert middleware._truncate_result("short") == "short"

        # Long result
        long_result = "x" * 300
        truncated = middleware._truncate_result(long_result)
        assert len(truncated) == 203  # 200 + "..."
        assert truncated.endswith("...")

        # Exact length
        exact = "x" * 200
        assert middleware._truncate_result(exact) == exact
