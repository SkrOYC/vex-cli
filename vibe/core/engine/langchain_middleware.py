"""LangChain 1.2.0 middleware for Mistral Vibe.

This module provides custom middleware implementations for the Vibe agent:
- ContextWarningMiddleware: Warns when context usage reaches threshold
- PriceLimitMiddleware: Stops execution when price limit is exceeded
- LoggerMiddleware: Provides deep observability into agent execution cycles

These middleware classes are designed to work natively with LangChain 1.2.0
without any DeepAgents dependencies.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import (  # type: ignore
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, ToolMessage  # type: ignore
from langgraph.prebuilt.tool_node import ToolCallRequest  # type: ignore
from langgraph.types import Command  # type: ignore

from vibe.core.utils import logger

if TYPE_CHECKING:
    from langgraph.runtime import Runtime  # type: ignore


class ContextWarningMiddleware(AgentMiddleware):
    """Warn user when context usage reaches threshold.

    This middleware checks the current token count before each model call
    and injects a warning message into the state when usage exceeds
    the configured threshold percentage of max_context.

    Token counts are obtained from AIMessage.usage_metadata when available,
    falling back to character-based estimation if not present.

    Example:
        middleware = ContextWarningMiddleware(
            threshold_percent=0.5,  # Warn at 50%
            max_context=100000,      # 100K token max context
        )
    """

    def __init__(
        self, threshold_percent: float = 0.5, max_context: int | None = None
    ) -> None:
        """Initialize the context warning middleware.

        Args:
            threshold_percent: Percentage of max_context at which to warn (0.0-1.0)
            max_context: Maximum context window size in tokens
        """
        self.threshold_percent = threshold_percent
        self.max_context = max_context
        self._warned = False

    def before_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Check context usage before model call and inject warning if needed.

        Args:
            state: Current agent state containing messages
            runtime: Runtime context for the middleware

        Returns:
            dict with "warning" key if threshold exceeded, None otherwise
        """
        if self._warned or self.max_context is None:
            return None

        # Get actual token count from usage metadata if available
        current_tokens = self._get_current_token_count(state)

        if current_tokens >= self.max_context * self.threshold_percent:
            self._warned = True
            warning_message = self._create_warning(current_tokens, self.max_context)
            return {"warning": warning_message}

        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """No action needed after model call."""
        return None

    def _get_current_token_count(self, state: AgentState) -> int:
        """Get current token count from usage metadata or estimate from messages.

        Args:
            state: Current agent state containing messages

        Returns:
            Token count from usage_metadata or estimated count
        """
        # Check if we have usage metadata from previous AI responses
        messages = state.get("messages", [])
        for msg in reversed(messages):
            # Only AIMessage has usage_metadata
            if isinstance(msg, AIMessage) and msg.usage_metadata:
                total_tokens = msg.usage_metadata.get("total_tokens", 0)
                if total_tokens > 0:
                    return total_tokens

        # Fall back to estimation if no usage metadata available
        return self._estimate_tokens(messages)

    def _estimate_tokens(self, messages: list) -> int:
        """Rough token estimation: ~4 characters per token.

        This is a fallback when usage_metadata is not available.

        Args:
            messages: List of messages to estimate

        Returns:
            Estimated token count
        """
        total_chars = 0
        for msg in messages:
            content = getattr(msg, "content", msg)
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                # Handle multi-part content (e.g., vision models)
                for part in content:
                    if isinstance(part, str):
                        total_chars += len(part)
                    elif isinstance(part, dict) and "text" in part:
                        total_chars += len(str(part.get("text", "")))
            elif isinstance(content, dict) and "text" in content:
                total_chars += len(str(content.get("text", "")))
        return total_chars // 4

    def _create_warning(self, current_tokens: int, max_tokens: int) -> str:
        """Create a formatted warning message.

        Args:
            current_tokens: Current token count
            max_tokens: Maximum allowed tokens

        Returns:
            Formatted warning message string
        """
        percentage_used = (current_tokens / max_tokens) * 100
        return (
            f"You have used {percentage_used:.0f}% of your total context "
            f"({current_tokens:,}/{max_tokens:,} tokens)"
        )


class PriceLimitMiddleware(AgentMiddleware):
    """Stop execution when price limit is exceeded.

    This middleware tracks cumulative cost across model calls and raises
    a RuntimeError when the total exceeds the configured max_price.

    Cost is calculated using actual token counts from usage_metadata
    multiplied by the per-token pricing rates.

    Example:
        middleware = PriceLimitMiddleware(
            max_price=10.0,  # $10 limit
            model_name="gpt-4o",
            pricing={
                "gpt-4o": (0.000005, 0.000015),  # $5/$15 per 1M tokens
            },
        )
    """

    def __init__(
        self,
        max_price: float,
        model_name: str,
        pricing: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        """Initialize the price limit middleware.

        Args:
            max_price: Maximum total cost allowed in dollars
            model_name: The name of the model to use for pricing lookup
            pricing: Dict mapping model names to (input_rate, output_rate) tuples
                    Rates are per-token (not per-million-tokens)
        """
        self.max_price = max_price
        self.model_name = model_name
        self.pricing = pricing or {}  # model_name -> (input_rate, output_rate)
        self._total_cost = 0.0

    def before_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """No action needed before model call."""
        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Track cost after each model call and raise error if limit exceeded.

        Args:
            state: Current agent state containing messages
            runtime: Runtime context for the middleware

        Returns:
            None

        Raises:
            RuntimeError: If cumulative cost exceeds max_price
        """
        # Get the latest AI message from state
        messages = state.get("messages", [])
        if messages and isinstance(messages[-1], AIMessage):
            latest_response = messages[-1]
            if latest_response.usage_metadata:
                # Get pricing for the model using stored model_name
                input_rate, output_rate = self.pricing.get(self.model_name, (0.0, 0.0))

                # Calculate cost using actual token counts
                input_tokens = latest_response.usage_metadata.get("input_tokens", 0)
                output_tokens = latest_response.usage_metadata.get("output_tokens", 0)
                cost = (input_tokens * input_rate) + (output_tokens * output_rate)
                self._total_cost += cost

                # Check limit
                if self._total_cost > self.max_price:
                    raise RuntimeError(
                        f"Price limit exceeded: ${self._total_cost:.4f} > ${self.max_price:.2f}"
                    )

        return None


class LoggerMiddleware(AgentMiddleware):
    """Log agent execution for observability and debugging.

    This middleware provides deep visibility into the agent's internal reasoning
    process by logging session lifecycle, model interactions, and tool executions.
    Logs are routed through Vibe's existing logging infrastructure.

    Example:
        middleware = LoggerMiddleware(enabled=True)
    """

    def __init__(self, enabled: bool = True) -> None:
        """Initialize the logger middleware.

        Args:
            enabled: Whether to enable logging (default: True)
        """
        self.enabled = enabled

    def before_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Log the start of a new agent session."""
        if self.enabled:
            session_id = getattr(runtime, "session_id", "unknown")
            logger.info(f"[SESSION START] Agent session {session_id} initiated")
        return None

    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Log the completion of the agent session."""
        if self.enabled:
            session_id = getattr(runtime, "session_id", "unknown")
            logger.info(f"[SESSION END] Agent session {session_id} completed")
        return None

    def _log_model_request(self, request: ModelRequest) -> None:
        """Logs model request details."""
        model_name = getattr(request.model, "model_name", "unknown")
        message_count = len(request.messages)
        tools_count = len(request.tools) if request.tools else 0
        logger.info(
            f"[MODEL REQUEST] Model: {model_name}, Messages: {message_count}, Tools: {tools_count}"
        )

    def _log_model_response(self, response: ModelResponse) -> None:
        """Logs model response details, including token usage."""
        if response.result:
            usage = getattr(response.result[0], "usage_metadata", None)
            if usage:
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)
                logger.info(
                    f"[MODEL RESPONSE] Usage: input={input_tokens}, output={output_tokens}, total={total_tokens}"
                )
            else:
                logger.info("[MODEL RESPONSE] No usage metadata available")
        else:
            logger.info("[MODEL RESPONSE] No usage metadata available")

        if response.structured_response:
            logger.info(
                f"[MODEL RESPONSE] Structured response: {response.structured_response}"
            )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Log model request and response for observability."""
        if not self.enabled:
            return handler(request)

        self._log_model_request(request)
        try:
            response = handler(request)
            self._log_model_response(response)
            return response
        except Exception as e:
            logger.error(f"[MODEL ERROR] Model call failed: {e}")
            raise

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Async version of wrap_model_call."""
        if not self.enabled:
            return await handler(request)

        self._log_model_request(request)
        try:
            response = await handler(request)
            self._log_model_response(response)
            return response
        except Exception as e:
            logger.error(f"[MODEL ERROR] Model call failed: {e}")
            raise

    def _log_tool_request(self, tool_name: str, tool_args: dict) -> None:
        """Logs tool call request details."""
        logger.info(f"[TOOL CALL] {tool_name}({tool_args})")

    def _log_tool_result(self, tool_name: str, result: Any) -> None:
        """Logs tool call result, with truncation."""
        content = result.content if isinstance(result, ToolMessage) else result
        truncated_result = self._truncate_result(content)
        logger.info(f"[TOOL RESULT] {tool_name}: {truncated_result}")

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Log tool execution for observability."""
        if not self.enabled:
            return handler(request)

        tool_name = request.tool_call.get("name", "unknown")
        self._log_tool_request(tool_name, request.tool_call.get("arguments", {}))

        try:
            result = handler(request)
            self._log_tool_result(tool_name, result)
            return result
        except Exception as e:
            logger.error(f"[TOOL ERROR] {tool_name} failed: {e}")
            raise

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Async version of wrap_tool_call."""
        if not self.enabled:
            return await handler(request)

        tool_name = request.tool_call.get("name", "unknown")
        self._log_tool_request(tool_name, request.tool_call.get("arguments", {}))

        try:
            result = await handler(request)
            self._log_tool_result(tool_name, result)
            return result
        except Exception as e:
            logger.error(f"[TOOL ERROR] {tool_name} failed: {e}")
            raise

    def _truncate_result(self, result: Any, max_length: int = 200) -> str:
        """Truncate result for logging if too large."""
        result_str = str(result)
        if len(result_str) > max_length:
            return result_str[:max_length] + "..."
        return result_str
