from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage

if TYPE_CHECKING:
    from langchain.chat_models.base import BaseChatModel
    from langgraph.runtime import Runtime

    from vibe.core.config import VibeConfig


class ContextWarningMiddleware(AgentMiddleware):
    """Warn user when context usage reaches threshold.

    Preserves Vibe's context warning functionality by injecting warnings
    into the conversation when token usage exceeds the configured threshold.
    """

    @property
    def name(self) -> str:
        return "context_warning"

    def __init__(
        self, threshold_percent: float = 0.5, max_context: int | None = None
    ) -> None:
        self.threshold_percent = threshold_percent
        self.max_context = max_context
        self._warned = False

    def before_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Check context usage before model call and inject warning if needed."""
        if self._warned or self.max_context is None:
            return None

        # Get actual token count from usage metadata if available
        current_tokens = self._get_current_token_count(state)

        if current_tokens >= self.max_context * self.threshold_percent:
            self._warned = True
            # Inject warning message into the conversation
            warning_message = self._create_warning(current_tokens, self.max_context)
            # Return dict to modify state - the runtime will handle displaying this
            return {"warning": warning_message}

        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """No action needed after model call."""
        return None

    def _get_current_token_count(self, state: AgentState) -> int:
        """Get current token count from usage metadata or estimate from messages."""
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
        """Rough token estimation: ~4 characters per token."""
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
        """Create a formatted warning message."""
        percentage_used = (current_tokens / max_tokens) * 100
        return (
            f"You have used {percentage_used:.0f}% of your total context "
            f"({current_tokens:,}/{max_tokens:,} tokens)"
        )


class PriceLimitMiddleware(AgentMiddleware):
    """Stop execution when price limit is exceeded.

    Preserves Vibe's cost control functionality by tracking cumulative
    cost across model calls and raising an interrupt when the limit is reached.
    """

    @property
    def name(self) -> str:
        return "price_limit"

    def __init__(
        self,
        max_price: float,
        model_name: str,
        pricing: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self.max_price = max_price
        self.model_name = model_name
        self.pricing = pricing or {}  # model_name -> (input_rate, output_rate)
        self._total_cost = 0.0

    def before_model(
        self,
        state: AgentState,
        runtime: Any,  # type: ignore
    ) -> dict[str, Any] | None:
        """No action needed before model call."""
        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Track cost after each model call and raise error if limit exceeded."""
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


def build_middleware_stack(
    config: VibeConfig,
    model: BaseChatModel,  # type: ignore
    backend: Any,  # type: ignore
) -> list[AgentMiddleware]:
    """Build the complete middleware stack for the agent.

    Order is important for correct execution:
    1. Subagents (SubAgentMiddleware, optional) - DeepAgents provides TodoList and Filesystem by default
    2. Context warnings (ContextWarningMiddleware, Vibe-specific)
    3. Price limit (PriceLimitMiddleware, Vibe-specific)
    4. Human-in-the-loop (HumanInTheLoopMiddleware, for approvals)
    """
    from langchain.agents.middleware import HumanInTheLoopMiddleware

    middleware: list[AgentMiddleware] = []

    # DeepAgents provides TodoListMiddleware, FilesystemMiddleware, and SubAgentMiddleware by default
    # Only add custom middleware that's not already provided by DeepAgents

    # 1. Subagents (optional, Vibe-specific) - handled by DeepAgents automatically
    # Note: Don't add SubAgentMiddleware manually as it causes duplicate middleware error

    # 2. Context warnings (Vibe-specific)
    if config.context_warnings:
        middleware.append(
            ContextWarningMiddleware(
                threshold_percent=0.5, max_context=config.auto_compact_threshold
            )
        )

    # 3. Price limit (Vibe-specific)
    if config.max_price is not None:
        # Get pricing from model config
        pricing = {}
        for model_config in config.models:
            pricing[model_config.name] = (
                model_config.input_price / 1_000_000,  # Convert to per-token rate
                model_config.output_price / 1_000_000,
            )

        # Get the active model's name for pricing lookup
        active_model = config.get_active_model()
        model_name = active_model.name

        middleware.append(PriceLimitMiddleware(config.max_price, model_name, pricing))

    # 4. Human-in-the-loop (for approvals) - independent of price limit
    from vibe.core.engine.permissions import build_interrupt_config

    interrupt_on = build_interrupt_config(config)
    if interrupt_on:
        from langchain.agents.middleware import HumanInTheLoopMiddleware

        middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))

    return middleware
