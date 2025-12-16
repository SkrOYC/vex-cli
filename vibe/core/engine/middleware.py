from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import AgentMiddleware, AgentState

if TYPE_CHECKING:
    from vibe.core.config import VibeConfig


class ContextWarningMiddleware(AgentMiddleware):
    """Warn user when context usage reaches threshold.

    Preserves Vibe's context warning functionality by injecting warnings
    into the conversation when token usage exceeds the configured threshold.
    """

    def __init__(
        self, threshold_percent: float = 0.5, max_context: int | None = None
    ) -> None:
        self.threshold_percent = threshold_percent
        self.max_context = max_context
        self._warned = False

    def before_model(
        self,
        state: AgentState,
        runtime,  # type: ignore
    ) -> dict[str, Any] | None:
        """Check context usage before model call and inject warning if needed."""
        if self._warned or self.max_context is None:
            return None

        # Estimate token count from messages
        messages = state.get("messages", [])
        estimated_tokens = self._estimate_tokens(messages)

        if estimated_tokens >= self.max_context * self.threshold_percent:
            self._warned = True
            # Inject warning message into the conversation
            warning_message = self._create_warning(estimated_tokens, self.max_context)
            # Return dict to modify state - the runtime will handle displaying this
            return {"warning": warning_message}

        return None

    def after_model(
        self,
        state: AgentState,
        runtime,  # type: ignore
    ) -> dict[str, Any] | None:
        """No action needed after model call."""
        return None

    def _estimate_tokens(self, messages: list) -> int:
        """Rough token estimation: ~4 characters per token."""
        total_chars = 0
        for msg in messages:
            if hasattr(msg, "content"):
                total_chars += len(str(msg.content))
            else:
                # Assume msg is a string
                total_chars += len(str(msg))
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

    def __init__(
        self, max_price: float, pricing: dict[str, tuple[float, float]] | None = None
    ):
        self.max_price = max_price
        self.pricing = pricing or {}  # model_name -> (input_rate, output_rate)
        self._total_cost = 0.0

    def before_model(
        self,
        state: AgentState,
        runtime,  # type: ignore
    ) -> dict[str, Any] | None:
        """No action needed before model call."""
        return None

    def after_model(
        self,
        state: AgentState,
        runtime,  # type: ignore
    ) -> dict[str, Any] | None:
        """Track cost after each model call and raise error if limit exceeded."""
        # Extract usage metadata from state (assuming it's added by the agent)
        usage = state.get("last_usage")
        if usage:
            # Get pricing for the model
            model_name = state.get("model_name", "default")
            input_rate, output_rate = self.pricing.get(model_name, (0.0, 0.0))

            # Calculate cost
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
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
    model: "BaseChatModel",  # type: ignore
    backend: "BackendProtocol",  # type: ignore
) -> list[AgentMiddleware]:
    """Build the complete middleware stack for the agent.

    Order is important for correct execution:
    1. Planning (TodoListMiddleware)
    2. Filesystem (FilesystemMiddleware)
    3. Subagents (SubAgentMiddleware, optional)
    4. Summarization (SummarizationMiddleware)
    5. Context warnings (ContextWarningMiddleware, Vibe-specific)
    6. Price limit (PriceLimitMiddleware, Vibe-specific)
    7. Human-in-the-loop (HumanInTheLoopMiddleware, for approvals)
    """
    from langchain.agents.middleware import HumanInTheLoopMiddleware, TodoListMiddleware
    from langchain.agents.middleware.summarization import SummarizationMiddleware

    from deepagents.middleware.filesystem import FilesystemMiddleware
    from deepagents.middleware.subagents import SubAgentMiddleware

    middleware: list[AgentMiddleware] = []

    # 1. Planning
    middleware.append(TodoListMiddleware())

    # 2. Filesystem
    middleware.append(FilesystemMiddleware(backend=backend))

    # 3. Subagents (optional)
    if config.enable_subagents:
        middleware.append(
            SubAgentMiddleware(
                default_model=model,
                default_tools=[],  # Will inherit from main agent
                general_purpose_agent=True,
            )
        )

    # 4. Summarization (replaces AutoCompactMiddleware)
    if model is not None:
        middleware.append(
            SummarizationMiddleware(
                model=model,
                trigger=("tokens", config.auto_compact_threshold),
                keep=("messages", 6),  # Keep last 6 messages
            )
        )

    # 5. Context warnings (Vibe-specific)
    if config.context_warnings:
        middleware.append(
            ContextWarningMiddleware(
                threshold_percent=0.5, max_context=config.auto_compact_threshold
            )
        )

    # 6. Price limit (Vibe-specific)
    if config.max_price is not None:
        # Get pricing from model config
        pricing = {}
        for model_config in config.models:
            pricing[model_config.name] = (
                model_config.input_price / 1_000_000,  # Convert to per-token rate
                model_config.output_price / 1_000_000,
            )

        middleware.append(PriceLimitMiddleware(config.max_price, pricing))

    # 7. Human-in-the-loop (for approvals)
    # This would be added based on interrupt config, but for now placeholder
    # interrupt_on = build_interrupt_config(config)
    # if interrupt_on:
    #     middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))

    return middleware
