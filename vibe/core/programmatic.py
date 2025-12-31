from __future__ import annotations

import asyncio

from vibe.core.config import VibeConfig
from vibe.core.engine.langchain_engine import VibeLangChainEngine
from vibe.core.output_formatters import create_formatter
from vibe.core.types import AssistantEvent, LLMMessage, OutputFormat, Role
from vibe.core.utils import ConversationLimitException, logger


def run_programmatic(
    config: VibeConfig,
    prompt: str,
    max_turns: int | None = None,
    max_price: float | None = None,
    output_format: OutputFormat = OutputFormat.TEXT,
    previous_messages: list[LLMMessage] | None = None,
    auto_approve: bool = True,
) -> str | None:
    """Run in programmatic mode using VibeLangChainEngine.

    Args:
        config: Configuration for Vibe agent
        prompt: The user prompt to process
        max_turns: Maximum number of assistant turns (LLM calls) to allow
        max_price: Maximum cost in dollars before stopping
        output_format: Format for the output
        previous_messages: Optional messages from a previous session to continue
        auto_approve: Whether to automatically approve tool execution

    Returns:
        The final assistant response text, or None if no response
    """
    formatter = create_formatter(output_format)

    # Use VibeLangChainEngine directly
    engine = VibeLangChainEngine(
        config=config,
        approval_callback=None,  # Programmatic mode auto-approves all tools
    )
    logger.info("USER: %s", prompt)

    async def _async_run() -> str | None:
        # Initialize the engine
        engine.initialize()

        # Note: VibeLangChainEngine handles state internally via LangGraph checkpointing
        # Previous messages are managed at the LangGraph level, not application level
        # We skip manually loading them as the checkpoint system handles persistence

        async for event in engine.run(prompt):
            formatter.on_event(event)
            if isinstance(event, AssistantEvent) and event.stopped_by_middleware:
                raise ConversationLimitException(event.content)

        return formatter.finalize()

    return asyncio.run(_async_run())
