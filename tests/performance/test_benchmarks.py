"""Performance benchmark tests for Agent vs VibeEngine."""

import asyncio
import time
import os
import psutil
from typing import AsyncGenerator

import pytest

from vibe.core.agent import Agent
from vibe.core.engine import VibeEngine
from vibe.core.types import BaseEvent, AssistantEvent
from tests.stubs.fake_backend import FakeBackend
from tests.mock.utils import mock_llm_chunk


@pytest.fixture
def mock_agent(config):
    """Create a legacy Agent with mock backend."""
    backend = FakeBackend([mock_llm_chunk(content="Mock response")])
    agent = Agent(config=config, auto_approve=True, backend=backend)
    return agent


@pytest.fixture
def mock_vibe_engine(config):
    """Create a VibeEngine with mock backend."""
    # For VibeEngine, we need to patch the backend creation
    # For now, we'll create a basic VibeEngine and note that full mocking needs DeepAgents mocking
    engine = VibeEngine(config=config)
    return engine


def measure_memory_usage(func):
    """Decorator to measure memory usage of a function."""

    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        return result, {
            "initial_memory": initial_memory,
            "final_memory": final_memory,
            "memory_growth": final_memory - initial_memory,
            "duration": end_time - start_time,
        }

    return wrapper


def test_simple_conversation_latency(benchmark, mock_agent, short_conversation):
    """Measure response time for simple conversation (3-4 messages, no tools)."""

    def agent_run():
        async def _run():
            events = []
            async for event in mock_agent.act(short_conversation["input"]):
                events.append(event)
            return events

        return asyncio.run(_run())

    # Benchmark agent as baseline
    benchmark.pedantic(agent_run, iterations=5, rounds=3)


def test_multi_turn_latency(benchmark, mock_agent, long_conversation):
    """Measure response time for 10+ message conversation."""

    def agent_run():
        async def _run():
            events = []
            for message in long_conversation["messages"]:
                async for event in mock_agent.act(message):
                    events.append(event)
            return events

        return asyncio.run(_run())

    benchmark.pedantic(agent_run, iterations=3, rounds=2)


def test_tool_execution_latency(benchmark, mock_agent, tool_scenarios):
    """Measure file operation latency."""

    def agent_run():
        async def _run():
            events = []
            async for event in mock_agent.act(tool_scenarios["file_read_scenario"]):
                events.append(event)
            return events

        return asyncio.run(_run())

    benchmark.pedantic(agent_run, iterations=5, rounds=3)


def test_memory_growth(benchmark, mock_agent, long_conversation):
    """Measure memory growth over conversation."""

    @measure_memory_usage
    def agent_memory_test():
        async def _run():
            events = []
            for message in long_conversation["messages"]:
                async for event in mock_agent.act(message):
                    events.append(event)
            return events

        return asyncio.run(_run())

    benchmark.pedantic(agent_memory_test, iterations=5, rounds=3)


def test_streaming_throughput(benchmark, mock_agent, short_conversation):
    """Measure words/second during streaming."""

    def agent_streaming():
        async def _run():
            words = 0
            async for event in mock_agent.act(short_conversation["input"]):
                if isinstance(event, AssistantEvent) and event.content:
                    words += len(event.content.split())
            return words

        return asyncio.run(_run())

    benchmark.pedantic(agent_streaming, iterations=5, rounds=3)
