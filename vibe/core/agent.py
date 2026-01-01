"""Legacy Agent exception classes.

The Agent class has been removed as part of the DeepAgents migration.
These exception classes are kept for backward compatibility with tests.
See tests/migration/__init__.py for migration guidance.
"""

from __future__ import annotations


class AgentError(Exception):
    """Base exception for Agent errors."""


class AgentStateError(AgentError):
    """Raised when agent is in an invalid state."""


class LLMResponseError(AgentError):
    """Raised when LLM response is malformed or missing expected data."""



