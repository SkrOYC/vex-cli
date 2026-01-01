from __future__ import annotations

import json

from langchain.agents.middleware.human_in_the_loop import HITLRequest, HITLResponse

from vibe.core.types import LLMChunk, LLMMessage, LLMUsage, Role, ToolCall

MOCK_DATA_ENV_VAR = "VIBE_MOCK_LLM_DATA"


def mock_llm_chunk(
    content: str = "Hello!",
    role: Role = Role.assistant,
    tool_calls: list[ToolCall] | None = None,
    name: str | None = None,
    tool_call_id: str | None = None,
    finish_reason: str | None = None,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> LLMChunk:
    message = LLMMessage(
        role=role,
        content=content,
        tool_calls=tool_calls,
        name=name,
        tool_call_id=tool_call_id,
    )
    return LLMChunk(
        message=message,
        usage=LLMUsage(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        ),
        finish_reason=finish_reason,
    )


def get_mocking_env(mock_chunks: list[LLMChunk] | None = None) -> dict[str, str]:
    if mock_chunks is None:
        mock_chunks = [mock_llm_chunk()]

    mock_data = [LLMChunk.model_dump(mock_chunk) for mock_chunk in mock_chunks]

    return {MOCK_DATA_ENV_VAR: json.dumps(mock_data)}


def create_mock_hitl_request(
    tool_name: str = "bash",
    args: dict | None = None,
    description: str = "",
    allowed_decisions: list[str] | None = None,
) -> HITLRequest:
    """Create a mock HITLRequest for testing."""
    if args is None:
        args = {"command": "test"}

    if not description:
        description = f"Execute {tool_name}"

    if allowed_decisions is None:
        allowed_decisions = ["approve", "reject"]

    return HITLRequest(
        action_requests=[
            {
                "name": tool_name,
                "args": args,
                "description": description,
            }
        ],
        review_configs=[
            {
                "action_name": tool_name,
                "allowed_decisions": allowed_decisions,
            }
        ],
    )


def create_mock_hitl_response(
    approved: bool,
    message: str | None = None,
) -> HITLResponse:
    """Create a mock HITLResponse for testing."""
    if approved:
        return HITLResponse(decisions=[{"type": "approve"}])
    else:
        return HITLResponse(
            decisions=[{"type": "reject", "message": message or "Rejected"}]
        )


def create_mock_multi_tool_response(
    approvals: list[bool],
    feedbacks: list[str | None],
) -> HITLResponse:
    """Create a mock HITLResponse for multiple tools."""
    decisions = []
    for approved, feedback in zip(approvals, feedbacks, strict=True):
        if approved:
            decisions.append({"type": "approve"})
        else:
            decisions.append({"type": "reject", "message": feedback})
    return HITLResponse(decisions=decisions)
