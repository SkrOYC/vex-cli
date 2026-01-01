"""Unit tests for HITLResponse format and construction."""

from __future__ import annotations

from langchain.agents.middleware.human_in_the_loop import HITLResponse

from tests.mock.utils import (
    create_mock_hitl_response,
    create_mock_multi_tool_response,
)


class TestHITLResponseFormat:
    """Test HITLResponse structure and format."""

    def test_approve_decision_format(self):
        """Test approve decision has correct format."""
        response = create_mock_hitl_response(approved=True)

        assert len(response["decisions"]) == 1
        assert response["decisions"][0]["type"] == "approve"
        assert "message" not in response["decisions"][0]

    def test_reject_decision_format(self):
        """Test reject decision has correct format."""
        response = create_mock_hitl_response(approved=False, message="Unsafe")

        assert len(response["decisions"]) == 1
        assert response["decisions"][0]["type"] == "reject"
        assert response["decisions"][0]["message"] == "Unsafe"

    def test_reject_with_default_message(self):
        """Test reject with default message."""
        response = create_mock_hitl_response(approved=False)

        assert response["decisions"][0]["type"] == "reject"
        assert response["decisions"][0]["message"] == "Rejected"

    def test_multi_decision_format(self):
        """Test multiple decisions in single HITLResponse."""
        response = create_mock_multi_tool_response(
            approvals=[True, False, True],
            feedbacks=[None, "Skip", None],
        )

        assert len(response["decisions"]) == 3
        assert response["decisions"][0]["type"] == "approve"
        assert response["decisions"][1]["type"] == "reject"
        assert response["decisions"][1]["message"] == "Skip"
        assert response["decisions"][2]["type"] == "approve"

    def test_all_approve_format(self):
        """Test all approve decisions."""
        response = create_mock_multi_tool_response(
            approvals=[True, True, True],
            feedbacks=[None, None, None],
        )

        assert len(response["decisions"]) == 3
        assert all(d["type"] == "approve" for d in response["decisions"])
        assert all("message" not in d for d in response["decisions"])

    def test_all_reject_format(self):
        """Test all reject decisions."""
        response = create_mock_multi_tool_response(
            approvals=[False, False, False],
            feedbacks=["Bad", "Worse", "Worst"],
        )

        assert len(response["decisions"]) == 3
        assert all(d["type"] == "reject" for d in response["decisions"])
        assert all("message" in d for d in response["decisions"])

    def test_mixed_decisions(self):
        """Test mixed approve/reject decisions."""
        response = create_mock_multi_tool_response(
            approvals=[True, False, True, False],
            feedbacks=[None, "Nope", None, "Nah"],
        )

        assert len(response["decisions"]) == 4
        assert response["decisions"][0]["type"] == "approve"
        assert response["decisions"][1]["type"] == "reject"
        assert response["decisions"][2]["type"] == "approve"
        assert response["decisions"][3]["type"] == "reject"

    def test_single_decision_multi_tool(self):
        """Test multi-tool helper with single decision."""
        response = create_mock_multi_tool_response(
            approvals=[True],
            feedbacks=[None],
        )

        assert len(response["decisions"]) == 1
        assert response["decisions"][0]["type"] == "approve"

    def test_empty_decisions_list(self):
        """Test empty decisions list."""
        response = create_mock_multi_tool_response(
            approvals=[],
            feedbacks=[],
        )

        assert len(response["decisions"]) == 0

    def test_reject_with_none_message(self):
        """Test reject decision with None message becomes default."""
        response = create_mock_multi_tool_response(
            approvals=[False, True],
            feedbacks=[None, None],
        )

        assert len(response["decisions"]) == 2
        assert response["decisions"][0]["type"] == "reject"
        assert response["decisions"][0]["message"] is None
        assert response["decisions"][1]["type"] == "approve"

    def test_large_multi_tool_response(self):
        """Test creating response with many decisions."""
        approvals = [True, False] * 5
        feedbacks = [None, "Unsafe"] * 5

        response = create_mock_multi_tool_response(approvals, feedbacks)

        assert len(response["decisions"]) == 10
        for i, decision in enumerate(response["decisions"]):
            if i % 2 == 0:
                assert decision["type"] == "approve"
                assert "message" not in decision
            else:
                assert decision["type"] == "reject"
                assert decision["message"] == "Unsafe"

    def test_response_structure_matches_langchain_format(self):
        """Test that response structure matches LangChain HITLResponse format."""
        response = HITLResponse(
            decisions=[
                {"type": "approve"},
                {"type": "reject", "message": "Feedback"},
            ],
        )

        assert "decisions" in response
        assert len(response["decisions"]) == 2
        assert response["decisions"][0]["type"] == "approve"
        assert response["decisions"][1]["type"] == "reject"
        assert response["decisions"][1]["message"] == "Feedback"

    def test_response_with_all_types_of_decisions(self):
        """Test response containing all decision types."""
        response = create_mock_multi_tool_response(
            approvals=[True, False, True, False],
            feedbacks=[None, "Error1", None, "Error2"],
        )

        assert len(response["decisions"]) == 4

        # Verify approve decisions
        approve_decisions = [d for d in response["decisions"] if d["type"] == "approve"]
        assert len(approve_decisions) == 2

        # Verify reject decisions
        reject_decisions = [d for d in response["decisions"] if d["type"] == "reject"]
        assert len(reject_decisions) == 2
        assert all("message" in d for d in reject_decisions)
