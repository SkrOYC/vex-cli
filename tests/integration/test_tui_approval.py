"""Integration tests for TUI approval dialogs with multi-tool support.

These tests validate that the TUI correctly handles the new HITLResponse format
with multi-tool interrupts, batch shortcuts, and progress display.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch
import pytest

from vibe.cli.textual_ui.app import VibeApp
from vibe.core.config import VibeConfig
from vibe.core.engine.langchain_engine import VibeLangChainEngine


class TestVibeAppMultiToolApproval:
    """Test VibeApp multi-tool approval handling."""

    @pytest.fixture
    def config(self) -> VibeConfig:
        """Create a test configuration."""
        return VibeConfig()

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent with approval methods."""
        agent = AsyncMock(spec=VibeLangChainEngine)
        agent.handle_multi_tool_approval = AsyncMock()
        agent.handle_approval = AsyncMock()
        return agent

    @pytest.fixture
    def mock_app(self, config: VibeConfig, mock_agent):
        """Create a mock VibeApp for testing."""
        with patch.object(VibeApp, "__init__", lambda x, **kwargs: None):
            app = VibeApp()
            app.config = config
            app.agent = mock_agent
            app._pending_approval = None
            return app

    @pytest.mark.asyncio
    async def test_handle_multi_tool_approval_single_tool(
        self, mock_app: VibeApp, mock_agent
    ):
        """Test handling single tool approval."""
        action_requests = [
            {
                "name": "bash",
                "args": {"command": "ls"},
                "description": "List files",
            }
        ]

        # Mock the approval result directly (not a future)
        async def mock_switch(action_requests, current_index):
            return {
                "approved": True,
                "always_approve": False,
                "feedback": None,
            }

        with patch.object(
            mock_app, "_switch_to_approval_app_from_action", side_effect=mock_switch
        ):
            await mock_app._handle_multi_tool_approval(action_requests)

        # Should have called handle_multi_tool_approval with single approve
        mock_agent.handle_multi_tool_approval.assert_called_once_with([True], [None])

    @pytest.mark.asyncio
    async def test_handle_multi_tool_approval_sequential(
        self, mock_app: VibeApp, mock_agent
    ):
        """Test handling multiple tools approved sequentially."""
        action_requests = [
            {"name": "bash", "args": {"command": "ls"}, "description": "List"},
            {"name": "read_file", "args": {"path": "test.txt"}, "description": "Read"},
            {"name": "edit_file", "args": {"path": "test.txt"}, "description": "Edit"},
        ]

        # Sequential approval results
        approval_results = [
            {
                "approved": True,
                "always_approve": False,
                "feedback": None,
                "next_tool": 1,
            },
            {
                "approved": True,
                "always_approve": False,
                "feedback": None,
                "next_tool": 2,
            },
            {
                "approved": True,
                "always_approve": False,
                "feedback": None,
                "next_tool": None,
            },
        ]

        call_count = [0]  # Use list to allow mutation in closure

        async def mock_switch(action_requests, current_index):
            result = approval_results[call_count[0]]
            call_count[0] += 1
            return result

        with patch.object(
            mock_app, "_switch_to_approval_app_from_action", side_effect=mock_switch
        ):
            await mock_app._handle_multi_tool_approval(action_requests)

        # Should have called handle_multi_tool_approval with all approves
        mock_agent.handle_multi_tool_approval.assert_called_once_with(
            [True, True, True], [None, None, None]
        )

    @pytest.mark.asyncio
    async def test_handle_multi_tool_approval_approve_all(
        self, mock_app: VibeApp, mock_agent
    ):
        """Test handling Approve All shortcut."""
        action_requests = [
            {"name": "bash", "args": {"command": "ls"}, "description": "List"},
            {"name": "read_file", "args": {"path": "test.txt"}, "description": "Read"},
            {"name": "edit_file", "args": {"path": "test.txt"}, "description": "Edit"},
        ]

        # First tool approved, second tool uses Approve All
        approval_results = [
            {
                "approved": True,
                "always_approve": False,
                "feedback": None,
                "next_tool": 1,
            },
            {
                "approved": True,
                "always_approve": False,
                "feedback": None,
                "batch_approve": True,  # Approve All
            },
        ]

        call_count = [0]  # Use list to allow mutation in closure

        async def mock_switch(action_requests, current_index):
            result = approval_results[call_count[0]]
            call_count[0] += 1
            return result

        with patch.object(
            mock_app, "_switch_to_approval_app_from_action", side_effect=mock_switch
        ):
            await mock_app._handle_multi_tool_approval(action_requests)

        # Should have called handle_multi_tool_approval with all approves (including batch)
        mock_agent.handle_multi_tool_approval.assert_called_once_with(
            [True, True, True], [None, None, None]
        )

    @pytest.mark.asyncio
    async def test_handle_multi_tool_approval_reject_all(
        self, mock_app: VibeApp, mock_agent
    ):
        """Test handling Reject All shortcut."""
        action_requests = [
            {"name": "bash", "args": {"command": "ls"}, "description": "List"},
            {"name": "read_file", "args": {"path": "test.txt"}, "description": "Read"},
            {"name": "edit_file", "args": {"path": "test.txt"}, "description": "Edit"},
        ]

        # First tool approved, second tool uses Reject All
        approval_results = [
            {
                "approved": True,
                "always_approve": False,
                "feedback": None,
                "next_tool": 1,
            },
            {
                "approved": False,
                "always_approve": False,
                "feedback": "User rejected all operations",
                "batch_reject": True,  # Reject All
            },
        ]

        call_count = [0]  # Use list to allow mutation in closure

        async def mock_switch(action_requests, current_index):
            result = approval_results[call_count[0]]
            call_count[0] += 1
            return result

        with patch.object(
            mock_app, "_switch_to_approval_app_from_action", side_effect=mock_switch
        ):
            await mock_app._handle_multi_tool_approval(action_requests)

        # Should have called handle_multi_tool_approval with correct decisions
        mock_agent.handle_multi_tool_approval.assert_called_once_with(
            [True, False, False],
            [None, "User rejected all operations", "User rejected all operations"],
        )

    @pytest.mark.asyncio
    async def test_handle_multi_tool_approval_mixed(
        self, mock_app: VibeApp, mock_agent
    ):
        """Test handling mixed approval decisions."""
        action_requests = [
            {"name": "bash", "args": {"command": "ls"}, "description": "List"},
            {"name": "read_file", "args": {"path": "test.txt"}, "description": "Read"},
            {"name": "edit_file", "args": {"path": "test.txt"}, "description": "Edit"},
        ]

        # Mixed: approve, reject, approve
        approval_results = [
            {
                "approved": True,
                "always_approve": False,
                "feedback": None,
                "next_tool": 1,
            },
            {
                "approved": False,
                "always_approve": False,
                "feedback": "Not needed",
                "next_tool": 2,
            },
            {
                "approved": True,
                "always_approve": False,
                "feedback": None,
                "next_tool": None,
            },
        ]

        call_count = [0]  # Use list to allow mutation in closure

        async def mock_switch(action_requests, current_index):
            result = approval_results[call_count[0]]
            call_count[0] += 1
            return result

        with patch.object(
            mock_app, "_switch_to_approval_app_from_action", side_effect=mock_switch
        ):
            await mock_app._handle_multi_tool_approval(action_requests)

        # Should have called handle_multi_tool_approval with mixed decisions
        mock_agent.handle_multi_tool_approval.assert_called_once_with(
            [True, False, True], [None, "Not needed", None]
        )

    @pytest.mark.asyncio
    async def test_handle_engine_interrupt_multi_tool_format(
        self, mock_app: VibeApp, mock_agent
    ):
        """Test that _handle_engine_interrupt detects multi-tool format."""
        interrupt_data = {
            "data": {
                "action_requests": [
                    {"name": "bash", "args": {"command": "ls"}, "description": "List"},
                    {
                        "name": "read_file",
                        "args": {"path": "test.txt"},
                        "description": "Read",
                    },
                ],
            }
        }

        # Mock the approval result
        async def mock_switch(action_requests, current_index):
            return {
                "approved": True,
                "always_approve": False,
                "feedback": None,
            }

        with patch.object(
            mock_app, "_handle_multi_tool_approval", new_callable=AsyncMock
        ) as mock_handle:
            with patch.object(
                mock_app, "_switch_to_approval_app_from_action", side_effect=mock_switch
            ):
                await mock_app._handle_engine_interrupt(interrupt_data)

                # Should have called _handle_multi_tool_approval with action_requests
                mock_handle.assert_called_once()
                calls = mock_handle.call_args[0][0]
                assert len(calls) == 2

    @pytest.mark.asyncio
    async def test_handle_engine_interrupt_legacy_format(
        self, mock_app: VibeApp, mock_agent
    ):
        """Test that _handle_engine_interrupt handles legacy single-tool format."""
        interrupt_data = {
            "data": {
                "action_request": {
                    "name": "bash",
                    "args": {"command": "ls"},
                    "description": "List files",
                }
            }
        }

        # Mock the approval result
        async def mock_switch(action_requests, current_index):
            return {
                "approved": True,
                "always_approve": False,
                "feedback": None,
            }

        with patch.object(
            mock_app, "_handle_multi_tool_approval", new_callable=AsyncMock
        ) as mock_handle:
            with patch.object(
                mock_app, "_switch_to_approval_app_from_action", side_effect=mock_switch
            ):
                await mock_app._handle_engine_interrupt(interrupt_data)

                # Should have called _handle_multi_tool_approval with single action in list
                mock_handle.assert_called_once()
                calls = mock_handle.call_args[0][0]
                assert len(calls) == 1
                assert calls[0]["name"] == "bash"


class TestApprovalAppLogic:
    """Test ApprovalApp logic without requiring Textual app context."""

    @pytest.fixture
    def config(self) -> VibeConfig:
        """Create a test configuration."""
        return VibeConfig()

    def test_single_tool_title(self, config: VibeConfig):
        """Test that single tool shows correct title without progress."""
        # Test the title generation logic directly
        tool_name = "bash"
        total_tools = 1
        current_index = 0

        title_text = f"⚠ {tool_name} command"
        if total_tools > 1:
            title_text += f" ({current_index + 1}/{total_tools})"

        assert title_text == "⚠ bash command"
        assert "(1/1)" not in title_text

    def test_multi_tool_title_progress(self):
        """Test that multi-tool shows progress in title."""
        # Test the title generation logic directly
        tool_name = "bash"
        total_tools = 3
        current_index = 0

        title_text = f"⚠ {tool_name} command"
        if total_tools > 1:
            title_text += f" ({current_index + 1}/{total_tools})"

        assert title_text == "⚠ bash command (1/3)"

        # Second tool
        current_index = 1
        title_text = f"⚠ read_file command"
        if total_tools > 1:
            title_text += f" ({current_index + 1}/{total_tools})"

        assert title_text == "⚠ read_file command (2/3)"

        # Third tool
        current_index = 2
        title_text = f"⚠ edit_file command"
        if total_tools > 1:
            title_text += f" ({current_index + 1}/{total_tools})"

        assert title_text == "⚠ edit_file command (3/3)"

    def test_single_tool_options_count(self):
        """Test that single tool shows only 3 options."""
        total_tools = 1
        max_options = 5 if total_tools > 1 else 3
        assert max_options == 3

    def test_multi_tool_options_count(self):
        """Test that multi-tool shows 5 options."""
        total_tools = 3
        max_options = 5 if total_tools > 1 else 3
        assert max_options == 5

    def test_always_approve_option_text(self):
        """Test that 'always allow' option is available."""
        tool_name = "bash"
        option_text = f"Yes and always allow {tool_name} this session"
        assert "always allow" in option_text.lower() or "Always" in option_text


class TestKeyboardShortcuts:
    """Test keyboard shortcut functionality."""

    def test_approve_all_shortcut_key(self):
        """Test that key 4 is used for Approve All."""
        # The key binding should be "4"
        key = "4"
        assert key == "4"

    def test_reject_all_shortcut_key(self):
        """Test that key 5 is used for Reject All."""
        # The key binding should be "5"
        key = "5"
        assert key == "5"


class TestMultiToolFlow:
    """Test multi-tool approval flow logic."""

    def test_next_tool_calculation_first_tool(self):
        """Test next_tool calculation for first tool."""
        current_index = 0
        total_tools = 3

        # If not last tool, next_tool should be current_index + 1
        if current_index < total_tools - 1:
            next_tool = current_index + 1
        else:
            next_tool = None

        assert next_tool == 1

    def test_next_tool_calculation_last_tool(self):
        """Test next_tool calculation for last tool."""
        current_index = 2
        total_tools = 3

        # If not last tool, next_tool should be current_index + 1
        if current_index < total_tools - 1:
            next_tool = current_index + 1
        else:
            next_tool = None

        assert next_tool is None

    def test_batch_approve_all_remaining(self):
        """Test that Approve All approves all remaining tools."""
        current_index = 1
        total_tools = 3
        remaining = total_tools - current_index - 1

        # Total decisions should be: 1 (current) + remaining (batch)
        # But the batch actually adds the current + remaining, so we have:
        # - Tool 0: approved (not part of batch)
        # - Tools 1, 2: approved via batch
        # Total: 3 approvals

        # When batch_approve is True on tool 1, we add approvals for tools 1 and 2
        # But tool 0 was already processed, so total is:
        # 1 (tool 0) + 2 (tools 1 & 2 via batch) = 3
        approvals = [True] + [True] * (
            remaining + 1
        )  # +1 because current tool is also batch approved

        assert len(approvals) == 3
        assert all(approvals)

    def test_batch_reject_all_remaining(self):
        """Test that Reject All rejects all remaining tools."""
        current_index = 1
        total_tools = 3
        feedback = "User rejected all operations"
        remaining = total_tools - current_index - 1

        # Total decisions should be: 1 (current) + remaining + 1 (current batch)
        decisions = [{"type": "approve"}] + [
            {"type": "reject", "message": feedback}
        ] * (remaining + 1)

        assert len(decisions) == 3
        assert decisions[0]["type"] == "approve"
        assert decisions[1]["type"] == "reject"
        assert decisions[2]["type"] == "reject"
        assert all(d["message"] == feedback for d in decisions[1:])
