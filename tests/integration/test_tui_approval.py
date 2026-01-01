"""Integration tests for TUI approval dialogs with multi-tool support.

These tests validate that the TUI correctly handles the new HITLResponse format
with multi-tool interrupts, batch shortcuts, and progress display.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from vibe.cli.textual_ui.app import VibeApp
from vibe.cli.textual_ui.widgets.approval_app import ApprovalApp
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

    @pytest.mark.skip(
        reason="Legacy format handling removed - test no longer applicable"
    )
    @pytest.mark.asyncio
    async def test_handle_engine_interrupt_legacy_format(
        self, mock_app: VibeApp, mock_agent
    ):
        """Test that _handle_engine_interrupt handles legacy single-tool format."""
        pass


class TestKeyboardShortcuts:
    """Test keyboard shortcut functionality."""

    def test_approve_all_shortcut_key(self):
        """Test that key 4 is used for Approve All."""
        binding = next((b for b in ApprovalApp.BINDINGS if b.key == "4"), None)
        assert binding is not None, "Binding for key '4' not found"
        assert binding.action == "select_4"
        assert binding.description == "Approve All"

    def test_reject_all_shortcut_key(self):
        """Test that key 5 is used for Reject All."""
        binding = next((b for b in ApprovalApp.BINDINGS if b.key == "5"), None)
        assert binding is not None, "Binding for key '5' not found"
        assert binding.action == "select_5"
        assert binding.description == "Reject All"
