"""Unit tests for ApprovalBridge with DeepAgents integration."""

import pytest
import asyncio
from unittest.mock import AsyncMock
from vibe.core.engine.adapters import ApprovalBridge


class TestApprovalBridge:
    """Test ApprovalBridge functionality."""

    def test_initialization(self):
        """Test ApprovalBridge initializes correctly."""
        bridge = ApprovalBridge()
        assert bridge._pending_approvals == {}

    @pytest.mark.asyncio
    async def test_handle_interrupt_no_action_request(self):
        """Test handling interrupt with no action request."""
        bridge = ApprovalBridge()
        interrupt = {"type": "interrupt", "data": {}}

        result = await bridge.handle_interrupt(interrupt)
        assert result == {"approved": True}

    @pytest.mark.asyncio
    async def test_handle_interrupt_with_action_request_timeout(self):
        """Test handling interrupt with action request times out."""
        bridge = ApprovalBridge()
        interrupt = {
            "type": "interrupt",
            "data": {
                "action_request": {
                    "name": "write_file",
                    "args": {"path": "/test.txt", "content": "hello"},
                    "description": "Write to file",
                }
            },
        }

        # Should timeout and auto-reject
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(bridge.handle_interrupt(interrupt), timeout=0.1)

    @pytest.mark.asyncio
    async def test_extract_action_request_from_data(self):
        """Test extracting action request from interrupt data."""
        bridge = ApprovalBridge()
        interrupt = {
            "data": {
                "action_request": {
                    "name": "bash",
                    "args": {"command": "ls"},
                    "description": "Run command",
                }
            }
        }

        result = bridge._extract_action_request(interrupt)
        assert result == {
            "name": "bash",
            "args": {"command": "ls"},
            "description": "Run command",
        }

    @pytest.mark.asyncio
    async def test_extract_action_request_fallback(self):
        """Test fallback action request extraction."""
        bridge = ApprovalBridge()
        interrupt = {
            "name": "write_file",
            "args": {"path": "/file.txt"},
            "description": "Write file",
        }

        result = bridge._extract_action_request(interrupt)
        assert result == {
            "name": "write_file",
            "args": {"path": "/file.txt"},
            "description": "Write file",
        }

    @pytest.mark.asyncio
    async def test_respond_with_request_id(self):
        """Test responding to specific request ID."""
        bridge = ApprovalBridge()

        # Create a pending approval
        future = asyncio.Future()
        bridge._pending_approvals["test-id"] = future

        # Respond
        await bridge.respond(True, "test-id", "approved")

        # Check the future was resolved
        assert future.done()
        assert future.result() == {"approved": True, "feedback": "approved"}

        # Check cleanup
        assert "test-id" not in bridge._pending_approvals

    @pytest.mark.asyncio
    async def test_respond_unknown_request_id(self):
        """Test responding to unknown request ID."""
        bridge = ApprovalBridge()

        # Should not raise
        await bridge.respond(True, "unknown-id")

    @pytest.mark.asyncio
    async def test_multiple_concurrent_interrupts(self):
        """Test handling multiple interrupts concurrently."""
        bridge = ApprovalBridge()

        interrupt1 = {
            "data": {
                "action_request": {
                    "name": "write_file",
                    "args": {"path": "/file1.txt"},
                    "description": "Write file 1",
                }
            }
        }

        interrupt2 = {
            "data": {
                "action_request": {
                    "name": "bash",
                    "args": {"command": "ls"},
                    "description": "List files",
                }
            }
        }

        # Start both interrupts (they will timeout)
        task1 = asyncio.create_task(bridge.handle_interrupt(interrupt1))
        task2 = asyncio.create_task(bridge.handle_interrupt(interrupt2))

        # Both should timeout and auto-reject
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(task1, timeout=0.1)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(task2, timeout=0.1)

        # Check that multiple request IDs were created
        assert len(bridge._pending_approvals) == 0  # All cleaned up after timeout

    @pytest.mark.asyncio
    async def test_malformed_interrupt_data(self):
        """Test handling malformed interrupt data."""
        bridge = ApprovalBridge()

        # Test with None data
        result = await bridge.handle_interrupt({"type": "interrupt", "data": None})
        assert result == {"approved": True}

        # Test with empty dict
        result = await bridge.handle_interrupt({})
        assert result == {"approved": True}

        # Test with invalid action_request structure
        interrupt = {"data": {"action_request": "invalid_string_instead_of_dict"}}
        result = await bridge.handle_interrupt(interrupt)
        assert result == {"approved": True}  # Should auto-approve on extraction failure

    @pytest.mark.asyncio
    async def test_respond_to_already_resolved_future(self):
        """Test responding to a future that's already been resolved."""
        bridge = ApprovalBridge()

        # Create and immediately resolve a future
        future = asyncio.Future()
        future.set_result({"approved": False, "feedback": "already done"})
        bridge._pending_approvals["test-id"] = future

        # Responding again should not raise
        await bridge.respond(True, "test-id", "new response")

        # Future should still have original result
        assert future.result() == {"approved": False, "feedback": "already done"}

    @pytest.mark.asyncio
    async def test_extract_action_request_edge_cases(self):
        """Test action request extraction with edge cases."""
        bridge = ApprovalBridge()

        # Test with missing fields in action_request
        interrupt = {
            "data": {
                "action_request": {
                    "name": "test_tool"
                    # Missing args and description
                }
            }
        }
        result = bridge._extract_action_request(interrupt)
        assert result == {"name": "test_tool", "args": {}, "description": ""}

        # Test with None values
        interrupt = {
            "data": {
                "action_request": {"name": None, "args": None, "description": None}
            }
        }
        result = bridge._extract_action_request(interrupt)
        assert result is not None  # Should handle None values gracefully
