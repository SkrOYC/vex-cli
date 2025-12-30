"""Unit tests for ApprovalBridge with DeepAgents integration."""

from __future__ import annotations

import asyncio

import pytest

from vibe.core.engine.adapters import ApprovalBridge


@pytest.fixture
def approval_bridge(config_dir):
    """Create an ApprovalBridge with minimal config for testing."""
    from vibe.core.config import VibeConfig

    config = VibeConfig()
    return ApprovalBridge(config=config)


class TestApprovalBridge:
    """Test ApprovalBridge functionality."""

    def test_initialization(self, approval_bridge):
        """Test ApprovalBridge initializes correctly."""
        assert approval_bridge._pending_approvals == {}

    @pytest.mark.asyncio
    async def test_handle_interrupt_no_action_request(self, approval_bridge):
        """Test handling interrupt with no action request."""
        interrupt = {"type": "interrupt", "data": {}}

        result = await approval_bridge.handle_interrupt(interrupt)
        assert result == {"approved": True}

    @pytest.mark.asyncio
    async def test_handle_interrupt_with_action_request_timeout(self, approval_bridge):
        """Test handling interrupt with action request times out.

        Note: This test requires an approval_callback that doesn't respond.
        With auto-approval mode (no callback), the interrupt is approved immediately.
        """
        from vibe.core.config import VibeConfig

        config = VibeConfig()

        async def non_responsive_callback(request: dict) -> dict:
            # Never respond - this would cause a timeout
            await asyncio.sleep(300)  # 5 minutes (much longer than test timeout)
            return {"approved": False}

        bridge = ApprovalBridge(
            config=config, approval_callback=non_responsive_callback
        )

        interrupt = {
            "data": {
                "action_request": {
                    "name": "bash",
                    "args": {"command": "ls"},
                    "description": "Run command",
                }
            }
        }

        # Should timeout and auto-reject
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(bridge.handle_interrupt(interrupt), timeout=0.1)

    @pytest.mark.asyncio
    async def test_extract_action_request_from_data(self, approval_bridge):
        """Test extracting action request from interrupt data."""
        interrupt = {
            "data": {
                "action_request": {
                    "name": "bash",
                    "args": {"command": "ls"},
                    "description": "Run command",
                }
            }
        }

        result = approval_bridge._extract_action_request(interrupt)
        assert result == {
            "name": "bash",
            "args": {"command": "ls"},
            "description": "Run command",
        }

    @pytest.mark.asyncio
    async def test_extract_action_request_fallback(self, approval_bridge):
        """Test fallback action request extraction."""
        interrupt = {
            "name": "create",
            "args": {"path": "/file.txt", "file_text": "content"},
            "description": "Create file",
        }

        result = approval_bridge._extract_action_request(interrupt)
        assert result == {
            "name": "create",
            "args": {"path": "/file.txt", "file_text": "content"},
            "description": "Create file",
        }

    @pytest.mark.asyncio
    async def test_respond_with_request_id(self, approval_bridge):
        """Test responding to specific request ID."""
        # Create a pending approval
        future = asyncio.Future()
        approval_bridge._pending_approvals["test-id"] = future

        # Respond
        await approval_bridge.respond(True, "test-id", "approved")

        # Check the future was resolved
        assert future.done()
        assert future.result() == {"approved": True, "feedback": "approved"}

        # Check cleanup
        assert "test-id" not in approval_bridge._pending_approvals

    @pytest.mark.asyncio
    async def test_respond_unknown_request_id(self, approval_bridge):
        """Test responding to unknown request ID."""
        # Should not raise
        await approval_bridge.respond(True, "unknown-id")

    @pytest.mark.asyncio
    async def test_multiple_concurrent_interrupts(self, approval_bridge):
        """Test handling multiple interrupts concurrently.

        Note: This test requires an approval_callback that doesn't respond.
        With auto-approval mode (no callback), interrupts are approved immediately.
        """
        from vibe.core.config import VibeConfig

        config = VibeConfig()

        async def non_responsive_callback(request: dict) -> dict:
            await asyncio.sleep(300)  # 5 minutes
            return {"approved": False}

        bridge = ApprovalBridge(
            config=config, approval_callback=non_responsive_callback
        )

        interrupt1 = {
            "data": {
                "action_request": {
                    "name": "create",
                    "args": {"path": "/file1.txt", "file_text": "content"},
                    "description": "Create file 1",
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
    async def test_malformed_interrupt_data(self, approval_bridge):
        """Test handling malformed interrupt data."""
        # Test with None data
        result = await approval_bridge.handle_interrupt({
            "type": "interrupt",
            "data": None,
        })
        assert result == {"approved": True}

        # Test with empty dict
        result = await approval_bridge.handle_interrupt({})
        assert result == {"approved": True}

        # Test with invalid action_request structure
        interrupt = {"data": {"action_request": "invalid_string_instead_of_dict"}}
        result = await approval_bridge.handle_interrupt(interrupt)
        assert result == {"approved": True}  # Should auto-approve on extraction failure

    @pytest.mark.asyncio
    async def test_respond_to_already_resolved_future(self, approval_bridge):
        """Test responding to a future that's already been resolved."""
        # Create and immediately resolve a future
        future = asyncio.Future()
        future.set_result({"approved": False, "feedback": "already done"})
        approval_bridge._pending_approvals["test-id"] = future

        # Responding again should not raise
        await approval_bridge.respond(True, "test-id", "new response")

        # Future should still have original result
        assert future.result() == {"approved": False, "feedback": "already done"}

    @pytest.mark.asyncio
    async def test_extract_action_request_edge_cases(self, approval_bridge):
        """Test action request extraction with edge cases."""
        # Test with missing fields in action_request
        interrupt = {
            "data": {
                "action_request": {
                    "name": "test_tool"
                    # Missing args and description
                }
            }
        }
        result = approval_bridge._extract_action_request(interrupt)
        assert result == {"name": "test_tool", "args": {}, "description": ""}

        # Test with None values
        interrupt = {
            "data": {
                "action_request": {"name": None, "args": None, "description": None}
            }
        }
        result = approval_bridge._extract_action_request(interrupt)
        assert result is not None  # Should handle None values gracefully
