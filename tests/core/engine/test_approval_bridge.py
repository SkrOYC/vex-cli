"""Tests for ApprovalBridge."""

import pytest
from vibe.core.engine.adapters import ApprovalBridge


class TestApprovalBridge:
    async def test_handle_interrupt(self):
        """Test handling of interrupts."""
        bridge = ApprovalBridge()
        interrupt = {"type": "tool_approval", "tool": "bash"}
        result = await bridge.handle_interrupt(interrupt)
        assert result == {"approved": True}

    async def test_respond_without_pending(self):
        """Test responding when no pending approval."""
        bridge = ApprovalBridge()
        # Should not raise
        await bridge.respond(True)