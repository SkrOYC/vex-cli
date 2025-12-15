"""Tests for ApprovalBridge."""

import asyncio

import pytest
from vibe.core.engine.adapters import ApprovalBridge


class TestApprovalBridge:
    @pytest.mark.asyncio
    async def test_handle_interrupt(self):
        """Test handling of interrupts."""
        bridge = ApprovalBridge()
        interrupt = {"type": "tool_approval", "tool": "bash"}

        # For skeleton implementation, handle_interrupt should return immediately
        # In real implementation, it would wait for approval
        result = await bridge.handle_interrupt(interrupt)
        assert result == {"approved": True}  # Placeholder for now

    @pytest.mark.asyncio
    async def test_respond_without_pending(self):
        """Test responding when no pending approval."""
        bridge = ApprovalBridge()
        # Should not raise
        await bridge.respond(True)