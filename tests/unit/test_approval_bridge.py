"""Unit tests for ApprovalBridge with DeepAgents integration."""

import pytest
import asyncio
from vibe.core.engine.adapters import ApprovalBridge


class TestApprovalBridge:
    """Test ApprovalBridge functionality."""
    
    def test_initialization(self):
        """Test ApprovalBridge initializes correctly."""
        bridge = ApprovalBridge()
        assert bridge._pending_approval is None
    
    @pytest.mark.asyncio
    async def test_handle_interrupt(self):
        """Test handling of interrupts."""
        bridge = ApprovalBridge()
        interrupt = {"type": "tool_approval", "tool": "bash"}

        # For skeleton implementation, handle_interrupt should return immediately
        result = await bridge.handle_interrupt(interrupt)
        assert result == {"approved": True}  # Placeholder for now
    
    @pytest.mark.asyncio
    async def test_respond_without_pending(self):
        """Test responding when no pending approval."""
        bridge = ApprovalBridge()
        # Should not raise
        await bridge.respond(True)
        
    @pytest.mark.asyncio
    async def test_respond_with_pending_approval(self):
        """Test responding when there is a pending approval."""
        bridge = ApprovalBridge()
        
        # Create a future to simulate pending approval
        bridge._pending_approval = asyncio.Future()
        
        # Respond to the pending approval
        await bridge.respond(True, "test feedback")
        
        # The future should be completed
        assert bridge._pending_approval is None