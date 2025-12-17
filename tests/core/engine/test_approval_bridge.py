"""Tests for ApprovalBridge."""

import asyncio

import pytest
from vibe.core.config import VibeConfig
from vibe.core.engine.adapters import ApprovalBridge
from vibe.core.tools.base import BaseToolConfig, ToolPermission


class TestApprovalBridge:
    @pytest.mark.asyncio
    async def test_handle_interrupt_with_callback(self):
        """Test handling of interrupts with callback."""
        config = VibeConfig()
        
        # Mock approval callback
        callback_results = []
        
        async def mock_callback(action_request):
            result = {"approved": True, "always_approve": False, "feedback": None}
            callback_results.append(result)
            return result

        bridge = ApprovalBridge(config=config, approval_callback=mock_callback)
        interrupt = {
            "data": {
                "action_request": {
                    "name": "test_tool",
                    "args": {"param": "value"},
                    "description": "Test tool",
                }
            }
        }

        result = await bridge.handle_interrupt(interrupt)
        
        assert result["approved"] is True
        assert len(callback_results) == 1

    @pytest.mark.asyncio
    async def test_handle_interrupt_auto_approve_no_callback(self):
        """Test auto-approve when no callback provided."""
        config = VibeConfig()
        bridge = ApprovalBridge(config=config)
        interrupt = {
            "data": {
                "action_request": {
                    "name": "test_tool",
                    "args": {"param": "value"},
                    "description": "Test tool",
                }
            }
        }

        result = await bridge.handle_interrupt(interrupt)
        
        assert result["approved"] is True
        assert "no approval callback" in result["feedback"]

    @pytest.mark.asyncio
    async def test_respond_without_pending(self):
        """Test responding when no pending approval."""
        config = VibeConfig()
        bridge = ApprovalBridge(config=config)
        # Should not raise - use a fake request_id since there's no pending request
        await bridge.respond(True, "fake_request_id")

    @pytest.mark.asyncio
    async def test_session_auto_approve_functionality(self):
        """Test session-wide auto-approval tracking."""
        config = VibeConfig()
    
        # Mock callback that always approves with "always" option
        async def mock_callback(action_request):
            return {"approved": True, "always_approve": True, "feedback": None}
    
        bridge = ApprovalBridge(config=config, approval_callback=mock_callback)
    
        interrupt = {
            "data": {
                "action_request": {
                    "name": "test_tool",
                    "args": {"param": "value"},
                    "description": "Test tool",
                }
            }
        }
    
        # First call should add to session auto-approve
        result1 = await bridge.handle_interrupt(interrupt)
        assert result1["approved"] is True
        assert result1["always_approve"] is True
        assert "test_tool" in bridge._session_auto_approve

        # Second call should be auto-approved
        result2 = await bridge.handle_interrupt(interrupt)
        assert result2["approved"] is True
        assert "Auto-approved for session" in result2["feedback"]
