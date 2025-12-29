# 06 - Approval System Migration

## Overview

Replace the `ApprovalBridge` adapter with native LangChain 1.2.0 `HumanInTheLoopMiddleware`, simplifying the approval workflow by using built-in interrupt handling.

## Current Approval System

### `vibe/core/engine/adapters.py` (ApprovalBridge - 256 lines)

```python
class ApprovalBridge:
    """Bridge LangGraph interrupts to Vibe TUI approval flow."""
    
    def __init__(self, config, approval_callback=None):
        self.config = config
        self.approval_callback = approval_callback
        self._pending_approvals: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._session_auto_approve: set[str] = set()
    
    async def handle_interrupt(self, interrupt: dict[str, Any]) -> dict[str, Any]:
        """Handle LangGraph interrupts for approval."""
        # Complex translation logic
        action_request = self._extract_action_request(interrupt)
        # ... approval flow with timeouts and pattern matching
    
    async def respond(self, approved: bool, request_id: str, feedback: str | None = None):
        """Respond to pending approval."""
        # Set future result for waiting code
```

## New Approval System

### Native `HumanInTheLoopMiddleware` Integration

```python
# New: vibe/core/engine/langchain_engine.py (partial)

def _build_interrupt_config(self) -> dict[str, Any]:
    """Build HITL interrupt config from Vibe tool permissions."""
    from vibe.core.engine.permissions import build_interrupt_config
    return build_interrupt_config(self.config)

def _build_middleware_stack(self) -> list[AgentMiddleware]:
    """Build the custom middleware stack."""
    from langchain.agents.middleware import HumanInTheLoopMiddleware
    
    middleware: list[AgentMiddleware] = []
    
    # ... other middleware ...
    
    # Human-in-the-loop (native LangChain 1.2.0)
    interrupt_on = self._build_interrupt_config()
    if interrupt_on:
        middleware.append(
            HumanInTheLoopMiddleware(interrupt_on=interrupt_on)
        )
    
    return middleware
```

### Native Interrupt Handling

```python
# New: vibe/core/engine/langchain_engine.py (partial)

async def handle_approval(
    self, approved: bool, request_id: str, feedback: str | None = None
) -> None:
    """Handle approval decision from TUI using native LangGraph Command."""
    from langgraph.types import Command
    
    config = {"configurable": {"thread_id": self._thread_id}}
    
    if approved:
        await self._agent.ainvoke(
            Command(resume={"approved": True, "feedback": feedback}),
            config=config,
        )
    else:
        await self._agent.ainvoke(
            Command(resume={"approved": False, "feedback": feedback}),
            config=config,
        )

async def reject_execution(self, feedback: str | None = None) -> None:
    """Reject execution and abort operation."""
    from langchain_core.messages import HumanMessage
    
    config = {"configurable": {"thread_id": self._thread_id}}
    
    rejection_message = HumanMessage(
        content=f"I reject this operation: {feedback or 'Operation rejected by user'}"
    )
    
    await self._agent.aupdate_state(
        config,
        {"messages": [rejection_message]},
        as_node="human",
    )
```

## TUI Integration

```python
# vibe/cli/textual_ui/widgets/approval_app.py

class ApprovalApp(Container):
    """Approval dialog using native LangGraph interrupts."""
    
    async def on_approval_decision(self, approved: bool, feedback: str | None = None):
        """Handle approval decision using native Command pattern."""
        # NEW: Direct call to native interrupt resume
        await self.engine.handle_approval(
            approved=approved,
            request_id=self.current_request_id,
            feedback=feedback,
        )
```

## Comparison

| Aspect | Before (DeepAgents) | After (LangChain 1.2.0) |
|--------|---------------------|--------------------------|
| **Interrupt Handling** | Custom `ApprovalBridge` | Native `HumanInTheLoopMiddleware` |
| **Resume Logic** | Custom `invoke(None, config)` | Native `Command(resume=...)` |
| **Reject Logic** | Custom `update_state()` | Native `Command(resume=...)` |
| **Pattern Matching** | Manual in `ApprovalBridge` | Via `interrupt_on` config |
| **Timeout Handling** | Manual in `ApprovalBridge` | Native in middleware |
| **Code LOC** | ~256 lines | ~50 lines |

## Validation Checklist

- [ ] Native HITL middleware works correctly
- [ ] Approval dialog displays for interrupted tools
- [ ] Approve/reject decisions resume execution
- [ ] Pattern-based auto-approval works
- [ ] No ApprovalBridge code remains
- [ ] Timeout handling works correctly
