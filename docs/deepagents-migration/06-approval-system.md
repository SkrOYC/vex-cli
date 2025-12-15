# 06 - Approval System Migration

## Overview

Migrate from Mistral Vibe's sophisticated approval system to DeepAgents' HumanInTheLoopMiddleware while preserving the rich TUI approval dialogs.

## Current Approval System

### Complex Permission Model

```python
# vibe/core/tools/base.py
class ToolPermission(StrEnum):
    ALWAYS = auto()  # No approval needed
    ASK = auto()     # Ask user for approval
    NEVER = auto()   # Tool disabled

class BaseToolConfig(BaseModel):
    permission: ToolPermission = ToolPermission.ASK
    allowlist: list[str] = Field(default_factory=list)  # Regex patterns
    denylist: list[str] = Field(default_factory=list)   # Regex patterns
```

### Approval Flow

```python
# vibe/core/agent.py
async def _execute_tools(self, tool_calls: list[ToolCall]):
    for tool_call in tool_calls:
        if not self.auto_approve:
            # Get approval from user
            approved = await self._get_approval(tool_call)
            if not approved:
                # Skip or handle rejection
                continue
        # Execute tool
```

### TUI Approval Dialogs

```python
# vibe/cli/textual_ui/widgets/approval_app.py
class ApprovalApp(Container):
    # Rich approval UI with keyboard navigation
    # Shows tool details, diffs, and decision options
```

## DeepAgents Approval System

### HumanInTheLoopMiddleware

```python
from langchain.agents.middleware import HumanInTheLoopMiddleware, InterruptOnConfig

# Simple configuration
interrupt_on = {
    "write_file": True,      # Always ask
    "execute": True,         # Always ask
    "read_file": False,      # Never ask
}

middleware = HumanInTheLoopMiddleware(interrupt_on=interrupt_on)
```

### Interrupt Flow

DeepAgents uses LangGraph interrupts:

```python
# When tool needs approval, execution pauses
# Control returns to caller with interrupt info
# Caller can resume with decision

# Resume with approval
agent.invoke(None, {"configurable": {"thread_id": thread_id}})

# Or modify before resuming
agent.update_state(
    config,
    {"messages": [HumanMessage(content="Approved")]},
    as_node="human",
)
```

## Migration Strategy

### Phase 1: Approval Bridge

Create a bridge between DeepAgents HITL and Vibe's TUI:

```python
# vibe/core/engine/adapters.py

from typing import Any, Protocol
from langchain.agents.middleware.human_in_the_loop import ActionRequest


class ApprovalCallback(Protocol):
    """Protocol for approval callbacks."""
    async def request_approval(self, action: ActionRequest) -> bool: ...


class ApprovalBridge:
    """Bridge between DeepAgents HITL and Vibe TUI."""
    
    def __init__(self, callback: ApprovalCallback | None = None):
        self.callback = callback
        self._pending_approvals: dict[str, ActionRequest] = {}
    
    async def handle_interrupt(self, interrupt: dict[str, Any]) -> dict[str, Any]:
        """Handle a LangGraph interrupt for approval."""
        if not self.callback:
            return {"approved": True}  # Auto-approve if no callback
        
        action_request = self._extract_action_request(interrupt)
        if not action_request:
            return {"approved": True}
        
        # Store for potential resume
        request_id = str(uuid4())
        self._pending_approvals[request_id] = action_request
        
        # Request approval via TUI
        approved = await self.callback.request_approval(action_request)
        
        return {
            "approved": approved,
            "request_id": request_id,
            "action": action_request,
        }
    
    def _extract_action_request(self, interrupt: dict[str, Any]) -> ActionRequest | None:
        """Extract ActionRequest from LangGraph interrupt."""
        # Parse interrupt data into ActionRequest format
        # This depends on how HITL structures the interrupt
        return None


def build_interrupt_config(config: VibeConfig) -> dict[str, bool | InterruptOnConfig]:
    """Build DeepAgents interrupt config from Vibe tool permissions."""
    interrupt_on = {}
    
    # Map Vibe permissions to DeepAgents interrupts
    for tool_name, tool_config in config.tools.items():
        match tool_config.permission:
            case ToolPermission.ALWAYS:
                # No interrupt needed
                pass
            case ToolPermission.ASK:
                interrupt_on[tool_name] = True
            case ToolPermission.NEVER:
                # Tool will be filtered out during tool loading
                pass
    
    # Add default dangerous tools
    interrupt_on.update({
        "write_file": True,
        "edit_file": True,
        "execute": True,
        "bash": True,
    })
    
    return interrupt_on
```

### Phase 2: TUI Integration

Update the TUI to handle DeepAgents interrupts:

```python
# vibe/cli/textual_ui/app.py

class VibeApp(App):
    def __init__(self, config: VibeConfig):
        super().__init__()
        self.engine = VibeEngine(
            config=config,
            approval_callback=self._create_approval_bridge(),
        )

    def _create_approval_bridge(self) -> ApprovalBridge:
        """Create approval bridge for TUI integration."""
        async def request_approval(action: ActionRequest) -> bool:
            # Show approval dialog
            result = await self.push_screen_wait(ApprovalScreen(action))
            return result.approved
        
        return ApprovalBridge(callback=request_approval)

    async def on_engine_interrupt(self, interrupt: dict[str, Any]):
        """Handle engine interrupt (approval needed)."""
        decision = await self.approval_bridge.handle_interrupt(interrupt)
        
        if decision["approved"]:
            # Resume execution
            await self.engine.resume_execution(decision)
        else:
            # Handle rejection
            await self.engine.reject_execution(decision)
```

### Phase 3: Enhanced Approval UI

Preserve and enhance the approval dialogs:

```python
# vibe/cli/textual_ui/widgets/approval_screen.py

class ApprovalScreen(ModalScreen[bool]):
    """Enhanced approval screen for DeepAgents actions."""
    
    def __init__(self, action_request: ActionRequest):
        super().__init__()
        self.action_request = action_request
    
    def compose(self) -> ComposeResult:
        with Vertical():
            yield Header()
            
            with Container(id="action-details"):
                yield Static(f"Tool: {self.action_request['name']}")
                yield Static(f"Action: {self.action_request.get('description', 'No description')}")
                
                # Show arguments
                args = self.action_request.get("args", {})
                if args:
                    yield Static("Arguments:")
                    for key, value in args.items():
                        yield Static(f"  {key}: {truncate_value(str(value))}")
            
            # Show diff if available (for file operations)
            if hasattr(self.action_request, "diff"):
                yield Static("Changes:")
                yield DiffDisplay(self.action_request.diff)
            
            with Horizontal(id="buttons"):
                yield Button("Approve", id="approve", variant="primary")
                yield Button("Reject", id="reject", variant="error")
                yield Button("Edit Args", id="edit", variant="default")
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "approve":
            self.dismiss(True)
        elif event.button.id == "reject":
            self.dismiss(False)
        elif event.button.id == "edit":
            # Show edit dialog
            await self._show_edit_dialog()
```

## Permission Mapping

### Vibe Permissions → DeepAgents Interrupts

| Vibe Permission | DeepAgents Interrupt | Behavior |
|-----------------|---------------------|----------|
| `ALWAYS` | No interrupt | Tool executes immediately |
| `ASK` | `True` | Interrupt for approval |
| `NEVER` | Tool filtered out | Tool not available |

### Pattern-Based Permissions

Vibe's allowlist/denylist patterns need custom handling:

```python
# vibe/core/engine/permissions.py

def check_allowlist_denylist(
    tool_name: str, 
    args: dict[str, Any], 
    config: VibeConfig
) -> ToolPermission:
    """Check if tool call matches allowlist/denylist patterns."""
    tool_config = config.get_tool_config(tool_name)
    
    # Check denylist first
    for pattern in tool_config.denylist:
        if matches_pattern(args, pattern):
            return ToolPermission.NEVER
    
    # Check allowlist
    for pattern in tool_config.allowlist:
        if matches_pattern(args, pattern):
            return ToolPermission.ALWAYS
    
    # No match, use default permission
    return tool_config.permission

def matches_pattern(args: dict[str, Any], pattern: str) -> bool:
    """Check if args match a pattern (file paths, etc.)."""
    # Implement pattern matching logic
    # e.g., fnmatch for file paths
    return False
```

## Interrupt Handling

### DeepAgents Interrupt Flow

```python
# When HITL interrupts, execution pauses
# The interrupt contains information about what needs approval

interrupt = {
    "type": "human_input",
    "data": {
        "action_request": {
            "name": "write_file",
            "args": {"path": "/file.txt", "content": "hello"},
            "description": "Write content to file"
        }
    }
}

# TUI shows approval dialog
# User makes decision
# Execution resumes with decision
```

### Resume Patterns

```python
# Simple approval
agent.invoke(None, config)

# With modification
agent.update_state(
    config,
    {"messages": [HumanMessage(content="Approved with modifications")]},
    as_node="human",
)
```

## Files to Preserve/Modify

### Preserve (Enhanced)
- `vibe/cli/textual_ui/widgets/approval_app.py` - Core approval UI
- `vibe/core/tools/base.py` - Permission enums and logic

### Modify
- `vibe/core/agent.py` - Remove approval logic (moved to engine)
- `vibe/cli/textual_ui/app.py` - Add interrupt handling

### Remove
- Complex approval state management from agent
- Direct approval callbacks in conversation loop

## Validation Checklist

- [ ] Tool permissions map correctly to interrupts
- [ ] Approval dialogs display correctly
- [ ] Interrupt/resume flow works
- [ ] Pattern-based permissions (allowlist/denylist) preserved
- [ ] File operation diffs show in approval UI
- [ ] Keyboard navigation preserved
- [ ] Auto-approve mode works
- [ ] Rejection handling works
- [ ] Edit args functionality works
