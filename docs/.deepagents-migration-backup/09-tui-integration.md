# 09 - Textual UI Integration

## Overview

Preserve Mistral Vibe's rich Textual TUI while integrating it with DeepAgents' LangGraph-based execution engine.

## Current TUI Architecture

### Core Components

```python
# vibe/cli/textual_ui/app.py (~400 lines)

class VibeApp(App):
    """Main Textual application."""
    
    def __init__(self, config: VibeConfig):
        super().__init__()
        self.config = config
        self.agent = Agent(config)  # Direct agent integration
        
    async def on_user_message(self, message: str):
        """Handle user input and run conversation."""
        async for event in self.agent.run(message):
            await self._handle_event(event)
    
    async def _handle_event(self, event: BaseEvent):
        """Route events to UI components."""
        match event:
            case AssistantEvent():
                self._append_assistant_message(event.content)
            case ToolCallEvent():
                self._show_tool_call(event.tool_call, event.tool_config)
            case ToolResultEvent():
                self._show_tool_result(event.tool_call, event.result, event.success)
```

### UI Widgets

```python
# vibe/cli/textual_ui/widgets/

├── approval_app.py      # Approval dialogs
├── chat_input.py        # Message input
├── compact.py          # Context compaction display
├── config_app.py       # Configuration UI
├── context_progress.py  # Token usage display
├── loading.py          # Loading indicators
├── messages.py         # Message display
├── mode_indicator.py   # Current mode display
├── path_display.py     # Working directory
├── tool_widgets.py     # Tool call/result display
├── welcome.py          # Welcome banner
```

## DeepAgents TUI Integration

### Phase 1: Engine Abstraction

Create an abstraction layer so TUI doesn't know about DeepAgents:

```python
# vibe/cli/textual_ui/engine_interface.py

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any

from vibe.core.types import BaseEvent


class EngineInterface(ABC):
    """Abstract interface for conversation engines."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the engine."""
        pass
    
    @abstractmethod
    async def run_conversation(self, message: str) -> AsyncGenerator[BaseEvent, None]:
        """Run a conversation turn, yielding events."""
        pass
    
    @abstractmethod
    async def handle_approval(self, approved: bool, feedback: str | None = None) -> None:
        """Handle approval decision."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset conversation state."""
        pass
    
    @property
    @abstractmethod
    def stats(self) -> dict[str, Any]:
        """Get current statistics."""
        pass


class DeepAgentsEngineInterface(EngineInterface):
    """DeepAgents implementation of engine interface."""
    
    def __init__(self, config: VibeConfig, approval_callback=None):
        from vibe.core.engine import VibeEngine
        self.engine = VibeEngine(config, approval_callback)
    
    async def initialize(self) -> None:
        self.engine.initialize()
    
     async def run_conversation(self, message: str) -> AsyncGenerator[BaseEvent, None]:
         async for event in self.engine.run(message):
             yield event
    
    async def handle_approval(self, approved: bool, feedback: str | None = None) -> None:
        await self.engine.handle_approval(approved, feedback)
    
    def reset(self) -> None:
        self.engine.reset()
    
    @property
    def stats(self) -> dict[str, Any]:
        return self.engine.stats


# Backward compatibility
class LegacyAgentInterface(EngineInterface):
    """Legacy Agent implementation for comparison."""
    
    def __init__(self, config: VibeConfig, approval_callback=None):
        from vibe.core.agent import Agent
        self.agent = Agent(config, auto_approve=approval_callback is None)
    
    async def initialize(self) -> None:
        pass  # Legacy agent doesn't need initialization
    
    async def run_conversation(self, message: str) -> AsyncGenerator[BaseEvent, None]:
        async for event in self.agent.run(message):
            yield event
    
    async def handle_approval(self, approved: bool, feedback: str | None = None) -> None:
        # Legacy agent handles approval internally
        pass
    
    def reset(self) -> None:
        # Legacy agent reset logic
        pass
    
    @property
    def stats(self) -> dict[str, Any]:
        return self.agent.stats
```

### Phase 2: TUI Engine Integration

Update VibeApp to use the engine interface:

```python
# vibe/cli/textual_ui/app.py

class VibeApp(App):
    """Main Textual application with pluggable engine."""
    
    def __init__(self, config: VibeConfig, engine: EngineInterface | None = None):
        super().__init__()
        self.config = config
        
        # Use DeepAgents engine by default
        if engine is None:
            engine = DeepAgentsEngineInterface(
                config=config,
                approval_callback=self._create_approval_callback(),
            )
        
        self.engine = engine
        self._initialized = False
    
    async def on_mount(self) -> None:
        """Initialize on mount."""
        await self.engine.initialize()
        self._initialized = True
    
    async def on_user_message(self, message: str) -> None:
        """Handle user input through engine interface."""
        if not self._initialized:
            await self.engine.initialize()
            self._initialized = True
        
        async for event in self.engine.run_conversation(message):
            await self._handle_event(event)
    
    def _create_approval_callback(self):
        """Create approval callback for engine."""
        async def approval_callback(action_request):
            # Show approval dialog
            result = await self.push_screen_wait(ApprovalScreen(action_request))
            return result.approved
        
        return approval_callback
    
    async def action_reset_conversation(self) -> None:
        """Reset conversation."""
        self.engine.reset()
        # Clear UI state
        self._clear_messages()
        self._show_welcome_message()
```

### Phase 3: UI Component Updates

### Approval Dialogs

Update approval UI to work with DeepAgents interrupts:

```python
# vibe/cli/textual_ui/widgets/approval_screen.py

class ApprovalScreen(ModalScreen[bool]):
    """Approval screen for DeepAgents actions."""
    
    def __init__(self, action_request: dict[str, Any]):
        super().__init__()
        self.action_request = action_request
    
    def compose(self) -> ComposeResult:
        with Vertical():
            yield Header()
            
            # Action details
            with Container(id="action-details"):
                yield Static(f"Tool: {self.action_request.get('name', 'Unknown')}")
                
                args = self.action_request.get("args", {})
                if args:
                    yield Static("Arguments:")
                    for key, value in args.items():
                        yield Static(f"  {key}: {truncate_value(str(value))}")
            
            # Show diff for file operations
            if self._is_file_operation():
                diff = self._generate_diff()
                if diff:
                    yield Static("Changes:")
                    yield DiffDisplay(diff)
            
            # Action buttons
            with Horizontal(id="buttons"):
                yield Button("Approve", id="approve", variant="primary")
                yield Button("Reject", id="reject", variant="error")
                yield Button("Edit", id="edit", variant="default")
    
    def _is_file_operation(self) -> bool:
        """Check if this is a file operation that can show diffs."""
        return self.action_request.get("name") in ["write_file", "edit_file"]
    
    def _generate_diff(self) -> str | None:
        """Generate diff for file operations."""
        # Extract old/new content from action request
        # This requires enriching the action request with file state
        return None
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "approve":
            self.dismiss(True)
        elif event.button.id == "reject":
            self.dismiss(False)
        elif event.button.id == "edit":
            await self._show_edit_dialog()
```

### Message Display

Update message widgets to handle new event types:

```python
# vibe/cli/textual_ui/widgets/messages.py

class AssistantMessage(Widget):
    """Display assistant messages with streaming support."""
    
    def __init__(self, content: str = "", streaming: bool = False):
        super().__init__()
        self.content = content
        self.streaming = streaming
    
    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("Assistant", classes="message-header")
            yield Static(self.content, id="content", classes="message-content")
            if self.streaming:
                yield LoadingWidget()
    
    def update_content(self, new_content: str, streaming: bool = False) -> None:
        """Update message content (for streaming)."""
        self.content = new_content
        self.streaming = streaming
        
        content_widget = self.query_one("#content", Static)
        content_widget.update(new_content)
        
        # Update loading indicator
        loading = self.query_one(LoadingWidget)
        if streaming:
            loading.display = True
        else:
            loading.display = False

class ToolCallMessage(Widget):
    """Display tool calls with DeepAgents formatting."""
    
    def __init__(self, tool_name: str, tool_args: dict[str, Any]):
        super().__init__()
        self.tool_name = tool_name
        self.tool_args = tool_args
    
    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("Tool Call", classes="message-header")
            
            # Format tool call like DeepAgents UI
            formatted = format_tool_display(self.tool_name, self.tool_args)
            yield Static(formatted, classes="tool-call")
            
            yield LoadingWidget()  # Show while tool is running
```

### Context Progress

Update token tracking to work with LangGraph state:

```python
# vibe/cli/textual_ui/widgets/context_progress.py

class ContextProgress(Widget):
    """Show context usage and token counts."""
    
    def __init__(self, engine: EngineInterface):
        super().__init__()
        self.engine = engine
    
    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Static("Context: ", id="label")
            yield ProgressBar(id="progress", total=100)
            yield Static("", id="tokens")
    
    async def update_progress(self) -> None:
        """Update progress from engine stats."""
        stats = self.engine.stats
        
        # Estimate context usage
        estimated_tokens = stats.get("messages", 0) * 50  # Rough estimate
        max_context = 200000  # From config
        
        percentage = min(100, (estimated_tokens / max_context) * 100)
        
        progress = self.query_one("#progress", ProgressBar)
        progress.progress = percentage
        
        tokens_display = self.query_one("#tokens", Static)
        tokens_display.update(f"{estimated_tokens:,} / {max_context:,}")
```

## Phase 4: Command Integration

### Slash Commands

Preserve and enhance slash commands:

```python
# vibe/cli/commands.py

async def handle_command(command: str, engine: EngineInterface, token_tracker) -> str | bool:
    """Handle slash commands with engine abstraction."""
    cmd = command.lower().strip().lstrip("/")
    
    if cmd == "clear":
        engine.reset()
        token_tracker.reset()
        # Clear UI
        return True
    
    if cmd == "tokens":
        stats = engine.stats
        # Display stats
        return True
    
    # Other commands...
```

### Mode Indicator

Update mode indicator for DeepAgents features:

```python
# vibe/cli/textual_ui/widgets/mode_indicator.py

class ModeIndicator(Widget):
    """Show current agent mode and features."""
    
    def __init__(self, config: VibeConfig):
        super().__init__()
        self.config = config
    
    def compose(self) -> ComposeResult:
        features = []
        if self.config.enable_subagents:
            features.append("🤖 Subagents")
        if self.config.enable_planning:
            features.append("📋 Planning")
        if self.config.enable_memory:
            features.append("🧠 Memory")
        
        feature_text = " | ".join(features) if features else "Standard"
        
        with Horizontal():
            yield Static("Mode: ", classes="mode-label")
            yield Static(feature_text, classes="mode-features")
```

## Phase 5: Error Handling

### Engine Errors

Handle DeepAgents-specific errors in TUI:

```python
# vibe/cli/textual_ui/app.py

class VibeApp(App):
    async def on_user_message(self, message: str) -> None:
        try:
            async for event in self.engine.run_conversation(message):
                await self._handle_event(event)
        except Exception as e:
            # Show error in UI
            await self._show_error(f"Engine error: {e}")
            # Offer to reset or retry
```

### Interrupt Handling

Handle LangGraph interrupts gracefully:

```python
# vibe/cli/textual_ui/app.py

async def _handle_engine_interrupt(self, interrupt: dict[str, Any]) -> None:
    """Handle DeepAgents interrupts (approvals, etc.)."""
    interrupt_type = interrupt.get("type")
    
    if interrupt_type == "human_input":
        # Show approval dialog
        approved = await self._show_approval_dialog(interrupt["data"])
        await self.engine.handle_approval(approved)
    elif interrupt_type == "error":
        # Show error dialog
        await self._show_error_dialog(interrupt["data"])
    else:
        # Log unknown interrupt
        logger.warning(f"Unknown interrupt type: {interrupt_type}")
```

## Files to Modify

### Core TUI
- `vibe/cli/textual_ui/app.py` - Main app integration
- `vibe/cli/textual_ui/engine_interface.py` - New abstraction layer

### Widgets
- `vibe/cli/textual_ui/widgets/approval_screen.py` - Update for DeepAgents
- `vibe/cli/textual_ui/widgets/messages.py` - Handle new event types
- `vibe/cli/textual_ui/widgets/context_progress.py` - Update for LangGraph stats
- `vibe/cli/textual_ui/widgets/mode_indicator.py` - Show DeepAgents features

### Commands
- `vibe/cli/commands.py` - Update for engine interface

## Validation Checklist

- [ ] TUI launches with DeepAgents engine
- [ ] User input flows to engine correctly
- [ ] Events display in UI properly
- [ ] Streaming tokens work smoothly
- [ ] Tool calls show with proper formatting
- [ ] Approval dialogs work with interrupts
- [ ] Context progress updates correctly
- [ ] Error handling works gracefully
- [ ] Slash commands function properly
- [ ] Mode indicator shows correct features
- [ ] Performance matches or exceeds original
- [ ] All existing keyboard shortcuts work
- [ ] Visual design preserved
