# 09 - TUI Integration

## Overview

Update the Textual TUI to work with the new LangChain 1.2.0 engine, ensuring the user experience remains unchanged while simplifying the integration code.

## TUI Changes

### `vibe/cli/textual_ui/app.py`

```python
# Before: Using VibeEngine with adapters
from vibe.core.engine.engine import VibeEngine

class VibeApp(App):
    def __init__(self, config: VibeConfig) -> None:
        super().__init__()
        self.config = config
        self.engine = VibeEngine(
            config=config,
            approval_callback=self._create_approval_callback(),
        )

# After: Using VibeLangChainEngine with native events
from vibe.core.engine.langchain_engine import VibeLangChainEngine

class VibeApp(App):
    def __init__(self, config: VibeConfig) -> None:
        super().__init__()
        self.config = config
        self.engine = VibeLangChainEngine(  # Changed class name
            config=config,
            approval_callback=self._create_approval_callback(),
        )
```

### Event Handling

```python
# Before: Handling translated events
async def on_user_message(self, message: str) -> None:
    async for event in self.engine.run(message):
        translated = EventTranslator.translate(event)  # Adapter!
        await self._handle_event(translated)

# After: Handling native events (or mapped events)
async def on_user_message(self, message: str) -> None:
    async for event in self.engine.run(message):
        # Direct event handling - no translation!
        await self._handle_event(event)
```

### Approval Flow

```python
# Before: Using ApprovalBridge
async def on_approval_decision(self, approved: bool, feedback: str | None = None):
    await self.engine.approval_bridge.respond(approved, request_id, feedback)

# After: Using native Command pattern
async def on_approval_decision(self, approved: bool, feedback: str | None = None):
    await self.engine.handle_approval(approved, request_id, feedback)
```

## What Stays the Same

- **Chat Input Container** - No changes
- **Message Display** - No changes (handles both event types)
- **Approval Dialogs** - No changes to UI
- **Context Progress** - No changes
- **Keyboard Shortcuts** - No changes
- **All Widgets** - No changes

## TUI Integration Checklist

- [ ] TUI loads and initializes correctly
- [ ] Chat messages display properly
- [ ] Tool execution events are shown
- [ ] Approval dialogs work (native HITL)
- [ ] Context warnings display
- [ ] All keyboard shortcuts work
- [ ] No regressions in UX
