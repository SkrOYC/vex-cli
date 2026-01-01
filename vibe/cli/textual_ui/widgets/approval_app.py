from __future__ import annotations

from typing import Any, ClassVar

from textual import events
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Container, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Static

from vibe.cli.textual_ui.renderers import get_renderer
from vibe.core.config import VibeConfig


class ApprovalApp(Container):
    can_focus = True
    can_focus_children = False

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("up", "move_up", "Up", show=False),
        Binding("down", "move_down", "Down", show=False),
        Binding("enter", "select", "Select", show=False),
        Binding("1", "select_1", "Yes", show=False),
        Binding("y", "select_1", "Yes", show=False),
        Binding("2", "select_2", "Always Tool Session", show=False),
        Binding("3", "select_3", "No", show=False),
        Binding("n", "select_3", "No", show=False),
        Binding("4", "select_4", "Approve All", show=False),
        Binding("5", "select_5", "Reject All", show=False),
    ]

    class ApprovalGranted(Message):
        def __init__(self, action_request: dict[str, Any]) -> None:
            super().__init__()
            self.action_request = action_request
            self.tool_name = action_request.get("name", "")
            self.tool_args = action_request.get("args", {})

    class ApprovalGrantedAlwaysTool(Message):
        def __init__(
            self, action_request: dict[str, Any], save_permanently: bool
        ) -> None:
            super().__init__()
            self.action_request = action_request
            self.tool_name = action_request.get("name", "")
            self.tool_args = action_request.get("args", {})
            self.save_permanently = save_permanently

    class ApprovalRejected(Message):
        def __init__(self, action_request: dict[str, Any]) -> None:
            super().__init__()
            self.action_request = action_request
            self.tool_name = action_request.get("name", "")
            self.tool_args = action_request.get("args", {})

    def __init__(
        self,
        action_requests: list[dict[str, Any]],
        workdir: str,
        config: VibeConfig,
        current_index: int = 0,
    ) -> None:
        super().__init__(id="approval-app")
        self.action_requests = action_requests
        self.current_index = current_index
        self.total_tools = len(action_requests)
        self.max_options = 5 if self.total_tools > 1 else 3

        # Get current tool info
        self.action_request = action_requests[current_index]
        self.tool_name = self.action_request.get("name", "")
        self.tool_args = self.action_request.get("args", {})
        self.workdir = workdir
        self.config = config
        self.selected_option = 0
        self.content_container: Vertical | None = None
        self.title_widget: Static | None = None
        self.tool_info_container: Vertical | None = None
        self.option_widgets: list[Static] = []
        self.help_widget: Static | None = None

        # Get reference to app's pending approval future
        self.app = self.app
        self.pending_approval_future = getattr(self.app, "_pending_approval", None)

    def compose(self) -> ComposeResult:
        with Vertical(id="approval-content"):
            self.title_widget = Static(self._get_title_text(), classes="approval-title")
            yield self.title_widget

            with VerticalScroll(classes="approval-tool-info-scroll"):
                self.tool_info_container = Vertical(
                    classes="approval-tool-info-container"
                )
                yield self.tool_info_container

            yield Static("")

            for _ in range(5):
                widget = Static("", classes="approval-option")
                self.option_widgets.append(widget)
                yield widget

            yield Static("")

            self.help_widget = Static(
                "↑↓ navigate  Enter select  ESC reject", classes="approval-help"
            )
            yield self.help_widget

    def _get_title_text(self) -> str:
        """Get title text, showing progress for multi-tool scenarios."""
        title_text = f"⚠ {self.tool_name} command"
        if self.total_tools > 1:
            title_text += f" ({self.current_index + 1}/{self.total_tools})"
        return title_text

    def _update_title(self) -> None:
        """Update title to show progress for multi-tool."""
        if self.title_widget:
            self.title_widget.update(self._get_title_text())

    async def on_mount(self) -> None:
        await self._update_tool_info()
        self._update_title()
        self._update_options()
        self.focus()

    async def _update_tool_info(self) -> None:
        if not self.tool_info_container:
            return

        renderer = get_renderer(self.tool_name)
        widget_class, data = renderer.get_approval_widget(self.tool_args)

        await self.tool_info_container.remove_children()
        approval_widget = widget_class(data)
        await self.tool_info_container.mount(approval_widget)

    def _update_options(self) -> None:
        options = [
            ("Yes", "yes"),
            (f"Yes and always allow {self.tool_name} this session", "yes"),
            ("No and tell the agent what to do instead", "no"),
        ]

        # Add batch shortcuts for multi-tool scenarios
        if self.total_tools > 1:
            options.append(("Approve All (remaining tools)", "yes"))
            options.append(("Reject All (remaining tools)", "no"))

        for idx, ((text, color_type), widget) in enumerate(
            zip(
                options[: self.max_options],
                self.option_widgets[: self.max_options],
                strict=True,
            )
        ):
            is_selected = idx == self.selected_option

            cursor = "› " if is_selected else "  "
            option_text = f"{cursor}{idx + 1}. {text}"

            widget.update(option_text)

            widget.remove_class("approval-cursor-selected")
            widget.remove_class("approval-option-selected")
            widget.remove_class("approval-option-yes")
            widget.remove_class("approval-option-no")

            if is_selected:
                widget.add_class("approval-cursor-selected")
                if color_type == "yes":
                    widget.add_class("approval-option-yes")
                else:
                    widget.add_class("approval-option-no")
            else:
                widget.add_class("approval-option-selected")
                if color_type == "yes":
                    widget.add_class("approval-option-yes")
                else:
                    widget.add_class("approval-option-no")

    def action_move_up(self) -> None:
        self.selected_option = (self.selected_option - 1) % self.max_options
        self._update_options()

    def action_move_down(self) -> None:
        self.selected_option = (self.selected_option + 1) % self.max_options
        self._update_options()

    def action_select(self) -> None:
        self._handle_selection(self.selected_option)

    def action_select_1(self) -> None:
        self.selected_option = 0
        self._handle_selection(0)

    def action_select_2(self) -> None:
        self.selected_option = 1
        self._handle_selection(1)

    def action_select_3(self) -> None:
        self.selected_option = 2
        self._handle_selection(2)

    def action_select_4(self) -> None:
        """Approve All shortcut (key 4)."""
        if self.total_tools > 1:
            self.selected_option = 3
            self._handle_selection(3)

    def action_select_5(self) -> None:
        """Reject All shortcut (key 5)."""
        if self.total_tools > 1:
            self.selected_option = 4
            self._handle_selection(4)

    def action_reject(self) -> None:
        self.selected_option = 2
        self._handle_selection(2)

    def _handle_selection(self, option: int) -> None:

        match option:
            case 0:
                # Approve once
                self._handle_approve(always_approve=False)
            case 1:
                # Approve always for session
                self._handle_approve(always_approve=True)
            case 2:
                # Reject
                self._handle_reject()
            case 3 if self.total_tools > 1:
                # Approve All
                self._handle_approve_all()
            case 4 if self.total_tools > 1:
                # Reject All
                self._handle_reject_all()

    def _handle_approve(self, always_approve: bool) -> None:
        """Handle approval decision."""
        result = {
            "approved": True,
            "always_approve": always_approve,
            "feedback": None
            if not always_approve
            else f"Auto-approve {self.tool_name} for this session",
        }

        # If more tools to approve, continue to next
        if self.total_tools > 1 and self.current_index < self.total_tools - 1:
            result["next_tool"] = self.current_index + 1

        self._set_approval_result(result)

    def _handle_reject(self) -> None:
        """Handle rejection decision."""
        result = {
            "approved": False,
            "always_approve": False,
            "feedback": "User rejected operation",
        }

        # Reject current tool, but continue with others
        if self.total_tools > 1 and self.current_index < self.total_tools - 1:
            result["next_tool"] = self.current_index + 1

        self._set_approval_result(result)

    def _handle_approve_all(self) -> None:
        """Handle Approve All shortcut."""
        result = {
            "approved": True,
            "always_approve": False,
            "feedback": None,
            "batch_approve": True,  # Signal to app
        }
        self._set_approval_result(result)

    def _handle_reject_all(self) -> None:
        """Handle Reject All shortcut."""
        result = {
            "approved": False,
            "always_approve": False,
            "feedback": "User rejected all operations",
            "batch_reject": True,  # Signal to app
        }
        self._set_approval_result(result)

    def _set_approval_result(self, result: dict[str, Any]) -> None:
        """Set the approval result on the app's pending approval future."""
        if self.pending_approval_future and not self.pending_approval_future.done():
            self.pending_approval_future.set_result(result)

    def on_blur(self, event: events.Blur) -> None:
        self.call_after_refresh(self.focus)
