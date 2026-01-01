"""Test snapshot for basic conversation with VibeLangChainEngine."""

from __future__ import annotations

from textual.pilot import Pilot

from tests.mock.utils import mock_llm_chunk
from tests.snapshots.base_snapshot_test_app import BaseSnapshotTestApp, default_config
from tests.snapshots.snap_compare import SnapCompare
from tests.stubs.fake_backend import FakeBackend
from vibe.core.engine import VibeLangChainEngine


class SnapshotTestAppWithConversation(BaseSnapshotTestApp):
    def __init__(self) -> None:
        config = default_config()
        engine = VibeLangChainEngine(config=config)
        engine.initialize()
        super().__init__(config=config)
        self.agent = engine


def test_snapshot_shows_basic_conversation(snap_compare: SnapCompare) -> None:
    async def run_before(pilot: Pilot) -> None:
        await pilot.press(*"Hello there, who are you?")
        await pilot.press("enter")
        await pilot.pause(0.4)

    assert snap_compare(
        "test_ui_snapshot_basic_conversation.py:SnapshotTestAppWithConversation",
        terminal_size=(120, 36),
        run_before=run_before,
    )
