"""Unit tests for filesystem shared services."""

from __future__ import annotations

import time

import pytest

from vibe.core.tools.filesystem.shared import ViewTrackerService


class TestViewTrackerService:
    """Tests for ViewTrackerService class."""

    @pytest.fixture
    def tracker(self) -> ViewTrackerService:
        """Create a fresh ViewTrackerService for each test."""
        return ViewTrackerService()

    def test_initially_empty(self, tracker: ViewTrackerService) -> None:
        """Test ViewTrackerService starts with no view records."""
        assert tracker.has_been_viewed("/any/path") is False

    def test_record_view(self, tracker: ViewTrackerService) -> None:
        """Test record_view stores the file path with timestamp."""
        tracker.record_view("/project/src/main.py")

        assert tracker.has_been_viewed("/project/src/main.py") is True

    def test_record_view_returns_none(self, tracker: ViewTrackerService) -> None:
        """Test record_view returns None."""
        result = tracker.record_view("/project/src/main.py")
        assert result is None

    def test_has_been_viewed_returns_true_for_recorded(
        self, tracker: ViewTrackerService
    ) -> None:
        """Test has_been_viewed returns True for recorded files."""
        tracker.record_view("/project/src/main.py")

        assert tracker.has_been_viewed("/project/src/main.py") is True

    def test_has_been_viewed_returns_false_for_unrecorded(
        self, tracker: ViewTrackerService
    ) -> None:
        """Test has_been_viewed returns False for unrecorded files."""
        assert tracker.has_been_viewed("/project/src/unviewed.py") is False

    def test_get_last_view_timestamp_returns_timestamp(
        self, tracker: ViewTrackerService
    ) -> None:
        """Test get_last_view_timestamp returns a valid timestamp."""
        before = int(time.time() * 1000)
        tracker.record_view("/project/src/main.py")
        after = int(time.time() * 1000)

        timestamp = tracker.get_last_view_timestamp("/project/src/main.py")

        assert timestamp is not None
        assert before <= timestamp <= after

    def test_get_last_view_timestamp_returns_none_for_unrecorded(
        self, tracker: ViewTrackerService
    ) -> None:
        """Test get_last_view_timestamp returns None for unrecorded files."""
        timestamp = tracker.get_last_view_timestamp("/project/src/unviewed.py")
        assert timestamp is None

    def test_clear_view_record_removes_specific_entry(
        self, tracker: ViewTrackerService
    ) -> None:
        """Test clear_view_record removes only the specified file."""
        tracker.record_view("/project/src/main.py")
        tracker.record_view("/project/src/other.py")

        tracker.clear_view_record("/project/src/main.py")

        assert tracker.has_been_viewed("/project/src/main.py") is False
        assert tracker.has_been_viewed("/project/src/other.py") is True

    def test_clear_view_record_nonexistent_path(
        self, tracker: ViewTrackerService
    ) -> None:
        """Test clear_view_record does not raise for nonexistent path."""
        tracker.record_view("/project/src/main.py")

        # Should not raise
        tracker.clear_view_record("/project/src/nonexistent.py")

        # Original file should still be tracked
        assert tracker.has_been_viewed("/project/src/main.py") is True

    def test_clear_all_removes_all_entries(self, tracker: ViewTrackerService) -> None:
        """Test clear_all removes all view records."""
        tracker.record_view("/project/src/main.py")
        tracker.record_view("/project/src/other.py")
        tracker.record_view("/project/src/third.py")

        tracker.clear_all()

        assert tracker.has_been_viewed("/project/src/main.py") is False
        assert tracker.has_been_viewed("/project/src/other.py") is False
        assert tracker.has_been_viewed("/project/src/third.py") is False

    def test_clear_all_on_empty_tracker(self, tracker: ViewTrackerService) -> None:
        """Test clear_all does not raise on empty tracker."""
        # Should not raise
        tracker.clear_all()

        assert tracker.has_been_viewed("/any/path") is False

    def test_record_view_updates_existing_timestamp(
        self, tracker: ViewTrackerService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test record_view updates the timestamp if file was already viewed."""
        current_time = 1000000000.0

        monkeypatch.setattr(time, "time", lambda: current_time)
        tracker.record_view("/project/src/main.py")
        first_timestamp = tracker.get_last_view_timestamp("/project/src/main.py")

        # Advance time
        current_time += 10
        tracker.record_view("/project/src/main.py")
        second_timestamp = tracker.get_last_view_timestamp("/project/src/main.py")

        assert first_timestamp is not None
        assert second_timestamp is not None
        assert second_timestamp > first_timestamp

    def test_has_been_viewed_with_empty_string(
        self, tracker: ViewTrackerService
    ) -> None:
        """Test has_been_viewed returns False for empty string path."""
        assert tracker.has_been_viewed("") is False

    def test_record_view_with_empty_string(self, tracker: ViewTrackerService) -> None:
        """Test record_view accepts empty string path."""
        tracker.record_view("")

        assert tracker.has_been_viewed("") is True
        assert tracker.get_last_view_timestamp("") is not None

    def test_get_last_view_timestamp_empty_string(
        self, tracker: ViewTrackerService
    ) -> None:
        """Test get_last_view_timestamp returns None for unrecorded empty string."""
        timestamp = tracker.get_last_view_timestamp("")
        assert timestamp is None

    def test_track_multiple_files(self, tracker: ViewTrackerService) -> None:
        """Test tracking multiple files independently."""
        tracker.record_view("/project/src/file1.py")
        tracker.record_view("/project/src/file2.py")
        tracker.record_view("/project/src/file3.py")

        assert tracker.has_been_viewed("/project/src/file1.py") is True
        assert tracker.has_been_viewed("/project/src/file2.py") is True
        assert tracker.has_been_viewed("/project/src/file3.py") is True
        assert tracker.has_been_viewed("/project/src/file4.py") is False

    def test_track_same_file_multiple_times(self, tracker: ViewTrackerService) -> None:
        """Test tracking the same file multiple times."""
        tracker.record_view("/project/src/main.py")
        tracker.record_view("/project/src/main.py")
        tracker.record_view("/project/src/main.py")

        assert tracker.has_been_viewed("/project/src/main.py") is True

        # Should have the latest timestamp only
        timestamp = tracker.get_last_view_timestamp("/project/src/main.py")
        assert timestamp is not None

    def test_clear_one_file_keeps_others(self, tracker: ViewTrackerService) -> None:
        """Test clearing one file does not affect others."""
        tracker.record_view("/project/src/main.py")
        tracker.record_view("/project/src/other.py")

        tracker.clear_view_record("/project/src/main.py")

        assert tracker.has_been_viewed("/project/src/main.py") is False
        assert tracker.has_been_viewed("/project/src/other.py") is True
        assert tracker.get_last_view_timestamp("/project/src/other.py") is not None

    def test_timestamps_are_milliseconds(self, tracker: ViewTrackerService) -> None:
        """Test that timestamps are in milliseconds (Unix epoch * 1000)."""
        tracker.record_view("/project/src/main.py")
        timestamp = tracker.get_last_view_timestamp("/project/src/main.py")

        assert timestamp is not None
        # Should be a large number (milliseconds since epoch)
        assert timestamp > 1_000_000_000_000  # After year 2001
        # Should not be in seconds (would be ~1/1000 of this)
        assert timestamp < 10_000_000_000_000  # Before year 2286
