"""Shared services for filesystem tools.

This module provides services that track file state and support filesystem operations.
"""

from __future__ import annotations

import time


class ViewTrackerService:
    """Tracks file views to enforce "view before edit" workflow.

    This service maintains an in-memory record of which files have been viewed
    during the current session. This enables tools to enforce that users must
    view a file before editing it, ensuring they understand the current content.

    The tracker uses a simple dictionary to store absolute file paths mapped to
    timestamps when the file was last viewed. All paths are normalized to absolute
    paths for consistent tracking.

    Attributes:
        _views: Dictionary mapping absolute file paths to view timestamps.

    Example:
        ```python
        tracker = ViewTrackerService()

        # Record that a file was viewed
        tracker.record_view("/project/src/main.py")

        # Check if file was viewed
        if tracker.has_been_viewed("/project/src/main.py"):
            print("File has been viewed")

        # Get the last view timestamp
        timestamp = tracker.get_last_view_timestamp("/project/src/main.py")
        ```
    """

    def __init__(self) -> None:
        """Initialize the ViewTrackerService with empty view records."""
        self._views: dict[str, int] = {}

    def record_view(self, file_path: str) -> None:
        """Record that a file was viewed.

        Stores the current timestamp for the given file path, enabling tracking
        of when files were last examined.

        Args:
            file_path: Absolute path to the file that was viewed.
        """
        self._views[file_path] = int(time.time() * 1000)

    def has_been_viewed(self, file_path: str) -> bool:
        """Check if a file has been viewed during this session.

        Args:
            file_path: Absolute path to check.

        Returns:
            True if the file has been viewed at least once, False otherwise.
        """
        return file_path in self._views

    def get_last_view_timestamp(self, file_path: str) -> int | None:
        """Get the timestamp when a file was last viewed.

        Args:
            file_path: Absolute path to check.

        Returns:
            Unix timestamp in milliseconds when the file was last viewed,
            or None if the file has never been viewed.
        """
        return self._views.get(file_path)

    def clear_view_record(self, file_path: str) -> None:
        """Remove the view record for a specific file.

        After calling this method, has_been_viewed() will return False for
        the given file path until record_view() is called again.

        Args:
            file_path: Absolute path to clear from the view records.
        """
        self._views.pop(file_path, None)

    def clear_all(self) -> None:
        """Clear all view records.

        This resets the tracker, so has_been_viewed() will return False
        for all files until they are viewed again.
        """
        self._views.clear()
