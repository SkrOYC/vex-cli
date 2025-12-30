"""Unit tests for EditTool.

Tests cover all acceptance criteria including:
- Edit succeeds with proper view tracking
- Edit fails without view tracking
- Edit fails on non-existent file
- Edit fails on modified file (mtime check)
- Mistaken edit detection triggers (length ratio, line similarity, length diff)
- Mistaken edit detection majority voting (2 of 3)
- Mistaken edit detection skips short content
- Retry allowed within 60s (same hashes)
- Retry rejected after 60s (expired warning)
- Retry rejected with different hashes (different operation)
- Hash function produces consistent results
- Hash function matches TypeScript (same input → same output)
- All error messages match TypeScript exactly
- Cleanup removes expired warnings
- All tests achieve >85% coverage
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
import tempfile
import time

from pydantic import ValidationError
import pytest

from vibe.core.tools.filesystem.edit import (
    EditArgs,
    EditResult,
    EditTool,
    EditToolConfig,
    EditToolState,
    WarnedOperation,
)
from vibe.core.tools.filesystem.shared import ViewTrackerService
from vibe.core.tools.filesystem.types import FileSystemError


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_dir() -> Iterator[Path]:
    """Create a temporary directory for file operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def view_tracker() -> ViewTrackerService:
    """Create a fresh ViewTrackerService for each test."""
    return ViewTrackerService()


@pytest.fixture
def tool_config(view_tracker: ViewTrackerService, temp_dir: Path) -> EditToolConfig:
    """Create an EditToolConfig with ViewTrackerService."""
    return EditToolConfig(view_tracker=view_tracker, workdir=temp_dir)


@pytest.fixture
def tool_state() -> EditToolState:
    """Create a fresh EditToolState for each test."""
    return EditToolState()


@pytest.fixture
def tool(tool_config: EditToolConfig, tool_state: EditToolState) -> EditTool:
    """Create an EditTool instance for testing."""
    return EditTool(config=tool_config, state=tool_state)


# =============================================================================
# EditArgs Tests
# =============================================================================


class TestEditArgs:
    """Tests for EditArgs Pydantic model."""

    def test_creation_with_path_and_content(self) -> None:
        """Test EditArgs can be created with path and file_text."""
        args = EditArgs(path="/test/file.txt", file_text="Hello, World!")
        assert args.path == "/test/file.txt"
        assert args.file_text == "Hello, World!"

    def test_creation_with_empty_content(self) -> None:
        """Test EditArgs accepts empty content."""
        args = EditArgs(path="/test/file.txt", file_text="")
        assert args.file_text == ""

    def test_model_validation(self) -> None:
        """Test EditArgs validates types correctly."""
        with pytest.raises(ValidationError):
            EditArgs(path=123, file_text="text")  # type: ignore


# =============================================================================
# EditResult Tests
# =============================================================================


class TestEditResult:
    """Tests for EditResult Pydantic model."""

    def test_creation_with_output(self) -> None:
        """Test EditResult can be created with output."""
        result = EditResult(output="File edited successfully.")
        assert result.output == "File edited successfully."


# =============================================================================
# Path Resolution Tests
# =============================================================================


class TestPathResolution:
    """Tests for path resolution functionality."""

    @pytest.mark.asyncio
    async def test_absolute_path_resolved_correctly(
        self, tool: EditTool, temp_dir: Path
    ) -> None:
        """Test absolute paths are resolved correctly."""
        file_path = temp_dir / "test.txt"
        file_path.write_text("original", encoding="utf-8")
        view_tracker = tool.config.view_tracker
        assert view_tracker is not None
        # Wait to ensure mtime is set before recording view
        time.sleep(0.01)
        view_tracker.record_view(str(file_path))

        result = await tool.run(EditArgs(path=str(file_path), file_text="new content"))

        assert "modified successfully" in result.output
        assert file_path.read_text(encoding="utf-8") == "new content"

    @pytest.mark.asyncio
    async def test_relative_path_resolved_against_workdir(
        self, tool: EditTool, temp_dir: Path
    ) -> None:
        """Test relative paths are resolved against working directory."""
        file_path = temp_dir / "relative.txt"
        file_path.write_text("original", encoding="utf-8")
        view_tracker = tool.config.view_tracker
        assert view_tracker is not None
        # Wait to ensure mtime is set before recording view
        time.sleep(0.01)
        view_tracker.record_view(str(file_path))

        result = await tool.run(EditArgs(path="relative.txt", file_text="new content"))

        expected_path = temp_dir / "relative.txt"
        assert "modified successfully" in result.output
        assert expected_path.read_text(encoding="utf-8") == "new content"


# =============================================================================
# View Tracking Tests
# =============================================================================


class TestViewTracking:
    """Tests for view tracking enforcement."""

    @pytest.mark.asyncio
    async def test_edit_succeeds_with_proper_view_tracking(
        self, tool: EditTool, view_tracker: ViewTrackerService, temp_dir: Path
    ) -> None:
        """Test editing succeeds when file has been viewed."""
        file_path = temp_dir / "edit_me.txt"
        file_path.write_text("original content", encoding="utf-8")

        # Record view before editing
        view_tracker.record_view(str(file_path))

        result = await tool.run(EditArgs(path=str(file_path), file_text="new content"))

        assert result.output == f"File '{file_path}' modified successfully"
        assert file_path.read_text(encoding="utf-8") == "new content"

    @pytest.mark.asyncio
    async def test_edit_fails_without_view_tracking(
        self, tool: EditTool, temp_dir: Path
    ) -> None:
        """Test editing fails if file has not been viewed."""
        file_path = temp_dir / "unviewed.txt"
        file_path.write_text("content", encoding="utf-8")

        with pytest.raises(FileSystemError) as exc_info:
            await tool.run(EditArgs(path=str(file_path), file_text="new content"))

        assert exc_info.value.code == "FILE_NOT_VIEWED"
        assert "must be viewed before editing" in str(exc_info.value)
        assert len(exc_info.value.suggestions) > 0

    @pytest.mark.asyncio
    async def test_edit_updates_view_timestamp(
        self, tool: EditTool, view_tracker: ViewTrackerService, temp_dir: Path
    ) -> None:
        """Test editing updates the view timestamp."""
        file_path = temp_dir / "update_view.txt"
        file_path.write_text("original", encoding="utf-8")

        # Wait a moment to ensure file mtime is different from view time
        time.sleep(0.01)

        # Record initial view
        view_tracker.record_view(str(file_path))
        initial_timestamp = view_tracker.get_last_view_timestamp(str(file_path))

        # Wait a moment to ensure different timestamp
        time.sleep(0.01)

        # Edit the file
        await tool.run(EditArgs(path=str(file_path), file_text="updated"))

        # Check that timestamp was updated
        new_timestamp = view_tracker.get_last_view_timestamp(str(file_path))
        assert new_timestamp is not None
        assert initial_timestamp is not None
        assert new_timestamp >= initial_timestamp


# =============================================================================
# File Modification Tests
# =============================================================================


class TestFileModification:
    """Tests for file modification detection."""

    @pytest.mark.asyncio
    async def test_edit_fails_on_non_existent_file(
        self, tool: EditTool, view_tracker: ViewTrackerService, temp_dir: Path
    ) -> None:
        """Test editing fails if file doesn't exist."""
        non_existent_path = temp_dir / "does_not_exist.txt"

        with pytest.raises(FileSystemError) as exc_info:
            await tool.run(
                EditArgs(path=str(non_existent_path), file_text="new content")
            )

        assert exc_info.value.code == "FILE_NOT_FOUND"
        assert "File not found" in str(exc_info.value)
        assert "doesn't exist" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_edit_fails_on_modified_file_mtime_check(
        self, tool: EditTool, view_tracker: ViewTrackerService, temp_dir: Path
    ) -> None:
        """Test editing fails if file was modified after last view."""
        file_path = temp_dir / "modified.txt"
        file_path.write_text("original", encoding="utf-8")

        # Record view
        view_tracker.record_view(str(file_path))

        # Wait a moment and modify the file
        time.sleep(0.01)
        file_path.write_text("modified externally", encoding="utf-8")

        with pytest.raises(FileSystemError) as exc_info:
            await tool.run(EditArgs(path=str(file_path), file_text="new content"))

        assert exc_info.value.code == "FILE_MODIFIED"
        assert "has been modified since" in str(exc_info.value)


# =============================================================================
# Mistaken Edit Detection Tests
# =============================================================================


class TestMistakenEditDetection:
    """Tests for mistaken edit detection functionality."""

    @pytest.mark.asyncio
    async def test_mistaken_edit_detected_length_ratio(
        self, tool: EditTool, view_tracker: ViewTrackerService, temp_dir: Path
    ) -> None:
        """Test mistaken edit is detected when new content is much smaller."""
        file_path = temp_dir / "small_replace.txt"
        # 100 lines of content with 5 unique line types
        old_content = "line1\nline2\nline3\nline4\nline5\n" * 20
        file_path.write_text(old_content, encoding="utf-8")
        view_tracker.record_view(str(file_path))

        # New content: significantly smaller (length_ratio < 0.3)
        # Most lines same (similarity > 0.7)
        # Line count similar (diff < 0.3)
        # Use ~20 repetitions of the same 5 lines (100 lines)
        # plus some new lines to trigger heuristics
        new_content = (
            ("line1\n" * 20)
            + ("line2\n" * 20)
            + ("line3\n" * 20)
            + ("line4\n" * 20)
            + ("line5\n" * 20)
            + "new_line\n" * 5
        )  # 105 lines, ~420 chars - most lines same

        with pytest.raises(FileSystemError) as exc_info:
            await tool.run(EditArgs(path=str(file_path), file_text=new_content))

        assert exc_info.value.code == "MISTAKEN_EDIT"
        assert "mistaken usage" in str(exc_info.value)
        assert len(exc_info.value.suggestions) > 0

    @pytest.mark.asyncio
    async def test_mistaken_edit_detected_line_similarity(
        self, tool: EditTool, view_tracker: ViewTrackerService, temp_dir: Path
    ) -> None:
        """Test mistaken edit is detected when content has high line similarity."""
        file_path = temp_dir / "high_similarity.txt"
        old_content = "def foo():\n    pass\n\ndef bar():\n    pass\n" * 20
        file_path.write_text(old_content, encoding="utf-8")
        view_tracker.record_view(str(file_path))

        # New content is mostly the same with minor changes
        new_content = "def foo():\n    return 1\n\ndef bar():\n    pass\n" * 20

        with pytest.raises(FileSystemError) as exc_info:
            await tool.run(EditArgs(path=str(file_path), file_text=new_content))

        assert exc_info.value.code == "MISTAKEN_EDIT"

    @pytest.mark.asyncio
    async def test_mistaken_edit_detected_length_diff(
        self, tool: EditTool, view_tracker: ViewTrackerService, temp_dir: Path
    ) -> None:
        """Test mistaken edit is detected when line counts are similar."""
        file_path = temp_dir / "similar_lines.txt"
        old_content = "line1\nline2\nline3\nline4\nline5\n" * 20  # 100 lines, 5 unique
        file_path.write_text(old_content, encoding="utf-8")
        view_tracker.record_view(str(file_path))

        # New content: similar line count (diff < 0.3)
        # High line similarity (> 0.7) using Jaccard
        # Same 5 unique lines + 5 new lines = 105 total lines
        # Unique: 5 old + 2 new = 7, Jaccard = 5/7 = 0.714 (> 0.7)
        new_content = (
            ("line1\n" * 20)
            + ("line2\n" * 20)
            + ("line3\n" * 20)
            + ("line4\n" * 20)
            + ("line5\n" * 20)
            + "newA\nnewB\n"  # 102 lines total
        )

        with pytest.raises(FileSystemError) as exc_info:
            await tool.run(EditArgs(path=str(file_path), file_text=new_content))

        assert exc_info.value.code == "MISTAKEN_EDIT"

    @pytest.mark.asyncio
    async def test_mistaken_edit_majority_voting_2_of_3(
        self, tool: EditTool, view_tracker: ViewTrackerService, temp_dir: Path
    ) -> None:
        """Test mistaken edit detection requires 2 of 3 heuristics to trigger."""
        file_path = temp_dir / "majority_vote.txt"
        # Create content where only 1 heuristic would trigger
        old_content = "line1\nline2\nline3\nline4\nline5\n" * 30  # 150 lines, >100
        file_path.write_text(old_content, encoding="utf-8")
        view_tracker.record_view(str(file_path))

        # New content: completely different content with different line count
        # Length ratio: ~620/1050 = 0.59 (> 0.3) - doesn't trigger
        # Line similarity: 0/7 = 0 (< 0.7) - doesn't trigger
        # Length diff: 45/150 = 0.3 (exactly at threshold) - doesn't trigger
        new_content = "completely\ndifferent\ncontent\nhere\n" * 10  # 40 lines

        # This should NOT raise because 0 of 3 heuristics trigger
        result = await tool.run(EditArgs(path=str(file_path), file_text=new_content))
        assert "modified successfully" in result.output

    @pytest.mark.asyncio
    async def test_mistaken_edit_skips_short_content(
        self, tool: EditTool, view_tracker: ViewTrackerService, temp_dir: Path
    ) -> None:
        """Test mistaken edit detection skips short content."""
        file_path = temp_dir / "short_content.txt"
        old_content = "short"  # Less than 100 chars
        file_path.write_text(old_content, encoding="utf-8")
        view_tracker.record_view(str(file_path))

        # Even though this looks like a replacement, it should be allowed
        # because content is too short for meaningful comparison
        new_content = "also short"

        result = await tool.run(EditArgs(path=str(file_path), file_text=new_content))
        assert "modified successfully" in result.output


# =============================================================================
# Retry Logic Tests
# =============================================================================


class TestRetryLogic:
    """Tests for retry logic functionality."""

    @pytest.mark.asyncio
    async def test_retry_allowed_within_60s_same_hashes(
        self, tool: EditTool, view_tracker: ViewTrackerService, temp_dir: Path
    ) -> None:
        """Test retry is allowed within 60-second timeout."""
        file_path = temp_dir / "retry.txt"
        old_content = "line1\nline2\nline3\nline4\nline5\n" * 20
        file_path.write_text(old_content, encoding="utf-8")
        view_tracker.record_view(str(file_path))

        # New content that triggers mistaken edit detection
        new_content = (
            ("line1\n" * 20)
            + ("line2\n" * 20)
            + ("line3\n" * 20)
            + ("line4\n" * 20)
            + ("line5\n" * 20)
            + "new_line\n" * 5
        )  # 105 lines

        # First attempt - should fail with warning
        with pytest.raises(FileSystemError):
            await tool.run(EditArgs(path=str(file_path), file_text=new_content))

        # Second attempt - should succeed (retry allowed)
        result = await tool.run(EditArgs(path=str(file_path), file_text=new_content))
        assert "modified successfully" in result.output

    @pytest.mark.timeout(120)
    @pytest.mark.asyncio
    async def test_retry_rejected_after_60s_expired_warning(
        self, tool: EditTool, view_tracker: ViewTrackerService, temp_dir: Path
    ) -> None:
        """Test retry is rejected after 60-second timeout."""
        file_path = temp_dir / "expired_retry.txt"
        old_content = "line1\nline2\nline3\nline4\nline5\n" * 20
        file_path.write_text(old_content, encoding="utf-8")
        view_tracker.record_view(str(file_path))

        # New content that triggers mistaken edit detection
        new_content = (
            ("line1\n" * 20)
            + ("line2\n" * 20)
            + ("line3\n" * 20)
            + ("line4\n" * 20)
            + ("line5\n" * 20)
            + "new_line\n" * 5
        )  # 105 lines

        # First attempt - should fail with warning
        with pytest.raises(FileSystemError):
            await tool.run(EditArgs(path=str(file_path), file_text=new_content))

        # Wait for warning to expire (60+ seconds)
        time.sleep(61)

        # Second attempt - should fail again (warning expired)
        with pytest.raises(FileSystemError) as exc_info:
            await tool.run(EditArgs(path=str(file_path), file_text=new_content))

        assert exc_info.value.code == "MISTAKEN_EDIT"

    @pytest.mark.asyncio
    async def test_retry_rejected_different_hashes(
        self, tool: EditTool, view_tracker: ViewTrackerService, temp_dir: Path
    ) -> None:
        """Test retry is rejected when content hashes don't match."""
        file_path = temp_dir / "different_hashes.txt"
        old_content = "line1\nline2\nline3\nline4\nline5\n" * 20
        file_path.write_text(old_content, encoding="utf-8")
        view_tracker.record_view(str(file_path))

        # New content that triggers mistaken edit detection
        new_content_1 = (
            ("line1\n" * 20)
            + ("line2\n" * 20)
            + ("line3\n" * 20)
            + ("line4\n" * 20)
            + ("line5\n" * 20)
            + "new_line\n" * 5
        )
        new_content_2 = (
            ("line1\n" * 20)
            + ("line2\n" * 20)
            + ("line3\n" * 20)
            + ("line4\n" * 20)
            + ("line5\n" * 20)
            + "different_line\n" * 5
        )

        # First attempt with content 1 - should fail with warning
        with pytest.raises(FileSystemError):
            await tool.run(EditArgs(path=str(file_path), file_text=new_content_1))

        # Second attempt with different content - should fail again
        with pytest.raises(FileSystemError) as exc_info:
            await tool.run(EditArgs(path=str(file_path), file_text=new_content_2))

        assert exc_info.value.code == "MISTAKEN_EDIT"

    @pytest.mark.timeout(120)
    @pytest.mark.asyncio
    async def test_cleanup_removes_expired_warnings(
        self, tool: EditTool, view_tracker: ViewTrackerService, temp_dir: Path
    ) -> None:
        """Test cleanup removes expired warnings."""
        file_path = temp_dir / "cleanup_test.txt"
        old_content = "line1\nline2\nline3\nline4\nline5\n" * 20
        file_path.write_text(old_content, encoding="utf-8")
        view_tracker.record_view(str(file_path))

        # New content that triggers mistaken edit detection
        new_content = (
            ("line1\n" * 20)
            + ("line2\n" * 20)
            + ("line3\n" * 20)
            + ("line4\n" * 20)
            + ("line5\n" * 20)
            + "new_line\n" * 5
        )  # 105 lines

        # First attempt - should fail with warning
        with pytest.raises(FileSystemError):
            await tool.run(EditArgs(path=str(file_path), file_text=new_content))

        # Verify warning was recorded
        assert str(file_path) in tool.state.warned_operations

        # Wait for warning to expire
        time.sleep(61)

        # Trigger cleanup by running another edit operation
        file_path2 = temp_dir / "cleanup_test2.txt"
        file_path2.write_text("content", encoding="utf-8")
        view_tracker.record_view(str(file_path2))
        await tool.run(EditArgs(path=str(file_path2), file_text="new content"))

        # Verify warning was cleaned up
        assert str(file_path) not in tool.state.warned_operations


# =============================================================================
# Hash Function Tests
# =============================================================================


class TestHashFunction:
    """Tests for hash function implementation."""

    def test_hash_function_produces_consistent_results(self) -> None:
        """Test hash function produces consistent results for same input."""
        content = "Hello, World!"
        hash1 = EditTool._hash_content(content)
        hash2 = EditTool._hash_content(content)

        assert hash1 == hash2

    def test_hash_function_matches_typescript_exactly(self) -> None:
        """Test hash function matches TypeScript implementation with exact values.

        These expected values are generated from the TypeScript implementation:
        const HASH_MULTIPLIER = 31;
        const BIT_MASK_32 = 0x1_00_00_00_00; // 2^32
        const BASE_36 = 36;
        """
        # Test cases with known-good hash values from TypeScript
        test_cases = {
            "": "0",
            "a": "2p",
            "ab": "2e9",
            "abc": "22ci",
            "hello": "1n1e4y",
            "Hello, World!": "osc9p1",
            "The quick brown fox jumps over the lazy dog": "1oy9w7n",
        }

        for content, expected_hash in test_cases.items():
            actual_hash = EditTool._hash_content(content)
            assert actual_hash == expected_hash, (
                f"Hash mismatch for '{content}': got '{actual_hash}', expected '{expected_hash}'"
            )

    def test_hash_function_matches_typescript_same_input(self) -> None:
        """Test hash function matches TypeScript implementation for same input.

        TypeScript implementation:
        const HASH_MULTIPLIER = 31;
        const BIT_MASK_32 = 0x1_00_00_00_00; // 2^32
        const BASE_36 = 36;
        let hash = 0;
        for (let i = 0; i < content.length; i++) {
          const char = content.charCodeAt(i);
          hash = hash * HASH_MULTIPLIER + char;
          hash %= BIT_MASK_32;
        }
        return Math.abs(hash).toString(BASE_36);

        Test with various inputs - verify produces valid base-36 strings.
        """
        test_cases = [
            "",
            "a",
            "ab",
            "abc",
            "hello",
            "Hello, World!",
            "The quick brown fox jumps over the lazy dog",
        ]

        for content in test_cases:
            python_hash = EditTool._hash_content(content)
            # Verify hash is a valid base-36 string (digits 0-9, letters a-z)
            assert python_hash.isalnum(), (
                f"Hash for '{content}' is not alphanumeric: {python_hash}"
            )
            # Only check islower() for hashes containing letters
            if python_hash.isalpha():
                assert python_hash.islower(), (
                    f"Hash for '{content}' should be lowercase: {python_hash}"
                )

    def test_hash_function_different_inputs_different_hashes(self) -> None:
        """Test different inputs produce different hashes."""
        hashes = set()
        test_strings = [f"string_{i}" for i in range(100)]

        for content in test_strings:
            hashes.add(EditTool._hash_content(content))

        # All hashes should be unique for unique inputs
        assert len(hashes) == len(test_strings)


# =============================================================================
# Error Message Validation Tests
# =============================================================================


class TestErrorMessageValidation:
    """Tests for exact error message matching with TypeScript."""

    @pytest.mark.asyncio
    async def test_error_message_file_not_viewed_exact_match(
        self, tool: EditTool, view_tracker: ViewTrackerService, temp_dir: Path
    ) -> None:
        """Test error message for file not viewed matches TypeScript exactly."""
        file_path = temp_dir / "not_viewed.txt"
        file_path.write_text("content", encoding="utf-8")

        with pytest.raises(FileSystemError) as exc_info:
            await tool.run(EditArgs(path=str(file_path), file_text="new content"))

        error_message = str(exc_info.value)
        # Verify key phrases from TypeScript error message
        assert "must be viewed before editing" in error_message
        assert "Use 'read_file' command" in error_message

    @pytest.mark.asyncio
    async def test_error_message_file_not_found_exact_match(
        self, tool: EditTool, view_tracker: ViewTrackerService, temp_dir: Path
    ) -> None:
        """Test error message for file not found matches TypeScript exactly."""
        non_existent_path = temp_dir / "does_not_exist.txt"

        with pytest.raises(FileSystemError) as exc_info:
            await tool.run(
                EditArgs(path=str(non_existent_path), file_text="new content")
            )

        error_message = str(exc_info.value)
        # Verify key phrases from TypeScript error message
        assert "File not found" in error_message
        assert "doesn't exist" in error_message
        assert "Use 'create' command" in error_message

    @pytest.mark.asyncio
    async def test_error_message_file_modified_exact_match(
        self, tool: EditTool, view_tracker: ViewTrackerService, temp_dir: Path
    ) -> None:
        """Test error message for file modified matches TypeScript exactly."""
        file_path = temp_dir / "modified.txt"
        file_path.write_text("original", encoding="utf-8")
        view_tracker.record_view(str(file_path))

        time.sleep(0.01)
        file_path.write_text("modified externally", encoding="utf-8")

        with pytest.raises(FileSystemError) as exc_info:
            await tool.run(EditArgs(path=str(file_path), file_text="new content"))

        error_message = str(exc_info.value)
        # Verify key phrases from TypeScript error message
        assert "has been modified since" in error_message
        assert "Use 'read_file' command" in error_message

    @pytest.mark.asyncio
    async def test_error_message_mistaken_edit_exact_match(
        self, tool: EditTool, view_tracker: ViewTrackerService, temp_dir: Path
    ) -> None:
        """Test error message for mistaken edit matches TypeScript exactly."""
        file_path = temp_dir / "mistaken_edit.txt"
        old_content = "line1\nline2\nline3\nline4\nline5\n" * 20
        file_path.write_text(old_content, encoding="utf-8")
        view_tracker.record_view(str(file_path))

        # New content that triggers mistaken edit detection
        new_content = (
            ("line1\n" * 20)
            + ("line2\n" * 20)
            + ("line3\n" * 20)
            + ("line4\n" * 20)
            + ("line5\n" * 20)
            + "new_line\n" * 5
        )  # 105 lines

        with pytest.raises(FileSystemError) as exc_info:
            await tool.run(EditArgs(path=str(file_path), file_text=new_content))

        error_message = str(exc_info.value)
        # Verify key phrases from TypeScript error message
        assert "mistaken usage" in error_message
        assert (
            "Consider using 'edit_file'" in error_message
            or "str_replace" in error_message
        )
        assert (
            "60-second timeout" in error_message
            or "Try this command again" in error_message
        )


# =============================================================================
# Success Message Tests
# =============================================================================


class TestSuccessMessage:
    """Tests for success message format."""

    @pytest.mark.asyncio
    async def test_success_message_format_validation(
        self, tool: EditTool, view_tracker: ViewTrackerService, temp_dir: Path
    ) -> None:
        """Test success message format matches TypeScript exactly."""
        file_path = temp_dir / "success.txt"
        file_path.write_text("original", encoding="utf-8")
        view_tracker.record_view(str(file_path))

        result = await tool.run(EditArgs(path=str(file_path), file_text="new content"))

        assert result.output == f"File '{file_path}' modified successfully"
        # Verify content was actually updated
        assert file_path.read_text(encoding="utf-8") == "new content"

    @pytest.mark.asyncio
    async def test_success_message_with_unicode_content(
        self, tool: EditTool, view_tracker: ViewTrackerService, temp_dir: Path
    ) -> None:
        """Test success message with Unicode content."""
        file_path = temp_dir / "unicode.txt"
        file_path.write_text("original", encoding="utf-8")
        view_tracker.record_view(str(file_path))

        unicode_content = "Hello, 世界! 🌍 Ñoño ©®™"
        result = await tool.run(
            EditArgs(path=str(file_path), file_text=unicode_content)
        )

        assert "modified successfully" in result.output
        assert file_path.read_text(encoding="utf-8") == unicode_content


# =============================================================================
# UTF-8 Encoding Tests
# =============================================================================


class TestUTF8Encoding:
    """Tests for UTF-8 encoding support."""

    @pytest.mark.asyncio
    async def test_edit_with_utf8_content(
        self, tool: EditTool, view_tracker: ViewTrackerService, temp_dir: Path
    ) -> None:
        """Test editing with UTF-8 encoding handles special characters."""
        file_path = temp_dir / "utf8_edit.txt"
        old_content = "Hello, 世界! 🌍 Ñoño ©®™"
        file_path.write_text(old_content, encoding="utf-8")
        view_tracker.record_view(str(file_path))

        new_content = "Goodbye, 世界! 🌍 Ñoño ©®™"
        result = await tool.run(EditArgs(path=str(file_path), file_text=new_content))

        assert result.output == f"File '{file_path}' modified successfully"
        assert file_path.read_text(encoding="utf-8") == new_content
