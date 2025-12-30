# Development Commands

## Package Management (Always use uv)
```bash
# Install dependencies
uv sync

# Install all extras (including build dependencies)
uv sync --all-extras

# Add/remove dependencies
uv add <package>
uv remove <package>

# Run any Python tool within the uv environment
uv run pytest
uv run ruff check
uv run pyright
uv run python script.py
```

## Linting & Formatting
```bash
# Check and fix linting issues
uv run ruff check --fix --unsafe-fixes

# Format code
uv run ruff format --check  # Check only
uv run ruff format           # Format in-place

# Run type checker
uv run pyright

# Run all pre-commit hooks manually
uv run pre-commit run --all-files

# Run spell check
uv run typos --write-changes
```

## Testing
```bash
# Run all tests (excluding snapshots)
uv run pytest --ignore tests/snapshots

# Run all tests including snapshots
uv run pytest

# Run snapshot tests only
uv run pytest tests/snapshots

# Run performance benchmarks
uv run pytest tests/performance/test_benchmarks.py

# Run single test file
uv run pytest tests/unit/test_config.py

# Run single test function
uv run pytest tests/unit/test_config.py::test_config_creation

# Run tests in verbose mode: uv run pytest -v
# Stop on first failure: uv run pytest -x
# Run tests without parallel execution: uv run pytest -n 0
```

## CLI Verification
```bash
# Verify CLI tools work correctly
uv run vibe --help
uv run vibe-acp --help
```

## Nix Flake (Alternative to uv)
```bash
# Enter Nix development environment
nix develop

# Run commands in Nix environment
nix develop --command uv run pytest
nix develop --command vibe --help

# Check flake
nix flake check
```

# Code Style Guidelines

## Python Version
- Use Python 3.12+ features
- Always start files with: `from __future__ import annotations`

## Type Hints
- Use built-in generics: `list[str]`, `dict[str, int]` (not `List`, `Dict`)
- Use pipe operator for unions: `int | None` (not `Optional[int]`)
- All public functions and classes must have type annotations
- No inline `# type: ignore` or `# noqa` - fix at source instead

## Formatting (Ruff)
- Line length: 88 characters
- Ban relative imports: use absolute imports from project root
- Known first-party package: `vibe`
- Combine `from foo import bar, baz` on same line
- Import third-party before local imports when possible

## Naming Conventions
- Classes: `PascalCase` (e.g., `VibeEngine`, `VibeConfig`)
- Functions/variables: `snake_case` (e.g., `create_engine`, `user_name`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `MAX_TOKENS`)
- Private members: `_leading_underscore` (e.g., `_agent`, `_config`)
- Enum members: `UPPERCASE` with `auto()` for values

## Modern Python Syntax
- Use `match-case` for pattern matching instead of if/elif/else
- Use walrus operator `:=` when assignment + conditional simplifies code
- Use f-strings for all string formatting
- Use comprehensions for building lists/dicts

## File Operations
- Always use `pathlib.Path` instead of `os.path`
- Path operations: `path.read_text()`, `path.write_text()`, `path.exists()`

## Modern Enum Usage
- Use StrEnum for string-based enums, IntEnum/IntFlag for integer-based
- Use auto() for automatic value assignment, always UPPERCASE for members
- Use @property for computed attributes rather than storing values

## Pydantic v2
- Use `model_validate()` for external data parsing
- Use `field_validator` instead of `@validator` decorator
- Use `BaseModel.model_dump()` instead of `.dict()`
- Use `from_attributes=True` for converting objects to models
- Keep normalization logic inside model validators
- Prefer thin wrappers over custom constructors like `from_sdk`

## Pydantic Discriminated Unions
- Don't narrow a field type in a subclass (violates Liskov substitution)
- Use sibling classes plus a shared mixin for common fields
- Compose union with `Annotated[Union[...], Field(discriminator='transport')]`
- Use `match` on discriminator to narrow types at call sites

## Error Handling
- Define custom exceptions inheriting from `RuntimeError` or `ValueError`
- Include relevant attributes in exceptions (e.g., `self.env_key`)
- Don't document obvious built-in exceptions (TypeError, ValueError)
- Only document exceptions explicitly raised by your function
- For public APIs, document exceptions from called functions if they propagate

## Code Structure
- "Never nester": avoid deep nesting, use early returns/guard clauses
- Keep functions small: max 50 statements (ruff PLR0915)
- Max 15 branches, 15 locals, 9 args, 6 returns
- Extract helpers to reduce complexity

## Testing
- Use pytest fixtures in `tests/conftest.py` for shared setup
- Test directories: `tests/unit/`, `tests/integration/`, `tests/snapshots/`, `tests/performance/`
- Use descriptive test names: `test_function_name_scenario`
- Mock external dependencies (API calls, file I/O)

## Documentation
- Docstrings for all public classes and functions
- Use Google-style docstrings (not NumPy or reST)
- Keep exception documentation minimal and accurate

## Development Workflow
1. Make changes
2. Run linting: `uv run ruff check --fix --unsafe-fixes && uv run ruff format`
3. Run type checking: `uv run pyright`
4. Run tests: `uv run pytest -x` (stop on first failure)

---

# Implementation Notes

## Nix FHS Environment for Testing

Some tests require the FHS (Filesystem Hierarchy Standard) environment because dependencies like `tokenizers` (from Hugging Face) require `libstdc++.so.6` which isn't available in the default Nix shell.

```bash
# Run tests requiring FHS environment
nix develop .#fhs --command python -m pytest tests/tools/filesystem/test_list_files.py -v

# Or with uv
nix develop .#fhs --command uv run pytest tests/tools/filesystem/test_list_files.py
```

## Tool Design Pattern: find_files vs list Modes

When implementing filesystem tools that support both file discovery (find_files) and directory listing (list), design for dual modes based on input:

- **With patterns** → find_files mode (glob-based file discovery)
- **Without patterns** → list mode (directory contents)

Example from `ListFilesTool`:
```python
if args.patterns:
    output = await self._find_files(resolved_path, args)
else:
    output = await self._list(resolved_path, args)
```

## Pattern Matching: pathlib vs fnmatch

- Use `Path.glob()` and `Path.rglob()` for basic patterns (`*`, `**`)
- Use `fnmatch.fnmatch()` for advanced patterns (`?`, `[abc]`)
- For recursive patterns with `**`, process the pattern specially if needed

```python
if "**" in pattern:
    parts = pattern.split("**")
    if len(parts) == self._DOUBLE_STAR_PARTS_COUNT:
        base_pattern = parts[1].lstrip("/")
        return fnmatch(file_name, base_pattern)
```

## Recursive vs Non-Recursive List Output

The list command has different output formats based on `recursive` flag:

- **`recursive=False`**: Simple list with `[DIR]`/`[FILE]` markers and file metadata
- **`recursive=True`**: Tree structure with indentation, no markers

## Test File Organization

For new tool packages, create a dedicated conftest.py in the test directory to avoid importing heavy dependencies from the root conftest.py:

```python
# tests/tools/filesystem/conftest.py
import pytest
from pathlib import Path
import tempfile

from vibe.core.tools.filesystem.list_files import (
    ListFilesTool, ListFilesArgs, ListFilesToolConfig, ListFilesToolState
)

@pytest.fixture
def temp_dir() -> Path:
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

# ... other fixtures
```

## Path Resolution Pattern

Always resolve paths against `config.effective_workdir` and validate:

```python
def _resolve_path(self, path: str) -> Path:
    if Path(path).is_absolute():
        return Path(path).resolve()
    else:
        return (self.config.effective_workdir / path).resolve()
```
