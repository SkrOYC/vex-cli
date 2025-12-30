# Development Commands

**All commands must run under `nix develop .#fhs --command`**

## Package Management
```bash
nix develop .#fhs --command uv sync                    # Install dependencies
nix develop .#fhs --command uv sync --all-extras       # Install all extras
nix develop .#fhs --command uv add <package>           # Add dependency
nix develop .#fhs --command uv remove <package>        # Remove dependency
nix develop .#fhs --command uv run <command>           # Run any tool
```

## Linting & Formatting
```bash
nix develop .#fhs --command uv run ruff check --fix --unsafe-fixes  # Lint + fix
nix develop .#fhs --command uv run ruff format                      # Format code
nix develop .#fhs --command uv run pyright                          # Type check
nix develop .#fhs --command uv run pre-commit run --all-files       # Run hooks
nix develop .#fhs --command uv run typos --write-changes            # Spell check
```

## Testing
```bash
nix develop .#fhs --command uv run pytest                          # All tests
nix develop .#fhs --command uv run pytest --ignore tests/snapshots  # Exclude snapshots
nix develop .#fhs --command uv run pytest tests/snapshots          # Snapshots only
nix develop .#fhs --command uv run pytest tests/performance/test_benchmarks.py  # Benchmarks
nix develop .#fhs --command uv run pytest <file>::<test>           # Single test
nix develop .#fhs --command uv run pytest -v -x -n 0               # Verbose, stop on fail, no parallel
```

## CLI Verification
```bash
nix develop .#fhs --command uv run vibe --help
nix develop .#fhs --command uv run vibe-acp --help
```

## Nix Operations
```bash
nix develop .#fhs                              # Enter interactive env
nix flake check                                # Check flake
```

# Code Style Guidelines

- **Python 3.12+**: Start files with `from __future__ import annotations`
- **Type hints**: `list[str]`, `dict[str, int]`, `int | None`; annotate all public APIs; no inline `# type: ignore`
- **Ruff**: 88 char lines; absolute imports only; known first-party: `vibe`
- **Naming**: Classes `PascalCase`, functions/vars `snake_case`, constants `UPPER_SNAKE_CASE`, private `_underscore`, enums `UPPERCASE`
- **Modern syntax**: `match-case`, walrus `:=`, f-strings, comprehensions
- **Paths**: Always use `pathlib.Path`, not `os.path`
- **Enums**: StrEnum for strings, IntEnum/IntFlag for ints; use `auto()`, UPPERCASE members; use `@property` for computed
- **Pydantic v2**: `model_validate()`, `field_validator`, `model_dump()`, `from_attributes=True`
- **Pydantic unions**: Sibling classes + mixin; `Annotated[Union[...], Field(discriminator='transport')]`
- **Errors**: Custom from `RuntimeError`/`ValueError`; include attrs; only document explicitly raised exceptions
- **Structure**: Early returns; max 50 statements, 15 branches, 15 locals, 9 args, 6 returns
- **Testing**: Fixtures in `tests/conftest.py`; mock externals; descriptive names
- **Docs**: Google-style docstrings for public APIs

**Workflow**: Make changes → lint → type check → `pytest -x`

# Implementation Notes

## FHS Environment Requirement
All tests and commands require `libstdc++.so.6` which is only available in FHS environment.
