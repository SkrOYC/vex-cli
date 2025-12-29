# 08 - Configuration Preservation

## Overview

Verify that all existing configuration continues to work with LangChain 1.2.0, with no changes required to the configuration system.

## Configuration Comparison

| Aspect | Before | After | Status |
|--------|--------|-------|--------|
| **Config File** | `pyproject.toml` | `pyproject.toml` | ✅ No change |
| **Environment Variables** | `VIBE_*` | `VIBE_*` | ✅ No change |
| **Tool Permissions** | Per-tool config | Per-tool config | ✅ No change |
| **Model Configuration** | `models` list | `models` list | ✅ No change |
| **Provider Configuration** | `providers` list | `providers` list | ✅ No change |
| **MCP Servers** | `mcp_servers` list | `mcp_servers` list | ✅ No change |
| **DeepAgents Flags** | `use_deepagents` | REMOVE | 🗑️ To remove |
| **Feature Flags** | Various | Various | ✅ No change |

## Configuration to Remove

```python
# In vibe/core/config.py

class VibeConfig(BaseSettings):
    # REMOVE these DeepAgents-specific flags:
    use_deepagents: bool = Field(default=False)
    enable_subagents: bool = Field(default=True)
    enable_planning: bool = Field(default=True)
    
    # KEEP these general flags:
    context_warnings: bool = Field(default=True)
    max_price: float | None = Field(default=None)
    auto_compact_threshold: int = Field(default=170000)
```

## Feature Flag Migration

```python
# Before: Feature flag in pyproject.toml or .env
VIBE_USE_DEEPAGENTS=true

# After: Use LangChain by default (no flag needed)
# Can add flag for testing during migration:
VIBE_USE_LANGCHAIN=false  # Only for rollback during migration
```

## Configuration Preservation Checklist

- [ ] All existing TOML configs work
- [ ] Environment variables are preserved
- [ ] Tool permissions work
- [ ] Model/provider configs work
- [ ] MCP server configs work
- [ ] DeepAgents flags are removed
- [ ] New feature flags are added if needed
