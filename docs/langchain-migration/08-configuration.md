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

## ⚠️ Critical Issues Found (Post-Migration Audit)

1. **interrupt_before Parameter Removed** ✅
   - `interrupt_before=["tools"]` must be deleted from `create_agent()` call
   - Conflicts with `HumanInTheLoopMiddleware` (two competing interrupt mechanisms)
   - Defeats permission system (ALWAYS tools still pause)
   - **Fix:** Delete line 188 from `langchain_engine.py` (see Priority 0 in 00-overview.md)
   - **Impact:** Critical - Breaks HITL approval system

2. **ContextWarningMiddleware Token Tracking** ⚠️
   - Currently returns per-message `usage_metadata["total_tokens"]` instead of cumulative
   - Example: After 2 turns with 1000 + 1500 tokens, returns 1500 instead of 2500
   - Warnings trigger at wrong percentages (30% instead of 50%)
   - **Fix:** Add `_cumulative_tokens` tracking in `after_model` hook (see Priority 1 in 00-overview.md)
   - **Impact:** High - Incorrect context warnings, potential budget overruns

3. **Missing Async Middleware Variants** ⚠️
   - All middlewares only implement sync hooks
   - Will crash if LangGraph calls async variants during async execution
   - **Fix:** Implement `abefore_model`, `aafter_model`, etc. for all middleware (see Priority 1 in 00-overview.md)
   - **Impact:** High - Crashes on async execution path

4. **VibeAgentState Missing LangGraph Type Annotations** ⚠️
   - Missing `NotRequired`, `EphemeralValue`, `PrivateStateAttr`, `OmitFromInput`, `OmitFromOutput`
   - Breaks LangGraph state management behavior
   - Warnings persist incorrectly across graph steps (not reset between turns)
   - **Fix:** Add proper LangGraph annotations (see Priority 1 in 00-overview.md)
   - **Impact:** Medium - State may behave unexpectedly

5. **PriceLimitMiddleware Model Name Lookup Bug** ⚠️
   - Uses `state.get("model_name", "default")` instead of constructor `self.model_name`
   - `model_name` is NOT in standard `AgentState` schema
   - Always looks up "default" pricing key (may not exist)
   - Price limits never trigger (rates = 0.0, 0.0)
   - **Fix:** Use `self.model_name` from constructor in `after_model` (see Priority 1 in 00-overview.md)
   - **Impact:** Medium - Price limits don't work

**Action Required:**
- Fix these 5 issues before production deployment
- Estimated effort: 2-3 hours for Priority 0-1 issues
- See 00-overview.md for detailed priority action plan
