# 08 - Configuration System Preservation

## Overview

Preserve Mistral Vibe's sophisticated configuration system while adapting it to work with DeepAgents' simpler parameter-based approach.

## Current Configuration System

### Complex TOML-Based Config

```python
# vibe/core/config.py (~400 lines)

class VibeConfig(BaseSettings):
    """Comprehensive configuration with TOML loading."""
    
    # Core settings
    active_model: str = "devstral-2"
    auto_compact_threshold: int = 200_000
    context_warnings: bool = False
    
    # Model configurations
    models: list[ModelConfig] = Field(default_factory=lambda: list(DEFAULT_MODELS))
    
    # Provider configurations  
    providers: list[ProviderConfig] = Field(default_factory=lambda: list(DEFAULT_PROVIDERS))
    
    # Tool configurations
    tools: dict[str, BaseToolConfig] = Field(default_factory=dict)
    
    # Project detection
    workdir: Path | None = Field(default=None, exclude=True)
    
    # Complex loading logic
    @classmethod
    def load(cls, workdir: Path | None = None) -> VibeConfig:
        # Load global config
        # Load project config
        # Merge configurations
        # Validate settings
        pass
```

### Hierarchical Loading

```python
# Configuration loading order:
# 1. Global config (~/.vibe/config.toml)
# 2. Project config (.vibe/config.toml) 
# 3. Environment variables
# 4. Command line overrides
```

### Project-Aware Features

```python
# Automatic project detection via .git
# Project-specific model settings
# Tool permission overrides per project
# Context-aware configuration
```

## DeepAgents Configuration Approach

### Simple Parameters

DeepAgents takes simple parameters to `create_deep_agent()`:

```python
agent = create_deep_agent(
    model=ChatMistralAI(...),      # LangChain model
    tools=[...],                   # List of tools
    system_prompt="...",           # System prompt
    middleware=[...],              # Middleware stack
    backend=FilesystemBackend(...), # File backend
    interrupt_on={...},            # Approval config
    checkpointer=InMemorySaver(),  # State persistence
)
```

No built-in configuration file system.

## Migration Strategy

### Phase 1: Preserve VibeConfig

Keep the existing configuration system intact:

```python
# vibe/core/config.py - PRESERVED ENTIRELY

class VibeConfig(BaseSettings):
    # All existing fields and methods preserved
    active_model: str = "devstral-2"
    auto_compact_threshold: int = 200_000
    context_warnings: bool = False
    models: list[ModelConfig] = Field(default_factory=lambda: list(DEFAULT_MODELS))
    providers: list[ProviderConfig] = Field(default_factory=lambda: list(DEFAULT_PROVIDERS))
    tools: dict[str, BaseToolConfig] = Field(default_factory=dict)
    workdir: Path | None = Field(default=None, exclude=True)
    
    # All existing methods preserved
    def get_model_config(self) -> ModelConfig: ...
    def get_provider_config(self, provider: str) -> ProviderConfig: ...
    def get_tool_config(self, tool_name: str) -> BaseToolConfig: ...
    def effective_workdir(self) -> Path: ...
```

### Phase 2: Configuration Bridge

Create a bridge that converts VibeConfig to DeepAgents parameters:

```python
# vibe/core/engine/config_bridge.py

from typing import Any
from langchain.agents.middleware import InterruptOnConfig
from langchain_core.language_models import BaseChatModel

from deepagents.backends import FilesystemBackend
from deepagents.middleware import FilesystemMiddleware

from vibe.core.config import VibeConfig
from vibe.core.engine.models import create_model_from_config


class DeepAgentsConfig:
    """Bridge between VibeConfig and DeepAgents parameters."""
    
    @staticmethod
    def create_model(config: VibeConfig) -> BaseChatModel:
        """Create LangChain model from Vibe config."""
        return create_model_from_config(config)
    
    @staticmethod
    def create_backend(config: VibeConfig) -> FilesystemBackend:
        """Create filesystem backend from Vibe config."""
        return FilesystemBackend(
            root_dir=config.effective_workdir,
            virtual_mode=False,
            max_file_size_mb=10,  # Could be configurable
        )
    
    @staticmethod
    def create_interrupt_config(config: VibeConfig) -> dict[str, bool | InterruptOnConfig]:
        """Create interrupt config from Vibe tool permissions."""
        interrupt_on = {}
        
        # Map Vibe permissions to DeepAgents interrupts
        for tool_name, tool_config in config.tools.items():
            match tool_config.permission:
                case "ask":
                    interrupt_on[tool_name] = True
                case "never":
                    # Tool will be filtered out during loading
                    pass
                case "always":
                    # No interrupt needed
                    pass
        
        # Add default dangerous tools
        interrupt_on.update({
            "write_file": True,
            "edit_file": True,
            "execute": True,
            "bash": True,
        })
        
        return interrupt_on
    
    @staticmethod
    def create_system_prompt(config: VibeConfig) -> str:
        """Create system prompt from Vibe config."""
        from vibe.core.system_prompt import get_universal_system_prompt
        return get_universal_system_prompt(config)
    
    @staticmethod
    def get_tools(config: VibeConfig) -> list[Any]:
        """Get tools configured for the agent."""
        from vibe.core.engine.tools import VibeToolAdapter
        return VibeToolAdapter.get_all_tools(config)
    
    @staticmethod
    def get_middleware(config: VibeConfig) -> list[Any]:
        """Get middleware stack for the agent."""
        from vibe.core.engine.middleware import build_middleware_stack
        model = DeepAgentsConfig.create_model(config)
        backend = DeepAgentsConfig.create_backend(config)
        return build_middleware_stack(config, model, backend)
```

### Phase 3: Engine Integration

Use the config bridge in VibeEngine:

```python
# vibe/core/engine/engine.py

class VibeEngine:
    def __init__(self, config: VibeConfig, approval_callback=None):
        self.config = config
        self.approval_bridge = approval_callback
        self._agent = None
    
    def initialize(self) -> None:
        """Initialize the DeepAgents engine using Vibe config."""
        from deepagents import create_deep_agent
        from langgraph.checkpoint.memory import InMemorySaver
        
        # Use config bridge to get all parameters
        model = DeepAgentsConfig.create_model(self.config)
        tools = DeepAgentsConfig.get_tools(self.config)
        backend = DeepAgentsConfig.create_backend(self.config)
        middleware = DeepAgentsConfig.get_middleware(self.config)
        interrupt_on = DeepAgentsConfig.create_interrupt_config(self.config)
        system_prompt = DeepAgentsConfig.create_system_prompt(self.config)
        
        self._agent = create_deep_agent(
            model=model,
            tools=tools,
            system_prompt=system_prompt,
            middleware=middleware,
            backend=backend,
            interrupt_on=interrupt_on,
            checkpointer=InMemorySaver(),
        )
```

## Configuration Extensions

### DeepAgents-Specific Settings

Add new config options for DeepAgents features:

```python
# vibe/core/config.py - Add to VibeConfig

class VibeConfig(BaseSettings):
    # Existing fields...
    
    # DeepAgents-specific settings
    enable_subagents: bool = True
    enable_planning: bool = True
    enable_memory: bool = False  # Future feature
    max_recursion_depth: int = 1000
    context_compaction_threshold: int = 170000  # DeepAgents default
    
    # Backend settings
    filesystem_backend_root: Path | None = None
    filesystem_max_file_size_mb: int = 10
    filesystem_virtual_mode: bool = False
```

### Backward Compatibility

Ensure all existing config files continue to work:

```python
# vibe/core/config.py

class VibeConfig(BaseSettings):
    # All existing fields with defaults
    active_model: str = "devstral-2"  # Existing default
    auto_compact_threshold: int = 200_000  # Existing default
    
    # New fields with safe defaults
    enable_subagents: bool = True  # Enable new features by default
    context_compaction_threshold: int = 170000  # DeepAgents optimized
    
    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_config(cls, data: Any) -> Any:
        """Migrate legacy config to new format."""
        if isinstance(data, dict):
            # Handle old field names or formats
            # e.g., rename deprecated fields
            pass
        return data
```

## Project Configuration

### Preserve Project Detection

Keep the existing project-aware configuration:

```python
# vibe/core/config.py - PRESERVED

def load_config(workdir: Path | None = None) -> VibeConfig:
    """Load configuration with project awareness."""
    # Existing logic preserved
    # 1. Load global config
    # 2. Load project config if in project
    # 3. Merge with environment variables
    # 4. Apply command line overrides
    pass

def find_project_root(start_path: Path | None = None) -> Path | None:
    """Find project root via .git detection."""
    # Existing logic preserved
    pass
```

### Project-Specific DeepAgents Config

Allow project-specific DeepAgents settings:

```toml
# .vibe/config.toml (project-specific)

[deepagents]
enable_subagents = true
enable_planning = true
max_recursion_depth = 50  # Lower for faster project work

[deepagents.backend]
max_file_size_mb = 5  # Smaller for project files
virtual_mode = false
```

## Environment Variables

### Existing Variables (Preserved)

```bash
MISTRAL_API_KEY=your_key
VIBE_ACTIVE_MODEL=devstral-2
VIBE_AUTO_COMPACT_THRESHOLD=200000
```

### New Variables (Optional)

```bash
# DeepAgents features
VIBE_ENABLE_SUBAGENTS=true
VIBE_ENABLE_PLANNING=true

# LangChain settings
LANGCHAIN_TRACING=false  # Disable tracing by default
```

## Configuration Validation

### Schema Validation

Ensure DeepAgents compatibility:

```python
# vibe/core/config.py

class VibeConfig(BaseSettings):
    @model_validator(mode="after")
    def validate_deepagents_config(self) -> Self:
        """Validate configuration for DeepAgents compatibility."""
        
        # Check model compatibility
        model_config = self.get_model_config()
        if model_config.provider not in ["mistral", "anthropic", "openai"]:
            raise ValueError(f"Unsupported provider: {model_config.provider}")
        
        # Check tool permissions
        for tool_name, tool_config in self.tools.items():
            if tool_config.permission not in ["always", "ask", "never"]:
                raise ValueError(f"Invalid permission for {tool_name}: {tool_config.permission}")
        
        # Check backend settings
        if self.filesystem_max_file_size_mb > 100:
            raise ValueError("Max file size too large for DeepAgents backend")
        
        return self
```

## Files to Preserve

### Core Configuration (PRESERVED)
- `vibe/core/config.py` - Main configuration class
- `vibe/core/config_path.py` - Path management
- `vibe/core/prompts/` - System prompts
- `vibe/core/system_prompt.py` - Prompt generation

### Files to Add
- `vibe/core/engine/config_bridge.py` - DeepAgents parameter conversion

### Files to Modify
- `vibe/core/config.py` - Add DeepAgents-specific settings

## Validation Checklist

- [ ] Existing config files load without changes
- [ ] Project detection works as before
- [ ] TOML configuration preserved
- [ ] Environment variables respected
- [ ] New DeepAgents settings work
- [ ] Backward compatibility maintained
- [ ] Configuration validation works
- [ ] CLI config commands preserved
- [ ] Help text updated for new options
