# 05 - Backend Systems

## Overview

Migrate from Mistral Vibe's custom LLM backends and direct filesystem access to DeepAgents' unified backend system.

## Current Backend Architecture

### LLM Backends

```
vibe/core/llm/backend/
├── __init__.py
├── factory.py      # BACKEND_FACTORY mapping
├── mistral.py      # MistralBackend (~200 lines)
└── generic.py      # GenericBackend for OpenAI-compatible APIs
```

### BackendLike Protocol

```python
# vibe/core/llm/types.py
class BackendLike(Protocol):
    async def __aenter__(self) -> BackendLike: ...
    async def __aexit__(self, ...): ...
    
    async def complete(
        self,
        model: ModelConfig,
        messages: list[LLMMessage],
        temperature: float,
        tools: list[AvailableTool] | None,
        max_tokens: int | None,
        tool_choice: StrToolChoice | AvailableTool | None,
        extra_headers: dict[str, str] | None,
    ) -> LLMChunk: ...

    async def complete_streaming(
        self,
        model: ModelConfig,
        ...
    ) -> AsyncGenerator[LLMChunk]: ...

    async def count_tokens(self, messages: list[LLMMessage]) -> int: ...
```

### File Operations

Currently handled directly in tools:
- `read_file.py` - Direct `aiofiles` operations
- `write_file.py` - Direct `aiofiles` operations  
- `search_replace.py` - Direct `aiofiles` operations

## DeepAgents Backend Architecture

### LLM: BaseChatModel

DeepAgents uses LangChain's `BaseChatModel` abstraction:

```python
from langchain_mistralai import ChatMistralAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

# Simple model creation
model = ChatMistralAI(
    model="devstral-small-latest",
    api_key=os.getenv("MISTRAL_API_KEY"),
    temperature=0.2,
)
```

### Filesystem: BackendProtocol

```python
# deepagents/backends/protocol.py
class BackendProtocol(abc.ABC):
    def ls_info(self, path: str) -> list[FileInfo]: ...
    def read(self, file_path: str, offset: int, limit: int) -> str: ...
    def write(self, file_path: str, content: str) -> WriteResult: ...
    def edit(self, file_path: str, old_string: str, new_string: str) -> EditResult: ...
    def grep_raw(self, pattern: str, path: str, glob: str) -> list[GrepMatch]: ...
    def glob_info(self, pattern: str, path: str) -> list[FileInfo]: ...
```

### Available Backends

| Backend | Purpose | Use Case |
|---------|---------|----------|
| `FilesystemBackend` | Real filesystem | Local development |
| `StateBackend` | In-memory (ephemeral) | Testing, sandboxed |
| `StoreBackend` | LangGraph store | Persistent across sessions |
| `CompositeBackend` | Multiple backends | Read from one, write to another |
| `SandboxBackendProtocol` | Remote execution | Modal, Runloop, Daytona |

## Migration Plan

### Phase 1: LLM Backend Migration

Replace custom backends with LangChain models:

```python
# vibe/core/engine/models.py

from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel

from vibe.core.config import VibeConfig, Backend


def create_model_from_config(config: VibeConfig) -> BaseChatModel:
    """Create a LangChain model from Vibe configuration."""
    model_config = config.get_model_config()
    provider_config = config.get_provider_config(model_config.provider)
    
    match provider_config.backend:
        case Backend.MISTRAL:
            return ChatMistralAI(
                model=model_config.name,
                api_key=provider_config.get_api_key(),
                temperature=model_config.temperature,
                max_tokens=16384,
                base_url=provider_config.api_base,
            )
        
        case Backend.GENERIC:
            # OpenAI-compatible endpoints (llama.cpp, etc.)
            return ChatOpenAI(
                model=model_config.name,
                api_key=provider_config.get_api_key() or "not-needed",
                temperature=model_config.temperature,
                base_url=provider_config.api_base,
            )
        
        case _:
            raise ValueError(f"Unknown backend: {provider_config.backend}")
```

### Phase 2: Filesystem Backend Integration

Use DeepAgents' `FilesystemBackend`:

```python
# vibe/core/engine/engine.py

from deepagents.backends import FilesystemBackend
from deepagents.middleware import FilesystemMiddleware


class VibeEngine:
    def _create_backend(self) -> FilesystemBackend:
        """Create filesystem backend for file operations."""
        return FilesystemBackend(
            root_dir=self.config.effective_workdir,
            virtual_mode=False,  # Use real filesystem paths
            max_file_size_mb=10,
        )

    def initialize(self) -> None:
        backend = self._create_backend()
        
        self._agent = create_deep_agent(
            model=self._create_model(),
            backend=backend,  # Passed to FilesystemMiddleware
            ...
        )
```

### Phase 3: Token Counting

LangChain handles token counting internally, but for UI display:

```python
# vibe/core/engine/stats.py

from langchain_core.language_models import BaseChatModel


def estimate_tokens(model: BaseChatModel, messages: list) -> int:
    """Estimate token count for messages."""
    # LangChain models may have get_num_tokens method
    if hasattr(model, "get_num_tokens"):
        text = "\n".join(str(m.content) for m in messages)
        return model.get_num_tokens(text)
    
    # Fallback: rough estimate (4 chars per token)
    total_chars = sum(len(str(m.content)) for m in messages)
    return total_chars // 4
```

## Files to Remove

After migration:

```
vibe/core/llm/
├── backend/
│   ├── factory.py      # Remove - use create_model_from_config
│   ├── mistral.py      # Remove - use ChatMistralAI
│   └── generic.py      # Remove - use ChatOpenAI
├── types.py            # Remove BackendLike protocol
└── format.py           # May need to preserve for message formatting
```

## Comparison

### Before: Custom MistralBackend

```python
# vibe/core/llm/backend/mistral.py (~200 lines)
class MistralBackend:
    def __init__(self, provider: ProviderConfig, timeout: float):
        self._client: mistralai.Mistral | None = None
        self._provider = provider
        self._mapper = MistralMapper()
        # ... complex initialization

    async def __aenter__(self):
        self._client = mistralai.Mistral(...)
        await self._client.__aenter__()
        return self

    async def complete_streaming(self, ...):
        # 50+ lines of streaming logic
        messages = [self._mapper.prepare_message(msg) for msg in messages]
        async for chunk in self._client.chat.stream_async(...):
            yield self._mapper.parse_chunk(chunk)
```

### After: LangChain ChatMistralAI

```python
# 5 lines
from langchain_mistralai import ChatMistralAI

model = ChatMistralAI(
    model="devstral-small-latest",
    api_key=os.getenv("MISTRAL_API_KEY"),
    temperature=0.2,
)
```

### Before: Direct File Operations

```python
# vibe/core/tools/builtins/read_file.py
async def run(self, args: ReadFileArgs) -> ReadFileResult:
    file_path = self._prepare_and_validate_path(args)
    async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
        lines = await f.readlines()
    # ... 50+ lines of processing
```

### After: FilesystemBackend

```python
# Handled automatically by FilesystemMiddleware
# Just configure the backend
backend = FilesystemBackend(
    root_dir=config.effective_workdir,
    virtual_mode=False,
)
# FilesystemMiddleware provides read_file, write_file, edit_file tools
```

## Configuration Preservation

The `VibeConfig` model/provider system is preserved, just mapped to LangChain:

```python
# vibe/core/config.py (preserved)
class ModelConfig(BaseModel):
    name: str
    provider: str
    alias: str
    temperature: float = 0.2
    input_price: float = 0.0
    output_price: float = 0.0

class ProviderConfig(BaseModel):
    name: str
    api_base: str
    api_key_env_var: str
    backend: Backend = Backend.GENERIC

# Usage in engine
model_config = config.get_model_config()  # Works as before
provider_config = config.get_provider_config(model_config.provider)
model = create_model_from_config(config)  # New function, same config
```

## Validation Checklist

- [ ] ChatMistralAI connects to Mistral API
- [ ] ChatOpenAI works with local llama.cpp
- [ ] FilesystemBackend reads files correctly
- [ ] FilesystemBackend writes files correctly
- [ ] FilesystemBackend edit works (search/replace)
- [ ] Path validation and security preserved
- [ ] Token counting available for UI
- [ ] Streaming works through LangGraph
- [ ] Configuration system unchanged
- [ ] Cost tracking still functional
