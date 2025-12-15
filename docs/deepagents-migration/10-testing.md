# 10 - Testing Strategy & Validation

## Overview

Comprehensive testing strategy to ensure the DeepAgents migration maintains feature parity while adding new capabilities.

## Testing Principles

1. **Feature Parity First**: Ensure all existing Mistral Vibe features work before adding new ones
2. **Incremental Validation**: Test each migration phase independently
3. **Parallel Testing**: Run both old and new implementations side-by-side during transition
4. **User-Centric Validation**: Focus on end-user experience preservation

## Test Categories

### 1. Unit Tests

#### Engine Component Tests

```python
# tests/unit/test_engine.py

class TestVibeEngine:
    """Test VibeEngine functionality."""
    
    def test_initialization(self):
        """Test engine initializes with config."""
        config = create_test_config()
        engine = VibeEngine(config)
        
        assert engine._agent is None
        engine.initialize()
        assert engine._agent is not None
    
    def test_model_creation(self):
        """Test LangChain model creation from Vibe config."""
        config = create_test_config()
        model = DeepAgentsConfig.create_model(config)
        
        assert isinstance(model, ChatMistralAI)
        assert model.model_name == "devstral-small-latest"
    
    def test_backend_creation(self):
        """Test filesystem backend creation."""
        config = create_test_config()
        backend = DeepAgentsConfig.create_backend(config)
        
        assert isinstance(backend, FilesystemBackend)
        assert backend.root_dir == config.effective_workdir
    
    def test_interrupt_config(self):
        """Test approval interrupt configuration."""
        config = create_test_config()
        interrupts = DeepAgentsConfig.create_interrupt_config(config)
        
        assert "write_file" in interrupts
        assert interrupts["write_file"] is True
```

#### Tool Adapter Tests

```python
# tests/unit/test_tool_adapters.py

class TestToolAdapters:
    """Test tool adaptation from Vibe to DeepAgents."""
    
    def test_bash_tool_creation(self):
        """Test bash tool creation."""
        config = create_test_config()
        tools = VibeToolAdapter.get_all_tools(config)
        
        bash_tool = next(t for t in tools if t.name == "bash")
        assert bash_tool is not None
        assert "execute" in bash_tool.description.lower()
    
    def test_filesystem_tools(self):
        """Test that filesystem tools are available via middleware."""
        # FilesystemMiddleware provides ls, read_file, write_file, etc.
        # Test that middleware is properly configured
        config = create_test_config()
        middleware = build_middleware_stack(config, create_test_model(), create_test_backend())
        
        fs_middleware = next(m for m in middleware if isinstance(m, FilesystemMiddleware))
        assert fs_middleware is not None
```

#### Event Translation Tests

```python
# tests/unit/test_event_translation.py

class TestEventTranslation:
    """Test event translation between DeepAgents and Vibe."""
    
    def test_tool_call_translation(self):
        """Test tool call event translation."""
        langgraph_event = {
            "event": "on_tool_start",
            "name": "read_file",
            "run_id": "abc-123",
            "data": {"input": {"path": "/test.txt"}}
        }
        
        event = EventTranslator.translate(langgraph_event)
        assert isinstance(event, ToolCallEvent)
        assert event.tool_call.function.name == "read_file"
    
    def test_streaming_translation(self):
        """Test streaming token translation."""
        langgraph_event = {
            "event": "on_chat_model_stream",
            "data": {"chunk": type('MockChunk', (), {"content": "Hello"})()}
        }
        
        event = EventTranslator.translate(langgraph_event)
        assert isinstance(event, AssistantEvent)
        assert event.content == "Hello"
```

### 2. Integration Tests

#### End-to-End Conversation Tests

```python
# tests/integration/test_conversation.py

class TestConversationFlow:
    """Test complete conversation flows."""
    
    async def test_simple_conversation(self):
        """Test basic conversation without tools."""
        config = create_test_config()
        engine = VibeEngine(config)
        await engine.initialize()
        
         events = []
         async for event in engine.run("Hello"):
             events.append(event)
        
        assert len(events) > 0
        assert any(isinstance(e, AssistantEvent) for e in events)
    
    async def test_tool_execution(self):
        """Test conversation with tool execution."""
        config = create_test_config()
        engine = VibeEngine(config)
        await engine.initialize()
        
        # Mock approval callback
        approvals = []
        async def mock_approve(action):
            approvals.append(action)
            return True
        
        engine.approval_bridge = ApprovalBridge(callback=mock_approve)
        
         events = []
         async for event in engine.run("Create a file called test.txt"):
             events.append(event)
        
        # Should have tool call and result events
        tool_calls = [e for e in events if isinstance(e, ToolCallEvent)]
        tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
        
        assert len(tool_calls) > 0
        assert len(tool_results) > 0
```

#### TUI Integration Tests

```python
# tests/integration/test_tui_integration.py

class TestTUIIntegration:
    """Test TUI integration with DeepAgents engine."""
    
    async def test_tui_initialization(self):
        """Test TUI initializes with engine."""
        config = create_test_config()
        app = VibeApp(config)
        
        # Mock the event loop for testing
        await app._on_mount()
        
        assert app.engine is not None
        assert app._initialized is True
    
    async def test_message_handling(self):
        """Test user message handling."""
        config = create_test_config()
        app = VibeApp(config)
        
        # Mock message handling
        message_events = []
        original_handle = app._handle_event
        
        async def mock_handle(event):
            message_events.append(event)
            await original_handle(event)
        
        app._handle_event = mock_handle
        
        # Simulate user message
        await app.on_user_message("Test message")
        
        assert len(message_events) > 0
```

### 3. Feature Parity Tests

#### Configuration Tests

```python
# tests/feature/test_configuration.py

class TestConfigurationParity:
    """Test that configuration works identically."""
    
    def test_config_loading(self):
        """Test config loading preserves behavior."""
        # Load config the old way
        old_config = load_config_old_way()
        
        # Load config the new way
        new_config = VibeConfig.load()
        
        # Compare key properties
        assert old_config.active_model == new_config.active_model
        assert old_config.auto_compact_threshold == new_config.auto_compact_threshold
        assert len(old_config.models) == len(new_config.models)
    
    def test_project_detection(self):
        """Test project detection works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create .git directory
            os.makedirs(os.path.join(tmpdir, ".git"))
            
            # Test project detection
            project_root = find_project_root(Path(tmpdir))
            assert project_root == Path(tmpdir)
```

#### Tool Permission Tests

```python
# tests/feature/test_permissions.py

class TestPermissionParity:
    """Test tool permissions work identically."""
    
    def test_allowlist_denylist(self):
        """Test allowlist/denylist patterns."""
        config = create_test_config()
        
        # Test denylist
        config.tools["read_file"].denylist = ["*.secret"]
        
        # Should deny access to secret files
        permission = check_allowlist_denylist(
            "read_file", 
            {"path": "/secret.txt"}, 
            config
        )
        assert permission == ToolPermission.NEVER
    
    def test_approval_mapping(self):
        """Test permission mapping to interrupts."""
        config = create_test_config()
        config.tools["write_file"].permission = ToolPermission.ASK
        
        interrupts = DeepAgentsConfig.create_interrupt_config(config)
        
        assert "write_file" in interrupts
        assert interrupts["write_file"] is True
```

### 4. Performance Tests

#### Benchmarking Tests

```python
# tests/performance/test_benchmarks.py

class TestPerformance:
    """Performance regression tests."""
    
    async def test_conversation_speed(self):
        """Test conversation response time."""
        config = create_test_config()
        engine = VibeEngine(config)
        await engine.initialize()
        
        start_time = time.time()
        
         events = []
         async for event in engine.run("Hello world"):
             events.append(event)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete within reasonable time
        assert duration < 5.0  # 5 seconds max
        
        # Should have response
        assert any(isinstance(e, AssistantEvent) for e in events)
    
    def test_memory_usage(self):
        """Test memory usage doesn't regress."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Run conversation
        # ... test code ...
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 100 * 1024 * 1024  # 100MB max increase
```

### 5. Compatibility Tests

#### Backward Compatibility

```python
# tests/compatibility/test_backward_compat.py

class TestBackwardCompatibility:
    """Test that existing functionality still works."""
    
    def test_legacy_config_files(self):
        """Test old config files still load."""
        # Create old-style config file
        old_config_content = """
        active_model = "devstral-2"
        auto_compact_threshold = 200000
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml') as f:
            f.write(old_config_content)
            f.flush()
            
            # Should load without errors
            config = VibeConfig.load(f.name)
            assert config.active_model == "devstral-2"
    
    def test_legacy_api_usage(self):
        """Test that old APIs still work."""
        # Test that old Agent class can still be instantiated
        # (even if not used in production)
        config = create_test_config()
        agent = Agent(config)  # Legacy class
        
        assert agent is not None
        assert agent.config == config
```

## Test Execution Strategy

### Development Testing

```bash
# Run unit tests
uv run pytest tests/unit/ -v

# Run integration tests
uv run pytest tests/integration/ -v

# Run feature parity tests
uv run pytest tests/feature/ -v

# Run performance tests
uv run pytest tests/performance/ -v
```

### CI/CD Integration

```yaml
# .github/workflows/test.yml
name: Test
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - name: Install dependencies
        run: uv sync --all-groups
      - name: Run tests
        run: uv run pytest tests/ -v --cov=vibe --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v4
```

### Manual Testing Checklist

#### Pre-Migration Validation
- [ ] All existing tests pass
- [ ] TUI launches and accepts input
- [ ] Basic conversation works
- [ ] Tool execution works
- [ ] Approval dialogs appear
- [ ] Configuration loading works
- [ ] Project detection works

#### Post-Migration Validation
- [ ] DeepAgents engine initializes
- [ ] Conversation flows work
- [ ] New features (planning, subagents) available
- [ ] Performance meets expectations
- [ ] Error handling works
- [ ] Configuration preserved
- [ ] TUI appearance unchanged

#### Feature Validation
- [ ] File operations work (read, write, edit)
- [ ] Bash commands execute
- [ ] Todo management functional
- [ ] Subagent delegation works
- [ ] Context compaction triggers
- [ ] Memory usage reasonable
- [ ] Streaming responses smooth

## Test Data Management

### Test Fixtures

```python
# tests/conftest.py

@pytest.fixture
def test_config():
    """Create test configuration."""
    return VibeConfig(
        active_model="devstral-small-latest",
        models=[ModelConfig(
            name="devstral-small-latest",
            provider="mistral",
            alias="test-model"
        )],
        providers=[ProviderConfig(
            name="mistral",
            api_base="https://api.mistral.ai/v1",
            api_key_env_var="MISTRAL_API_KEY",
            backend=Backend.MISTRAL
        )]
    )

@pytest.fixture
def mock_mistral_key(monkeypatch):
    """Mock Mistral API key."""
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
```

### Mock Services

```python
# tests/mocks.py

class MockMistralBackend:
    """Mock Mistral backend for testing."""
    
    async def complete(self, **kwargs):
        return LLMChunk(
            content="Mock response",
            finish_reason="stop",
            usage=LLMUsage(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30
            )
        )

class MockFilesystemBackend:
    """Mock filesystem backend."""
    
    def read(self, path: str, offset: int = 0, limit: int = 500) -> str:
        return f"Mock content for {path}"
    
    def write(self, path: str, content: str) -> WriteResult:
        return WriteResult(path=path)
```

## Success Criteria

### Functional Completeness
- [ ] All existing features work identically
- [ ] All new DeepAgents features functional
- [ ] TUI experience preserved
- [ ] Configuration system unchanged
- [ ] Performance meets or exceeds original

### Quality Assurance
- [ ] Test coverage > 90%
- [ ] No regressions in existing functionality
- [ ] Error handling robust
- [ ] Memory leaks absent
- [ ] Race conditions handled

### User Experience
- [ ] TUI launches without issues
- [ ] Conversations flow naturally
- [ ] Tool approvals work smoothly
- [ ] Error messages helpful
- [ ] Performance feels responsive
