# Performance Benchmarks

This document describes the performance benchmarking suite for comparing the legacy Agent vs DeepAgents VibeEngine.

## Methodology

Benchmarks are run using `pytest-benchmark` with mock responses to avoid API variability and costs. Tests measure latency, memory usage, and throughput across different conversation scenarios.

## Test Categories

### Latency Tests
- **Simple Conversation**: 3-4 messages, no tools (target: ≤1s variance)
- **Multi-turn Conversation**: 10+ messages (target: ≤5% variance)
- **Tool Execution**: File operations (target: ≤10% overhead)

### Memory Usage Tests
- Initial memory footprint on engine creation
- Memory growth over conversation
- Peak memory during tool execution

### Throughput Tests
- Tokens per second during streaming
- Messages per second for batch processing

## Running Benchmarks

### Local Execution
```bash
# Run all benchmarks
pytest tests/performance/test_benchmarks.py --benchmark-json=results.json

# Generate comparison report
python tests/performance/benchmark_agent_comparison.py
```

### CI Integration
Benchmarks run automatically on pull requests and can be viewed in the PR comments.

## Results

*Baseline results will be added after initial runs.*

## Regression Thresholds

- **Response Latency**: VibeEngine ≤ Agent + 10%
- **Memory Usage**: VibeEngine ≤ Agent + 20%
- **Throughput**: VibeEngine ≥ Agent - 5%

## Optimization Tips

- Use streaming mode for better throughput
- Monitor memory growth in long conversations
- Consider auto-compaction for context management