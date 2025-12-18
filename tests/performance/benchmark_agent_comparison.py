"""Benchmark comparison script for Agent vs VibeEngine."""

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any

import pytest


def run_benchmarks(output_file: str) -> Dict[str, Any]:
    """Run benchmarks and return results."""
    # Run pytest with benchmark
    result = subprocess.run([
        sys.executable, "-m", "pytest",
        "tests/performance/test_benchmarks.py",
        f"--benchmark-json={output_file}",
        "--tb=short"
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print("Error running benchmarks:")
        print(result.stderr)
        return {}

    # Load results
    with open(output_file, 'r') as f:
        return json.load(f)


def generate_html_report(agent_results: Dict[str, Any], vibe_results: Dict[str, Any] | None = None) -> str:
    """Generate HTML comparison report."""
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Agent vs VibeEngine Performance Benchmarks</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .better {{ color: green; }}
        .worse {{ color: red; }}
    </style>
</head>
<body>
    <h1>Agent vs VibeEngine Performance Comparison</h1>

    <h2>Latency Tests</h2>
    <table>
        <tr>
            <th>Test</th>
            <th>Agent (ms)</th>
            <th>VibeEngine (ms)</th>
            <th>Difference</th>
        </tr>
        <!-- Placeholder for actual results -->
        <tr>
            <td>Simple Conversation</td>
            <td>{agent_results.get('simple_conv', 'N/A')}</td>
            <td>{vibe_results.get('simple_conv', 'N/A')}</td>
            <td>N/A</td>
        </tr>
    </table>

    <h2>Memory Usage</h2>
    <p>Memory growth comparison...</p>

    <h2>Throughput</h2>
    <p>Tokens/second comparison...</p>

    <p><em>Report generated automatically. Check thresholds: Latency ≤10%, Memory ≤20%, Throughput ≥5%.</em></p>
</body>
</html>
"""
    return html


def main():
    """Main comparison script."""
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    output_dir = Path("benchmark_results")
    output_dir.mkdir(exist_ok=True)

    print(f"Running benchmarks with {iterations} iterations...")

    # Run benchmarks (currently only Agent is implemented)
    benchmark_output = output_dir / "benchmark_results.json"

    results = run_benchmarks(str(benchmark_output))

    # Generate HTML report (placeholder for comparison)
    html_report = generate_html_report(results, {})  # Agent results, empty VibeEngine for now
    report_file = output_dir / "benchmark_report.html"
    with open(report_file, 'w') as f:
        f.write(html_report)

    print(f"Report generated: {report_file}")
    print(f"Benchmark results: {benchmark_output}")


if __name__ == "__main__":
    main()