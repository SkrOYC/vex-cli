"""Benchmark comparison script for Agent vs VibeEngine."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any


def run_benchmarks(output_file: str) -> dict[str, Any]:
    """Run benchmarks and return results."""
    # Run pytest with benchmark
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/performance/test_benchmarks.py",
            f"--benchmark-json={output_file}",
            "--tb=short",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print("Error running benchmarks:")
        print(result.stderr)
        return {}

    # Load results
    with open(output_file) as f:
        return json.load(f)


def extract_benchmark_table(
    agent_results: dict[str, Any], vibe_results: dict[str, Any] | None = None
) -> str:
    """Extract benchmark data from pytest-benchmark JSON structure."""
    if not agent_results.get("benchmarks"):
        return "<tr><td colspan='4'>No benchmark data available</td></tr>"

    rows = []
    for benchmark in agent_results["benchmarks"]:
        name = benchmark["name"]
        mean_time = (
            benchmark["stats"].get("mean", 0) * 1000000
        )  # Convert to microseconds
        rows.append(f"""
        <tr>
            <td>{name}</td>
            <td>{mean_time:.2f}</td>
            <td>N/A</td>
            <td>N/A</td>
        </tr>
        """)

    return "".join(rows)


def generate_html_report(
    agent_results: dict[str, Any], vibe_results: dict[str, Any] | None = None
) -> str:
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
        <!-- Extract actual results from pytest-benchmark JSON structure -->
        {extract_benchmark_table(agent_results, vibe_results)}
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
    output_dir = Path("benchmark_results")
    output_dir.mkdir(exist_ok=True)

    print("Running benchmarks...")

    # Run benchmarks (currently only Agent is implemented)
    benchmark_output = output_dir / "benchmark_results.json"

    results = run_benchmarks(str(benchmark_output))

    # Generate HTML report (placeholder for comparison)
    html_report = generate_html_report(
        results, {}
    )  # Agent results, empty VibeEngine for now
    report_file = output_dir / "benchmark_report.html"
    with open(report_file, "w") as f:
        f.write(html_report)

    print(f"Report generated: {report_file}")
    print(f"Benchmark results: {benchmark_output}")


if __name__ == "__main__":
    main()
