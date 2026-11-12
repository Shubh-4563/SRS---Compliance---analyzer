"""
Convenience evaluation script for running the Declarative Benchmark Runner (Phase 2.6.3).
Usage:
    python3 scripts/evaluate_manifest_benchmark.py --all
    python3 scripts/evaluate_manifest_benchmark.py --manifest data/benchmarks/pair_a_lab_report.json
    python3 scripts/evaluate_manifest_benchmark.py --manifest data/benchmarks/pair_b_srs.json --output-json reports/bench_b.json
"""
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarks.benchmark_runner import main

if __name__ == "__main__":
    main()

