#!/usr/bin/env python3
"""
Evaluation Script for Router Benchmarking.

Runs full benchmark with actual API calls, quality evaluation, and cost tracking.
Compares multiple baseline routers and prints comparison results.

Requirements:
- OPENROUTER_API_KEY environment variable must be set
- Or create a .env file with OPENROUTER_API_KEY=your_key
"""

import asyncio
import sys

from src.config import OPENROUTER_API_KEY
from src.model_registry import MODEL_REGISTRY, ModelTier
from src.benchmarking import (
    benchmark_all_routers,
    print_benchmark_summary,
    print_router_comparison,
    SAMPLE_QUERIES,
    ALL_QUERIES
)


async def main():
    """Run full benchmark evaluation."""
    print("=" * 80)
    print("ROUTER EVALUATION BENCHMARK")
    print("=" * 80)

    # Check for API key
    if not OPENROUTER_API_KEY:
        print("\nError: OPENROUTER_API_KEY environment variable not set.")
        print("Set it in your .env file or export it in your shell:")
        print("  export OPENROUTER_API_KEY=your_key_here")
        sys.exit(1)

    # Show available models
    print("\nAvailable Models by Tier:")
    print("-" * 60)
    for tier in ModelTier:
        models = [k for k, v in MODEL_REGISTRY.items() if v.tier == tier]
        print(f"  {tier.value.upper():<10}: {', '.join(models)}")

    # Show queries to be tested
    print("\nQueries to benchmark:")
    print("-" * 60)
    input_queries = ALL_QUERIES if "--full" in sys.argv else SAMPLE_QUERIES
    total_queries = 0
    for category, queries in input_queries.items():
        print(f"  {category}: {len(queries)} queries")
        total_queries += len(queries)
    print(f"  Total: {total_queries} queries")

    # Run benchmark
    print("\n" + "=" * 80)
    print("RUNNING FULL BENCHMARK")
    print("=" * 80)

    all_results = await benchmark_all_routers(
        queries=input_queries,
        evaluator_model="trinity-mini",
        seed=42,
        max_concurrent=3
    )

    # Print individual results
    for router_name, results in all_results.items():
        print_benchmark_summary(results, router_name)

    # Print comparison
    print_router_comparison(all_results)


if __name__ == "__main__":
    asyncio.run(main())
