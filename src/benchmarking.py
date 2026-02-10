"""
Benchmarking Module for Router Evaluation.

Provides utilities for benchmarking routers with actual API calls,
quality evaluation, and cost tracking.
"""

import time
import random
import asyncio
import json
from pathlib import Path
from typing import Literal, Optional, Dict, List
from dataclasses import dataclass, field

import httpx

from .config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL
from .model_registry import MODEL_REGISTRY, ModelTier
from .latency import get_latency_multiplier
from .quality import QualityEvaluation, evaluate_quality, evaluate_quality_batch
from .router import BaseRouter, NaiveRouter, StaticRouter


# Load sample queries from JSON file
def load_queries(filename: str) -> Dict[str, List[str]]:
    """Load sample queries from the data directory."""
    data_path = Path(__file__).parent.parent / "data" / filename
    if data_path.exists():
        with open(data_path) as f:
            return json.load(f)
    # Fallback to empty dict if file doesn't exist
    return {}


SAMPLE_QUERIES = load_queries("sample_queries.json")
ALL_QUERIES = load_queries("all_queries.json")


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run"""
    query: str
    model_key: str
    deployment: str
    latency_ms: float  # Final latency (API latency * deployment multiplier)
    quality: Optional[QualityEvaluation]
    response: str
    cost_estimate: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class RoutingBenchmarkResult:
    """Result from benchmarking a router on a single query"""
    query: str
    query_category: str
    model_key: str
    deployment: str
    model_tier: ModelTier
    latency_ms: float
    timed_out: bool
    # Extended fields for full benchmark (with API calls)
    quality_score: Optional[float] = None
    cost_estimate: float = 0.0
    response: Optional[str] = None


@dataclass
class InferenceResult:
    """Intermediate result from inference phase (before quality evaluation)."""
    query: str
    query_category: str
    model_key: str
    deployment: str
    model_tier: ModelTier
    latency_ms: float
    timed_out: bool
    cost_estimate: float
    response: str


async def run_single_benchmark(
    query: str,
    model_key: str,
    deployment: Literal["edge", "cloud"],
    evaluate: bool = True
) -> BenchmarkResult:
    """
    Run a single benchmark iteration.

    This simulates the full pipeline:
    1. Simulate latency for the deployment
    2. Actually call the model via OpenRouter
    3. Optionally evaluate quality
    """
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    model_config = MODEL_REGISTRY[model_key]

    # Get latency multiplier for this deployment
    latency_multiplier = get_latency_multiplier(model_key, deployment)

    # Make actual API call
    async with httpx.AsyncClient() as client:
        start_time = time.time()
        api_response = await client.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_config.model_id,
                "messages": [{"role": "user", "content": query}],
                "max_tokens": 1000
            },
            timeout=60.0
        )
        actual_latency = (time.time() - start_time) * 1000

    api_response.raise_for_status()
    result = api_response.json()

    response_text = result["choices"][0]["message"]["content"]

    # Estimate cost
    usage = result.get("usage", {})
    input_tokens = usage.get("prompt_tokens", len(query.split()) * 1.3)
    output_tokens = usage.get("completion_tokens", len(response_text.split()) * 1.3)

    cost = (
        (input_tokens / 1_000_000) * model_config.cost_per_million_input +
        (output_tokens / 1_000_000) * model_config.cost_per_million_output
    )

    # Evaluate quality if requested
    quality = None
    if evaluate:
        quality = await evaluate_quality(
            query=query,
            response=response_text,
            model_key=model_key
        )

    # Apply deployment multiplier to actual API latency
    final_latency_ms = actual_latency * latency_multiplier

    return BenchmarkResult(
        query=query,
        model_key=model_key,
        deployment=deployment,
        latency_ms=final_latency_ms,
        quality=quality,
        response=response_text,
        cost_estimate=cost
    )


async def _process_single_query_inference(
    query: str,
    category: str,
    router: BaseRouter,
    semaphore: asyncio.Semaphore,
    client: httpx.AsyncClient,
    max_retries: int = 3,
    base_delay: float = 10.0
) -> Optional[InferenceResult]:
    """Process a single query - inference only, no evaluation."""
    async with semaphore:
        last_error = None
        for attempt in range(max_retries):
            try:
                # Get routing decision
                model_key, deployment = router.route(query)
                model_config = MODEL_REGISTRY[model_key]

                # Get latency multiplier for this deployment
                latency_multiplier = get_latency_multiplier(model_key, deployment)

                # Make actual API call
                start_time = time.time()
                api_response = await client.post(
                    f"{OPENROUTER_BASE_URL}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model_config.model_id,
                        "messages": [{"role": "user", "content": query}],
                        "max_tokens": 1000
                    },
                    timeout=60.0
                )
                actual_latency = (time.time() - start_time) * 1000

                # Check for rate limiting
                if api_response.status_code == 429:
                    delay = base_delay * (2 ** attempt)
                    print(f"  Rate limited, waiting {delay:.0f}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                    continue

                api_response.raise_for_status()
                result = api_response.json()

                # Validate response structure
                if "choices" not in result or not result["choices"]:
                    error_msg = result.get("error", {}).get("message", str(result))
                    print(f"  API error: {error_msg[:100]}")
                    await asyncio.sleep(base_delay)
                    continue

                response_text = result["choices"][0]["message"]["content"]

                # Calculate cost
                usage = result.get("usage", {})
                input_tokens = usage.get("prompt_tokens", len(query.split()) * 1.3)
                output_tokens = usage.get("completion_tokens", len(response_text.split()) * 1.3)
                cost = (
                    (input_tokens / 1_000_000) * model_config.cost_per_million_input +
                    (output_tokens / 1_000_000) * model_config.cost_per_million_output
                )

                # Apply deployment multiplier to actual API latency
                total_latency = actual_latency * latency_multiplier

                print(f"  [{category}] {model_key}@{deployment}: latency={total_latency:.0f}ms")

                return InferenceResult(
                    query=query,
                    query_category=category,
                    model_key=model_key,
                    deployment=deployment,
                    model_tier=model_config.tier,
                    latency_ms=total_latency,
                    timed_out=False,
                    cost_estimate=cost,
                    response=response_text
                )

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    delay = base_delay * (2 ** attempt)
                    print(f"  Rate limited, waiting {delay:.0f}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                    last_error = e
                else:
                    print(f"  HTTP error: {e}")
                    return None
            except Exception as e:
                last_error = e
                if "429" in str(e) or "rate" in str(e).lower():
                    delay = base_delay * (2 ** attempt)
                    print(f"  Rate limited, waiting {delay:.0f}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                else:
                    print(f"  Error processing query: {e}")
                    return None

        print(f"  Failed after {max_retries} retries: {last_error}")
        return None


async def benchmark_router(
    router: BaseRouter,
    queries: Optional[Dict[str, List[str]]] = None,
    evaluator_model: str = "trinity-mini",
    seed: Optional[int] = None,
    max_concurrent: int = 3
) -> List[RoutingBenchmarkResult]:
    """
    Full benchmark with actual API calls, quality evaluation, and cost tracking.

    Uses a two-phase approach:
    - Phase 1: Run all inference API calls in parallel
    - Phase 2: Batch evaluate all responses in a single API call

    Args:
        router: Router instance to benchmark
        queries: Dict of category -> list of queries (defaults to SAMPLE_QUERIES)
        evaluator_model: Model to use for quality evaluation
        seed: Random seed for reproducibility
        max_concurrent: Maximum number of concurrent API calls

    Returns:
        List of RoutingBenchmarkResult with quality scores and costs
    """
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    if seed is not None:
        random.seed(seed)

    queries = queries or SAMPLE_QUERIES

    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(max_concurrent)

    # PHASE 1: Run all inference calls in parallel
    print("  Phase 1: Running inference...")
    all_tasks = []
    async with httpx.AsyncClient() as client:
        for category, query_list in queries.items():
            for query in query_list:
                task = _process_single_query_inference(
                    query=query,
                    category=category,
                    router=router,
                    semaphore=semaphore,
                    client=client
                )
                all_tasks.append(task)

        # Run all inference in parallel (limited by semaphore)
        inference_results = await asyncio.gather(*all_tasks)

    # Filter out None results (from errors)
    inference_results = [r for r in inference_results if r is not None]

    if not inference_results:
        return []

    # PHASE 2: Batch evaluate all responses in one API call
    print(f"  Phase 2: Batch evaluating {len(inference_results)} responses...")
    items = [(r.query, r.response, r.model_key) for r in inference_results]

    try:
        evaluations = await evaluate_quality_batch(items, evaluator_model)
    except Exception as e:
        print(f"  Batch evaluation failed: {e}")
        # Return results without quality scores if batch eval fails
        return [
            RoutingBenchmarkResult(
                query=r.query[:50] + "..." if len(r.query) > 50 else r.query,
                query_category=r.query_category,
                model_key=r.model_key,
                deployment=r.deployment,
                model_tier=r.model_tier,
                latency_ms=r.latency_ms,
                timed_out=r.timed_out,
                quality_score=None,
                cost_estimate=r.cost_estimate,
                response=r.response[:100] + "..." if len(r.response) > 100 else r.response
            )
            for r in inference_results
        ]

    # Combine inference results with evaluations
    final_results = []
    for r, eval_result in zip(inference_results, evaluations):
        print(f"  [{r.query_category}] {r.model_key}@{r.deployment}: quality={eval_result.overall_score:.1f}/10")
        final_results.append(RoutingBenchmarkResult(
            query=r.query[:50] + "..." if len(r.query) > 50 else r.query,
            query_category=r.query_category,
            model_key=r.model_key,
            deployment=r.deployment,
            model_tier=r.model_tier,
            latency_ms=r.latency_ms,
            timed_out=r.timed_out,
            quality_score=eval_result.overall_score,
            cost_estimate=r.cost_estimate,
            response=r.response[:100] + "..." if len(r.response) > 100 else r.response
        ))

    return final_results


def print_benchmark_summary(
    results: List[RoutingBenchmarkResult],
    router_name: str = "Router"
) -> None:
    """Print a formatted summary of benchmark results."""
    if not results:
        print("No results to display.")
        return

    # Check if we have quality data
    has_quality = any(r.quality_score is not None for r in results)
    has_cost = any(r.cost_estimate > 0 for r in results)

    print("\n" + "=" * 100)
    print(f"{router_name.upper()} BENCHMARK RESULTS")
    print("=" * 100)

    # Group by category
    categories: Dict[str, List[RoutingBenchmarkResult]] = {}
    for r in results:
        if r.query_category not in categories:
            categories[r.query_category] = []
        categories[r.query_category].append(r)

    # Overall stats
    total_latency = sum(r.latency_ms for r in results)
    avg_latency = total_latency / len(results)
    timeout_count = sum(1 for r in results if r.timed_out)
    timeout_rate = timeout_count / len(results) * 100
    total_cost = sum(r.cost_estimate for r in results)

    print(f"\nOverall Statistics:")
    print(f"  Total queries:     {len(results)}")
    print(f"  Average latency:   {avg_latency:.1f} ms")
    print(f"  Timeout rate:      {timeout_rate:.1f}%")

    if has_quality:
        quality_results = [r for r in results if r.quality_score is not None]
        if quality_results:
            avg_quality = sum(r.quality_score for r in quality_results) / len(quality_results)
            print(f"  Average quality:   {avg_quality:.2f}/10")

    if has_cost:
        print(f"  Total cost:        ${total_cost:.6f}")

    # Stats by category
    print(f"\nResults by Query Category:")
    print("-" * 100)
    if has_quality:
        print(f"{'Category':<12} {'Count':>6} {'Latency':>10} {'Quality':>10} {'Cost':>12}   {'Top Models':<40}")
    else:
        print(f"{'Category':<12} {'Count':>6} {'Latency':>10}   {'Routing Distribution':<60}")
    print("-" * 100)

    for category in ["simple", "moderate", "complex", "reasoning", "coding"]:
        if category in categories:
            cat_results = categories[category]
            cat_avg_latency = sum(r.latency_ms for r in cat_results) / len(cat_results)

            # Count model/deployment combinations
            routing_counts: Dict[str, int] = {}
            for r in cat_results:
                key = f"{r.model_key}@{r.deployment}"
                routing_counts[key] = routing_counts.get(key, 0) + 1

            if has_quality:
                cat_quality_results = [r for r in cat_results if r.quality_score is not None]
                cat_avg_quality = (
                    sum(r.quality_score for r in cat_quality_results) / len(cat_quality_results)
                    if cat_quality_results else 0
                )
                cat_cost = sum(r.cost_estimate for r in cat_results)
                # Show top 2 models
                top_models = sorted(routing_counts.items(), key=lambda x: -x[1])[:2]
                top_str = ", ".join(f"{k}({v})" for k, v in top_models)
                print(f"{category:<12} {len(cat_results):>6} {cat_avg_latency:>8.0f}ms "
                      f"{cat_avg_quality:>9.2f}/10 ${cat_cost:>10.6f}   {top_str}")
            else:
                dist_parts = [f"{k}: {v}" for k, v in sorted(routing_counts.items())]
                dist = ", ".join(dist_parts)
                if len(dist) > 60:
                    dist = dist[:57] + "..."
                print(f"{category:<12} {len(cat_results):>6} {cat_avg_latency:>8.0f}ms   {dist}")

    # Stats by model tier
    print(f"\nResults by Model Tier:")
    print("-" * 100)
    tiers: Dict[str, List[RoutingBenchmarkResult]] = {}
    for r in results:
        tier_name = r.model_tier.value
        if tier_name not in tiers:
            tiers[tier_name] = []
        tiers[tier_name].append(r)

    if has_quality:
        print(f"  {'Tier':<10} {'Count':>6} {'Avg Latency':>12} {'Avg Quality':>12} {'Total Cost':>12}")
        print(f"  {'-'*10} {'-'*6} {'-'*12} {'-'*12} {'-'*12}")
        for tier_name in ["small", "medium", "large", "reasoning"]:
            if tier_name in tiers:
                tier_results = tiers[tier_name]
                tier_avg_latency = sum(r.latency_ms for r in tier_results) / len(tier_results)
                tier_quality_results = [r for r in tier_results if r.quality_score is not None]
                tier_avg_quality = (
                    sum(r.quality_score for r in tier_quality_results) / len(tier_quality_results)
                    if tier_quality_results else 0
                )
                tier_cost = sum(r.cost_estimate for r in tier_results)
                print(f"  {tier_name:<10} {len(tier_results):>6} {tier_avg_latency:>10.0f}ms "
                      f"{tier_avg_quality:>11.2f}/10 ${tier_cost:>10.6f}")
    else:
        for tier_name, tier_results in sorted(tiers.items()):
            tier_avg = sum(r.latency_ms for r in tier_results) / len(tier_results)
            print(f"  {tier_name:<10}: {len(tier_results):>4} queries, avg {tier_avg:>8.1f} ms")

    # Stats by deployment
    print(f"\nResults by Deployment:")
    print("-" * 100)
    deployments: Dict[str, List[RoutingBenchmarkResult]] = {}
    for r in results:
        if r.deployment not in deployments:
            deployments[r.deployment] = []
        deployments[r.deployment].append(r)

    if has_quality:
        print(f"  {'Deploy':<8} {'Count':>6} {'Avg Latency':>12} {'Avg Quality':>12} {'Total Cost':>12}")
        print(f"  {'-'*8} {'-'*6} {'-'*12} {'-'*12} {'-'*12}")
        for deploy in ["edge", "cloud"]:
            if deploy in deployments:
                deploy_results = deployments[deploy]
                deploy_avg_latency = sum(r.latency_ms for r in deploy_results) / len(deploy_results)
                deploy_quality_results = [r for r in deploy_results if r.quality_score is not None]
                deploy_avg_quality = (
                    sum(r.quality_score for r in deploy_quality_results) / len(deploy_quality_results)
                    if deploy_quality_results else 0
                )
                deploy_cost = sum(r.cost_estimate for r in deploy_results)
                print(f"  {deploy:<8} {len(deploy_results):>6} {deploy_avg_latency:>10.0f}ms "
                      f"{deploy_avg_quality:>11.2f}/10 ${deploy_cost:>10.6f}")
    else:
        for deploy, deploy_results in sorted(deployments.items()):
            deploy_avg = sum(r.latency_ms for r in deploy_results) / len(deploy_results)
            print(f"  {deploy:<6}: {len(deploy_results):>4} queries, avg {deploy_avg:>8.1f} ms")


def print_router_comparison(
    all_results: Dict[str, List[RoutingBenchmarkResult]]
) -> None:
    """
    Print a comparison table of multiple routers.

    Args:
        all_results: Dict mapping router name to list of benchmark results
    """
    if not all_results:
        print("No results to compare.")
        return

    print("\n" + "=" * 100)
    print("ROUTER COMPARISON")
    print("=" * 100)

    # Header
    print(f"\n{'Router':<35} {'Queries':>8} {'Avg Latency':>12} {'Avg Quality':>12} {'Total Cost':>14}")
    print("-" * 100)

    for router_name, results in all_results.items():
        if not results:
            print(f"{router_name:<35} {'N/A':>8} {'N/A':>12} {'N/A':>12} {'N/A':>14}")
            continue

        avg_latency = sum(r.latency_ms for r in results) / len(results)
        quality_results = [r for r in results if r.quality_score is not None]
        avg_quality = (
            sum(r.quality_score for r in quality_results) / len(quality_results)
            if quality_results else 0
        )
        total_cost = sum(r.cost_estimate for r in results)

        print(f"{router_name:<35} {len(results):>8} {avg_latency:>10.0f}ms "
              f"{avg_quality:>11.2f}/10 ${total_cost:>12.6f}")

    print("-" * 100)

    # Find best router for each metric
    best_latency = min(
        all_results.items(),
        key=lambda x: sum(r.latency_ms for r in x[1]) / len(x[1]) if x[1] else float('inf')
    )
    best_quality = max(
        all_results.items(),
        key=lambda x: (
            sum(r.quality_score for r in x[1] if r.quality_score) /
            len([r for r in x[1] if r.quality_score])
            if any(r.quality_score for r in x[1]) else 0
        )
    )
    best_cost = min(
        all_results.items(),
        key=lambda x: sum(r.cost_estimate for r in x[1]) if x[1] else float('inf')
    )

    print(f"\nBest Latency:  {best_latency[0]}")
    print(f"Best Quality:  {best_quality[0]}")
    print(f"Lowest Cost:   {best_cost[0]}")


async def benchmark_all_routers(
    queries: Optional[Dict[str, List[str]]] = None,
    evaluator_model: str = "trinity-mini",
    seed: Optional[int] = None,
    max_concurrent: int = 3
) -> Dict[str, List[RoutingBenchmarkResult]]:
    """
    Benchmark all baseline routers and return comparison results.

    Args:
        queries: Dict of category -> list of queries
        evaluator_model: Model to use for quality evaluation
        seed: Random seed for reproducibility
        max_concurrent: Maximum concurrent API calls per router

    Returns:
        Dict mapping router name to benchmark results
    """
    queries = queries or SAMPLE_QUERIES
    all_results: Dict[str, List[RoutingBenchmarkResult]] = {}

    # Define routers to benchmark
    routers: List[BaseRouter] = [
        NaiveRouter(edge_probability=0.5),
        StaticRouter("gemma-3-4b"),       # Edge static (SMALL model)
        StaticRouter("mistral-small-24b"),    # Cloud static (LARGE model)
    ]

    for router in routers:
        print(f"\n  Benchmarking: {router.name}")
        print(f"  {'-' * 40}")

        results = await benchmark_router(
            router=router,
            queries=queries,
            evaluator_model=evaluator_model,
            seed=seed,
            max_concurrent=max_concurrent
        )
        all_results[router.name] = results

    return all_results
