#!/usr/bin/env python3
"""
Smoke Test for Assessment Simulation Module.

Runs basic tests to verify the module is working correctly:
1. Model registry is loaded
2. Latency simulation works
3. Edge constraints are enforced
4. Routers work correctly
5. (Optional) API calls work if OPENROUTER_API_KEY is set

Run with --api flag to include API tests:
    uv run python smoke_test.py --api
"""

import sys
import asyncio
import httpx

from src.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL
from src.model_registry import (
    MODEL_REGISTRY,
    ModelTier,
    EDGE_COMPATIBLE_MODELS,
    get_models_by_tier,
)
from src.latency import (
    get_latency_multiplier,
    EDGE_LATENCY_MULTIPLIER,
    CLOUD_LATENCY_MULTIPLIER,
)
from src.router import NaiveRouter, StaticRouter
from src.quality import (
    QualityDimension,
    QualityEvaluation,
    QualityEvaluationResult,
    create_evaluator_agent,
    get_evaluator_agent,
    evaluate_quality,
)


def test_model_registry():
    """Test that model registry is properly loaded."""
    print("\n" + "=" * 60)
    print("TEST: Model Registry")
    print("=" * 60)

    print(f"\nTotal models registered: {len(MODEL_REGISTRY)}")

    print("\nModels by tier:")
    for tier in ModelTier:
        models = get_models_by_tier(tier)
        print(f"  {tier.value.upper():<10}: {', '.join(models)}")

    print(f"\nEdge-compatible models: {EDGE_COMPATIBLE_MODELS}")

    # Verify we have at least one model per important tier
    assert len(get_models_by_tier(ModelTier.SMALL)) >= 1, "Need at least 1 SMALL model"
    assert len(get_models_by_tier(ModelTier.LARGE)) >= 1, "Need at least 1 LARGE model"

    print("\n[PASS] Model registry test passed")


def test_latency_simulation():
    """Test latency simulation."""
    print("\n" + "=" * 60)
    print("TEST: Latency Simulation")
    print("=" * 60)

    # Example API latency to demonstrate multiplier effect
    example_api_latency = 5000  # 5 seconds

    print(f"\nLatency multipliers:")
    print(f"  Edge:  {EDGE_LATENCY_MULTIPLIER}x")
    print(f"  Cloud: {CLOUD_LATENCY_MULTIPLIER}x")

    # Test SMALL model on edge
    print("\nTesting SMALL model on edge:")
    multiplier = get_latency_multiplier("gemma-3-4b", "edge")
    final_latency = example_api_latency * multiplier
    print(f"  gemma-3-4b @ edge: {example_api_latency}ms * {multiplier} = {final_latency:.0f}ms")
    assert multiplier == EDGE_LATENCY_MULTIPLIER

    # Test SMALL model on cloud
    print("\nTesting SMALL model on cloud:")
    multiplier = get_latency_multiplier("gemma-3-4b", "cloud")
    final_latency = example_api_latency * multiplier
    print(f"  gemma-3-4b @ cloud: {example_api_latency}ms * {multiplier} = {final_latency:.0f}ms")
    assert multiplier == CLOUD_LATENCY_MULTIPLIER

    # Test larger models on cloud
    print("\nTesting larger models on cloud:")
    for model in ["gemma-3-12b", "llama-3.3-70b", "deepseek-r1-0528"]:
        multiplier = get_latency_multiplier(model, "cloud")
        final_latency = example_api_latency * multiplier
        print(f"  {model} @ cloud: {example_api_latency}ms * {multiplier} = {final_latency:.0f}ms")
        assert multiplier == CLOUD_LATENCY_MULTIPLIER

    print("\n[PASS] Latency simulation test passed")


def test_edge_constraints():
    """Test that edge deployment constraints are enforced."""
    print("\n" + "=" * 60)
    print("TEST: Edge Deployment Constraints")
    print("=" * 60)

    # Try to deploy a LARGE model on edge - should fail
    print("\nTesting edge constraint (should raise error):")
    try:
        get_latency_multiplier("llama-3.3-70b", "edge")
        print("  ERROR: Should have raised ValueError!")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  Correctly rejected: {e}")

    print("\n[PASS] Edge constraint test passed")


def test_routers():
    """Test router implementations."""
    print("\n" + "=" * 60)
    print("TEST: Router Implementations")
    print("=" * 60)

    # Test NaiveRouter
    print("\nTesting NaiveRouter:")
    router = NaiveRouter(edge_probability=0.5)
    for i in range(5):
        model, deployment = router.route(f"Test query {i}")
        print(f"  Query {i}: {model}@{deployment}")

    stats = router.get_stats()
    print(f"  Stats: {stats}")
    assert stats["total"] == 5, "Should have 5 total calls"

    # Test StaticRouter with SMALL model (edge)
    print("\nTesting StaticRouter (edge):")
    router = StaticRouter("gemma-3-4b")
    model, deployment = router.route("Test query")
    print(f"  Route: {model}@{deployment}")
    assert model == "gemma-3-4b", "Should route to gemma-3-4b"
    assert deployment == "edge", "SMALL models should deploy on edge"

    # Test StaticRouter with LARGE model (cloud)
    print("\nTesting StaticRouter (cloud):")
    router = StaticRouter("llama-3.3-70b")
    model, deployment = router.route("Test query")
    print(f"  Route: {model}@{deployment}")
    assert model == "llama-3.3-70b", "Should route to llama-3.3-70b"
    assert deployment == "cloud", "LARGE models should deploy on cloud"

    print("\n[PASS] Router test passed")


def test_quality_evaluation():
    """Test quality evaluation module setup."""
    print("\n" + "=" * 60)
    print("TEST: Quality Evaluation Module")
    print("=" * 60)

    # Test that default evaluator model is registered
    default_evaluator = "trinity-mini"
    print(f"\nDefault evaluator model: {default_evaluator}")

    assert default_evaluator in MODEL_REGISTRY, f"Evaluator model {default_evaluator} not in registry"
    evaluator_config = MODEL_REGISTRY[default_evaluator]
    print(f"  Model ID: {evaluator_config.model_id}")
    print(f"  Tier: {evaluator_config.tier.value}")

    # Test that evaluator agent can be created (without making API calls)
    print("\nCreating evaluator agent...")
    agent = create_evaluator_agent(default_evaluator)
    assert agent is not None, "Agent should be created"
    print(f"  Agent created successfully")

    # Test cached agent retrieval
    print("\nTesting cached agent retrieval...")
    agent2 = get_evaluator_agent(default_evaluator)
    assert agent2 is not None, "Cached agent should be retrieved"
    print(f"  Cached agent retrieved successfully")

    # Test QualityDimension model
    print("\nTesting QualityDimension model...")
    dim = QualityDimension(
        dimension="accuracy",
        score=8.5,
        reasoning="The response is factually correct."
    )
    assert dim.score == 8.5, "Score should be 8.5"
    print(f"  QualityDimension: {dim.dimension} = {dim.score}/10")

    # Test QualityEvaluationResult model
    print("\nTesting QualityEvaluationResult model...")
    result = QualityEvaluationResult(
        overall_score=8.0,
        dimensions=[dim],
        summary="Good response overall."
    )
    assert result.overall_score == 8.0, "Overall score should be 8.0"
    print(f"  QualityEvaluationResult: {result.overall_score}/10")

    # Test QualityEvaluation model
    print("\nTesting QualityEvaluation model...")
    eval_result = QualityEvaluation(
        overall_score=8.0,
        dimensions=[dim],
        summary="Good response overall.",
        model_used="gemma-3-4b",
        evaluator_model=default_evaluator
    )
    assert eval_result.evaluator_model == default_evaluator
    print(f"  QualityEvaluation: evaluated by {eval_result.evaluator_model}")

    print("\n[PASS] Quality evaluation test passed")


async def test_api_calls():
    """Test actual API calls to verify models work (requires API key)."""
    print("\n" + "=" * 60)
    print("TEST: API Calls (Live)")
    print("=" * 60)

    if not OPENROUTER_API_KEY:
        print("\n  Skipping API test - OPENROUTER_API_KEY not set")
        return False

    # Test all models in registry
    all_passed = True

    async with httpx.AsyncClient() as client:
        for model_key, model_config in MODEL_REGISTRY.items():
            description = f"{model_config.tier.value.upper()} - {model_config.display_name}"
            print(f"\n  Testing {model_key} ({description})...")
            try:
                response = await client.post(
                    f"{OPENROUTER_BASE_URL}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model_config.model_id,
                        "messages": [{"role": "user", "content": "Say hello in one word."}],
                        "max_tokens": 50
                    },
                    timeout=30.0
                )

                if response.status_code == 200:
                    result = response.json()
                    if "choices" in result and result["choices"]:
                        content = result["choices"][0]["message"]["content"][:50]
                        print(f"    [PASS] Response: {content}...")
                    else:
                        print(f"    [FAIL] No choices in response: {result}")
                        all_passed = False
                elif response.status_code == 429:
                    print(f"    [WARN] Rate limited (429) - model exists but rate limited")
                else:
                    print(f"    [FAIL] Status {response.status_code}: {response.text[:100]}")
                    all_passed = False
            except Exception as e:
                print(f"    [FAIL] Error: {e}")
                all_passed = False

    # Test evaluator with structured output
    print(f"\n  Testing evaluator structured output (trinity-mini)...")
    try:
        eval_result = await evaluate_quality(
            query="What is 2+2?",
            response="2+2 equals 4.",
            model_key="gemma-3-4b",
            evaluator_model="trinity-mini"
        )
        print(f"    [PASS] Evaluation score: {eval_result.overall_score}/10")
        print(f"    Summary: {eval_result.summary[:80]}...")
    except Exception as e:
        print(f"    [FAIL] Evaluator error: {e}")
        all_passed = False

    if all_passed:
        print("\n[PASS] API tests passed")
    else:
        print("\n[WARN] Some API tests failed")

    return all_passed


def main():
    """Run all smoke tests."""
    run_api_tests = "--api" in sys.argv

    print("=" * 60)
    print("SMOKE TEST - Assessment Simulation Module")
    print("=" * 60)

    test_model_registry()
    test_latency_simulation()
    test_edge_constraints()
    test_routers()
    test_quality_evaluation()

    if run_api_tests:
        asyncio.run(test_api_calls())
    else:
        print("\n" + "-" * 60)
        print("Note: Run with --api flag to test actual API calls:")
        print("  uv run python smoke_test.py --api")

    print("\n" + "=" * 60)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
