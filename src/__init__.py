"""
Shared module for technical assessments.

Provides simulation utilities for latency and quality evaluation,
routing implementations, and benchmarking tools.
"""

# Configuration
from .config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
)

# Model Registry
from .model_registry import (
    MODEL_REGISTRY,
    ModelTier,
    ModelConfig,
    EDGE_COMPATIBLE_MODELS,
    EDGE_MODELS,
    CLOUD_MODELS,
    get_models_by_tier,
    get_edge_compatible_models,
    get_cloud_models,
)

# Latency Simulation
from .latency import (
    get_latency_multiplier,
    EDGE_LATENCY_MULTIPLIER,
    CLOUD_LATENCY_MULTIPLIER,
)

# Quality Evaluation
from .quality import (
    QualityDimension,
    QualityEvaluation,
    QualityEvaluationResult,
    evaluate_quality,
    create_evaluator_agent,
    get_evaluator_agent,
    DEFAULT_EVALUATION_PROMPT,
)

# Routing
from .router import (
    BaseRouter,
    NaiveRouter,
    StaticRouter,
    Router,
)

# Benchmarking
from .benchmarking import (
    BenchmarkResult,
    RoutingBenchmarkResult,
    run_single_benchmark,
    benchmark_router,
    benchmark_all_routers,
    print_benchmark_summary,
    print_router_comparison,
    SAMPLE_QUERIES,
)

__all__ = [
    # Configuration
    "OPENROUTER_API_KEY",
    "OPENROUTER_BASE_URL",
    # Model Registry
    "MODEL_REGISTRY",
    "ModelTier",
    "ModelConfig",
    "EDGE_COMPATIBLE_MODELS",
    "EDGE_MODELS",
    "CLOUD_MODELS",
    "get_models_by_tier",
    "get_edge_compatible_models",
    "get_cloud_models",
    # Latency Simulation
    "get_latency_multiplier",
    "EDGE_LATENCY_MULTIPLIER",
    "CLOUD_LATENCY_MULTIPLIER",
    # Quality Evaluation
    "QualityDimension",
    "QualityEvaluation",
    "QualityEvaluationResult",
    "evaluate_quality",
    "create_evaluator_agent",
    "get_evaluator_agent",
    "DEFAULT_EVALUATION_PROMPT",
    # Routing
    "BaseRouter",
    "NaiveRouter",
    "StaticRouter",
    "Router",
    # Benchmarking
    "BenchmarkResult",
    "RoutingBenchmarkResult",
    "run_single_benchmark",
    "benchmark_router",
    "benchmark_all_routers",
    "print_benchmark_summary",
    "print_router_comparison",
    "SAMPLE_QUERIES",
]
