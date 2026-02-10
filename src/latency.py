"""
Latency Simulation Module.

Simulates edge vs cloud deployment latency using simple multipliers on actual API latency.

LIMITATIONS (candidates should ignore these):
1. Does not account for input/output token count
2. Does not model network conditions realistically
3. Does not simulate cold starts
4. Does not account for batching effects
5. Does not model geographic latency
"""

from typing import Literal

from .model_registry import (
    MODEL_REGISTRY,
    ModelTier,
    EDGE_COMPATIBLE_MODELS,
)


# Simple latency multipliers:
# - Edge: 0.2x (local inference, no network round-trip)
# - Cloud: 1.0x (actual API latency)
EDGE_LATENCY_MULTIPLIER = 0.2
CLOUD_LATENCY_MULTIPLIER = 1.0


def get_latency_multiplier(
    model_key: str,
    deployment: Literal["edge", "cloud"]
) -> float:
    """
    Get latency multiplier for a model deployment.

    Args:
        model_key: Key from MODEL_REGISTRY
        deployment: "edge" or "cloud"

    Returns:
        Multiplier to apply to actual API latency:
        - Edge: 0.2x (5x faster than cloud)
        - Cloud: 1.0x (actual API latency)

    Raises:
        ValueError: If model_key is unknown or if non-SMALL model is deployed on edge
    """
    if model_key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_key}. Available: {list(MODEL_REGISTRY.keys())}"
        )

    model_config = MODEL_REGISTRY[model_key]

    # Validate edge deployment constraint: only SMALL models on edge
    if deployment == "edge" and model_config.tier != ModelTier.SMALL:
        raise ValueError(
            f"Model '{model_key}' ({model_config.tier.value}) cannot be deployed on edge. "
            f"Only SMALL tier models are edge-compatible: {EDGE_COMPATIBLE_MODELS}"
        )

    if deployment == "edge":
        return EDGE_LATENCY_MULTIPLIER
    else:
        return CLOUD_LATENCY_MULTIPLIER
