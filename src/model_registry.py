"""
Model Registry for LLM models.

Defines model tiers, configurations, and the registry of available models.
All models are accessed via OpenRouter API using free tier models.
"""

from typing import Dict, List
from dataclasses import dataclass
from enum import Enum


class ModelTier(Enum):
    """Model capability tiers - affects quality and latency profiles"""
    SMALL = "small"      # Fast, lower quality (e.g., Haiku, Gemini Flash)
    MEDIUM = "medium"    # Balanced (e.g., Sonnet, GPT-4o-mini)
    LARGE = "large"      # High quality, slower (e.g., Opus, GPT-4, Claude 3.5)
    REASONING = "reasoning"  # Specialized reasoning models (o1, DeepSeek R1)


@dataclass
class ModelConfig:
    """Configuration for a specific model"""
    model_id: str                    # OpenRouter model identifier
    display_name: str                # Human-readable name
    tier: ModelTier                  # Capability tier
    cost_per_million_input: float = 0.0   # Emulated USD per 1M input tokens
    cost_per_million_output: float = 0.0  # Emulated USD per 1M output tokens


# Static model registry - VERIFIED FREE MODELS from OpenRouter
# See: https://openrouter.ai/models?q=free
# Note: Free models require opting-in to data training on OpenRouter
# Only includes models verified to work as of testing date
MODEL_REGISTRY: Dict[str, ModelConfig] = {
    # =========================================================================
    # SMALL TIER - Fast, lower quality models (edge-compatible)
    # =========================================================================
    "gemma-3-4b": ModelConfig(
        model_id="google/gemma-3-4b-it:free",
        display_name="Gemma 3 4B",
        tier=ModelTier.SMALL,
        cost_per_million_input=0.05,   # Emulated: $0.05 per 1M input tokens
        cost_per_million_output=0.10   # Emulated: $0.10 per 1M output tokens
    ),
    "llama-3.2-3b": ModelConfig(
        model_id="meta-llama/llama-3.2-3b-instruct:free",
        display_name="Llama 3.2 3B",
        tier=ModelTier.SMALL,
        cost_per_million_input=0.03,   # Emulated: $0.03 per 1M input tokens
        cost_per_million_output=0.06   # Emulated: $0.06 per 1M output tokens
    ),
    "gemma-3n-e4b": ModelConfig(
        model_id="google/gemma-3n-e4b-it:free",
        display_name="Gemma 3N E4B",
        tier=ModelTier.SMALL,
        cost_per_million_input=0.04,   # Emulated: $0.04 per 1M input tokens
        cost_per_million_output=0.08   # Emulated: $0.08 per 1M output tokens
    ),

    # =========================================================================
    # MEDIUM TIER - Balanced models
    # =========================================================================
    "trinity-mini": ModelConfig(
        model_id="arcee-ai/trinity-mini:free",
        display_name="Trinity Mini",
        tier=ModelTier.MEDIUM,
        cost_per_million_input=0.10,   # Emulated: $0.10 per 1M input tokens
        cost_per_million_output=0.30   # Emulated: $0.30 per 1M output tokens
    ),
    "gemma-3-12b": ModelConfig(
        model_id="google/gemma-3-12b-it:free",
        display_name="Gemma 3 12B",
        tier=ModelTier.MEDIUM,
        cost_per_million_input=0.10,   # Emulated: $0.10 per 1M input tokens
        cost_per_million_output=0.30   # Emulated: $0.30 per 1M output tokens
    ),
    "mistral-small-24b": ModelConfig(
        model_id="mistralai/mistral-small-3.1-24b-instruct:free",
        display_name="Mistral Small 3.1 24B",
        tier=ModelTier.MEDIUM,
        cost_per_million_input=0.20,   # Emulated: $0.20 per 1M input tokens
        cost_per_million_output=0.50   # Emulated: $0.50 per 1M output tokens
    ),
    "gemma-3-27b": ModelConfig(
        model_id="google/gemma-3-27b-it:free",
        display_name="Gemma 3 27B",
        tier=ModelTier.MEDIUM,
        cost_per_million_input=0.25,   # Emulated: $0.25 per 1M input tokens
        cost_per_million_output=0.60   # Emulated: $0.60 per 1M output tokens
    ),
    "nemotron-nano": ModelConfig(
        model_id="nvidia/nemotron-3-nano-30b-a3b:free",
        display_name="NVIDIA Nemotron 3 Nano 30B",
        tier=ModelTier.MEDIUM,
        cost_per_million_input=0.35,   # Emulated: $0.35 per 1M input tokens
        cost_per_million_output=0.75   # Emulated: $0.75 per 1M output tokens
    ),

    # =========================================================================
    # LARGE TIER - High quality models
    # =========================================================================
    "llama-3.3-70b": ModelConfig(
        model_id="meta-llama/llama-3.3-70b-instruct:free",
        display_name="Llama 3.3 70B",
        tier=ModelTier.LARGE,
        cost_per_million_input=0.50,   # Emulated: $0.50 per 1M input tokens
        cost_per_million_output=1.00   # Emulated: $1.00 per 1M output tokens
    ),

    # =========================================================================
    # REASONING TIER - Specialized reasoning models
    # =========================================================================
    "deepseek-r1-0528": ModelConfig(
        model_id="deepseek/deepseek-r1-0528:free",
        display_name="DeepSeek R1 0528",
        tier=ModelTier.REASONING,
        cost_per_million_input=2.00,   # Emulated: $2.00 per 1M input tokens
        cost_per_million_output=8.00   # Emulated: $8.00 per 1M output tokens
    ),
}


def get_models_by_tier(tier: ModelTier) -> List[str]:
    """Get all model keys for a given tier."""
    return [k for k, v in MODEL_REGISTRY.items() if v.tier == tier]


def get_edge_compatible_models() -> List[str]:
    """Get models compatible with edge deployment (SMALL tier only)."""
    return get_models_by_tier(ModelTier.SMALL)


def get_cloud_models() -> List[str]:
    """Get all models (all can run on cloud)."""
    return list(MODEL_REGISTRY.keys())


# Convenience lists
EDGE_COMPATIBLE_MODELS = get_edge_compatible_models()
EDGE_MODELS = EDGE_COMPATIBLE_MODELS  # Alias
CLOUD_MODELS = get_cloud_models()
