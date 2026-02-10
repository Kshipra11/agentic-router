"""
Custom Router Implementation.

================================================================================
CANDIDATE INSTRUCTIONS
================================================================================

This is the file where you should implement your final routing strategy.
Extend the BaseRouter class and implement the required methods.

You may add additional files in this solutions/ folder as needed:
- Helper modules (e.g., classifier.py, features.py)
- Data files (e.g., training data, lookup tables)
- Tests for your implementation

However, your FINAL router implementation must be in this file (custom_router.py)
and the main class must be named `CustomRouter`.

================================================================================
GOAL
================================================================================

Create a router that intelligently routes queries to the best model/deployment
combination based on query characteristics, optimizing for quality, latency, and cost.

================================================================================
BASELINE LIMITATIONS TO ADDRESS (pick one or more)
================================================================================

1. Query complexity detection - route simple queries to fast/cheap models
2. Intent classification - match query type to model strengths
3. Cost optimization - minimize cost while meeting quality thresholds
4. Latency optimization - meet latency SLAs by smart edge/cloud decisions
5. Quality optimization - maximize quality within budget constraints
6. Adaptive routing - learn from past performance
7. Fallback strategies - handle model failures gracefully

================================================================================
AVAILABLE RESOURCES
================================================================================

- MODEL_REGISTRY: Dict of model configurations (tier, costs, etc.)
- ModelTier: Enum (SMALL, MEDIUM, LARGE, REASONING)
- EDGE_COMPATIBLE_MODELS: List of models that can run on edge (SMALL tier only)
- get_latency_multiplier(): Get latency multiplier for model/deployment combo

================================================================================
CONSTRAINTS
================================================================================

- Only SMALL tier models can be deployed on edge
- Edge deployment has 0.2x latency multiplier (5x faster than cloud)
- Cloud deployment has 1.0x latency multiplier (baseline)
"""

from typing import Optional, List, Tuple, Dict, Any

from src.router import BaseRouter
from src.model_registry import (
    MODEL_REGISTRY,
    ModelTier,
    ModelConfig,
    EDGE_COMPATIBLE_MODELS,
    CLOUD_MODELS,
    get_models_by_tier,
)


class CustomRouter(BaseRouter):
    """
    Custom router implementation for intelligent query routing.

    TODO: Implement your routing strategy here.

    Your router should decide:
    1. Which model to use for a given query
    2. Where to deploy (edge or cloud)

    Consider these factors:
    - Query complexity (simple vs complex)
    - Query type/intent (coding, reasoning, factual, creative, etc.)
    - Latency requirements
    - Cost constraints
    - Quality requirements
    """

    def __init__(
        self,
        # TODO: Add any configuration parameters your router needs
    ):
        """
        Initialize the custom router.

        TODO: Implement initialization logic.

        Consider initializing:
        - Configuration parameters
        - Query classifier (if using ML-based classification)
        - Model performance cache (if doing adaptive routing)
        - Any pre-computed routing rules
        """
        super().__init__()

        # TODO: Initialize your router's state here
        pass

    @property
    def name(self) -> str:
        """
        Return a descriptive name for this router.

        TODO: Return a name that describes your routing strategy.

        Examples:
        - "ComplexityAware"
        - "IntentBased"
        - "CostOptimized"
        - "AdaptiveQuality"
        """
        return "Custom"

    def route(
        self,
        query: str,
        available_models: Optional[List[str]] = None
    ) -> Tuple[str, str]:
        """
        Route a query to a model and deployment.

        TODO: Implement your routing logic here.

        Args:
            query: The user query to route
            available_models: Optional list of models to choose from
                            (defaults to all models if None)

        Returns:
            Tuple of (model_key, deployment) where:
            - model_key: Key from MODEL_REGISTRY (e.g., "gemma-3-4b", "llama-3.3-70b")
            - deployment: "edge" or "cloud"

        IMPORTANT CONSTRAINTS:
        - Only SMALL tier models can be deployed on "edge"
        - All models can be deployed on "cloud"
        - If you return an invalid combination, the benchmark will fail
        """
        self.call_count += 1
        pass

    # =========================================================================
    # OPTIONAL HELPER METHODS
    # Implement these if they help your routing strategy
    # =========================================================================

    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze a query to extract features for routing decisions.

        TODO: Implement query analysis (optional but recommended).

        Suggested features to extract:
        - length: Number of characters/words/tokens
        - complexity_score: Estimated complexity (simple/moderate/complex)
        - intent: Query type (factual, reasoning, coding, creative, etc.)
        - domain: Subject area (math, science, general, etc.)
        - requires_reasoning: Whether deep reasoning is needed
        - requires_code: Whether code generation is needed

        Returns:
            Dict with query features
        """
        # TODO: Implement query analysis
        # Example:
        # return {
        #     "length": len(query),
        #     "word_count": len(query.split()),
        #     "complexity": self._estimate_complexity(query),
        #     "intent": self._classify_intent(query),
        #     "has_code_keywords": any(kw in query.lower() for kw in ["code", "function", "implement"]),
        # }

        pass

    def _estimate_complexity(self, query: str) -> str:
        """
        Estimate query complexity.

        TODO: Implement complexity estimation (optional).

        Approaches:
        - Rule-based: keyword matching, length thresholds
        - ML-based: trained classifier
        - Hybrid: rules + simple heuristics

        Returns:
            One of: "simple", "moderate", "complex"
        """
        # TODO: Implement complexity estimation
        # Example rule-based approach:
        # if len(query.split()) < 10:
        #     return "simple"
        # elif any(kw in query.lower() for kw in ["explain", "analyze", "compare"]):
        #     return "complex"
        # else:
        #     return "moderate"

        pass

    def _classify_intent(self, query: str) -> str:
        """
        Classify the intent/type of query.

        TODO: Implement intent classification (optional).

        Suggested categories:
        - "factual": Simple fact lookup
        - "reasoning": Requires logical reasoning
        - "coding": Code generation/debugging
        - "creative": Creative writing
        - "analysis": Data/text analysis
        - "math": Mathematical problems

        Returns:
            Intent category string
        """
        # TODO: Implement intent classification
        # Example keyword-based approach:
        # query_lower = query.lower()
        # if any(kw in query_lower for kw in ["write code", "implement", "function"]):
        #     return "coding"
        # elif any(kw in query_lower for kw in ["why", "explain", "reason"]):
        #     return "reasoning"
        # else:
        #     return "factual"

        pass

    def _select_tier(self, query_features: Dict[str, Any]) -> ModelTier:
        """
        Select the appropriate model tier based on query features.

        TODO: Implement tier selection logic (optional).

        Guidelines:
        - SMALL: Simple queries, low latency requirements
        - MEDIUM: Moderate complexity, balanced quality/speed
        - LARGE: Complex queries requiring high quality
        - REASONING: Queries requiring deep reasoning/math

        Args:
            query_features: Features extracted from _analyze_query()

        Returns:
            ModelTier enum value
        """
        # TODO: Implement tier selection
        # Example:
        # complexity = query_features.get("complexity", "moderate")
        # intent = query_features.get("intent", "factual")
        #
        # if intent == "reasoning" or complexity == "complex":
        #     return ModelTier.REASONING
        # elif complexity == "simple":
        #     return ModelTier.SMALL
        # else:
        #     return ModelTier.MEDIUM

        pass

    def _select_model(
        self,
        tier: ModelTier,
        available_models: Optional[List[str]] = None
    ) -> str:
        """
        Select a specific model from the target tier.

        TODO: Implement model selection within tier (optional).

        Considerations:
        - Cost: Choose cheaper model if quality requirements are met
        - Availability: Filter by available_models if provided
        - Specialization: Some models may be better for certain tasks

        Args:
            tier: Target model tier
            available_models: Optional filter list

        Returns:
            model_key from MODEL_REGISTRY
        """
        # TODO: Implement model selection
        # Example:
        # tier_models = get_models_by_tier(tier)
        # if available_models:
        #     tier_models = [m for m in tier_models if m in available_models]
        #
        # if not tier_models:
        #     # Fallback to any available model
        #     return available_models[0] if available_models else "gemma-3-4b"
        #
        # # Select cheapest model in tier
        # return min(tier_models, key=lambda m: MODEL_REGISTRY[m].cost_per_million_input)

        pass

    def update_from_feedback(
        self,
        query: str,
        model_key: str,
        deployment: str,
        quality_score: float,
        latency_ms: float
    ) -> None:
        """
        Update router state based on feedback from a completed request.

        TODO: Implement adaptive learning (optional, for advanced implementations).

        This method is called after each request completes with the actual
        quality and latency observed. Use this to:
        - Track model performance over time
        - Adjust routing decisions based on observed outcomes
        - Implement online learning

        Args:
            query: The original query
            model_key: Model that was used
            deployment: Deployment that was used
            quality_score: Quality score (0-10) from evaluation
            latency_ms: Actual latency in milliseconds
        """
        # TODO: Implement feedback processing for adaptive routing
        # Example:
        # key = f"{model_key}@{deployment}"
        # if key not in self.model_performance_cache:
        #     self.model_performance_cache[key] = {"quality": [], "latency": []}
        # self.model_performance_cache[key]["quality"].append(quality_score)
        # self.model_performance_cache[key]["latency"].append(latency_ms)

        pass


# =============================================================================
# ALTERNATIVE ROUTER TEMPLATES
# Uncomment and modify one of these if it better fits your approach
# =============================================================================

# class ComplexityAwareRouter(BaseRouter):
#     """Router that routes based on query complexity."""
#
#     def __init__(self, complexity_threshold: int = 50):
#         super().__init__()
#         self.complexity_threshold = complexity_threshold
#
#     @property
#     def name(self) -> str:
#         return "ComplexityAware"
#
#     def route(self, query: str, available_models: Optional[List[str]] = None) -> Tuple[str, str]:
#         self.call_count += 1
#         # Simple heuristic: longer queries are more complex
#         if len(query) < self.complexity_threshold:
#             model_key, deployment = "gemma-3-4b", "edge"
#         else:
#             model_key, deployment = "llama-3.3-70b", "cloud"
#         self.routing_history.append((query[:50], model_key, deployment))
#         return (model_key, deployment)


# class IntentBasedRouter(BaseRouter):
#     """Router that routes based on query intent."""
#
#     INTENT_MODEL_MAP = {
#         "coding": "llama-3.3-70b",
#         "reasoning": "deepseek-r1-0528",
#         "simple": "gemma-3-4b",
#         "default": "mistral-small-24b",
#     }
#
#     def __init__(self):
#         super().__init__()
#
#     @property
#     def name(self) -> str:
#         return "IntentBased"
#
#     def _detect_intent(self, query: str) -> str:
#         query_lower = query.lower()
#         if any(kw in query_lower for kw in ["code", "function", "implement", "debug"]):
#             return "coding"
#         elif any(kw in query_lower for kw in ["prove", "derive", "calculate", "solve"]):
#             return "reasoning"
#         elif len(query.split()) < 10:
#             return "simple"
#         return "default"
#
#     def route(self, query: str, available_models: Optional[List[str]] = None) -> Tuple[str, str]:
#         self.call_count += 1
#         intent = self._detect_intent(query)
#         model_key = self.INTENT_MODEL_MAP.get(intent, "mistral-small-24b")
#         deployment = "edge" if MODEL_REGISTRY[model_key].tier == ModelTier.SMALL else "cloud"
#         self.routing_history.append((query[:50], model_key, deployment))
#         return (model_key, deployment)
