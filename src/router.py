"""
Router Module for Query Routing.

Provides abstract base class and baseline implementations for routing queries
to appropriate models and deployments.

BASELINE ROUTER LIMITATIONS (candidates may choose to address one or more of these):
1. Completely ignores query content/complexity
2. No learning from past performance
3. No quality consideration
4. No cost optimization
5. No adaptive behavior based on load/latency
6. No fallback/retry logic on failure
"""

import random
from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Tuple

from .model_registry import (
    MODEL_REGISTRY,
    ModelTier,
    EDGE_MODELS,
    CLOUD_MODELS,
)


class BaseRouter(ABC):
    """
    Abstract base class for query routers.

    Candidates should extend this class to implement their own routing strategies.

    A router decides:
    1. Which model to use for a given query
    2. Where to deploy (edge or cloud)

    The routing decision should consider:
    - Query complexity and requirements
    - Model capabilities and costs
    - Latency requirements
    - Quality requirements
    """

    def __init__(self):
        self.call_count = 0
        self.routing_history: List[Tuple[str, str, str]] = []  # (query, model, deploy)

    @abstractmethod
    def route(
        self,
        query: str,
        available_models: Optional[List[str]] = None
    ) -> Tuple[str, str]:
        """
        Route a query to a model and deployment.

        Args:
            query: The user query to route
            available_models: Optional list of models to choose from

        Returns:
            Tuple of (model_key, deployment) where deployment is "edge" or "cloud"
        """
        pass

    def get_stats(self) -> Dict[str, int]:
        """Get routing statistics."""
        stats: Dict[str, int] = {"total": self.call_count, "edge": 0, "cloud": 0}
        for _, _, deploy in self.routing_history:
            stats[deploy] = stats.get(deploy, 0) + 1
        return stats

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a descriptive name for this router."""
        pass


class NaiveRouter(BaseRouter):
    """
    Naive baseline router that makes completely random decisions.

    Strategy:
    1. Randomly select deployment: edge or cloud
    2. Randomly select a model compatible with that deployment

    LIMITATIONS (candidates should identify and address these):
    1. Completely ignores query content/complexity
    2. No learning from past performance
    3. No quality consideration
    4. No cost optimization
    5. No adaptive behavior based on load/latency
    6. No fallback/retry logic on failure
    7. Random selection means inconsistent results
    """

    def __init__(self, edge_probability: float = 0.5):
        """
        Args:
            edge_probability: Probability of routing to edge (0.0 to 1.0)
        """
        super().__init__()
        self.edge_probability = edge_probability

    def route(
        self,
        query: str,
        available_models: Optional[List[str]] = None
    ) -> Tuple[str, str]:
        """
        Randomly route a query to a model and deployment.

        Args:
            query: The user query (ignored in naive implementation)
            available_models: Optional list of models to choose from

        Returns:
            Tuple of (model_key, deployment)
        """
        self.call_count += 1

        # Step 1: Randomly choose deployment
        deployment = "edge" if random.random() < self.edge_probability else "cloud"

        # Step 2: Get models compatible with chosen deployment
        if deployment == "edge":
            compatible_models = EDGE_MODELS
        else:
            compatible_models = available_models or CLOUD_MODELS

        # Step 3: Randomly select a model
        model_key = random.choice(compatible_models)

        # Track history (could be used for analysis)
        self.routing_history.append((query[:50], model_key, deployment))

        return (model_key, deployment)

    @property
    def name(self) -> str:
        return "Random"


class StaticRouter(BaseRouter):
    """
    Static router that always routes to a specific model.

    Deployment is determined by model tier:
    - SMALL models -> edge deployment
    - All other tiers -> cloud deployment
    """

    def __init__(self, model_key: str):
        """
        Args:
            model_key: The model to always route to
        """
        super().__init__()

        if model_key not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_key}")

        self.model_key = model_key
        self.model_config = MODEL_REGISTRY[model_key]

        # Determine deployment based on tier
        if self.model_config.tier == ModelTier.SMALL:
            self.deployment = "edge"
        else:
            self.deployment = "cloud"

    def route(
        self,
        query: str,
        available_models: Optional[List[str]] = None
    ) -> Tuple[str, str]:
        """
        Always route to the configured model.

        Args:
            query: The user query (ignored - always uses same model)
            available_models: Ignored in static router

        Returns:
            Tuple of (model_key, deployment)
        """
        self.call_count += 1
        self.routing_history.append((query[:50], self.model_key, self.deployment))
        return (self.model_key, self.deployment)

    @property
    def name(self) -> str:
        return f"Static({self.model_key}@{self.deployment})"


# Type alias for any router
Router = BaseRouter
