"""
Custom Router Implementation.

This module provides the CustomRouter class that wraps the LangGraph pipeline
and implements the BaseRouter interface required by the benchmarking system.

The router:
1. Invokes the LangGraph pipeline for each query
2. Extracts the routing decision from the pipeline result
3. Applies guardrails to ensure valid routing decisions
4. Returns (model_key, deployment) tuple as required by BaseRouter
"""

import logging
import time
from typing import Optional, Tuple, List, Dict, Any

from src.router import BaseRouter
from src.model_registry import MODEL_REGISTRY, get_edge_compatible_models, CLOUD_MODELS

from .pipeline import build_routing_graph

# Set up logging
logger = logging.getLogger(__name__)


class CustomRouter(BaseRouter):
    """
    Agentic router using LangGraph pipeline.
    
    This router uses a multi-step LLM-powered pipeline to intelligently route
    queries to the optimal model and deployment location. The pipeline:
    
    1. Classifies query intent (coding, reasoning, factual, etc.)
    2. Scores mission-criticality (how important is correctness?)
    3. Scores latency-criticality (how time-sensitive is the response?)
    4. Makes final routing decision (selects model + deployment)
    
    The router applies guardrails to ensure all routing decisions are valid
    and never crashes, even if the pipeline fails.
    
    Example:
        >>> router = CustomRouter()
        >>> model_key, deployment = router.route("Write a Python function")
        >>> print(f"{model_key}@{deployment}")
        mistral-small-24b@cloud
    """
    
    def __init__(self):
        """
        Initialize the CustomRouter.
        
        Builds the LangGraph pipeline and sets up internal state.
        """
        super().__init__()
        self.graph = build_routing_graph()
        self.last_trace = None  # Store last pipeline execution trace for debugging
        self.trace_history: List[Dict[str, Any]] = []  # Store structured traces for evaluation
    
    @property
    def name(self) -> str:
        """
        Return the router name for identification in benchmarks.
        
        Returns:
            Router name string
        """
        return "AgenticRouter"
    
    def route(
        self,
        query: str,
        available_models: Optional[list[str]] = None
    ) -> Tuple[str, str]:
        """
        Route a query to a model and deployment using the agentic pipeline.
        
        This method:
        1. Invokes the LangGraph pipeline with the query
        2. Extracts the routing decision from the pipeline result
        3. Applies guardrails to ensure the decision is valid
        4. Returns (model_key, deployment) tuple
        
        Args:
            query: The user query to route
            available_models: Optional list of model keys to choose from.
                            If None, defaults to all CLOUD_MODELS.
                            
        Returns:
            Tuple of (model_key, deployment) where:
            - model_key: Key from MODEL_REGISTRY (e.g., "gemma-3-4b", "mistral-small-24b")
            - deployment: "edge" or "cloud"
            
        Note:
            The router never crashes. If the pipeline fails, it returns a safe
            fallback routing decision.
        """
        self.call_count += 1
        
        # Determine allowed models (default to all cloud models if not specified)
        allowed_models = available_models or CLOUD_MODELS
        
        # Start timing for routing overhead
        routing_start_time = time.time()
        
        logger.info("=" * 80)
        logger.info(f"[CUSTOM_ROUTER] Routing query (call #{self.call_count})")
        logger.info(f"Query: {query[:100]}{'...' if len(query) > 100 else ''}")
        logger.info(f"Allowed models: {len(allowed_models)} models")
        logger.info("=" * 80)
        
        try:
            # Invoke the LangGraph pipeline
            pipeline_result = self.graph.invoke({
                "query": query,
                "allowed_models": allowed_models,
            })
            
            # Calculate routing overhead (time spent in pipeline)
            routing_overhead_ms = (time.time() - routing_start_time) * 1000
            
            # Extract costs and timing from traces
            traces = pipeline_result.get("traces", {})
            total_routing_cost = 0.0
            step_timings = {}
            
            for step_name, step_trace in traces.items():
                if isinstance(step_trace, dict):
                    # Extract cost
                    step_cost = step_trace.get("cost", 0.0)
                    total_routing_cost += step_cost
                    
                    # Extract timing
                    step_duration = step_trace.get("step_duration_s", 0.0)
                    step_timings[step_name] = step_duration * 1000  # Convert to ms
            
            # Store structured trace for evaluation
            trace_data = {
                "query": query,
                "timestamp": time.time(),
                "routing_overhead_ms": routing_overhead_ms,
                "routing_cost": total_routing_cost,
                "step_timings": step_timings,
                "intent": pipeline_result.get("intent"),
                "mission_score": (
                    pipeline_result.get("mission_criticality").score 
                    if pipeline_result.get("mission_criticality") else None
                ),
                "latency_score": (
                    pipeline_result.get("latency_criticality").score 
                    if pipeline_result.get("latency_criticality") else None
                ),
                "decision": pipeline_result.get("routing_decision"),
                "full_state": pipeline_result,  # Complete LangGraph state
            }
            
            # Store trace for debugging/inspection
            self.last_trace = pipeline_result
            self.trace_history.append(trace_data)
            
            # Extract routing decision
            decision = pipeline_result.get("routing_decision")
            
            if decision is None:
                logger.warning("[CUSTOM_ROUTER] Pipeline returned None decision. Using fallback.")
                return self._get_fallback_route(allowed_models)
            
            # Extract model_key and deployment from decision
            model_key = decision.model_key
            deployment = decision.deployment
            
            logger.info(f"[CUSTOM_ROUTER] Pipeline decision: {model_key}@{deployment}")
            logger.info(f"[CUSTOM_ROUTER] Routing overhead: {routing_overhead_ms:.2f}ms, Cost: ${total_routing_cost:.6f}")
            
        except Exception as e:
            logger.error(f"[CUSTOM_ROUTER] Pipeline execution failed: {e}")
            logger.warning("[CUSTOM_ROUTER] Using fallback route")
            # Still record trace with error
            routing_overhead_ms = (time.time() - routing_start_time) * 1000
            trace_data = {
                "query": query,
                "timestamp": time.time(),
                "routing_overhead_ms": routing_overhead_ms,
                "routing_cost": 0.0,
                "step_timings": {},
                "error": str(e),
            }
            self.trace_history.append(trace_data)
            return self._get_fallback_route(allowed_models)
        
        # Apply guardrails to ensure valid routing decision
        model_key, deployment = self._apply_guardrails(
            model_key=model_key,
            deployment=deployment,
            allowed_models=allowed_models
        )
        
        # Record in history
        self.routing_history.append((query[:50], model_key, deployment))
        
        logger.info(f"[CUSTOM_ROUTER] ✓ Final route: {model_key}@{deployment}")
        logger.info("=" * 80)
        
        return (model_key, deployment)
    
    def _apply_guardrails(
        self,
        model_key: str,
        deployment: str,
        allowed_models: list[str]
    ) -> Tuple[str, str]:
        """
        Apply guardrails to ensure routing decision is valid.
        
        This function ensures:
        1. model_key exists in MODEL_REGISTRY
        2. model_key is in the allowed_models list
        3. Edge deployment only for SMALL tier models
        4. Always returns a valid (model_key, deployment) tuple
        
        Args:
            model_key: Model key from pipeline decision
            deployment: Deployment location from pipeline decision
            allowed_models: List of allowed model keys
            
        Returns:
            Tuple of (validated_model_key, validated_deployment)
        """
        original_model = model_key
        original_deployment = deployment
        
        # Find the registry key - could be a direct key or a model_id
        registry_key = None
        if model_key in MODEL_REGISTRY:
            registry_key = model_key
        else:
            # Try to find it as a model_id
            for k, v in MODEL_REGISTRY.items():
                if v.model_id == model_key:
                    registry_key = k
                    break
        
        # Guardrail 1: Ensure model exists in registry
        if registry_key is None:
            logger.warning(
                f"[CUSTOM_ROUTER] Model '{model_key}' not in MODEL_REGISTRY. "
                f"Using fallback."
            )
            registry_key = allowed_models[0] if allowed_models else "gemma-3-4b"
            deployment = "cloud"
        
        # Guardrail 2: Ensure model is in allowed list (allowed_models contains registry keys)
        if registry_key not in allowed_models:
            logger.warning(
                f"[CUSTOM_ROUTER] Model '{registry_key}' not in allowed_models. "
                f"Using fallback."
            )
            registry_key = allowed_models[0] if allowed_models else "gemma-3-4b"
            deployment = "cloud"
        
        # Guardrail 3: Edge deployment constraint using get_edge_compatible_models()
        if deployment == "edge":
            edge_compatible_keys = get_edge_compatible_models()
            if registry_key not in edge_compatible_keys:
                logger.warning(
                    f"[CUSTOM_ROUTER] Model '{registry_key}' is not edge-compatible. "
                    f"Switching to cloud."
                )
                deployment = "cloud"
        
        # Log if we made any changes
        if registry_key != original_model or deployment != original_deployment:
            logger.info(
                f"[CUSTOM_ROUTER] Guardrails applied: "
                f"{original_model}@{original_deployment} -> {registry_key}@{deployment}"
            )
        
        return (registry_key, deployment)
    
    def _get_fallback_route(self, allowed_models: list[str]) -> Tuple[str, str]:
        """
        Get a safe fallback routing decision.
        
        This is used when the pipeline fails or returns an invalid decision.
        The fallback strategy:
        1. Try to use a SMALL tier model on edge (fastest, cheapest)
        2. Fall back to MEDIUM tier on cloud (balanced)
        3. Last resort: first allowed model on cloud
        
        Args:
            allowed_models: List of allowed model keys
            
        Returns:
            Tuple of (model_key, deployment)
        """
        # Strategy 1: Try SMALL tier on edge (if available and allowed)
        edge_compatible_keys = get_edge_compatible_models()
        for model_key in edge_compatible_keys:
            if model_key in allowed_models:
                logger.info(f"[CUSTOM_ROUTER] Fallback: Using {model_key}@edge")
                return (model_key, "edge")
        
        # Strategy 2: Use first allowed model on cloud
        if allowed_models:
            fallback_model = allowed_models[0]
            logger.info(f"[CUSTOM_ROUTER] Fallback: Using {fallback_model}@cloud")
            return (fallback_model, "cloud")
        
        # Strategy 3: Last resort - use default
        logger.warning("[CUSTOM_ROUTER] Fallback: Using default gemma-3-4b@cloud")
        return ("gemma-3-4b", "cloud")
    
    def get_routing_metrics(self) -> Dict[str, Any]:
        """
        Get aggregated routing metrics for evaluation.
        
        Returns:
            Dictionary with:
            - total_queries: Number of queries routed
            - avg_routing_overhead_ms: Average time spent in pipeline
            - total_routing_cost: Total cost of all routing LLM calls
            - avg_routing_cost_per_query: Average routing cost per query
            - step_timings_avg: Average time per pipeline step
        """
        if not self.trace_history:
            return {
                "total_queries": 0,
                "avg_routing_overhead_ms": 0.0,
                "total_routing_cost": 0.0,
                "avg_routing_cost_per_query": 0.0,
                "step_timings_avg": {},
            }
        
        total_overhead = sum(t["routing_overhead_ms"] for t in self.trace_history)
        total_cost = sum(t["routing_cost"] for t in self.trace_history)
        
        # Aggregate step timings
        step_timings_sum = {}
        step_timings_count = {}
        for trace in self.trace_history:
            for step, timing in trace.get("step_timings", {}).items():
                step_timings_sum[step] = step_timings_sum.get(step, 0) + timing
                step_timings_count[step] = step_timings_count.get(step, 0) + 1
        
        step_timings_avg = {
            step: step_timings_sum[step] / step_timings_count[step]
            for step in step_timings_sum
        }
        
        n = len(self.trace_history)
        return {
            "total_queries": n,
            "avg_routing_overhead_ms": total_overhead / n,
            "total_routing_cost": total_cost,
            "avg_routing_cost_per_query": total_cost / n,
            "step_timings_avg": step_timings_avg,
        }