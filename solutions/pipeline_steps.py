"""
Pipeline step implementations for the agentic routing system.

This module contains the four core pipeline steps:
1. Intent Classification - Classify the query type
2. Mission Criticality Scoring - Assess importance of correctness
3. Latency Criticality Scoring - Assess time-sensitivity
4. Routing Decision - Select model and deployment

Each step:
- Builds a prompt using prompts.py
- Calls the LLM using llm_client.py
- Returns a validated Pydantic model with debug info
"""

import logging
import time
from typing import List, Tuple, Dict, Any

from src.model_registry import MODEL_REGISTRY, get_models_by_tier, get_edge_compatible_models, ModelTier

from .models import (
    IntentClassification,
    MissionCriticality,
    LatencyCriticality,
    RoutingDecision,
    QueryIntent,
)
from .prompts import (
    build_intent_classification_prompt,
    build_mission_criticality_prompt,
    build_latency_criticality_prompt,
    build_routing_decision_prompt,
)
from .llm_client import call_llm_structured

# Set up logging
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

# Model to use for routing pipeline steps (should be fast and cheap)
# Using free tier models that are stable and support our prompt format
ROUTER_MODEL_ID = "google/gemma-3-4b-it:free"
# ROUTER_MODEL_ID = "google/gemma-3n-e4b-it:free"

# Default parameters for LLM calls
# DEFAULT_MAX_TOKENS = 100
DEFAULT_MAX_TOKENS = 200
DEFAULT_TEMPERATURE = 0.0  # Deterministic outputs
DEFAULT_RETRIES = 2
DEFAULT_TIMEOUT = 60


# ============================================================================
# Pipeline Step Functions
# ============================================================================

def classify_intent(query: str) -> Tuple[IntentClassification, Dict[str, Any]]:
    """
    Step 1: Classify the intent of the user query.
    
    This step analyzes the query to determine what type of task it requires:
    - Is it a simple fact lookup?
    - Does it need reasoning?
    - Is it a coding task?
    - etc.
    
    Args:
        query: The user query to classify
        
    Returns:
        Tuple of (IntentClassification, debug_info) where:
        - IntentClassification: Pydantic model with intent, confidence, reasoning
        - debug_info: Dictionary with API call metadata
        
    Example:
        >>> intent, debug = classify_intent("Write a Python function to sort a list")
        >>> print(intent.intent)
        QueryIntent.CODING
    """
    logger.info("=" * 80)
    logger.info("[PIPELINE STEP 1] Intent Classification")
    logger.info(f"Query: {query[:100]}{'...' if len(query) > 100 else ''}")
    
    # Track step timing
    step_start = time.time()
    
    # Build prompt
    prompt = build_intent_classification_prompt(query)
    
    # Call LLM
    intent_result, debug_info = call_llm_structured(
        model_id=ROUTER_MODEL_ID,
        prompt=prompt,
        output_model=IntentClassification,
        max_tokens=DEFAULT_MAX_TOKENS,
        temperature=DEFAULT_TEMPERATURE,
        retries=DEFAULT_RETRIES,
        timeout_s=DEFAULT_TIMEOUT,
        step_name="INTENT_CLASSIFICATION",
    )
    
    # Add step-level timing
    debug_info["step_duration_s"] = time.time() - step_start
    
    # Log result
    logger.info(f"[PIPELINE STEP 1] ✓ Classified as: {intent_result.intent.value}")
    logger.info(f"  Confidence: {intent_result.confidence:.2f}")
    logger.info(f"  Reasoning: {intent_result.reasoning}")
    logger.info("=" * 80)
    
    return intent_result, debug_info


def score_mission_criticality(
    query: str,
    intent: QueryIntent
) -> Tuple[MissionCriticality, Dict[str, Any]]:
    """
    Step 2: Score the mission-criticality of the query.
    
    This step assesses how important it is to get a correct, high-quality answer.
    High scores indicate critical decisions where wrong answers could cause harm.
    Low scores indicate casual queries where errors are harmless.
    
    Args:
        query: The user query
        intent: The classified intent from Step 1
        
    Returns:
        Tuple of (MissionCriticality, debug_info) where:
        - MissionCriticality: Pydantic model with score (0.0-1.0) and reasoning
        - debug_info: Dictionary with API call metadata
        
    Example:
        >>> mission, debug = score_mission_criticality(
        ...     "What medication should I take?",
        ...     QueryIntent.SIMPLE_FACTUAL
        ... )
        >>> print(mission.score)  # Should be high (0.8-1.0)
        0.9
    """
    logger.info("=" * 80)
    logger.info("[PIPELINE STEP 2] Mission Criticality Scoring")
    logger.info(f"Intent: {intent.value}")
    logger.info(f"Query: {query[:100]}{'...' if len(query) > 100 else ''}")
    
    # Track step timing
    step_start = time.time()
    
    # Build prompt
    prompt = build_mission_criticality_prompt(query, intent)
    
    # Call LLM
    mission_result, debug_info = call_llm_structured(
        model_id=ROUTER_MODEL_ID,
        prompt=prompt,
        output_model=MissionCriticality,
        max_tokens=DEFAULT_MAX_TOKENS,
        temperature=DEFAULT_TEMPERATURE,
        retries=DEFAULT_RETRIES,
        timeout_s=DEFAULT_TIMEOUT,
        step_name="MISSION_CRITICALITY",
    )
    
    # Add step-level timing
    debug_info["step_duration_s"] = time.time() - step_start
    
    # Log result
    logger.info(f"[PIPELINE STEP 2] ✓ Mission Score: {mission_result.score:.2f}/1.0")
    logger.info(f"  Reasoning: {mission_result.reasoning}")
    logger.info("=" * 80)
    
    return mission_result, debug_info


def score_latency_criticality(
    query: str,
    intent: QueryIntent
) -> Tuple[LatencyCriticality, Dict[str, Any]]:
    """
    Step 3: Score the latency-criticality of the query.
    
    This step assesses how time-sensitive the response is.
    High scores indicate the user needs an instant response (real-time, interactive).
    Low scores indicate the user can wait for a thorough, high-quality answer.
    
    Args:
        query: The user query
        intent: The classified intent from Step 1
        
    Returns:
        Tuple of (LatencyCriticality, debug_info) where:
        - LatencyCriticality: Pydantic model with score (0.0-1.0) and reasoning
        - debug_info: Dictionary with API call metadata
        
    Example:
        >>> latency, debug = score_latency_criticality(
        ...     "What time is it?",
        ...     QueryIntent.SIMPLE_FACTUAL
        ... )
        >>> print(latency.score)  # Should be high (0.8-1.0)
        0.9
    """
    logger.info("=" * 80)
    logger.info("[PIPELINE STEP 3] Latency Criticality Scoring")
    logger.info(f"Intent: {intent.value}")
    logger.info(f"Query: {query[:100]}{'...' if len(query) > 100 else ''}")
    
    # Track step timing
    step_start = time.time()
    
    # Build prompt
    prompt = build_latency_criticality_prompt(query, intent)
    
    # Call LLM
    latency_result, debug_info = call_llm_structured(
        model_id=ROUTER_MODEL_ID,
        prompt=prompt,
        output_model=LatencyCriticality,
        max_tokens=DEFAULT_MAX_TOKENS,
        temperature=DEFAULT_TEMPERATURE,
        retries=DEFAULT_RETRIES,
        timeout_s=DEFAULT_TIMEOUT,
        step_name="LATENCY_CRITICALITY",
    )
    
    # Add step-level timing
    debug_info["step_duration_s"] = time.time() - step_start
    
    # Log result
    logger.info(f"[PIPELINE STEP 3] ✓ Latency Score: {latency_result.score:.2f}/1.0")
    logger.info(f"  Reasoning: {latency_result.reasoning}")
    logger.info("=" * 80)
    
    return latency_result, debug_info


def make_routing_decision(
    query: str,
    intent: QueryIntent,
    mission_criticality: MissionCriticality,
    latency_criticality: LatencyCriticality,
) -> Tuple[RoutingDecision, Dict[str, Any]]:
    logger.info("=" * 80)
    logger.info("[PIPELINE STEP 4] Routing Decision")
    logger.info(f"Intent: {intent.value}")
    logger.info(f"Mission Score: {mission_criticality.score:.2f}/1.0")
    logger.info(f"Latency Score: {latency_criticality.score:.2f}/1.0")

    # Get models by tier using helper functions
    small_models = get_models_by_tier(ModelTier.SMALL)
    medium_models = get_models_by_tier(ModelTier.MEDIUM)
    large_models = get_models_by_tier(ModelTier.LARGE)
    reasoning_models = get_models_by_tier(ModelTier.REASONING)

    # Filter candidate models based on intent
    candidate_models: List[str] = []
    if intent == QueryIntent.SIMPLE_FACTUAL:
        candidate_models.extend(small_models)
    elif intent == QueryIntent.MULTI_HOP_FACTUAL:
        candidate_models.extend(medium_models)
        candidate_models.extend(small_models)
    elif intent == QueryIntent.COMPLEX_REASONING:
        candidate_models.extend(medium_models)
        # candidate_models.extend(reasoning_models)
    elif intent == QueryIntent.CODING:
        candidate_models.extend(medium_models)
        candidate_models.extend(large_models)
    else:
        logger.warning(f"Unhandled intent '{intent.value}', defaulting to MEDIUM tier.")
        candidate_models.extend(small_models)
        candidate_models.extend(medium_models)

    
    # Remove duplicates
    candidate_models = list(set(candidate_models))
    
    # Ensure at least one model
    if not candidate_models:
        logger.warning(f"[ROUTING_DECISION] No models found. Using fallback: gemma-3n-e4b.")
        candidate_models = ["gemma-3n-e4b"]
    
    # Pass registry keys directly to the prompt (never expose model IDs)
    allowed_model_keys = [k for k in candidate_models if k in MODEL_REGISTRY]
    
    logger.info(f"Allowed models: {len(allowed_model_keys)} models")
    if len(allowed_model_keys) <= 10:
        logger.info(f"  {allowed_model_keys}")
    else:
        logger.info(f"  {allowed_model_keys[:5]} ... ({len(allowed_model_keys) - 5} more)")

    # Track step timing
    step_start = time.time()

    # Build prompt
    prompt = build_routing_decision_prompt(
        query=query,
        intent=intent,
        mission_score=mission_criticality.score,
        latency_score=latency_criticality.score,
        allowed_model_keys=allowed_model_keys,
    )
    
    # Call LLM
    decision_result, debug_info = call_llm_structured(
        model_id=ROUTER_MODEL_ID,
        prompt=prompt,
        output_model=RoutingDecision,
        max_tokens=DEFAULT_MAX_TOKENS,
        temperature=DEFAULT_TEMPERATURE,
        retries=DEFAULT_RETRIES,
        timeout_s=DEFAULT_TIMEOUT,
        step_name="ROUTING_DECISION",
    )
    
    # Add step-level timing
    debug_info["step_duration_s"] = time.time() - step_start
    
    # Simple validation: ensure registry key is in allowed list and deployment is valid
    decision_key = decision_result.model_key
    
    # Check if it's a registry key (what we expect from LLM)
    if decision_key not in MODEL_REGISTRY:
        logger.warning(f"[ROUTING_DECISION] Model key '{decision_key}' not found in registry. Using fallback.")
        decision_key = "gemma-3n-e4b"
    
    # Check if registry key is in allowed list
    if decision_key not in allowed_model_keys:
        logger.warning(f"[ROUTING_DECISION] Model key '{decision_key}' not in allowed list. Using first allowed model.")
        decision_key = allowed_model_keys[0]
    
    # Validate deployment: edge only for SMALL tier
    final_deployment = decision_result.deployment
    if final_deployment == "edge":
        if decision_key not in get_edge_compatible_models():
            logger.warning(f"[ROUTING_DECISION] Model '{decision_key}' cannot be deployed on edge. Switching to cloud.")
            final_deployment = "cloud"
    
    # Create validated decision (keep as registry key, model_id conversion happens in custom_router)
    decision_result = RoutingDecision(
        model_key=decision_key,
        deployment=final_deployment,
        reasoning=decision_result.reasoning if decision_key == decision_result.model_key and final_deployment == decision_result.deployment else f"Fixed: {decision_result.reasoning}"
    )
    
    # Log result
    logger.info(f"[PIPELINE STEP 4] ✓ Selected: {decision_result.model_key}@{decision_result.deployment}")
    logger.info(f"  Reasoning: {decision_result.reasoning}")
    logger.info("=" * 80)
    
    return decision_result, debug_info


