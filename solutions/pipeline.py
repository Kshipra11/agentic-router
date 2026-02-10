"""
LangGraph pipeline implementation for the agentic routing system.

This module defines the routing pipeline as a LangGraph state machine with
four sequential steps:
1. classify_intent -> 2. score_mission -> 3. score_latency -> 4. make_decision

The pipeline manages state through a TypedDict and executes each step
sequentially, passing results from one step to the next.
"""

import os
from typing import TypedDict, Optional, Dict, Any

from langgraph.graph import StateGraph, END

from src.model_registry import CLOUD_MODELS

from .models import (
    IntentClassification,
    MissionCriticality,
    LatencyCriticality,
    RoutingDecision,
)
from .pipeline_steps import (
    classify_intent,
    score_mission_criticality,
    score_latency_criticality,
    make_routing_decision,
)


# ============================================================================
# Pipeline State Definition
# ============================================================================

class RouterState(TypedDict, total=False):
    """
    State dictionary for the routing pipeline.
    
    This TypedDict defines all the data that flows through the pipeline.
    Each node updates specific fields and passes the state to the next node.
    
    Fields:
        query: The user query to route (required, set at start)
        allowed_models: List of model keys to choose from (optional, defaults to CLOUD_MODELS)
        intent: Intent classification result from Step 1
        mission_criticality: Mission-criticality score from Step 2
        latency_criticality: Latency-criticality score from Step 3
        routing_decision: Final routing decision from Step 4
        traces: Optional debug traces from each step (if debug mode enabled)
    """
    # Input (required)
    query: str
    
    # Configuration (optional)
    allowed_models: Optional[list[str]]
    
    # Pipeline step outputs (set by each node)
    intent: Optional[IntentClassification]
    mission_criticality: Optional[MissionCriticality]
    latency_criticality: Optional[LatencyCriticality]
    routing_decision: Optional[RoutingDecision]
    
    # Debug traces (optional, only if debug mode enabled)
    traces: Optional[Dict[str, Any]]


# ============================================================================
# Debug Mode Helper
# ============================================================================

def _is_debug_enabled() -> bool:
    """
    Check if debug mode is enabled via environment variable.
    
    When enabled, the pipeline captures debug traces from each step
    for inspection and analysis.
    
    Returns:
        True if ROUTER_DEBUG environment variable is set to a truthy value
    """
    debug_env = os.environ.get("ROUTER_DEBUG", "").strip().lower()
    return debug_env in {"1", "true", "yes", "on"}


# ============================================================================
# Graph Node Functions
# ============================================================================

def classify_intent_node(state: RouterState) -> RouterState:
    """
    Graph node: Step 1 - Classify query intent.
    
    This node:
    1. Takes the query from state
    2. Calls classify_intent() pipeline step
    3. Stores the result in state["intent"]
    4. Optionally stores debug traces if debug mode is enabled
    
    Args:
        state: Current pipeline state
        
    Returns:
        Updated state with intent classification result
    """
    query = state["query"]
    
    # Call pipeline step
    intent_result, debug_info = classify_intent(query)
    
    # Update state
    new_state: RouterState = {
        "intent": intent_result,
    }
    
    # Always store traces for evaluation (not just in debug mode)
    traces = dict(state.get("traces") or {})
    traces["intent"] = debug_info
    new_state["traces"] = traces
    
    return new_state


def score_mission_node(state: RouterState) -> RouterState:
    """
    Graph node: Step 2 - Score mission-criticality.
    
    This node:
    1. Takes query and intent from state
    2. Calls score_mission_criticality() pipeline step
    3. Stores the result in state["mission_criticality"]
    4. Optionally stores debug traces if debug mode is enabled
    
    Args:
        state: Current pipeline state (must have "intent" from previous step)
        
    Returns:
        Updated state with mission-criticality score
    """
    query = state["query"]
    intent = state["intent"]
    
    if intent is None:
        raise ValueError("Intent classification must be completed before mission scoring")
    
    # Call pipeline step
    mission_result, debug_info = score_mission_criticality(query, intent.intent)
    
    # Update state
    new_state: RouterState = {
        "mission_criticality": mission_result,
    }
    
    # Always store traces for evaluation (not just in debug mode)
    traces = dict(state.get("traces") or {})
    traces["mission"] = debug_info
    new_state["traces"] = traces
    
    return new_state


def score_latency_node(state: RouterState) -> RouterState:
    """
    Graph node: Step 3 - Score latency-criticality.
    
    This node:
    1. Takes query and intent from state
    2. Calls score_latency_criticality() pipeline step
    3. Stores the result in state["latency_criticality"]
    4. Optionally stores debug traces if debug mode is enabled
    
    Args:
        state: Current pipeline state (must have "intent" from Step 1)
        
    Returns:
        Updated state with latency-criticality score
    """
    query = state["query"]
    intent = state["intent"]
    
    if intent is None:
        raise ValueError("Intent classification must be completed before latency scoring")
    
    # Call pipeline step
    latency_result, debug_info = score_latency_criticality(query, intent.intent)
    
    # Update state
    new_state: RouterState = {
        "latency_criticality": latency_result,
    }
    
    # Store traces for evaluation 
    traces = dict(state.get("traces") or {})
    traces["latency"] = debug_info
    new_state["traces"] = traces
    
    return new_state


def make_decision_node(state: RouterState) -> RouterState:
    """
    Graph node: Step 4 - Make final routing decision.
    
    This node:
    1. Takes query, intent, mission_criticality, latency_criticality from state
    2. Gets allowed_models from state (or defaults to CLOUD_MODELS)
    3. Calls make_routing_decision() pipeline step
    4. Stores the result in state["routing_decision"]
    5. Optionally stores debug traces if debug mode is enabled
    6. Handles errors gracefully with fallback decisions
    
    Args:
        state: Current pipeline state (must have all previous step results)
        
    Returns:
        Updated state with routing decision
    """
    query = state["query"]
    intent = state["intent"]
    mission = state["mission_criticality"]
    latency = state["latency_criticality"]
    
    # Validate required state
    if intent is None:
        raise ValueError("Intent classification must be completed before routing decision")
    if mission is None:
        raise ValueError("Mission-criticality scoring must be completed before routing decision")
    if latency is None:
        raise ValueError("Latency-criticality scoring must be completed before routing decision")
    
    # Get allowed models (default to CLOUD_MODELS if not specified)
    allowed_models = state.get("allowed_models") or CLOUD_MODELS
    
    try:
        # Call pipeline step
        decision_result, debug_info = make_routing_decision(
            query=query,
            intent=intent.intent,
            mission_criticality=mission,
            latency_criticality=latency,
        )
        
        # Update state
        new_state: RouterState = {
            "routing_decision": decision_result,
        }
        
        # Always store traces for evaluation (not just in debug mode)
        traces = dict(state.get("traces") or {})
        traces["decision"] = debug_info
        new_state["traces"] = traces
        
        return new_state
        
    except Exception as e:
        # Handle errors gracefully with fallback decision
        from .models import RoutingDecision
        
        logger = __import__("logging").getLogger(__name__)
        logger.error(f"[PIPELINE] Error in routing decision: {e}")
        logger.warning("[PIPELINE] Using fallback routing decision")
        
        # Create safe fallback decision
        fallback_model = allowed_models[0] if allowed_models else "gemma-3-4b"
        fallback_decision = RoutingDecision(
            model_key=fallback_model,
            deployment="cloud",
            reasoning=f"Fallback due to error: {str(e)}"
        )
        
        new_state: RouterState = {
            "routing_decision": fallback_decision,
        }
        
        # Always store traces for evaluation (not just in debug mode)
        traces = dict(state.get("traces") or {})
        traces["decision"] = {"error": str(e)}
        new_state["traces"] = traces
        
        return new_state


# ============================================================================
# Graph Construction
# ============================================================================

def build_routing_graph():
    """
    Build and compile the LangGraph routing pipeline.
    
    The pipeline is a linear graph:
    classify_intent -> score_mission -> score_latency -> make_decision -> END
    
    Returns:
        Compiled LangGraph StateGraph ready to invoke
    """
    # Create graph
    graph = StateGraph(RouterState)
    
    # Add nodes (each node is a pipeline step)
    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("score_mission", score_mission_node)
    graph.add_node("score_latency", score_latency_node)
    graph.add_node("make_decision", make_decision_node)
    
    # Define edges (linear flow)
    graph.set_entry_point("classify_intent")
    graph.add_edge("classify_intent", "score_mission")
    graph.add_edge("score_mission", "score_latency")
    graph.add_edge("score_latency", "make_decision")
    graph.add_edge("make_decision", END)
    
    # Compile and return
    return graph.compile()
