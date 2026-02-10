"""
Prompt templates for the agentic routing pipeline.

This module contains all prompt-building functions for each pipeline step.
Prompts are designed to be clear, explicit, and easy for small language models
to understand and follow.
"""

import json
from typing import List, Dict, Optional

from src.model_registry import MODEL_REGISTRY

from .models import (
    IntentClassification,
    MissionCriticality,
    LatencyCriticality,
    RoutingDecision,
    QueryIntent,
)
from .utils import pydantic_to_json_schema_string


def _schema_block(schema: dict) -> str:
    """Format JSON schema as a string for embedding in prompts."""
    return json.dumps(schema, indent=2)


def _tier_table(allowed_model_keys: Optional[List[str]] = None) -> str:
    """Format model registry by tier for inclusion in prompts.
    
    Args:
        allowed_model_keys: Optional list of model keys to filter by. If provided,
            only shows models in this list. If None, shows all models.
    """
    tiers: Dict[str, List[str]] = {"SMALL": [], "MEDIUM": [], "LARGE": [], "REASONING": []}
    for k, cfg in MODEL_REGISTRY.items():
        # Only include if it's in allowed list (if provided)
        if allowed_model_keys is None or k in allowed_model_keys:
            tiers[cfg.tier.name].append(k)
    return (
        f"SMALL: {tiers['SMALL']}\n"
        f"MEDIUM: {tiers['MEDIUM']}\n"
        f"LARGE: {tiers['LARGE']}\n"
        f"REASONING: {tiers['REASONING']}\n"
    )


def build_intent_classification_prompt(query: str) -> str:
    """
    Build prompt for intent classification step.
    
    Args:
        query: The user query to classify
        
    Returns:
        Complete prompt string ready to send to LLM
    """
    schema = IntentClassification.model_json_schema()
    
    prompt = f"""
You are an expert Intent Classification Agent.
Your task is to accurately categorize the user's query into one of the provided intent categories.

**RESPONSE FORMAT:**
Output a JSON object that STRICTLY adheres to this schema:
{_schema_block(schema)}

**CRITICAL RULES FOR OUTPUT:**
1.  **DO NOT include any text outside the JSON object.**
2.  **DO NOT wrap the JSON in markdown code blocks** (e.g., ```json or ```).
3.  `reasoning` must be concise (maximum 20 words).

**EXAMPLE JSON OUTPUT:**
```json
{{
  "intent": "simple_factual",
  "confidence": 0.9,
  "reasoning": "The query asks for a direct fact about a person."
}}
```

**Intent Categories & Definitions:**
- `simple_factual`: Single, direct fact lookup or definition.
- `multi_hop_factual`: Requires synthesizing multiple facts or light comparison/reasoning.
- `complex_reasoning`: Involves logical deduction, math, proofs, constraint satisfaction.
- `coding`: Code generation, explanation, debugging, API usage.

---
**USER QUERY:**
{query}
""".strip()
    
    return prompt


def build_mission_criticality_prompt(query: str, intent: QueryIntent) -> str:
    """
    Build prompt for mission-criticality scoring step.
    
    Args:
        query: The user query
        intent: The classified intent from previous step
        
    Returns:
        Complete prompt string ready to send to LLM
    """
    schema = MissionCriticality.model_json_schema()
    
    prompt = f"""
You are a Mission Criticality Scoring Agent.
Your task is to assess the potential impact of an incorrect answer to the user's query.

**RESPONSE FORMAT:**
Output a JSON object that STRICTLY adheres to this schema:
{_schema_block(schema)}

**CRITICAL RULES FOR OUTPUT:**
1.  **DO NOT include any text outside the JSON object.**
2.  **DO NOT wrap the JSON in markdown code blocks** (e.g., ```json or ```).
3.  `reasoning` must be concise (maximum 20 words).

**EXAMPLE JSON OUTPUT:**
```json
{{
  "score": 0.75,
  "reasoning": "Incorrect information could lead to moderate financial loss."
}}
```

**Mission Score Anchors (0.0 - 1.0):**
- `0.0-0.2`: Casual, recreational, minimal or no real-world consequences.
- `0.3-0.5`: Informational, minor inconvenience or easily correctable if wrong.
- `0.6-0.8`: User relies on information for decisions; wrong answer leads to real, moderate cost.
- `0.9-1.0`: High stakes (medical, legal, financial, safety, irreversible decisions); critical cost if wrong.

---
**INPUTS:**
- **Intent:** {intent.value}
- **User Query:** {query}
""".strip()
    
    return prompt


def build_latency_criticality_prompt(query: str, intent: QueryIntent) -> str:
    """
    Build prompt for latency-criticality scoring step.
    
    Args:
        query: The user query
        intent: The classified intent from previous step
        
    Returns:
        Complete prompt string ready to send to LLM
    """
    schema = LatencyCriticality.model_json_schema()
    
    prompt = f"""
You are a Latency Criticality Scoring Agent.
Your task is to assess how quickly the user expects a response to their query.

**RESPONSE FORMAT:**
Output a JSON object that STRICTLY adheres to this schema:
{_schema_block(schema)}

**CRITICAL RULES FOR OUTPUT:**
1.  **DO NOT include any text outside the JSON object.**
2.  **DO NOT wrap the JSON in markdown code blocks** (e.g., ```json or ```).
3.  `reasoning` must be concise (maximum 20 words).

**EXAMPLE JSON OUTPUT:**
```json
{{
  "score": 0.9,
  "reasoning": "User expects immediate response in an interactive chat."
}}
```

**Latency Score Anchors (0.0 - 1.0):**
- `0.0-0.2`: Non-urgent, can wait for deep analysis or complex processing.
- `0.3-0.5`: Standard interactive request, a few seconds response time is acceptable.
- `0.6-0.8`: User expects a quick response, like chat or simple lookups.
- `0.9-1.0`: Real-time interaction, urgent, part of an interactive loop.

---
**INPUTS:**
- **Intent:** {intent.value}
- **User Query:** {query}
""".strip()
    
    return prompt


def build_routing_decision_prompt(
    query: str,
    intent: QueryIntent,
    mission_score: float,
    latency_score: float,
    allowed_model_keys: List[str],
) -> str:
    """
    Build prompt for final routing decision step.
    
    Args:
        query: The user query
        intent: The classified intent
        mission_score: Mission-criticality score (0.0-1.0)
        latency_score: Latency-criticality score (0.0-1.0)
        allowed_model_keys: List of model keys to choose from (actual model_ids)
        
    Returns:
        Complete prompt string ready to send to LLM
    """
    schema = RoutingDecision.model_json_schema()
    
    prompt = f"""
You are a Routing Decision Agent. Select the BEST model_key and deployment for this query.

OUTPUT FORMAT (return valid JSON only, no markdown):
{{
  "model_key": "gemma-3-4b",
  "deployment": "edge",
  "reasoning": "High latency, low mission: SMALL tier on edge for speed."
}}

ROUTING POLICY (apply in order):

1. If mission_score >= 0.80:
   → LARGE tier model on cloud
   → If intent == complex_reasoning: REASONING tier model on cloud

2. If intent == complex_reasoning AND mission_score >= 0.70:
   → REASONING tier model on cloud

3. If latency_score >= 0.70 AND mission_score <= 0.50:
   → SMALL tier model on edge

4. Otherwise:
   → MEDIUM tier model on cloud

SELECTION CRITERIA:
- Evaluate ALL models in the selected tier
- Choose based on: query requirements, model capabilities, cost efficiency
- Do NOT default to any specific provider
- If multiple models fit, choose the lowest-cost option

INPUTS:
Intent: {intent.value}
Mission Score: {mission_score:.2f}
Latency Score: {latency_score:.2f}
Query: {query}

ALLOWED MODEL KEYS (choose exactly one):
{allowed_model_keys}

MODELS BY TIER:
{_tier_table(allowed_model_keys)}

SCHEMA (reference only):
{_schema_block(schema)}
""".strip()
    
    return prompt