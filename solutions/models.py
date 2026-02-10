"""
Pydantic models for the agentic routing pipeline.

This module defines all the structured output models used by each step
of the routing pipeline. Each model represents the output of an LLM call
with validation constraints.

"""

from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field


# ============================================================================
# Enums and Type Aliases
# ============================================================================

class QueryIntent(str, Enum):
    """Intent categories for query classification."""
    SIMPLE_FACTUAL = "simple_factual"
    MULTI_HOP_FACTUAL = "multi_hop_factual"
    COMPLEX_REASONING = "complex_reasoning"
    CODING = "coding"
    CREATIVE = "creative"
    PLANNING = "planning"
    OPINION_ADVICE = "opinion_advice"
    # OTHER = "other"


Deployment = Literal["edge", "cloud"]


# ============================================================================
# Pipeline Step Output Models
# ============================================================================

class IntentClassification(BaseModel):
    """
    Output of the intent classification step.
    
    Classifies what type of query this is (coding, reasoning, factual, etc.)
    """
    intent: QueryIntent = Field(
        description="The primary intent category of the query. Choose from: simple_factual (basic fact lookup), multi_hop_factual (requires multiple facts), complex_reasoning (needs logical reasoning), coding (code generation/debugging), troubleshooting (problem solving), creative (creative writing), planning (planning/organization), opinion_advice (opinions/recommendations), other (doesn't fit above categories)"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the intent classification. 0.0 = uncertain, 1.0 = very confident. Use 0.7-1.0 for clear cases, 0.4-0.7 for ambiguous queries"
    )
    reasoning: str = Field(
        description="Brief explanation (1-2 sentences) of why this intent was chosen. Mention key words or patterns that led to this classification"
    )


class MissionCriticality(BaseModel):
    """
    Output of the mission-criticality scoring step.
    
    Scores how important it is to get a high-quality, correct answer.
    High scores = user is making important decisions, wrong answer could cause harm.
    Low scores = casual query, exploratory, low stakes.
    """
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Mission-criticality score from 0.0 to 1.0. Use 0.8-1.0 for: critical decisions, financial/medical/legal queries, safety-related, high-stakes business decisions. Use 0.5-0.8 for: important but not critical, factual accuracy matters. Use 0.0-0.5 for: casual/exploratory queries, entertainment, low-stakes questions where wrong answer is harmless"
    )
    reasoning: str = Field(
        description="Brief explanation (1-2 sentences) of why this score was assigned. Mention what makes this query high/low stakes, or what consequences a wrong answer might have"
    )


class LatencyCriticality(BaseModel):
    """
    Output of the latency-criticality scoring step.
    
    Scores how time-sensitive the response is.
    High scores = user needs instant response (interactive, real-time).
    Low scores = can wait, thoroughness more important than speed.
    """
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Latency-criticality score from 0.0 to 1.0. Use 0.8-1.0 for: real-time interactions, quick lookups, time-sensitive decisions, user waiting for response. Use 0.5-0.8 for: moderate time sensitivity, user expects reasonable response time. Use 0.0-0.5 for: deep analysis queries, research questions, thoroughness more important than speed, user can wait"
    )
    reasoning: str = Field(
        description="Brief explanation (1-2 sentences) of why this score was assigned. Mention time-sensitive indicators (real-time, quick lookup) or indicators that thoroughness is more important than speed"
    )


class RoutingDecision(BaseModel):
    """
    Output of the final routing step.
    
    Contains the selected model and deployment location based on
    the intent, mission score, and latency score.
    """
    model_key: str = Field(
        description="The model key from MODEL_REGISTRY to use for this query. Must be one of the allowed model keys provided in the prompt. Choose based on: tier (SMALL for simple, MEDIUM for moderate, LARGE for complex, REASONING for reasoning tasks), cost efficiency, and query requirements"
    )
    deployment: Deployment = Field(
        description="Deployment location: 'edge' for low latency (only SMALL tier models can be on edge, 0.2x latency multiplier), 'cloud' for higher quality (all models can be on cloud, 1.0x latency multiplier). Choose 'edge' when latency_score is high and mission_score is low. Choose 'cloud' for high quality requirements"
    )
    reasoning: str = Field(
        description="Brief explanation (1-2 sentences) of why this model and deployment were chosen. Mention how intent, mission_score, and latency_score influenced the decision"
    )