"""
Quality Evaluation Module (LLM-as-a-Judge).

Uses Pydantic AI to evaluate response quality with structured outputs.

LIMITATIONS (candidates should ignore these):
1. Single evaluator bias - no ensemble/voting
2. No calibration against human judgments
3. Evaluation prompt is not optimized
4. No handling of edge cases (refusals, errors)
5. Does not account for task-specific criteria
6. Potential for evaluator model to favor its own outputs
"""

from typing import Dict, List
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from dataclasses import dataclass

from .model_registry import MODEL_REGISTRY


class QualityDimension(BaseModel):
    """Individual quality dimension score"""
    dimension: str = Field(description="Name of the quality dimension")
    score: float = Field(ge=0, le=10, description="Score from 0-10")
    reasoning: str = Field(description="Brief explanation for the score")


class QualityEvaluation(BaseModel):
    """Complete quality evaluation result"""
    overall_score: float = Field(ge=0, le=10, description="Overall quality score")
    dimensions: List[QualityDimension] = Field(description="Individual dimension scores")
    summary: str = Field(description="Brief summary of the evaluation")
    model_used: str = Field(description="Model that generated the response")
    evaluator_model: str = Field(description="Model used for evaluation")


class QualityEvaluationResult(BaseModel):
    """Structured output for quality evaluation (used by Pydantic AI)"""
    overall_score: float = Field(ge=0, le=10, description="Overall quality score from 0-10")
    dimensions: List[QualityDimension] = Field(description="Individual dimension scores")
    summary: str = Field(description="Brief 2-3 sentence summary of the evaluation")


class BatchQualityEvaluationResult(BaseModel):
    """Structured output for batch evaluation of multiple items."""
    evaluations: List[QualityEvaluationResult] = Field(
        description="Evaluation results in the same order as the input items"
    )


# Default evaluation system prompt - candidates may improve this
DEFAULT_EVALUATION_PROMPT = """
You are an expert evaluator assessing the quality of AI assistant responses.

Evaluate responses on these dimensions (0-10 scale):

1. **Accuracy**: Is the information factually correct?
2. **Relevance**: Does it directly address the query?
3. **Completeness**: Does it cover all important aspects?
4. **Clarity**: Is it well-organized and easy to understand?
5. **Helpfulness**: Would this response satisfy the user's need?

Provide an overall score (0-10) that reflects the weighted importance of these dimensions,
along with individual dimension scores and brief reasoning for each.

Be objective and consistent in your evaluations. Consider the context and difficulty
of the original query when assigning scores.
"""

EVALUATOR_SYSTEM_PROMPT = """You are a strict expert evaluator assessing AI assistant responses. Use the FULL 0-10 scale.

## Score Calibration (USE THE FULL RANGE)
- 9-10: Exceptional. Flawless, comprehensive, expert-level. Reserve for truly outstanding responses.
- 7-8: Good. Solid answer with minor gaps or room for improvement.
- 5-6: Adequate. Gets the job done but has notable weaknesses.
- 3-4: Poor. Significant issues, missing key information, or partially incorrect.
- 1-2: Very poor. Mostly wrong, unhelpful, or misses the point.
- 0: Completely wrong or harmful.

## Dimensions to Evaluate
1. accuracy - Factually correct? Any errors or hallucinations? (Be strict - any factual error drops this significantly)
2. relevance - Directly addresses the query? Stays on topic?
3. completeness - Covers all important aspects? What's missing?
4. clarity - Well-organized? Easy to understand?
5. helpfulness - Would this actually help the user?

## Critical Instructions
- DO NOT default to high scores. Most responses are 5-8, not 9-10.
- A "correct but shallow" answer is 5-6, not 8-9.
- Deduct points for: verbosity, missing edge cases, lack of examples when helpful, generic advice.
- Perfect 10s should be rare - the response must be genuinely exceptional."""


def create_evaluator_agent(
    evaluator_model: str = "trinity-mini"
) -> Agent[None, QualityEvaluationResult]:
    """Create a pydantic-ai Agent for quality evaluation with structured output."""
    evaluator_config = MODEL_REGISTRY.get(evaluator_model)
    if not evaluator_config:
        raise ValueError(f"Unknown evaluator model: {evaluator_model}")

    # Use openrouter: prefix with the model ID
    model_string = f"openrouter:{evaluator_config.model_id}"

    return Agent(
        model=model_string,
        system_prompt=EVALUATOR_SYSTEM_PROMPT,
        output_type=QualityEvaluationResult,
        retries=3
    )


# Cache for evaluator agents
_evaluator_agents: Dict[str, Agent[None, QualityEvaluationResult]] = {}


def get_evaluator_agent(
    evaluator_model: str = "trinity-mini"
) -> Agent[None, QualityEvaluationResult]:
    """Get or create a cached evaluator agent."""
    if evaluator_model not in _evaluator_agents:
        _evaluator_agents[evaluator_model] = create_evaluator_agent(evaluator_model)
    return _evaluator_agents[evaluator_model]


async def evaluate_quality(
    query: str,
    response: str,
    model_key: str,
    evaluator_model: str = "trinity-mini"
) -> QualityEvaluation:
    """
    Evaluate response quality using LLM-as-a-judge via Pydantic AI.

    Args:
        query: Original user query
        response: Model response to evaluate
        model_key: Which model generated the response
        evaluator_model: Model to use for evaluation

    Returns:
        QualityEvaluation with scores and reasoning
    """
    agent = get_evaluator_agent(evaluator_model)

    # Build the evaluation prompt
    user_prompt = f"""Evaluate the following response:

## Original Query
{query}

## Response to Evaluate
{response}

Provide scores for accuracy, relevance, completeness, clarity, and helpfulness."""

    # Run the agent with structured output
    result = await agent.run(user_prompt)

    return QualityEvaluation(
        overall_score=result.output.overall_score,
        dimensions=result.output.dimensions,
        summary=result.output.summary,
        model_used=model_key,
        evaluator_model=evaluator_model
    )


@dataclass
class BatchEvalDeps:
    """Dependencies for batch evaluation agent."""
    expected_count: int


def create_batch_evaluator_agent(
    evaluator_model: str
) -> Agent[BatchEvalDeps, BatchQualityEvaluationResult]:
    """Create a batch evaluator agent with output validation."""
    evaluator_config = MODEL_REGISTRY.get(evaluator_model)
    if not evaluator_config:
        raise ValueError(f"Unknown evaluator model: {evaluator_model}")

    model_string = f"openrouter:{evaluator_config.model_id}"

    agent = Agent(
        model=model_string,
        system_prompt=EVALUATOR_SYSTEM_PROMPT,
        output_type=BatchQualityEvaluationResult,
        deps_type=BatchEvalDeps,
        retries=3
    )

    @agent.output_validator
    async def validate_eval_count(
        ctx: RunContext[BatchEvalDeps],
        result: BatchQualityEvaluationResult
    ) -> BatchQualityEvaluationResult:
        """Validate the number of evaluations matches expected count."""
        expected = ctx.deps.expected_count
        actual = len(result.evaluations)
        if actual != expected:
            raise ModelRetry(f"Expected {expected} evaluations but got {actual}. Please provide exactly {expected} evaluations.")
        return result

    return agent


async def _evaluate_batch_chunk(
    items: List[tuple],  # List of (query, response, model_key)
    evaluator_model: str
) -> List[QualityEvaluation]:
    """Evaluate a small batch of items (internal helper)."""
    # Build multi-item prompt
    user_prompt = "Evaluate each of the following responses:\n\n"
    for i, (query, response, _) in enumerate(items, 1):
        user_prompt += f"## Item {i}\n"
        user_prompt += f"**Query:** {query}\n"
        user_prompt += f"**Response:** {response}\n\n"

    user_prompt += f"\nProvide exactly {len(items)} evaluations in the same order as the items above."

    # Create agent with output validator
    agent = create_batch_evaluator_agent(evaluator_model)
    deps = BatchEvalDeps(expected_count=len(items))

    result = await agent.run(user_prompt, deps=deps)

    # Map to QualityEvaluation with model_used metadata
    return [
        QualityEvaluation(
            overall_score=e.overall_score,
            dimensions=e.dimensions,
            summary=e.summary,
            model_used=items[i][2],  # model_key from input tuple
            evaluator_model=evaluator_model
        )
        for i, e in enumerate(result.output.evaluations)
    ]


async def evaluate_quality_batch(
    items: List[tuple],  # List of (query, response, model_key)
    evaluator_model: str = "trinity-mini",
    chunk_size: int = 5
) -> List[QualityEvaluation]:
    """
    Evaluate multiple query-response pairs in batched API calls.

    Splits large batches into smaller chunks to avoid model limitations
    with large tool call outputs.

    Args:
        items: List of (query, response, model_key) tuples
        evaluator_model: Model to use for evaluation
        chunk_size: Number of items to evaluate per API call (default 5)

    Returns:
        List of QualityEvaluation in the same order as input items
    """
    if not items:
        return []

    # Process in chunks to avoid tool_calls validation issues
    all_results = []
    for i in range(0, len(items), chunk_size):
        chunk = items[i:i + chunk_size]
        chunk_num = (i // chunk_size) + 1
        total_chunks = (len(items) + chunk_size - 1) // chunk_size
        print(f"    Evaluating batch {chunk_num}/{total_chunks} ({len(chunk)} items)...")

        chunk_results = await _evaluate_batch_chunk(chunk, evaluator_model)
        all_results.extend(chunk_results)

    return all_results
