# Agentic Router Solution

## Overview

This solution implements an **agentic routing pipeline** that intelligently routes user queries to the optimal LLM model and deployment location (edge vs. cloud) using a multi-step LLM-powered workflow. The pipeline uses LangGraph to structure a sequential decision-making process where each step is an LLM call with structured output.

## Architecture Overview

### Pipeline Diagram

```
┌─────────────────┐
│   User Query    │
└────────┬────────┘
         │
┌────────▼────────────────────────┐
│  Step 1: Intent Classification │  LLM analyzes query type
│  - simple_factual               │  (coding, reasoning, factual, etc.)
│  - multi_hop_factual            │
│  - complex_reasoning            │
│  - coding                       │
└────────┬────────────────────────┘
         │
┌────────▼────────────────────────┐
│  Step 2: Mission Criticality    │  LLM scores importance
│  Score: 0.0 - 1.0               │  (0 = low stakes, 1 = must be correct)
│  Reasoning: ...                 │
└────────┬────────────────────────┘
         │
┌────────▼────────────────────────┐
│  Step 3: Latency Criticality    │  LLM scores time-sensitivity
│  Score: 0.0 - 1.0               │  (0 = can wait, 1 = needs instant response)
│  Reasoning: ...                 │
└────────┬────────────────────────┘
         │
┌────────▼────────────────────────┐
│  Step 4: Routing Decision       │  LLM selects optimal model
│  - model_key: "gemma-3-4b"      │  based on intent + scores
│  - deployment: "edge" or "cloud"│
│  - reasoning: ...               │
└────────┬────────────────────────┘
         │
┌────────▼────────────────────────┐
│  Guardrails & Validation        │  Ensure valid routing decision
│  - Model exists in registry     │  - Edge only for SMALL tier
│  - Model in allowed list        │  - Fallback if invalid
└────────┬────────────────────────┘
         │
┌────────▼────────────────────────┐
│  Final Route: (model, deploy)   │
└──────────────────────────────────┘
```

### Implementation Structure

The solution is organized into modular components:

```
solutions2/
├── models.py              # Pydantic models for structured outputs
├── prompts.py             # Prompt templates for each pipeline step
├── utils.py              # JSON parsing and utility functions
├── llm_client.py         # OpenRouter API client with retry logic
├── pipeline_steps.py     # Core logic for each pipeline step
├── pipeline.py           # LangGraph pipeline definition
└── custom_router.py      # CustomRouter class wrapping the pipeline
```

### Key Design Decisions

1. **LangGraph for State Management**: Used LangGraph's `StateGraph` to manage pipeline state and ensure sequential execution of steps.

2. **Structured Output via Pydantic**: Each pipeline step returns a Pydantic model, ensuring type safety and validation.

3. **Prompt-based JSON Schema**: Since free-tier models don't support native structured output, we embed JSON schemas in prompts and parse responses with robust error handling.

4. **Model Key vs Model ID Separation**: The pipeline works with registry keys (e.g., `"gemma-3-4b"`) internally, and converts to full model IDs (e.g., `"google/gemma-3-4b-it:free"`) only when making API calls.

5. **Dynamic Model Filtering**: The routing decision step filters candidate models based on intent classification to improve routing quality and efficiency.

6. **Robust Error Handling**: Each step has retry logic, fallback defaults, and validation to ensure the router never crashes.

## Prompt Design Decisions

### Step 1: Intent Classification

**Goal**: Accurately categorize queries into one of four intent types.

**Design Decisions**:
- **Explicit Categories**: Defined seven clear categories with concise definitions:
  - `simple_factual`: Single, direct fact lookup
  - `multi_hop_factual`: Requires synthesizing multiple facts
  - `complex_reasoning`: Logical deduction, math, proofs
  - `coding`: Code generation, debugging, API usage
  - `creative`: Creative writing, storytelling, brainstorming, artistic content generation
  - `planning`: Task planning, project organization, step-by-step strategy development
  - `opinion-advice`: Seeking recommendations, personal opinions, or advice on decisions

- **JSON Example**: Included a concrete JSON example in the prompt to guide small models.

- **Strict Output Rules**: Explicitly instructed the model to:
  - Output ONLY valid JSON (no markdown wrapping)
  - Keep reasoning concise (max 20 words)
  - Match intent values exactly

**Rationale**: Small models need clear, explicit instructions with examples to produce consistent structured output.

### Step 2: Mission Criticality Scoring

**Goal**: Score how critical it is to get a high-quality answer (0.0 = low stakes, 1.0 = must be correct).

**Design Decisions**:
- **Context-Aware Scoring**: The prompt considers:
  - Decision-making implications
  - Potential harm from wrong answers
  - Factual vs. exploratory nature
  - User's apparent need for accuracy

- **Intent Integration**: Uses the classified intent to inform scoring (e.g., coding queries often have higher mission scores).

- **Concrete Examples**: Included examples of high vs. low mission-criticality queries.

**Rationale**: Mission score directly influences model tier selection, so accuracy is crucial for routing quality.

### Step 3: Latency Criticality Scoring

**Goal**: Score how time-sensitive the response is (0.0 = can wait, 1.0 = needs instant response).

**Design Decisions**:
- **User Experience Focus**: Considers:
  - Interactive vs. batch processing context
  - User's apparent urgency
  - Query complexity vs. expected response time

- **Intent Integration**: Uses intent to inform scoring (e.g., simple factual queries often have higher latency scores).

- **Balanced Scoring**: Encourages nuanced scoring rather than binary high/low values.

**Rationale**: Latency score influences deployment location (edge vs. cloud) and model selection, affecting user experience.

### Step 4: Routing Decision

**Goal**: Select the optimal `(model_key, deployment)` combination based on intent, mission score, and latency score.

**Design Decisions**:
- **Hierarchical Policy**: Implemented a clear routing policy with priority order:
  1. High mission (≥0.80) → LARGE/REASONING tier on cloud
  2. Complex reasoning with high mission (≥0.70) → REASONING tier on cloud
  3. High latency (≥0.70) + low mission (≤0.50) → SMALL tier on edge
  4. Otherwise → MEDIUM tier on cloud

- **Dynamic Model Filtering**: Filters candidate models based on intent before presenting to the LLM:
  - `simple_factual` → SMALL tier only
  - `multi_hop_factual` → MEDIUM + SMALL tiers
  - `complex_reasoning` → REASONING + LARGE tiers
  - `coding` → MEDIUM + LARGE tiers

- **Registry Key Only**: Only shows registry keys (e.g., `"gemma-3-4b"`) in the prompt, never full model IDs, to prevent confusion.

- **Tier Table**: Presents models organized by tier for easy selection.

- **Explicit JSON Example**: Includes a concrete example of the expected output format.

**Rationale**: The routing decision is the most critical step. Clear policy, filtered candidates, and explicit examples help the LLM make optimal choices.

### Common Prompt Patterns

All prompts follow these patterns for consistency:

1. **Role Definition**: "You are an expert [Agent Type]"
2. **Task Description**: Clear, concise task statement
3. **Output Format**: JSON schema with explicit rules
4. **Examples**: Concrete JSON examples (not schema definitions)
5. **Strict Rules**: Explicit do's and don'ts
6. **Schema Reference**: JSON schema at the end (marked as "for validation only")

## Key Results

### Quality Comparison

**Overall Mean Quality Scores:**
- **AgenticRouter**: 7.40/10 (best)
- **StaticRouter (small)**: 6.60/10
- **NaiveRouter**: 5.60/10

**Quality by Query Category:**

| Category | AgenticRouter | NaiveRouter | StaticRouter (small) |
|----------|---------------|-------------|---------------------|
| coding | 9.0/10 | 5.0/10 | 6.0/10 |
| complex | 5.0/10 | 5.0/10 | 8.0/10 |
| moderate | 7.0/10 | 5.0/10 | 6.0/10 |
| reasoning | 9.0/10 | 7.0/10 | 5.0/10 |
| simple | 7.0/10 | 6.0/10 | 8.0/10 |

**Key Findings:**
- AgenticRouter outperforms baselines overall (+1.8 points vs. NaiveRouter, +0.8 vs. StaticRouter)
- Strong performance on coding (9.0/10) and reasoning (9.0/10) queries
- Weak performance on complex queries (5.0/10) - needs improvement
- Consistent performance across simple and moderate queries

### Routing Decisions Analysis

**Model Selection Patterns:**
- **Simple queries**: `gemma-3-4b@edge` (SMALL tier, edge deployment)
- **Moderate queries**: `gemma-3-4b@edge` (SMALL tier, edge deployment)
- **Complex queries**: `gemma-3-12b@cloud` (MEDIUM tier, cloud deployment)
- **Reasoning queries**: `gemma-3-12b@cloud` (MEDIUM tier, cloud deployment)
- **Coding queries**: `gemma-3-12b@cloud` (MEDIUM tier, cloud deployment)

**Mission/Latency Score Distributions:**
- **Mission Score by Model Tier**:
  - MEDIUM tier: 0.73 (higher mission → higher tier models)
  - SMALL tier: 0.48 (lower mission → lower tier models)
- **Latency Score by Deployment**:
  - Edge deployment: 0.65 (higher latency score → edge for speed)
  - Cloud deployment: 0.47 (lower latency score → cloud for quality)

### Latency Analysis

**Overall Latency Comparison:**
- AgenticRouter has higher total latency due to routing overhead
- Average routing overhead: ~8,260ms per query
- Step-by-step breakdown:
  - Intent classification: ~2,400ms
  - Mission criticality: ~2,000ms
  - Latency criticality: ~1,800ms
  - Routing decision: ~2,000ms

**Routing Overhead Breakdown:**
- Routing overhead represents a significant portion of total latency
- The overhead is primarily from the 4 sequential LLM calls in the pipeline
- Rate limiting can add additional delays (10-40 seconds in some cases)

**Comparison with Baselines:**
- AgenticRouter adds ~8-10 seconds of routing overhead per query
- This overhead is acceptable when quality improvement justifies it
- For simple queries, the overhead may exceed the inference time

### Cost Analysis

**Routing Cost:**
- Average routing cost per query: ~$0.000144 - $0.000161
- Total routing cost for 5 queries: ~$0.0007 - $0.0008
- Cost breakdown by step:
  - Each pipeline step (4 steps) costs ~$0.000036 - $0.000040
  - Using free-tier models keeps routing costs minimal

**Inference Cost:**
- Inference costs vary by model selected
- SMALL tier models: ~$0.000001 - $0.000005 per query
- MEDIUM tier models: ~$0.000024 - $0.000302 per query

**Total Cost vs. Baselines:**
- AgenticRouter has higher total cost due to routing overhead
- However, routing costs are minimal (~$0.00015 per query)
- The quality improvement (7.40 vs 5.60 vs NaiveRouter) justifies the additional cost
- Cost per quality point: Very low due to free-tier models


### Overhead Mitigation Strategies

1. **Use Paid/Faster Models**: Switch from free-tier models to paid models for routing steps:
   - Paid models have higher rate limits and faster response times
   - Could reduce routing overhead from ~8 seconds to ~2-3 seconds
   - Trade-off: Higher cost per routing decision

2. **Parallel Execution**: Run mission and latency scoring in parallel:
   - These steps are independent and can execute simultaneously
   - Would reduce overhead by ~2-4 seconds (saving the time of one sequential step)
   - Requires minimal code changes to pipeline structure

3. **Caching**: Cache intent classifications for similar queries:
   - Many queries have similar intents (e.g., "What is X?" → simple_factual)
   - Cache intent results to skip Step 1 for repeated query patterns
   - Could reduce overhead by ~2 seconds for cached queries

4. **Token Optimization**: Further reduce `max_tokens` for routing steps:
   - Current: 200 tokens per step
   - Could reduce to 100-150 tokens for faster responses
   - Trade-off: May reduce prompt clarity or require more retries

## Known Limitations

### 1. Rate Limiting

**Issue**: Free-tier models have strict rate limits (`free-models-per-min`), causing:
- Long wait times (10-40 seconds)
- Failed queries after retries
- Inconsistent latency measurements

**Impact**: 
- Routing overhead includes rate limit waits
- Some queries fail after 3 retries
- Evaluation results may be skewed by rate limits

**Mitigation**:
- Reduced `MAX_CONCURRENT` in benchmarks
- Exponential backoff with reset time detection
- Consider using paid models for production

### 2. Intent Classification Accuracy

**Issue**: Intent misclassification can cascade to poor routing decisions.

**Impact**: If a complex reasoning query is misclassified as `simple_factual`, it will be routed to a SMALL tier model, resulting in poor quality.

**Mitigation**:
- Clear intent definitions in prompts
- Confidence scores (though not currently used in routing)
- Fallback to broader model pool if confidence is low (future work)

### 3. Sequential Execution

**Issue**: Pipeline steps run sequentially, adding latency.

**Impact**: Total routing overhead = sum of all step latencies.

**Mitigation**:
- Mission and latency scoring could run in parallel (independent of each other)
- Would reduce overhead by ~200-500ms

### 4. JSON Parsing Robustness

**Issue**: Small models sometimes return invalid JSON or schema definitions instead of instances.

**Impact**: Requires retries and fallback defaults, adding latency.

**Mitigation**:
- Robust JSON extraction with markdown stripping
- Explicit examples in prompts
- Retry logic with improved prompts on failure


## Future Improvements

1. **Parallel Execution**: Run mission and latency scoring in parallel
2. **Model Performance Feedback**: Track which models perform well for which intents
3. **Confidence-Based Routing**: Use intent confidence to adjust model pool
4. **Caching**: Cache intent classifications for similar queries
5. **Cost-Aware Routing**: More explicit cost considerations in routing policy
6. **Edge Deployment Optimization**: Better edge model selection for low-latency scenarios


## Conclusion

This solution implements a routing pipeline using LangGraph and structured LLM outputs. The pipeline makes intelligent routing decisions based on query intent, mission criticality, and latency requirements.

### Key Achievements

1. **Quality Improvement**: AgenticRouter achieves 7.40/10 average quality, outperforming NaiveRouter (5.60/10) by +1.8 points and StaticRouter (6.60/10) by +0.8 points.

2. **Strong Performance on Specialized Queries**: Excellent performance on coding (9.0/10) and reasoning (9.0/10) queries, demonstrating the value of intelligent routing.

3. **Coherent Pipeline Logic**: Mission and latency scores align with routing decisions - high latency queries route to edge, high mission queries route to higher-tier models.

4. **Minimal Cost Impact**: Routing costs are negligible (~$0.00015 per query) due to free-tier models, making the quality improvement cost-effective.

