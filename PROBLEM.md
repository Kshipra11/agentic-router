# Agentic AI Engineer Take-Home Assessment
## Junior / Mid-Level Track

## Overview

**Role**: Agentic AI Engineer
**Level**: Junior / Mid-Level
**Time Estimate**: 4-6 hours
**Submission**: GitHub repository with code, documentation, and results

## Problem Statement

Build an **agentic routing pipeline** that intelligently routes user queries to the right model and deployment (edge vs. cloud) using a multi-step LLM-powered workflow.

Unlike the LLM Researcher track (which uses trained classifiers), your router should use **LLM agents at routing time** to analyze and classify queries through a structured pipeline.

### What You'll Build

A static agentic routing graph using **Pydantic AI** or **LangGraph** with the following pipeline:

```
┌─────────────────┐
│   User Query    │
└────────┬────────┘
         │
┌────────▼────────┐
│     Intent      │     "What is this query about?"
│  Classification │     (coding, reasoning, factual, creative, etc.)
└────────┬────────┘
         │
┌────────▼────────┐
│    Mission-     │     "How important is getting this right?"
│   Criticality   │     (high = needs best model, low = cheap model is fine)
│    Scoring      │
└────────┬────────┘
         │
┌────────▼────────┐
│    Latency-     │     "How time-sensitive is this?"
│   Criticality   │     (high = needs fast response, low = can wait)
│    Scoring      │
└────────┬────────┘
         │
┌────────▼────────┐
│   Final Route   │     Uses intent + mission score + latency score
│    Decision     │     to select (model_key, deployment)
└────────┬────────┘
         │
┌────────▼────────┐
│  (model, deploy) │
└─────────────────┘
```

Each step is an LLM call with structured output. The final step combines all signals to make the routing decision.

### Constraints

- **Agentic workflows required** - Each pipeline step should be an LLM call (using a fast, cheap model like a SMALL tier model)
- **No training, profiling, or RL** - This is a pure inference-time pipeline
- **Coding assistants allowed** - You may use AI coding assistants to help write code
- **Use Pydantic AI or LangGraph** - Structure your pipeline as a graph
- You must work with the provided model registry (OpenRouter integration)
- **All your work must go in `solutions/` and `notebooks/` folders only** - do not modify files in `src/` or other directories

## Available Tools & Resources

### Frameworks

| Framework | Docs |
|-----------|------|
| **Pydantic AI** | https://ai.pydantic.dev/ |
| **LangGraph** | https://langchain-ai.github.io/langgraph/ |

### Provided Infrastructure

- **Model Registry**: 10 LLM models across 4 tiers (SMALL, MEDIUM, LARGE, REASONING)
- **Latency Simulation**: Edge (0.2x latency) vs Cloud (1.0x latency)
- **Quality Evaluator**: LLM-as-a-judge for measuring response quality
- **Baseline Routers**: NaiveRouter (random) and StaticRouter (always same model)
- **Benchmarking**: Full benchmark suite with comparison tables

## Your Tasks

### Task 1: Define Structured Output Models

Define Pydantic models for each pipeline step's output.

```python
from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional

class QueryIntent(str, Enum):
    """Define your intent categories"""
    SIMPLE_FACTUAL = "simple_factual"
    COMPLEX_REASONING = "complex_reasoning"
    CODING = "coding"
    CREATIVE = "creative"
    # ... add more as you see fit

class IntentClassification(BaseModel):
    """Output of the intent classification step"""
    intent: QueryIntent
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str

class MissionCriticality(BaseModel):
    """Output of the mission-criticality scoring step"""
    score: float = Field(ge=0.0, le=1.0, description="0 = low stakes, 1 = must be correct")
    reasoning: str

class LatencyCriticality(BaseModel):
    """Output of the latency-criticality scoring step"""
    score: float = Field(ge=0.0, le=1.0, description="0 = can wait, 1 = needs instant response")
    reasoning: str

class RoutingDecision(BaseModel):
    """Output of the final routing step"""
    model_key: str
    deployment: str  # "edge" or "cloud"
    reasoning: str
```

**Deliverable**: Structured output models in `solutions/custom_router.py` or a separate `solutions/models.py`

### Task 2: Build Pipeline Steps

Implement each step as an LLM call with structured output. Use a fast, cheap model (SMALL tier) for the routing agents themselves.

**Step 1 - Intent Classification**: Given a query, classify its intent. The LLM should analyze what kind of task the query requires.

**Step 2 - Mission-Criticality Scoring**: Given a query and its intent, score how critical it is to get a high-quality answer. Consider:
- Is the user making a decision based on this?
- Could a wrong answer cause harm?
- Is this a factual question with a definitive answer?
- Is this casual/exploratory?

**Step 3 - Latency-Criticality Scoring**: Given a query and its intent, score how time-sensitive the response is. Consider:
- Is this a quick lookup or a deep analysis?
- Would the user expect an instant response?
- Is latency more important than thoroughness?

**Step 4 - Final Routing Decision**: Given the intent, mission score, and latency score, select the best `(model_key, deployment)` combination from the model registry. The agent should reason about:
- High mission + low latency → Large/Reasoning model on cloud
- Low mission + high latency → Small model on edge
- High mission + high latency → Medium model on cloud (balanced)
- Low mission + low latency → Small model on edge (cheapest)

**Deliverable**: Pipeline implementation in `solutions/custom_router.py`

### Task 3: Wire It Into a Graph

Use Pydantic AI or LangGraph to structure the pipeline as a graph with proper state management.

```python
# Example with LangGraph (you can also use Pydantic AI)
from langgraph.graph import StateGraph
from typing import TypedDict

class RouterState(TypedDict):
    query: str
    intent: Optional[IntentClassification]
    mission_criticality: Optional[MissionCriticality]
    latency_criticality: Optional[LatencyCriticality]
    routing_decision: Optional[RoutingDecision]

def build_routing_graph() -> StateGraph:
    graph = StateGraph(RouterState)

    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("score_mission", score_mission_node)
    graph.add_node("score_latency", score_latency_node)
    graph.add_node("make_decision", make_decision_node)

    graph.set_entry_point("classify_intent")
    graph.add_edge("classify_intent", "score_mission")
    graph.add_edge("score_mission", "score_latency")
    graph.add_edge("score_latency", "make_decision")

    return graph.compile()
```

Then wrap it in `CustomRouter`:

```python
from src.router import BaseRouter

class CustomRouter(BaseRouter):
    def __init__(self):
        super().__init__()
        self.graph = build_routing_graph()

    @property
    def name(self) -> str:
        return "AgenticRouter"

    def route(self, query: str, available_models=None) -> Tuple[str, str]:
        self.call_count += 1
        result = self.graph.invoke({"query": query})
        decision = result["routing_decision"]
        return (decision.model_key, decision.deployment)
```

**Deliverable**: `solutions/custom_router.py` with `CustomRouter` class

### Task 4: Evaluation & Analysis

Benchmark your agentic router against the baselines.

**Required Experiments**:

1. **Quality Comparison**
   - Mean quality score vs. NaiveRouter and StaticRouter
   - Quality by query category (simple, moderate, complex, reasoning, coding)

2. **Routing Decisions Analysis**
   - What models does your router pick for each category?
   - How do mission/latency scores distribute across query types?
   - Show the pipeline reasoning for a few example queries

3. **Latency Analysis**
   - Total latency (including routing overhead from LLM calls)
   - Routing overhead breakdown (how much time is spent in the pipeline itself?)

4. **Cost Analysis**
   - Cost of routing (the LLM calls made by the pipeline agents)
   - Cost of inference (the actual model calls)
   - Total cost vs. baselines

5. **Pipeline Coherence**
   - Do the intermediate scores (mission/latency) make intuitive sense?
   - Show examples of the full pipeline trace for different query types

**Deliverable**: `notebooks/evaluation.ipynb` with visualizations and analysis

## Where to Put Your Work

**All your code and files must go in `solutions/` and `notebooks/` only.** Do not modify the provided `src/`, `data/`, or root-level files.

Your final router must be in `solutions/custom_router.py`:
```
solutions/
├── custom_router.py      # REQUIRED: CustomRouter wrapping your graph
├── models.py             # Pydantic output models (optional, can be in custom_router.py)
├── pipeline.py           # Pipeline step implementations (optional)
└── ...
```

## Evaluation Criteria

| Criterion | Weight | What We Look For |
|-----------|--------|------------------|
| **Pipeline Design** | 25% | Clear step separation, good structured outputs, sensible prompts |
| **Graph Implementation** | 25% | Proper use of Pydantic AI or LangGraph, state management |
| **Routing Quality** | 25% | Does the pipeline make sensible routing decisions? |
| **Analysis & Evaluation** | 25% | Thorough benchmarking, pipeline coherence analysis |

## Submission Requirements

1. **Your submission should include**:
   ```
   solutions/
   ├── custom_router.py      # Your router (REQUIRED)
   └── ...                   # Additional helper files

   notebooks/
   └── evaluation.ipynb

   SOLUTION.md               # Your writeup (see below)
   ```

2. **SOLUTION.md** must include:
   - Architecture overview with pipeline diagram
   - Prompt design decisions for each step
   - Key results (quality, latency, cost vs. baselines)
   - Analysis of routing overhead
   - Known limitations

3. **Reproducibility**:
   - All prompts included in code
   - requirements.txt with pinned versions (if adding new dependencies)
   - Clear instructions to run

## What's NOT Allowed

- **Trained classifiers**: No sklearn, XGBoost, or pre-trained models for routing
- **Profiling or RL**: No offline data collection, no reward optimization
- **Modifying `src/` or other provided files**: Work only in `solutions/` and `notebooks/`

## What IS Allowed

- **Coding assistants**: GitHub Copilot, Claude, ChatGPT for coding help
- **LLM calls at routing time**: This is the entire point - your pipeline uses LLM agents
- **Any agentic framework**: Pydantic AI, LangGraph, or similar
- **Prompt engineering**: Spend time crafting good prompts for each step
- **Additional pipeline steps**: You may add more steps beyond the four required ones

## Post-Submission Discussion Topics

Be prepared to discuss:

1. **Prompt design**: How did you design prompts for each step? What alternatives did you try?
2. **Routing overhead**: The pipeline adds LLM calls before the actual inference. When is this worth it?
3. **Pipeline model choice**: What model did you use for the routing agents? Why?
4. **Error propagation**: What happens if the intent classifier gets it wrong?
5. **Parallelization**: Could mission and latency scoring run in parallel? What are the tradeoffs?
6. **Scaling**: How would this work at 10,000 QPS given the routing overhead?
7. **Adding new models**: A new model is added to the registry. What changes?

## Resources

- Provided code: `src/` folder (model registry, benchmarking, quality evaluation)
- OpenRouter API: https://openrouter.ai/docs
- Pydantic AI: https://ai.pydantic.dev/
- LangGraph: https://langchain-ai.github.io/langgraph/

**Good luck! Focus on clean pipeline design and sensible prompts. The quality of your routing decisions matters more than complexity.**
