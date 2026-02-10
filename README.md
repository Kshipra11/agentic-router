# Technical Assessment Package

## Overview

This package contains boilerplate code for your take-home technical assessment.

## Package Structure

```
├── PROBLEM.md                          # Your assessment instructions (START HERE)
├── README.md                           # This file
├── pyproject.toml                      # Project configuration (uv package manager)
├── .env.template                       # Environment variables template (copy to .env)
├── smoke_test.py                       # Quick local test
├── eval.py                             # Full benchmark evaluation
├── src/                                # Framework code (DO NOT MODIFY)
│   ├── __init__.py                     # Module exports
│   ├── config.py                       # Configuration and API settings
│   ├── model_registry.py               # Model definitions and tiers
│   ├── latency.py                      # Latency simulation
│   ├── quality.py                      # Quality evaluation (LLM-as-judge)
│   ├── router.py                       # Router base class and implementations
│   └── benchmarking.py                 # Benchmarking utilities
├── solutions/                          # YOUR IMPLEMENTATION GOES HERE
│   ├── custom_router.py                # Your router implementation (REQUIRED)
│   └── ...                             # Add any additional files as needed
├── notebooks/                          # YOUR ANALYSIS GOES HERE
│   └── ...                             # Jupyter notebooks for evaluation
└── data/
    ├── sample_queries.json             # Sample queries (5 total, 1 per category)
    └── all_queries.json                # Full query set (15 total, 3 per category)
```

## Getting Started

1. **Read `PROBLEM.md`** - This contains your specific assessment instructions
2. **Set up your environment**:

```bash
# Install dependencies with uv
uv sync

# Copy the environment template and add your API key
cp .env.template .env
# Edit .env and add your OPENROUTER_API_KEY

# Run smoke test (no API key needed)
uv run python smoke_test.py

# Run smoke test with API verification (requires OPENROUTER_API_KEY)
uv run python smoke_test.py --api

# Run benchmark with sample queries (5 queries, ~3 routers)
uv run python eval.py

# Run full benchmark with all queries (uses data/all_queries.json)
uv run python eval.py --full
```

## Implementing Your Solution

All your work should go in **two folders only**. Do not modify files in `src/` or other directories.

| Folder | Purpose |
|--------|---------|
| `solutions/` | Code - your router, helpers, trained models |
| `notebooks/` | Analysis - Jupyter notebooks for evaluation |

### Requirements

1. **Main file**: `solutions/custom_router.py`
2. **Main class**: `CustomRouter` (must extend `BaseRouter`)
3. **Required method**: `route(query, available_models) -> (model_key, deployment)`
4. **Writeup**: `SOLUTION.md` in the project root

### Example Usage

```python
from solutions.custom_router import CustomRouter
from src.benchmarking import benchmark_router

# Create your router
router = CustomRouter()

# Benchmark it
results = await benchmark_router(router)
```

## Shared Components

### Module Structure (`src/`)

1. **Config** (`config.py`): API configuration
   - Loads environment variables from `.env`
   - OpenRouter API key and base URL

2. **Model Registry** (`model_registry.py`): Static model configuration
   - All free OpenRouter models (`:free` suffix, no real cost)
   - Model tiers: SMALL (edge-compatible), MEDIUM, LARGE, REASONING
   - Emulated costs per million tokens for benchmarking optimization
   - SMALL tier models: gemma-3-4b, llama-3.2-3b, gemma-3n-e4b
   - Edge deployment restricted to SMALL tier only

3. **Latency Simulation** (`latency.py`): Simple deployment latency multipliers
   - Edge deployment: 0.2× actual API latency (simulates local inference)
   - Cloud deployment: 1.0× actual API latency (baseline)
   - Only SMALL tier models can be deployed on edge
   - **Intentional limitations** for candidates to identify

4. **Quality Evaluation** (`quality.py`): LLM-as-a-judge via Pydantic AI
   - Multi-dimensional quality scoring (accuracy, relevance, completeness, clarity, helpfulness)
   - Structured output with pydantic-ai and output validation
   - Batch evaluation support for efficiency (reduces API calls)
   - Uses trinity-mini (free tier) as evaluator

5. **Router** (`router.py`): Abstract base class and baselines
   - `BaseRouter`: Abstract class for candidates to extend
   - `NaiveRouter`: Random routing (baseline)
   - `StaticRouter`: Always routes to a specific model

6. **Benchmarking** (`benchmarking.py`): Evaluation tools
   - Two-phase approach: parallel inference, then batch quality evaluation
   - Parallel query processing with rate limiting
   - Quality evaluation and cost tracking
   - Router comparison tables
   - Sample queries (`data/sample_queries.json`) and full queries (`data/all_queries.json`)

### Key Design Decisions

The boilerplate is intentionally limited to:
- Force candidates to read and understand the code
- Require candidates to identify limitations
- Enable measurement of improvement
- Provide `BaseRouter` class to extend
- Prevent "just plug in an LLM" solutions

### What We're Really Evaluating

| Skill | How It's Assessed |
|-------|-------------------|
| Research ability | Literature review quality, paper synthesis |
| Implementation | Code quality, tests, documentation |
| Empirical rigor | Experimental design, statistical analysis |
| Systems thinking | Architecture decisions, production considerations |
| Communication | Documentation clarity, result presentation |
| Judgment | Tradeoff analysis, knowing when NOT to add complexity |

## Need Help?

If you get stuck or want to review syntax, examples, or concepts, these resources may be helpful:

- **LLMs from Scratch**: https://stuli.ai/llms-from-scratch/README.html
- **Build Your Own Super Agents**: https://stuli.ai/build-your-own-super-agents/README.html
