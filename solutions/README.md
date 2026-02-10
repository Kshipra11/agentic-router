# Solutions Folder

This folder is where candidates should implement their routing solution.

## Structure

```
solutions/
├── README.md           # This file
├── custom_router.py    # Your main router implementation (REQUIRED)
└── ...                 # Any additional files you need
```

## Requirements

1. **Main Implementation**: Your final router must be in `custom_router.py`
2. **Class Name**: The main router class must be named `CustomRouter`
3. **Inheritance**: Must extend `BaseRouter` from `src.router`

## Adding Additional Files

You may add any additional files as needed:

- **Helper modules**: `classifier.py`, `features.py`, `utils.py`, etc.
- **Data files**: Training data, lookup tables, configuration files
- **Tests**: Unit tests for your implementation
- **Documentation**: Notes, diagrams, or analysis

## Usage

Your router will be imported and benchmarked like this:

```python
from solutions.custom_router import CustomRouter

router = CustomRouter()
results = await benchmark_router(router)
```

## Getting Started

1. Open `custom_router.py`
2. Implement the `route()` method
3. Run `uv run python smoke_test.py` to verify setup
4. Run `uv run python eval.py` to benchmark your router
