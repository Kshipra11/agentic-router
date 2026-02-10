"""
Utility functions for the agentic routing pipeline.

This module provides helper functions for:
- Converting Pydantic models to JSON schemas (for embedding in prompts)
- Extracting JSON from LLM responses (robust parsing)
"""

import json
import re
from typing import Type
from pydantic import BaseModel


def pydantic_to_json_schema_string(pydantic_model: Type[BaseModel]) -> str:
    """
    Convert a Pydantic model to a JSON schema string for embedding in prompts.
    
    This function:
    1. Generates the JSON schema from the Pydantic model
    2. Formats it as a readable JSON string
    3. Escapes braces so it can be safely used in Python's .format() or f-strings
    
    Args:
        pydantic_model: Pydantic model class (e.g., IntentClassification)
        
    Returns:
        JSON schema as a formatted string with escaped braces
        
    Example:
        >>> schema_str = pydantic_to_json_schema_string(IntentClassification)
        >>> prompt = f"Return JSON matching this schema: {schema_str}"
    """
    # Generate JSON schema from Pydantic model
    schema = pydantic_model.model_json_schema()
    
    # Convert to formatted JSON string
    json_str = json.dumps(schema, indent=2)
    
    # Escape braces for safe use in .format() calls
    # When using f-strings, you don't need this, but for .format() you do
    # We escape them so the function works with both
    escaped = json_str.replace("{", "{{").replace("}", "}}")
    
    return escaped


def extract_json_from_text(text: str) -> dict:
    """
    Extract JSON from text, handling various formats that LLMs might return.
    
    This function is robust and handles:
    - Pure JSON: {"key": "value"}
    - Markdown-wrapped: ```json\n{"key": "value"}\n```
    - Text with extra content: "Here's the JSON: {\"key\": \"value\"}"
    - Multiple JSON objects (returns the first valid one)
    
    Args:
        text: Response text that may contain JSON
        
    Returns:
        Extracted JSON as a dictionary
        
    Raises:
        json.JSONDecodeError: If no valid JSON can be extracted
        
    Example:
        >>> text = '```json\\n{"intent": "coding", "confidence": 0.9}\\n```'
        >>> result = extract_json_from_text(text)
        >>> print(result)
        {'intent': 'coding', 'confidence': 0.9}
    """
    if not text or not text.strip():
        raise json.JSONDecodeError("Empty text", text, 0)
    
    text = text.strip()
    
    # Strategy 1: Try direct JSON parsing first (fastest path)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Remove markdown code blocks if present
    # Handles: ```json\n{...}\n``` or ```\n{...}\n```
    if text.startswith('```'):
        # Remove opening ```json or ```
        text = re.sub(r'^```(?:json)?\s*\n?', '', text, flags=re.IGNORECASE)
        # Remove closing ```
        text = re.sub(r'\n?```\s*$', '', text)
        text = text.strip()
        
        # Try parsing again after removing markdown
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    
    # Strategy 3: Find JSON object using regex (handles nested objects)
    # Look for {...} pattern that might be embedded in text
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    
    # Strategy 4: Last resort - find first { to last }
    # This handles cases where there's text before/after the JSON
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    
    # If all strategies fail, raise an error
    raise json.JSONDecodeError(
        "Could not extract valid JSON from text",
        text,
        0
    )
