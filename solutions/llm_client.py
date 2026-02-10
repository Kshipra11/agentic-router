"""
LLM Client for OpenRouter API calls.

This module provides a clean interface for making structured LLM calls
via OpenRouter API. It includes:
- Comprehensive logging before and after API calls
- Robust error handling with retries
- Rate limit handling
- JSON extraction and Pydantic validation
- Terminal-only logging (no file logging)
"""

import os
import json
import time
import logging
import requests
import certifi
import urllib3
from typing import Type, TypeVar, Dict, Any, Tuple, Optional
from pydantic import BaseModel, ValidationError

from src.config import OPENROUTER_BASE_URL
from src.model_registry import MODEL_REGISTRY
from .utils import extract_json_from_text

# Configure TLS certificates for requests
# Set environment variables if not already set
if 'REQUESTS_CA_BUNDLE' not in os.environ:
    os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
if 'SSL_CERT_FILE' not in os.environ:
    os.environ['SSL_CERT_FILE'] = certifi.where()

# Disable SSL warnings (optional)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Set up logging
logger = logging.getLogger(__name__)

# Type variable for Pydantic models
T = TypeVar("T", bound=BaseModel)

# OpenRouter API endpoint
OPENROUTER_URL = f"{OPENROUTER_BASE_URL}/chat/completions"


def _get_step_name(output_model: Type[BaseModel]) -> str:
    """
    Extract a readable step name from the output model class name.
    
    Args:
        output_model: Pydantic model class
        
    Returns:
        Human-readable step name (e.g., "IntentClassification" -> "INTENT")
    """
    name = output_model.__name__
    # Remove common suffixes
    name = name.replace("Output", "").replace("Classification", "").replace("Criticality", "")
    # Convert to uppercase for logging
    return name.upper()


def call_llm_structured(
    *,
    model_id: str,
    prompt: str,
    output_model: Type[T],
    max_tokens: int = 200,
    temperature: float = 0.0,
    retries: int = 2,
    timeout_s: int = 60,
    step_name: Optional[str] = None,
) -> Tuple[T, Dict[str, Any]]:
    """
    Call OpenRouter API with a prompt and parse structured output.
    
    This function:
    1. Logs the request details before making the call
    2. Makes the API call to OpenRouter
    3. Extracts and validates JSON from the response
    4. Returns validated Pydantic model and debug info
    5. Handles retries, rate limits, and errors gracefully
    
    Args:
        model_id: OpenRouter model identifier (e.g., "google/gemma-3-4b-it:free")
        prompt: The prompt to send to the model
        output_model: Pydantic model class to validate output against
        max_tokens: Maximum tokens in response (default: 200)
        temperature: Sampling temperature (default: 0.0 for deterministic)
        retries: Number of retry attempts on failure (default: 2)
        timeout_s: Request timeout in seconds (default: 60)
        step_name: Optional step name for logging (auto-detected if not provided)
        
    Returns:
        Tuple of (validated_output, debug_info) where:
        - validated_output: Parsed and validated Pydantic model instance
        - debug_info: Dictionary with request/response metadata
        
    Raises:
        RuntimeError: If API call fails after all retries
        ValidationError: If response cannot be parsed or validated
    """
    
    # Get step name for logging
    if step_name is None:
        step_name = _get_step_name(output_model)
    
    # Track last error for final error message
    last_error: Optional[Exception] = None
    
    # ========================================================================
    # PRE-REQUEST LOGGING
    # ========================================================================
    logger.info("=" * 80)
    logger.info(f"[{step_name}] Starting LLM call")
    logger.info(f"  Model: {model_id}")
    logger.info(f"  Max tokens: {max_tokens}")
    logger.info(f"  Temperature: {temperature}")
    logger.info(f"  Output model: {output_model.__name__}")
    logger.info(f"  Prompt length: {len(prompt)} characters")
    logger.info("-" * 80)
    
    # Retry loop
    for attempt in range(retries + 1):
        if attempt > 0:
            logger.warning(f"[{step_name}] Retry attempt {attempt + 1}/{retries + 1}")
        
        try:
            # ====================================================================
            # BUILD REQUEST PAYLOAD
            # ====================================================================
            payload = {
                "model": model_id,
                "messages": [
                    {"role": "user", "content": prompt}  # User-only message (works with all models)
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "provider": {"allow_fallbacks": True},  # Allow fallback providers
            }
            
            logger.debug(f"[{step_name}] Request payload: {json.dumps(payload, indent=2)}")
            
            # ====================================================================
            # MAKE API CALL
            # ====================================================================
            logger.info(f"[{step_name}] Sending request to OpenRouter API...")
            request_start_time = time.time()
            
            # Dynamically retrieve the API key from environment or src.config
            # Prioritize environment variable, then fallback to src.config
            openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
            if not openrouter_api_key:
                # If not in environment, try getting it from src.config
                # We re-import here to ensure we get the latest value from the reloaded src.config
                from src.config import OPENROUTER_API_KEY as CONFIG_API_KEY
                openrouter_api_key = CONFIG_API_KEY

            # Validate API key
            if not openrouter_api_key:
                error_msg = "OPENROUTER_API_KEY not found. Ensure it's set in your environment variables (e.g., .env file) or directly in the notebook."
                logger.error(f"[{step_name}] API Error: {error_msg}")
                raise RuntimeError(error_msg)

            # Check if API key looks valid (optional, but good for debugging)
            if not openrouter_api_key.startswith("sk-"):
                logger.warning(f"[{step_name}] API key does not start with 'sk-'. It may be invalid: {openrouter_api_key[:20]}...")

            # Construct headers dynamically
            headers = {
                "Authorization": f"Bearer {openrouter_api_key}",
                "Content-Type": "application/json",
            }

            logger.debug(f"[{step_name}] Using API key: {openrouter_api_key[:10]}...{openrouter_api_key[-4:] if len(openrouter_api_key) > 14 else 'SHORT'}")


            response = requests.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=timeout_s,
                verify=True  # Use certificate bundle from environment
            )
            
            request_duration = time.time() - request_start_time
            logger.info(f"[{step_name}] API call completed in {request_duration:.2f}s")
            logger.info(f"[{step_name}] HTTP status: {response.status_code}")
            
            # ====================================================================
            # PARSE RESPONSE
            # ====================================================================
            try:
                response_data = response.json()
            except json.JSONDecodeError as e:
                logger.error(f"[{step_name}] Failed to parse response as JSON: {e}")
                logger.error(f"[{step_name}] Response text: {response.text[:500]}")
                raise RuntimeError(f"Invalid JSON response: {e}") from e
            
            # Extract usage and calculate cost
            usage = response_data.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            
            cost = 0.0
            try:
                # Try to find model in registry by matching model_id
                model_key_candidates = [
                    model_id.split("/")[-1].split(":")[0].replace("-it", ""),
                    model_id.split("/")[-1].split(":")[0],
                    model_id.split("/")[-1],
                ]
                
                for candidate in model_key_candidates:
                    if candidate in MODEL_REGISTRY:
                        model_config = MODEL_REGISTRY[candidate]
                        cost = (
                            (input_tokens / 1_000_000) * model_config.cost_per_million_input +
                            (output_tokens / 1_000_000) * model_config.cost_per_million_output
                        )
                        break
            except (KeyError, AttributeError, IndexError):
                # If model not found in registry, cost remains 0.0
                pass
            
            # Build debug info
            debug_info: Dict[str, Any] = {
                "step_name": step_name,
                "model_id": model_id,
                "request": {
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "prompt_length": len(prompt),
                },
                "response": {
                    "http_status": response.status_code,
                    "duration_seconds": request_duration,
                },
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                },
                "cost": cost,
            }
            
            # ====================================================================
            # HANDLE API ERRORS
            # ====================================================================
            if "error" in response_data:
                error_info = response_data["error"]
                error_code = error_info.get("code")
                error_message = error_info.get("message", str(error_info))
                
                debug_info["error"] = error_info
                logger.error(f"[{step_name}] API Error: {error_message}")
                logger.error(f"[{step_name}] Error code: {error_code}")
                
                # Handle rate limiting with intelligent retry
                if error_code == 429 and attempt < retries:
                    # Try to get rate limit reset time from response headers
                    reset_timestamp = None
                    metadata = error_info.get("metadata", {})
                    headers = metadata.get("headers", {})
                    
                    if "X-RateLimit-Reset" in headers:
                        reset_timestamp = int(headers["X-RateLimit-Reset"]) / 1000  # Convert ms to seconds
                        current_time = time.time()
                        wait_time = max(reset_timestamp - current_time + 1, 1.2 * (attempt + 1))
                        logger.warning(
                            f"[{step_name}] Rate limited. Waiting until reset "
                            f"({wait_time:.1f}s)..."
                        )
                    else:
                        # Exponential backoff if no reset time available
                        wait_time = 1.2 * (2 ** attempt)
                        logger.warning(
                            f"[{step_name}] Rate limited. Waiting {wait_time:.1f}s "
                            f"(exponential backoff)..."
                        )
                    
                    time.sleep(wait_time)
                    continue
                
                # For other errors, raise immediately
                raise RuntimeError(f"API error: {error_message}")
            
            # ====================================================================
            # EXTRACT AND VALIDATE RESPONSE
            # ====================================================================
            if "choices" not in response_data or not response_data["choices"]:
                error_msg = "No choices in API response"
                logger.error(f"[{step_name}] {error_msg}")
                logger.error(f"[{step_name}] Response data: {json.dumps(response_data, indent=2)}")
                raise RuntimeError(error_msg)
            
            # Get raw content from response
            raw_content = response_data["choices"][0]["message"]["content"]
            debug_info["raw_content"] = raw_content
            
            logger.info(f"[{step_name}] Received response ({len(raw_content)} characters)")
            logger.debug(f"[{step_name}] Raw response preview: {raw_content[:200]}...")
            
            # Convert to string if needed
            if not isinstance(raw_content, str):
                raw_content = json.dumps(raw_content)
            
            # ====================================================================
            # EXTRACT JSON FROM RESPONSE
            # ====================================================================
            logger.info(f"[{step_name}] Extracting JSON from response...")
            try:
                json_dict = extract_json_from_text(raw_content)
                debug_info["parsed_json"] = json_dict
                logger.info(f"[{step_name}] ✓ Successfully extracted JSON")
                logger.debug(f"[{step_name}] Parsed JSON: {json.dumps(json_dict, indent=2)}")
            except json.JSONDecodeError as e:
                last_error = e
                debug_info["json_extraction_error"] = str(e)
                logger.error(f"[{step_name}] ✗ Failed to extract JSON: {e}")
                logger.error(f"[{step_name}] Raw content: {raw_content[:500]}...")
                
                if attempt < retries:
                    wait_time = 0.5 * (attempt + 1)
                    logger.info(f"[{step_name}] Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                raise RuntimeError(f"Failed to extract JSON from response: {e}") from e
            
            # ====================================================================
            # VALIDATE WITH PYDANTIC
            # ====================================================================
            logger.info(f"[{step_name}] Validating with Pydantic model: {output_model.__name__}...")
            try:
                validated_output = output_model.model_validate(json_dict)
                logger.info(f"[{step_name}] ✓ Validation successful")
                
                # Log the validated output
                output_json = validated_output.model_dump_json(indent=2)
                logger.info(f"[{step_name}] Validated output:")
                logger.info(f"[{step_name}] {output_json}")
                
            except ValidationError as e:
                last_error = e
                debug_info["validation_error"] = str(e)
                logger.error(f"[{step_name}] ✗ Validation failed: {e}")
                logger.error(f"[{step_name}] Extracted JSON: {json.dumps(json_dict, indent=2)}")
                
                if attempt < retries:
                    wait_time = 0.5 * (attempt + 1)
                    logger.info(f"[{step_name}] Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                raise RuntimeError(f"Validation failed: {e}") from e
            
            # ====================================================================
            # POST-REQUEST LOGGING (SUCCESS)
            # ====================================================================
            logger.info("-" * 80)
            logger.info(f"[{step_name}] ✓ Successfully completed")
            logger.info(f"  Duration: {request_duration:.2f}s")
            logger.info(f"  Model: {model_id}")
            logger.info(f"  Output: {output_model.__name__}")
            logger.info("=" * 80)
            
            return validated_output, debug_info
            
        except requests.exceptions.Timeout as e:
            last_error = e
            logger.error(f"[{step_name}] Request timeout after {timeout_s}s")
            if attempt < retries:
                wait_time = 1.0 * (attempt + 1)
                logger.info(f"[{step_name}] Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
                continue
            raise RuntimeError(f"Request timeout: {e}") from e
            
        except requests.exceptions.RequestException as e:
            last_error = e
            logger.error(f"[{step_name}] Request exception: {e}")
            if attempt < retries:
                wait_time = 1.0 * (attempt + 1)
                logger.info(f"[{step_name}] Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
                continue
            raise RuntimeError(f"Request failed: {e}") from e
            
        except Exception as e:
            last_error = e
            logger.error(f"[{step_name}] Unexpected error: {e}")
            if attempt < retries:
                wait_time = 0.5 * (attempt + 1)
                logger.info(f"[{step_name}] Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
                continue
            raise
    
    # ========================================================================
    # FINAL ERROR LOGGING
    # ========================================================================
    logger.error("=" * 80)
    logger.error(f"[{step_name}] ✗ Failed after {retries + 1} attempts")
    if last_error:
        logger.error(f"[{step_name}] Last error: {last_error}")
    logger.error("=" * 80)
    
    raise RuntimeError(f"Failed after {retries + 1} attempts. Last error: {last_error}")
