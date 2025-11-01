"""OpenRouter API client for LLM interactions."""

import os
import logging
import time
import json
from typing import Dict, Any, Optional
import requests
from dotenv import load_dotenv

from src.utils import load_config

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class OpenRouterClient:
    """Client for interacting with OpenRouter API."""
    
    def __init__(self, config_path: str = "config/config.yaml", model: Optional[str] = None):
        """
        Initialize OpenRouter client.

        Args:
            config_path: Path to configuration file
            model: Optional model override (if not provided, uses OPENROUTER_MODEL env var, then config default)
        """
        self.config = load_config(config_path)
        self.llm_config = self.config['llm']

        # Get API key from environment
        self.api_key = os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")

        self.api_base_url = self.llm_config['api_base_url']

        # Model priority: explicit parameter > env var > config default
        self.default_model = (
            model if model
            else os.getenv('OPENROUTER_MODEL', self.llm_config['default_model'])
        )

        self.temperature = self.llm_config['temperature']
        self.max_tokens = self.llm_config['max_tokens']
        self.timeout = self.llm_config['timeout']

        logger.info(f"Initialized OpenRouter client with model: {self.default_model}")
    
    def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            model: Optional model override
            
        Returns:
            Dictionary with 'content', 'usage', and 'model' keys
        """
        # Use defaults if not specified
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        model = model if model is not None else self.default_model
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Build request payload
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # Make API call with retry logic
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Calling OpenRouter API (attempt {attempt + 1}/{max_retries})")
                
                response = requests.post(
                    self.api_base_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://github.com/meal-shopping-assistant",
                        "X-Title": "Meal Shopping Assistant"
                    },
                    json=payload,
                    timeout=self.timeout
                )
                
                response.raise_for_status()
                
                result = response.json()
                
                # Extract response content
                content = result['choices'][0]['message']['content']
                usage = result.get('usage', {})
                
                logger.info(f"API call successful. Tokens used: {usage.get('total_tokens', 'unknown')}")
                
                return {
                    'content': content,
                    'usage': usage,
                    'model': result.get('model', model)
                }
                
            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise Exception("API request timed out after multiple retries")
            
            except requests.exceptions.HTTPError as e:
                logger.error(f"HTTP error: {e}")
                if e.response.status_code == 429:  # Rate limit
                    if attempt < max_retries - 1:
                        logger.warning("Rate limited, waiting before retry...")
                        time.sleep(retry_delay * 2)
                        retry_delay *= 2
                    else:
                        raise Exception("Rate limit exceeded")
                else:
                    raise Exception(f"API request failed: {e}")
            
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise Exception(f"API request failed: {e}")
        
        raise Exception("Failed to get response from API")
    
    def parse_json_response(self, content: str) -> Any:
        """
        Parse JSON from LLM response with robust error handling and repair.

        Args:
            content: Response content from LLM

        Returns:
            Parsed JSON object
        """
        try:
            # Try to find JSON in code blocks
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                if end == -1:  # No closing backticks (truncated)
                    json_str = content[start:].strip()
                else:
                    json_str = content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                if end == -1:  # No closing backticks (truncated)
                    json_str = content[start:].strip()
                else:
                    json_str = content[start:end].strip()
            else:
                json_str = content.strip()

            # Apply multiple repair strategies
            json_str = self._repair_json(json_str)

            # Try to parse
            return json.loads(json_str)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.error(f"Content preview: {content[:1000]}...")

            # Try advanced recovery strategies
            try:
                repaired = self._advanced_json_repair(content)
                if repaired:
                    logger.info("Successfully repaired JSON using advanced recovery")
                    return repaired
            except Exception as repair_error:
                logger.error(f"Advanced repair also failed: {repair_error}")

            raise Exception(f"Failed to parse JSON response: {e}")

    def _repair_json(self, json_str: str) -> str:
        """
        Apply common JSON repair strategies.

        Args:
            json_str: JSON string to repair

        Returns:
            Repaired JSON string
        """
        # Remove any trailing commas before closing brackets/braces
        json_str = json_str.replace(",]", "]").replace(",}", "}")

        # Remove any leading/trailing whitespace
        json_str = json_str.strip()

        # Fix common quote issues
        # Replace smart quotes with regular quotes
        json_str = json_str.replace('"', '"').replace('"', '"')
        json_str = json_str.replace("'", "'").replace("'", "'")

        return json_str

    def _advanced_json_repair(self, content: str) -> Any:
        """
        Advanced JSON repair for truncated or malformed responses.

        Args:
            content: Full content from LLM

        Returns:
            Parsed JSON object or None
        """
        # Extract JSON from code blocks if present
        if "```json" in content:
            start = content.find("```json") + 7
            json_str = content[start:].strip()
        elif "```" in content:
            start = content.find("```") + 3
            json_str = content[start:].strip()
        else:
            json_str = content.strip()

        # Remove trailing backticks if present
        if json_str.endswith("```"):
            json_str = json_str[:-3].strip()

        # Try to find the first valid JSON array
        start_idx = json_str.find('[')
        if start_idx == -1:
            return None

        json_str = json_str[start_idx:]

        # Count brackets to find where truncation might have occurred
        bracket_count = 0
        brace_count = 0
        last_valid_pos = 0
        in_string = False
        escape_next = False

        for i, char in enumerate(json_str):
            if escape_next:
                escape_next = False
                continue

            if char == '\\':
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if in_string:
                continue

            if char == '[':
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
                if bracket_count == 0:
                    # Found complete array
                    last_valid_pos = i + 1
                    break
            elif char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and bracket_count == 1:
                    # Completed an object inside the array
                    last_valid_pos = i + 1

        # If we didn't find a complete array, try to close it
        if bracket_count > 0 or brace_count > 0:
            logger.warning(f"JSON appears truncated. Bracket count: {bracket_count}, Brace count: {brace_count}")

            # Find the last complete object
            if last_valid_pos > 0:
                json_str = json_str[:last_valid_pos]

            # Try to close any open braces and brackets
            json_str = json_str.rstrip()

            # Remove trailing comma if present
            if json_str.endswith(','):
                json_str = json_str[:-1]

            # Close open structures
            json_str += '}' * brace_count
            json_str += ']' * bracket_count

        # Apply basic repairs
        json_str = self._repair_json(json_str)

        # Try to parse
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # If still failing, try to extract just complete objects
            return self._extract_complete_objects(json_str)
    
    def _extract_complete_objects(self, json_str: str) -> Any:
        """
        Extract complete JSON objects from a potentially malformed array.

        Args:
            json_str: Malformed JSON string

        Returns:
            List of complete objects or None
        """
        try:
            # Try to find complete objects within the string
            objects = []
            depth = 0
            current_obj = []
            in_string = False
            escape_next = False

            for i, char in enumerate(json_str):
                if escape_next:
                    if depth > 0:
                        current_obj.append(char)
                    escape_next = False
                    continue

                if char == '\\':
                    if depth > 0:
                        current_obj.append(char)
                    escape_next = True
                    continue

                if char == '"' and not escape_next:
                    in_string = not in_string
                    if depth > 0:
                        current_obj.append(char)
                    continue

                if in_string:
                    if depth > 0:
                        current_obj.append(char)
                    continue

                if char == '{':
                    depth += 1
                    current_obj.append(char)
                elif char == '}':
                    depth -= 1
                    current_obj.append(char)
                    if depth == 0 and current_obj:
                        # Complete object found
                        obj_str = ''.join(current_obj)
                        try:
                            obj = json.loads(obj_str)
                            objects.append(obj)
                        except:
                            pass
                        current_obj = []
                elif depth > 0:
                    current_obj.append(char)

            if objects:
                logger.info(f"Extracted {len(objects)} complete objects from malformed JSON")
                return objects

            return None

        except Exception as e:
            logger.error(f"Failed to extract complete objects: {e}")
            return None

    def test_connection(self) -> bool:
        """
        Test the API connection.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = self.generate_response(
                "Say 'Hello' if you can hear me.",
                temperature=0.1,
                max_tokens=50
            )
            return 'content' in response and len(response['content']) > 0
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

