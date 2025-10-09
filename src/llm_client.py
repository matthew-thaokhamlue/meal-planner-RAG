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
            model: Optional model override (if not provided, uses default from config)
        """
        self.config = load_config(config_path)
        self.llm_config = self.config['llm']

        # Get API key from environment
        self.api_key = os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")

        self.api_base_url = self.llm_config['api_base_url']
        self.default_model = model if model else self.llm_config['default_model']
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
        Parse JSON from LLM response.
        
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
                json_str = content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                json_str = content[start:end].strip()
            else:
                json_str = content.strip()
            
            return json.loads(json_str)
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.error(f"Content: {content[:500]}...")
            raise Exception(f"Failed to parse JSON response: {e}")
    
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

