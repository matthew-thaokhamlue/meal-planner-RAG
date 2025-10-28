import pytest
import os
import json
from unittest.mock import Mock, patch
from src.llm_client import OpenRouterClient

@pytest.fixture
def mock_env(monkeypatch):
    """Mock environment variables"""
    monkeypatch.setenv('OPENROUTER_API_KEY', 'test_api_key')

@pytest.fixture
def llm_client(mock_env):
    """Create an LLM client with mocked environment"""
    return OpenRouterClient()

def test_client_initialization(llm_client):
    """Test client initialization"""
    assert llm_client.api_key == 'test_api_key'
    assert llm_client.default_model is not None
    assert llm_client.temperature is not None

def test_missing_api_key(monkeypatch):
    """Test that missing API key raises error"""
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)

    with pytest.raises(ValueError, match="OPENROUTER_API_KEY not found"):
        OpenRouterClient()

def test_model_from_env_variable(monkeypatch):
    """Test that OPENROUTER_MODEL env variable is used"""
    monkeypatch.setenv('OPENROUTER_API_KEY', 'test_api_key')
    monkeypatch.setenv('OPENROUTER_MODEL', 'anthropic/claude-haiku-4.5')

    client = OpenRouterClient()

    assert client.default_model == 'anthropic/claude-haiku-4.5'

def test_model_parameter_overrides_env(monkeypatch):
    """Test that explicit model parameter overrides env variable"""
    monkeypatch.setenv('OPENROUTER_API_KEY', 'test_api_key')
    monkeypatch.setenv('OPENROUTER_MODEL', 'anthropic/claude-haiku-4.5')

    client = OpenRouterClient(model='openai/gpt-4')

    assert client.default_model == 'openai/gpt-4'

def test_model_defaults_to_config(monkeypatch):
    """Test that model defaults to config when no env var or parameter"""
    monkeypatch.setenv('OPENROUTER_API_KEY', 'test_api_key')
    monkeypatch.delenv('OPENROUTER_MODEL', raising=False)

    client = OpenRouterClient()

    # Should use the default from config.yaml
    assert client.default_model is not None

@patch('src.llm_client.requests.post')
def test_generate_response_success(mock_post, llm_client):
    """Test successful API response"""
    # Mock successful response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        'choices': [
            {
                'message': {
                    'content': 'Test response content'
                }
            }
        ],
        'usage': {
            'total_tokens': 100,
            'prompt_tokens': 50,
            'completion_tokens': 50
        },
        'model': 'test-model'
    }
    mock_post.return_value = mock_response
    
    result = llm_client.generate_response("Test prompt")
    
    assert result['content'] == 'Test response content'
    assert result['usage']['total_tokens'] == 100
    assert result['model'] == 'test-model'
    
    # Verify API was called correctly
    mock_post.assert_called_once()
    call_args = mock_post.call_args
    assert call_args[1]['json']['messages'][0]['content'] == 'Test prompt'

@patch('src.llm_client.requests.post')
def test_generate_response_with_system_prompt(mock_post, llm_client):
    """Test API call with system prompt"""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        'choices': [{'message': {'content': 'Response'}}],
        'usage': {},
        'model': 'test-model'
    }
    mock_post.return_value = mock_response
    
    result = llm_client.generate_response(
        "User prompt",
        system_prompt="System prompt"
    )
    
    # Verify both system and user messages were sent
    call_args = mock_post.call_args
    messages = call_args[1]['json']['messages']
    assert len(messages) == 2
    assert messages[0]['role'] == 'system'
    assert messages[0]['content'] == 'System prompt'
    assert messages[1]['role'] == 'user'
    assert messages[1]['content'] == 'User prompt'

@patch('src.llm_client.requests.post')
def test_generate_response_timeout_retry(mock_post, llm_client):
    """Test retry logic on timeout"""
    import requests
    
    # First two calls timeout, third succeeds
    mock_post.side_effect = [
        requests.exceptions.Timeout(),
        requests.exceptions.Timeout(),
        Mock(
            status_code=200,
            json=lambda: {
                'choices': [{'message': {'content': 'Success'}}],
                'usage': {},
                'model': 'test-model'
            }
        )
    ]
    
    result = llm_client.generate_response("Test prompt")
    
    assert result['content'] == 'Success'
    assert mock_post.call_count == 3

@patch('src.llm_client.requests.post')
def test_generate_response_max_retries_exceeded(mock_post, llm_client):
    """Test that max retries raises exception"""
    import requests
    
    # All calls timeout
    mock_post.side_effect = requests.exceptions.Timeout()
    
    with pytest.raises(Exception, match="timed out"):
        llm_client.generate_response("Test prompt")
    
    assert mock_post.call_count == 3

def test_parse_json_response_with_code_block(llm_client):
    """Test parsing JSON from code block"""
    content = '''Here's the response:
```json
{
    "key": "value",
    "number": 42
}
```
'''
    
    result = llm_client.parse_json_response(content)
    
    assert result['key'] == 'value'
    assert result['number'] == 42

def test_parse_json_response_plain(llm_client):
    """Test parsing plain JSON"""
    content = '{"key": "value", "number": 42}'
    
    result = llm_client.parse_json_response(content)
    
    assert result['key'] == 'value'
    assert result['number'] == 42

def test_parse_json_response_invalid(llm_client):
    """Test parsing invalid JSON raises exception"""
    content = 'This is not JSON'
    
    with pytest.raises(Exception, match="Failed to parse JSON"):
        llm_client.parse_json_response(content)

@patch('src.llm_client.requests.post')
def test_test_connection_success(mock_post, llm_client):
    """Test connection test success"""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        'choices': [{'message': {'content': 'Hello'}}],
        'usage': {},
        'model': 'test-model'
    }
    mock_post.return_value = mock_response
    
    result = llm_client.test_connection()
    
    assert result is True

@patch('src.llm_client.requests.post')
def test_test_connection_failure(mock_post, llm_client):
    """Test connection test failure"""
    import requests
    mock_post.side_effect = requests.exceptions.ConnectionError()
    
    result = llm_client.test_connection()
    
    assert result is False

