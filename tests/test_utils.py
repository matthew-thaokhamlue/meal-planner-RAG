import pytest
import os
import json
from datetime import datetime, timedelta
from src.utils import (
    load_config,
    load_prompts,
    save_json,
    load_json,
    parse_time_query,
    format_shopping_list,
    get_days_since,
    ensure_dir
)

def test_load_config():
    """Test loading configuration"""
    config = load_config('config/config.yaml')
    assert 'rag' in config
    assert 'llm' in config
    assert config['rag']['top_k'] == 10

def test_load_prompts():
    """Test loading prompts"""
    prompts = load_prompts('config/prompts.yaml')
    assert 'system_prompt' in prompts
    assert 'meal_planning_prompt' in prompts

def test_save_and_load_json(tmp_path):
    """Test JSON save and load"""
    test_data = {'test': 'data', 'number': 42}
    filepath = tmp_path / "test.json"
    
    save_json(test_data, str(filepath))
    assert os.path.exists(filepath)
    
    loaded = load_json(str(filepath))
    assert loaded == test_data

def test_load_json_nonexistent():
    """Test loading non-existent JSON file"""
    result = load_json('nonexistent.json')
    assert result is None

def test_parse_time_query_days():
    """Test parsing day-based queries"""
    result = parse_time_query("what should I shop for 3 days")
    assert result['num_days'] == 3
    assert result['occasion'] == 'general'

def test_parse_time_query_weekend():
    """Test parsing weekend query"""
    result = parse_time_query("shopping for the weekend")
    assert result['num_days'] == 2
    assert result['occasion'] == 'weekend'

def test_parse_time_query_week():
    """Test parsing week query"""
    result = parse_time_query("meal plan for a week")
    assert result['num_days'] == 7
    assert result['occasion'] == 'week'

def test_parse_time_query_party():
    """Test parsing party query"""
    result = parse_time_query("party for 15 people")
    assert result['occasion'] == 'party'
    assert result['num_people'] == 15

def test_format_shopping_list():
    """Test shopping list formatting"""
    shopping_list = {
        'produce': [
            {'item': 'tomatoes', 'quantity': '6', 'unit': 'medium'},
            {'item': 'onions', 'quantity': '2', 'unit': 'large'}
        ],
        'household': [
            {'item': 'dish soap', 'reason': 'Running low'}
        ]
    }
    
    formatted = format_shopping_list(shopping_list)
    assert '🥬 Produce' in formatted
    assert '6 medium tomatoes' in formatted
    assert '🧹 Household' in formatted
    assert 'dish soap' in formatted

def test_get_days_since():
    """Test days calculation"""
    yesterday = (datetime.now() - timedelta(days=1)).isoformat()
    assert get_days_since(yesterday) == 1
    
    week_ago = (datetime.now() - timedelta(days=7)).isoformat()
    assert get_days_since(week_ago) == 7

def test_ensure_dir(tmp_path):
    """Test directory creation"""
    new_dir = tmp_path / "test" / "nested" / "dir"
    ensure_dir(str(new_dir))
    assert os.path.exists(new_dir)

