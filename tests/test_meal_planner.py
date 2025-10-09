import pytest
import os
import json
from unittest.mock import Mock, MagicMock
from src.meal_planner import MealPlanner

@pytest.fixture
def mock_rag_engine():
    """Create a mock RAG engine"""
    engine = Mock()
    engine.get_stats.return_value = {
        'total_documents': 10,
        'by_type': {'meal': 5, 'preference': 3, 'bill': 2}
    }
    engine.get_context.return_value = "Mock context about meals and preferences"
    engine.query.return_value = [
        {
            'document': 'Household item: dish soap',
            'metadata': {
                'type': 'household_item',
                'item_name': 'dish soap',
                'date': '2024-01-15',
                'store': 'Store A'
            }
        }
    ]
    return engine

@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client"""
    client = Mock()
    client.generate_response.return_value = {
        'content': '''```json
[
  {
    "option_name": "Test Meal Plan",
    "description": "A test meal plan",
    "meals": [
      {
        "day": "Monday",
        "meal_name": "Test Meal",
        "description": "A test meal",
        "main_ingredients": ["ingredient1", "ingredient2"]
      }
    ],
    "shopping_list": {
      "produce": [
        {"item": "tomatoes", "quantity": "3", "unit": "whole"}
      ],
      "household": [
        {"item": "dish soap", "reason": "Running low"}
      ]
    },
    "estimated_cost": "50-60",
    "prep_difficulty": "easy"
  }
]
```''',
        'usage': {'total_tokens': 500},
        'model': 'test-model'
    }
    client.parse_json_response.return_value = [
        {
            "option_name": "Test Meal Plan",
            "description": "A test meal plan",
            "meals": [
                {
                    "day": "Monday",
                    "meal_name": "Test Meal",
                    "description": "A test meal",
                    "main_ingredients": ["ingredient1", "ingredient2"]
                }
            ],
            "shopping_list": {
                "produce": [
                    {"item": "tomatoes", "quantity": "3", "unit": "whole"}
                ],
                "household": [
                    {"item": "dish soap", "reason": "Running low"}
                ]
            },
            "estimated_cost": "50-60",
            "prep_difficulty": "easy"
        }
    ]
    return client

@pytest.fixture
def meal_planner(mock_rag_engine, mock_llm_client):
    """Create a meal planner with mocked dependencies"""
    return MealPlanner(
        rag_engine=mock_rag_engine,
        llm_client=mock_llm_client
    )

def test_meal_planner_initialization(meal_planner):
    """Test meal planner initialization"""
    assert meal_planner.rag_engine is not None
    assert meal_planner.llm_client is not None
    assert meal_planner.prompts is not None

def test_generate_meal_plans_success(meal_planner, mock_rag_engine, mock_llm_client):
    """Test successful meal plan generation"""
    plans = meal_planner.generate_meal_plans("3 days")
    
    assert len(plans) > 0
    assert plans[0]['option_name'] == 'Test Meal Plan'
    assert 'meals' in plans[0]
    assert 'shopping_list' in plans[0]
    assert 'generated_at' in plans[0]
    assert 'query' in plans[0]
    
    # Verify RAG and LLM were called
    mock_rag_engine.get_stats.assert_called_once()
    mock_rag_engine.get_context.assert_called_once()
    mock_llm_client.generate_response.assert_called_once()

def test_generate_meal_plans_empty_database(mock_llm_client):
    """Test meal plan generation with empty database"""
    empty_rag = Mock()
    empty_rag.get_stats.return_value = {'total_documents': 0}
    
    planner = MealPlanner(
        rag_engine=empty_rag,
        llm_client=mock_llm_client
    )
    
    plans = planner.generate_meal_plans("3 days")
    
    assert len(plans) > 0
    assert plans[0].get('error') is True
    assert 'No Data Available' in plans[0]['option_name']

def test_generate_meal_plans_llm_error(mock_rag_engine):
    """Test meal plan generation when LLM fails"""
    error_llm = Mock()
    error_llm.generate_response.side_effect = Exception("API Error")
    
    planner = MealPlanner(
        rag_engine=mock_rag_engine,
        llm_client=error_llm
    )
    
    plans = planner.generate_meal_plans("3 days")
    
    assert len(plans) > 0
    assert plans[0].get('error') is True

def test_log_user_choice(meal_planner, tmp_path, monkeypatch):
    """Test logging user choice"""
    # Mock the choices file path
    choices_file = tmp_path / "user_choices.json"
    monkeypatch.setattr('src.meal_planner.load_json', lambda x: [])
    
    saved_data = []
    def mock_save_json(data, filepath):
        saved_data.append(data)
    
    monkeypatch.setattr('src.meal_planner.save_json', mock_save_json)
    
    selected_option = {
        'option_name': 'Test Plan',
        'meals': []
    }
    
    all_options = [
        selected_option,
        {'option_name': 'Other Plan', 'meals': []}
    ]
    
    meal_planner.log_user_choice(selected_option, "test query", all_options)
    
    assert len(saved_data) > 0
    assert saved_data[0][0]['query'] == 'test query'
    assert saved_data[0][0]['selected_index'] == 0

def test_get_choice_history(meal_planner, monkeypatch):
    """Test getting choice history"""
    mock_choices = [
        {
            'timestamp': '2024-01-20T10:00:00',
            'query': 'test query 1',
            'selected_option': {'option_name': 'Plan 1'}
        },
        {
            'timestamp': '2024-01-19T10:00:00',
            'query': 'test query 2',
            'selected_option': {'option_name': 'Plan 2'}
        }
    ]
    
    monkeypatch.setattr('src.meal_planner.load_json', lambda x: mock_choices)
    
    history = meal_planner.get_choice_history(limit=1)
    
    assert len(history) == 1
    assert history[0]['query'] == 'test query 1'  # Most recent

def test_analyze_household_items(meal_planner, mock_rag_engine):
    """Test household items analysis"""
    items = meal_planner.analyze_household_items()
    
    assert len(items) > 0
    assert items[0]['item'] == 'dish soap'
    assert 'last_purchase' in items[0]
    
    mock_rag_engine.query.assert_called_once()

def test_export_shopping_list_txt(meal_planner, tmp_path):
    """Test exporting shopping list as text"""
    meal_plan = {
        'option_name': 'Test Plan',
        'shopping_list': {
            'produce': [
                {'item': 'tomatoes', 'quantity': '3', 'unit': 'whole'}
            ],
            'household': [
                {'item': 'dish soap'}
            ]
        }
    }
    
    filepath = tmp_path / "shopping_list.txt"
    meal_planner.export_shopping_list(meal_plan, str(filepath), format='txt')
    
    assert os.path.exists(filepath)
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    assert 'Test Plan' in content
    assert 'tomatoes' in content
    assert 'dish soap' in content

def test_export_shopping_list_json(meal_planner, tmp_path):
    """Test exporting shopping list as JSON"""
    meal_plan = {
        'option_name': 'Test Plan',
        'shopping_list': {
            'produce': [
                {'item': 'tomatoes', 'quantity': '3', 'unit': 'whole'}
            ]
        }
    }
    
    filepath = tmp_path / "shopping_list.json"
    meal_planner.export_shopping_list(meal_plan, str(filepath), format='json')
    
    assert os.path.exists(filepath)
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    assert 'produce' in data
    assert data['produce'][0]['item'] == 'tomatoes'

