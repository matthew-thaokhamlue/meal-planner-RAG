import os
import pytest
import yaml
from pathlib import Path

def test_project_structure():
    """Test that all required directories exist"""
    required_dirs = [
        'src',
        'data/raw',
        'data/processed',
        'data/logs',
        'config',
        'tests'
    ]
    
    for dir_path in required_dirs:
        assert os.path.exists(dir_path), f"Directory {dir_path} does not exist"

def test_config_files_exist():
    """Test that configuration files exist"""
    assert os.path.exists('config/config.yaml')
    assert os.path.exists('config/prompts.yaml')
    assert os.path.exists('.env.example')

def test_config_yaml_valid():
    """Test that config.yaml is valid YAML"""
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    assert 'rag' in config
    assert 'llm' in config
    assert 'logging' in config
    assert config['rag']['top_k'] == 10

def test_prompts_yaml_valid():
    """Test that prompts.yaml is valid YAML"""
    with open('config/prompts.yaml', 'r') as f:
        prompts = yaml.safe_load(f)
    
    assert 'system_prompt' in prompts
    assert 'meal_planning_prompt' in prompts
    assert '{context}' in prompts['meal_planning_prompt']
    assert '{query}' in prompts['meal_planning_prompt']

def test_requirements_exist():
    """Test that requirements.txt exists and has content"""
    assert os.path.exists('requirements.txt')
    
    with open('requirements.txt', 'r') as f:
        content = f.read()
    
    assert 'sentence-transformers' in content
    assert 'requests' in content
    assert 'click' in content

