# agents.md - AI Agent Implementation Guide

## Project: RAG-Powered Home Meal & Shopping Assistant

This document provides detailed implementation instructions for AI agents to build the complete project. Each task includes implementation details, test requirements, and success criteria.

---

## 🎯 Project Context

**Goal**: Build a CLI application that uses RAG-Anything to suggest personalized meal plans and comprehensive shopping lists based on user's cooking history and preferences.

**Key Requirements**:
- Use RAG-Anything for context retrieval
- Use OpenRouter API with `requests` library (no OpenAI SDK)
- CLI-based interaction
- Include both food ingredients and household items
- Log user choices for learning
- Test-driven development

---

## 📋 Implementation Phases

---

## PHASE 1: Setup & Environment

### Task 1.1: Repository Setup

**Instructions**:
```bash
# Create project structure
mkdir -p meal-shopping-assistant/{src,data/{raw,processed,logs},config,tests}
cd meal-shopping-assistant
```

**Files to Create**:

1. **Project Structure**:
```
meal-shopping-assistant/
├── src/
│   ├── __init__.py
│   ├── cli.py
│   ├── rag_engine.py
│   ├── llm_client.py
│   ├── data_ingestion.py
│   ├── meal_planner.py
│   └── utils.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── logs/
├── config/
│   ├── config.yaml
│   └── prompts.yaml
├── tests/
│   ├── __init__.py
│   ├── test_rag_engine.py
│   ├── test_llm_client.py
│   ├── test_data_ingestion.py
│   ├── test_meal_planner.py
│   └── fixtures/
│       ├── sample_meals.json
│       ├── sample_preferences.json
│       ├── sample_bills.json
│       └── sample_grocery_lists.json
├── requirements.txt
├── setup.py
├── README.md
├── .env.example
└── .gitignore
```

2. **.gitignore**:
```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
dist/
*.egg-info/

# Environment
.env

# Data
data/processed/*
data/logs/*
!data/processed/.gitkeep
!data/logs/.gitkeep

# IDE
.vscode/
.idea/
*.swp
*.swo

# RAG database
vector_db/
*.faiss
*.index
```

3. **.env.example**:
```bash
# OpenRouter API Configuration
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_MODEL=anthropic/claude-3.5-sonnet

# RAG Configuration
VECTOR_DB_PATH=./data/processed/vector_db
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Logging
LOG_LEVEL=INFO
LOG_FILE=./data/logs/app.log
```

4. **README.md** (Basic):
```markdown
# RAG-Powered Home Meal & Shopping Assistant

A CLI application that suggests personalized meal plans and shopping lists using RAG and AI.

## Setup

1. Clone repository
2. Install dependencies: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and add your OpenRouter API key
4. Run: `meal-assistant --help`

## Development

Run tests: `pytest tests/ -v`
```

**Test Requirements**:
- [ ] Verify all directories are created
- [ ] Verify `.gitignore` excludes sensitive files
- [ ] Verify `.env.example` is tracked but `.env` is not

**Success Criteria**:
✅ Clean project structure exists
✅ Git repository initialized
✅ All placeholder files created

---

### Task 1.2: Dependencies Setup

**Instructions**:

1. **requirements.txt**:
```txt
# RAG and Embeddings
sentence-transformers==2.2.2
faiss-cpu==1.7.4
numpy==1.24.3

# LLM
requests==2.31.0
python-dotenv==1.0.0

# CLI
click==8.1.7
rich==13.7.0
colorama==0.4.6

# Data Processing
pyyaml==6.0.1
pandas==2.0.3

# Utilities
python-dateutil==2.8.2
pytz==2023.3

# Testing
pytest==7.4.3
pytest-cov==4.1.0
pytest-mock==3.12.0

# Development
black==23.12.1
flake8==6.1.0
```

2. **setup.py**:
```python
from setuptools import setup, find_packages

setup(
    name="meal-shopping-assistant",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'sentence-transformers>=2.2.2',
        'faiss-cpu>=1.7.4',
        'requests>=2.31.0',
        'python-dotenv>=1.0.0',
        'click>=8.1.7',
        'rich>=13.7.0',
        'pyyaml>=6.0.1',
        'pandas>=2.0.3',
        'python-dateutil>=2.8.2',
    ],
    entry_points={
        'console_scripts': [
            'meal-assistant=src.cli:main',
        ],
    },
    python_requires='>=3.8',
)
```

3. **config/config.yaml**:
```yaml
# Application Configuration

rag:
  vector_db_path: "./data/processed/vector_db"
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  chunk_size: 500
  chunk_overlap: 50
  top_k: 10

llm:
  api_base_url: "https://openrouter.ai/api/v1/chat/completions"
  default_model: "anthropic/claude-3.5-sonnet"
  temperature: 0.7
  max_tokens: 2000
  timeout: 30

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "./data/logs/app.log"

choices:
  log_file: "./data/logs/user_choices.json"
  
household_items:
  # Common household items to suggest
  categories:
    - cleaning_supplies
    - paper_products
    - personal_care
    - kitchen_supplies
  
  restock_interval_days:
    cleaning_supplies: 30
    paper_products: 14
    personal_care: 45
    kitchen_supplies: 60
```

4. **config/prompts.yaml**:
```yaml
system_prompt: |
  You are a helpful meal planning assistant. Your role is to suggest personalized meal plans and comprehensive shopping lists based on the user's cooking history, preferences, and household needs.
  
  Key responsibilities:
  1. Suggest meals that align with user preferences and past cooking patterns
  2. Create complete shopping lists with ingredients AND household items
  3. Consider dietary restrictions and allergies
  4. Provide variety while respecting favorites
  5. Be practical and budget-conscious

meal_planning_prompt: |
  Based on the following context about the user's meal history, preferences, and shopping patterns:
  
  {context}
  
  User request: {query}
  
  Please generate 3 distinct meal plan options that include:
  
  1. **Option Name**: A catchy, descriptive name
  2. **Meals**: List of {num_days} days of meals with:
     - Meal name
     - Brief description
     - Main ingredients
  3. **Shopping List**: Comprehensive list organized by category:
     - Produce
     - Proteins
     - Dairy
     - Grains & Pasta
     - Pantry Items
     - Household Items (cleaning supplies, paper products, etc.)
  
  Important guidelines:
  - Consider the user's preferences and past choices
  - Include variety across the options
  - Add household items if it's been a while since last purchase
  - Consolidate duplicate ingredients across meals
  - Be specific with quantities
  
  Return the response as a JSON array with this structure:
  ```json
  [
    {{
      "option_name": "Mediterranean Week",
      "description": "Light, healthy meals inspired by Mediterranean cuisine",
      "meals": [
        {{
          "day": "Monday",
          "meal_name": "Greek Salad with Grilled Chicken",
          "description": "Fresh vegetables with lemon herb chicken",
          "main_ingredients": ["chicken breast", "feta cheese", "tomatoes", "cucumbers"]
        }}
      ],
      "shopping_list": {{
        "produce": [
          {{"item": "tomatoes", "quantity": "6", "unit": "medium"}},
          {{"item": "cucumbers", "quantity": "3", "unit": "whole"}}
        ],
        "proteins": [...],
        "dairy": [...],
        "grains_pasta": [...],
        "pantry": [...],
        "household": [
          {{"item": "dish soap", "reason": "Last purchased 3 weeks ago"}},
          {{"item": "paper towels", "reason": "Frequently purchased item"}}
        ]
      }},
      "estimated_cost": "75-85",
      "prep_difficulty": "easy"
    }}
  ]
  ```

fallback_response: |
  I apologize, but I don't have enough information to make personalized suggestions yet. Please ingest some data first using:
  
  meal-assistant ingest --all ./data/raw
```

**Test Requirements**:

Create `tests/test_setup.py`:
```python
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
```

**Run Tests**:
```bash
pytest tests/test_setup.py -v
```

**Success Criteria**:
✅ All dependencies installable: `pip install -r requirements.txt`
✅ Config files are valid YAML
✅ All tests pass
✅ Package installable: `pip install -e .`

---

## PHASE 2: Utilities & Logging

### Task 2.1: Utilities Module

**Instructions**:

Create `src/utils.py`:
```python
"""Utility functions for the meal shopping assistant."""

import os
import json
import logging
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def setup_logging(config_path: str = "config/config.yaml") -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configured logger instance
    """
    config = load_config(config_path)
    log_config = config.get('logging', {})
    
    # Create logs directory if it doesn't exist
    log_file = log_config.get('file', './data/logs/app.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_config.get('level', 'INFO')),
        format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_prompts(prompts_path: str = "config/prompts.yaml") -> Dict[str, str]:
    """
    Load prompt templates from YAML file.
    
    Args:
        prompts_path: Path to prompts file
        
    Returns:
        Prompts dictionary
    """
    with open(prompts_path, 'r') as f:
        return yaml.safe_load(f)

def ensure_dir(directory: str) -> None:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        directory: Directory path to create
    """
    os.makedirs(directory, exist_ok=True)

def save_json(data: Any, filepath: str, indent: int = 2) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        filepath: Path to save file
        indent: JSON indentation level
    """
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent, default=str)

def load_json(filepath: str) -> Any:
    """
    Load data from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Loaded data
    """
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, 'r') as f:
        return json.load(f)

def parse_time_query(query: str) -> Dict[str, Any]:
    """
    Parse time-based queries like "3 days", "weekend", "week", etc.
    
    Args:
        query: User query string
        
    Returns:
        Dictionary with 'num_days' and 'occasion' keys
    """
    query_lower = query.lower()
    
    # Check for specific occasions
    if 'party' in query_lower or 'celebration' in query_lower:
        occasion = 'party'
        # Try to extract number of people
        import re
        people_match = re.search(r'(\d+)\s*people', query_lower)
        num_people = int(people_match.group(1)) if people_match else 10
        num_days = 1
        return {
            'num_days': num_days,
            'occasion': occasion,
            'num_people': num_people
        }
    
    # Parse number of days
    if 'weekend' in query_lower:
        num_days = 2
        occasion = 'weekend'
    elif 'week' in query_lower:
        num_days = 7
        occasion = 'week'
    else:
        # Try to extract number
        import re
        match = re.search(r'(\d+)\s*days?', query_lower)
        if match:
            num_days = int(match.group(1))
            occasion = 'general'
        else:
            num_days = 3  # Default
            occasion = 'general'
    
    return {
        'num_days': num_days,
        'occasion': occasion,
        'num_people': 2  # Default household size
    }

def format_shopping_list(shopping_list: Dict[str, List[Dict]]) -> str:
    """
    Format shopping list for display.
    
    Args:
        shopping_list: Shopping list dictionary organized by category
        
    Returns:
        Formatted string for display
    """
    output = []
    
    category_names = {
        'produce': '🥬 Produce',
        'proteins': '🍗 Proteins',
        'dairy': '🧀 Dairy',
        'grains_pasta': '🌾 Grains & Pasta',
        'pantry': '🏺 Pantry Items',
        'household': '🧹 Household Items'
    }
    
    for category, items in shopping_list.items():
        if items:
            display_name = category_names.get(category, category.replace('_', ' ').title())
            output.append(f"\n{display_name}:")
            for item in items:
                if isinstance(item, dict):
                    quantity = item.get('quantity', '')
                    unit = item.get('unit', '')
                    name = item.get('item', '')
                    reason = item.get('reason', '')
                    
                    if quantity and unit:
                        line = f"  • {quantity} {unit} {name}"
                    else:
                        line = f"  • {name}"
                    
                    if reason:
                        line += f" ({reason})"
                    
                    output.append(line)
                else:
                    output.append(f"  • {item}")
    
    return '\n'.join(output)

def get_days_since(date_str: str) -> int:
    """
    Calculate days since a given date string.
    
    Args:
        date_str: Date string in ISO format
        
    Returns:
        Number of days since the date
    """
    try:
        date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        now = datetime.now(date.tzinfo) if date.tzinfo else datetime.now()
        delta = now - date
        return delta.days
    except Exception:
        return 999  # Return large number if parsing fails
```

**Test Requirements**:

Create `tests/test_utils.py`:
```python
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
```

**Run Tests**:
```bash
pytest tests/test_utils.py -v
```

**Success Criteria**:
✅ All utility functions work correctly
✅ Logging is properly configured
✅ JSON operations work
✅ Time parsing works for various formats
✅ All tests pass

---

## PHASE 3: RAG Engine

### Task 3.1: RAG Engine Implementation

**Instructions**:

Create `src/rag_engine.py`:
```python
"""RAG engine for document storage and retrieval using FAISS and sentence transformers."""

import os
import logging
import pickle
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

from src.utils import load_config, ensure_dir

logger = logging.getLogger(__name__)

class RAGEngine:
    """RAG engine for storing and retrieving documents."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize RAG engine.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.rag_config = self.config['rag']
        
        # Initialize paths
        self.vector_db_path = self.rag_config['vector_db_path']
        ensure_dir(self.vector_db_path)
        
        self.index_path = os.path.join(self.vector_db_path, "faiss.index")
        self.metadata_path = os.path.join(self.vector_db_path, "metadata.pkl")
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {self.rag_config['embedding_model']}")
        self.embedding_model = SentenceTransformer(self.rag_config['embedding_model'])
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize or load FAISS index
        self.index = None
        self.documents = []
        self.metadata = []
        
        self._load_or_create_index()
    
    def _load_or_create_index(self) -> None:
        """Load existing index or create a new one."""
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            logger.info("Loading existing FAISS index")
            self.index = faiss.read_index(self.index_path)
            
            with open(self.metadata_path, 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.metadata = data['metadata']
            
            logger.info(f"Loaded index with {len(self.documents)} documents")
        else:
            logger.info("Creating new FAISS index")
            self.index = faiss.IndexFlatL2(self.embedding_dim)
    
    def add_documents(
        self,
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Add documents to the RAG database.
        
        Args:
            documents: List of document strings
            metadata: Optional list of metadata dictionaries
        """
        if not documents:
            logger.warning("No documents to add")
            return
        
        logger.info(f"Adding {len(documents)} documents to RAG database")
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(
            documents,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Store documents and metadata
        self.documents.extend(documents)
        
        if metadata is None:
            metadata = [{}] * len(documents)
        self.metadata.extend(metadata)
        
        # Save index
        self._save_index()
        
        logger.info(f"Successfully added {len(documents)} documents. Total: {len(self.documents)}")
    
    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Query the RAG database.
        
        Args:
            query: Query string
            top_k: Number of results to return (default from config)
            filter_metadata: Optional metadata filters
            
        Returns:
            List of result dictionaries with 'document', 'metadata', and 'score'
        """
        if top_k is None:
            top_k = self.rag_config['top_k']
        
        if len(self.documents) == 0:
            logger.warning("No documents in database")
            return []
        
        logger.info(f"Querying RAG database: '{query[:50]}...'")
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True
        ).astype('float32')
        
        # Search FAISS index
        distances, indices = self.index.search(query_embedding, min(top_k, len(self.documents)))
        
        # Prepare results
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            
            doc_metadata = self.metadata[idx]
            
            # Apply metadata filter if provided
            if filter_metadata:
                match = all(
                    doc_metadata.get(key) == value
                    for key, value in filter_metadata.items()
                )
                if not match:
                    continue
            
            results.append({
                'document': self.documents[idx],
                'metadata': doc_metadata,
                'score': float(distance)
            })
        
        logger.info(f"Found {len(results)} relevant documents")
        return results
    
    def get_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Get formatted context string for LLM prompt.
        
        Args:
            query: Query string
            top_k: Number of results to include
            filter_metadata: Optional metadata filters
            
        Returns:
            Formatted context string
        """
        results = self.query(query, top_k, filter_metadata)
        
        if not results:
            return "No relevant information found in the database."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            doc_type = result['metadata'].get('type', 'unknown')
            context_parts.append(f"[Document {i} - {doc_type}]")
            context_parts.append(result['document'])
            context_parts.append("")  # Empty line
        
        return '\n'.join(context_parts)
    
    def _save_index(self) -> None:
        """Save FAISS index and metadata to disk."""
        logger.info("Saving FAISS index")
        faiss.write_index(self.index, self.index_path)
        
        with open(self.metadata_path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.metadata
            }, f)
    
    def reset_database(self) -> None:
        """Reset the database (clear all documents)."""
        logger.warning("Resetting RAG database")
        
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.documents = []
        self.metadata = []
        
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.metadata_path):
            os.remove(self.metadata_path)
        
        logger.info("Database reset complete")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the database.
        
        Returns:
            Dictionary with statistics
        """
        type_counts = {}
        for meta in self.metadata:
            doc_type = meta.get('type', 'unknown')
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        
        return {
            'total_documents': len(self.documents),
            'by_type': type_counts,
            'index_size': self.index.ntotal
        }
```

**Test Requirements**:

Create `tests/test_rag_engine.py`:
```python
import pytest
import os
import shutil
from src.rag_engine import RAGEngine

@pytest.fixture
def temp_rag_engine(tmp_path, monkeypatch):
    """Create a temporary RAG engine for testing"""
    # Create temporary config
    config = {
        'rag': {
            'vector_db_path': str(tmp_path / 'vector_db'),
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'chunk_size': 500,
            'chunk_overlap': 50,
            'top_k': 5
        }
    }
    
# Mock load_config to use temporary config
def mock_load_config(path):
    return config

monkeypatch.setattr('src.rag_engine.load_config', mock_load_config)

engine = RAGEngine()
yield engine

# Cleanup
if os.path.exists(str(tmp_path / 'vector_db')):
    shutil.rmtree(str(tmp_path / 'vector_db'))

def test_rag_engine_initialization(temp_rag_engine):
    """Test RAG engine initialization"""
    assert temp_rag_engine.index is not None
    assert temp_rag_engine.embedding_model is not None
    assert temp_rag_engine.embedding_dim > 0

def test_add_documents(temp_rag_engine):
    """Test adding documents"""
    documents = [
        "I cooked spaghetti carbonara on Monday with eggs, pasta, and bacon.",
        "User prefers vegetarian meals on weekdays.",
        "Bought dish soap and paper towels at the store."
    ]
    
    metadata = [
        {'type': 'meal', 'date': '2024-01-15'},
        {'type': 'preference', 'category': 'dietary'},
        {'type': 'bill', 'date': '2024-01-20'}
    ]
    
    temp_rag_engine.add_documents(documents, metadata)
    
    assert len(temp_rag_engine.documents) == 3
    assert len(temp_rag_engine.metadata) == 3
    assert temp_rag_engine.index.ntotal == 3

def test_query_documents(temp_rag_engine):
    """Test querying documents"""
    documents = [
        "I cooked spaghetti carbonara with pasta, eggs, bacon, and parmesan cheese.",
        "User loves Italian food, especially pasta dishes.",
        "Bought pasta, eggs, and bacon at grocery store for $15.50."
    ]
    
    metadata = [
        {'type': 'meal', 'cuisine': 'italian'},
        {'type': 'preference', 'cuisine': 'italian'},
        {'type': 'bill', 'store': 'grocery'}
    ]
    
    temp_rag_engine.add_documents(documents, metadata)
    
    results = temp_rag_engine.query("Italian pasta meals", top_k=2)
    
    assert len(results) <= 2
    assert all('document' in r for r in results)
    assert all('metadata' in r for r in results)
    assert all('score' in r for r in results)

def test_query_with_metadata_filter(temp_rag_engine):
    """Test querying with metadata filters"""
    documents = [
        "Cooked chicken curry on Tuesday",
        "User prefers spicy food",
        "Bought chicken at store"
    ]
    
    metadata = [
        {'type': 'meal', 'date': '2024-01-16'},
        {'type': 'preference', 'taste': 'spicy'},
        {'type': 'bill', 'date': '2024-01-15'}
    ]
    
    temp_rag_engine.add_documents(documents, metadata)
    
    # Filter by type
    results = temp_rag_engine.query(
        "chicken",
        top_k=5,
        filter_metadata={'type': 'meal'}
    )
    
    assert len(results) >= 1
    assert all(r['metadata']['type'] == 'meal' for r in results)

def test_get_context(temp_rag_engine):
    """Test getting formatted context"""
    documents = [
        "User loves Mexican food",
        "Cooked tacos last week"
    ]
    
    metadata = [
        {'type': 'preference'},
        {'type': 'meal'}
    ]
    
    temp_rag_engine.add_documents(documents, metadata)
    
    context = temp_rag_engine.get_context("Mexican food preferences", top_k=2)
    
    assert isinstance(context, str)
    assert len(context) > 0
    assert 'Document 1' in context or 'Document 2' in context

def test_empty_query(temp_rag_engine):
    """Test querying empty database"""
    results = temp_rag_engine.query("test query")
    assert results == []

def test_get_stats(temp_rag_engine):
    """Test getting database statistics"""
    documents = [
        "Meal 1",
        "Meal 2",
        "Preference 1",
        "Bill 1"
    ]
    
    metadata = [
        {'type': 'meal'},
        {'type': 'meal'},
        {'type': 'preference'},
        {'type': 'bill'}
    ]
    
    temp_rag_engine.add_documents(documents, metadata)
    
    stats = temp_rag_engine.get_stats()
    
    assert stats['total_documents'] == 4
    assert stats['by_type']['meal'] == 2
    assert stats['by_type']['preference'] == 1
    assert stats['by_type']['bill'] == 1
    assert stats['index_size'] == 4

def test_reset_database(temp_rag_engine):
    """Test resetting database"""
    documents = ["Test document"]
    temp_rag_engine.add_documents(documents)
    
    assert len(temp_rag_engine.documents) == 1
    
    temp_rag_engine.reset_database()
    
    assert len(temp_rag_engine.documents) == 0
    assert len(temp_rag_engine.metadata) == 0
    assert temp_rag_engine.index.ntotal == 0

def test_persistence(temp_rag_engine, tmp_path, monkeypatch):
    """Test that index persists between sessions"""
    documents = ["Persistent document"]
    metadata = [{'type': 'test'}]
    
    temp_rag_engine.add_documents(documents, metadata)
    
    # Create new instance (simulating restart)
    config = {
        'rag': {
            'vector_db_path': str(tmp_path / 'vector_db'),
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'chunk_size': 500,
            'chunk_overlap': 50,
            'top_k': 5
        }
    }
    
    def mock_load_config(path):
        return config
    
    monkeypatch.setattr('src.rag_engine.load_config', mock_load_config)
    
    new_engine = RAGEngine()
    
    assert len(new_engine.documents) == 1
    assert new_engine.documents[0] == "Persistent document"
```

**Run Tests**:
```bash
pytest tests/test_rag_engine.py -v
```

**Success Criteria**:
✅ RAG engine initializes correctly
✅ Documents can be added with metadata
✅ Queries return relevant results
✅ Metadata filtering works
✅ Index persists between sessions
✅ All tests pass

---

## PHASE 4: Data Ingestion

### Task 4.1: Data Schemas and Sample Data

**Instructions**:

Create `tests/fixtures/sample_meals.json`:
```json
[
  {
    "date": "2024-01-15",
    "meal_name": "Spaghetti Carbonara",
    "cuisine": "Italian",
    "servings": 4,
    "ingredients": [
      "spaghetti pasta (400g)",
      "eggs (4 large)",
      "bacon (200g)",
      "parmesan cheese (100g)",
      "black pepper",
      "salt"
    ],
    "prep_time_minutes": 30,
    "difficulty": "medium",
    "notes": "Family loved it! Make more next time.",
    "rating": 5
  },
  {
    "date": "2024-01-16",
    "meal_name": "Chicken Stir Fry",
    "cuisine": "Asian",
    "servings": 3,
    "ingredients": [
      "chicken breast (500g)",
      "broccoli (1 head)",
      "bell peppers (2)",
      "soy sauce",
      "garlic (4 cloves)",
      "ginger (1 inch)",
      "rice (2 cups)"
    ],
    "prep_time_minutes": 25,
    "difficulty": "easy",
    "notes": "Quick weeknight dinner",
    "rating": 4
  },
  {
    "date": "2024-01-18",
    "meal_name": "Vegetarian Tacos",
    "cuisine": "Mexican",
    "servings": 4,
    "ingredients": [
      "black beans (2 cans)",
      "corn (1 can)",
      "avocado (2)",
      "tomatoes (3)",
      "lettuce (1 head)",
      "tortillas (8)",
      "cheese (200g)",
      "lime (2)",
      "cilantro"
    ],
    "prep_time_minutes": 20,
    "difficulty": "easy",
    "notes": "Great for Meatless Monday",
    "rating": 5
  },
  {
    "date": "2024-01-20",
    "meal_name": "Salmon with Roasted Vegetables",
    "cuisine": "Contemporary",
    "servings": 2,
    "ingredients": [
      "salmon fillets (2, 150g each)",
      "asparagus (1 bunch)",
      "cherry tomatoes (200g)",
      "olive oil",
      "lemon (1)",
      "garlic (3 cloves)",
      "herbs (rosemary, thyme)"
    ],
    "prep_time_minutes": 35,
    "difficulty": "medium",
    "notes": "Healthy and delicious",
    "rating": 5
  },
  {
    "date": "2024-01-22",
    "meal_name": "Beef Curry",
    "cuisine": "Indian",
    "servings": 6,
    "ingredients": [
      "beef stew meat (800g)",
      "onions (3)",
      "curry paste (3 tbsp)",
      "coconut milk (400ml)",
      "potatoes (4)",
      "carrots (3)",
      "rice (3 cups)",
      "garlic (5 cloves)",
      "ginger (2 inches)"
    ],
    "prep_time_minutes": 90,
    "difficulty": "medium",
    "notes": "Made extra for leftovers",
    "rating": 5
  }
]
```

Create `tests/fixtures/sample_preferences.json`:
```json
{
  "dietary_restrictions": [
    "No shellfish (allergy)"
  ],
  "favorite_cuisines": [
    "Italian",
    "Mexican",
    "Asian"
  ],
  "disliked_foods": [
    "cilantro (for John)",
    "mushrooms (for Sarah)"
  ],
  "preferred_proteins": [
    "chicken",
    "fish",
    "beans"
  ],
  "cooking_frequency": {
    "weekday": "quick meals under 30 minutes",
    "weekend": "can spend more time, try new recipes"
  },
  "household_size": 4,
  "notes": [
    "Kids prefer mild flavors",
    "Try to incorporate more vegetables",
    "Monday is meatless day",
    "Friday is usually pizza or takeout night"
  ],
  "budget": {
    "weekly_grocery": 150,
    "currency": "USD"
  },
  "meal_preferences": {
    "breakfast": "usually light - yogurt, fruit, toast",
    "lunch": "leftovers or sandwiches",
    "dinner": "main meal of the day, prefer home-cooked"
  }
}
```

Create `tests/fixtures/sample_bills.json`:
```json
[
  {
    "date": "2024-01-14",
    "store": "Whole Foods",
    "total_amount": 87.45,
    "currency": "USD",
    "items": [
      {"name": "spaghetti pasta", "quantity": 2, "unit": "pack", "price": 3.99, "category": "grains"},
      {"name": "eggs", "quantity": 1, "unit": "dozen", "price": 4.50, "category": "dairy"},
      {"name": "bacon", "quantity": 1, "unit": "pack", "price": 8.99, "category": "meat"},
      {"name": "parmesan cheese", "quantity": 1, "unit": "block", "price": 7.99, "category": "dairy"},
      {"name": "chicken breast", "quantity": 1.5, "unit": "lb", "price": 12.50, "category": "meat"},
      {"name": "broccoli", "quantity": 2, "unit": "head", "price": 3.98, "category": "produce"},
      {"name": "bell peppers", "quantity": 3, "unit": "each", "price": 4.50, "category": "produce"},
      {"name": "dish soap", "quantity": 1, "unit": "bottle", "price": 4.99, "category": "household"},
      {"name": "paper towels", "quantity": 1, "unit": "pack", "price": 12.99, "category": "household"},
      {"name": "olive oil", "quantity": 1, "unit": "bottle", "price": 14.99, "category": "pantry"},
      {"name": "rice", "quantity": 1, "unit": "bag", "price": 8.99, "category": "grains"}
    ]
  },
  {
    "date": "2024-01-19",
    "store": "Trader Joe's",
    "total_amount": 65.30,
    "currency": "USD",
    "items": [
      {"name": "salmon fillets", "quantity": 2, "unit": "pack", "price": 14.99, "category": "seafood"},
      {"name": "asparagus", "quantity": 1, "unit": "bunch", "price": 3.99, "category": "produce"},
      {"name": "cherry tomatoes", "quantity": 2, "unit": "pack", "price": 5.98, "category": "produce"},
      {"name": "avocados", "quantity": 4, "unit": "each", "price": 5.96, "category": "produce"},
      {"name": "black beans", "quantity": 3, "unit": "can", "price": 2.97, "category": "canned"},
      {"name": "tortillas", "quantity": 2, "unit": "pack", "price": 5.98, "category": "grains"},
      {"name": "cheese", "quantity": 1, "unit": "pack", "price": 6.99, "category": "dairy"},
      {"name": "lettuce", "quantity": 1, "unit": "head", "price": 2.99, "category": "produce"},
      {"name": "laundry detergent", "quantity": 1, "unit": "bottle", "price": 11.99, "category": "household"}
    ]
  },
  {
    "date": "2024-01-21",
    "store": "Local Market",
    "total_amount": 45.20,
    "currency": "USD",
    "items": [
      {"name": "beef stew meat", "quantity": 2, "unit": "lb", "price": 18.99, "category": "meat"},
      {"name": "onions", "quantity": 5, "unit": "each", "price": 2.50, "category": "produce"},
      {"name": "potatoes", "quantity": 5, "unit": "lb", "price": 4.99, "category": "produce"},
      {"name": "carrots", "quantity": 2, "unit": "lb", "price": 2.99, "category": "produce"},
      {"name": "coconut milk", "quantity": 2, "unit": "can", "price": 5.98, "category": "canned"},
      {"name": "curry paste", "quantity": 1, "unit": "jar", "price": 4.99, "category": "pantry"},
      {"name": "toilet paper", "quantity": 1, "unit": "pack", "price": 9.99, "category": "household"}
    ]
  }
]
```

Create `tests/fixtures/sample_grocery_lists.json`:
```json
[
  {
    "date": "2024-01-07",
    "occasion": "weekly shopping",
    "items": [
      "milk",
      "bread",
      "eggs",
      "chicken",
      "vegetables",
      "rice",
      "pasta",
      "cheese",
      "yogurt",
      "fruit",
      "dish soap",
      "trash bags"
    ],
    "completed": true,
    "notes": "Regular weekly stock-up"
  },
  {
    "date": "2024-01-12",
    "occasion": "party preparation",
    "items": [
      "ground beef",
      "burger buns",
      "lettuce",
      "tomatoes",
      "chips",
      "soda",
      "ice cream",
      "napkins",
      "paper plates",
      "cups"
    ],
    "completed": true,
    "notes": "Birthday party for 15 people"
  }
]
```

### Task 4.2: Data Ingestion Module

**Instructions**:

Create `src/data_ingestion.py`:
```python
"""Data ingestion module for processing and loading various data types into RAG."""

import os
import json
import logging
from typing import List, Dict, Any, Tuple
from datetime import datetime
from src.utils import load_json, get_days_since

logger = logging.getLogger(__name__)

class DataIngestion:
    """Handle ingestion of meal data, preferences, bills, and grocery lists."""
    
    def __init__(self):
        """Initialize data ingestion."""
        self.processed_documents = []
        self.processed_metadata = []
    
    def load_meals(self, filepath: str) -> Tuple[List[str], List[Dict]]:
        """
        Load and process meal data.
        
        Args:
            filepath: Path to meals JSON file
            
        Returns:
            Tuple of (documents, metadata)
        """
        logger.info(f"Loading meals from {filepath}")
        
        meals = load_json(filepath)
        if not meals:
            logger.warning(f"No meals found in {filepath}")
            return [], []
        
        documents = []
        metadata = []
        
        for meal in meals:
            # Create rich text representation
            doc_text = self._format_meal_document(meal)
            documents.append(doc_text)
            
            # Create metadata
            meta = {
                'type': 'meal',
                'date': meal.get('date'),
                'meal_name': meal.get('meal_name'),
                'cuisine': meal.get('cuisine'),
                'rating': meal.get('rating'),
                'difficulty': meal.get('difficulty')
            }
            metadata.append(meta)
        
        logger.info(f"Processed {len(documents)} meals")
        return documents, metadata
    
    def _format_meal_document(self, meal: Dict[str, Any]) -> str:
        """Format meal data as a document string."""
        parts = []
        
        # Basic info
        parts.append(f"Meal: {meal.get('meal_name', 'Unknown')}")
        parts.append(f"Date: {meal.get('date', 'Unknown')}")
        parts.append(f"Cuisine: {meal.get('cuisine', 'Unknown')}")
        parts.append(f"Servings: {meal.get('servings', 'Unknown')}")
        
        # Ingredients
        ingredients = meal.get('ingredients', [])
        if ingredients:
            parts.append(f"Ingredients: {', '.join(ingredients)}")
        
        # Cooking details
        if meal.get('prep_time_minutes'):
            parts.append(f"Prep time: {meal['prep_time_minutes']} minutes")
        if meal.get('difficulty'):
            parts.append(f"Difficulty: {meal['difficulty']}")
        
        # Rating and notes
        if meal.get('rating'):
            parts.append(f"Rating: {meal['rating']}/5")
        if meal.get('notes'):
            parts.append(f"Notes: {meal['notes']}")
        
        return '\n'.join(parts)
    
    def load_preferences(self, filepath: str) -> Tuple[List[str], List[Dict]]:
        """
        Load and process user preferences.
        
        Args:
            filepath: Path to preferences JSON file
            
        Returns:
            Tuple of (documents, metadata)
        """
        logger.info(f"Loading preferences from {filepath}")
        
        prefs = load_json(filepath)
        if not prefs:
            logger.warning(f"No preferences found in {filepath}")
            return [], []
        
        documents = []
        metadata = []
        
        # Process dietary restrictions
        if prefs.get('dietary_restrictions'):
            doc = f"Dietary Restrictions: {', '.join(prefs['dietary_restrictions'])}"
            documents.append(doc)
            metadata.append({'type': 'preference', 'category': 'dietary_restrictions'})
        
        # Process favorite cuisines
        if prefs.get('favorite_cuisines'):
            doc = f"Favorite Cuisines: {', '.join(prefs['favorite_cuisines'])}"
            documents.append(doc)
            metadata.append({'type': 'preference', 'category': 'favorite_cuisines'})
        
        # Process disliked foods
        if prefs.get('disliked_foods'):
            doc = f"Disliked Foods: {', '.join(prefs['disliked_foods'])}"
            documents.append(doc)
            metadata.append({'type': 'preference', 'category': 'disliked_foods'})
        
        # Process preferred proteins
        if prefs.get('preferred_proteins'):
            doc = f"Preferred Proteins: {', '.join(prefs['preferred_proteins'])}"
            documents.append(doc)
            metadata.append({'type': 'preference', 'category': 'proteins'})
        
        # Process cooking frequency
        if prefs.get('cooking_frequency'):
            freq = prefs['cooking_frequency']
            doc = f"Cooking Frequency: Weekdays - {freq.get('weekday', 'N/A')}, Weekends - {freq.get('weekend', 'N/A')}"
            documents.append(doc)
            metadata.append({'type': 'preference', 'category': 'cooking_frequency'})
        
        # Process household info
        if prefs.get('household_size'):
            doc = f"Household Size: {prefs['household_size']} people"
            documents.append(doc)
            metadata.append({'type': 'preference', 'category': 'household'})
        
        # Process meal preferences
        if prefs.get('meal_preferences'):
            meal_prefs = prefs['meal_preferences']
            for meal_type, pref in meal_prefs.items():
                doc = f"{meal_type.title()} Preference: {pref}"
                documents.append(doc)
                metadata.append({'type': 'preference', 'category': 'meal_preferences'})
        
        # Process notes
        if prefs.get('notes'):
            for note in prefs['notes']:
                documents.append(f"Preference Note: {note}")
                metadata.append({'type': 'preference', 'category': 'notes'})
        
        # Process budget
        if prefs.get('budget'):
            budget = prefs['budget']
            doc = f"Weekly Grocery Budget: {budget.get('currency', 'USD')} {budget.get('weekly_grocery', 'N/A')}"
            documents.append(doc)
            metadata.append({'type': 'preference', 'category': 'budget'})
        
        logger.info(f"Processed {len(documents)} preference items")
        return documents, metadata
    
    def load_bills(self, filepath: str) -> Tuple[List[str], List[Dict]]:
        """
        Load and process shopping bills.
        
        Args:
            filepath: Path to bills JSON file
            
        Returns:
            Tuple of (documents, metadata)
        """
        logger.info(f"Loading bills from {filepath}")
        
        bills = load_json(filepath)
        if not bills:
            logger.warning(f"No bills found in {filepath}")
            return [], []
        
        documents = []
        metadata = []
        
        for bill in bills:
            # Create document for overall bill
            doc_text = self._format_bill_document(bill)
            documents.append(doc_text)
            
            meta = {
                'type': 'bill',
                'date': bill.get('date'),
                'store': bill.get('store'),
                'total': bill.get('total_amount')
            }
            metadata.append(meta)
            
            # Create separate documents for household items (for restocking logic)
            items = bill.get('items', [])
            for item in items:
                if item.get('category') == 'household':
                    item_doc = (
                        f"Household Item Purchase: {item.get('name')} "
                        f"bought on {bill.get('date')} at {bill.get('store')} "
                        f"for {item.get('price')}"
                    )
                    documents.append(item_doc)
                    
                    item_meta = {
                        'type': 'household_item',
                        'date': bill.get('date'),
                        'item_name': item.get('name'),
                        'store': bill.get('store'),
                        'category': 'household'
                    }
                    metadata.append(item_meta)
        
        logger.info(f"Processed {len(documents)} bill items")
        return documents, metadata
    
    def _format_bill_document(self, bill: Dict[str, Any]) -> str:
        """Format bill data as a document string."""
        parts = []
        
        parts.append(f"Shopping Bill from {bill.get('store', 'Unknown Store')}")
        parts.append(f"Date: {bill.get('date', 'Unknown')}")
        parts.append(f"Total: {bill.get('currency', 'USD')} {bill.get('total_amount', 0)}")
        
        # Summarize items by category
        items = bill.get('items', [])
        if items:
            categories = {}
            for item in items:
                cat = item.get('category', 'other')
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(item.get('name'))
            
            parts.append("Items purchased:")
            for category, item_names in categories.items():
                parts.append(f"  {category.title()}: {', '.join(item_names)}")
        
        return '\n'.join(parts)
    
    def load_grocery_lists(self, filepath: str) -> Tuple[List[str], List[Dict]]:
        """
        Load and process historical grocery lists.
        
        Args:
            filepath: Path to grocery lists JSON file
            
        Returns:
            Tuple of (documents, metadata)
        """
        logger.info(f"Loading grocery lists from {filepath}")
        
        lists = load_json(filepath)
        if not lists:
            logger.warning(f"No grocery lists found in {filepath}")
            return [], []
        
        documents = []
        metadata = []
        
        for grocery_list in lists:
            doc_text = self._format_grocery_list_document(grocery_list)
            documents.append(doc_text)
            
            meta = {
                'type': 'grocery_list',
                'date': grocery_list.get('date'),
                'occasion': grocery_list.get('occasion'),
                'completed': grocery_list.get('completed')
            }
            metadata.append(meta)
        
        logger.info(f"Processed {len(documents)} grocery lists")
        return documents, metadata
    
    def _format_grocery_list_document(self, grocery_list: Dict[str, Any]) -> str:
        """Format grocery list as a document string."""
        parts = []
        
        parts.append(f"Grocery List for {grocery_list.get('occasion', 'Unknown')}")
        parts.append(f"Date: {grocery_list.get('date', 'Unknown')}")
        
        items = grocery_list.get('items', [])
        if items:
            parts.append(f"Items: {', '.join(items)}")
        
        if grocery_list.get('notes'):
            parts.append(f"Notes: {grocery_list['notes']}")
        
        status = "Completed" if grocery_list.get('completed') else "Pending"
        parts.append(f"Status: {status}")
        
        return '\n'.join(parts)
    
    def ingest_all_data(
        self,
        data_dir: str,
        rag_engine
    ) -> Dict[str, int]:
        """
        Ingest all data files from a directory.
        
        Args:
            data_dir: Directory containing data files
            rag_engine: RAG engine instance to add documents to
            
        Returns:
            Dictionary with counts of ingested items by type
        """
        logger.info(f"Ingesting all data from {data_dir}")
        
        counts = {
            'meals': 0,
            'preferences': 0,
            'bills': 0,
            'grocery_lists': 0
        }
        
        # Load meals
        meals_path = os.path.join(data_dir, 'meals.json')
        if os.path.exists(meals_path):
            docs, meta = self.load_meals(meals_path)
            if docs:
                rag_engine.add_documents(docs, meta)
                counts['meals'] = len(docs)
        
        # Load preferences
        prefs_path = os.path.join(data_dir, 'preferences.json')
        if os.path.exists(prefs_path):
            docs, meta = self.load_preferences(prefs_path)
            if docs:
                rag_engine.add_documents(docs, meta)
                counts['preferences'] = len(docs)
        
        # Load bills
        bills_path = os.path.join(data_dir, 'bills.json')
        if os.path.exists(bills_path):
            docs, meta = self.load_bills(bills_path)
            if docs:
                rag_engine.add_documents(docs, meta)
                counts['bills'] = len(docs)
        
        # Load grocery lists
        lists_path = os.path.join(data_dir, 'grocery_lists.json')
        if os.path.exists(lists_path):
            docs, meta = self.load_grocery_lists(lists_path)
            if docs:
                rag_engine.add_documents(docs, meta)
                counts['grocery_lists'] = len(docs)
        
        logger.info(f"Ingestion complete: {counts}")
        return counts
```

**Test Requirements**:

Create `tests/test_data_ingestion.py`:
```python
import pytest
import os
import json
from src.data_ingestion import DataIngestion

@pytest.fixture
def data_ingestion():
    """Create DataIngestion instance"""
    return DataIngestion()

@pytest.fixture
def fixtures_dir():
    """Get fixtures directory path"""
    return os.path.join('tests', 'fixtures')

def test_load_meals(data_ingestion, fixtures_dir):
    """Test loading meals"""
    filepath = os.path.join(fixtures_dir, 'sample_meals.json')
    docs, meta = data_ingestion.load_meals(filepath)
    
    assert len(docs) == 5
    ```python
    assert len(meta) == 5
    assert all(m['type'] == 'meal' for m in meta)
    assert 'Spaghetti Carbonara' in docs[0]
    assert 'Italian' in docs[0]

def test_load_preferences(data_ingestion, fixtures_dir):
    """Test loading preferences"""
    filepath = os.path.join(fixtures_dir, 'sample_preferences.json')
    docs, meta = data_ingestion.load_preferences(filepath)
    
    assert len(docs) > 0
    assert all(m['type'] == 'preference' for m in meta)
    
    # Check that various preference types are loaded
    categories = [m['category'] for m in meta]
    assert 'dietary_restrictions' in categories
    assert 'favorite_cuisines' in categories

def test_load_bills(data_ingestion, fixtures_dir):
    """Test loading bills"""
    filepath = os.path.join(fixtures_dir, 'sample_bills.json')
    docs, meta = data_ingestion.load_bills(filepath)
    
    assert len(docs) > 0
    
    # Should have both bill documents and household item documents
    bill_docs = [m for m in meta if m['type'] == 'bill']
    household_docs = [m for m in meta if m['type'] == 'household_item']
    
    assert len(bill_docs) == 3
    assert len(household_docs) > 0

def test_load_grocery_lists(data_ingestion, fixtures_dir):
    """Test loading grocery lists"""
    filepath = os.path.join(fixtures_dir, 'sample_grocery_lists.json')
    docs, meta = data_ingestion.load_grocery_lists(filepath)
    
    assert len(docs) == 2
    assert len(meta) == 2
    assert all(m['type'] == 'grocery_list' for m in meta)

def test_format_meal_document(data_ingestion):
    """Test meal document formatting"""
    meal = {
        'meal_name': 'Test Meal',
        'date': '2024-01-15',
        'cuisine': 'Italian',
        'servings': 4,
        'ingredients': ['pasta', 'sauce'],
        'rating': 5,
        'notes': 'Delicious!'
    }
    
    doc = data_ingestion._format_meal_document(meal)
    
    assert 'Test Meal' in doc
    assert '2024-01-15' in doc
    assert 'Italian' in doc
    assert 'pasta' in doc
    assert 'Delicious!' in doc

def test_format_bill_document(data_ingestion):
    """Test bill document formatting"""
    bill = {
        'store': 'Test Store',
        'date': '2024-01-15',
        'total_amount': 50.00,
        'currency': 'USD',
        'items': [
            {'name': 'milk', 'category': 'dairy'},
            {'name': 'bread', 'category': 'grains'}
        ]
    }
    
    doc = data_ingestion._format_bill_document(bill)
    
    assert 'Test Store' in doc
    assert '2024-01-15' in doc
    assert '50.0' in doc
    assert 'milk' in doc

def test_format_grocery_list_document(data_ingestion):
    """Test grocery list document formatting"""
    grocery_list = {
        'date': '2024-01-15',
        'occasion': 'weekly shopping',
        'items': ['milk', 'bread', 'eggs'],
        'completed': True,
        'notes': 'Regular shopping'
    }
    
    doc = data_ingestion._format_grocery_list_document(grocery_list)
    
    assert 'weekly shopping' in doc
    assert '2024-01-15' in doc
    assert 'milk' in doc
    assert 'Completed' in doc

def test_load_nonexistent_file(data_ingestion):
    """Test loading non-existent file"""
    docs, meta = data_ingestion.load_meals('nonexistent.json')
    assert docs == []
    assert meta == []

def test_ingest_all_data(data_ingestion, fixtures_dir, temp_rag_engine):
    """Test ingesting all data files"""
    counts = data_ingestion.ingest_all_data(fixtures_dir, temp_rag_engine)
    
    assert counts['meals'] > 0
    assert counts['preferences'] > 0
    assert counts['bills'] > 0
    assert counts['grocery_lists'] > 0
    
    # Verify data was added to RAG engine
    stats = temp_rag_engine.get_stats()
    assert stats['total_documents'] > 0
```

**Run Tests**:
```bash
pytest tests/test_data_ingestion.py -v
```

**Success Criteria**:
✅ All data types load correctly
✅ Documents are properly formatted
✅ Metadata is correctly generated
✅ Batch ingestion works
✅ All tests pass

---

## PHASE 5: LLM Client

### Task 5.1: OpenRouter Client Implementation

**Instructions**:

Create `src/llm_client.py`:
```python
"""OpenRouter API client for LLM interactions."""

import os
import json
import logging
import time
import requests
from typing import Dict, Any, Optional, List
from src.utils import load_config

logger = logging.getLogger(__name__)

class OpenRouterClient:
    """Client for OpenRouter API using requests library."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize OpenRouter client.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.llm_config = self.config['llm']
        
        # Get API key from environment
        self.api_key = os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not found in environment. "
                "Please set it in your .env file."
            )
        
        # API configuration
        self.api_base_url = self.llm_config['api_base_url']
        self.default_model = os.getenv(
            'OPENROUTER_MODEL',
            self.llm_config['default_model']
        )
        self.temperature = self.llm_config['temperature']
        self.max_tokens = self.llm_config['max_tokens']
        self.timeout = self.llm_config['timeout']
        
        logger.info(f"Initialized OpenRouter client with model: {self.default_model}")
    
    def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate response from OpenRouter API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (default from config)
            max_tokens: Maximum tokens to generate (default from config)
            model: Model to use (default from config)
            
        Returns:
            Dictionary with 'content', 'model', 'usage', and 'finish_reason'
        """
        if temperature is None:
            temperature = self.temperature
        if max_tokens is None:
            max_tokens = self.max_tokens
        if model is None:
            model = self.default_model
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/meal-shopping-assistant",
            "X-Title": "Meal Shopping Assistant"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        logger.info(f"Sending request to OpenRouter API (model: {model})")
        logger.debug(f"Messages: {json.dumps(messages, indent=2)}")
        
        # Retry logic with exponential backoff
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_base_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                
                # Handle rate limiting
                if response.status_code == 429:
                    if attempt < max_retries - 1:
                        logger.warning(f"Rate limited. Retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    else:
                        raise Exception("Rate limit exceeded. Please try again later.")
                
                # Handle other errors
                response.raise_for_status()
                
                result = response.json()
                
                # Extract response
                content = result['choices'][0]['message']['content']
                finish_reason = result['choices'][0].get('finish_reason', 'unknown')
                usage = result.get('usage', {})
                
                logger.info(
                    f"Received response: {usage.get('total_tokens', 'unknown')} tokens, "
                    f"finish_reason: {finish_reason}"
                )
                logger.debug(f"Response content: {content[:200]}...")
                
                return {
                    'content': content,
                    'model': result.get('model', model),
                    'usage': usage,
                    'finish_reason': finish_reason
                }
                
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    logger.warning(f"Request timeout. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise Exception("Request timeout after multiple retries")
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error: {str(e)}")
                if attempt < max_retries - 1 and response.status_code >= 500:
                    logger.warning(f"Server error. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise Exception(f"API request failed: {str(e)}")
            
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                raise
        
        raise Exception("Failed to get response after multiple retries")
    
    def generate_meal_plans(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> str:
        """
        Generate meal plans using the LLM.
        
        Args:
            system_prompt: System prompt with instructions
            user_prompt: User prompt with context and query
            
        Returns:
            Generated meal plan content
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        result = self.generate_response(messages)
        return result['content']
    
    def parse_json_response(self, content: str) -> Any:
        """
        Parse JSON from LLM response, handling markdown code blocks.
        
        Args:
            content: Response content that may contain JSON
            
        Returns:
            Parsed JSON object
        """
        # Remove markdown code blocks if present
        content = content.strip()
        
        # Remove ```json or ``` markers
        if content.startswith('```json'):
            content = content[7:]
        elif content.startswith('```'):
            content = content[3:]
        
        if content.endswith('```'):
            content = content[:-3]
        
        content = content.strip()
        
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.error(f"Content: {content[:500]}")
            raise ValueError(f"Invalid JSON response from LLM: {str(e)}")
```

**Test Requirements**:

Create `tests/test_llm_client.py`:
```python
import pytest
import os
import json
from unittest.mock import Mock, patch, MagicMock
from src.llm_client import OpenRouterClient

@pytest.fixture
def mock_env(monkeypatch):
    """Mock environment variables"""
    monkeypatch.setenv('OPENROUTER_API_KEY', 'test-api-key')

@pytest.fixture
def llm_client(mock_env, monkeypatch):
    """Create LLM client with mocked config"""
    mock_config = {
        'llm': {
            'api_base_url': 'https://openrouter.ai/api/v1/chat/completions',
            'default_model': 'anthropic/claude-3.5-sonnet',
            'temperature': 0.7,
            'max_tokens': 2000,
            'timeout': 30
        }
    }
    
    def mock_load_config(path):
        return mock_config
    
    monkeypatch.setattr('src.llm_client.load_config', mock_load_config)
    
    return OpenRouterClient()

def test_client_initialization(llm_client):
    """Test client initializes correctly"""
    assert llm_client.api_key == 'test-api-key'
    assert llm_client.default_model == 'anthropic/claude-3.5-sonnet'
    assert llm_client.temperature == 0.7

def test_client_missing_api_key(monkeypatch):
    """Test client raises error without API key"""
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    
    mock_config = {
        'llm': {
            'api_base_url': 'https://openrouter.ai/api/v1/chat/completions',
            'default_model': 'anthropic/claude-3.5-sonnet',
            'temperature': 0.7,
            'max_tokens': 2000,
            'timeout': 30
        }
    }
    
    def mock_load_config(path):
        return mock_config
    
    monkeypatch.setattr('src.llm_client.load_config', mock_load_config)
    
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY not found"):
        OpenRouterClient()

@patch('src.llm_client.requests.post')
def test_generate_response_success(mock_post, llm_client):
    """Test successful API response"""
    # Mock successful response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        'choices': [{
            'message': {'content': 'Test response'},
            'finish_reason': 'stop'
        }],
        'usage': {'total_tokens': 100},
        'model': 'anthropic/claude-3.5-sonnet'
    }
    mock_post.return_value = mock_response
    
    messages = [{"role": "user", "content": "Test message"}]
    result = llm_client.generate_response(messages)
    
    assert result['content'] == 'Test response'
    assert result['finish_reason'] == 'stop'
    assert result['usage']['total_tokens'] == 100
    
    # Verify request was made correctly
    mock_post.assert_called_once()
    call_args = mock_post.call_args
    assert call_args[1]['json']['messages'] == messages

@patch('src.llm_client.requests.post')
def test_generate_response_with_custom_params(mock_post, llm_client):
    """Test API call with custom parameters"""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        'choices': [{
            'message': {'content': 'Response'},
            'finish_reason': 'stop'
        }],
        'usage': {},
        'model': 'test-model'
    }
    mock_post.return_value = mock_response
    
    messages = [{"role": "user", "content": "Test"}]
    result = llm_client.generate_response(
        messages,
        temperature=0.5,
        max_tokens=500,
        model='test-model'
    )
    
    call_args = mock_post.call_args
    payload = call_args[1]['json']
    
    assert payload['temperature'] == 0.5
    assert payload['max_tokens'] == 500
    assert payload['model'] == 'test-model'

@patch('src.llm_client.requests.post')
@patch('src.llm_client.time.sleep')
def test_retry_on_rate_limit(mock_sleep, mock_post, llm_client):
    """Test retry logic on rate limit"""
    # First call returns 429, second succeeds
    mock_response_429 = Mock()
    mock_response_429.status_code = 429
    
    mock_response_200 = Mock()
    mock_response_200.status_code = 200
    mock_response_200.json.return_value = {
        'choices': [{
            'message': {'content': 'Success'},
            'finish_reason': 'stop'
        }],
        'usage': {},
        'model': 'test-model'
    }
    
    mock_post.side_effect = [mock_response_429, mock_response_200]
    
    messages = [{"role": "user", "content": "Test"}]
    result = llm_client.generate_response(messages)
    
    assert result['content'] == 'Success'
    assert mock_post.call_count == 2
    mock_sleep.assert_called_once()

@patch('src.llm_client.requests.post')
def test_max_retries_exceeded(mock_post, llm_client):
    """Test exception after max retries"""
    mock_response = Mock()
    mock_response.status_code = 429
    mock_post.return_value = mock_response
    
    messages = [{"role": "user", "content": "Test"}]
    
    with pytest.raises(Exception, match="Rate limit exceeded"):
        llm_client.generate_response(messages)

def test_parse_json_response(llm_client):
    """Test JSON parsing from LLM response"""
    # Test with clean JSON
    json_str = '{"key": "value"}'
    result = llm_client.parse_json_response(json_str)
    assert result == {"key": "value"}
    
    # Test with markdown code block
    json_with_markdown = '```json\n{"key": "value"}\n```'
    result = llm_client.parse_json_response(json_with_markdown)
    assert result == {"key": "value"}
    
    # Test with plain code block
    json_with_block = '```\n{"key": "value"}\n```'
    result = llm_client.parse_json_response(json_with_block)
    assert result == {"key": "value"}

def test_parse_json_response_invalid(llm_client):
    """Test JSON parsing with invalid JSON"""
    invalid_json = "This is not JSON"
    
    with pytest.raises(ValueError, match="Invalid JSON response"):
        llm_client.parse_json_response(invalid_json)

def test_generate_meal_plans(llm_client):
    """Test meal plans generation wrapper"""
    with patch.object(llm_client, 'generate_response') as mock_generate:
        mock_generate.return_value = {
            'content': 'Meal plan content',
            'model': 'test-model',
            'usage': {},
            'finish_reason': 'stop'
        }
        
        result = llm_client.generate_meal_plans(
            "System prompt",
            "User prompt"
        )
        
        assert result == 'Meal plan content'
        mock_generate.assert_called_once()
        
        # Verify messages structure
        call_args = mock_generate.call_args[0][0]
        assert len(call_args) == 2
        assert call_args[0]['role'] == 'system'
        assert call_args[1]['role'] == 'user'
```

**Run Tests**:
```bash
pytest tests/test_llm_client.py -v
```

**Success Criteria**:
✅ Client initializes with API key
✅ API requests are properly formatted
✅ Retry logic works for rate limits
✅ JSON parsing handles various formats
✅ All tests pass

---

## PHASE 6: Meal Planner

### Task 6.1: Meal Planner Implementation

**Instructions**:

Create `src/meal_planner.py`:
```python
"""Meal planning logic combining RAG and LLM."""

import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.rag_engine import RAGEngine
from src.llm_client import OpenRouterClient
from src.utils import (
    load_prompts,
    parse_time_query,
    save_json,
    load_json,
    get_days_since,
    load_config
)

logger = logging.getLogger(__name__)

class MealPlanner:
    """Main meal planning orchestrator."""
    
    def __init__(
        self,
        config_path: str = "config/config.yaml",
        prompts_path: str = "config/prompts.yaml"
    ):
        """
        Initialize meal planner.
        
        Args:
            config_path: Path to configuration file
            prompts_path: Path to prompts file
        """
        logger.info("Initializing Meal Planner")
        
        self.config = load_config(config_path)
        self.prompts = load_prompts(prompts_path)
        
        self.rag_engine = RAGEngine(config_path)
        self.llm_client = OpenRouterClient(config_path)
        
        self.choices_log_file = self.config.get('choices', {}).get(
            'log_file',
            './data/logs/user_choices.json'
        )
    
    def generate_meal_plans(
        self,
        query: str,
        num_options: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Generate meal plan options based on query.
        
        Args:
            query: User query (e.g., "what should I shop for 3 days")
            num_options: Number of options to generate
            
        Returns:
            List of meal plan option dictionaries
        """
        logger.info(f"Generating meal plans for query: '{query}'")
        
        # Check if database has data
        stats = self.rag_engine.get_stats()
        if stats['total_documents'] == 0:
            logger.warning("No data in RAG database")
            return self._get_fallback_response()
        
        # Parse query to understand time frame and context
        query_params = parse_time_query(query)
        logger.info(f"Parsed query: {query_params}")
        
        # Retrieve relevant context from RAG
        context = self._get_relevant_context(query, query_params)
        
        # Generate meal plans using LLM
        meal_plans = self._generate_with_llm(
            query,
            context,
            query_params,
            num_options
        )
        
        # Enhance with household items
        meal_plans = self._add_household_items(meal_plans)
        
        return meal_plans
    
    def _get_relevant_context(
        self,
        query: str,
        query_params: Dict[str, Any]
    ) -> str:
        """
        Retrieve relevant context from RAG database.
        
        Args:
            query: User query
            query_params: Parsed query parameters
            
        Returns:
            Formatted context string
        """
        logger.info("Retrieving context from RAG database")
        
        # Get context from different sources
        contexts = []
        
        # 1. Get meal history
        meal_context = self.rag_engine.get_context(
            f"{query} meals history preferences",
            top_k=5,
            filter_metadata={'type': 'meal'}
        )
        if meal_context != "No relevant information found in the database.":
            contexts.append("=== MEAL HISTORY ===")
            contexts.append(meal_context)
        
        # 2. Get preferences
        pref_context = self.rag_engine.get_context(
            "user preferences dietary restrictions favorites",
            top_k=5,
            filter_metadata={'type': 'preference'}
        )
        if pref_context != "No relevant information found in the database.":
            contexts.append("=== USER PREFERENCES ===")
            contexts.append(pref_context)
        
        # 3. Get shopping history for budget context
        bill_context = self.rag_engine.get_context(
            "shopping bills grocery spending",
            top_k=3,
            filter_metadata={'type': 'bill'}
        )
        if bill_context != "No relevant information found in the database.":
            contexts.append("=== SHOPPING HISTORY ===")
            contexts.append(bill_context)
        
        # 4. Get past grocery lists for pattern recognition
        list_context = self.rag_engine.get_context(
            f"grocery lists {query_params.get('occasion', '')}",
            top_k=2,
            filter_metadata={'type': 'grocery_list'}
        )
        if list_context != "No relevant information found in the database.":
            contexts.append("=== PAST GROCERY LISTS ===")
            contexts.append(list_context)
        
        # 5. Get past user choices
        choice_context = self._get_choice_history_context()
        if choice_context:
            contexts.append("=== PAST CHOICES ===")
            contexts.append(choice_context)
        
        return "\n\n".join(contexts) if contexts else "No relevant context available."
    
    def _get_choice_history_context(self) -> str:
        """Get context from past user choices."""
        choices = load_json(self.choices_log_file)
        if not choices:
            return ""
        
        # Get last 5 choices
        recent_choices = choices[-5:] if len(choices) > 5 else choices
        
        context_parts = []
        for choice in recent_choices:
            context_parts.append(
                f"On {choice.get('timestamp', 'unknown date')}, "
                f"user chose: {choice.get('selected_option', {}).get('option_name', 'Unknown')}"
            )
        
        return "\n".join(context_parts)
    
    def _generate_with_llm(
        self,
        query: str,
        context: str,
        query_params: Dict[str, Any],
        num_options: int
    ) -> List[Dict[str, Any]]:
        """
        Generate meal plans using LLM.
        
        Args:
            query: User query
            context: Retrieved context
            query_params: Parsed query parameters
            num_options: Number of options to generate
            
        Returns:
            List of meal plan options
        """
        logger.info("Generating meal plans with LLM")
        
        # Prepare prompts
        system_prompt = self.prompts['system_prompt']
        
        user_prompt_template = self.prompts['meal_planning_prompt']
        user_prompt = user_prompt_template.format(
            context=context,
            query=query,
            num_days=query_params.get('num_days', 3)
        )
        
        # Generate response
        try:
            response_content = self.llm_client.generate_meal_plans(
                system_prompt,
                user_prompt
            )
            
            # Parse JSON response
            meal_plans = self.llm_client.parse_json_response(response_content)
            
            # Validate structure
            if not isinstance(meal_plans, list):
                logger.error("Response is not a list")
                return self._get_fallback_response()
            
            logger.info(f"Successfully generated {len(meal_plans)} meal plan options")
            return meal_plans[:num_options]
            
        except Exception as e:
            logger.error(f"Error generating meal plans: {str(e)}")
            return self._get_fallback_response()
    
    def _add_household_items(
        self,
        meal_plans: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Add household items to shopping lists based on purchase history.
        
        Args:
            meal_plans: List of meal plan options
            
        Returns:
            Meal plans with enhanced shopping lists
        """
        logger.info("Adding household items to shopping lists")
        
        # Get household item purchase history
        household_results = self.rag_engine.query(
            "household items",
            top_k=20,
            filter_metadata={'type': 'household_item'}
        )
        
        if not household_results:
            return meal_plans
        
        # Analyze restock needs
        restock_items = []
        restock_intervals = self.config.get('household_items', {}).get(
            'restock_interval_days',
            {}
        )
        
        # Track last purchase date for each item
        item_last_purchase = {}
        for result in household_results:
            item_name = result['metadata'].get('item_name', '').lower()
            date_str = result['metadata'].get('date', '')
            
            if item_name and date_str:
                days_since = get_days_since(date_str)
                
                if item_name not in item_last_purchase:
                    item_last_purchase[item_name] = days_since
                else:
                    item_last_purchase[item_name] = min(
                        item_last_purchase[item_name],
                        days_since
                    )
        
        # Determine what needs restocking
        default_interval = 30
        for item_name, days_since in item_last_purchase.items():
            # Guess category from item name
            category = self._guess_household_category(item_name)
            interval = restock_intervals.get(category, default_interval)
            
            if days_since >= interval * 0.8:  # Restock at 80% of interval
                restock_items.append({
                    'item': item_name,
                    'reason': f'Last purchased {days_since} days ago'
                })
        
        # Add to each meal plan's shopping list
        for plan in meal_plans:
            if 'shopping_list' not in plan:
                plan['shopping_list'] = {}
            
            if 'household' not in plan['shopping_list']:
                plan['shopping_list']['household'] = []
            
            # Add restock items (limit to 3-4 items per plan for variety)
            plan['shopping_list']['household'].extend(restock_items[:4])
        
        return meal_plans
    
    def _guess_household_category(self, item_name: str) -> str:
        """Guess household item category from name."""
        item_lower = item_name.lower()
        
        if any(word in item_lower for word in ['soap', 'detergent', 'cleaner', 'spray', 'wipe']):
            return 'cleaning_supplies'
        elif any(word in item_lower for word in ['paper', 'towel', 'tissue', 'napkin']):
            return 'paper_products'
        elif any(word in item_lower for word in ['shampoo', 'toothpaste', 'deodorant']):
            return 'personal_care'

        ```python
        else:
            return 'kitchen_supplies'
    
    def _get_fallback_response(self) -> List[Dict[str, Any]]:
        """
        Get fallback response when generation fails or no data available.
        
        Returns:
            List with single fallback option
        """
        logger.warning("Using fallback response")
        
        return [{
            'option_name': 'No Personalized Suggestions Available',
            'description': self.prompts.get('fallback_response', 
                'Please ingest some data first using: meal-assistant ingest --all ./data/raw'),
            'meals': [],
            'shopping_list': {},
            'estimated_cost': 'N/A',
            'prep_difficulty': 'N/A'
        }]
    
    def log_user_choice(
        self,
        query: str,
        options: List[Dict[str, Any]],
        selected_index: int
    ) -> None:
        """
        Log user's choice for future learning.
        
        Args:
            query: Original query
            options: All options presented
            selected_index: Index of selected option (0-based)
        """
        logger.info(f"Logging user choice: option {selected_index + 1}")
        
        # Load existing choices
        choices = load_json(self.choices_log_file)
        if choices is None:
            choices = []
        
        # Create choice record
        choice_record = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'selected_index': selected_index,
            'selected_option': options[selected_index],
            'all_options': options
        }
        
        choices.append(choice_record)
        
        # Save updated choices
        save_json(choices, self.choices_log_file)
        
        logger.info(f"Choice logged to {self.choices_log_file}")
        
        # Add choice to RAG for future context
        self._add_choice_to_rag(choice_record)
    
    def _add_choice_to_rag(self, choice_record: Dict[str, Any]) -> None:
        """
        Add user choice to RAG database for future context.
        
        Args:
            choice_record: Choice record dictionary
        """
        selected = choice_record['selected_option']
        
        # Format as document
        doc_parts = [
            f"User Choice on {choice_record['timestamp']}",
            f"Query: {choice_record['query']}",
            f"Selected Plan: {selected.get('option_name', 'Unknown')}",
            f"Description: {selected.get('description', '')}"
        ]
        
        # Add meals if available
        meals = selected.get('meals', [])
        if meals:
            meal_names = [m.get('meal_name', 'Unknown') for m in meals]
            doc_parts.append(f"Meals: {', '.join(meal_names)}")
        
        document = '\n'.join(doc_parts)
        
        metadata = {
            'type': 'user_choice',
            'timestamp': choice_record['timestamp'],
            'query': choice_record['query']
        }
        
        self.rag_engine.add_documents([document], [metadata])
        logger.info("Added choice to RAG database")
    
    def get_choice_history(
        self,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get user choice history.
        
        Args:
            limit: Maximum number of choices to return (most recent)
            
        Returns:
            List of choice records
        """
        choices = load_json(self.choices_log_file)
        if not choices:
            return []
        
        if limit:
            return choices[-limit:]
        
        return choices
```

**Test Requirements**:

Create `tests/test_meal_planner.py`:
```python
import pytest
import os
import json
from unittest.mock import Mock, patch, MagicMock
from src.meal_planner import MealPlanner

@pytest.fixture
def mock_rag_engine():
    """Mock RAG engine"""
    mock = Mock()
    mock.get_stats.return_value = {
        'total_documents': 10,
        'by_type': {'meal': 5, 'preference': 3, 'bill': 2}
    }
    mock.get_context.return_value = "Mock context from RAG"
    mock.query.return_value = []
    mock.add_documents.return_value = None
    return mock

@pytest.fixture
def mock_llm_client():
    """Mock LLM client"""
    mock = Mock()
    mock.generate_meal_plans.return_value = json.dumps([
        {
            'option_name': 'Test Plan',
            'description': 'Test description',
            'meals': [
                {
                    'day': 'Monday',
                    'meal_name': 'Test Meal',
                    'description': 'Test meal description',
                    'main_ingredients': ['ingredient1', 'ingredient2']
                }
            ],
            'shopping_list': {
                'produce': [
                    {'item': 'tomatoes', 'quantity': '5', 'unit': 'medium'}
                ]
            },
            'estimated_cost': '50-60',
            'prep_difficulty': 'easy'
        }
    ])
    mock.parse_json_response.return_value = [
        {
            'option_name': 'Test Plan',
            'description': 'Test description',
            'meals': [],
            'shopping_list': {},
            'estimated_cost': '50-60',
            'prep_difficulty': 'easy'
        }
    ]
    return mock

@pytest.fixture
def meal_planner(monkeypatch, mock_rag_engine, mock_llm_client):
    """Create meal planner with mocked dependencies"""
    
    mock_config = {
        'choices': {'log_file': './data/logs/test_choices.json'},
        'household_items': {
            'restock_interval_days': {
                'cleaning_supplies': 30,
                'paper_products': 14
            }
        }
    }
    
    mock_prompts = {
        'system_prompt': 'Test system prompt',
        'meal_planning_prompt': 'Context: {context}\nQuery: {query}\nDays: {num_days}',
        'fallback_response': 'No data available'
    }
    
    def mock_load_config(path):
        return mock_config
    
    def mock_load_prompts(path):
        return mock_prompts
    
    monkeypatch.setattr('src.meal_planner.load_config', mock_load_config)
    monkeypatch.setattr('src.meal_planner.load_prompts', mock_load_prompts)
    monkeypatch.setattr('src.meal_planner.RAGEngine', lambda x: mock_rag_engine)
    monkeypatch.setattr('src.meal_planner.OpenRouterClient', lambda x: mock_llm_client)
    
    planner = MealPlanner()
    return planner

def test_meal_planner_initialization(meal_planner):
    """Test meal planner initializes correctly"""
    assert meal_planner.rag_engine is not None
    assert meal_planner.llm_client is not None
    assert meal_planner.config is not None
    assert meal_planner.prompts is not None

def test_generate_meal_plans_success(meal_planner, mock_llm_client):
    """Test successful meal plan generation"""
    plans = meal_planner.generate_meal_plans("what should I shop for 3 days")
    
    assert len(plans) > 0
    assert 'option_name' in plans[0]
    assert 'shopping_list' in plans[0]

def test_generate_meal_plans_empty_database(meal_planner, mock_rag_engine):
    """Test meal plan generation with empty database"""
    mock_rag_engine.get_stats.return_value = {'total_documents': 0}
    
    plans = meal_planner.generate_meal_plans("test query")
    
    assert len(plans) > 0
    assert 'No Personalized Suggestions' in plans[0]['option_name']

def test_get_relevant_context(meal_planner, mock_rag_engine):
    """Test context retrieval"""
    query_params = {'num_days': 3, 'occasion': 'general'}
    
    context = meal_planner._get_relevant_context("test query", query_params)
    
    assert isinstance(context, str)
    assert len(context) > 0
    # Should call rag_engine.get_context multiple times
    assert mock_rag_engine.get_context.call_count >= 3

def test_add_household_items(meal_planner, mock_rag_engine):
    """Test adding household items to shopping lists"""
    # Mock household item results
    mock_rag_engine.query.return_value = [
        {
            'document': 'Dish soap purchased',
            'metadata': {
                'type': 'household_item',
                'item_name': 'dish soap',
                'date': '2024-01-01'
            }
        }
    ]
    
    meal_plans = [
        {
            'option_name': 'Test',
            'shopping_list': {
                'produce': []
            }
        }
    ]
    
    enhanced_plans = meal_planner._add_household_items(meal_plans)
    
    assert 'household' in enhanced_plans[0]['shopping_list']
    # Should have added household items based on restock logic

def test_guess_household_category(meal_planner):
    """Test household item category guessing"""
    assert meal_planner._guess_household_category('dish soap') == 'cleaning_supplies'
    assert meal_planner._guess_household_category('paper towels') == 'paper_products'
    assert meal_planner._guess_household_category('shampoo') == 'personal_care'
    assert meal_planner._guess_household_category('aluminum foil') == 'kitchen_supplies'

def test_log_user_choice(meal_planner, tmp_path, monkeypatch):
    """Test logging user choice"""
    # Use temporary file for testing
    temp_log = tmp_path / "test_choices.json"
    meal_planner.choices_log_file = str(temp_log)
    
    query = "test query"
    options = [
        {'option_name': 'Option 1', 'meals': []},
        {'option_name': 'Option 2', 'meals': []}
    ]
    selected_index = 1
    
    meal_planner.log_user_choice(query, options, selected_index)
    
    # Verify file was created and contains data
    assert temp_log.exists()
    
    with open(temp_log, 'r') as f:
        choices = json.load(f)
    
    assert len(choices) == 1
    assert choices[0]['query'] == query
    assert choices[0]['selected_index'] == 1
    assert choices[0]['selected_option']['option_name'] == 'Option 2'

def test_get_choice_history(meal_planner, tmp_path):
    """Test retrieving choice history"""
    # Create test choice log
    temp_log = tmp_path / "test_choices.json"
    meal_planner.choices_log_file = str(temp_log)
    
    test_choices = [
        {'timestamp': '2024-01-01', 'query': 'query1'},
        {'timestamp': '2024-01-02', 'query': 'query2'},
        {'timestamp': '2024-01-03', 'query': 'query3'}
    ]
    
    with open(temp_log, 'w') as f:
        json.dump(test_choices, f)
    
    # Get all history
    history = meal_planner.get_choice_history()
    assert len(history) == 3
    
    # Get limited history
    history = meal_planner.get_choice_history(limit=2)
    assert len(history) == 2
    assert history[0]['query'] == 'query2'

def test_get_choice_history_empty(meal_planner, tmp_path):
    """Test retrieving empty choice history"""
    temp_log = tmp_path / "nonexistent.json"
    meal_planner.choices_log_file = str(temp_log)
    
    history = meal_planner.get_choice_history()
    assert history == []

def test_fallback_response(meal_planner):
    """Test fallback response structure"""
    fallback = meal_planner._get_fallback_response()
    
    assert len(fallback) == 1
    assert 'option_name' in fallback[0]
    assert 'description' in fallback[0]
    assert fallback[0]['meals'] == []

def test_add_choice_to_rag(meal_planner, mock_rag_engine):
    """Test adding choice to RAG database"""
    choice_record = {
        'timestamp': '2024-01-01T12:00:00',
        'query': 'test query',
        'selected_option': {
            'option_name': 'Test Plan',
            'description': 'Test description',
            'meals': [
                {'meal_name': 'Test Meal'}
            ]
        }
    }
    
    meal_planner._add_choice_to_rag(choice_record)
    
    # Verify RAG engine add_documents was called
    mock_rag_engine.add_documents.assert_called_once()
    
    # Check the document content
    call_args = mock_rag_engine.add_documents.call_args
    documents = call_args[0][0]
    metadata = call_args[0][1]
    
    assert len(documents) == 1
    assert 'Test Plan' in documents[0]
    assert metadata[0]['type'] == 'user_choice'
```

**Run Tests**:
```bash
pytest tests/test_meal_planner.py -v
```

**Success Criteria**:
✅ Meal planner generates options successfully
✅ Context retrieval works correctly
✅ Household items are added based on history
✅ User choices are logged and added to RAG
✅ Fallback works when no data available
✅ All tests pass

---

## PHASE 7: CLI Interface

### Task 7.1: CLI Implementation

**Instructions**:

Create `src/cli.py`:
```python
"""Command-line interface for the meal shopping assistant."""

import os
import sys
import logging
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from rich import box

from src.meal_planner import MealPlanner
from src.rag_engine import RAGEngine
from src.data_ingestion import DataIngestion
from src.utils import setup_logging, format_shopping_list, ensure_dir

console = Console()
logger = None

def init_logging(verbose: bool = False):
    """Initialize logging."""
    global logger
    logger = setup_logging()
    if verbose:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)

@click.group()
@click.option('--verbose', is_flag=True, help='Enable verbose logging')
def main(verbose):
    """
    Meal Shopping Assistant - AI-powered meal planning and shopping lists.
    """
    init_logging(verbose)
    
    # Display banner
    banner = """
    ╔═══════════════════════════════════════════════════╗
    ║   🍽️  Meal Shopping Assistant  🛒                 ║
    ║   AI-Powered Meal Planning & Shopping Lists      ║
    ╚═══════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold cyan")

@main.command()
@click.option('--meals', type=click.Path(exists=True), help='Path to meals JSON file')
@click.option('--preferences', type=click.Path(exists=True), help='Path to preferences JSON file')
@click.option('--bills', type=click.Path(exists=True), help='Path to bills JSON file')
@click.option('--grocery-lists', type=click.Path(exists=True), help='Path to grocery lists JSON file')
@click.option('--all', 'all_dir', type=click.Path(exists=True), help='Directory containing all data files')
def ingest(meals, preferences, bills, grocery_lists, all_dir):
    """
    Ingest data into the RAG database.
    
    Examples:
    
      meal-assistant ingest --meals data/raw/meals.json
      
      meal-assistant ingest --all data/raw
    """
    try:
        console.print("\n[bold blue]Starting data ingestion...[/bold blue]\n")
        
        rag_engine = RAGEngine()
        data_ingestion = DataIngestion()
        
        total_count = 0
        
        if all_dir:
            # Batch ingest all files from directory
            with console.status("[bold green]Processing all files..."):
                counts = data_ingestion.ingest_all_data(all_dir, rag_engine)
            
            # Display results
            table = Table(title="Ingestion Results", box=box.ROUNDED)
            table.add_column("Data Type", style="cyan")
            table.add_column("Count", style="green", justify="right")
            
            for data_type, count in counts.items():
                table.add_row(data_type.replace('_', ' ').title(), str(count))
                total_count += count
            
            console.print(table)
        
        else:
            # Individual file ingestion
            if meals:
                with console.status("[bold green]Loading meals..."):
                    docs, meta = data_ingestion.load_meals(meals)
                    if docs:
                        rag_engine.add_documents(docs, meta)
                        total_count += len(docs)
                        console.print(f"✓ Loaded {len(docs)} meals", style="green")
            
            if preferences:
                with console.status("[bold green]Loading preferences..."):
                    docs, meta = data_ingestion.load_preferences(preferences)
                    if docs:
                        rag_engine.add_documents(docs, meta)
                        total_count += len(docs)
                        console.print(f"✓ Loaded {len(docs)} preference items", style="green")
            
            if bills:
                with console.status("[bold green]Loading bills..."):
                    docs, meta = data_ingestion.load_bills(bills)
                    if docs:
                        rag_engine.add_documents(docs, meta)
                        total_count += len(docs)
                        console.print(f"✓ Loaded {len(docs)} bill items", style="green")
            
            if grocery_lists:
                with console.status("[bold green]Loading grocery lists..."):
                    docs, meta = data_ingestion.load_grocery_lists(grocery_lists)
                    if docs:
                        rag_engine.add_documents(docs, meta)
                        total_count += len(docs)
                        console.print(f"✓ Loaded {len(docs)} grocery lists", style="green")
        
        # Display final stats
        stats = rag_engine.get_stats()
        console.print(f"\n[bold green]✓ Successfully ingested {total_count} items![/bold green]")
        console.print(f"[dim]Total documents in database: {stats['total_documents']}[/dim]\n")
        
    except Exception as e:
        console.print(f"\n[bold red]✗ Error during ingestion: {str(e)}[/bold red]\n")
        if logger:
            logger.exception("Ingestion failed")
        sys.exit(1)

@main.command()
@click.argument('query', nargs=-1, required=True)
@click.option('--num-options', default=3, help='Number of options to generate (default: 3)')
@click.option('--export', type=click.Path(), help='Export selected plan to file')
def plan(query, num_options, export):
    """
    Generate meal plans and shopping lists.
    
    Examples:
    
      meal-assistant plan "what should I shop for 3 days"
      
      meal-assistant plan "weekend party for 10 people"
      
      meal-assistant plan "quick weeknight dinners" --num-options 5
    """
    try:
        query_str = ' '.join(query)
        console.print(f"\n[bold blue]Query:[/bold blue] {query_str}\n")
        
        # Generate meal plans
        meal_planner = MealPlanner()
        
        with console.status("[bold green]Generating meal plan options..."):
            plans = meal_planner.generate_meal_plans(query_str, num_options)
        
        if not plans or 'No Personalized Suggestions' in plans[0].get('option_name', ''):
            console.print(plans[0].get('description', 'No suggestions available'), style="yellow")
            sys.exit(0)
        
        # Display options
        display_meal_plans(plans)
        
        # Ask user to select
        console.print("\n")
        choice = click.prompt(
            "Select an option (1-{}) or 'q' to quit".format(len(plans)),
            type=str
        )
        
        if choice.lower() == 'q':
            console.print("Cancelled.", style="yellow")
            sys.exit(0)
        
        try:
            selected_index = int(choice) - 1
            if selected_index < 0 or selected_index >= len(plans):
                raise ValueError()
        except ValueError:
            console.print("Invalid choice.", style="red")
            sys.exit(1)
        
        # Log choice
        meal_planner.log_user_choice(query_str, plans, selected_index)
        
        selected_plan = plans[selected_index]
        console.print(f"\n[bold green]✓ You selected: {selected_plan['option_name']}[/bold green]")
        console.print("[dim]This choice has been logged for future personalization.[/dim]\n")
        
        # Export if requested
        if export:
            export_meal_plan(selected_plan, export)
            console.print(f"[green]✓ Plan exported to {export}[/green]\n")
        
    except KeyboardInterrupt:
        console.print("\n\nCancelled.", style="yellow")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[bold red]✗ Error: {str(e)}[/bold red]\n")
        if logger:
            logger.exception("Plan generation failed")
        sys.exit(1)

@main.command()
@click.option('--last', default=10, help='Show last N choices (default: 10)')
def history(last):
    """
    Show history of past meal plan choices.
    
    Examples:
    
      meal-assistant history
      
      meal-assistant history --last 5
    """
    try:
        meal_planner = MealPlanner()
        choices = meal_planner.get_choice_history(limit=last)
        
        if not choices:
            console.print("[yellow]No choice history found.[/yellow]\n")
            sys.exit(0)
        
        console.print(f"\n[bold blue]Last {len(choices)} Choices:[/bold blue]\n")
        
        for i, choice in enumerate(reversed(choices), 1):
            timestamp = choice.get('timestamp', 'Unknown time')
            query = choice.get('query', 'Unknown query')
            selected = choice.get('selected_option', {})
            option_name = selected.get('option_name', 'Unknown')
            
            panel_content = f"""
[bold]Query:[/bold] {query}
[bold]Selected:[/bold] {option_name}
[bold]Time:[/bold] {timestamp}
            """.strip()
            
            console.print(Panel(panel_content, title=f"Choice #{len(choices) - i + 1}", border_style="cyan"))
        
        console.print()
        
    except Exception as e:
        console.print(f"\n[bold red]✗ Error: {str(e)}[/bold red]\n")
        if logger:
            logger.exception("History retrieval failed")
        sys.exit(1)

@main.command()
@click.confirmation_option(prompt='Are you sure you want to reset the database? This cannot be undone.')
def reset():
    """
    Reset the RAG database (clear all data).
    
    WARNING: This will delete all ingested data!
    """
    try:
        console.print("\n[bold yellow]Resetting database...[/bold yellow]\n")
        
        rag_engine = RAGEngine()
        rag_engine.reset_database()
        
        console.print("[bold green]✓ Database reset successfully![/bold green]\n")
        
    except Exception as e:
        console.print(f"\n[bold red]✗ Error: {str(e)}[/bold red]\n")
        if logger:
            logger.exception("Reset failed")
        sys.exit(1)

@main.command()
def stats():
    """Show database statistics."""
    try:
        rag_engine = RAGEngine()
        stats = rag_engine.get_stats()
        
        console.print("\n[bold blue]Database Statistics:[/bold blue]\n")
        
        table = Table(box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")
        
        table.add_row("Total Documents", str(stats['total_documents']))
        table.add_row("Index Size", str(stats['index_size']))
        
        console.print(table)
        
        if stats['by_type']:
            console.print("\n[bold]Documents by Type:[/bold]\n")
            
            type_table = Table(box=box.ROUNDED)
            type_table.add_column("Type", style="cyan")
            type_table.add_column("Count", style="green", justify="right")
            
            for doc_type, count in stats['by_type'].items():
                type_table.add_row(doc_type.replace('_', ' ').title(), str(count))
            
            console.print(type_table)
        
        console.print()
        
    except Exception as e:
        console.print(f"\n[bold red]✗ Error: {str(e)}[/bold red]\n")
        if logger:
            logger.exception("Stats retrieval failed")
        sys.exit(1)

def display_meal_plans(plans):
    """Display meal plans in a nice format."""
    for i, plan in enumerate(plans, 1):
        # Create option header
        console.print(f"\n[bold cyan]Option {i}: {plan.get('option_name', 'Unknown')}[/bold cyan]")
        console.print(f"[dim]{plan.get('description', '')}[/dim]")
        console.print(f"[dim]Difficulty: {plan.get('prep_difficulty', 'N/A')} | "
                     f"Estimated Cost: ${plan.get('estimated_cost', 'N/A')}[/dim]\n")
        
        # Display meals
        meals = plan.get('meals', [])
        if meals:
            console.print("[bold]📅 Meals:[/bold]")
            for meal in meals:
                day = meal.get('day', 'Day')
                meal_name = meal.get('meal_name', 'Unknown')
                description = meal.get('description', '')
                console.print(f"  • [green]{day}[/green]: {meal_name}")
                if description:
                    console.print(f"    [dim]{description}[/dim]")
            console.print()
        
        # Display shopping list
        shopping_list = plan.get('shopping_list', {})
        if shopping_list:
            console.print("[bold]🛒 Shopping List:[/bold]")
            formatted_list = format_shopping_list(shopping_list)
            console.print(formatted_list)
        
        # Separator between options
        if i < len(plans):
            console.print("\n" + "─" * 60)

def export_meal_plan(plan, filepath):
    """Export meal plan to a file."""
    import json
    ensure_dir(os.path.dirname(filepath))
    
    with open(filepath, 'w') as f:
        json.dump(plan, f, indent=2)

if __name__ == '__main__':
    main()
```

**Test Requirements**:

Create `tests/test_cli.py`:
```python
import pytest
import json
import os
from click.testing import CliRunner
from unittest.mock import Mock, patch, MagicMock
from src.cli import main, display_meal_plans, export_meal_plan

@pytest.fixture
def cli_runner():
    """Create CLI runner"""
    return CliRunner()

@pytest.fixture
def sample_meal_plans():
    """Sample meal plans for testing"""
    return [
        {
            'option_name': 'Mediterranean Week',
            'description': 'Healthy Mediterranean meals',
            'meals': [
                {
                    'day': 'Monday',
                    'meal_name': 'Greek Salad',
                    'description': 'Fresh and healthy'
                }
            ],
            'shopping_list': {
                'produce': [
                    {'item': 'tomatoes', 'quantity': '5', 'unit': 'medium'}
                ]
            },
            'estimated_cost': '50-60',
            'prep_difficulty': 'easy'
        }
    ]

@patch('src.cli.RAGEngine')
@patch('src.cli.DataIngestion')
def test_ingest_all(mock_data_ingestion, mock_rag_engine, cli_runner, tmp_path):
    """Test ingest command with --all flag"""
    # Setup mocks
    mock_rag_instance = Mock()
    mock_rag_engine.return_value = mock_rag_instance
    mock_rag_instance.get_stats.return_value = {'total_documents': 10}
    
    mock_ingestion_instance = Mock()
    mock_data_ingestion.return_value = mock_ingestion_instance
    mock_ingestion_instance.ingest_all_data.return_value = {
        'meals': 5,
        'preferences': 3,
        'bills': 2,
        'grocery_lists': 0
    }
    
    # Create test directory
    test_dir = tmp_path / "data"
    test_dir.mkdir()
    
    result = cli_runner.invoke(main, ['ingest', '--all', str(test_dir)])
    
    assert result.exit_code == 0
    assert 'Successfully ingested' in result.output

@patch('src.cli.MealPlanner')
def test_plan_command(mock_meal_planner, cli_runner, sample_meal_plans):
    """Test plan command"""
    # Setup mock
    mock_planner_instance = Mock()
    mock_meal_planner.return_value = mock_planner_instance
    mock_planner_instance.generate_meal_plans.return_value = sample_meal_plans
    
    # Simulate user selecting option 1
    result = cli_runner.invoke(
        main,
        ['plan', 'what should I shop for 3 days'],
        input='1\n'
    )
    
    assert result.exit_code == 0
    assert 'Mediterranean Week' in result.output
    assert 'Greek Salad' in result.output

@patch('src.cli.MealPlanner')
def test_plan_command_quit(mock_meal_planner, cli_runner, sample_meal
```python
_plans):
    """Test plan command with quit option"""
    # Setup mock
    mock_planner_instance = Mock()
    mock_meal_planner.return_value = mock_planner_instance
    mock_planner_instance.generate_meal_plans.return_value = sample_meal_plans
    
    # Simulate user quitting
    result = cli_runner.invoke(
        main,
        ['plan', 'weekend meals'],
        input='q\n'
    )
    
    assert result.exit_code == 0
    assert 'Cancelled' in result.output

@patch('src.cli.MealPlanner')
def test_history_command(mock_meal_planner, cli_runner):
    """Test history command"""
    # Setup mock
    mock_planner_instance = Mock()
    mock_meal_planner.return_value = mock_planner_instance
    mock_planner_instance.get_choice_history.return_value = [
        {
            'timestamp': '2024-01-15T12:00:00',
            'query': 'test query',
            'selected_option': {'option_name': 'Test Plan'}
        }
    ]
    
    result = cli_runner.invoke(main, ['history'])
    
    assert result.exit_code == 0
    assert 'Test Plan' in result.output

@patch('src.cli.MealPlanner')
def test_history_command_empty(mock_meal_planner, cli_runner):
    """Test history command with no history"""
    # Setup mock
    mock_planner_instance = Mock()
    mock_meal_planner.return_value = mock_planner_instance
    mock_planner_instance.get_choice_history.return_value = []
    
    result = cli_runner.invoke(main, ['history'])
    
    assert result.exit_code == 0
    assert 'No choice history' in result.output

@patch('src.cli.RAGEngine')
def test_reset_command(mock_rag_engine, cli_runner):
    """Test reset command"""
    # Setup mock
    mock_rag_instance = Mock()
    mock_rag_engine.return_value = mock_rag_instance
    
    result = cli_runner.invoke(main, ['reset'], input='y\n')
    
    assert result.exit_code == 0
    assert 'reset successfully' in result.output
    mock_rag_instance.reset_database.assert_called_once()

@patch('src.cli.RAGEngine')
def test_stats_command(mock_rag_engine, cli_runner):
    """Test stats command"""
    # Setup mock
    mock_rag_instance = Mock()
    mock_rag_engine.return_value = mock_rag_instance
    mock_rag_instance.get_stats.return_value = {
        'total_documents': 25,
        'by_type': {
            'meal': 10,
            'preference': 5,
            'bill': 10
        },
        'index_size': 25
    }
    
    result = cli_runner.invoke(main, ['stats'])
    
    assert result.exit_code == 0
    assert '25' in result.output
    assert 'meal' in result.output.lower()

def test_export_meal_plan(tmp_path, sample_meal_plans):
    """Test exporting meal plan to file"""
    export_path = tmp_path / "exported_plan.json"
    
    export_meal_plan(sample_meal_plans[0], str(export_path))
    
    assert export_path.exists()
    
    with open(export_path, 'r') as f:
        exported = json.load(f)
    
    assert exported['option_name'] == 'Mediterranean Week'

@patch('src.cli.MealPlanner')
def test_plan_with_export(mock_meal_planner, cli_runner, sample_meal_plans, tmp_path):
    """Test plan command with export flag"""
    # Setup mock
    mock_planner_instance = Mock()
    mock_meal_planner.return_value = mock_planner_instance
    mock_planner_instance.generate_meal_plans.return_value = sample_meal_plans
    
    export_path = tmp_path / "export.json"
    
    result = cli_runner.invoke(
        main,
        ['plan', 'test query', '--export', str(export_path)],
        input='1\n'
    )
    
    assert result.exit_code == 0
    assert export_path.exists()

def test_display_meal_plans(sample_meal_plans):
    """Test meal plans display function"""
    # This is mainly a visual function, just ensure it doesn't crash
    try:
        display_meal_plans(sample_meal_plans)
    except Exception as e:
        pytest.fail(f"display_meal_plans raised an exception: {e}")

@patch('src.cli.RAGEngine')
@patch('src.cli.DataIngestion')
def test_ingest_individual_files(mock_data_ingestion, mock_rag_engine, cli_runner, tmp_path):
    """Test ingesting individual files"""
    # Setup mocks
    mock_rag_instance = Mock()
    mock_rag_engine.return_value = mock_rag_instance
    mock_rag_instance.get_stats.return_value = {'total_documents': 5}
    
    mock_ingestion_instance = Mock()
    mock_data_ingestion.return_value = mock_ingestion_instance
    mock_ingestion_instance.load_meals.return_value = (['doc1'], [{'type': 'meal'}])
    
    # Create test file
    test_file = tmp_path / "meals.json"
    test_file.write_text('[]')
    
    result = cli_runner.invoke(main, ['ingest', '--meals', str(test_file)])
    
    assert result.exit_code == 0

@patch('src.cli.MealPlanner')
def test_plan_no_data(mock_meal_planner, cli_runner):
    """Test plan command with no data in database"""
    # Setup mock to return fallback response
    mock_planner_instance = Mock()
    mock_meal_planner.return_value = mock_planner_instance
    mock_planner_instance.generate_meal_plans.return_value = [
        {
            'option_name': 'No Personalized Suggestions Available',
            'description': 'Please ingest data first',
            'meals': [],
            'shopping_list': {}
        }
    ]
    
    result = cli_runner.invoke(main, ['plan', 'test query'])
    
    assert result.exit_code == 0
    assert 'ingest data' in result.output.lower()

def test_main_help(cli_runner):
    """Test main command help"""
    result = cli_runner.invoke(main, ['--help'])
    
    assert result.exit_code == 0
    assert 'Meal Shopping Assistant' in result.output

def test_ingest_help(cli_runner):
    """Test ingest command help"""
    result = cli_runner.invoke(main, ['ingest', '--help'])
    
    assert result.exit_code == 0
    assert 'ingest' in result.output.lower()

def test_plan_help(cli_runner):
    """Test plan command help"""
    result = cli_runner.invoke(main, ['plan', '--help'])
    
    assert result.exit_code == 0
    assert 'meal plan' in result.output.lower()
```

**Run Tests**:
```bash
pytest tests/test_cli.py -v
```

**Success Criteria**:
✅ CLI commands work correctly
✅ User can ingest data
✅ User can generate and select meal plans
✅ User can view history
✅ Database can be reset
✅ Stats display correctly
✅ Export functionality works
✅ All tests pass

---

## PHASE 8: Integration & End-to-End Testing

### Task 8.1: Integration Tests

**Instructions**:

Create `tests/test_integration.py`:
```python
"""Integration tests for the complete workflow."""

import pytest
import os
import shutil
from src.rag_engine import RAGEngine
from src.data_ingestion import DataIngestion
from src.meal_planner import MealPlanner
from src.llm_client import OpenRouterClient

@pytest.fixture
def integration_env(tmp_path, monkeypatch):
    """Setup complete integration environment"""
    # Set up temporary directories
    vector_db_path = tmp_path / 'vector_db'
    logs_path = tmp_path / 'logs'
    
    vector_db_path.mkdir()
    logs_path.mkdir()
    
    # Mock config
    config = {
        'rag': {
            'vector_db_path': str(vector_db_path),
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'chunk_size': 500,
            'chunk_overlap': 50,
            'top_k': 5
        },
        'llm': {
            'api_base_url': 'https://openrouter.ai/api/v1/chat/completions',
            'default_model': 'anthropic/claude-3.5-sonnet',
            'temperature': 0.7,
            'max_tokens': 2000,
            'timeout': 30
        },
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file': str(logs_path / 'app.log')
        },
        'choices': {
            'log_file': str(logs_path / 'choices.json')
        },
        'household_items': {
            'restock_interval_days': {
                'cleaning_supplies': 30,
                'paper_products': 14
            }
        }
    }
    
    prompts = {
        'system_prompt': 'You are a helpful meal planning assistant.',
        'meal_planning_prompt': 'Context: {context}\nQuery: {query}\nDays: {num_days}',
        'fallback_response': 'No data available. Please ingest data first.'
    }
    
    def mock_load_config(path):
        return config
    
    def mock_load_prompts(path):
        return prompts
    
    monkeypatch.setattr('src.rag_engine.load_config', mock_load_config)
    monkeypatch.setattr('src.data_ingestion.load_config', mock_load_config)
    monkeypatch.setattr('src.meal_planner.load_config', mock_load_config)
    monkeypatch.setattr('src.meal_planner.load_prompts', mock_load_prompts)
    
    # Set API key
    monkeypatch.setenv('OPENROUTER_API_KEY', 'test-key')
    
    yield {
        'vector_db_path': vector_db_path,
        'logs_path': logs_path,
        'config': config,
        'prompts': prompts
    }
    
    # Cleanup
    if vector_db_path.exists():
        shutil.rmtree(vector_db_path)
    if logs_path.exists():
        shutil.rmtree(logs_path)

@pytest.mark.integration
def test_complete_workflow(integration_env, fixtures_dir, monkeypatch):
    """Test complete workflow: ingest -> query -> select -> log"""
    
    # Mock LLM client to avoid actual API calls
    from unittest.mock import Mock, patch
    import json
    
    mock_llm_response = json.dumps([
        {
            'option_name': 'Integration Test Plan',
            'description': 'A test meal plan',
            'meals': [
                {
                    'day': 'Monday',
                    'meal_name': 'Test Meal',
                    'description': 'Test description',
                    'main_ingredients': ['ingredient1', 'ingredient2']
                }
            ],
            'shopping_list': {
                'produce': [
                    {'item': 'tomatoes', 'quantity': '5', 'unit': 'medium'}
                ],
                'household': []
            },
            'estimated_cost': '50-60',
            'prep_difficulty': 'easy'
        }
    ])
    
    with patch('src.llm_client.requests.post') as mock_post:
        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{
                'message': {'content': mock_llm_response},
                'finish_reason': 'stop'
            }],
            'usage': {'total_tokens': 100},
            'model': 'test-model'
        }
        mock_post.return_value = mock_response
        
        # Step 1: Initialize components
        rag_engine = RAGEngine()
        data_ingestion = DataIngestion()
        
        # Step 2: Ingest data
        counts = data_ingestion.ingest_all_data(fixtures_dir, rag_engine)
        
        assert counts['meals'] > 0
        assert counts['preferences'] > 0
        
        # Verify data in RAG
        stats = rag_engine.get_stats()
        assert stats['total_documents'] > 0
        
        # Step 3: Generate meal plans
        meal_planner = MealPlanner()
        plans = meal_planner.generate_meal_plans("what should I shop for 3 days")
        
        assert len(plans) > 0
        assert 'option_name' in plans[0]
        assert 'shopping_list' in plans[0]
        
        # Step 4: Log user choice
        meal_planner.log_user_choice("test query", plans, 0)
        
        # Step 5: Verify choice was logged
        history = meal_planner.get_choice_history()
        assert len(history) > 0
        assert history[-1]['query'] == "test query"
        
        # Step 6: Verify choice was added to RAG
        choice_results = rag_engine.query("user choice", top_k=1)
        assert len(choice_results) > 0

@pytest.mark.integration
def test_data_persistence(integration_env, fixtures_dir, monkeypatch):
    """Test that data persists between sessions"""
    
    # Session 1: Ingest data
    rag_engine1 = RAGEngine()
    data_ingestion = DataIngestion()
    
    counts = data_ingestion.ingest_all_data(fixtures_dir, rag_engine1)
    total_docs = sum(counts.values())
    
    stats1 = rag_engine1.get_stats()
    assert stats1['total_documents'] == total_docs
    
    # Session 2: Load existing data
    rag_engine2 = RAGEngine()
    stats2 = rag_engine2.get_stats()
    
    assert stats2['total_documents'] == stats1['total_documents']
    assert stats2['by_type'] == stats1['by_type']

@pytest.mark.integration
def test_rag_retrieval_quality(integration_env, fixtures_dir):
    """Test that RAG retrieves relevant documents"""
    
    # Ingest data
    rag_engine = RAGEngine()
    data_ingestion = DataIngestion()
    data_ingestion.ingest_all_data(fixtures_dir, rag_engine)
    
    # Query for Italian food
    results = rag_engine.query("Italian pasta meals", top_k=3)
    
    assert len(results) > 0
    # Should find the Spaghetti Carbonara meal
    assert any('carbonara' in r['document'].lower() or 'italian' in r['document'].lower() 
               for r in results)

@pytest.mark.integration
def test_household_items_logic(integration_env, fixtures_dir, monkeypatch):
    """Test household items are added based on purchase history"""
    
    from unittest.mock import Mock, patch
    import json
    
    mock_llm_response = json.dumps([
        {
            'option_name': 'Test Plan',
            'description': 'Test',
            'meals': [],
            'shopping_list': {'produce': []},
            'estimated_cost': '50',
            'prep_difficulty': 'easy'
        }
    ])
    
    with patch('src.llm_client.requests.post') as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{
                'message': {'content': mock_llm_response},
                'finish_reason': 'stop'
            }],
            'usage': {},
            'model': 'test'
        }
        mock_post.return_value = mock_response
        
        # Ingest data (including bills with household items)
        rag_engine = RAGEngine()
        data_ingestion = DataIngestion()
        data_ingestion.ingest_all_data(fixtures_dir, rag_engine)
        
        # Generate plan
        meal_planner = MealPlanner()
        plans = meal_planner.generate_meal_plans("shopping for weekend")
        
        # Should have household items in shopping list
        assert len(plans) > 0
        # Household items should be present if purchase history indicates restocking
        if 'household' in plans[0].get('shopping_list', {}):
            household_items = plans[0]['shopping_list']['household']
            assert isinstance(household_items, list)

@pytest.mark.integration
def test_empty_database_handling(integration_env):
    """Test system handles empty database gracefully"""
    
    meal_planner = MealPlanner()
    plans = meal_planner.generate_meal_plans("test query")
    
    assert len(plans) > 0
    assert 'No Personalized Suggestions' in plans[0]['option_name']

@pytest.mark.integration
def test_metadata_filtering(integration_env, fixtures_dir):
    """Test RAG metadata filtering works correctly"""
    
    # Ingest data
    rag_engine = RAGEngine()
    data_ingestion = DataIngestion()
    data_ingestion.ingest_all_data(fixtures_dir, rag_engine)
    
    # Query only meals
    meal_results = rag_engine.query(
        "food",
        top_k=10,
        filter_metadata={'type': 'meal'}
    )
    
    assert all(r['metadata']['type'] == 'meal' for r in meal_results)
    
    # Query only preferences
    pref_results = rag_engine.query(
        "preferences",
        top_k=10,
        filter_metadata={'type': 'preference'}
    )
    
    assert all(r['metadata']['type'] == 'preference' for r in pref_results)

@pytest.fixture
def fixtures_dir():
    """Get fixtures directory"""
    return os.path.join('tests', 'fixtures')
```

**Run Integration Tests**:
```bash
pytest tests/test_integration.py -v -m integration
```

**Run All Tests**:
```bash
# Run all tests with coverage
pytest tests/ -v --cov=src --cov-report=html --cov-report=term

# Run only unit tests (exclude integration)
pytest tests/ -v -m "not integration"

# Run with verbose output
pytest tests/ -v -s
```

**Success Criteria**:
✅ Complete workflow works end-to-end
✅ Data persists between sessions
✅ RAG retrieves relevant documents
✅ Household items logic works
✅ Empty database is handled gracefully
✅ All tests pass with >80% coverage

---

## PHASE 9: Documentation

### Task 9.1: Complete README

**Instructions**:

Update `README.md`:
```markdown
# RAG-Powered Home Meal & Shopping Assistant

An intelligent CLI application that uses RAG (Retrieval-Augmented Generation) and AI to suggest personalized meal plans and comprehensive shopping lists based on your cooking history, preferences, and household needs.

## Features

- 🍽️ **Personalized Meal Plans**: Get AI-generated meal suggestions based on your history
- 🛒 **Comprehensive Shopping Lists**: Includes both food ingredients and household items
- 📊 **RAG-Powered Context**: Uses vector database to retrieve relevant past data
- 🧠 **Learning System**: Improves suggestions based on your choices
- 🏠 **Household Items**: Smart restocking suggestions based on purchase history
- 💾 **Data Persistence**: All your data stays local and private

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenRouter API key ([Get one here](https://openrouter.ai))

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/meal-shopping-assistant.git
   cd meal-shopping-assistant
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the package**:
   ```bash
   pip install -e .
   ```

5. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenRouter API key
   ```

## Quick Start

### 1. Prepare Your Data

Create JSON files with your meal history, preferences, bills, and grocery lists. See [Data Formats](#data-formats) below.

Example structure:
```
data/raw/
├── meals.json
├── preferences.json
├── bills.json
└── grocery_lists.json
```

### 2. Ingest Data

```bash
# Ingest all data files at once
meal-assistant ingest --all data/raw

# Or ingest individually
meal-assistant ingest --meals data/raw/meals.json
meal-assistant ingest --preferences data/raw/preferences.json
```

### 3. Generate Meal Plans

```bash
# Get suggestions for 3 days
meal-assistant plan "what should I shop for 3 days"

# Weekend planning
meal-assistant plan "weekend meals"

# Party planning
meal-assistant plan "party for 15 people"

# With custom number of options
meal-assistant plan "quick weeknight dinners" --num-options 5
```

### 4. View History

```bash
# See your past choices
meal-assistant history

# Last 5 choices
meal-assistant history --last 5
```

### 5. Check Database Stats

```bash
meal-assistant stats
```

## Usage

### Commands

#### `ingest` - Add data to the system

```bash
# Ingest all data from a directory
meal-assistant ingest --all data/raw

# Ingest specific files
meal-assistant ingest --meals meals.json
meal-assistant ingest --preferences prefs.json
meal-assistant ingest --bills bills.json
meal-assistant ingest --grocery-lists lists.json
```

#### `plan` - Generate meal plans

```bash
# Basic usage
meal-assistant plan "what should I shop for 3 days"

# Export to file
meal-assistant plan "weekend meals" --export my_plan.json

# Custom number of options
meal-assistant plan "dinner ideas" --num-options 5
```

#### `history` - View past choices

```bash
meal-assistant history
meal-assistant history --last 10
```

#### `stats` - Database statistics

```bash
meal-assistant stats
```

#### `reset` - Clear database

```bash
meal-assistant reset
```

## Data Formats

### Meals (`meals.json`)

```json
[
  {
    "date": "2024-01-15",
    "meal_name": "Spaghetti Carbonara",
    "cuisine": "Italian",
    "servings": 4,
    "ingredients": [
      "spaghetti pasta (400g)",
      "eggs (4 large)",
      "bacon (200g)",
      "parmesan cheese (100g)"
    ],
    "prep_time_minutes": 30,
    "difficulty": "medium",
    "notes": "Family loved it!",
    "rating": 5
  }
]
```

### Preferences (`preferences.json`)

```json
{
  "dietary_restrictions": ["No shellfish (allergy)"],
  "favorite_cuisines": ["Italian", "Mexican", "Asian"],
  "disliked_foods": ["mushrooms"],
  "preferred_proteins": ["chicken", "fish", "beans"],
  "household_size": 4,
  "notes": ["Monday is meatless day"],
  "budget": {
    "weekly_grocery": 150,
    "currency": "USD"
  }
}
```

### Bills (`bills.json`)

```json
[
  {
    "date": "2024-01-14",
    "store": "Whole Foods",
    "total_amount": 87.45,
    "currency": "USD",
    "items": [
      {
        "name": "chicken breast",
        "quantity": 1.5,
        "unit": "lb",
        "price": 12.50,
        "category": "meat"
      },
      {
        "name": "dish soap",
        "quantity": 1,
        "unit": "bottle",
        "price": 4.99,
        "category": "household"
      }
    ]
  }
]
```

### Grocery Lists (`grocery_lists.json`)

```json
[
  {
    "date": "2024-01-07",
    "occasion": "weekly shopping",
    "items": ["milk", "bread", "eggs"],
    "completed": true,
    "notes": "Regular weekly stock-up"
  }
]
```

## Configuration

Edit `config/config.yaml` to customize:

- RAG settings (embedding model, top_k results)
- LLM settings (model, temperature, max_tokens)
- Household item restock intervals
- Logging preferences

Edit `config/prompts.yaml` to customize AI prompts.

## Development

### Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Integration tests only
pytest tests/ -m integration

# Unit tests only
pytest tests/ -m "not integration"
```

### Project Structure

```
meal-shopping-assistant/
├── src/
│   ├── cli.py              # CLI interface
│   ├── rag_engine.py       # RAG/vector database
│   ├── llm_client.py       # OpenRouter API client
│   ├── data_ingestion.py   # Data processing
│   ├── meal_planner.py     # Main orchestration
│   └── utils.py            # Utilities
├── tests/                  # Test suite
├── config/                 # Configuration files
├── data/                   # Data storage
└── requirements.txt        # Dependencies
```

## Troubleshooting

### "OPENROUTER_API_KEY not found"

Make sure you've created a `.env` file with your API key:
```bash
OPENROUTER_API_KEY=your_key_here
```

### "No relevant information found"

You need to ingest data first:
```bash
meal-assistant ingest --all data/raw
```

### Low-quality suggestions

- Add more meal history data
- Make sure preferences are detailed
- Log your choices to improve future suggestions

### API Rate Limiting

The system includes automatic retry logic. If you hit rate limits frequently, consider:
- Reducing `--num-options`
- Waiting a few moments between requests

## Privacy & Data

- All data is stored locally on your machine
- No data is sent anywhere except LLM API calls for generation
- Your meal history and preferences never leave your control
- Vector database is stored in `./data/processed/vector_db`

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Built with [sentence-transformers](https://www.sbert.net/) for embeddings
- Uses [FAISS](https://github.com/facebookresearch/faiss) for vector search
- Powered by [OpenRouter](https://openrouter.ai) for LLM access
- CLI built with [Click](https://click.palletsprojects.com/) and [Rich](https://rich.readthedocs.io/)

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing issues for solutions
- Read the [FAQ](docs/FAQ.md)

---

**Happy meal planning! 🍽️🛒**
```

### Task 9.2: Additional Documentation

Create `docs/FAQ.md`:
```markdown
# Frequently Asked Questions

## General

**Q: What is RAG?**
A: RAG (Retrieval-Augmented Generation) combines vector search with AI generation. It retrieves relevant context from your data before generating suggestions.

**Q: Is my data private?**
A: Yes! All your data stays on your machine. Only the generated prompts (with context) are sent to the LLM API.

**Q: Do I need an internet connection?**
A: Yes, for generating meal plans (LLM API calls). Data ingestion works offline.

## Usage

**Q: How much data do I need to start?**
A: Minimum: 5-10 meals and basic preferences. More data = better suggestions.

**Q: Can I edit my data after ingesting?**
A: Yes! Edit your JSON files and re-ingest. Use `meal-assistant reset` first to clear old data.

**Q: How does the learning work?**
A: Your choices are logged and added to the RAG database, influencing future suggestions.

## Technical

**Q: Which LLM models are supported?**
A: Any model available on OpenRouter. Default is Claude Sonnet 3.5.

**Q: Can I use local LLMs?**
A: Not currently, but you could modify `llm_client.py` to support local models.

**Q: How large can my database get?**
A: FAISS can handle millions of vectors. Typical usage: <10MB.

## Troubleshooting

**Q: Suggestions aren't relevant**
A: Ensure you have diverse data (meals, preferences, bills). Log your choices to improve.

**Q: API costs too high?**
A: Use cheaper models, reduce `max_tokens`, or generate fewer options.

**Q: Installation fails?**
A: Make sure you're using Python 3.8+. Try updating pip: `pip install --upgrade pip`
```

Create `.github/workflows/tests.yml` (optional, for CI):
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10', '3.11']
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest tests/ -v -m "not integration" --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v2
      with:
        file: ./coverage.xml
```

**Success Criteria**:
✅ Complete, clear README
✅ FAQ addresses common questions
✅ Data format examples provided
✅ Troubleshooting guide included
✅ Contributing guidelines present

---

## Final Checklist

Before considering the project complete, verify:

- [ ] All dependencies install cleanly
- [ ] All tests pass (`pytest tests/ -v`)
- [ ] Test coverage >80% (`pytest --cov=src`)
- [ ] CLI commands work as expected
- [ ] Sample data can be ingested
- [ ] Meal plans generate successfully
- [ ] User
choices are logged properly
- [ ] RAG retrieval is accurate
- [ ] Documentation is complete
- [ ] `.env.example` is provided
- [ ] `.gitignore` excludes sensitive files
- [ ] Code is formatted consistently
- [ ] Error handling is robust
- [ ] Logging works properly
- [ ] Export functionality works
- [ ] Database persistence works
- [ ] Reset command works safely

---

## PHASE 10: Polish & Optimization

### Task 10.1: Code Quality & Formatting

**Instructions**:

1. **Format code with Black**:
```bash
black src/ tests/
```

2. **Check with flake8**:
```bash
flake8 src/ tests/ --max-line-length=100 --extend-ignore=E203,W503
```

3. **Add type hints** (optional but recommended):

Update critical functions with type hints. Example for `src/meal_planner.py`:
```python
from typing import Dict, Any, List, Optional

def generate_meal_plans(
    self,
    query: str,
    num_options: int = 3
) -> List[Dict[str, Any]]:
    """Generate meal plan options based on query."""
    ...
```

### Task 10.2: Performance Optimization

**Instructions**:

Add caching to RAG queries in `src/rag_engine.py`:

```python
from functools import lru_cache
import hashlib

class RAGEngine:
    def __init__(self, config_path: str = "config/config.yaml"):
        # ... existing code ...
        self._query_cache = {}
    
    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Query with caching."""
        # Create cache key
        cache_key = self._create_cache_key(query, top_k, filter_metadata)
        
        # Check cache
        if cache_key in self._query_cache:
            logger.debug(f"Cache hit for query: {query[:50]}")
            return self._query_cache[cache_key]
        
        # Perform query
        results = self._perform_query(query, top_k, filter_metadata)
        
        # Cache results (limit cache size)
        if len(self._query_cache) > 100:
            # Remove oldest entry
            self._query_cache.pop(next(iter(self._query_cache)))
        
        self._query_cache[cache_key] = results
        return results
    
    def _create_cache_key(
        self,
        query: str,
        top_k: Optional[int],
        filter_metadata: Optional[Dict[str, Any]]
    ) -> str:
        """Create cache key from query parameters."""
        key_data = f"{query}:{top_k}:{str(filter_metadata)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _perform_query(
        self,
        query: str,
        top_k: Optional[int],
        filter_metadata: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Actual query implementation (existing code)."""
        # Move existing query logic here
        ...
```

### Task 10.3: Enhanced Error Messages

**Instructions**:

Update `src/cli.py` with better error messages:

```python
@main.command()
def plan(query, num_options, export):
    """Generate meal plans and shopping lists."""
    try:
        # Check if API key is set
        if not os.getenv('OPENROUTER_API_KEY'):
            console.print(
                "\n[bold red]Error: OPENROUTER_API_KEY not found![/bold red]\n"
                "\n[yellow]Please follow these steps:[/yellow]\n"
                "1. Copy .env.example to .env\n"
                "2. Add your OpenRouter API key to .env\n"
                "3. Get a key at: https://openrouter.ai\n",
                style="yellow"
            )
            sys.exit(1)
        
        # Check if database has data
        rag_engine = RAGEngine()
        stats = rag_engine.get_stats()
        
        if stats['total_documents'] == 0:
            console.print(
                "\n[bold yellow]No data found in database![/bold yellow]\n"
                "\n[cyan]Get started by ingesting your data:[/cyan]\n"
                "1. Prepare your data files (see README.md for formats)\n"
                "2. Run: meal-assistant ingest --all data/raw\n"
                "3. Then try planning again!\n",
                style="cyan"
            )
            sys.exit(0)
        
        # Continue with existing plan logic...
        
    except FileNotFoundError as e:
        console.print(
            f"\n[bold red]File not found: {e.filename}[/bold red]\n"
            "[yellow]Please check the file path and try again.[/yellow]\n"
        )
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]✗ Error: {str(e)}[/bold red]\n")
        console.print(
            "[yellow]If this persists, try:[/yellow]\n"
            "1. Check your .env file has a valid API key\n"
            "2. Ensure you have ingested data\n"
            "3. Run with --verbose flag for more details\n"
            "4. Check logs at data/logs/app.log\n"
        )
        if logger:
            logger.exception("Plan generation failed")
        sys.exit(1)
```

### Task 10.4: Sample Data Generator

**Instructions**:

Create `scripts/generate_sample_data.py`:

```python
"""Generate sample data for testing the meal assistant."""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

def generate_sample_meals(num_meals: int = 20) -> list:
    """Generate sample meal data."""
    cuisines = ['Italian', 'Mexican', 'Asian', 'American', 'Mediterranean', 'Indian']
    difficulties = ['easy', 'medium', 'hard']
    
    meal_templates = [
        ('Spaghetti Carbonara', 'Italian', ['pasta', 'eggs', 'bacon', 'parmesan']),
        ('Chicken Tacos', 'Mexican', ['chicken', 'tortillas', 'cheese', 'lettuce']),
        ('Stir Fry', 'Asian', ['chicken', 'vegetables', 'soy sauce', 'rice']),
        ('Burger', 'American', ['ground beef', 'buns', 'lettuce', 'tomato']),
        ('Greek Salad', 'Mediterranean', ['feta', 'olives', 'cucumber', 'tomato']),
        ('Curry', 'Indian', ['chicken', 'curry paste', 'coconut milk', 'rice']),
    ]
    
    meals = []
    start_date = datetime.now() - timedelta(days=90)
    
    for i in range(num_meals):
        template = random.choice(meal_templates)
        date = start_date + timedelta(days=random.randint(0, 90))
        
        meal = {
            'date': date.strftime('%Y-%m-%d'),
            'meal_name': template[0],
            'cuisine': template[1],
            'servings': random.randint(2, 6),
            'ingredients': template[2],
            'prep_time_minutes': random.randint(15, 90),
            'difficulty': random.choice(difficulties),
            'notes': 'Sample meal',
            'rating': random.randint(3, 5)
        }
        meals.append(meal)
    
    return sorted(meals, key=lambda x: x['date'])

def generate_sample_preferences() -> dict:
    """Generate sample preferences."""
    return {
        'dietary_restrictions': ['No shellfish'],
        'favorite_cuisines': ['Italian', 'Mexican', 'Asian'],
        'disliked_foods': ['mushrooms', 'cilantro'],
        'preferred_proteins': ['chicken', 'fish', 'beans'],
        'household_size': 4,
        'notes': [
            'Kids prefer mild flavors',
            'Monday is meatless day',
            'Try to incorporate vegetables'
        ],
        'budget': {
            'weekly_grocery': 150,
            'currency': 'USD'
        },
        'cooking_frequency': {
            'weekday': 'quick meals under 30 minutes',
            'weekend': 'can spend more time'
        }
    }

def generate_sample_bills(num_bills: int = 10) -> list:
    """Generate sample shopping bills."""
    stores = ['Whole Foods', 'Trader Joes', 'Local Market', 'Costco']
    categories = {
        'produce': ['tomatoes', 'lettuce', 'onions', 'carrots'],
        'meat': ['chicken breast', 'ground beef', 'salmon'],
        'dairy': ['milk', 'cheese', 'eggs', 'yogurt'],
        'household': ['dish soap', 'paper towels', 'laundry detergent']
    }
    
    bills = []
    start_date = datetime.now() - timedelta(days=60)
    
    for i in range(num_bills):
        date = start_date + timedelta(days=i * 6)
        
        items = []
        total = 0
        
        for category, item_list in categories.items():
            num_items = random.randint(1, 3)
            for _ in range(num_items):
                item_name = random.choice(item_list)
                price = random.uniform(2, 15)
                
                items.append({
                    'name': item_name,
                    'quantity': random.randint(1, 3),
                    'unit': 'each',
                    'price': round(price, 2),
                    'category': category
                })
                total += price
        
        bill = {
            'date': date.strftime('%Y-%m-%d'),
            'store': random.choice(stores),
            'total_amount': round(total, 2),
            'currency': 'USD',
            'items': items
        }
        bills.append(bill)
    
    return bills

def generate_sample_grocery_lists(num_lists: int = 5) -> list:
    """Generate sample grocery lists."""
    occasions = ['weekly shopping', 'quick restock', 'party prep']
    
    items_pool = [
        'milk', 'bread', 'eggs', 'chicken', 'vegetables',
        'rice', 'pasta', 'cheese', 'yogurt', 'fruit'
    ]
    
    lists = []
    start_date = datetime.now() - timedelta(days=30)
    
    for i in range(num_lists):
        date = start_date + timedelta(days=i * 7)
        
        grocery_list = {
            'date': date.strftime('%Y-%m-%d'),
            'occasion': random.choice(occasions),
            'items': random.sample(items_pool, k=random.randint(5, 10)),
            'completed': True,
            'notes': 'Sample grocery list'
        }
        lists.append(grocery_list)
    
    return lists

def main():
    """Generate all sample data files."""
    output_dir = Path('data/raw')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating sample data...")
    
    # Generate meals
    meals = generate_sample_meals(20)
    with open(output_dir / 'meals.json', 'w') as f:
        json.dump(meals, f, indent=2)
    print(f"✓ Generated {len(meals)} meals")
    
    # Generate preferences
    preferences = generate_sample_preferences()
    with open(output_dir / 'preferences.json', 'w') as f:
        json.dump(preferences, f, indent=2)
    print("✓ Generated preferences")
    
    # Generate bills
    bills = generate_sample_bills(10)
    with open(output_dir / 'bills.json', 'w') as f:
        json.dump(bills, f, indent=2)
    print(f"✓ Generated {len(bills)} bills")
    
    # Generate grocery lists
    grocery_lists = generate_sample_grocery_lists(5)
    with open(output_dir / 'grocery_lists.json', 'w') as f:
        json.dump(grocery_lists, f, indent=2)
    print(f"✓ Generated {len(grocery_lists)} grocery lists")
    
    print(f"\n✓ All sample data generated in {output_dir}")
    print("\nNext steps:")
    print("1. Run: meal-assistant ingest --all data/raw")
    print("2. Run: meal-assistant plan 'what should I shop for 3 days'")

if __name__ == '__main__':
    main()
```

Make it executable:
```bash
chmod +x scripts/generate_sample_data.py
```

Add to README:
```markdown
## Quick Demo

Want to try it out without preparing data?

```bash
# Generate sample data
python scripts/generate_sample_data.py

# Ingest sample data
meal-assistant ingest --all data/raw

# Generate a plan
meal-assistant plan "what should I shop for 3 days"
```
```

### Task 10.5: Final Testing Script

**Instructions**:

Create `scripts/run_all_tests.sh`:

```bash
#!/bin/bash

echo "======================================"
echo "Running Complete Test Suite"
echo "======================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓ $2${NC}"
    else
        echo -e "${RED}✗ $2${NC}"
        exit 1
    fi
}

# 1. Code formatting check
echo "1. Checking code formatting..."
black --check src/ tests/ 2>/dev/null
print_status $? "Code formatting"

# 2. Linting
echo "2. Running linter..."
flake8 src/ tests/ --max-line-length=100 --extend-ignore=E203,W503 2>/dev/null
print_status $? "Linting"

# 3. Unit tests
echo "3. Running unit tests..."
pytest tests/ -v -m "not integration" --tb=short
print_status $? "Unit tests"

# 4. Integration tests
echo "4. Running integration tests..."
pytest tests/ -v -m integration --tb=short
print_status $? "Integration tests"

# 5. Coverage check
echo "5. Checking test coverage..."
pytest tests/ --cov=src --cov-report=term --cov-fail-under=80 -q
print_status $? "Coverage (>80%)"

# 6. Type checking (if mypy is installed)
if command -v mypy &> /dev/null; then
    echo "6. Running type checker..."
    mypy src/ --ignore-missing-imports 2>/dev/null
    print_status $? "Type checking"
fi

echo ""
echo "======================================"
echo -e "${GREEN}All tests passed!${NC}"
echo "======================================"
echo ""
echo "Ready for deployment ✨"
```

Make it executable:
```bash
chmod +x scripts/run_all_tests.sh
```

---

## Final Deployment Checklist

### Pre-Release

- [ ] Run `scripts/run_all_tests.sh` - all pass
- [ ] Generate sample data and test end-to-end
- [ ] Review all documentation
- [ ] Test on fresh Python environment
- [ ] Check all examples in README work
- [ ] Verify `.env.example` is correct
- [ ] Review error messages are helpful
- [ ] Test CLI help text is clear
- [ ] Check logging works properly
- [ ] Verify data privacy (no data leaks)

### Release

- [ ] Tag version: `git tag v0.1.0`
- [ ] Push to repository
- [ ] Create GitHub release with notes
- [ ] Update README with any final changes
- [ ] Consider publishing to PyPI (optional)

### Post-Release

- [ ] Monitor for issues
- [ ] Respond to user feedback
- [ ] Plan future enhancements

---

## Summary

You now have a complete implementation guide for the RAG-Powered Home Meal & Shopping Assistant. The project includes:

✅ **Core Functionality**:
- RAG engine with FAISS vector database
- OpenRouter API integration
- Data ingestion for multiple formats
- Intelligent meal planning
- Household item suggestions
- User choice logging and learning

✅ **Quality Assurance**:
- Comprehensive test suite (unit + integration)
- >80% test coverage
- Error handling throughout
- Input validation
- Retry logic for API calls

✅ **User Experience**:
- Beautiful CLI with Rich formatting
- Helpful error messages
- Sample data generator
- Export functionality
- Database statistics

✅ **Documentation**:
- Complete README
- FAQ document
- Code comments
- Type hints
- Usage examples

✅ **Developer Tools**:
- Automated testing script
- CI/CD workflow (optional)
- Code formatting
- Linting configuration

The assistant is now ready to start implementing each phase sequentially, running tests at each step to ensure quality!