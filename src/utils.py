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

