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
        """Format grocery list data as a document string."""
        parts = []

        parts.append(f"Grocery List from {grocery_list.get('date', 'Unknown')}")
        parts.append(f"Occasion: {grocery_list.get('occasion', 'Unknown')}")

        items = grocery_list.get('items', [])
        if items:
            parts.append(f"Items: {', '.join(items)}")

        if grocery_list.get('notes'):
            parts.append(f"Notes: {grocery_list['notes']}")

        return '\n'.join(parts)

    def ingest_all_data(self, data_dir: str, rag_engine) -> Dict[str, int]:
        """
        Ingest all data files from a directory.

        Args:
            data_dir: Directory containing data files
            rag_engine: RAG engine instance to add documents to

        Returns:
            Dictionary with counts of ingested items
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

