"""Meal planning logic that combines RAG and LLM."""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.rag_engine import RAGEngine
from src.llm_client import OpenRouterClient
from src.utils import load_prompts, parse_time_query, save_json, load_json

logger = logging.getLogger(__name__)

class MealPlanner:
    """Core meal planning logic."""
    
    def __init__(
        self,
        rag_engine: Optional[RAGEngine] = None,
        llm_client: Optional[OpenRouterClient] = None,
        prompts_path: str = "config/prompts.yaml"
    ):
        """
        Initialize meal planner.
        
        Args:
            rag_engine: RAG engine instance (creates new if None)
            llm_client: LLM client instance (creates new if None)
            prompts_path: Path to prompts configuration
        """
        self.rag_engine = rag_engine if rag_engine else RAGEngine()
        self.llm_client = llm_client if llm_client else OpenRouterClient()
        self.prompts = load_prompts(prompts_path)
        
        logger.info("Meal planner initialized")
    
    def generate_meal_plans(
        self,
        query: str,
        num_options: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Generate meal plan options based on user query.
        
        Args:
            query: User query (e.g., "3 days", "weekend", "party for 10")
            num_options: Number of options to generate (default: 3)
            
        Returns:
            List of meal plan dictionaries
        """
        logger.info(f"Generating meal plans for query: '{query}'")
        
        # Parse the query to extract time information
        time_info = parse_time_query(query)
        num_days = time_info['num_days']
        occasion = time_info['occasion']
        num_people = time_info.get('num_people', 2)
        
        logger.info(f"Parsed query: {num_days} days, occasion: {occasion}, people: {num_people}")
        
        # Check if database has data
        stats = self.rag_engine.get_stats()
        if stats['total_documents'] == 0:
            logger.warning("No data in RAG database")
            return self._generate_fallback_response()

        # Retrieve relevant context from RAG with diversity
        # diversity_factor=0.5 means retrieve 2x documents and randomly sample for variety
        context = self.rag_engine.get_context(query, top_k=15, diversity_factor=0.5)
        
        # Build the prompt
        system_prompt = self.prompts['system_prompt']
        user_prompt = self.prompts['meal_planning_prompt'].format(
            context=context,
            query=query,
            num_days=num_days
        )
        
        # Generate response from LLM
        # Use higher max_tokens for meal planning since we now include breakfast, lunch, and dinner
        try:
            response = self.llm_client.generate_response(
                user_prompt,
                system_prompt=system_prompt,
                max_tokens=8000  # Increased to accommodate breakfast, lunch, dinner for multiple days
            )
            
            # Parse JSON response
            parsed_response = self.llm_client.parse_json_response(response['content'])

            # Handle both direct array and object with meal_plans key
            if isinstance(parsed_response, dict) and 'meal_plans' in parsed_response:
                meal_plans = parsed_response['meal_plans']
            elif isinstance(parsed_response, list):
                meal_plans = parsed_response
            else:
                logger.error(f"Unexpected response format: {type(parsed_response)}")
                return self._generate_fallback_response()
            
            # Limit to requested number of options
            meal_plans = meal_plans[:num_options]
            
            # Add metadata
            for plan in meal_plans:
                plan['generated_at'] = datetime.now().isoformat()
                plan['query'] = query
                plan['num_days'] = num_days
                plan['occasion'] = occasion
            
            logger.info(f"Successfully generated {len(meal_plans)} meal plans")
            return meal_plans
            
        except Exception as e:
            logger.error(f"Error generating meal plans: {e}")
            return self._generate_fallback_response()
    
    def _generate_fallback_response(self) -> List[Dict[str, Any]]:
        """
        Generate a fallback response when LLM fails or no data available.
        
        Returns:
            List with fallback message
        """
        return [{
            'option_name': 'No Data Available',
            'description': self.prompts['fallback_response'],
            'meals': [],
            'shopping_list': {},
            'error': True
        }]
    
    def log_user_choice(
        self,
        selected_option: Dict[str, Any],
        query: str,
        all_options: List[Dict[str, Any]]
    ) -> None:
        """
        Log user's choice for future learning.
        
        Args:
            selected_option: The option the user selected
            query: Original query
            all_options: All options that were presented
        """
        logger.info(f"Logging user choice: {selected_option.get('option_name')}")
        
        # Load existing choices
        choices_file = "data/logs/user_choices.json"
        choices = load_json(choices_file) or []
        
        # Create choice record
        choice_record = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'selected_option': selected_option,
            'num_options_presented': len(all_options),
            'selected_index': next(
                (i for i, opt in enumerate(all_options) if opt == selected_option),
                -1
            )
        }
        
        choices.append(choice_record)
        
        # Save choices
        save_json(choices, choices_file)
        
        logger.info("User choice logged successfully")
    
    def get_choice_history(
        self,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get history of user choices.
        
        Args:
            limit: Optional limit on number of records to return
            
        Returns:
            List of choice records
        """
        choices_file = "data/logs/user_choices.json"
        choices = load_json(choices_file) or []
        
        # Sort by timestamp (most recent first)
        choices.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        if limit:
            choices = choices[:limit]
        
        return choices
    
    def analyze_household_items(self) -> List[Dict[str, Any]]:
        """
        Analyze household items that might need restocking.
        
        Returns:
            List of household items with purchase history
        """
        logger.info("Analyzing household items")
        
        # Query for household items
        results = self.rag_engine.query(
            "household items cleaning supplies paper products",
            top_k=20,
            filter_metadata={'type': 'household_item'}
        )
        
        # Group by item name and find most recent purchase
        items_dict = {}
        for result in results:
            item_name = result['metadata'].get('item_name', 'unknown')
            date = result['metadata'].get('date', '')
            
            if item_name not in items_dict or date > items_dict[item_name]['last_purchase']:
                items_dict[item_name] = {
                    'item': item_name,
                    'last_purchase': date,
                    'store': result['metadata'].get('store', 'unknown')
                }
        
        # Convert to list and sort by date
        items_list = list(items_dict.values())
        items_list.sort(key=lambda x: x['last_purchase'], reverse=True)
        
        return items_list
    
    def export_shopping_list(
        self,
        meal_plan: Dict[str, Any],
        filepath: str,
        format: str = 'txt'
    ) -> None:
        """
        Export shopping list to file.
        
        Args:
            meal_plan: Meal plan dictionary
            filepath: Path to save file
            format: Format ('txt' or 'json')
        """
        logger.info(f"Exporting shopping list to {filepath}")
        
        shopping_list = meal_plan.get('shopping_list', {})
        
        if format == 'json':
            save_json(shopping_list, filepath)
        else:
            # Text format
            lines = [f"Shopping List: {meal_plan.get('option_name', 'Meal Plan')}\n"]
            lines.append("=" * 50 + "\n\n")
            
            for category, items in shopping_list.items():
                if items:
                    lines.append(f"{category.replace('_', ' ').title()}:\n")
                    for item in items:
                        if isinstance(item, dict):
                            quantity = item.get('quantity', '')
                            unit = item.get('unit', '')
                            name = item.get('item', '')
                            if quantity and unit:
                                lines.append(f"  [ ] {quantity} {unit} {name}\n")
                            else:
                                lines.append(f"  [ ] {name}\n")
                        else:
                            lines.append(f"  [ ] {item}\n")
                    lines.append("\n")
            
            with open(filepath, 'w') as f:
                f.writelines(lines)
        
        logger.info("Shopping list exported successfully")

