"""Simplified meal planner for individual meal selection workflow."""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import defaultdict

from src.rag_engine import RAGEngine
from src.llm_client import OpenRouterClient
from src.utils import load_prompts

logger = logging.getLogger(__name__)


class SimpleMealPlanner:
    """Simplified meal planner for generating individual meal options."""
    
    def __init__(
        self,
        rag_engine: Optional[RAGEngine] = None,
        llm_client: Optional[OpenRouterClient] = None,
        prompts_path: str = "config/prompts.yaml"
    ):
        """
        Initialize simple meal planner.
        
        Args:
            rag_engine: RAG engine instance (creates new if None)
            llm_client: LLM client instance (creates new if None)
            prompts_path: Path to prompts configuration
        """
        self.rag_engine = rag_engine if rag_engine else RAGEngine()
        self.llm_client = llm_client if llm_client else OpenRouterClient()
        self.prompts = load_prompts(prompts_path)
        
        logger.info("Simple meal planner initialized")
    
    def generate_meal_options(
        self,
        num_meals: int = 15,
        diversity_factor: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Generate individual meal options from RAG database.
        
        Args:
            num_meals: Number of meal options to generate
            diversity_factor: Diversity factor for RAG retrieval (0.0-1.0)
            
        Returns:
            List of meal dictionaries with name, description, ingredients
        """
        logger.info(f"Generating {num_meals} meal options")
        
        # Check if database has data
        stats = self.rag_engine.get_stats()
        if stats['total_documents'] == 0:
            logger.warning("No data in RAG database")
            return []
        
        # Retrieve diverse meal context from RAG
        query = self._get_diverse_query()
        logger.info(f"Using dynamic query: '{query}'")
        
        context = self.rag_engine.get_context(
            query,
            top_k=num_meals,
            diversity_factor=diversity_factor
        )
        
        # Build the prompt for individual meals
        system_prompt = self.prompts['system_prompt']
        user_prompt = self.prompts.get('individual_meals_prompt', self._get_default_individual_meals_prompt()).format(
            context=context,
            num_meals=num_meals
        )
        
        # Generate response from LLM
        try:
            response = self.llm_client.generate_response(
                user_prompt,
                system_prompt=system_prompt,
                max_tokens=6000
            )
            
            # Parse JSON response
            parsed_response = self.llm_client.parse_json_response(response['content'])
            
            # Handle both direct array and object with meals key
            if isinstance(parsed_response, dict) and 'meals' in parsed_response:
                meals = parsed_response['meals']
            elif isinstance(parsed_response, list):
                meals = parsed_response
            else:
                logger.error(f"Unexpected response format: {type(parsed_response)}")
                return []
            
            # Add metadata
            for i, meal in enumerate(meals):
                meal['id'] = i
                meal['generated_at'] = datetime.now().isoformat()
            
            logger.info(f"Successfully generated {len(meals)} meal options")
            return meals[:num_meals]
            
        except Exception as e:
            logger.error(f"Error generating meal options: {e}")
            return []
    
    def calculate_shopping_list(
        self,
        selected_meals: List[Dict[str, Any]],
        include_household_items: bool = True
    ) -> Dict[str, Any]:
        """
        Calculate shopping list from selected meals using LLM for smart merging.

        Args:
            selected_meals: List of selected meal dictionaries
            include_household_items: Whether to include household items checklist

        Returns:
            Shopping list dictionary organized by category with simple item names
        """
        logger.info(f"Calculating shopping list for {len(selected_meals)} meals")

        if not selected_meals:
            return {}

        # Collect all ingredients from selected meals
        all_ingredients = []
        meal_names = []

        for meal in selected_meals:
            meal_names.append(meal.get('meal_name', 'Unknown'))
            ingredients = meal.get('ingredients', [])
            all_ingredients.extend(ingredients)

        # Use LLM to intelligently merge and categorize ingredients
        shopping_list = self._generate_smart_shopping_list(all_ingredients, meal_names)

        # Add household items if requested
        if include_household_items:
            shopping_list['household_checklist'] = [
                item['item'] for item in self._get_household_items()
            ]

        return shopping_list

    def _generate_smart_shopping_list(
        self,
        ingredients: List[str],
        meal_names: List[str]
    ) -> Dict[str, List[str]]:
        """
        Use LLM to generate a smart, easy-to-understand shopping list.

        Args:
            ingredients: List of all ingredients from selected meals
            meal_names: List of meal names for context

        Returns:
            Dictionary of categorized shopping items
        """
        # Build prompt for LLM
        prompt = f"""You are helping create a simple, easy-to-understand shopping list.

Selected meals:
{', '.join(meal_names)}

All ingredients from these meals:
{chr(10).join(f"- {ing}" for ing in ingredients)}

Please create a smart shopping list that:
1. Merges duplicate ingredients (e.g., "500g chicken thighs" + "300g chicken breast" = "chicken")
2. Uses simple, generic names (e.g., "chicken" not "chicken thighs")
3. Removes all quantities and units
4. Groups similar items intelligently
5. Categorizes items into: produce, proteins, dairy, grains_pasta, pantry

Return ONLY valid JSON in this format:
```json
{{
  "produce": ["garlic", "ginger", "cilantro"],
  "proteins": ["chicken", "pork", "eggs"],
  "dairy": ["coconut milk"],
  "grains_pasta": ["rice", "noodles"],
  "pantry": ["soy sauce", "fish sauce", "curry paste"]
}}
```

Important:
- Each item should appear only ONCE across all categories
- Use simple, common names that are easy to remember
- Sort items alphabetically within each category
- NO quantities, NO units, NO preparation notes
- NO markdown, NO explanations, ONLY JSON
"""

        try:
            # Generate response from LLM
            response = self.llm_client.generate_response(
                prompt,
                system_prompt="You are a helpful assistant that creates simple, easy-to-understand shopping lists.",
                max_tokens=2000
            )

            # Parse JSON response
            shopping_list = self.llm_client.parse_json_response(response['content'])

            # Validate and clean the response
            if isinstance(shopping_list, dict):
                # Ensure all values are lists of strings
                cleaned = {}
                for category, items in shopping_list.items():
                    if isinstance(items, list):
                        cleaned[category] = [str(item).strip() for item in items if item]

                logger.info(f"Generated smart shopping list with {sum(len(v) for v in cleaned.values())} items")
                return cleaned
            else:
                logger.error(f"Unexpected shopping list format: {type(shopping_list)}")
                return self._fallback_shopping_list(ingredients)

        except Exception as e:
            logger.error(f"Error generating smart shopping list: {e}")
            return self._fallback_shopping_list(ingredients)

    def _fallback_shopping_list(self, ingredients: List[str]) -> Dict[str, List[str]]:
        """
        Fallback method if LLM fails - use simple extraction.

        Args:
            ingredients: List of ingredient strings

        Returns:
            Dictionary of categorized shopping items
        """
        logger.info("Using fallback shopping list generation")

        shopping_list = defaultdict(set)

        for ingredient in ingredients:
            category = self._categorize_ingredient(ingredient)
            item_name = self._extract_item_name(ingredient)
            shopping_list[category].add(item_name)

        # Convert sets to sorted lists
        return {
            category: sorted(list(items))
            for category, items in shopping_list.items()
        }
    
    def _extract_item_name(self, ingredient: str) -> str:
        """
        Extract clean item name from ingredient string.
        Removes quantities, units, and preparation notes.

        Examples:
            "500g chicken thighs" -> "chicken"
            "2 cups rice" -> "rice"
            "1 tsp cornstarch mixed with water" -> "cornstarch"
            "3 kaffir lime leaves" -> "kaffir lime leaves"
        """
        ingredient = ingredient.strip().lower()

        # Remove common quantity patterns at the start
        # Pattern: number + optional decimal + optional unit
        import re

        # Remove leading quantities and units
        patterns = [
            r'^\d+\.?\d*\s*(g|kg|lb|oz|ml|l|cup|cups|tbsp|tsp|tablespoon|teaspoon|piece|pieces|whole|unit|units|bunch|bunches|clove|cloves|stalk|stalks|slice|slices)\s+',
            r'^\d+\.?\d*\s+',  # Just numbers
        ]

        for pattern in patterns:
            ingredient = re.sub(pattern, '', ingredient)

        # Remove preparation notes (anything after common keywords)
        prep_keywords = [
            'mixed with', 'chopped', 'diced', 'sliced', 'minced', 'crushed',
            'peeled', 'deveined', 'boneless', 'skinless', 'cut into',
            'thinly sliced', 'finely chopped', 'roughly chopped'
        ]

        for keyword in prep_keywords:
            if keyword in ingredient:
                ingredient = ingredient.split(keyword)[0].strip()

        # Extract the main ingredient name (remove adjectives for common items)
        # Map specific ingredients to generic names
        ingredient_map = {
            'chicken thighs': 'chicken',
            'chicken breast': 'chicken',
            'chicken drumsticks': 'chicken',
            'chicken wings': 'chicken',
            'pork shoulder': 'pork',
            'pork chops': 'pork',
            'pork belly': 'pork',
            'beef chuck': 'beef',
            'beef sirloin': 'beef',
            'ground beef': 'beef',
            'ground pork': 'pork',
            'fish sauce': 'fish sauce',
            'soy sauce': 'soy sauce',
            'oyster sauce': 'oyster sauce',
            'coconut milk': 'coconut milk',
            'coconut cream': 'coconut milk',
            'thai basil': 'thai basil',
            'holy basil': 'holy basil',
            'sweet basil': 'basil',
            'kaffir lime leaves': 'kaffir lime leaves',
            'lime leaves': 'kaffir lime leaves',
            'green curry paste': 'green curry paste',
            'red curry paste': 'red curry paste',
            'yellow curry paste': 'yellow curry paste',
        }

        # Check if ingredient matches any mapping
        for key, value in ingredient_map.items():
            if key in ingredient:
                return value

        # If no mapping found, return cleaned ingredient
        return ingredient.strip()

    def _categorize_ingredient(self, ingredient: str) -> str:
        """Categorize ingredient into shopping list category."""
        ingredient_lower = ingredient.lower()

        # Produce
        if any(word in ingredient_lower for word in [
            'tomato', 'onion', 'garlic', 'ginger', 'pepper', 'carrot', 'lettuce',
            'cucumber', 'cabbage', 'broccoli', 'spinach', 'mushroom', 'herb',
            'lemongrass', 'lime', 'lemon', 'cilantro', 'basil', 'mint', 'shallot',
            'galangal', 'kaffir', 'chili', 'vegetable', 'produce', 'scallion',
            'bean sprout', 'bok choy', 'eggplant', 'zucchini', 'squash'
        ]):
            return 'produce'

        # Proteins
        if any(word in ingredient_lower for word in [
            'chicken', 'pork', 'beef', 'fish', 'shrimp', 'seafood', 'egg',
            'tofu', 'meat', 'protein', 'salmon', 'tuna', 'duck', 'lamb'
        ]):
            return 'proteins'

        # Dairy
        if any(word in ingredient_lower for word in [
            'milk', 'cheese', 'yogurt', 'butter', 'cream', 'dairy'
        ]):
            return 'dairy'

        # Grains & Pasta
        if any(word in ingredient_lower for word in [
            'rice', 'pasta', 'noodle', 'bread', 'flour', 'grain', 'quinoa', 'oat'
        ]):
            return 'grains_pasta'

        # Pantry
        return 'pantry'
    
    def _get_household_items(self) -> List[Dict[str, str]]:
        """Get standard household items checklist."""
        return [
            {'item': 'dish soap', 'category': 'cleaning'},
            {'item': 'sponges', 'category': 'cleaning'},
            {'item': 'laundry detergent', 'category': 'cleaning'},
            {'item': 'trash bags', 'category': 'kitchen essentials'},
            {'item': 'paper towels', 'category': 'paper products'},
            {'item': 'toilet paper', 'category': 'paper products'},
            {'item': 'hand soap', 'category': 'toiletries'},
        ]
    
    def _get_diverse_query(self) -> str:
        """Get a random diverse query for RAG retrieval."""
        import random
        queries = [
            "easy weeknight dinners for family",
            "budget friendly meals with common ingredients",
            "healthy lunch ideas for work",
            "30 minute meals for busy days",
            "one pot meals for easy cleanup",
            "comfort food for dinner",
            "vegetarian and plant based options",
            "high protein meals for energy",
            "simple pasta and rice dishes",
            "quick stir fry recipes",
            "oven baked dinner recipes",
            "slow cooker and instant pot meals"
        ]
        return random.choice(queries)

    def _get_default_individual_meals_prompt(self) -> str:
        """Get default prompt for individual meals generation."""
        return """Based on the following meal history and preferences:

{context}

Please generate {num_meals} diverse individual meal options suitable for day-to-day grocery shopping and cooking. 
Each meal should be a complete dish that can be prepared independently.

Focus on:
1. **Practicality**: Meals that are realistic for a weeknight dinner (30-45 mins).
2. **Accessibility**: Ingredients found in a standard grocery store.
3. **Variety**: Mix of cuisines and protein sources, but keep it approachable.

Return ONLY valid JSON in this EXACT format:
```json
[
  {{
    "meal_name": "Thai Green Curry with Chicken",
    "cuisine": "Thai",
    "description": "Creamy coconut curry with chicken and vegetables",
    "ingredients": ["500g chicken thighs", "400ml coconut milk", "2 tbsp green curry paste", "1 cup bamboo shoots", "3 kaffir lime leaves", "1 bunch thai basil"],
    "prep_time_minutes": 30,
    "difficulty": "medium"
  }}
]
```

Important:
- Make meals diverse (different cuisines, proteins, cooking methods)
- Include specific quantities in ingredients
- Keep it practical and based on the user's history
- NO markdown, NO explanations, ONLY JSON
"""

