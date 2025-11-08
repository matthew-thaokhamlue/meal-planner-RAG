"""Streamlit app for meal planning and shopping list generation."""

import streamlit as st
import logging
from typing import List, Dict, Any

from src.simple_meal_planner import SimpleMealPlanner
from src.utils import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Meal Planner & Shopping List",
    page_icon="🍽",
    layout="wide"
)

# Initialize session state
if 'meal_options' not in st.session_state:
    st.session_state.meal_options = []
if 'selected_meals' not in st.session_state:
    st.session_state.selected_meals = []
if 'shopping_list' not in st.session_state:
    st.session_state.shopping_list = {}


def format_shopping_list_text(shopping_list: Dict[str, Any]) -> str:
    """Format shopping list as plain text for clipboard."""
    lines = []
    lines.append("SHOPPING LIST")
    lines.append("=" * 50)
    lines.append("")

    # Category display names
    category_names = {
        'produce': 'Produce',
        'proteins': 'Proteins',
        'dairy': 'Dairy',
        'grains_pasta': 'Grains & Pasta',
        'pantry': 'Pantry Items',
        'household_checklist': 'Household Items Checklist'
    }

    for category, items in shopping_list.items():
        if not items:
            continue

        display_name = category_names.get(category, category.replace('_', ' ').title())
        lines.append(f"{display_name}:")
        lines.append("-" * 30)

        for item in items:
            # Simple format: just item name
            if isinstance(item, str):
                lines.append(f"  [ ] {item}")
            else:
                # Fallback for dict format (shouldn't happen with new code)
                item_name = item.get('item', str(item))
                lines.append(f"  [ ] {item_name}")

        lines.append("")

    return '\n'.join(lines)


def main():
    """Main Streamlit app."""
    
    st.title("Meal Planner & Shopping List Generator")
    st.markdown("---")
    
    # Initialize planner
    if 'planner' not in st.session_state:
        with st.spinner("Initializing meal planner..."):
            st.session_state.planner = SimpleMealPlanner()
    
    planner = st.session_state.planner
    
    # Step 1: Generate meal options
    st.header("Step 1: Generate Meal Options")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        num_meals = st.slider(
            "How many meal options do you want to see?",
            min_value=5,
            max_value=30,
            value=15,
            step=1
        )
    
    with col2:
        if st.button("Generate Meals", type="primary", use_container_width=True):
            with st.spinner(f"Generating {num_meals} meal options..."):
                st.session_state.meal_options = planner.generate_meal_options(
                    num_meals=num_meals,
                    diversity_factor=0.6
                )
                st.session_state.selected_meals = []
                st.session_state.shopping_list = {}
            
            if st.session_state.meal_options:
                st.success(f"Generated {len(st.session_state.meal_options)} meal options!")
            else:
                st.error("Failed to generate meals. Please check if data is ingested.")
    
    # Step 2: Display and select meals
    if st.session_state.meal_options:
        st.markdown("---")
        st.header("Step 2: Select Your Meals")
        
        st.info(f"Select the meals you want to cook. You can choose as many as you like from the {len(st.session_state.meal_options)} options below.")
        
        # Create a grid of meal cards
        cols_per_row = 3
        meal_options = st.session_state.meal_options
        
        for i in range(0, len(meal_options), cols_per_row):
            cols = st.columns(cols_per_row)
            
            for j, col in enumerate(cols):
                idx = i + j
                if idx >= len(meal_options):
                    break
                
                meal = meal_options[idx]
                
                with col:
                    # Create a card-like container
                    with st.container():
                        # Checkbox for selection
                        is_selected = st.checkbox(
                            f"Select",
                            key=f"meal_{idx}",
                            value=meal in st.session_state.selected_meals
                        )
                        
                        # Update selected meals
                        if is_selected and meal not in st.session_state.selected_meals:
                            st.session_state.selected_meals.append(meal)
                        elif not is_selected and meal in st.session_state.selected_meals:
                            st.session_state.selected_meals.remove(meal)
                        
                        # Meal details
                        st.subheader(meal.get('meal_name', 'Unknown Meal'))
                        st.caption(f"{meal.get('cuisine', 'Unknown')} | {meal.get('difficulty', 'medium')} | {meal.get('prep_time_minutes', 30)} min")
                        st.write(meal.get('description', ''))
                        
                        # Ingredients expander
                        with st.expander("Ingredients"):
                            ingredients = meal.get('ingredients', [])
                            for ingredient in ingredients:
                                st.write(f"- {ingredient}")
        
        # Show selected count
        st.markdown("---")
        st.subheader(f"Selected: {len(st.session_state.selected_meals)} meals")
        
        # Step 3: Generate shopping list
        if st.session_state.selected_meals:
            st.markdown("---")
            st.header("Step 3: Generate Shopping List")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                include_household = st.checkbox(
                    "Include household items checklist",
                    value=True
                )
            
            with col2:
                if st.button("Generate Shopping List", type="primary", use_container_width=True):
                    with st.spinner("Calculating shopping list..."):
                        st.session_state.shopping_list = planner.calculate_shopping_list(
                            st.session_state.selected_meals,
                            include_household_items=include_household
                        )
                    
                    if st.session_state.shopping_list:
                        st.success("Shopping list generated!")
    
    # Step 4: Display shopping list
    if st.session_state.shopping_list:
        st.markdown("---")
        st.header("Your Shopping List")
        
        # Format for display and clipboard
        shopping_text = format_shopping_list_text(st.session_state.shopping_list)
        
        # Display in a text area for easy copying
        st.text_area(
            "Copy this shopping list:",
            value=shopping_text,
            height=400,
            help="Click inside and press Ctrl+A (Cmd+A on Mac) to select all, then Ctrl+C (Cmd+C) to copy"
        )
        
        # Also show formatted version
        with st.expander("View formatted shopping list"):
            category_names = {
                'produce': 'Produce',
                'proteins': 'Proteins',
                'dairy': 'Dairy',
                'grains_pasta': 'Grains & Pasta',
                'pantry': 'Pantry Items',
                'household_checklist': 'Household Items Checklist'
            }

            for category, items in st.session_state.shopping_list.items():
                if not items:
                    continue

                display_name = category_names.get(category, category.replace('_', ' ').title())
                st.subheader(display_name)

                for item in items:
                    # Simple format: just item name
                    if isinstance(item, str):
                        st.write(f"- {item}")
                    else:
                        # Fallback for dict format
                        item_name = item.get('item', str(item))
                        st.write(f"- {item_name}")
        
        # Download button
        st.download_button(
            label="Download Shopping List (TXT)",
            data=shopping_text,
            file_name="shopping_list.txt",
            mime="text/plain"
        )


if __name__ == "__main__":
    main()

