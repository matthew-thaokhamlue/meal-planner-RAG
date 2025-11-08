# Streamlit Meal Planner Guide

## Overview

The Streamlit app provides a simple, interactive web interface for meal planning and shopping list generation.

## Workflow

### 1. Generate Meal Options
- Choose how many meal options you want to see (5-30 meals)
- Click "Generate Meals" to get diverse meal suggestions from your cooking history
- The app uses RAG to retrieve relevant meals and LLM to generate diverse options

### 2. Select Your Meals
- Browse through the generated meal options
- Each meal card shows:
  - Meal name
  - Cuisine type
  - Difficulty level
  - Preparation time
  - Description
  - Ingredients (expandable)
- Check the boxes to select the meals you want to cook
- You can select as many or as few as you like

### 3. Generate Shopping List
- Once you've selected your meals, click "Generate Shopping List"
- Choose whether to include household items checklist
- The app will consolidate all ingredients from your selected meals

### 4. Copy Shopping List
- The shopping list is displayed in a text area for easy copying
- Click inside the text area and press:
  - **Ctrl+A** (Windows/Linux) or **Cmd+A** (Mac) to select all
  - **Ctrl+C** (Windows/Linux) or **Cmd+C** (Mac) to copy
- You can also download the shopping list as a TXT file

## Running the App

### Start the Streamlit App

```bash
# Activate virtual environment
source venv/bin/activate

# Run the app
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

### First Time Setup

Make sure you have data ingested:

```bash
meal-assistant ingest --all ./data/raw
```

## Features

### AI-Powered Smart Shopping Lists
Uses Claude AI (anthropic/claude-haiku-4.5) to intelligently merge and simplify ingredients, creating shopping lists that are easy to understand and remember.

### No Emojis
The shopping list output is plain text without emojis, making it easy to copy and paste into notes apps, messaging apps, or print.

### Clipboard-Friendly Format
The shopping list is formatted as simple, merged items without quantities or units:

```
SHOPPING LIST
==================================================

Produce:
------------------------------
  [ ] garlic
  [ ] ginger
  [ ] cilantro
  [ ] lime

Proteins:
------------------------------
  [ ] chicken
  [ ] pork
  [ ] eggs

Grains & Pasta:
------------------------------
  [ ] rice
  [ ] noodles

...
```

### Smart AI-Powered Merging
The app uses Claude AI to intelligently merge and simplify ingredients:
- **Smart merging**: "500g chicken thighs" + "300g chicken breast" = `[ ] chicken`
- **Intelligent extraction**: "1 tsp cornstarch mixed with water" = `[ ] cornstarch`
- **Common names**: "jasmine rice" + "day-old rice" = `[ ] rice`
- **No duplicates**: If 3 meals need garlic, it appears once
- **No quantities or units**: Just simple, easy-to-remember items

The AI understands context and creates a shopping list that makes sense to humans!

### Household Items
Optionally include a checklist of common household items to restock:
- Cleaning supplies
- Paper products
- Toiletries
- Kitchen essentials

## Tips

1. **Start with more options**: Generate 15-20 meal options to have good variety
2. **Mix cuisines**: The app generates diverse meals across different cuisines
3. **Simple shopping**: The list shows only unique items (no quantities), making it easy to remember
4. **Save your list**: Use the download button to save the shopping list for later
5. **Merged items**: If multiple meals need the same ingredient, it appears only once in the list

## Troubleshooting

### No meals generated
- Make sure you have ingested data: `meal-assistant ingest --all ./data/raw`
- Check that the RAG database has documents

### App won't start
- Make sure streamlit is installed: `pip install streamlit>=1.28.0`
- Check that you're in the correct directory with `app.py`

### Slow generation
- Generating meals requires LLM API calls, which can take 10-30 seconds
- Be patient and wait for the spinner to complete

