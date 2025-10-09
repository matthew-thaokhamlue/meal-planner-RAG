# Usage Examples

This document provides detailed examples of how to use the RAG-Powered Home Meal & Shopping Assistant.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Adding Data Interactively](#adding-data-interactively)
3. [Data Ingestion](#data-ingestion)
4. [Meal Planning](#meal-planning)
5. [History Management](#history-management)
6. [Database Management](#database-management)
7. [Advanced Usage](#advanced-usage)

## Getting Started

### First Time Setup

```bash
# 1. Install the package
pip install -e .

# 2. Set up your API key
cp .env.example .env
# Edit .env and add: OPENROUTER_API_KEY=your_key_here

# 3. Prepare your data files in data/raw/
# - meals.json
# - preferences.json
# - bills.json
# - grocery_lists.json

# 4. Ingest your data
meal-assistant ingest --all ./data/raw

# 5. Verify data was loaded
meal-assistant stats
```

## Adding Data Interactively

The easiest way to add new data is through the interactive conversational interface. The system will guide you through adding meals, bills, and grocery lists step by step.

### Add a New Meal

```bash
meal-assistant add menu
```

The system will ask you:
- Meal name
- Cuisine type
- Ingredients (one by one)
- Servings
- Prep time
- Difficulty level
- Rating
- Notes

**Example interaction:**
```
🍽️  Let's add a new meal!

What's the name of the meal?: Pad Thai
What cuisine is it? (Asian): Thai
Let's add ingredients. Enter them one by one (press Enter with empty line when done):
  Ingredient 1 (): Rice noodles
  Ingredient 2 (): Shrimp
  Ingredient 3 (): Bean sprouts
  Ingredient 4 (): Peanuts
  Ingredient 5 (): Lime
  Ingredient 6 (): Eggs
  Ingredient 7 ():
How many servings? (4): 2
Prep time in minutes? (45): 30
Difficulty level? [easy/medium/hard] (medium): easy
Rating (1-5)? (4): 5
Any notes? (): Quick and delicious Thai street food

📋 Meal Summary:
  Name: Pad Thai
  Cuisine: Thai
  Servings: 2
  Ingredients: 6 items
  Prep time: 30 minutes
  Difficulty: easy
  Rating: 5/5
  Notes: Quick and delicious Thai street food

Save this meal? [y/n] (y): y
✓ Meal saved to data/raw/meals.json
Add to RAG database now? [y/n] (y): y
✓ Added to RAG database!
```

### Add a Shopping Bill

```bash
meal-assistant add bill
```

The system will ask you:
- Store name
- Date
- Currency
- Items (name, quantity, unit, price, category)

**Example interaction:**
```
🧾 Let's add a shopping bill!

Which store did you shop at?: Asian Market
Date (YYYY-MM-DD)? (2025-10-08):
Currency? (USD):

Let's add items. Enter them one by one (press Enter with empty name when done):

Item 1:
  Name (): Fish sauce
  Quantity (1): 1
  Unit (item): bottle
  Price (0): 4.99
  Category [produce/protein/dairy/grains_pasta/pantry/household] (pantry): pantry

Item 2:
  Name (): Lemongrass
  Quantity (1): 3
  Unit (item): stalks
  Price (0): 2.99
  Category [produce/protein/dairy/grains_pasta/pantry/household] (pantry): produce

Item 3:
  Name ():

📋 Bill Summary:
  Store: Asian Market
  Date: 2025-10-08
  Items: 2
  Total: USD 7.98

Save this bill? [y/n] (y): y
✓ Bill saved to data/raw/bills.json
Add to RAG database now? [y/n] (y): y
✓ Added to RAG database!
```

### Add a Grocery List

```bash
meal-assistant add grocery
```

The system will ask you:
- Occasion
- Date
- Items (one by one)
- Notes
- Completion status

**Example interaction:**
```
📝 Let's add a grocery list!

What's the occasion? (weekly shopping): Weekend party prep
Date (YYYY-MM-DD)? (2025-10-08):
Enter items one by one (press Enter with empty line when done):
  Item 1 (): Chicken wings
  Item 2 (): BBQ sauce
  Item 3 (): Potato chips
  Item 4 (): Soda
  Item 5 ():
Any notes? (): Need to buy early Saturday morning
Is this list completed? [y/n] (n): n

📋 Grocery List Summary:
  Occasion: Weekend party prep
  Date: 2025-10-08
  Items: 4
  Completed: No
  Notes: Need to buy early Saturday morning

Save this grocery list? [y/n] (y): y
✓ Grocery list saved to data/raw/grocery_lists.json
Add to RAG database now? [y/n] (y): y
✓ Added to RAG database!
```

### Benefits of Interactive Adding

1. **No JSON editing required** - The system handles the file format for you
2. **Automatic validation** - Ensures data is in the correct format
3. **Immediate RAG integration** - Optionally add to database right away
4. **Conversational** - Natural question-and-answer flow
5. **Error-free** - No syntax errors or formatting issues

## Data Ingestion

### Ingest All Data at Once

```bash
# Ingest all data files from a directory
meal-assistant ingest --all ./data/raw
```

Output:
```
🔄 Starting data ingestion...

Ingesting all data from: ./data/raw
      Ingestion Summary      
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Data Type     ┃ Documents ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ Meals         │ 5         │
│ Preferences   │ 14        │
│ Bills         │ 7         │
│ Grocery Lists │ 2         │
└───────────────┴───────────┘

✓ Ingestion complete!
Total documents in database: 28
```

### Ingest Individual Files

```bash
# Ingest only meals
meal-assistant ingest --meals ./data/raw/meals.json

# Ingest only preferences
meal-assistant ingest --preferences ./data/raw/preferences.json

# Ingest multiple specific files
meal-assistant ingest --meals ./data/raw/meals.json --bills ./data/raw/bills.json
```

## Meal Planning

### Basic Meal Planning

```bash
# Simple 3-day plan
meal-assistant plan "3 days of meals"

# Weekend planning
meal-assistant plan "weekend meals"

# Week-long planning
meal-assistant plan "meals for a week"
```

### Planning for Special Occasions

```bash
# Party planning
meal-assistant plan "party for 15 people"

# Holiday meal
meal-assistant plan "Thanksgiving dinner for 8"

# Date night
meal-assistant plan "romantic dinner for 2"
```

### Planning with Dietary Preferences

```bash
# Vegetarian meals
meal-assistant plan "5 days of vegetarian meals"

# Quick weeknight dinners
meal-assistant plan "3 quick weeknight dinners under 30 minutes"

# Budget-friendly meals
meal-assistant plan "budget meals for a week"
```

### Example Output

```
🍽️  Generating meal plans for: '3 days of meals'

✓ Generated 3 meal plan options!

Option 1: Global Comfort Classics
Familiar favorites with international flair, balanced for variety and ease

📅 Meals:
  • Monday: Mediterranean Chickpea Bowl
    Meatless Monday bowl with roasted vegetables and tahini dressing
  • Tuesday: Asian-Style Salmon Stir-Fry
    Quick weeknight dinner inspired by previous successful meals
  • Wednesday: Lighter Carbonara
    Modified version of family favorite with added vegetables

🛒 Shopping List:

🥬 Produce:
  • 2 large sweet potatoes
  • 3 whole bell peppers
  • 2 heads broccoli
  ...

💰 Estimated cost: $85-95
👨‍🍳 Difficulty: medium

────────────────────────────────────────────────────────────

[Options 2 and 3 displayed similarly...]

Which option would you like to choose? [1/2/3] (1): 1

✓ Logged your choice: Global Comfort Classics
```

### Export Shopping List

```bash
# Export to text file
meal-assistant plan "3 days" --export shopping_list.txt

# Export to JSON
meal-assistant plan "weekend" --export shopping_list.json
```

## History Management

### View Recent Choices

```bash
# View last 10 choices
meal-assistant history --last 10

# View last 5 choices
meal-assistant history --last 5
```

Example output:
```
📜 Your Recent Choices (last 5)

1. 2025-10-07T09:20:00.535664
   Query: 3 days of meals
   Selected: Global Comfort Classics

2. 2025-10-06T18:30:15.123456
   Query: weekend meals
   Selected: Quick & Fresh Favorites

...
```

### View All History

```bash
# View all recorded choices
meal-assistant history
```

## Database Management

### View Database Statistics

```bash
meal-assistant stats
```

Output:
```
📊 Database Statistics

    RAG Database Stats     
┏━━━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ Metric          ┃ Value ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━┩
│ Total Documents │ 28    │
│ Index Size      │ 28    │
└─────────────────┴───────┘

Documents by Type:
  • meal: 5
  • preference: 14
  • bill: 3
  • household_item: 4
  • grocery_list: 2
```

### Reset Database

```bash
meal-assistant reset
```

Output:
```
⚠️  Warning: This will delete all ingested data!

Are you sure you want to reset the database? [y/N]: y

✓ Database reset successfully!
```

## Advanced Usage

### Custom Configuration

Edit `config/config.yaml` to customize:

```yaml
llm:
  default_model: "anthropic/claude-3.5-sonnet"
  temperature: 0.7
  max_tokens: 4000  # Increase for longer responses
  timeout: 60

rag:
  top_k: 15  # Number of documents to retrieve
  
household_items:
  restock_interval_days:
    cleaning_supplies: 30
    paper_products: 14
```

### Custom Prompts

Edit `config/prompts.yaml` to customize LLM behavior:

```yaml
system_prompt: |
  You are a helpful meal planning assistant...
  [Customize the system prompt]

meal_planning_prompt: |
  Based on the following context...
  [Customize the meal planning prompt]
```

### Programmatic Usage

You can also use the components programmatically:

```python
from src.rag_engine import RAGEngine
from src.llm_client import OpenRouterClient
from src.meal_planner import MealPlanner

# Initialize components
rag_engine = RAGEngine()
llm_client = OpenRouterClient()
planner = MealPlanner(rag_engine, llm_client)

# Generate meal plans
plans = planner.generate_meal_plans("3 days of meals")

# Access the plans
for plan in plans:
    print(f"Option: {plan['option_name']}")
    print(f"Meals: {len(plan['meals'])}")
    print(f"Shopping list: {plan['shopping_list']}")
```

## Tips and Best Practices

1. **Use Interactive Adding**: Use `meal-assistant add menu/bill/grocery` for the easiest way to add data
2. **Regular Updates**: Update your data files regularly with new meals and shopping trips
3. **Detailed Preferences**: The more detailed your preferences, the better the recommendations
4. **Rate Your Meals**: Include ratings in your meal history to help the system learn
5. **Track Household Items**: Include household items in your bills for automatic restocking suggestions
6. **Use Natural Language**: The system understands natural queries like "quick weeknight dinners"
7. **Review History**: Check your history to see patterns in your choices
8. **Add Immediately to RAG**: When adding data interactively, choose "yes" to add to RAG database immediately

## Troubleshooting

### No Meal Plans Generated

If you get "No Data Available":
1. Check that data was ingested: `meal-assistant stats`
2. Verify your data files are in correct JSON format
3. Re-ingest data: `meal-assistant ingest --all ./data/raw`

### API Errors

If you get API errors:
1. Check your internet connection
2. Verify your OpenRouter API key in `.env`
3. Check the logs: `cat data/logs/app.log`

### Slow Performance

If the system is slow:
1. Reduce `top_k` in `config/config.yaml` (fewer documents retrieved)
2. Reduce `max_tokens` for faster LLM responses
3. Use a faster embedding model (though less accurate)

## Next Steps

- Experiment with different query styles
- Build up your meal history over time
- Customize prompts for your specific needs
- Export and share shopping lists with family members

