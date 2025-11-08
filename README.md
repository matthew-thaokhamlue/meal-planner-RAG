# RAG-Powered Home Meal & Shopping Assistant

A meal planning and shopping list application with both **Web UI (Streamlit)** and **CLI** interfaces, powered by RAG (Retrieval-Augmented Generation) and AI.

## Features

- **Web UI (Streamlit)**: Interactive meal selection and shopping list generation
- **Personalized Meal Planning**: Get diverse meal options based on your cooking history
- **Smart Shopping Lists**: Consolidated ingredients with automatic quantity calculation
- **RAG-Powered**: Uses FAISS and sentence transformers for intelligent context retrieval
- **Clipboard-Friendly**: Plain text output (no emojis) for easy copying
- **Household Items**: Optional checklist of common household items to restock

## Quick Start (Web UI - Recommended)

1. **Install and setup** (see Installation section below)

2. **Ingest your data**:
```bash
meal-assistant ingest --all ./data/raw
```

3. **Run the Streamlit app**:
```bash
streamlit run app.py
```

4. **Use the app**:
   - Choose how many meal options to generate (e.g., 15 meals)
   - Select the meals you want to cook (e.g., 3 meals)
   - Generate shopping list for selected meals
   - Copy the plain text shopping list to your clipboard

See [STREAMLIT_GUIDE.md](STREAMLIT_GUIDE.md) for detailed instructions.

## Setup

### Prerequisites

- Python 3.8 or higher
- OpenRouter API key ([Get one here](https://openrouter.ai/))

### Installation

1. Clone the repository:
```bash
cd meal-shopping-assistant
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your OpenRouter API key
```

5. Install the package:
```bash
pip install -e .
```

## Usage

### Add Data Interactively (Recommended)

The easiest way to add data is through the interactive interface. You can choose between:
- **Paste mode** - Copy/paste text from anywhere (recipes, receipts, lists) and let AI parse it
- **Interactive mode** - Answer questions one by one

```bash
# Add a new meal
meal-assistant add menu

# Add a shopping bill
meal-assistant add bill

# Add a grocery list
meal-assistant add grocery
```

**Paste Mode Example:**
```bash
$ meal-assistant add menu
How would you like to add the meal? [paste/interactive]: paste
[Paste your recipe in ANY language - Thai, Lao, Vietnamese, etc.]
[Press Ctrl+D]
✓ AI parses and translates to English automatically!
```

**🌍 Automatic Translation:** Paste mode automatically translates all text to English, ensuring consistent data and better RAG performance!

The system will guide you through the process, then automatically save and optionally add to the RAG database.

### Ingest Data (Bulk Import)

If you have existing data files, you can ingest them in bulk:

```bash
# Ingest all data from a directory
meal-assistant ingest --all ./data/raw

# Or ingest specific data types
meal-assistant ingest --meals ./data/raw/meals.json
meal-assistant ingest --preferences ./data/raw/preferences.json
meal-assistant ingest --bills ./data/raw/bills.json
meal-assistant ingest --grocery-lists ./data/raw/grocery_lists.json
```

### Generate Meal Plans

Ask for meal plan suggestions:

```bash
# Get meal plans for 3 days
meal-assistant plan "what should I shop for 3 days"

# Weekend meal planning
meal-assistant plan "weekend meals"

# Party planning
meal-assistant plan "party for 15 people"

# Specific dietary needs
meal-assistant plan "vegetarian meals for a week"
```

### View History

See your past choices:

```bash
# View recent choices
meal-assistant history --last 10

# View all history
meal-assistant history
```

### Reset Database

Clear all data and start fresh:

```bash
meal-assistant reset
```

## 🚀 Quick Reference

```bash
# Interactive data entry (easiest way)
meal-assistant add menu          # Add a meal conversationally
meal-assistant add bill          # Add a shopping bill
meal-assistant add grocery       # Add a grocery list

# Bulk data import
meal-assistant ingest --all ./data/raw

# Generate meal plans
meal-assistant plan "3 days of meals"
meal-assistant plan "weekend Thai food"

# View history and stats
meal-assistant history --last 10
meal-assistant stats

# Database management
meal-assistant reset
```

## Data Format

See the `tests/fixtures/` directory for complete example data formats.

### Meals Format (`meals.json`)

```json
[
  {
    "date": "2024-01-15",
    "meal_name": "Spaghetti Carbonara",
    "cuisine": "Italian",
    "servings": 4,
    "ingredients": ["400g spaghetti", "200g pancetta", "4 eggs"],
    "prep_time_minutes": 30,
    "difficulty": "medium",
    "notes": "Family loved it!",
    "rating": 5
  }
]
```

### Preferences Format (`preferences.json`)

```json
{
  "dietary_restrictions": ["no shellfish"],
  "favorite_cuisines": ["Italian", "Mexican", "Asian"],
  "disliked_foods": ["liver"],
  "preferred_proteins": ["chicken", "beef", "tofu"],
  "household_size": 4,
  "budget": {
    "weekly": "150-200",
    "currency": "USD"
  }
}
```

### Bills Format (`bills.json`)

```json
[
  {
    "date": "2024-01-20",
    "store": "Whole Foods",
    "total_amount": 87.50,
    "currency": "USD",
    "items": [
      {
        "name": "chicken breast",
        "quantity": "2",
        "unit": "lbs",
        "price": 12.99,
        "category": "protein"
      }
    ]
  }
]
```

### Grocery Lists Format (`grocery_lists.json`)

```json
[
  {
    "date": "2024-01-15",
    "occasion": "weekly shopping",
    "items": ["milk", "eggs", "bread"],
    "completed": true,
    "notes": "Forgot paper towels"
  }
]
```

## Development

### Run Tests

```bash
pytest tests/ -v
```

### Run Tests with Coverage

```bash
pytest tests/ --cov=src --cov-report=html
```

## Project Structure

```
meal-shopping-assistant/
├── src/                    # Source code
│   ├── cli.py             # CLI interface
│   ├── rag_engine.py      # RAG implementation
│   ├── llm_client.py      # OpenRouter API client
│   ├── data_ingestion.py  # Data processing
│   ├── meal_planner.py    # Meal planning logic
│   └── utils.py           # Utility functions
├── data/                   # Data storage
│   ├── raw/               # Input data
│   ├── processed/         # Vector database
│   └── logs/              # Application logs
├── config/                 # Configuration files
│   ├── config.yaml        # App settings
│   └── prompts.yaml       # LLM prompts
└── tests/                  # Test suite
```

## Troubleshooting

### "OPENROUTER_API_KEY not found"
- Make sure you've created a `.env` file with your API key
- Check that the key is named `OPENROUTER_API_KEY` (with underscore before KEY)

### "No data in RAG database"
- Run `meal-assistant ingest --all ./data/raw` to load your data
- Check that your data files are in the correct JSON format
- Use `meal-assistant stats` to verify data was loaded

### LLM Response Errors
- Check your internet connection
- Verify your OpenRouter API key is valid
- Check the logs in `data/logs/app.log` for detailed error messages
- Try increasing `max_tokens` in `config/config.yaml` if responses are truncated

### Database Issues
- Try resetting the database: `meal-assistant reset`
- Re-ingest your data: `meal-assistant ingest --all ./data/raw`
- Check file permissions in the `data/processed/` directory

## How It Works

1. **Data Ingestion**: Your meal history, preferences, and shopping data are processed and stored in a FAISS vector database using sentence-transformers embeddings
2. **Query Processing**: When you request a meal plan, the system parses your query to extract key information (duration, occasion, number of people)
3. **Context Retrieval**: The RAG engine performs semantic search to find the most relevant information from your history
4. **LLM Generation**: The retrieved context is sent to Claude 3.5 Sonnet via OpenRouter to generate personalized meal plans
5. **Choice Logging**: Your selections are logged to improve future recommendations

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📚 Documentation

### Quick References
- **[PASTE_MODE_GUIDE.md](PASTE_MODE_GUIDE.md)** - 🆕 Paste mode guide (copy/paste recipes, receipts, lists)
- **[TRANSLATION_FEATURE.md](TRANSLATION_FEATURE.md)** - 🌍 Automatic translation (paste in any language!)
- **[COMMAND_REFERENCE.md](COMMAND_REFERENCE.md)** - Complete command reference
- **[USAGE_EXAMPLES.md](USAGE_EXAMPLES.md)** - Detailed usage examples
- **[INTERACTIVE_DEMO.md](INTERACTIVE_DEMO.md)** - Interactive feature demonstrations

### Technical Documentation
- **[FEATURE_SUMMARY.md](FEATURE_SUMMARY.md)** - Technical implementation details
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - Project overview
- **[agents.md](agents.md)** - Agent architecture
- **[project_plan.md](project_plan.md)** - Original project plan

## Acknowledgments

- Built with [FAISS](https://github.com/facebookresearch/faiss) for vector similarity search
- Uses [sentence-transformers](https://www.sbert.net/) for text embeddings
- Powered by [OpenRouter](https://openrouter.ai/) for LLM access
- CLI built with [Click](https://click.palletsprojects.com/) and [Rich](https://rich.readthedocs.io/)

