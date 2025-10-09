# RAG-Powered Home Meal & Shopping Assistant - Project Plan

## Project Overview
A CLI-based application that uses RAG (Retrieval-Augmented Generation) to intelligently suggest meal plans and comprehensive shopping lists based on historical cooking data, preferences, and household needs.

## Core Features
1. **Data Ingestion**: Import and process meal history, preferences, bills, and grocery lists
2. **RAG-based Retrieval**: Query relevant context from stored data
3. **AI-Powered Suggestions**: Generate 3 meal plan options with complete shopping lists
4. **Choice Logging**: Track user selections for continuous learning
5. **Comprehensive Lists**: Include both food ingredients and household items

## Technology Stack
- **RAG Framework**: RAG-Anything (https://github.com/HKUDS/RAG-Anything)
- **LLM API**: OpenRouter API with `requests` library
- **Language**: Python 3.8+
- **CLI Framework**: `argparse` or `click`
- **Data Storage**: JSON/SQLite for logging, vector DB for RAG
- **Vector Store**: Based on RAG-Anything's recommendations (likely FAISS or Chroma)

---

## Task List

### Phase 1: Setup & Environment (Estimated: 1-2 days)

#### Task 1.1: Repository Setup
- [ ] Initialize git repository
- [ ] Create project structure:
  ```
  meal-shopping-assistant/
  ├── src/
  │   ├── __init__.py
  │   ├── cli.py
  │   ├── rag_engine.py
  │   ├── llm_client.py
  │   ├── data_ingestion.py
  │   └── utils.py
  ├── data/
  │   ├── raw/
  │   ├── processed/
  │   └── logs/
  ├── config/
  │   └── config.yaml
  ├── requirements.txt
  ├── README.md
  └── .env.example
  ```

#### Task 1.2: Install Dependencies
- [ ] Set up virtual environment
- [ ] Install RAG-Anything and its dependencies
- [ ] Install additional packages: `requests`, `python-dotenv`, `pyyaml`, `click`
- [ ] Create `requirements.txt`

#### Task 1.3: Configuration
- [ ] Create `.env.example` with OpenRouter API key placeholder
- [ ] Create `config.yaml` for app settings (model names, temperature, etc.)
- [ ] Set up logging configuration

---

### Phase 2: RAG-Anything Integration (Estimated: 2-3 days)

#### Task 2.1: Understand RAG-Anything
- [ ] Clone and study RAG-Anything repository
- [ ] Test basic examples from the repo
- [ ] Identify which components to use (likely the core RAG pipeline)
- [ ] Understand vector database integration options

#### Task 2.2: Build RAG Engine Module (`rag_engine.py`)
- [ ] Initialize RAG-Anything with appropriate configuration
- [ ] Implement `initialize_database()` - set up vector store
- [ ] Implement `add_documents(documents, metadata)` - add data to RAG
- [ ] Implement `query(question, top_k=5)` - retrieve relevant context
- [ ] Implement `reset_database()` - clear and reinitialize
- [ ] Add error handling and logging

#### Task 2.3: Test RAG Engine
- [ ] Create test data samples
- [ ] Test document ingestion
- [ ] Test retrieval with sample queries
- [ ] Verify metadata filtering works

---

### Phase 3: Data Ingestion (Estimated: 2-3 days)

#### Task 3.1: Define Data Schema
- [ ] Design JSON schema for:
  - Meal records (name, date, ingredients, servings, notes)
  - User preferences (dietary restrictions, favorites, dislikes)
  - Shopping bills (date, items, prices, store)
  - Grocery lists (historical lists with dates)
  - Household items inventory
- [ ] Create sample data files for testing

#### Task 3.2: Build Data Ingestion Module (`data_ingestion.py`)
- [ ] Implement `load_meals(filepath)` - parse meal data
- [ ] Implement `load_preferences(filepath)` - parse preferences
- [ ] Implement `load_bills(filepath)` - parse shopping bills
- [ ] Implement `load_grocery_lists(filepath)` - parse past lists
- [ ] Implement `process_and_chunk(data, data_type)` - prepare for RAG
- [ ] Implement `ingest_all_data(data_dir)` - batch process all files

#### Task 3.3: Document Formatting
- [ ] Create templates for converting data to text for RAG embedding
  - Example: "On 2024-01-15, cooked Spaghetti Carbonara using eggs, pasta, bacon..."
- [ ] Add metadata tagging (type, date, category)
- [ ] Test with sample data

---

### Phase 4: LLM Integration (Estimated: 2 days)

#### Task 4.1: OpenRouter Client (`llm_client.py`)
- [ ] Implement `OpenRouterClient` class
- [ ] Implement `__init__(api_key, model_name)` - initialize client
- [ ] Implement `generate_response(prompt, temperature, max_tokens)` - API call
- [ ] Add retry logic with exponential backoff
- [ ] Handle API errors gracefully
- [ ] Add token usage logging

#### Task 4.2: Prompt Engineering
- [ ] Create system prompt template for meal planning
- [ ] Create user prompt template with RAG context injection
- [ ] Design output format (structured JSON response)
- [ ] Test prompt with manual RAG context

#### Task 4.3: Response Parser
- [ ] Implement `parse_meal_options(response)` - extract 3 options
- [ ] Validate response format
- [ ] Handle malformed responses

---

### Phase 5: Core Logic (Estimated: 3-4 days)

#### Task 5.1: Meal Plan Generator
- [ ] Implement `generate_meal_plans(query, num_days, occasion)`:
  - Parse user query (e.g., "3 days", "weekend", "party")
  - Retrieve relevant context from RAG
  - Build comprehensive prompt
  - Call LLM API
  - Parse response into 3 options
- [ ] Add household items intelligence:
  - Analyze past bills for recurring household items
  - Include items based on time since last purchase

#### Task 5.2: Shopping List Aggregator
- [ ] Implement `aggregate_shopping_list(meal_plan, include_household=True)`
- [ ] Combine ingredients from multiple meals
- [ ] Remove duplicates and consolidate quantities
- [ ] Add suggested household items based on history
- [ ] Categorize items (produce, dairy, pantry, household)

#### Task 5.3: Choice Logging
- [ ] Implement `log_user_choice(option_selected, query_context, timestamp)`
- [ ] Store in SQLite/JSON format
- [ ] Include feedback mechanism for future RAG ingestion
- [ ] Implement `get_choice_history()` for analysis

---

### Phase 6: CLI Interface (Estimated: 2-3 days)

#### Task 6.1: CLI Structure (`cli.py`)
- [ ] Set up Click/argparse command structure
- [ ] Implement main command: `meal-assistant`
- [ ] Add subcommands:
  - `ingest` - add new data
  - `plan` - generate meal plans
  - `history` - view past choices
  - `reset` - clear database

#### Task 6.2: Implement `ingest` Command
- [ ] `meal-assistant ingest --meals <file>`
- [ ] `meal-assistant ingest --preferences <file>`
- [ ] `meal-assistant ingest --bills <file>`
- [ ] `meal-assistant ingest --grocery-lists <file>`
- [ ] `meal-assistant ingest --all <directory>` - batch ingest
- [ ] Add progress indicators
- [ ] Display summary after ingestion

#### Task 6.3: Implement `plan` Command
- [ ] `meal-assistant plan "what should I shop for 3 days"`
- [ ] Display 3 options with formatting:
  ```
  Option 1: Mediterranean Week
  ├── Meals:
  │   ├── Monday: Greek Salad
  │   ├── Tuesday: Chicken Souvlaki
  │   └── Wednesday: Spanakopita
  └── Shopping List:
      ├── Produce: tomatoes, cucumbers...
      ├── Protein: chicken breast...
      └── Household: dish soap, paper towels...
  ```
- [ ] Interactive selection: "Choose option [1/2/3]:"
- [ ] Log choice after selection
- [ ] Option to export to file

#### Task 6.4: Implement `history` Command
- [ ] `meal-assistant history --last 10` - show recent choices
- [ ] Display formatted history with dates
- [ ] Add filtering options (by date range, meal type)

#### Task 6.5: Polish CLI UX
- [ ] Add colorized output (using `colorama` or `rich`)
- [ ] Add ASCII art/banner
- [ ] Implement `--verbose` flag for debugging
- [ ] Add help text and examples

---

### Phase 7: Testing & Refinement (Estimated: 2-3 days)

#### Task 7.1: Unit Testing
- [ ] Write tests for RAG engine
- [ ] Write tests for data ingestion
- [ ] Write tests for LLM client
- [ ] Write tests for utilities

#### Task 7.2: Integration Testing
- [ ] Test end-to-end workflow with sample data
- [ ] Test with various query types:
  - "3 days"
  - "weekend party for 10 people"
  - "quick weeknight dinners"
  - "vegetarian meals"
- [ ] Test household item suggestions
- [ ] Test choice logging and retrieval

#### Task 7.3: Error Handling
- [ ] Handle missing API key gracefully
- [ ] Handle empty RAG database
- [ ] Handle API rate limits
- [ ] Handle malformed input data
- [ ] Add user-friendly error messages

---

### Phase 8: Documentation & Deployment (Estimated: 1-2 days)

#### Task 8.1: User Documentation
- [ ] Write comprehensive README with:
  - Installation instructions
  - Configuration guide
  - Usage examples
  - Data format specifications
  - Troubleshooting guide
- [ ] Create sample data files
- [ ] Add inline code documentation

#### Task 8.2: Developer Documentation
- [ ] Document architecture and design decisions
- [ ] Add code comments
- [ ] Create API documentation for modules
- [ ] Document RAG-Anything integration approach

#### Task 8.3: Packaging
- [ ] Make package installable with `pip install -e .`
- [ ] Add entry point for CLI command
- [ ] Test installation in fresh environment
- [ ] Create release checklist

---

## Optional Enhancements (Future Phases)

### Phase 9: Advanced Features
- [ ] **Budget awareness**: Factor in budget constraints from past bills
- [ ] **Seasonal suggestions**: Prefer seasonal ingredients
- [ ] **Nutrition tracking**: Include nutritional information
- [ ] **Recipe links**: Provide recipe URLs or instructions
- [ ] **Smart notifications**: Remind when to shop based on inventory
- [ ] **Multi-user support**: Handle multiple household members' preferences
- [ ] **Meal prep mode**: Optimize for batch cooking
- [ ] **Leftover tracking**: Suggest meals using leftovers

### Phase 10: UI Improvements
- [ ] **Web interface**: Optional web UI for easier data entry
- [ ] **Mobile app**: Companion mobile app for shopping lists
- [ ] **Voice interface**: Voice command support
- [ ] **Export formats**: PDF, Google Keep, Todoist integration

---

## Estimated Timeline
- **Minimum Viable Product**: 2-3 weeks (Phases 1-6)
- **Production Ready**: 3-4 weeks (includes Phases 7-8)
- **With Enhancements**: 6-8 weeks (includes optional phases)

## Success Metrics
1. Successfully ingest diverse data formats
2. Generate relevant, personalized meal suggestions
3. Include appropriate household items (not just food)
4. Learn from user choices over time
5. Response time under 10 seconds for plan generation

---

## Getting Started: First Steps
1. Fork/clone RAG-Anything repository
2. Set up development environment
3. Get OpenRouter API key
4. Create sample dataset (10 meals, 5 bills, preferences)
5. Build and test RAG engine with samples
6. Iterate from there!