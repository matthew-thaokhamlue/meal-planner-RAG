"""Interactive conversational interface for adding data."""

import os
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.markdown import Markdown
import json

from src.llm_client import OpenRouterClient
from src.utils import load_json, save_json, load_config
from src.rag_engine import RAGEngine
from src.data_ingestion import DataIngestion

logger = logging.getLogger(__name__)
console = Console()

class InteractiveDataAdder:
    """Conversational interface for adding data."""

    def __init__(self):
        """Initialize the interactive data adder."""
        self.llm_client = OpenRouterClient()
        self.rag_engine = RAGEngine()
        self.data_ingestion = DataIngestion()

        # Initialize a cheap LLM client for parsing
        # Only use parse_model from config if OPENROUTER_MODEL env var is not set
        config = load_config()
        parse_model = None
        if not os.getenv('OPENROUTER_MODEL'):
            parse_model = config.get('llm', {}).get('parse_model', 'openai/gpt-4o-mini')

        self.parse_llm = OpenRouterClient(model=parse_model)

    def _parse_meal_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse meal information from unstructured text using LLM."""
        prompt = f"""Extract meal information from the following text and return ONLY a valid JSON object with these fields:
- meal_name (string, in English)
- cuisine (string, in English)
- servings (integer, default 4)
- ingredients (array of strings, in English)
- prep_time_minutes (integer, default 45)
- difficulty (string: "easy", "medium", or "hard", default "medium")
- notes (string, optional, in English)
- rating (integer 1-5, default 4)

IMPORTANT: Translate ALL text to English. If the input is in another language (Thai, Lao, Vietnamese, etc.), translate meal names, ingredients, and notes to English.

Text to parse:
{text}

Return ONLY the JSON object with all text in English, no other text."""

        try:
            response_data = self.parse_llm.generate_response(prompt, temperature=0.3, max_tokens=2000)
            # Try to extract JSON from response
            response_text = response_data['content'].strip()

            # Remove markdown code blocks if present
            if response_text.startswith('```'):
                lines = response_text.split('\n')
                response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
                if response_text.startswith('json'):
                    response_text = response_text[4:].strip()

            meal_data = json.loads(response_text)

            # Add today's date
            meal_data['date'] = datetime.now().strftime("%Y-%m-%d")

            return meal_data
        except Exception as e:
            logger.error(f"Failed to parse meal from text: {e}")
            console.print(f"[yellow]⚠️  Could not parse meal data: {e}[/yellow]")
            return None

    def _parse_bill_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse bill information from unstructured text using LLM."""
        prompt = f"""Extract shopping bill information from the following text and return ONLY a valid JSON object with these fields:
- store (string, in English)
- date (string in YYYY-MM-DD format, use today if not specified)
- currency (string, default "USD")
- items (array of objects with: name, quantity, unit, price, category)
  - name: item name in English
  - category must be one of: "produce", "protein", "dairy", "grains_pasta", "pantry", "household"

IMPORTANT: Translate ALL item names to English. If the input is in another language (Thai, Lao, Vietnamese, etc.), translate all item names to English.

Text to parse:
{text}

Return ONLY the JSON object with all text in English, no other text. Calculate total_amount from items."""

        try:
            response_data = self.parse_llm.generate_response(prompt, temperature=0.3, max_tokens=3000)
            response_text = response_data['content'].strip()

            # Remove markdown code blocks if present
            if response_text.startswith('```'):
                lines = response_text.split('\n')
                response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
                if response_text.startswith('json'):
                    response_text = response_text[4:].strip()

            bill_data = json.loads(response_text)

            # Calculate total if not present
            if 'total_amount' not in bill_data and 'items' in bill_data:
                bill_data['total_amount'] = sum(item.get('price', 0) for item in bill_data['items'])

            # Ensure date is present
            if 'date' not in bill_data:
                bill_data['date'] = datetime.now().strftime("%Y-%m-%d")

            return bill_data
        except Exception as e:
            logger.error(f"Failed to parse bill from text: {e}")
            console.print(f"[yellow]⚠️  Could not parse bill data: {e}[/yellow]")
            return None

    def _parse_grocery_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse grocery list from unstructured text using LLM."""
        prompt = f"""Extract grocery list information from the following text and return ONLY a valid JSON object with these fields:
- occasion (string, default "shopping", in English)
- date (string in YYYY-MM-DD format, use today if not specified)
- items (array of strings, in English)
- notes (string, optional, in English)
- completed (boolean, default false)

IMPORTANT: Translate ALL text to English. If the input is in another language (Thai, Lao, Vietnamese, etc.), translate all item names, occasion, and notes to English.

Text to parse:
{text}

Return ONLY the JSON object with all text in English, no other text."""

        try:
            response_data = self.parse_llm.generate_response(prompt, temperature=0.3, max_tokens=2000)
            response_text = response_data['content'].strip()

            # Remove markdown code blocks if present
            if response_text.startswith('```'):
                lines = response_text.split('\n')
                response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
                if response_text.startswith('json'):
                    response_text = response_text[4:].strip()

            grocery_data = json.loads(response_text)

            # Ensure date is present
            if 'date' not in grocery_data:
                grocery_data['date'] = datetime.now().strftime("%Y-%m-%d")

            return grocery_data
        except Exception as e:
            logger.error(f"Failed to parse grocery list from text: {e}")
            console.print(f"[yellow]⚠️  Could not parse grocery list data: {e}[/yellow]")
            return None

    def add_meal(self):
        """Conversationally add a meal."""
        console.print("\n[bold blue]🍽️  Let's add a new meal![/bold blue]\n")

        # Ask for input mode
        mode = Prompt.ask(
            "[cyan]How would you like to add the meal?[/cyan]",
            choices=["paste", "file", "interactive"],
            default="interactive"
        )

        if mode == "paste":
            return self._add_meal_paste_mode()
        elif mode == "file":
            return self._add_meal_file_mode()
        else:
            return self._add_meal_interactive_mode()

    def _add_meal_paste_mode(self):
        """Add meal by pasting text."""
        console.print("\n[dim]Paste your recipe or meal information below.[/dim]")
        console.print("[dim]When done, type 'END' on a new line and press Enter:[/dim]")
        console.print("[dim](Or press Ctrl+D on Mac/Linux, Ctrl+Z on Windows)[/dim]\n")

        # Read multi-line input
        lines = []
        try:
            while True:
                line = input()
                # Check for END marker
                if line.strip() == 'END':
                    break
                lines.append(line)
        except EOFError:
            pass

        text = '\n'.join(lines)

        if not text.strip():
            console.print("[yellow]No text provided. Meal not saved.[/yellow]")
            return

        # Show character count for debugging
        console.print(f"[dim]Received {len(text)} characters[/dim]")

        console.print("\n[cyan]⏳ Parsing your text...[/cyan]\n")

        # Parse with LLM
        meal = self._parse_meal_from_text(text)

        if not meal:
            console.print("[red]Failed to parse meal data. Please try again.[/red]")
            return

        # Show parsed data
        console.print("[bold]📋 Parsed Meal Data:[/bold]")
        console.print(f"  Name: {meal.get('meal_name', 'Unknown')}")
        console.print(f"  Cuisine: {meal.get('cuisine', 'Unknown')}")
        console.print(f"  Servings: {meal.get('servings', 4)}")
        console.print(f"  Ingredients: {len(meal.get('ingredients', []))} items")
        if meal.get('ingredients'):
            for i, ing in enumerate(meal['ingredients'][:5], 1):
                console.print(f"    {i}. {ing}")
            if len(meal['ingredients']) > 5:
                console.print(f"    ... and {len(meal['ingredients']) - 5} more")
        console.print(f"  Prep time: {meal.get('prep_time_minutes', 45)} minutes")
        console.print(f"  Difficulty: {meal.get('difficulty', 'medium')}")
        console.print(f"  Rating: {meal.get('rating', 4)}/5")
        if meal.get('notes'):
            console.print(f"  Notes: {meal['notes']}")

        # Allow editing
        if Confirm.ask("\n[cyan]Edit any fields?[/cyan]", default=False):
            meal = self._edit_meal(meal)

        # Confirm save
        if not Confirm.ask("\n[cyan]Save this meal?[/cyan]", default=True):
            console.print("[yellow]Meal not saved.[/yellow]")
            return

        # Save to file
        meals_file = "data/raw/meals.json"
        meals = load_json(meals_file) or []
        meals.append(meal)
        save_json(meals, meals_file)

        console.print(f"\n[green]✓ Meal saved to {meals_file}[/green]")

        # Add to RAG database
        if Confirm.ask("[cyan]Add to RAG database now?[/cyan]", default=True):
            try:
                docs, meta = self.data_ingestion.load_meals(meals_file)
                if docs:
                    self.rag_engine.add_documents([docs[-1]], [meta[-1]])
                    console.print("[green]✓ Added to RAG database![/green]\n")
            except Exception as e:
                console.print(f"[yellow]⚠️  Could not add to RAG database: {e}[/yellow]")
                console.print("[dim]You can run 'meal-assistant ingest --meals data/raw/meals.json' later[/dim]\n")
        else:
            console.print("[dim]Run 'meal-assistant ingest --meals data/raw/meals.json' to add it later[/dim]\n")

    def _add_meal_file_mode(self):
        """Add meal by reading from a text file."""
        console.print("\n[dim]Save your recipe to a text file first, then provide the path.[/dim]")
        file_path = Prompt.ask("[cyan]Enter the file path[/cyan]")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            if not text.strip():
                console.print("[yellow]File is empty. Meal not saved.[/yellow]")
                return

            console.print(f"[dim]Read {len(text)} characters from file[/dim]")
            console.print("\n[cyan]⏳ Parsing your text...[/cyan]\n")

            # Parse with LLM
            meal = self._parse_meal_from_text(text)

            if not meal:
                console.print("[red]Failed to parse meal data. Please try again.[/red]")
                return

            # Show parsed data
            console.print("[bold]📋 Parsed Meal Data:[/bold]")
            console.print(f"  Name: {meal.get('meal_name', 'Unknown')}")
            console.print(f"  Cuisine: {meal.get('cuisine', 'Unknown')}")
            console.print(f"  Servings: {meal.get('servings', 4)}")
            console.print(f"  Ingredients: {len(meal.get('ingredients', []))} items")
            if meal.get('ingredients'):
                for i, ing in enumerate(meal['ingredients'][:5], 1):
                    console.print(f"    {i}. {ing}")
                if len(meal['ingredients']) > 5:
                    console.print(f"    ... and {len(meal['ingredients']) - 5} more")
            console.print(f"  Prep time: {meal.get('prep_time_minutes', 45)} minutes")
            console.print(f"  Difficulty: {meal.get('difficulty', 'medium')}")
            console.print(f"  Rating: {meal.get('rating', 4)}/5")
            if meal.get('notes'):
                console.print(f"  Notes: {meal['notes']}")

            # Allow editing
            if Confirm.ask("\n[cyan]Edit any fields?[/cyan]", default=False):
                meal = self._edit_meal(meal)

            # Confirm save
            if not Confirm.ask("\n[cyan]Save this meal?[/cyan]", default=True):
                console.print("[yellow]Meal not saved.[/yellow]")
                return

            # Save to file
            meals_file = "data/raw/meals.json"
            meals = load_json(meals_file) or []
            meals.append(meal)
            save_json(meals, meals_file)

            console.print(f"\n[green]✓ Meal saved to {meals_file}[/green]")

            # Add to RAG database
            if Confirm.ask("[cyan]Add to RAG database now?[/cyan]", default=True):
                try:
                    docs, meta = self.data_ingestion.load_meals(meals_file)
                    if docs:
                        self.rag_engine.add_documents([docs[-1]], [meta[-1]])
                        console.print("[green]✓ Added to RAG database![/green]\n")
                except Exception as e:
                    console.print(f"[yellow]⚠️  Could not add to RAG database: {e}[/yellow]")
                    console.print("[dim]You can run 'meal-assistant ingest --meals data/raw/meals.json' later[/dim]\n")
            else:
                console.print("[dim]Run 'meal-assistant ingest --meals data/raw/meals.json' to add it later[/dim]\n")

        except FileNotFoundError:
            console.print(f"[red]✗ File not found: {file_path}[/red]")
        except Exception as e:
            console.print(f"[red]✗ Error reading file: {e}[/red]")
            logger.error(f"File read error: {e}", exc_info=True)

    def _edit_meal(self, meal: Dict[str, Any]) -> Dict[str, Any]:
        """Allow editing of meal fields."""
        meal['meal_name'] = Prompt.ask("[cyan]Meal name[/cyan]", default=meal.get('meal_name', ''))
        meal['cuisine'] = Prompt.ask("[cyan]Cuisine[/cyan]", default=meal.get('cuisine', ''))
        meal['servings'] = int(Prompt.ask("[cyan]Servings[/cyan]", default=str(meal.get('servings', 4))))
        meal['prep_time_minutes'] = int(Prompt.ask("[cyan]Prep time (minutes)[/cyan]", default=str(meal.get('prep_time_minutes', 45))))
        meal['difficulty'] = Prompt.ask("[cyan]Difficulty[/cyan]", choices=["easy", "medium", "hard"], default=meal.get('difficulty', 'medium'))
        meal['rating'] = int(Prompt.ask("[cyan]Rating (1-5)[/cyan]", default=str(meal.get('rating', 4))))
        meal['notes'] = Prompt.ask("[cyan]Notes[/cyan]", default=meal.get('notes', ''))
        return meal

    def _add_meal_interactive_mode(self):
        """Add meal through interactive questions."""
        console.print("\n[dim]I'll ask you a few questions to capture the details.[/dim]\n")

        # Collect basic information
        meal_name = Prompt.ask("[cyan]What's the name of the meal?[/cyan]")
        cuisine = Prompt.ask("[cyan]What cuisine is it?[/cyan]", default="Asian")
        
        # Get ingredients
        console.print("\n[cyan]Let's add ingredients. Enter them one by one (press Enter with empty line when done):[/cyan]")
        ingredients = []
        while True:
            ingredient = Prompt.ask(f"  Ingredient {len(ingredients) + 1}", default="")
            if not ingredient:
                break
            ingredients.append(ingredient)
        
        if not ingredients:
            console.print("[yellow]No ingredients added. Meal not saved.[/yellow]")
            return
        
        # Get additional details
        servings = Prompt.ask("[cyan]How many servings?[/cyan]", default="4")
        prep_time = Prompt.ask("[cyan]Prep time in minutes?[/cyan]", default="45")
        difficulty = Prompt.ask("[cyan]Difficulty level?[/cyan]", choices=["easy", "medium", "hard"], default="medium")
        rating = Prompt.ask("[cyan]Rating (1-5)?[/cyan]", default="4")
        notes = Prompt.ask("[cyan]Any notes?[/cyan]", default="")
        
        # Create meal object
        meal = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "meal_name": meal_name,
            "cuisine": cuisine,
            "servings": int(servings),
            "ingredients": ingredients,
            "prep_time_minutes": int(prep_time),
            "difficulty": difficulty,
            "notes": notes,
            "rating": int(rating)
        }
        
        # Show summary
        console.print("\n[bold]📋 Meal Summary:[/bold]")
        console.print(f"  Name: {meal_name}")
        console.print(f"  Cuisine: {cuisine}")
        console.print(f"  Servings: {servings}")
        console.print(f"  Ingredients: {len(ingredients)} items")
        console.print(f"  Prep time: {prep_time} minutes")
        console.print(f"  Difficulty: {difficulty}")
        console.print(f"  Rating: {rating}/5")
        if notes:
            console.print(f"  Notes: {notes}")
        
        # Confirm
        if not Confirm.ask("\n[cyan]Save this meal?[/cyan]", default=True):
            console.print("[yellow]Meal not saved.[/yellow]")
            return
        
        # Save to file
        meals_file = "data/raw/meals.json"
        meals = load_json(meals_file) or []
        meals.append(meal)
        save_json(meals, meals_file)
        
        console.print(f"\n[green]✓ Meal saved to {meals_file}[/green]")
        
        # Add to RAG database
        if Confirm.ask("[cyan]Add to RAG database now?[/cyan]", default=True):
            try:
                docs, meta = self.data_ingestion.load_meals(meals_file)
                # Only add the last meal (the one we just added)
                if docs:
                    self.rag_engine.add_documents([docs[-1]], [meta[-1]])
                    console.print("[green]✓ Added to RAG database![/green]\n")
            except Exception as e:
                console.print(f"[yellow]⚠️  Could not add to RAG database: {e}[/yellow]")
                console.print("[dim]You can run 'meal-assistant ingest --meals data/raw/meals.json' later[/dim]\n")
        else:
            console.print("[dim]Run 'meal-assistant ingest --meals data/raw/meals.json' to add it later[/dim]\n")
    
    def add_bill(self):
        """Conversationally add a shopping bill."""
        console.print("\n[bold blue]🧾 Let's add a shopping bill![/bold blue]\n")

        # Ask for input mode
        mode = Prompt.ask(
            "[cyan]How would you like to add the bill?[/cyan]",
            choices=["paste", "file", "interactive"],
            default="interactive"
        )

        if mode == "paste":
            return self._add_bill_paste_mode()
        elif mode == "file":
            return self._add_bill_file_mode()
        else:
            return self._add_bill_interactive_mode()

    def _add_bill_paste_mode(self):
        """Add bill by pasting text."""
        console.print("\n[dim]Paste your receipt or bill information below.[/dim]")
        console.print("[dim]When done, type 'END' on a new line and press Enter:[/dim]")
        console.print("[dim](Or press Ctrl+D on Mac/Linux, Ctrl+Z on Windows)[/dim]\n")

        # Read multi-line input
        lines = []
        try:
            while True:
                line = input()
                # Check for END marker
                if line.strip() == 'END':
                    break
                lines.append(line)
        except EOFError:
            pass

        text = '\n'.join(lines)

        if not text.strip():
            console.print("[yellow]No text provided. Bill not saved.[/yellow]")
            return

        # Show character count for debugging
        console.print(f"[dim]Received {len(text)} characters[/dim]")

        console.print("\n[cyan]⏳ Parsing your text...[/cyan]\n")

        # Parse with LLM
        bill = self._parse_bill_from_text(text)

        if not bill:
            console.print("[red]Failed to parse bill data. Please try again.[/red]")
            return

        # Show parsed data
        console.print("[bold]📋 Parsed Bill Data:[/bold]")
        console.print(f"  Store: {bill.get('store', 'Unknown')}")
        console.print(f"  Date: {bill.get('date', 'Unknown')}")
        console.print(f"  Items: {len(bill.get('items', []))}")
        if bill.get('items'):
            for i, item in enumerate(bill['items'][:5], 1):
                console.print(f"    {i}. {item.get('name')} - ${item.get('price', 0):.2f}")
            if len(bill['items']) > 5:
                console.print(f"    ... and {len(bill['items']) - 5} more")
        console.print(f"  Total: {bill.get('currency', 'USD')} {bill.get('total_amount', 0):.2f}")

        # Allow editing
        if Confirm.ask("\n[cyan]Edit any fields?[/cyan]", default=False):
            bill = self._edit_bill(bill)

        # Confirm save
        if not Confirm.ask("\n[cyan]Save this bill?[/cyan]", default=True):
            console.print("[yellow]Bill not saved.[/yellow]")
            return

        # Save to file
        bills_file = "data/raw/bills.json"
        bills = load_json(bills_file) or []
        bills.append(bill)
        save_json(bills, bills_file)

        console.print(f"\n[green]✓ Bill saved to {bills_file}[/green]")

        # Add to RAG database
        if Confirm.ask("[cyan]Add to RAG database now?[/cyan]", default=True):
            try:
                docs, meta = self.data_ingestion.load_bills(bills_file)
                num_new_docs = len([d for d in meta if d.get('date') == bill['date']])
                if docs and num_new_docs > 0:
                    self.rag_engine.add_documents(docs[-num_new_docs:], meta[-num_new_docs:])
                    console.print("[green]✓ Added to RAG database![/green]\n")
            except Exception as e:
                console.print(f"[yellow]⚠️  Could not add to RAG database: {e}[/yellow]")
                console.print("[dim]You can run 'meal-assistant ingest --bills data/raw/bills.json' later[/dim]\n")
        else:
            console.print("[dim]Run 'meal-assistant ingest --bills data/raw/bills.json' to add it later[/dim]\n")

    def _add_bill_file_mode(self):
        """Add bill by reading from a text file."""
        console.print("\n[dim]Save your receipt to a text file first, then provide the path.[/dim]")
        file_path = Prompt.ask("[cyan]Enter the file path[/cyan]")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            if not text.strip():
                console.print("[yellow]File is empty. Bill not saved.[/yellow]")
                return

            console.print(f"[dim]Read {len(text)} characters from file[/dim]")
            console.print("\n[cyan]⏳ Parsing your text...[/cyan]\n")

            # Parse with LLM
            bill = self._parse_bill_from_text(text)

            if not bill:
                console.print("[red]Failed to parse bill data. Please try again.[/red]")
                return

            # Show parsed data
            console.print("[bold]📋 Parsed Bill Data:[/bold]")
            console.print(f"  Store: {bill.get('store', 'Unknown')}")
            console.print(f"  Date: {bill.get('date', 'Unknown')}")
            console.print(f"  Currency: {bill.get('currency', 'USD')}")
            console.print(f"  Items: {len(bill.get('items', []))} items")
            if bill.get('items'):
                for i, item in enumerate(bill['items'][:5], 1):
                    console.print(f"    {i}. {item.get('name', 'Unknown')} - {item.get('price', 0)} {bill.get('currency', 'USD')}")
                if len(bill['items']) > 5:
                    console.print(f"    ... and {len(bill['items']) - 5} more")
            console.print(f"  Total: {bill.get('total_amount', 0)} {bill.get('currency', 'USD')}")

            # Allow editing
            if Confirm.ask("\n[cyan]Edit any fields?[/cyan]", default=False):
                bill = self._edit_bill(bill)

            # Confirm save
            if not Confirm.ask("\n[cyan]Save this bill?[/cyan]", default=True):
                console.print("[yellow]Bill not saved.[/yellow]")
                return

            # Save to file
            bills_file = "data/raw/bills.json"
            bills = load_json(bills_file) or []
            bills.append(bill)
            save_json(bills, bills_file)

            console.print(f"\n[green]✓ Bill saved to {bills_file}[/green]")

            # Add to RAG database
            if Confirm.ask("[cyan]Add to RAG database now?[/cyan]", default=True):
                try:
                    docs, meta = self.data_ingestion.load_bills(bills_file)
                    num_new_docs = len([d for d in meta if d.get('date') == bill['date']])
                    if docs and num_new_docs > 0:
                        self.rag_engine.add_documents(docs[-num_new_docs:], meta[-num_new_docs:])
                        console.print("[green]✓ Added to RAG database![/green]\n")
                except Exception as e:
                    console.print(f"[yellow]⚠️  Could not add to RAG database: {e}[/yellow]")
                    console.print("[dim]You can run 'meal-assistant ingest --bills data/raw/bills.json' later[/dim]\n")
            else:
                console.print("[dim]Run 'meal-assistant ingest --bills data/raw/bills.json' to add it later[/dim]\n")

        except FileNotFoundError:
            console.print(f"[red]✗ File not found: {file_path}[/red]")
        except Exception as e:
            console.print(f"[red]✗ Error reading file: {e}[/red]")
            logger.error(f"File read error: {e}", exc_info=True)

    def _edit_bill(self, bill: Dict[str, Any]) -> Dict[str, Any]:
        """Allow editing of bill fields."""
        bill['store'] = Prompt.ask("[cyan]Store name[/cyan]", default=bill.get('store', ''))
        bill['date'] = Prompt.ask("[cyan]Date (YYYY-MM-DD)[/cyan]", default=bill.get('date', ''))
        bill['currency'] = Prompt.ask("[cyan]Currency[/cyan]", default=bill.get('currency', 'USD'))
        # For simplicity, don't edit individual items - user can use interactive mode for that
        return bill

    def _add_bill_interactive_mode(self):
        """Add bill through interactive questions."""
        console.print("\n[dim]I'll help you record your shopping trip.[/dim]\n")

        # Collect basic information
        store = Prompt.ask("[cyan]Which store did you shop at?[/cyan]")
        date = Prompt.ask("[cyan]Date (YYYY-MM-DD)?[/cyan]", default=datetime.now().strftime("%Y-%m-%d"))
        currency = Prompt.ask("[cyan]Currency?[/cyan]", default="USD")
        
        # Get items
        console.print("\n[cyan]Let's add items. Enter them one by one (press Enter with empty name when done):[/cyan]")
        items = []
        while True:
            console.print(f"\n[bold]Item {len(items) + 1}:[/bold]")
            name = Prompt.ask("  Name", default="")
            if not name:
                break
            
            quantity = Prompt.ask("  Quantity", default="1")
            unit = Prompt.ask("  Unit", default="item")
            price = Prompt.ask("  Price", default="0")
            category = Prompt.ask("  Category", 
                                 choices=["produce", "protein", "dairy", "grains_pasta", "pantry", "household"],
                                 default="pantry")
            
            items.append({
                "name": name,
                "quantity": quantity,
                "unit": unit,
                "price": float(price),
                "category": category
            })
        
        if not items:
            console.print("[yellow]No items added. Bill not saved.[/yellow]")
            return
        
        # Calculate total
        total_amount = sum(item['price'] for item in items)
        
        # Create bill object
        bill = {
            "date": date,
            "store": store,
            "total_amount": round(total_amount, 2),
            "currency": currency,
            "items": items
        }
        
        # Show summary
        console.print("\n[bold]📋 Bill Summary:[/bold]")
        console.print(f"  Store: {store}")
        console.print(f"  Date: {date}")
        console.print(f"  Items: {len(items)}")
        console.print(f"  Total: {currency} {total_amount:.2f}")
        
        # Confirm
        if not Confirm.ask("\n[cyan]Save this bill?[/cyan]", default=True):
            console.print("[yellow]Bill not saved.[/yellow]")
            return
        
        # Save to file
        bills_file = "data/raw/bills.json"
        bills = load_json(bills_file) or []
        bills.append(bill)
        save_json(bills, bills_file)
        
        console.print(f"\n[green]✓ Bill saved to {bills_file}[/green]")
        
        # Add to RAG database
        if Confirm.ask("[cyan]Add to RAG database now?[/cyan]", default=True):
            try:
                docs, meta = self.data_ingestion.load_bills(bills_file)
                # Only add the last bill's documents
                num_new_docs = len([d for d in meta if d.get('date') == date])
                if docs and num_new_docs > 0:
                    self.rag_engine.add_documents(docs[-num_new_docs:], meta[-num_new_docs:])
                    console.print("[green]✓ Added to RAG database![/green]\n")
            except Exception as e:
                console.print(f"[yellow]⚠️  Could not add to RAG database: {e}[/yellow]")
                console.print("[dim]You can run 'meal-assistant ingest --bills data/raw/bills.json' later[/dim]\n")
        else:
            console.print("[dim]Run 'meal-assistant ingest --bills data/raw/bills.json' to add it later[/dim]\n")
    
    def add_grocery_list(self):
        """Conversationally add a grocery list."""
        console.print("\n[bold blue]📝 Let's add a grocery list![/bold blue]\n")

        # Ask for input mode
        mode = Prompt.ask(
            "[cyan]How would you like to add the grocery list?[/cyan]",
            choices=["paste", "file", "interactive"],
            default="interactive"
        )

        if mode == "paste":
            return self._add_grocery_paste_mode()
        elif mode == "file":
            return self._add_grocery_file_mode()
        else:
            return self._add_grocery_interactive_mode()

    def _add_grocery_paste_mode(self):
        """Add grocery list by pasting text."""
        console.print("\n[dim]Paste your grocery list below.[/dim]")
        console.print("[dim]When done, type 'END' on a new line and press Enter:[/dim]")
        console.print("[dim](Or press Ctrl+D on Mac/Linux, Ctrl+Z on Windows)[/dim]\n")

        # Read multi-line input
        lines = []
        try:
            while True:
                line = input()
                # Check for END marker
                if line.strip() == 'END':
                    break
                lines.append(line)
        except EOFError:
            pass

        text = '\n'.join(lines)

        if not text.strip():
            console.print("[yellow]No text provided. Grocery list not saved.[/yellow]")
            return

        # Show character count for debugging
        console.print(f"[dim]Received {len(text)} characters[/dim]")

        console.print("\n[cyan]⏳ Parsing your text...[/cyan]\n")

        # Parse with LLM
        grocery_list = self._parse_grocery_from_text(text)

        if not grocery_list:
            console.print("[red]Failed to parse grocery list data. Please try again.[/red]")
            return

        # Show parsed data
        console.print("[bold]📋 Parsed Grocery List:[/bold]")
        console.print(f"  Occasion: {grocery_list.get('occasion', 'shopping')}")
        console.print(f"  Date: {grocery_list.get('date', 'Unknown')}")
        console.print(f"  Items: {len(grocery_list.get('items', []))}")
        if grocery_list.get('items'):
            for i, item in enumerate(grocery_list['items'][:10], 1):
                console.print(f"    {i}. {item}")
            if len(grocery_list['items']) > 10:
                console.print(f"    ... and {len(grocery_list['items']) - 10} more")
        console.print(f"  Completed: {'Yes' if grocery_list.get('completed') else 'No'}")
        if grocery_list.get('notes'):
            console.print(f"  Notes: {grocery_list['notes']}")

        # Allow editing
        if Confirm.ask("\n[cyan]Edit any fields?[/cyan]", default=False):
            grocery_list = self._edit_grocery(grocery_list)

        # Confirm save
        if not Confirm.ask("\n[cyan]Save this grocery list?[/cyan]", default=True):
            console.print("[yellow]Grocery list not saved.[/yellow]")
            return

        # Save to file
        lists_file = "data/raw/grocery_lists.json"
        lists = load_json(lists_file) or []
        lists.append(grocery_list)
        save_json(lists, lists_file)

        console.print(f"\n[green]✓ Grocery list saved to {lists_file}[/green]")

        # Add to RAG database
        if Confirm.ask("[cyan]Add to RAG database now?[/cyan]", default=True):
            try:
                docs, meta = self.data_ingestion.load_grocery_lists(lists_file)
                if docs:
                    self.rag_engine.add_documents([docs[-1]], [meta[-1]])
                    console.print("[green]✓ Added to RAG database![/green]\n")
            except Exception as e:
                console.print(f"[yellow]⚠️  Could not add to RAG database: {e}[/yellow]")
                console.print("[dim]You can run 'meal-assistant ingest --grocery-lists data/raw/grocery_lists.json' later[/dim]\n")
        else:
            console.print("[dim]Run 'meal-assistant ingest --grocery-lists data/raw/grocery_lists.json' to add it later[/dim]\n")

    def _add_grocery_file_mode(self):
        """Add grocery list by reading from a text file."""
        console.print("\n[dim]Save your grocery list to a text file first, then provide the path.[/dim]")
        file_path = Prompt.ask("[cyan]Enter the file path[/cyan]")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            if not text.strip():
                console.print("[yellow]File is empty. Grocery list not saved.[/yellow]")
                return

            console.print(f"[dim]Read {len(text)} characters from file[/dim]")
            console.print("\n[cyan]⏳ Parsing your text...[/cyan]\n")

            # Parse with LLM
            grocery_list = self._parse_grocery_from_text(text)

            if not grocery_list:
                console.print("[red]Failed to parse grocery list data. Please try again.[/red]")
                return

            # Show parsed data
            console.print("[bold]📋 Parsed Grocery List:[/bold]")
            console.print(f"  Occasion: {grocery_list.get('occasion', 'shopping')}")
            console.print(f"  Date: {grocery_list.get('date', 'Unknown')}")
            console.print(f"  Items: {len(grocery_list.get('items', []))} items")
            if grocery_list.get('items'):
                for i, item in enumerate(grocery_list['items'][:10], 1):
                    console.print(f"    {i}. {item}")
                if len(grocery_list['items']) > 10:
                    console.print(f"    ... and {len(grocery_list['items']) - 10} more")
            if grocery_list.get('notes'):
                console.print(f"  Notes: {grocery_list['notes']}")

            # Allow editing
            if Confirm.ask("\n[cyan]Edit any fields?[/cyan]", default=False):
                grocery_list = self._edit_grocery(grocery_list)

            # Confirm save
            if not Confirm.ask("\n[cyan]Save this grocery list?[/cyan]", default=True):
                console.print("[yellow]Grocery list not saved.[/yellow]")
                return

            # Save to file
            lists_file = "data/raw/grocery_lists.json"
            lists = load_json(lists_file) or []
            lists.append(grocery_list)
            save_json(lists, lists_file)

            console.print(f"\n[green]✓ Grocery list saved to {lists_file}[/green]")

            # Add to RAG database
            if Confirm.ask("[cyan]Add to RAG database now?[/cyan]", default=True):
                try:
                    docs, meta = self.data_ingestion.load_grocery_lists(lists_file)
                    if docs:
                        self.rag_engine.add_documents([docs[-1]], [meta[-1]])
                        console.print("[green]✓ Added to RAG database![/green]\n")
                except Exception as e:
                    console.print(f"[yellow]⚠️  Could not add to RAG database: {e}[/yellow]")
                    console.print("[dim]You can run 'meal-assistant ingest --grocery-lists data/raw/grocery_lists.json' later[/dim]\n")
            else:
                console.print("[dim]Run 'meal-assistant ingest --grocery-lists data/raw/grocery_lists.json' to add it later[/dim]\n")

        except FileNotFoundError:
            console.print(f"[red]✗ File not found: {file_path}[/red]")
        except Exception as e:
            console.print(f"[red]✗ Error reading file: {e}[/red]")
            logger.error(f"File read error: {e}", exc_info=True)

    def _edit_grocery(self, grocery_list: Dict[str, Any]) -> Dict[str, Any]:
        """Allow editing of grocery list fields."""
        grocery_list['occasion'] = Prompt.ask("[cyan]Occasion[/cyan]", default=grocery_list.get('occasion', 'shopping'))
        grocery_list['date'] = Prompt.ask("[cyan]Date (YYYY-MM-DD)[/cyan]", default=grocery_list.get('date', ''))
        grocery_list['notes'] = Prompt.ask("[cyan]Notes[/cyan]", default=grocery_list.get('notes', ''))
        grocery_list['completed'] = Confirm.ask("[cyan]Completed?[/cyan]", default=grocery_list.get('completed', False))
        return grocery_list

    def _add_grocery_interactive_mode(self):
        """Add grocery list through interactive questions."""
        console.print("\n[dim]Record what you plan to buy.[/dim]\n")

        # Collect basic information
        occasion = Prompt.ask("[cyan]What's the occasion?[/cyan]", default="weekly shopping")
        date = Prompt.ask("[cyan]Date (YYYY-MM-DD)?[/cyan]", default=datetime.now().strftime("%Y-%m-%d"))
        
        # Get items
        console.print("\n[cyan]Enter items one by one (press Enter with empty line when done):[/cyan]")
        items = []
        while True:
            item = Prompt.ask(f"  Item {len(items) + 1}", default="")
            if not item:
                break
            items.append(item)
        
        if not items:
            console.print("[yellow]No items added. Grocery list not saved.[/yellow]")
            return
        
        notes = Prompt.ask("[cyan]Any notes?[/cyan]", default="")
        completed = Confirm.ask("[cyan]Is this list completed?[/cyan]", default=False)
        
        # Create grocery list object
        grocery_list = {
            "date": date,
            "occasion": occasion,
            "items": items,
            "completed": completed,
            "notes": notes
        }
        
        # Show summary
        console.print("\n[bold]📋 Grocery List Summary:[/bold]")
        console.print(f"  Occasion: {occasion}")
        console.print(f"  Date: {date}")
        console.print(f"  Items: {len(items)}")
        console.print(f"  Completed: {'Yes' if completed else 'No'}")
        if notes:
            console.print(f"  Notes: {notes}")
        
        # Confirm
        if not Confirm.ask("\n[cyan]Save this grocery list?[/cyan]", default=True):
            console.print("[yellow]Grocery list not saved.[/yellow]")
            return
        
        # Save to file
        lists_file = "data/raw/grocery_lists.json"
        lists = load_json(lists_file) or []
        lists.append(grocery_list)
        save_json(lists, lists_file)
        
        console.print(f"\n[green]✓ Grocery list saved to {lists_file}[/green]")
        
        # Add to RAG database
        if Confirm.ask("[cyan]Add to RAG database now?[/cyan]", default=True):
            try:
                docs, meta = self.data_ingestion.load_grocery_lists(lists_file)
                if docs:
                    self.rag_engine.add_documents([docs[-1]], [meta[-1]])
                    console.print("[green]✓ Added to RAG database![/green]\n")
            except Exception as e:
                console.print(f"[yellow]⚠️  Could not add to RAG database: {e}[/yellow]")
                console.print("[dim]You can run 'meal-assistant ingest --grocery-lists data/raw/grocery_lists.json' later[/dim]\n")
        else:
            console.print("[dim]Run 'meal-assistant ingest --grocery-lists data/raw/grocery_lists.json' to add it later[/dim]\n")

