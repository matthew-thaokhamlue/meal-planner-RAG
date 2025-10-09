"""CLI interface for the meal shopping assistant."""

import click
import logging
import os
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich import print as rprint

from src.rag_engine import RAGEngine
from src.llm_client import OpenRouterClient
from src.meal_planner import MealPlanner
from src.data_ingestion import DataIngestion
from src.utils import setup_logging, format_shopping_list
from src.interactive_add import InteractiveDataAdder

# Initialize console for rich output
console = Console()

# Setup logging
logger = setup_logging()

@click.group()
@click.version_option(version='0.1.0')
def main():
    """RAG-Powered Home Meal & Shopping Assistant
    
    Generate personalized meal plans and shopping lists based on your
    cooking history, preferences, and household needs.
    """
    pass

@main.command()
@click.option('--meals', type=click.Path(exists=True), help='Path to meals JSON file')
@click.option('--preferences', type=click.Path(exists=True), help='Path to preferences JSON file')
@click.option('--bills', type=click.Path(exists=True), help='Path to bills JSON file')
@click.option('--grocery-lists', type=click.Path(exists=True), help='Path to grocery lists JSON file')
@click.option('--all', 'all_data', type=click.Path(exists=True), help='Directory containing all data files')
def ingest(meals, preferences, bills, grocery_lists, all_data):
    """Ingest data into the RAG database."""
    console.print("\n[bold blue]🔄 Starting data ingestion...[/bold blue]\n")
    
    try:
        rag_engine = RAGEngine()
        data_ingestion = DataIngestion()
        
        total_docs = 0
        
        if all_data:
            # Ingest all data from directory
            console.print(f"[cyan]Ingesting all data from: {all_data}[/cyan]")
            counts = data_ingestion.ingest_all_data(all_data, rag_engine)
            
            # Display results
            table = Table(title="Ingestion Summary")
            table.add_column("Data Type", style="cyan")
            table.add_column("Documents", style="green")
            
            for data_type, count in counts.items():
                table.add_row(data_type.replace('_', ' ').title(), str(count))
                total_docs += count
            
            console.print(table)
        else:
            # Ingest individual files
            if meals:
                console.print(f"[cyan]Ingesting meals from: {meals}[/cyan]")
                docs, meta = data_ingestion.load_meals(meals)
                if docs:
                    rag_engine.add_documents(docs, meta)
                    total_docs += len(docs)
                    console.print(f"[green]✓ Added {len(docs)} meal documents[/green]")
            
            if preferences:
                console.print(f"[cyan]Ingesting preferences from: {preferences}[/cyan]")
                docs, meta = data_ingestion.load_preferences(preferences)
                if docs:
                    rag_engine.add_documents(docs, meta)
                    total_docs += len(docs)
                    console.print(f"[green]✓ Added {len(docs)} preference documents[/green]")
            
            if bills:
                console.print(f"[cyan]Ingesting bills from: {bills}[/cyan]")
                docs, meta = data_ingestion.load_bills(bills)
                if docs:
                    rag_engine.add_documents(docs, meta)
                    total_docs += len(docs)
                    console.print(f"[green]✓ Added {len(docs)} bill documents[/green]")
            
            if grocery_lists:
                console.print(f"[cyan]Ingesting grocery lists from: {grocery_lists}[/cyan]")
                docs, meta = data_ingestion.load_grocery_lists(grocery_lists)
                if docs:
                    rag_engine.add_documents(docs, meta)
                    total_docs += len(docs)
                    console.print(f"[green]✓ Added {len(docs)} grocery list documents[/green]")
        
        # Display final stats
        stats = rag_engine.get_stats()
        console.print(f"\n[bold green]✓ Ingestion complete![/bold green]")
        console.print(f"[green]Total documents in database: {stats['total_documents']}[/green]\n")
        
    except Exception as e:
        console.print(f"[bold red]✗ Error during ingestion: {e}[/bold red]")
        logger.error(f"Ingestion error: {e}", exc_info=True)
        raise click.Abort()

@main.command()
@click.argument('query')
@click.option('--export', type=click.Path(), help='Export selected plan to file')
def plan(query, export):
    """Generate meal plans based on your query.
    
    Examples:
    
      meal-assistant plan "3 days"
      
      meal-assistant plan "weekend meals"
      
      meal-assistant plan "party for 15 people"
    """
    console.print(f"\n[bold blue]🍽️  Generating meal plans for: '{query}'[/bold blue]\n")
    
    try:
        # Initialize planner
        meal_planner = MealPlanner()
        
        # Generate plans
        with console.status("[cyan]Thinking...[/cyan]", spinner="dots"):
            plans = meal_planner.generate_meal_plans(query)
        
        # Check for errors
        if plans and plans[0].get('error'):
            console.print(Panel(
                plans[0]['description'],
                title="[yellow]⚠️  No Data Available[/yellow]",
                border_style="yellow"
            ))
            return
        
        # Display options
        console.print(f"[bold green]✓ Generated {len(plans)} meal plan options![/bold green]\n")
        
        for i, plan in enumerate(plans, 1):
            console.print(f"\n[bold cyan]Option {i}: {plan['option_name']}[/bold cyan]")
            console.print(f"[dim]{plan.get('description', '')}[/dim]\n")
            
            # Display meals
            console.print("[bold]📅 Meals:[/bold]")
            for meal in plan.get('meals', []):
                day = meal.get('day', 'Day')
                meal_name = meal.get('meal_name', 'Unknown')
                description = meal.get('description', '')
                console.print(f"  • [cyan]{day}[/cyan]: {meal_name}")
                if description:
                    console.print(f"    [dim]{description}[/dim]")
            
            # Display shopping list
            shopping_list = plan.get('shopping_list', {})
            if shopping_list:
                console.print(f"\n[bold]🛒 Shopping List:[/bold]")
                formatted_list = format_shopping_list(shopping_list)
                console.print(formatted_list)
            
            # Display additional info
            if plan.get('estimated_cost'):
                console.print(f"\n[dim]💰 Estimated cost: ${plan['estimated_cost']}[/dim]")
            if plan.get('prep_difficulty'):
                console.print(f"[dim]👨‍🍳 Difficulty: {plan['prep_difficulty']}[/dim]")
            
            console.print("\n" + "─" * 60)
        
        # Ask user to select
        if len(plans) > 1:
            console.print()
            choice = Prompt.ask(
                "[bold]Which option would you like to choose?[/bold]",
                choices=[str(i) for i in range(1, len(plans) + 1)],
                default="1"
            )
            selected_index = int(choice) - 1
            selected_plan = plans[selected_index]
            
            # Log choice
            meal_planner.log_user_choice(selected_plan, query, plans)
            console.print(f"\n[green]✓ Logged your choice: {selected_plan['option_name']}[/green]")
            
            # Export if requested
            if export:
                meal_planner.export_shopping_list(selected_plan, export)
                console.print(f"[green]✓ Shopping list exported to: {export}[/green]")
        
        console.print()
        
    except Exception as e:
        console.print(f"[bold red]✗ Error generating meal plans: {e}[/bold red]")
        logger.error(f"Plan generation error: {e}", exc_info=True)
        raise click.Abort()

@main.command()
@click.option('--last', type=int, default=10, help='Number of recent choices to show')
def history(last):
    """View your past meal plan choices."""
    console.print(f"\n[bold blue]📜 Your Recent Choices (last {last})[/bold blue]\n")
    
    try:
        meal_planner = MealPlanner()
        choices = meal_planner.get_choice_history(limit=last)
        
        if not choices:
            console.print("[yellow]No choices recorded yet.[/yellow]\n")
            return
        
        for i, choice in enumerate(choices, 1):
            timestamp = choice.get('timestamp', 'Unknown')
            query = choice.get('query', 'Unknown')
            option = choice.get('selected_option', {})
            option_name = option.get('option_name', 'Unknown')
            
            console.print(f"[cyan]{i}. {timestamp}[/cyan]")
            console.print(f"   Query: {query}")
            console.print(f"   Selected: [bold]{option_name}[/bold]\n")
        
    except Exception as e:
        console.print(f"[bold red]✗ Error retrieving history: {e}[/bold red]")
        logger.error(f"History error: {e}", exc_info=True)

@main.command()
def reset():
    """Reset the RAG database (clear all data)."""
    console.print("\n[bold yellow]⚠️  Warning: This will delete all ingested data![/bold yellow]\n")
    
    if Confirm.ask("Are you sure you want to reset the database?"):
        try:
            rag_engine = RAGEngine()
            rag_engine.reset_database()
            console.print("\n[bold green]✓ Database reset successfully![/bold green]\n")
        except Exception as e:
            console.print(f"[bold red]✗ Error resetting database: {e}[/bold red]")
            logger.error(f"Reset error: {e}", exc_info=True)
    else:
        console.print("\n[dim]Reset cancelled.[/dim]\n")

@main.command()
def stats():
    """Show database statistics."""
    console.print("\n[bold blue]📊 Database Statistics[/bold blue]\n")

    try:
        rag_engine = RAGEngine()
        stats = rag_engine.get_stats()

        table = Table(title="RAG Database Stats")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Documents", str(stats['total_documents']))
        table.add_row("Index Size", str(stats['index_size']))

        console.print(table)

        if stats['by_type']:
            console.print("\n[bold]Documents by Type:[/bold]")
            for doc_type, count in stats['by_type'].items():
                console.print(f"  • {doc_type}: {count}")

        console.print()

    except Exception as e:
        console.print(f"[bold red]✗ Error retrieving stats: {e}[/bold red]")
        logger.error(f"Stats error: {e}", exc_info=True)

@main.group()
def add():
    """Add new data interactively (menu, bill, grocery)."""
    pass

@add.command()
def menu():
    """Add a new meal/menu interactively."""
    try:
        adder = InteractiveDataAdder()
        adder.add_meal()
    except Exception as e:
        console.print(f"[bold red]✗ Error adding meal: {e}[/bold red]")
        logger.error(f"Add meal error: {e}", exc_info=True)

@add.command()
def bill():
    """Add a new shopping bill interactively."""
    try:
        adder = InteractiveDataAdder()
        adder.add_bill()
    except Exception as e:
        console.print(f"[bold red]✗ Error adding bill: {e}[/bold red]")
        logger.error(f"Add bill error: {e}", exc_info=True)

@add.command()
def grocery():
    """Add a new grocery list interactively."""
    try:
        adder = InteractiveDataAdder()
        adder.add_grocery_list()
    except Exception as e:
        console.print(f"[bold red]✗ Error adding grocery list: {e}[/bold red]")
        logger.error(f"Add grocery list error: {e}", exc_info=True)

if __name__ == '__main__':
    main()

