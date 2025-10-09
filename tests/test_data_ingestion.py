import pytest
import os
from src.data_ingestion import DataIngestion

@pytest.fixture
def data_ingestion():
    """Create a DataIngestion instance"""
    return DataIngestion()

def test_load_meals(data_ingestion):
    """Test loading meal data"""
    docs, meta = data_ingestion.load_meals('tests/fixtures/sample_meals.json')
    
    assert len(docs) == 5
    assert len(meta) == 5
    
    # Check first meal
    assert 'Spaghetti Carbonara' in docs[0]
    assert 'Italian' in docs[0]
    assert meta[0]['type'] == 'meal'
    assert meta[0]['cuisine'] == 'Italian'

def test_load_preferences(data_ingestion):
    """Test loading preferences"""
    docs, meta = data_ingestion.load_preferences('tests/fixtures/sample_preferences.json')
    
    assert len(docs) > 0
    assert len(meta) > 0
    
    # Check that various preference types are loaded
    types = [m['category'] for m in meta]
    assert 'dietary_restrictions' in types
    assert 'favorite_cuisines' in types
    assert 'household' in types

def test_load_bills(data_ingestion):
    """Test loading bills"""
    docs, meta = data_ingestion.load_bills('tests/fixtures/sample_bills.json')
    
    assert len(docs) > 0
    assert len(meta) > 0
    
    # Check that bills and household items are loaded
    types = [m['type'] for m in meta]
    assert 'bill' in types
    assert 'household_item' in types

def test_load_grocery_lists(data_ingestion):
    """Test loading grocery lists"""
    docs, meta = data_ingestion.load_grocery_lists('tests/fixtures/sample_grocery_lists.json')
    
    assert len(docs) == 2
    assert len(meta) == 2
    
    # Check first list
    assert 'weekly shopping' in docs[0]
    assert meta[0]['type'] == 'grocery_list'

def test_load_nonexistent_file(data_ingestion):
    """Test loading non-existent file"""
    docs, meta = data_ingestion.load_meals('nonexistent.json')
    
    assert docs == []
    assert meta == []

def test_format_meal_document(data_ingestion):
    """Test meal document formatting"""
    meal = {
        'meal_name': 'Test Meal',
        'date': '2024-01-01',
        'cuisine': 'Test',
        'servings': 2,
        'ingredients': ['ingredient1', 'ingredient2'],
        'prep_time_minutes': 30,
        'difficulty': 'easy',
        'rating': 5,
        'notes': 'Test notes'
    }
    
    doc = data_ingestion._format_meal_document(meal)
    
    assert 'Test Meal' in doc
    assert '2024-01-01' in doc
    assert 'ingredient1' in doc
    assert '30 minutes' in doc
    assert 'easy' in doc
    assert '5/5' in doc
    assert 'Test notes' in doc

def test_format_bill_document(data_ingestion):
    """Test bill document formatting"""
    bill = {
        'store': 'Test Store',
        'date': '2024-01-01',
        'currency': 'USD',
        'total_amount': 50.00,
        'items': [
            {'name': 'item1', 'category': 'produce'},
            {'name': 'item2', 'category': 'produce'},
            {'name': 'item3', 'category': 'dairy'}
        ]
    }
    
    doc = data_ingestion._format_bill_document(bill)
    
    assert 'Test Store' in doc
    assert '2024-01-01' in doc
    assert '50.0' in doc
    assert 'item1' in doc
    assert 'Produce' in doc
    assert 'Dairy' in doc

def test_ingest_all_data(data_ingestion, tmp_path, monkeypatch):
    """Test ingesting all data types"""
    # Create a mock RAG engine
    class MockRAGEngine:
        def __init__(self):
            self.documents = []
            self.metadata = []
        
        def add_documents(self, docs, meta):
            self.documents.extend(docs)
            self.metadata.extend(meta)
    
    mock_engine = MockRAGEngine()
    
    # Copy test fixtures to tmp directory
    import shutil
    fixtures_dir = 'tests/fixtures'
    test_dir = tmp_path / 'test_data'
    test_dir.mkdir()
    
    shutil.copy(os.path.join(fixtures_dir, 'sample_meals.json'), 
                test_dir / 'meals.json')
    shutil.copy(os.path.join(fixtures_dir, 'sample_preferences.json'), 
                test_dir / 'preferences.json')
    shutil.copy(os.path.join(fixtures_dir, 'sample_bills.json'), 
                test_dir / 'bills.json')
    shutil.copy(os.path.join(fixtures_dir, 'sample_grocery_lists.json'), 
                test_dir / 'grocery_lists.json')
    
    # Ingest all data
    counts = data_ingestion.ingest_all_data(str(test_dir), mock_engine)
    
    assert counts['meals'] == 5
    assert counts['preferences'] > 0
    assert counts['bills'] > 0
    assert counts['grocery_lists'] == 2
    
    assert len(mock_engine.documents) > 0
    assert len(mock_engine.metadata) > 0

