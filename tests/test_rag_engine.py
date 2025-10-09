import pytest
import os
import shutil
from src.rag_engine import RAGEngine

@pytest.fixture
def temp_rag_engine(tmp_path, monkeypatch):
    """Create a temporary RAG engine for testing"""
    # Create temporary config
    config = {
        'rag': {
            'vector_db_path': str(tmp_path / 'vector_db'),
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'chunk_size': 500,
            'chunk_overlap': 50,
            'top_k': 5
        }
    }
    
    # Mock load_config to use temporary config
    def mock_load_config(path):
        return config
    
    monkeypatch.setattr('src.rag_engine.load_config', mock_load_config)
    
    engine = RAGEngine()
    yield engine
    
    # Cleanup
    if os.path.exists(str(tmp_path / 'vector_db')):
        shutil.rmtree(str(tmp_path / 'vector_db'))

def test_rag_engine_initialization(temp_rag_engine):
    """Test RAG engine initialization"""
    assert temp_rag_engine.index is not None
    assert temp_rag_engine.embedding_model is not None
    assert temp_rag_engine.embedding_dim > 0

def test_add_documents(temp_rag_engine):
    """Test adding documents"""
    documents = [
        "I cooked spaghetti carbonara on Monday with eggs, pasta, and bacon.",
        "User prefers vegetarian meals on weekdays.",
        "Bought dish soap and paper towels at the store."
    ]
    
    metadata = [
        {'type': 'meal', 'date': '2024-01-15'},
        {'type': 'preference', 'category': 'dietary'},
        {'type': 'bill', 'date': '2024-01-20'}
    ]
    
    temp_rag_engine.add_documents(documents, metadata)
    
    assert len(temp_rag_engine.documents) == 3
    assert len(temp_rag_engine.metadata) == 3
    assert temp_rag_engine.index.ntotal == 3

def test_query_documents(temp_rag_engine):
    """Test querying documents"""
    documents = [
        "I cooked spaghetti carbonara with pasta, eggs, bacon, and parmesan cheese.",
        "User loves Italian food, especially pasta dishes.",
        "Bought pasta, eggs, and bacon at grocery store for $15.50."
    ]
    
    metadata = [
        {'type': 'meal', 'cuisine': 'italian'},
        {'type': 'preference', 'cuisine': 'italian'},
        {'type': 'bill', 'store': 'grocery'}
    ]
    
    temp_rag_engine.add_documents(documents, metadata)
    
    results = temp_rag_engine.query("Italian pasta meals", top_k=2)
    
    assert len(results) <= 2
    assert all('document' in r for r in results)
    assert all('metadata' in r for r in results)
    assert all('score' in r for r in results)

def test_query_with_metadata_filter(temp_rag_engine):
    """Test querying with metadata filters"""
    documents = [
        "Cooked chicken curry on Tuesday",
        "User prefers spicy food",
        "Bought chicken at store"
    ]
    
    metadata = [
        {'type': 'meal', 'date': '2024-01-16'},
        {'type': 'preference', 'taste': 'spicy'},
        {'type': 'bill', 'date': '2024-01-15'}
    ]
    
    temp_rag_engine.add_documents(documents, metadata)
    
    # Filter by type
    results = temp_rag_engine.query(
        "chicken",
        top_k=5,
        filter_metadata={'type': 'meal'}
    )
    
    assert len(results) >= 1
    assert all(r['metadata']['type'] == 'meal' for r in results)

def test_get_context(temp_rag_engine):
    """Test getting formatted context"""
    documents = [
        "User loves Mexican food",
        "Cooked tacos last week"
    ]
    
    metadata = [
        {'type': 'preference'},
        {'type': 'meal'}
    ]
    
    temp_rag_engine.add_documents(documents, metadata)
    
    context = temp_rag_engine.get_context("Mexican food preferences", top_k=2)
    
    assert isinstance(context, str)
    assert len(context) > 0
    assert 'Document 1' in context or 'Document 2' in context

def test_empty_query(temp_rag_engine):
    """Test querying empty database"""
    results = temp_rag_engine.query("test query")
    assert results == []

def test_get_stats(temp_rag_engine):
    """Test getting database statistics"""
    documents = [
        "Meal 1",
        "Meal 2",
        "Preference 1",
        "Bill 1"
    ]
    
    metadata = [
        {'type': 'meal'},
        {'type': 'meal'},
        {'type': 'preference'},
        {'type': 'bill'}
    ]
    
    temp_rag_engine.add_documents(documents, metadata)
    
    stats = temp_rag_engine.get_stats()
    
    assert stats['total_documents'] == 4
    assert stats['by_type']['meal'] == 2
    assert stats['by_type']['preference'] == 1
    assert stats['by_type']['bill'] == 1
    assert stats['index_size'] == 4

def test_reset_database(temp_rag_engine):
    """Test resetting database"""
    documents = ["Test document"]
    temp_rag_engine.add_documents(documents)
    
    assert len(temp_rag_engine.documents) == 1
    
    temp_rag_engine.reset_database()
    
    assert len(temp_rag_engine.documents) == 0
    assert len(temp_rag_engine.metadata) == 0
    assert temp_rag_engine.index.ntotal == 0

def test_persistence(tmp_path, monkeypatch):
    """Test that index persists between sessions"""
    # Create temporary config
    config = {
        'rag': {
            'vector_db_path': str(tmp_path / 'vector_db'),
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'chunk_size': 500,
            'chunk_overlap': 50,
            'top_k': 5
        }
    }
    
    def mock_load_config(path):
        return config
    
    monkeypatch.setattr('src.rag_engine.load_config', mock_load_config)
    
    # Create first engine and add document
    engine1 = RAGEngine()
    documents = ["Persistent document"]
    metadata = [{'type': 'test'}]
    engine1.add_documents(documents, metadata)
    
    # Create new instance (simulating restart)
    engine2 = RAGEngine()
    
    assert len(engine2.documents) == 1
    assert engine2.documents[0] == "Persistent document"
    
    # Cleanup
    if os.path.exists(str(tmp_path / 'vector_db')):
        shutil.rmtree(str(tmp_path / 'vector_db'))

