"""RAG engine for document storage and retrieval using FAISS and sentence transformers."""

import os
import logging
import pickle
import random
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

from src.utils import load_config, ensure_dir

logger = logging.getLogger(__name__)

class RAGEngine:
    """RAG engine for storing and retrieving documents."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize RAG engine.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.rag_config = self.config['rag']
        
        # Initialize paths
        self.vector_db_path = self.rag_config['vector_db_path']
        ensure_dir(self.vector_db_path)
        
        self.index_path = os.path.join(self.vector_db_path, "faiss.index")
        self.metadata_path = os.path.join(self.vector_db_path, "metadata.pkl")
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {self.rag_config['embedding_model']}")
        self.embedding_model = SentenceTransformer(self.rag_config['embedding_model'])
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize or load FAISS index
        self.index = None
        self.documents = []
        self.metadata = []
        
        self._load_or_create_index()
    
    def _load_or_create_index(self) -> None:
        """Load existing index or create a new one."""
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            logger.info("Loading existing FAISS index")
            self.index = faiss.read_index(self.index_path)
            
            with open(self.metadata_path, 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.metadata = data['metadata']
            
            logger.info(f"Loaded index with {len(self.documents)} documents")
        else:
            logger.info("Creating new FAISS index")
            self.index = faiss.IndexFlatL2(self.embedding_dim)
    
    def add_documents(
        self,
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Add documents to the RAG database.
        
        Args:
            documents: List of document strings
            metadata: Optional list of metadata dictionaries
        """
        if not documents:
            logger.warning("No documents to add")
            return
        
        logger.info(f"Adding {len(documents)} documents to RAG database")
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(
            documents,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Store documents and metadata
        self.documents.extend(documents)
        
        if metadata is None:
            metadata = [{}] * len(documents)
        self.metadata.extend(metadata)
        
        # Save index
        self._save_index()
        
        logger.info(f"Successfully added {len(documents)} documents. Total: {len(self.documents)}")
    
    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        diversity_factor: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Query the RAG database.

        Args:
            query: Query string
            top_k: Number of results to return (default from config)
            filter_metadata: Optional metadata filters
            diversity_factor: Float between 0.0 and 1.0. Higher values add more randomness.
                            0.0 = pure similarity ranking (default)
                            0.5 = moderate diversity (retrieves 2x documents, randomly samples)
                            1.0 = high diversity (retrieves 3x documents, randomly samples)

        Returns:
            List of result dictionaries with 'document', 'metadata', and 'score'
        """
        if top_k is None:
            top_k = self.rag_config['top_k']

        if len(self.documents) == 0:
            logger.warning("No documents in database")
            return []

        logger.info(f"Querying RAG database: '{query[:50]}...'")

        # Generate query embedding
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True
        ).astype('float32')

        # Calculate how many documents to retrieve based on diversity factor
        if diversity_factor > 0:
            # Retrieve more documents than needed, then randomly sample
            retrieval_multiplier = 1 + (2 * diversity_factor)  # 1.0 to 3.0
            search_k = min(int(top_k * retrieval_multiplier), len(self.documents))
            logger.info(f"Diversity mode: retrieving {search_k} docs, will sample {top_k}")
        else:
            search_k = min(top_k, len(self.documents))

        # Search FAISS index
        distances, indices = self.index.search(query_embedding, search_k)

        # Prepare results
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue

            doc_metadata = self.metadata[idx]

            # Apply metadata filter if provided
            if filter_metadata:
                match = all(
                    doc_metadata.get(key) == value
                    for key, value in filter_metadata.items()
                )
                if not match:
                    continue

            results.append({
                'document': self.documents[idx],
                'metadata': doc_metadata,
                'score': float(distance)
            })

        # Apply diversity sampling if requested
        if diversity_factor > 0 and len(results) > top_k:
            # Randomly sample from the results, weighted towards better scores
            # Use inverse distance as weight (lower distance = higher weight)
            weights = [1.0 / (1.0 + r['score']) for r in results]
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]

            # Random weighted sample
            sampled_indices = random.choices(
                range(len(results)),
                weights=weights,
                k=top_k
            )
            results = [results[i] for i in sampled_indices]
            logger.info(f"Sampled {len(results)} diverse documents from {len(distances[0])} candidates")

        logger.info(f"Found {len(results)} relevant documents")
        return results
    
    def get_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        diversity_factor: float = 0.0
    ) -> str:
        """
        Get formatted context string for LLM prompt.

        Args:
            query: Query string
            top_k: Number of results to include
            filter_metadata: Optional metadata filters
            diversity_factor: Float between 0.0 and 1.0 for diversity (see query method)

        Returns:
            Formatted context string
        """
        results = self.query(query, top_k, filter_metadata, diversity_factor)

        if not results:
            return "No relevant information found in the database."

        context_parts = []
        for i, result in enumerate(results, 1):
            doc_type = result['metadata'].get('type', 'unknown')
            context_parts.append(f"[Document {i} - {doc_type}]")
            context_parts.append(result['document'])
            context_parts.append("")  # Empty line

        return '\n'.join(context_parts)
    
    def _save_index(self) -> None:
        """Save FAISS index and metadata to disk."""
        logger.info("Saving FAISS index")
        faiss.write_index(self.index, self.index_path)
        
        with open(self.metadata_path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.metadata
            }, f)
    
    def reset_database(self) -> None:
        """Reset the database (clear all documents)."""
        logger.warning("Resetting RAG database")
        
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.documents = []
        self.metadata = []
        
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.metadata_path):
            os.remove(self.metadata_path)
        
        logger.info("Database reset complete")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the database.
        
        Returns:
            Dictionary with statistics
        """
        type_counts = {}
        for meta in self.metadata:
            doc_type = meta.get('type', 'unknown')
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        
        return {
            'total_documents': len(self.documents),
            'by_type': type_counts,
            'index_size': self.index.ntotal
        }

