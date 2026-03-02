from typing import List, Dict, Any, Optional, Tuple , cast
import chromadb
from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi  # You'll need to add this to requirements.txt
import asyncio

from ...core.interfaces.vector_store import VectorStore
from ...core.models.chunk import Chunk
from ...utils.logger import get_logger

logger = get_logger(__name__)

class ChromaService(VectorStore):
    def __init__(self, persist_directory: str = "./data/chroma_db"):
        self.persist_directory = Path(persist_directory)
        self.client = None
        self.collection = None
        self.bm25_index = None  # For keyword search
        self.all_chunks = []    # Store chunks for BM25
        self.bm25_ready = asyncio.Event()  # To signal when BM25 is built
    
    async def initialize(self) -> None:
        """Initialize ChromaDB client and collection"""
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("ChromaDB initialized")
    
    async def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> List[str]:
        """Add chunks to vector store and update BM25 index"""
        ids = [chunk.id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        # Add to ChromaDB
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        # Update local cache for BM25
        self.all_chunks.extend(chunks)
        await self._rebuild_bm25()
        
        logger.info(f"Added {len(chunks)} chunks to vector store")
        return ids
    
    async def similarity_search(
        self, 
        query_embedding: List[float], 
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Chunk, float]]:
        """Pure semantic search using embeddings"""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=filter
        )
        
        chunks = []
        if results['ids'] and results['ids'][0]:
            for i, doc_id in enumerate(results['ids'][0]):
                chunk = Chunk(
                    id=doc_id,
                    document_id=results['metadatas'][0][i].get('document_id', ''),
                    text=results['documents'][0][i],
                    metadata=results['metadatas'][0][i]
                )
                # Convert distance to similarity score (1 - distance for cosine)
                similarity = 1 - results['distances'][0][i] if results['distances'] else 0
                chunks.append((chunk, similarity))
        
        return chunks
    
    async def keyword_search(self , query: str, k: int = 10) -> List[Tuple[Chunk, float]]:
        """Pure keyword search using BM25"""
        if not self.bm25_index or not self.all_chunks:
            logger.warning("BM25 index not built, returning empty results")
            return []
        
        # Tokenize query
        query_tokens = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25_index.get_scores(query_tokens)
        
        # Get top k indices
        top_indices = np.argsort(scores)[-k:][::-1]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include if score > 0
                results.append((self.all_chunks[idx], float(scores[idx])))
        
        return results
    
    async def hybrid_search(
        self,
        query: str,
        query_embedding: List[float],
        k: int = 10,
        alpha: float = 0.5,  # Weight between semantic (alpha) and keyword (1-alpha)
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Chunk, float]]:
        """
        Hybrid search combining semantic and keyword search.
        
        Args:
            query: Original text query for keyword search
            query_embedding: Embedding for semantic search
            k: Number of results to return
            alpha: Weight for semantic search (0 = pure keyword, 1 = pure semantic)
            filter: Metadata filter for ChromaDB
        
        Returns:
            List of (chunk, score) tuples
        """
        logger.info(f"Performing hybrid search with alpha={alpha}")
        
        # Get more results for merging
        semantic_k = min(k * 2, 50)
        keyword_k = min(k * 2, 50)
        
        # Run both searches in parallel
        semantic_task = self.similarity_search(query_embedding, semantic_k, filter)
        keyword_task = self.keyword_search(query, keyword_k)
        
        semantic_results, keyword_results = await asyncio.gather(
            semantic_task, keyword_task, return_exceptions=True
        )
        
        # semantic_results_list: List[Tuple[Chunk, float]] = []
        # keyword_results_list: List[Tuple[Chunk, float]] = []

        # Handle potential errors
        if isinstance(semantic_results, Exception):
            logger.error(f"Semantic search failed: {semantic_results}")
            semantic_results = []
        else:
            semantic_results = cast(List[Tuple[Chunk, float]], semantic_results)
        

        if isinstance(keyword_results, Exception):
            logger.error(f"Keyword search failed: {keyword_results}")
            keyword_results = []
        else:
            keyword_results = cast(List[Tuple[Chunk, float]], keyword_results)
        

        # Type assertions for the editor
        # assert not isinstance(semantic_results, Exception)
        # assert not isinstance(keyword_results, Exception)
    
        # Now semantic_results and keyword_results are properly typed
        # semantic_list: List[Tuple[Chunk, float]] = semantic_results
        # keyword_list: List[Tuple[Chunk, float]] = keyword_results
        
        # Normalize and combine scores
        combined_scores = {}
        
        # Process semantic results
        semantic_scores = self._normalize_scores([score for _, score in semantic_results])
        for (chunk, _), norm_score in zip(semantic_results, semantic_scores):
            chunk_id = chunk.id
            if chunk_id not in combined_scores:
                combined_scores[chunk_id] = {'chunk': chunk, 'semantic': norm_score, 'keyword': 0}
            else:
                combined_scores[chunk_id]['semantic'] = norm_score
        
        # Process keyword results
        keyword_scores = self._normalize_scores([score for _, score in keyword_results])
        for (chunk, _), norm_score in zip(keyword_results, keyword_scores):
            chunk_id = chunk.id
            if chunk_id not in combined_scores:
                combined_scores[chunk_id] = {'chunk': chunk, 'semantic': 0, 'keyword': norm_score}
            else:
                combined_scores[chunk_id]['keyword'] = norm_score
        
        # Calculate hybrid scores
        hybrid_results = []
        for chunk_id, scores in combined_scores.items():
            # Weighted combination
            hybrid_score = (alpha * scores['semantic']) + ((1 - alpha) * scores['keyword'])
            hybrid_results.append((scores['chunk'], hybrid_score))
        
        # Sort by hybrid score and return top k
        hybrid_results.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Hybrid search returned {len(hybrid_results[:k])} results")
        return hybrid_results[:k]
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete all chunks for a document"""
        try:
            self.collection.delete(where={"document_id": document_id})
            # Also remove from local cache
            self.all_chunks = [c for c in self.all_chunks if c.document_id != document_id]
            await self._rebuild_bm25()
            logger.info(f"Deleted document {document_id} from vector store")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            return False
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            count = self.collection.count()
            return {
                "name": "documents",
                "count": count,
                "persist_directory": str(self.persist_directory),
                "bm25_ready": self.bm25_index is not None
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}
    
    async def _rebuild_bm25(self):
        """Rebuild BM25 index from all chunks"""
        if not self.all_chunks:
            self.bm25_index = None
            return
        
        try:
            # Tokenize all documents
            tokenized_docs = [chunk.text.lower().split() for chunk in self.all_chunks]
            self.bm25_index = BM25Okapi(tokenized_docs)
            self.bm25_ready.set()
            logger.info(f"BM25 index rebuilt with {len(self.all_chunks)} documents")
        except Exception as e:
            logger.error(f"Failed to rebuild BM25 index: {e}")
            self.bm25_index = None
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to [0, 1] range"""
        if not scores:
            return scores
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score - min_score == 0:
            return [1.0 for _ in scores]
        
        return [(s - min_score) / (max_score - min_score) for s in scores]