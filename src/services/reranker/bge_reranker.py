"""
BGE Reranker service for improving search result quality.
Reranks chunks based on their relevance to the query.
"""
from typing import List, Optional
import asyncio
from sentence_transformers import CrossEncoder

from ...core.models.chunk import Chunk
from ...utils.logger import get_logger
from ...utils.exceptions import RAGException

logger = get_logger(__name__)

class BGEReranker:
    """
    BGE Cross-Encoder reranker for improving search result relevance.
    
    This reraker uses a cross-encoder model to score query-chunk pairs,
    providing more accurate relevance scores than embedding similarity alone.
    """
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: str = "cpu"):
        """
        Initialize the reranker.
        
        Args:
            model_name: HuggingFace model name for the reranker
            device: Device to run the model on ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.device = device
        self.model: Optional[CrossEncoder] = None
        self.initialized = False
    
    async def initialize(self) -> None:
        """
        Initialize the reranker model.
        This loads the CrossEncoder model from HuggingFace.
        """
        try:
            logger.info(f"Loading reranker model: {self.model_name}")
            
            # Run model loading in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None,
                lambda: CrossEncoder(
                    self.model_name,
                    device=self.device,
                    max_length=512  # Limit sequence length for efficiency
                )
            )
            
            self.initialized = True
            logger.info(f"✅ Reranker model loaded: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}")
            raise RAGException(f"Reranker initialization failed: {e}")
    
    async def rerank(
        self, 
        query: str, 
        chunks: List[Chunk], 
        top_k: Optional[int] = None,
        batch_size: int = 32
    ) -> List[Chunk]:
        """
        Rerank chunks based on relevance to the query.
        
        Args:
            query: The user's query
            chunks: List of chunks to rerank
            top_k: Number of top chunks to return (None returns all)
            batch_size: Batch size for processing
            
        Returns:
            List of chunks sorted by relevance (most relevant first)
        """
        if not self.initialized or not self.model:
            logger.warning("Reranker not initialized, returning original order")
            return chunks[:top_k] if top_k else chunks
        
        if not chunks:
            return []
        
        try:
            logger.info(f"Reranking {len(chunks)} chunks for query: {query[:50]}...")
            
            # Prepare query-chunk pairs
            pairs = [[query, chunk.text] for chunk in chunks]
            
            # Get scores in batches
            all_scores = []
            loop = asyncio.get_event_loop()
            
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i + batch_size]
                batch_scores = await loop.run_in_executor(
                    None,
                    lambda: self.model.predict(batch_pairs).tolist()
                )
                all_scores.extend(batch_scores)
            
            # Pair chunks with scores and sort
            scored_chunks = list(zip(chunks, all_scores))
            scored_chunks.sort(key=lambda x: x[1], reverse=True)
            
            # Return top_k or all
            result_chunks = [chunk for chunk, _ in scored_chunks]
            if top_k:
                result_chunks = result_chunks[:top_k]
            
            logger.info(f"Reranking complete. Top score: {scored_chunks[0][1]:.4f}")
            return result_chunks
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Return original order on failure
            return chunks[:top_k] if top_k else chunks
    
    async def rerank_with_scores(
        self, 
        query: str, 
        chunks: List[Chunk], 
        top_k: Optional[int] = None,
        batch_size: int = 32
    ) -> List[tuple[Chunk, float]]:
        """
        Rerank chunks and return both chunks and their scores.
        
        Args:
            query: The user's query
            chunks: List of chunks to rerank
            top_k: Number of top chunks to return
            batch_size: Batch size for processing
            
        Returns:
            List of (chunk, score) tuples sorted by score
        """
        if not self.initialized or not self.model:
            logger.warning("Reranker not initialized")
            return [(chunk, 0.0) for chunk in chunks[:top_k]]
        
        if not chunks:
            return []
        
        try:
            # Prepare pairs
            pairs = [[query, chunk.text] for chunk in chunks]
            
            # Get scores in batches
            all_scores = []
            loop = asyncio.get_event_loop()
            
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i + batch_size]
                batch_scores = await loop.run_in_executor(
                    None,
                    lambda: self.model.predict(batch_pairs).tolist()
                )
                all_scores.extend(batch_scores)
            
            # Create scored pairs and sort
            scored_chunks = list(zip(chunks, all_scores))
            scored_chunks.sort(key=lambda x: x[1], reverse=True)
            
            # Return top_k
            if top_k:
                scored_chunks = scored_chunks[:top_k]
            
            return scored_chunks
            
        except Exception as e:
            logger.error(f"Reranking with scores failed: {e}")
            return [(chunk, 0.0) for chunk in chunks[:top_k]]
    
    async def batch_rerank(
        self,
        queries: List[str],
        chunks_list: List[List[Chunk]],
        top_k_per_query: Optional[int] = None
    ) -> List[List[Chunk]]:
        """
        Rerank multiple queries against their respective chunk lists.
        
        Args:
            queries: List of queries
            chunks_list: List of chunk lists for each query
            top_k_per_query: Number of top chunks to return per query
            
        Returns:
            List of reranked chunk lists
        """
        if not self.initialized or not self.model:
            logger.warning("Reranker not initialized")
            return [chunks[:top_k_per_query] for chunks in chunks_list]
        
        if len(queries) != len(chunks_list):
            raise ValueError("Number of queries must match number of chunk lists")
        
        results = []
        for query, chunks in zip(queries, chunks_list):
            reranked = await self.rerank(query, chunks, top_k_per_query)
            results.append(reranked)
        
        return results
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "initialized": self.initialized,
            "model_type": "CrossEncoder"
        }


# Optional: Simpler reranker without async for testing
class SimpleReranker:
    """
    A simple reranker that uses embedding similarity.
    Useful for testing when BGE reranker is not available.
    """
    
    def __init__(self):
        self.initialized = True
    
    async def rerank(self, query: str, chunks: List[Chunk], top_k: Optional[int] = None) -> List[Chunk]:
        """
        Simple reranking based on text length and query term matching.
        """
        if not chunks:
            return []
        
        # Simple scoring: count query term matches
        query_terms = set(query.lower().split())
        
        scored_chunks = []
        for chunk in chunks:
            chunk_terms = set(chunk.text.lower().split())
            matches = len(query_terms.intersection(chunk_terms))
            # Prefer chunks with more matches and reasonable length
            score = matches * (1.0 / max(1, abs(len(chunk.text) - 500) / 500))
            scored_chunks.append((chunk, score))
        
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        result = [chunk for chunk, _ in scored_chunks]
        return result[:top_k] if top_k else result