from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from .base_chunker import BaseChunker
from ...services.embedding.bge_service import BGEEmbeddingService
from ...utils.logger import get_logger
from .recursive_chunker import RecursiveChunker

logger = get_logger(__name__)

class SemanticChunker(BaseChunker):
    """
    Advanced chunker that uses embeddings to find semantic boundaries.
    Groups sentences that are semantically similar together.
    """
    
    def __init__(
        self, 
        chunk_size: int = 512, 
        chunk_overlap: int = 50, 
        embedding_service: Optional[BGEEmbeddingService] = None,
        similarity_threshold: float = 0.7,
        max_sentences_per_chunk: int = 20
    ):
        super().__init__(chunk_size, chunk_overlap)
        self.embedding_service = embedding_service
        self.similarity_threshold = similarity_threshold
        self.max_sentences_per_chunk = max_sentences_per_chunk
        
        # Import nltk for sentence tokenization
        self._init_nltk()
    
    def _init_nltk(self):
        """Initialize NLTK for sentence tokenization"""
        try:
            import nltk
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            import nltk
            nltk.download('punkt', quiet=True)
        except ImportError:
            logger.warning("NLTK not installed, falling back to simple sentence splitting")
    
    async def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Split text into semantic chunks based on embedding similarity.
        """
        if not text or not text.strip():
            return []
        
        # Clean the text
        text = text.strip()
        
        # If text is small enough, return as single chunk
        if len(text) <= self.chunk_size:
            return [text]
        
        # Split into sentences
        sentences = self._split_sentences(text)
        
        if len(sentences) <= 1:
            return [text]
        
        # If embedding service is available, use semantic chunking
        if self.embedding_service:
            try:
                chunks = await self._semantic_chunking(sentences)
            except Exception as e:
                logger.warning(f"Semantic chunking failed, falling back to recursive: {e}")
                chunks = await self._fallback_chunking(text)
        else:
            # Fallback to recursive chunking
            chunks = await self._fallback_chunking(text)
        
        # Merge very small chunks
        chunks = self._merge_small_chunks(chunks)
        
        return chunks
    
    async def _semantic_chunking(self, sentences: List[str]) -> List[str]:
        """
        Perform semantic chunking using embeddings.
        """
        # Get embeddings for all sentences
        embeddings = await self.embedding_service.embed(sentences)
        
        # Find semantic boundaries
        boundaries = self._find_semantic_boundaries(embeddings)
        
        # Create chunks from boundaries
        chunks = self._create_chunks_from_boundaries(sentences, boundaries)
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        """
        try:
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(text)
            return [s.strip() for s in sentences if s.strip()]
        except:
            # Simple fallback sentence splitting
            import re
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]
    
    def _find_semantic_boundaries(self, embeddings: List[List[float]]) -> List[int]:
        """
        Find indices where semantic shift occurs between sentences.
        Returns indices where new chunks should start.
        """
        boundaries = [0]  # Always start at first sentence
        consecutive_low = 0
        
        for i in range(1, len(embeddings)):
            # Calculate similarity between current and previous sentence
            sim = self._cosine_similarity(embeddings[i-1], embeddings[i])
            
            # If similarity is below threshold, it might be a boundary
            if sim < self.similarity_threshold:
                consecutive_low += 1
            else:
                consecutive_low = 0
            
            # Create boundary if we have multiple consecutive low similarities
            # OR if we're approaching max sentences per chunk
            last_boundary = boundaries[-1]
            sentences_since_boundary = i - last_boundary
            
            if (consecutive_low >= 2 or 
                sentences_since_boundary >= self.max_sentences_per_chunk):
                boundaries.append(i)
                consecutive_low = 0
        
        # Add end boundary
        boundaries.append(len(embeddings))
        
        return boundaries
    
    def _create_chunks_from_boundaries(self, sentences: List[str], boundaries: List[int]) -> List[str]:
        """
        Create chunks from sentence boundaries.
        """
        chunks = []
        
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i+1]
            chunk_sentences = sentences[start:end]
            
            if not chunk_sentences:
                continue
            
            chunk_text = ' '.join(chunk_sentences)
            
            # If chunk exceeds size limit, recursively split it
            if len(chunk_text) > self.chunk_size:
                # Create a recursive chunker for sub-splitting
                recursive = RecursiveChunker(self.chunk_size, self.chunk_overlap)
                # Note: In production, you'd want to properly await this
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # This is a simplification - in production, handle properly
                    sub_chunks = recursive._recursive_split(chunk_text, ["\n\n", "\n", ". ", " "])
                else:
                    sub_chunks = recursive._recursive_split(chunk_text, ["\n\n", "\n", ". ", " "])
                chunks.extend(sub_chunks)
            else:
                chunks.append(chunk_text)
        
        return chunks
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        """
        a = np.array(a)
        b = np.array(b)
        
        # Handle zero vectors
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    async def _fallback_chunking(self, text: str) -> List[str]:
        """
        Fallback to recursive chunking when semantic chunking fails.
        """
        chunker = RecursiveChunker(self.chunk_size, self.chunk_overlap)
        return await chunker.chunk(text)