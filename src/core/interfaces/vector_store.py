from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from ..models.chunk import Chunk

class VectorStore(ABC):
    @abstractmethod
    async def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> List[str]:
        """Store chunks with their vectors"""
        pass
    
    @abstractmethod
    async def similarity_search(self, query_embedding: List[float], k: int = 10) -> List[Tuple[Chunk, float]]:
        """Find similar chunks"""
        pass