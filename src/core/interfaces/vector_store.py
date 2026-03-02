from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from ..models.chunk import Chunk

class VectorStore(ABC):
    @abstractmethod
    async def initialize(self) -> None:
        pass
    
    @abstractmethod
    async def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> List[str]:
        pass
    
    @abstractmethod
    async def similarity_search(
        self, 
        query_embedding: List[float], 
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Chunk, float]]:
        pass
    
    @abstractmethod
    async def keyword_search(
        self, 
        query: str, 
        k: int = 10
    ) -> List[Tuple[Chunk, float]]:
        """NEW: Pure keyword search"""
        pass
    
    @abstractmethod
    async def hybrid_search(
        self,
        query: str,
        query_embedding: List[float],
        k: int = 10,
        alpha: float = 0.5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Chunk, float]]:
        """NEW: Hybrid search combining semantic and keyword"""
        pass