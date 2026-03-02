from abc import ABC, abstractmethod
from typing import List

class EmbeddingService(ABC):
    @abstractmethod
    async def initialize(self) -> None:
        pass
    
    @abstractmethod
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Convert text to vectors"""
        pass
    
    @abstractmethod
    async def embed_query(self, query: str) -> List[float]:
        pass