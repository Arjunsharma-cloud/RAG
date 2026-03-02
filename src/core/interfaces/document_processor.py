from abc import ABC, abstractmethod
from typing import AsyncGenerator
from ..models.document import Document
from ..models.chunk import Chunk

class DocumentProcessor(ABC):
    """Contract: Any document processor MUST do these things"""
    
    @abstractmethod
    async def process(self, document: Document) -> AsyncGenerator[Chunk, None]:
        """Take a document and turn it into chunks"""
        pass
    
    @abstractmethod
    async def validate(self, document: Document) -> bool:
        """Check if this processor can handle this document"""
        pass
    
    @abstractmethod
    async def extract_metadata(self, document: Document) -> dict:
        """Get metadata from the document"""
        pass