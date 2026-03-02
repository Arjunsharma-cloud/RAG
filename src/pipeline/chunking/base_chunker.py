from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseChunker(ABC):
    """
    Abstract base class for all chunking strategies.
    Defines the interface that all chunkers must implement.
    """
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Initialize the chunker with size and overlap parameters.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    @abstractmethod
    async def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Split text into chunks.
        
        Args:
            text: The text to split into chunks
            metadata: Optional metadata for context (document_id, etc.)
            
        Returns:
            List of text chunks
        """
        pass
    
    def _merge_small_chunks(self, chunks: List[str], min_size: int = 100) -> List[str]:
        """
        Merge chunks that are too small with adjacent chunks.
        
        Args:
            chunks: List of text chunks
            min_size: Minimum size threshold for chunks
            
        Returns:
            Merged chunks
        """
        if not chunks:
            return []
        
        merged = []
        current = []
        current_size = 0
        
        for chunk in chunks:
            chunk_size = len(chunk)
            
            # If current chunk is too small and we have accumulated chunks
            if chunk_size < min_size and current:
                current.append(chunk)
                current_size += chunk_size
            else:
                # Save current accumulated chunks
                if current:
                    merged.append(' '.join(current))
                    current = []
                    current_size = 0
                
                # If this chunk is still small, start new accumulation
                if chunk_size < min_size:
                    current = [chunk]
                    current_size = chunk_size
                else:
                    merged.append(chunk)
        
        # Add any remaining accumulated chunks
        if current:
            merged.append(' '.join(current))
        
        return merged