from typing import List, Dict, Any, Optional
from .base_chunker import BaseChunker

class RecursiveChunker(BaseChunker):
    """
    Recursively splits text by natural boundaries (paragraphs, sentences, words).
    Falls back to character splitting when no more boundaries exist.
    """
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        super().__init__(chunk_size, chunk_overlap)
        # Priority order of separators (from largest to smallest)
        self.separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
    
    async def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Recursively split text into chunks.
        """
        if not text or not text.strip():
            return []
        
        # Clean the text
        text = text.strip()
        
        # If text is already small enough, return as single chunk
        if len(text) <= self.chunk_size:
            return [text]
        
        # Start recursive splitting
        chunks = self._recursive_split(text, self.separators.copy())
        
        # Merge very small chunks
        chunks = self._merge_small_chunks(chunks)
        
        return chunks
    
    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """
        Internal recursive splitting logic.
        """
        # If text is small enough, return as is
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []
        
        # If no more separators, split by characters
        if not separators:
            return self._split_by_chars(text)
        
        # Get current separator
        separator = separators[0]
        remaining_separators = separators[1:]
        
        # Split by current separator
        if separator:
            parts = text.split(separator)
        else:
            # Empty separator means split by characters
            return self._split_by_chars(text)
        
        # If splitting didn't produce multiple parts, try next separator
        if len(parts) == 1:
            return self._recursive_split(text, remaining_separators)
        
        # Build chunks
        chunks = []
        current_chunk = []
        current_size = 0
        
        for part in parts:
            # Add separator back (except for empty separator case)
            part_with_sep = part + separator if separator else part
            part_size = len(part_with_sep)
            
            # If this part alone is too big, recursively split it
            if part_size > self.chunk_size:
                # First, save current chunk if it exists
                if current_chunk:
                    chunks.append(''.join(current_chunk).strip())
                    current_chunk = []
                    current_size = 0
                
                # Recursively split the large part
                sub_chunks = self._recursive_split(part, remaining_separators)
                chunks.extend(sub_chunks)
                continue
            
            # Check if adding this part would exceed chunk size
            if current_size + part_size <= self.chunk_size:
                current_chunk.append(part_with_sep)
                current_size += part_size
            else:
                # Save current chunk
                if current_chunk:
                    chunks.append(''.join(current_chunk).strip())
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap(current_chunk, self.chunk_overlap)
                if overlap_text:
                    current_chunk = [overlap_text, part_with_sep]
                    current_size = len(overlap_text) + part_size
                else:
                    current_chunk = [part_with_sep]
                    current_size = part_size
        
        # Add the last chunk
        if current_chunk:
            chunks.append(''.join(current_chunk).strip())
        
        return chunks
    
    def _split_by_chars(self, text: str) -> List[str]:
        """
        Split text by character count when no natural boundaries exist.
        """
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            # Calculate end position with overlap
            end = min(start + self.chunk_size, text_len)
            
            # Extract chunk
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position, accounting for overlap
            start = end - self.chunk_overlap
        
        return chunks
    
    def _get_overlap(self, chunk_parts: List[str], overlap_size: int) -> str:
        """
        Get overlap text from the end of previous chunk.
        """
        if not chunk_parts:
            return ""
        
        full_text = ''.join(chunk_parts)
        if len(full_text) <= overlap_size:
            return full_text
        
        # Get last `overlap_size` characters
        return full_text[-overlap_size:]