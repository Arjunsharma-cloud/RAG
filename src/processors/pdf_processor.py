import asyncio
from typing import AsyncGenerator
import pypdf
from pathlib import Path
from ..core.interfaces.document_processor import DocumentProcessor
from ..core.models.document import Document,  ProcessingStatus
from ..core.models.chunk import Chunk
from ..pipeline.chunking.base_chunker import BaseChunker
from ..utils.logger import get_logger

logger = get_logger(__name__)

class PDFProcessor(DocumentProcessor):
    def __init__(self, chunker: BaseChunker):
        self.chunker = chunker
    
    async def validate(self, document: Document) -> bool:
        return Path(document.source_path).suffix.lower() == '.pdf'
    
    async def extract_metadata(self, document: Document) -> dict:
        path = Path(document.source_path)
        with open(path, 'rb') as f:
            reader = pypdf.PdfReader(f)
            return {
                'num_pages': len(reader.pages),
                'file_name': path.name,
                'file_size': path.stat().st_size
            }
    
    async def process(self, document: Document) -> AsyncGenerator[Chunk, None]:
        try:
            logger.info(f"Processing PDF: {document.source_path}")
            text = await self._extract_text(document.source_path)
            
            document.content = text
            document.metadata.update(await self.extract_metadata(document))
            
            chunks = await self.chunker.chunk(text, {'document_id': document.id})
            
            for i, chunk_text in enumerate(chunks):
                yield Chunk(
                    document_id=document.id,
                    text=chunk_text,
                    metadata={**document.metadata, 'chunk_index': i},
                    index=i
                )
            
            document.status = ProcessingStatus.COMPLETED
        except Exception as e:
            document.status = ProcessingStatus.FAILED
            document.error_message = str(e)
            raise
    
    async def _extract_text(self, path: str) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_extract_text, path)
    
    def _sync_extract_text(self, path: str) -> str:
        text_parts = []
        with open(path, 'rb') as f:
            reader = pypdf.PdfReader(f)
            for page in reader.pages:
                text = page.extract_text()
                if text.strip():
                    text_parts.append(text)
        return '\n\n'.join(text_parts)