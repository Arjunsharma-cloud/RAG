import asyncio
import pandas as pd
from pathlib import Path
from typing import AsyncGenerator
from ..core.interfaces.document_processor import DocumentProcessor
from ..core.models.document import Document,  ProcessingStatus
from ..core.models.chunk import Chunk
from ..pipeline.chunking.base_chunker import BaseChunker
from ..utils.logger import get_logger


logger = get_logger(__name__)

class CSVProcessor(DocumentProcessor):
    def __init__(self, chunker: BaseChunker):
        self.chunker = chunker
    
    async def validate(self, document: Document) -> bool:
        return Path(document.source_path).suffix.lower() == '.csv'
    
    async def extract_metadata(self, document: Document) -> dict:
        path = Path(document.source_path)
        df = pd.read_csv(path, nrows=0)
        return {
            'columns': list(df.columns),
            'num_columns': len(df.columns),
            'file_name': path.name
        }
    
    async def process(self, document: Document) -> AsyncGenerator[Chunk, None]:
        try:
            logger.info(f"Processing CSV: {document.source_path}")
            text = await self._process_csv(document.source_path)
            
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
    
    async def _process_csv(self, path: str) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_process_csv, path)
    
    def _sync_process_csv(self, path: str) -> str:
        df = pd.read_csv(path)
        lines = [f"CSV File: {Path(path).name}", f"Rows: {len(df)}"]
        for idx, row in df.iterrows():
            lines.append(f"Row {idx + 1}: " + ", ".join([f"{col}: {val}" for col, val in row.items()]))
        return '\n'.join(lines)