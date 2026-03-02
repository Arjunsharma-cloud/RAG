import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from ..core.models.document import Document, DocumentType, ProcessingStatus
from ..processors import PDFProcessor, CSVProcessor
from ..services.embedding.bge_service import BGEEmbeddingService
from ..services.vector_store.chroma_service import ChromaService
from ..utils.logger import get_logger
from ..utils.exceptions import ProcessingError

logger = get_logger(__name__)

class IngestionPipeline:
    def __init__(
        self,
        embedding_service: BGEEmbeddingService,
        vector_store: ChromaService,
        pdf_processor: PDFProcessor,
        csv_processor: CSVProcessor,
        batch_size: int = 32
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.processors = {
            DocumentType.PDF: pdf_processor,
            DocumentType.CSV: csv_processor
        }
        self.batch_size = batch_size
    
    async def process_document(self, file_path: str, doc_type: DocumentType) -> Document:
        """Process a single document through the pipeline"""
        document = Document(type=doc_type, source_path=file_path)
        processor = self.processors[doc_type]
        
        try:
            logger.info(f"Processing {doc_type.value}: {file_path}")
            
            # Process into chunks
            chunks = []
            async for chunk in processor.process(document):
                chunks.append(chunk)
            
            if not chunks:
                logger.warning(f"No chunks generated for {file_path}")
                document.status = ProcessingStatus.COMPLETED
                return document
            
            # Generate embeddings in batches
            chunk_texts = [chunk.text for chunk in chunks]
            embeddings = []
            
            for i in range(0, len(chunk_texts), self.batch_size):
                batch = chunk_texts[i:i + self.batch_size]
                batch_embeddings = await self.embedding_service.embed(batch)
                embeddings.extend(batch_embeddings)
            
            # Store in vector DB
            await self.vector_store.add_chunks(chunks, embeddings)
            
            document.status = ProcessingStatus.COMPLETED
            document.processed_at = datetime.now()
            logger.info(f"Successfully processed: {file_path} - {len(chunks)} chunks")
            
        except Exception as e:
            document.status = ProcessingStatus.FAILED
            document.error_message = str(e)
            logger.error(f"Failed to process {file_path}: {e}")
            raise ProcessingError(f"Document processing failed: {e}")
        
        return document
    
    async def process_directory(self, directory: str) -> Dict[str, List[Document]]:
        """Process all supported files in a directory"""
        path = Path(directory)
        results = {}
        
        # Process PDFs
        pdf_files = list(path.glob("*.pdf"))
        if pdf_files:
            docs = []
            for pdf_file in pdf_files:
                try:
                    doc = await self.process_document(str(pdf_file), DocumentType.PDF)
                    docs.append(doc)
                except Exception as e:
                    logger.error(f"Error processing {pdf_file}: {e}")
            results['pdf'] = docs
        
        # Process CSVs
        csv_files = list(path.glob("*.csv"))
        if csv_files:
            docs = []
            for csv_file in csv_files:
                try:
                    doc = await self.process_document(str(csv_file), DocumentType.CSV)
                    docs.append(doc)
                except Exception as e:
                    logger.error(f"Error processing {csv_file}: {e}")
            results['csv'] = docs
        
        return results