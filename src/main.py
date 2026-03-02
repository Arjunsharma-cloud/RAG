import asyncio
from typing import Optional, Dict, Any, List
from pathlib import Path

from config.settings import load_config
from .core.models.document import Document, DocumentType
from .processors import PDFProcessor, CSVProcessor
from .services.embedding.bge_service import BGEEmbeddingService
from .services.vector_store.chroma_service import ChromaService
from .services.llm.ollama_service import OllamaService
from .services.llm.openrouter_service import OpenRouterService
from .services.memory.session_memory import SessionMemory
from .services.reranker.bge_reranker import BGEReranker
from .pipeline.ingestion_pipeline import IngestionPipeline
from .pipeline.query_pipeline import QueryPipeline
from .pipeline.chunking.recursive_chunker import RecursiveChunker
from .pipeline.chunking.semantic_chunker import SemanticChunker
from .utils.logger import setup_logger, get_logger
from .utils.exceptions import ConfigurationError

logger = get_logger(__name__)

class MultimodalRAGSystem:
    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path)
        self.initialized = False
        self.embedding_service = None
        self.vector_store = None
        self.llm_service = None
        self.memory_store = None
        self.reranker = None
        self.ingestion_pipeline = None
        self.query_pipeline = None
        self.pdf_processor = None
        self.csv_processor = None
    
    async def initialize(self):
        """Initialize all components"""
        logger.info("Initializing RAG System")
        
        try:
            # Create directories
            self.config.DATA_DIR.mkdir(parents=True, exist_ok=True)
            self.config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
            
            # Initialize embedding service
            self.embedding_service = BGEEmbeddingService(self.config.EMBEDDING_MODEL)
            await self.embedding_service.initialize()
            
            # Initialize vector store
            self.vector_store = ChromaService(str(self.config.CHROMA_DIR))
            await self.vector_store.initialize()
            
            # Initialize LLM service based on provider
            if self.config.LLM_PROVIDER == "openrouter":
                if not self.config.OPENROUTER_API_KEY:
                    raise ConfigurationError("OPENROUTER_API_KEY is required for OpenRouter provider")
                
                logger.info(f"Using OpenRouter with model: {self.config.OPENROUTER_MODEL}")
                self.llm_service = OpenRouterService(
                    api_key=self.config.OPENROUTER_API_KEY,
                    model=self.config.OPENROUTER_MODEL,
                    base_url=self.config.OPENROUTER_BASE_URL,
                    temperature=self.config.LLM_TEMPERATURE,
                    max_tokens=self.config.LLM_MAX_TOKENS,
                    site_url=self.config.OPENROUTER_SITE_URL,
                    site_name=self.config.OPENROUTER_SITE_NAME
                )
            else:
                logger.info(f"Using Ollama with model: {self.config.LLM_MODEL}")
                self.llm_service = OllamaService(
                    model=self.config.LLM_MODEL,
                    base_url=self.config.LLM_BASE_URL
                )
            
            # Initialize memory store
            self.memory_store = SessionMemory(
                ttl=self.config.MEMORY_TTL,
                max_turns=self.config.MAX_HISTORY_TURNS
            )
            
            # Initialize reranker if enabled
            if self.config.USE_RERANKER:
                self.reranker = BGEReranker(self.config.RERANKER_MODEL)
                await self.reranker.initialize()
            
            # Create chunker based on strategy
            if self.config.CHUNKING_STRATEGY == 'semantic':
                logger.info("Using Semantic Chunker for better topic coherence")
                chunker = SemanticChunker(
                    chunk_size=self.config.CHUNK_SIZE,
                    chunk_overlap=self.config.CHUNK_OVERLAP,
                    embedding_service=self.embedding_service,
                    similarity_threshold=0.7
                )
            else:
                logger.info("Using Recursive Chunker")
                chunker = RecursiveChunker(
                    chunk_size=self.config.CHUNK_SIZE,
                    chunk_overlap=self.config.CHUNK_OVERLAP
                )
            
            # Create processors
            self.pdf_processor = PDFProcessor(chunker)
            self.csv_processor = CSVProcessor(chunker)
            
            # Create pipelines
            self.ingestion_pipeline = IngestionPipeline(
                self.embedding_service,
                self.vector_store,
                self.pdf_processor,
                self.csv_processor,
                batch_size=self.config.BATCH_SIZE
            )
            
            self.query_pipeline = QueryPipeline(
                self.embedding_service,
                self.vector_store,
                self.llm_service,
                self.memory_store,
                self.reranker,
                top_k=self.config.TOP_K_RESULTS,
                use_hybrid_search=True
            )
            
            self.initialized = True
            logger.info("System ready!")
            
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            raise ConfigurationError(f"Initialization failed: {e}")
    
    async def process_document(self, file_path: str) -> Document:
        """Process a single document"""
        if not self.initialized:
            await self.initialize()
        
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine type
        if path.suffix.lower() == '.pdf':
            doc_type = DocumentType.PDF
        elif path.suffix.lower() == '.csv':
            doc_type = DocumentType.CSV
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        
        return await self.ingestion_pipeline.process_document(file_path, doc_type)
    
    async def process_directory(self, directory: str) -> Dict[str, List[Document]]:
        """Process all documents in a directory"""
        if not self.initialized:
            await self.initialize()
        return await self.ingestion_pipeline.process_directory(directory)
    
    async def query(self, user_query: str, session_id: str = "default",
                   filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Query the system"""
        if not self.initialized:
            await self.initialize()
        return await self.query_pipeline.query(user_query, session_id, filters)
    
    async def get_conversation_history(self, session_id: str):
        """Get conversation history"""
        if not self.initialized:
            await self.initialize()
        return await self.memory_store.get_conversation(session_id)
    
    async def clear_conversation(self, session_id: str) -> bool:
        """Clear conversation history"""
        if not self.initialized:
            await self.initialize()
        return await self.memory_store.delete_conversation(session_id)
    
    async def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        return {
            'embedding_model': self.config.EMBEDDING_MODEL,
            'llm_model': self.config.LLM_MODEL if self.config.LLM_PROVIDER == 'ollama' else self.config.OPENROUTER_MODEL,
            'llm_provider': self.config.LLM_PROVIDER,
            'chunk_size': self.config.CHUNK_SIZE,
            'chunking_strategy': self.config.CHUNKING_STRATEGY,
            'reranker_enabled': self.reranker is not None,
            'vector_store': 'ChromaDB'
        }
    
    async def close(self):
        """Clean up resources"""
        if self.llm_service:
            await self.llm_service.close()
        logger.info("System shutdown")