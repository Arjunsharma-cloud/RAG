"""
Utility modules for the Multimodal RAG System.
Provides logging, exception handling, text normalization, and helper functions.
"""

from .logger import setup_logger, get_logger
from .exceptions import (
    RAGException, 
    ConfigurationError,
    ProcessingError, 
    EmbeddingError, 
    VectorStoreError, 
    LLMError,

)
from .text_normalizer import TextNormalizer
from .async_utils import (  # Now this import will work!
    async_timed,
    gather_with_concurrency,
    run_async_in_thread,
    AsyncTaskManager,
    run_parallel,
    create_task
)

__all__ = [
    # Logging
    'setup_logger',
    'get_logger',
    
    # Exceptions
    'RAGException',
    'ConfigurationError',
    'ProcessingError',
    'EmbeddingError',
    'VectorStoreError',
    'LLMError',
    
    # Text Processing
    'TextNormalizer',
    
    # Async Utilities
    'async_timed',
    'gather_with_concurrency',
    'run_async_in_thread',
    'AsyncTaskManager',
    'run_parallel',
    'create_task'
]