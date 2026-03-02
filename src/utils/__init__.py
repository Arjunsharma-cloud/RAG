"""
Utils Module

Provides shared utilities for:
- Logging
- Exception handling
- Text normalization
- Async helpers

This module defines the public API surface for utils.
"""

# Logging
from .logger import setup_logger, get_logger

# Custom Exceptions
from .exceptions import (
    RAGException,
    ProcessingError,
    EmbeddingError,
    VectorStoreError,
    LLMError,
)

# Text Utilities
from .text_normalizer import TextNormalizer

# Async Utilities (if present)
from .async_utils import run_async, gather_with_concurrency


# Explicit public API
__all__ = [
    # Logging
    "setup_logger",
    "get_logger",

    # Exceptions
    "RAGException",
    "ProcessingError",
    "EmbeddingError",
    "VectorStoreError",
    "LLMError",

    # Text
    "TextNormalizer",

    # Async
    "run_async",
    "gather_with_concurrency",
]