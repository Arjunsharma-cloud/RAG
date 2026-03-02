class RAGException(Exception):
    """Base exception for our system"""
    pass

class ConfigurationError(RAGException):
    pass

class ProcessingError(RAGException):
    pass

class EmbeddingError(RAGException):
    pass

class VectorStoreError(RAGException):
    pass

class LLMError(RAGException):
    pass