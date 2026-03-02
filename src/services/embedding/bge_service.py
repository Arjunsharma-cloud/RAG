from typing import List
import asyncio
from sentence_transformers import SentenceTransformer
from ...core.interfaces.embedding_service import EmbeddingService
from ...utils.logger import get_logger
from ...utils.exceptions import EmbeddingError

logger = get_logger(__name__)

class BGEEmbeddingService(EmbeddingService):
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        self.model_name = model_name
        self.model = None
    
    async def initialize(self) -> None:
        try:
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None, 
                lambda: SentenceTransformer(self.model_name)
            )
            logger.info(f"Initialized embedding model")
        except Exception as e:
            raise EmbeddingError(f"Failed to initialize: {e}")
    
    async def embed(self, texts: List[str]) -> List[List[float]]:
        if not self.model:
            raise EmbeddingError("Model not initialized")
        
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self.model.encode(texts, normalize_embeddings=True).tolist()
        )
        return embeddings
    
    async def embed_query(self, query: str) -> List[float]:
        embeddings = await self.embed([query])
        return embeddings[0]