from typing import List, Dict, Any, Optional, Tuple
import chromadb
from pathlib import Path
from ...core.interfaces.vector_store import VectorStore
from ...core.models.chunk import Chunk
from ...utils.logger import get_logger

logger = get_logger(__name__)

class ChromaService(VectorStore):
    def __init__(self, persist_directory: str = "./data/chroma_db"):
        self.persist_directory = Path(persist_directory)
        self.client = None
        self.collection = None
    
    async def initialize(self) -> None:
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
    
    async def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> List[str]:
        ids = [chunk.id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        return ids
    
    async def similarity_search(self, query_embedding: List[float], k: int = 10) -> List[Tuple[Chunk, float]]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        chunks = []
        if results['ids'] and results['ids'][0]:
            for i, doc_id in enumerate(results['ids'][0]):
                chunk = Chunk(
                    id=doc_id,
                    document_id=results['metadatas'][0][i].get('document_id', ''),
                    text=results['documents'][0][i],
                    metadata=results['metadatas'][0][i]
                )
                chunks.append((chunk, results['distances'][0][i]))
        return chunks