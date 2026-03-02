from typing import List, Dict, Any, Optional, Tuple
from ..services.embedding.bge_service import BGEEmbeddingService
from ..services.vector_store.chroma_service import ChromaService
from ..services.llm.ollama_service import OllamaService
from ..services.memory.session_memory import SessionMemory
from ..services.reranker.bge_reranker import BGEReranker
from ..core.models.conversation import Conversation, Message  # Explicit imports
from ..core.models.chunk import Chunk
from ..utils.logger import get_logger
from ..utils.text_normalizer import TextNormalizer

logger = get_logger(__name__)

class QueryPipeline:
    def __init__(
        self,
        embedding_service: BGEEmbeddingService,
        vector_store: ChromaService,
        llm_service: OllamaService,
        memory_store: SessionMemory,
        reranker: Optional[BGEReranker] = None,
        top_k: int = 5,
        use_hybrid_search: bool = True
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.llm_service = llm_service
        self.memory_store = memory_store
        self.reranker = reranker
        self.top_k = top_k
        self.use_hybrid_search = use_hybrid_search
        self.normalizer = TextNormalizer()
    
    async def query(self, user_query: str, session_id: str, 
                    filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a query through the pipeline"""
        
        # Normalize query
        user_query = self.normalizer.normalize(user_query)
        
        # Get conversation history - returns Conversation object or None
        conversation: Optional[Conversation] = await self.memory_store.get_conversation(session_id)
        
        # Enhance query with conversation context if needed
        enhanced_query = await self._enhance_query(user_query, conversation)
        
        # Generate query embedding
        query_embedding = await self.embedding_service.embed_query(enhanced_query)
        
        # Search for relevant chunks
        if self.use_hybrid_search:
            chunks_with_scores = await self.vector_store.hybrid_search(
                query=enhanced_query,
                query_embedding=query_embedding,
                k=self.top_k * 2,
                filter=filters
            )
        else:
            chunks_with_scores = await self.vector_store.similarity_search(
                query_embedding, self.top_k * 2, filter=filters
            )
        
        # Extract chunks from tuples
        chunks = [chunk for chunk, _ in chunks_with_scores]
        
        # Rerank if available
        if self.reranker and chunks:
            chunks = await self.reranker.rerank(enhanced_query, chunks, self.top_k)
        else:
            chunks = chunks[:self.top_k]
        
        # Prepare context from chunks
        context = self._prepare_context(chunks)
        
        # Generate response
        prompt = self._build_prompt(user_query, context, conversation)
        response = await self.llm_service.generate(prompt)
        
        # Save to memory
        await self.memory_store.add_message(
            session_id, 
            Message(role="user", content=user_query)
        )
        await self.memory_store.add_message(
            session_id, 
            Message(role="assistant", content=response, metadata={"sources": [c.to_dict() for c in chunks]})
        )
        
        return {
            "answer": response,
            "sources": [self._format_source(c) for c in chunks],
            "conversation_id": session_id
        }
    
    async def _enhance_query(self, query: str, conversation: Optional[Conversation]) -> str:
        """Enhance query with conversation context if needed"""
        # Check if we have a conversation with messages
        if conversation is None:
            return query
        
        # Get the messages list from conversation
        messages: List[Message] = conversation.messages
        
        # Need at least 2 messages for context (user + assistant)
        if len(messages) < 2:
            return query
        
        # Check if query needs context (pronouns, references)
        needs_context = any(word in query.lower() 
                          for word in ['it', 'this', 'that', 'they', 'them', 'the above', 'previous', 'above'])
        
        if needs_context:
            # Get last 2 messages (last exchange)
            last_messages: List[Message] = messages[-2:]
            
            # Build context string
            context_parts = []
            for msg in last_messages:
                role_prefix = "Human" if msg.role == "user" else "Assistant"
                context_parts.append(f"{role_prefix}: {msg.content}")
            
            context = "\n".join(context_parts)
            return f"Previous conversation:\n{context}\n\nCurrent question: {query}"
        
        return query
    
    def _prepare_context(self, chunks: List[Chunk]) -> str:
        """Format chunks into context string"""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source_info = f"[Source {i}]"
            if 'file_name' in chunk.metadata:
                source_info += f" from {chunk.metadata['file_name']}"
            if 'page' in chunk.metadata:
                source_info += f" (page {chunk.metadata['page']})"
            
            context_parts.append(f"{source_info}:\n{chunk.text}")
        
        return "\n\n".join(context_parts)
    
    def _build_prompt(self, query: str, context: str, conversation: Optional[Conversation]) -> str:
        """Build the prompt for the LLM"""
        system_prompt = """You are a helpful AI assistant answering questions based on provided context.
Always base your answers on the given context. If the context doesn't contain relevant information,
say so clearly. Cite your sources using [Source X] notation."""
        
        # Add conversation history if available
        history = ""
        if conversation is not None:
            messages: List[Message] = conversation.messages
            if len(messages) > 2:
                # Get messages before the last exchange (for context)
                # We want the exchange before the current one
                if len(messages) >= 4:
                    recent = messages[-4:-2]  # Gets the exchange before last
                    if recent:
                        history_parts = ["Previous conversation:"]
                        for msg in recent:
                            role_prefix = "Human" if msg.role == "user" else "Assistant"
                            history_parts.append(f"{role_prefix}: {msg.content}")
                        history = "\n".join(history_parts) + "\n\n"
        
        # Build the full prompt
        prompt = f"""{system_prompt}

{history}Context:
{context}

Question: {query}

Answer:"""
        
        return prompt
    
    def _format_source(self, chunk: Chunk) -> Dict[str, Any]:
        """Format source for attribution"""
        return {
            "text": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
            "document_id": chunk.document_id,
            "metadata": chunk.metadata
        }