from typing import List, Dict, Any, Optional, Tuple
from ..services.embedding.bge_service import BGEEmbeddingService
from ..services.vector_store.chroma_service import ChromaService
from ..services.llm.ollama_service import OllamaService
from ..services.memory.session_memory import SessionMemory
from ..services.reranker.bge_reranker import BGEReranker
from ..core.models.conversation import Conversation, Message
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
        
        logger.info(f"Processing query: '{user_query[:50]}...' for session: {session_id}")
        
        # Normalize query
        user_query = self.normalizer.normalize(user_query)
        
        # Get conversation history
        conversation = await self.memory_store.get_conversation(session_id)
        logger.debug(f"Retrieved conversation with {len(conversation.messages) if conversation else 0} messages")
        
        # Enhance query with conversation context if needed
        enhanced_query = await self._enhance_query(user_query, conversation)
        logger.debug(f"Enhanced query: {enhanced_query[:100]}...")
        
        # Generate query embedding
        logger.info("Generating query embedding...")
        query_embedding = await self.embedding_service.embed_query(enhanced_query)
        logger.debug(f"Query embedding generated with dimension: {len(query_embedding)}")
        
        # Search for relevant chunks
        logger.info(f"Searching for relevant chunks (hybrid={self.use_hybrid_search})...")
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
        
        logger.info(f"Found {len(chunks_with_scores)} chunks from search")
        
        # Extract chunks from tuples
        chunks = [chunk for chunk, _ in chunks_with_scores]
        
        # Rerank if available
        if self.reranker and chunks:
            logger.info(f"Reranking {len(chunks)} chunks...")
            chunks = await self.reranker.rerank(enhanced_query, chunks, self.top_k)
            logger.info(f"Reranking complete, kept {len(chunks)} chunks")
        else:
            chunks = chunks[:self.top_k]
            logger.info(f"Using top {len(chunks)} chunks without reranking")
        
        if not chunks:
            logger.warning("No relevant chunks found!")
            response = "I couldn't find any relevant information in the documents to answer your question."
        else:
            # Prepare context from chunks
            context = self._prepare_context(chunks)
            logger.debug(f"Context prepared with {len(context)} characters")
            
            # Generate response
            logger.info("Generating LLM response...")
            prompt = self._build_prompt(user_query, context, conversation)
            logger.debug(f"Prompt built with {len(prompt)} characters")
            
            try:
                response = await self.llm_service.generate(prompt)
                logger.info(f"LLM response generated: {len(response)} characters")
                logger.debug(f"Response preview: {response[:200]}...")
            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
                response = f"Error generating response: {e}"
        
        # Save to memory
        logger.info("Saving conversation to memory...")
        await self.memory_store.add_message(
            session_id, 
            Message(role="user", content=user_query)
        )
        await self.memory_store.add_message(
            session_id, 
            Message(role="assistant", content=response, 
                   metadata={"sources": [c.to_dict() for c in chunks]})
        )
        
        result = {
            "answer": response,
            "sources": [self._format_source(c) for c in chunks],
            "conversation_id": session_id
        }
        
        logger.info("Query processing complete!")
        return result
    
    async def _enhance_query(self, query: str, conversation: Optional[Conversation]) -> str:
        """Enhance query with conversation context if needed"""
        if conversation is None:
            return query
        
        messages = conversation.messages
        if len(messages) < 2:
            return query
        
        # Check if query needs context (pronouns, references)
        needs_context = any(word in query.lower() 
                          for word in ['it', 'this', 'that', 'they', 'them', 'the above', 'previous', 'above'])
        
        if needs_context:
            last_messages = messages[-2:]
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
            messages = conversation.messages
            if len(messages) > 2:
                if len(messages) >= 4:
                    recent = messages[-4:-2]
                    if recent:
                        history_parts = ["Previous conversation:"]
                        for msg in recent:
                            role_prefix = "Human" if msg.role == "user" else "Assistant"
                            history_parts.append(f"{role_prefix}: {msg.content}")
                        history = "\n".join(history_parts) + "\n\n"
        
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