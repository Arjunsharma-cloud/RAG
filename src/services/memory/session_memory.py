from typing import Optional, List, Dict
import asyncio
from datetime import datetime, timedelta

from ...core.interfaces.memory_store import MemoryStore
from ...core.models.conversation import Conversation, Message
from ...utils.logger import get_logger

logger = get_logger(__name__)

class SessionMemory(MemoryStore):
    """In-memory session store with TTL and conversation turn limiting"""
    
    def __init__(self, ttl: int = 3600, max_turns: int = 10):
        """
        Initialize the session memory.
        
        Args:
            ttl: Time-to-live in seconds for conversations
            max_turns: Maximum number of conversation turns to keep (each turn = user+assistant)
        """
        self.ttl = ttl
        self.max_turns = max_turns
        self._conversations: Dict[str, Dict] = {}
        self._lock = asyncio.Lock()
    
    async def get_conversation(self, session_id: str) -> Optional[Conversation]:
        """Get conversation by session ID"""
        async with self._lock:
            if session_id not in self._conversations:
                return None
            
            conv_data = self._conversations[session_id]
            
            # Check TTL
            if datetime.now() > conv_data['expires_at']:
                del self._conversations[session_id]
                return None
            
            return conv_data['conversation']
    
    async def save_conversation(self, conversation: Conversation) -> None:
        """Save or update conversation"""
        async with self._lock:
            self._conversations[conversation.session_id] = {
                'conversation': conversation,
                'expires_at': datetime.now() + timedelta(seconds=self.ttl)
            }
    
    async def add_message(self, session_id: str, message: Message) -> Conversation:
        """Add message to conversation"""
        async with self._lock:
            # Get or create conversation
            conversation = await self.get_conversation(session_id)
            
            if not conversation:
                conversation = Conversation(session_id=session_id)
            
            # Add message
            conversation.add_message(message)
            
            # Trim if needed (max_turns * 2 because each turn has user+assistant)
            if len(conversation.messages) > self.max_turns * 2:
                # Keep only the most recent messages
                conversation.messages = conversation.messages[-(self.max_turns * 2):]
            
            # Save
            await self.save_conversation(conversation)
            
            return conversation
    
    async def delete_conversation(self, session_id: str) -> bool:
        """Delete conversation"""
        async with self._lock:
            if session_id in self._conversations:
                del self._conversations[session_id]
                return True
            return False
    
    async def list_conversations(self) -> List[str]:
        """List all active session IDs"""
        async with self._lock:
            # Clean expired first
            now = datetime.now()
            expired = [sid for sid, data in self._conversations.items() 
                      if now > data['expires_at']]
            for sid in expired:
                del self._conversations[sid]
            
            return list(self._conversations.keys())
    
    async def get_conversation_history(self, session_id: str, last_n: Optional[int] = None) -> List[Message]:
        """Get conversation history for a session"""
        conversation = await self.get_conversation(session_id)
        if not conversation:
            return []
        
        if last_n:
            return conversation.messages[-last_n:]
        return conversation.messages
    
    async def clear_all(self) -> None:
        """Clear all conversations"""
        async with self._lock:
            self._conversations.clear()
            logger.info("All conversations cleared")