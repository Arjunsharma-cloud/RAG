from typing import Optional, Dict
import asyncio
from datetime import datetime, timedelta
from ...core.interfaces.memory_store import MemoryStore
from ...core.models.conversation import Conversation, Message

class SessionMemory(MemoryStore):
    def __init__(self, ttl: int = 3600):
        self.ttl = ttl
        self._conversations: Dict[str, Dict] = {}
        self._lock = asyncio.Lock()
    
    async def get_conversation(self, session_id: str) -> Optional[Conversation]:
        async with self._lock:
            if session_id not in self._conversations:
                return None
            return self._conversations[session_id]['conversation']
    
    async def add_message(self, session_id: str, message: Message) -> Conversation:
        async with self._lock:
            if session_id not in self._conversations:
                conversation = Conversation(session_id=session_id)
            else:
                conversation = self._conversations[session_id]['conversation']
            
            conversation.add_message(message)
            
            self._conversations[session_id] = {
                'conversation': conversation,
                'expires_at': datetime.now() + timedelta(seconds=self.ttl)
            }
            return conversation