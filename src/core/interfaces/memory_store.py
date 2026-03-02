from abc import ABC, abstractmethod
from typing import Optional
from ..models.conversation import Conversation, Message

class MemoryStore(ABC):
    @abstractmethod
    async def get_conversation(self, session_id: str) -> Optional[Conversation]:
        pass
    
    @abstractmethod
    async def add_message(self, session_id: str, message: Message) -> Conversation:
        pass