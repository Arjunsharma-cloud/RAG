from abc import ABC, abstractmethod
from typing import Optional, List
from ..models.conversation import Conversation, Message

class MemoryStore(ABC):
    @abstractmethod
    async def get_conversation(self, session_id: str) -> Optional[Conversation]:
        """Get conversation by session ID"""
        pass
    
    @abstractmethod
    async def save_conversation(self, conversation: Conversation) -> None:
        """Save or update conversation"""
        pass
    
    @abstractmethod
    async def add_message(self, session_id: str, message: Message) -> Conversation:
        """Add message to conversation"""
        pass
    
    @abstractmethod
    async def delete_conversation(self, session_id: str) -> bool:
        """Delete conversation"""
        pass
    
    @abstractmethod
    async def list_conversations(self) -> List[str]:
        """List all session IDs"""
        pass