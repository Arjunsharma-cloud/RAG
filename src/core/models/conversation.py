from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any
import uuid

@dataclass
class Message:
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Conversation:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    messages: List[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def add_message(self, message: Message):
        """Add a message to the conversation"""
        self.messages.append(message)
        self.updated_at = datetime.now()
    
    def get_last_n_messages(self, n: int) -> List[Message]:
        """Get the last n messages"""
        return self.messages[-n:] if self.messages else []
    
    def get_conversation_context(self, max_messages: int = 10) -> str:
        """Format conversation for context"""
        context = []
        for msg in self.messages[-max_messages:]:
            prefix = "Human: " if msg.role == "user" else "Assistant: "
            context.append(f"{prefix}{msg.content}")
        return "\n".join(context)