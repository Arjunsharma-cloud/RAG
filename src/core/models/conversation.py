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