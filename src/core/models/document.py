from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum
import uuid

# These are the basic building blocks - what IS a document in our system?

class DocumentType(Enum):
    PDF = "pdf"
    CSV = "csv"
    IMAGE = "image"
    AUDIO = "audio"
    TEXT = "text"

class ProcessingStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Document:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: DocumentType = None
    source_path: str = ""
    content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: ProcessingStatus = ProcessingStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    processed_at: Optional[datetime] = None
    error_message: Optional[str] = None