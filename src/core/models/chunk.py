from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import uuid

@dataclass
class Chunk:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str = ""
    text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    index: int = 0