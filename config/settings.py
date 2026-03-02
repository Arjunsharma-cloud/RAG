import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Settings:
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    CHROMA_DIR: Path = DATA_DIR / "chroma_db"
    
    EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"
    EMBEDDING_DIMENSION: int = 768
    LLM_MODEL: str = "mistral:7b-instruct"
    LLM_TEMPERATURE: float = 0.1
    
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    
    TOP_K_RESULTS: int = 5
    
    def __post_init__(self):
        """Override with environment variables"""
        for field_name in self.__dataclass_fields__:
            env_value = os.getenv(field_name)
            if env_value is not None:
                setattr(self, field_name, env_value)

def load_config():
    return Settings()