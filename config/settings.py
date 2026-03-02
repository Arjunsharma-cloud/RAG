import os
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class Settings:
    # Project paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    CHROMA_DIR: Path = DATA_DIR / "chroma_db"
    
    # Model configurations
    EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"
    EMBEDDING_DIMENSION: int = 768
    LLM_MODEL: str = "mistral:7b-instruct"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 2048
    LLM_BASE_URL: str = "http://localhost:11434"
    
    # Chunking parameters
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    CHUNKING_STRATEGY: str = "semantic"
    
    # Search parameters
    TOP_K_RESULTS: int = 5
    USE_RERANKER: bool = False
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"
    HYBRID_SEARCH_WEIGHT: float = 0.5
    
    # Memory settings
    MEMORY_TTL: int = 3600
    MAX_HISTORY_TURNS: int = 10
    
    # Processing settings
    BATCH_SIZE: int = 32
    MAX_WORKERS: int = 4
    
    # OCR settings
    TESSERACT_CMD: str = "tesseract"
    
    # OpenRouter settings
    OPENROUTER_API_KEY: str = "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"
    OPENROUTER_MODEL: str = "mistralai/mistral-7b-instruct:free"
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_SITE_URL: str = ""
    OPENROUTER_SITE_NAME: str = "RAG System"
    
    # Choose which LLM to use
    LLM_PROVIDER: str = "openrouter"  # "ollama" or "openrouter"
    
    # Optional Neo4j
    NEO4J_URI: Optional[str] = None
    NEO4J_USER: Optional[str] = None
    NEO4J_PASSWORD: Optional[str] = None
    
    def __post_init__(self):
        """Override with environment variables"""
        for field_name in self.__dataclass_fields__:
            env_value = os.getenv(field_name)
            if env_value is not None:
                # Try to convert to appropriate type
                field_type = self.__dataclass_fields__[field_name].type
                if field_type == bool:
                    setattr(self, field_name, env_value.lower() in ('true', '1', 'yes'))
                elif field_type == int:
                    setattr(self, field_name, int(env_value))
                elif field_type == float:
                    setattr(self, field_name, float(env_value))
                elif field_type == Path:
                    setattr(self, field_name, Path(env_value))
                else:
                    setattr(self, field_name, env_value)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Settings":
        """Load settings from YAML file"""
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(**config)

# Define load_config as a standalone function (not inside the class)
def load_config(config_path: Optional[str] = None) -> Settings:
    """
    Load configuration from file or environment.
    
    Args:
        config_path: Optional path to YAML config file
    
    Returns:
        Settings object
    """
    if config_path and Path(config_path).exists():
        return Settings.from_yaml(config_path)
    return Settings()