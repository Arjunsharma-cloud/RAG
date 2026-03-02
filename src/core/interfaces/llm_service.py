from abc import ABC, abstractmethod
from typing import AsyncGenerator

class LLMService(ABC):
    @abstractmethod
    async def generate(self, prompt: str) -> str:
        """Generate text from prompt"""
        pass