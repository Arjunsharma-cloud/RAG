from typing import AsyncGenerator
import aiohttp
import json
from ...core.interfaces.llm_service import LLMService
from ...utils.logger import get_logger

logger = get_logger(__name__)

class OllamaService(LLMService):
    def __init__(self, model: str = "mistral:7b-instruct", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.session = None
    
    async def _ensure_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()
    
    async def generate(self, prompt: str) -> str:
        await self._ensure_session()
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        
        async with self.session.post(f"{self.base_url}/api/generate", json=payload) as response:
            result = await response.json()
            return result.get("response", "")
    
    async def close(self):
        if self.session:
            await self.session.close()