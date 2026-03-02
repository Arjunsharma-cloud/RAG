from typing import AsyncGenerator
import aiohttp
import json
import asyncio
from ...core.interfaces.llm_service import LLMService
from ...utils.logger import get_logger
from ...utils.exceptions import LLMError

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
        """Generate response from Ollama"""
        await self._ensure_session()
        
        try:
            logger.info(f"Sending request to Ollama model: {self.model}")
            logger.debug(f"Prompt length: {len(prompt)} characters")
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 2048
                }
            }
            
            async with self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)  # 60 second timeout
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Ollama API error: {response.status} - {error_text}")
                    raise LLMError(f"Ollama API error: {response.status}")
                
                result = await response.json()
                generated_text = result.get("response", "")
                logger.info(f"Ollama response received: {len(generated_text)} characters")
                return generated_text
                
        except asyncio.TimeoutError:
            logger.error("Ollama request timed out after 60 seconds")
            raise LLMError("Ollama request timed out")
        except aiohttp.ClientError as e:
            logger.error(f"Ollama connection error: {e}")
            logger.error(f"Make sure Ollama is running at {self.base_url}")
            raise LLMError(f"Cannot connect to Ollama. Is it running? Error: {e}")
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise LLMError(f"Failed to generate response: {e}")
    
    async def stream_generate(self, prompt: str) -> AsyncGenerator[str, None]:
        """Stream generate response"""
        await self._ensure_session()
        
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 2048
                }
            }
            
            async with self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Ollama API error: {response.status}")
                    return
                
                async for line in response.content:
                    if line:
                        try:
                            data = json.loads(line)
                            if "response" in data:
                                yield data["response"]
                            if data.get("done"):
                                break
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            logger.error(f"Stream generation failed: {e}")
            raise
    
    async def close(self):
        if self.session:
            await self.session.close()