"""
OpenRouter LLM service for accessing various models via API.
"""
from typing import AsyncGenerator, Optional, Dict, Any
import aiohttp
import json
import asyncio
import os

from ...core.interfaces.llm_service import LLMService
from ...utils.logger import get_logger
from ...utils.exceptions import LLMError

logger = get_logger(__name__)

class OpenRouterService(LLMService):
    """
    LLM service using OpenRouter API to access various models.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "arcee-ai/trinity-large-preview:free",
        base_url: str = "https://openrouter.ai/api/v1",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        site_url: Optional[str] = None,
        site_name: Optional[str] = None
    ):
        """
        Initialize OpenRouter service.
        
        Args:
            api_key: OpenRouter API key
            model: Model name (e.g., "mistralai/mistral-7b-instruct")
            base_url: OpenRouter API base URL
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            site_url: Your site URL for OpenRouter rankings
            site_name: Your site name for OpenRouter rankings
        """
                        
        self.api_key = "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"
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.site_url = site_url or os.getenv('SITE_URL', '')
        self.site_name = site_name or os.getenv('SITE_NAME', 'RAG System')
        self.session = None
    
    async def _ensure_session(self):
        """Ensure aiohttp session exists"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response using OpenRouter API.
        """
        await self._ensure_session()
        
        try:
            logger.info(f"Sending request to OpenRouter model: {self.model}")
            logger.debug(f"Prompt length: {len(prompt)} characters")
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Add site info for OpenRouter rankings (optional)
            if self.site_url:
                headers["HTTP-Referer"] = self.site_url
            if self.site_name:
                headers["X-Title"] = self.site_name
            
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": kwargs.get("temperature", self.temperature),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens)
            }
            
            async with self.session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"OpenRouter API error: {response.status} - {error_text}")
                    raise LLMError(f"OpenRouter API error: {response.status}")
                
                result = await response.json()
                generated_text = result["choices"][0]["message"]["content"]
                logger.info(f"OpenRouter response received: {len(generated_text)} characters")
                
                # Log token usage if available
                if "usage" in result:
                    logger.debug(f"Token usage: {result['usage']}")
                
                return generated_text
                
        except asyncio.TimeoutError:
            logger.error("OpenRouter request timed out after 30 seconds")
            raise LLMError("OpenRouter request timed out")
        except aiohttp.ClientError as e:
            logger.error(f"OpenRouter connection error: {e}")
            raise LLMError(f"Cannot connect to OpenRouter: {e}")
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise LLMError(f"Failed to generate response: {e}")
    
    async def stream_generate(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """
        Stream generate a response using OpenRouter API.
        """
        await self._ensure_session()
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            if self.site_url:
                headers["HTTP-Referer"] = self.site_url
            if self.site_name:
                headers["X-Title"] = self.site_name
            
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": kwargs.get("temperature", self.temperature),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "stream": True
            }
            
            async with self.session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"OpenRouter API error: {response.status}")
                    return
                
                async for line in response.content:
                    if line:
                        line = line.decode('utf-8').strip()
                        if line.startswith('data: ') and line != 'data: [DONE]':
                            try:
                                data = json.loads(line[6:])
                                if 'choices' in data and data['choices']:
                                    delta = data['choices'][0].get('delta', {})
                                    if 'content' in delta:
                                        yield delta['content']
                            except json.JSONDecodeError:
                                continue
                            
        except Exception as e:
            logger.error(f"Stream generation failed: {e}")
            raise
    
    async def close(self):
        """Close the session"""
        if self.session:
            await self.session.close()
            self.session = None