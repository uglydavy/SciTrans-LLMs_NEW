"""DeepSeek translation backend (OpenAI-compatible API)."""

import os
import time
from typing import Optional

try:
    from openai import OpenAI, AsyncOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

from ..base import TranslationBackend, TranslationRequest, TranslationResponse


class DeepSeekBackend(TranslationBackend):
    """DeepSeek-based translation backend (uses OpenAI-compatible API)."""
    
    BASE_URL = "https://api.deepseek.com"
    
    MODELS = {
        "deepseek-chat": {"cost_per_1k": 0.00014, "max_tokens": 64000},
        "deepseek-coder": {"cost_per_1k": 0.00014, "max_tokens": 64000}
    }
    
    def __init__(self, api_key: Optional[str] = None, model: str = "deepseek-chat"):
        if not HAS_OPENAI:
            raise ImportError("openai package required for DeepSeek. Run: pip install openai")
        
        api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        super().__init__(api_key, model)
        
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key, base_url=self.BASE_URL)
            self.async_client = AsyncOpenAI(api_key=self.api_key, base_url=self.BASE_URL)
        else:
            self.client = None
            self.async_client = None
    
    def _build_messages(self, request: TranslationRequest):
        """Build messages for DeepSeek API."""
        messages = []
        
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        else:
            system = f"You are a professional translator. Translate from {request.source_lang} to {request.target_lang}. Preserve all formatting."
            messages.append({"role": "system", "content": system})
        
        if request.context:
            context_text = "\n\n".join(request.context[-3:])
            messages.append({"role": "system", "content": f"Context:\n{context_text}"})
        
        if request.glossary:
            glossary_text = "\n".join([f"- {k} → {v}" for k, v in request.glossary.items()])
            messages.append({"role": "system", "content": f"Terminology:\n{glossary_text}"})
        
        messages.append({"role": "user", "content": f"Translate:\n\n{request.text}"})
        
        return messages
    
    async def translate(self, request: TranslationRequest) -> TranslationResponse:
        """Translate asynchronously."""
        if not self.async_client:
            raise ValueError("DeepSeek API key not configured")
        
        start_time = time.time()
        messages = self._build_messages(request)
        
        try:
            translations = []
            tokens_used = 0
            
            for _ in range(request.num_candidates):
                response = await self.async_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=request.temperature,
                    max_tokens=4096
                )
                
                translations.append(response.choices[0].message.content.strip())
                if response.usage:
                    tokens_used += response.usage.total_tokens
            
            cost_per_1k = self.MODELS.get(self.model, {}).get("cost_per_1k", 0.00014)
            cost = (tokens_used / 1000) * cost_per_1k
            latency = time.time() - start_time
            
            return TranslationResponse(
                translations=translations,
                backend="deepseek",
                model=self.model,
                tokens_used=tokens_used,
                cost=cost,
                latency=latency
            )
            
        except Exception as e:
            raise RuntimeError(f"DeepSeek translation failed: {str(e)}")
    
    def translate_sync(self, request: TranslationRequest) -> TranslationResponse:
        """Translate synchronously."""
        if not self.client:
            raise ValueError("DeepSeek API key not configured")
        
        start_time = time.time()
        messages = self._build_messages(request)
        
        try:
            translations = []
            tokens_used = 0
            
            for _ in range(request.num_candidates):
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=request.temperature,
                    max_tokens=4096
                )
                
                translations.append(response.choices[0].message.content.strip())
                if response.usage:
                    tokens_used += response.usage.total_tokens
            
            cost_per_1k = self.MODELS.get(self.model, {}).get("cost_per_1k", 0.00014)
            cost = (tokens_used / 1000) * cost_per_1k
            latency = time.time() - start_time
            
            return TranslationResponse(
                translations=translations,
                backend="deepseek",
                model=self.model,
                tokens_used=tokens_used,
                cost=cost,
                latency=latency
            )
            
        except Exception as e:
            raise RuntimeError(f"DeepSeek translation failed: {str(e)}")
