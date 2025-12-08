"""OpenAI translation backend."""

import os
import time
from typing import Optional
import asyncio

try:
    from openai import OpenAI, AsyncOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

from ..base import TranslationBackend, TranslationRequest, TranslationResponse


class OpenAIBackend(TranslationBackend):
    """OpenAI GPT-based translation backend."""
    
    MODELS = {
        "gpt-4o": {"cost_per_1k": 0.005, "max_tokens": 128000},
        "gpt-4-turbo": {"cost_per_1k": 0.01, "max_tokens": 128000},
        "gpt-4": {"cost_per_1k": 0.03, "max_tokens": 8192},
        "gpt-3.5-turbo": {"cost_per_1k": 0.0015, "max_tokens": 16385}
    }
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        if not HAS_OPENAI:
            raise ImportError("openai package not installed. Run: pip install openai")
        
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        super().__init__(api_key, model)
        
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
            self.async_client = AsyncOpenAI(api_key=self.api_key)
        else:
            self.client = None
            self.async_client = None
    
    def _build_messages(self, request: TranslationRequest):
        """Build messages for OpenAI API."""
        messages = []
        
        # System prompt
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        else:
            system = f"You are a professional translator specializing in scientific and technical documents. Translate from {request.source_lang} to {request.target_lang}."
            messages.append({"role": "system", "content": system})
        
        # Context (if provided)
        if request.context:
            context_text = "\n\n".join(request.context[-3:])  # Last 3 segments
            messages.append({
                "role": "system",
                "content": f"Context from previous translations:\n{context_text}"
            })
        
        # Glossary (if provided)
        if request.glossary:
            glossary_text = "\n".join([f"- {k} → {v}" for k, v in request.glossary.items()])
            messages.append({
                "role": "system",
                "content": f"Use these terminology translations:\n{glossary_text}"
            })
        
        # User text
        messages.append({
            "role": "user",
            "content": f"Translate the following text from {request.source_lang} to {request.target_lang}:\n\n{request.text}"
        })
        
        return messages
    
    async def translate(self, request: TranslationRequest) -> TranslationResponse:
        """Translate asynchronously."""
        if not self.async_client:
            raise ValueError("OpenAI API key not configured")
        
        start_time = time.time()
        messages = self._build_messages(request)
        
        try:
            if request.num_candidates > 1:
                # Generate multiple candidates
                response = await self.async_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=request.temperature,
                    n=request.num_candidates,
                    max_tokens=4096
                )
                
                translations = [choice.message.content.strip() for choice in response.choices]
                tokens_used = response.usage.total_tokens if response.usage else 0
            else:
                # Single translation
                response = await self.async_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=request.temperature,
                    max_tokens=4096
                )
                
                translations = [response.choices[0].message.content.strip()]
                tokens_used = response.usage.total_tokens if response.usage else 0
            
            # Calculate cost
            cost_per_1k = self.MODELS.get(self.model, {}).get("cost_per_1k", 0.01)
            cost = (tokens_used / 1000) * cost_per_1k
            
            latency = time.time() - start_time
            
            return TranslationResponse(
                translations=translations,
                backend="openai",
                model=self.model,
                tokens_used=tokens_used,
                cost=cost,
                latency=latency
            )
            
        except Exception as e:
            raise RuntimeError(f"OpenAI translation failed: {str(e)}")
    
    def translate_sync(self, request: TranslationRequest) -> TranslationResponse:
        """Translate synchronously."""
        if not self.client:
            raise ValueError("OpenAI API key not configured")
        
        start_time = time.time()
        messages = self._build_messages(request)
        
        try:
            if request.num_candidates > 1:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=request.temperature,
                    n=request.num_candidates,
                    max_tokens=4096
                )
                translations = [choice.message.content.strip() for choice in response.choices]
                tokens_used = response.usage.total_tokens if response.usage else 0
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=request.temperature,
                    max_tokens=4096
                )
                translations = [response.choices[0].message.content.strip()]
                tokens_used = response.usage.total_tokens if response.usage else 0
            
            cost_per_1k = self.MODELS.get(self.model, {}).get("cost_per_1k", 0.01)
            cost = (tokens_used / 1000) * cost_per_1k
            latency = time.time() - start_time
            
            return TranslationResponse(
                translations=translations,
                backend="openai",
                model=self.model,
                tokens_used=tokens_used,
                cost=cost,
                latency=latency
            )
            
        except Exception as e:
            raise RuntimeError(f"OpenAI translation failed: {str(e)}")
