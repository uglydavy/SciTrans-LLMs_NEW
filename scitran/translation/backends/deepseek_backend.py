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
    
    supports_batch_candidates = True  # DeepSeek supports n parameter
    
    BASE_URL = "https://api.deepseek.com"
    
    MODELS = {
        "deepseek-chat": {"cost_per_1k": 0.00014, "max_tokens": 64000},
        "deepseek-coder": {"cost_per_1k": 0.00014, "max_tokens": 64000}
    }
    
    def __init__(self, api_key: Optional[str] = None, model: str = "deepseek-chat"):
        if not HAS_OPENAI:
            raise ImportError("openai package required for DeepSeek. Run: pip install openai")
        
        # Load API key from parameter or environment
        api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        
        if not api_key:
            raise ValueError(
                "DeepSeek API key not found. Set DEEPSEEK_API_KEY environment variable "
                "or pass api_key parameter. Get your key at: https://platform.deepseek.com/"
            )
        
        super().__init__(api_key, model)
        
        # Initialize clients with official DeepSeek base URL
        self.client = OpenAI(api_key=self.api_key, base_url=self.BASE_URL)
        self.async_client = AsyncOpenAI(api_key=self.api_key, base_url=self.BASE_URL)
    
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
            # Use n parameter for batch candidates (single API call)
            n = max(1, request.num_candidates)
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=request.temperature,
                n=n,
                max_tokens=4096
            )
            
            translations = [choice.message.content.strip() for choice in response.choices]
            finish_reasons = [choice.finish_reason for choice in response.choices]
            tokens_used = response.usage.total_tokens if response.usage else 0
            
            cost_per_1k = self.MODELS.get(self.model, {}).get("cost_per_1k", 0.00014)
            cost = (tokens_used / 1000) * cost_per_1k
            latency = time.time() - start_time
            
            return TranslationResponse(
                translations=translations,
                backend="deepseek",
                model=self.model,
                tokens_used=tokens_used,
                cost=cost,
                latency=latency,
                finish_reasons=finish_reasons,
                metadata={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                }
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
            # Use n parameter for batch candidates (single API call)
            n = max(1, request.num_candidates)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=request.temperature,
                n=n,
                max_tokens=4096
            )
            
            translations = [choice.message.content.strip() for choice in response.choices]
            finish_reasons = [choice.finish_reason for choice in response.choices]
            tokens_used = response.usage.total_tokens if response.usage else 0
            
            cost_per_1k = self.MODELS.get(self.model, {}).get("cost_per_1k", 0.00014)
            cost = (tokens_used / 1000) * cost_per_1k
            latency = time.time() - start_time
            
            return TranslationResponse(
                translations=translations,
                backend="deepseek",
                model=self.model,
                tokens_used=tokens_used,
                cost=cost,
                latency=latency,
                finish_reasons=finish_reasons,
                metadata={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                }
            )
            
        except Exception as e:
            raise RuntimeError(f"DeepSeek translation failed: {str(e)}")
