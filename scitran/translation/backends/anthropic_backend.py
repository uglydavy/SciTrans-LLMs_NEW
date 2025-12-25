"""Anthropic Claude translation backend."""

import os
import time
import asyncio
import logging
from typing import Optional

try:
    from anthropic import Anthropic, AsyncAnthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

from ..base import TranslationBackend, TranslationRequest, TranslationResponse
from ..output_cleaner import clean_batch_outputs

logger = logging.getLogger(__name__)


class AnthropicBackend(TranslationBackend):
    """Anthropic Claude-based translation backend."""
    
    MODELS = {
        # Claude 3.5 Sonnet (Latest & Best - Recommended)
        "claude-3-5-sonnet-20241022": {"cost_per_1k": 0.003, "max_tokens": 200000, "quality": "best"},
        "claude-3-5-sonnet-20240620": {"cost_per_1k": 0.003, "max_tokens": 200000, "quality": "best"},
        
        # Claude 3 Opus (Highest Quality, Most Expensive)
        "claude-3-opus-20240229": {"cost_per_1k": 0.015, "max_tokens": 200000, "quality": "highest"},
        
        # Claude 3 Sonnet (Good Balance)
        "claude-3-sonnet-20240229": {"cost_per_1k": 0.003, "max_tokens": 200000, "quality": "good"},
        
        # Claude 3 Haiku (Fastest & Cheapest)
        "claude-3-haiku-20240307": {"cost_per_1k": 0.00025, "max_tokens": 200000, "quality": "fast"},
        
        # Note: Claude 4.x doesn't exist yet - 3.5 Sonnet is the latest
    }
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-5-sonnet-20241022", base_url: Optional[str] = None):
        if not HAS_ANTHROPIC:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
        
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        # Support custom base URL for third-party services (e.g., api.qidianai.xyz)
        base_url = base_url or os.getenv("ANTHROPIC_API_BASE_URL")
        super().__init__(api_key, model)
        
        if self.api_key:
            # Initialize clients with optional custom base URL
            client_kwargs = {"api_key": self.api_key}
            if base_url:
                # Fix: Remove trailing /v1 if present, Anthropic SDK adds it automatically
                # Also handle cases where base_url already includes /v1
                if base_url.endswith("/v1"):
                    base_url = base_url[:-3]  # Remove /v1
                elif base_url.endswith("/v1/"):
                    base_url = base_url[:-4]  # Remove /v1/
                client_kwargs["base_url"] = base_url
                logger.info(f"Using custom Anthropic API endpoint: {base_url}")
            
            self.client = Anthropic(**client_kwargs)
            self.async_client = AsyncAnthropic(**client_kwargs)
        else:
            self.client = None
            self.async_client = None
    
    def is_available(self) -> bool:
        """Check if backend is available and configured."""
        return self.api_key is not None and self.client is not None
    
    def _build_system_prompt(self, request: TranslationRequest):
        """Build system prompt for Claude (STEP 3: all instructions in system)."""
        if request.system_prompt:
            return request.system_prompt
        
        parts = [
            f"You are a professional translator specializing in scientific and technical documents.",
            f"Translate from {request.source_lang} to {request.target_lang}.",
            "",
            "CRITICAL RULES:",
            "- Output ONLY the translated text",
            "- No explanations, no commentary, no labels",
            "- Preserve all placeholder tokens EXACTLY (e.g., <<FORMULA_0001>>)",
            "- Preserve LaTeX formulas, code blocks, URLs unchanged",
            "- Maintain formatting and structure",
        ]
        
        if request.context:
            context_text = "\n".join(request.context[-3:])
            parts.append(f"\nContext from previous translations:\n{context_text}")
        
        if request.glossary:
            glossary_text = "\n".join([f"- {k} → {v}" for k, v in request.glossary.items()])
            parts.append(f"\nUse these terminology translations:\n{glossary_text}")
        
        return "\n".join(parts)
    
    async def translate(self, request: TranslationRequest) -> TranslationResponse:
        """Translate asynchronously with concurrent requests for multiple candidates."""
        if not self.async_client:
            raise ValueError("Anthropic API key not configured")
        
        start_time = time.time()
        
        # STEP 3: System prompt contains ALL instructions, user message is ONLY text
        system = self._build_system_prompt(request)
        user_prompt = request.text  # Just the text, no wrapper
        
        try:
            # Run multiple candidate requests concurrently
            async def get_candidate():
                response = await self.async_client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    temperature=request.temperature,
                    system=system,
                    messages=[{"role": "user", "content": user_prompt}]
                )
                return response
            
            # Gather all candidates concurrently
            responses = await asyncio.gather(*[get_candidate() for _ in range(request.num_candidates)])
            
            # STEP 3: Clean outputs
            raw_translations = [resp.content[0].text.strip() for resp in responses]
            translations = clean_batch_outputs(raw_translations)
            finish_reasons = [resp.stop_reason for resp in responses]
            tokens_used = sum(resp.usage.input_tokens + resp.usage.output_tokens for resp in responses)
            
            cost_per_1k = self.MODELS.get(self.model, {}).get("cost_per_1k", 0.003)
            cost = (tokens_used / 1000) * cost_per_1k
            latency = time.time() - start_time
            
            return TranslationResponse(
                translations=translations,
                backend="anthropic",
                model=self.model,
                tokens_used=tokens_used,
                cost=cost,
                latency=latency,
                finish_reasons=finish_reasons,
                metadata={
                    "stop_reasons": finish_reasons,
                }
            )
            
        except Exception as e:
            raise RuntimeError(f"Anthropic translation failed: {str(e)}")
    
    def translate_sync(self, request: TranslationRequest) -> TranslationResponse:
        """Translate synchronously."""
        if not self.client:
            raise ValueError("Anthropic API key not configured")
        
        start_time = time.time()
        
        # STEP 3: System prompt contains ALL instructions, user message is ONLY text
        system = self._build_system_prompt(request)
        user_prompt = request.text  # Just the text, no wrapper
        
        try:
            translations = []
            tokens_used = 0
            
            for _ in range(request.num_candidates):
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    temperature=request.temperature,
                    system=system,
                    messages=[{"role": "user", "content": user_prompt}]
                )
                
                translations.append(response.content[0].text.strip())
                tokens_used += response.usage.input_tokens + response.usage.output_tokens
            
            # STEP 3: Clean all translations
            translations = clean_batch_outputs(translations)
            
            cost_per_1k = self.MODELS.get(self.model, {}).get("cost_per_1k", 0.003)
            cost = (tokens_used / 1000) * cost_per_1k
            latency = time.time() - start_time
            
            return TranslationResponse(
                translations=translations,
                backend="anthropic",
                model=self.model,
                tokens_used=tokens_used,
                cost=cost,
                latency=latency
            )
            
        except Exception as e:
            raise RuntimeError(f"Anthropic translation failed: {str(e)}")
