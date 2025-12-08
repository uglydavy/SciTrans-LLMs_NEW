"""HuggingFace Translation Backend using Inference API."""

import os
import requests
from typing import Optional, List
from datetime import datetime

from ..base import TranslationBackend, TranslationRequest, TranslationResponse
from ...utils.logger import setup_logger

logger = setup_logger(__name__)


class HuggingFaceBackend(TranslationBackend):
    """
    HuggingFace Inference API backend.
    
    Supports various translation models:
    - facebook/mbart-large-50-many-to-many-mmt
    - Helsinki-NLP/opus-mt-en-fr (and other language pairs)
    - google/mt5-base
    
    FREE tier available with rate limits.
    """
    
    def __init__(self, model: str = "facebook/mbart-large-50-many-to-many-mmt", api_key: Optional[str] = None):
        super().__init__(api_key=api_key, model=model)
        self.name = "huggingface"
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
        self.base_url = "https://api-inference.huggingface.co/models"
        
        # Use multilingual mbart for all language pairs (most reliable)
        # Helsinki-NLP models are deprecated/moved
        self.default_models = {}
    
    def is_available(self) -> bool:
        """Check if API key is configured (optional for public models)."""
        return True  # HuggingFace has free tier
    
    def translate_sync(self, request: TranslationRequest) -> TranslationResponse:
        """Synchronous translation using HuggingFace API."""
        
        # Select best model for language pair
        lang_pair = f"{request.source_lang}-{request.target_lang}"
        model_to_use = self.default_models.get(lang_pair, self.model)
        
        url = f"{self.base_url}/{model_to_use}"
        
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        payload = {
            "inputs": request.text,
            "options": {
                "wait_for_model": True
            }
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            
            # Handle different response formats
            if isinstance(result, list) and len(result) > 0:
                if "translation_text" in result[0]:
                    translated = result[0]["translation_text"]
                elif "generated_text" in result[0]:
                    translated = result[0]["generated_text"]
                else:
                    translated = str(result[0])
            elif isinstance(result, dict):
                translated = result.get("translation_text") or result.get("generated_text") or str(result)
            else:
                translated = str(result)
            
            return TranslationResponse(
                translations=[translated],
                backend="huggingface",
                model=model_to_use,
                metadata={
                    "timestamp": datetime.now().isoformat(),
                    "api_response": result
                }
            )
            
        except requests.exceptions.RequestException as e:
            logger.error(f"HuggingFace API error: {e}")
            raise RuntimeError(f"HuggingFace translation failed: {e}")
    
    async def translate(self, request: TranslationRequest) -> TranslationResponse:
        """Async translation."""
        return self.translate_sync(request)
    
    def get_available_models(self) -> List[str]:
        """Get list of available translation models."""
        return [
            "facebook/mbart-large-50-many-to-many-mmt",
            "Helsinki-NLP/opus-mt-en-fr",
            "Helsinki-NLP/opus-mt-en-de",
            "Helsinki-NLP/opus-mt-en-es",
            "Helsinki-NLP/opus-mt-en-zh",
            "Helsinki-NLP/opus-mt-fr-en",
            "Helsinki-NLP/opus-mt-de-en",
            "google/mt5-base",
            "t5-base",
        ]
