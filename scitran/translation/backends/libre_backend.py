"""LibreTranslate backend (free, no-key by default, endpoint configurable)."""

import os
import time
from typing import Optional

import requests

from ..base import TranslationBackend, TranslationRequest, TranslationResponse


class LibreTranslateBackend(TranslationBackend):
    """LibreTranslate HTTP backend."""

    def __init__(self, api_key: Optional[str] = None, model: str = "libre"):
        super().__init__(api_key, model)
        self.endpoint = os.getenv("LIBRETRANSLATE_URL", "https://libretranslate.de")

    def translate_sync(self, request: TranslationRequest) -> TranslationResponse:
        start = time.time()
        url = f"{self.endpoint.rstrip('/')}/translate"
        payload = {
            "q": request.text,
            "source": request.source_lang,
            "target": request.target_lang,
            "format": "text",
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            payload["api_key"] = self.api_key

        try:
            resp = requests.post(url, json=payload, timeout=8)
            resp.raise_for_status()
            
            # Check if response has content
            if not resp.text or not resp.text.strip():
                raise ValueError("Empty response from LibreTranslate")
            
            try:
                data = resp.json()
            except ValueError as json_err:
                # If JSON parsing fails, log the response and raise
                raise ValueError(f"Invalid JSON response from LibreTranslate: {resp.text[:200]}")
            
            translation = data.get("translatedText")
            if not translation:
                # Fallback to source text if translation is missing
                translation = request.text
                
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"LibreTranslate request failed: {e}")
        except ValueError as e:
            raise RuntimeError(f"LibreTranslate response error: {e}")
        latency = time.time() - start
        return TranslationResponse(
            translations=[translation] * max(1, request.num_candidates),
            backend="libre",
            model=self.model,
            tokens_used=0,
            cost=0.0,
            latency=latency,
            metadata={"endpoint": self.endpoint},
        )

    async def translate(self, request: TranslationRequest) -> TranslationResponse:
        return self.translate_sync(request)

    def is_available(self) -> bool:
        return True

