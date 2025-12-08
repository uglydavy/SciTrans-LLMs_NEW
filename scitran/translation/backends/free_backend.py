"""Free translation backends (Google Translate, etc.)."""

import time
from typing import Optional

try:
    from deep_translator import GoogleTranslator
    HAS_DEEP_TRANSLATOR = True
except ImportError:
    HAS_DEEP_TRANSLATOR = False

from ..base import TranslationBackend, TranslationRequest, TranslationResponse


class FreeBackend(TranslationBackend):
    """Free translation backend using deep-translator."""
    
    LANG_CODES = {
        "en": "en", "english": "en",
        "fr": "fr", "french": "fr",
        "es": "es", "spanish": "es",
        "de": "de", "german": "de",
        "it": "it", "italian": "it",
        "pt": "pt", "portuguese": "pt",
        "ru": "ru", "russian": "ru",
        "zh": "zh-CN", "chinese": "zh-CN",
        "ja": "ja", "japanese": "ja",
        "ko": "ko", "korean": "ko",
        "ar": "ar", "arabic": "ar"
    }
    
    def __init__(self, api_key: Optional[str] = None, model: str = "google"):
        if not HAS_DEEP_TRANSLATOR:
            raise ImportError("deep-translator not installed. Run: pip install deep-translator")
        
        super().__init__(api_key, model)
    
    def _normalize_lang(self, lang: str) -> str:
        """Normalize language code."""
        lang = lang.lower().strip()
        return self.LANG_CODES.get(lang, lang)
    
    async def translate(self, request: TranslationRequest) -> TranslationResponse:
        """Translate asynchronously (wraps sync)."""
        return self.translate_sync(request)
    
    def translate_sync(self, request: TranslationRequest) -> TranslationResponse:
        """Translate synchronously."""
        start_time = time.time()
        
        source_lang = self._normalize_lang(request.source_lang)
        target_lang = self._normalize_lang(request.target_lang)
        
        try:
            translator = GoogleTranslator(source=source_lang, target=target_lang)
            
            # Split long texts (Google has limits)
            text = request.text
            if len(text) > 4500:
                # Split by sentences for better quality
                chunks = []
                current_chunk = ""
                for sentence in text.split(". "):
                    if len(current_chunk) + len(sentence) > 4500:
                        chunks.append(current_chunk)
                        current_chunk = sentence
                    else:
                        current_chunk += sentence + ". "
                if current_chunk:
                    chunks.append(current_chunk)
                
                translation = " ".join([translator.translate(chunk) for chunk in chunks])
            else:
                translation = translator.translate(text)
            
            # Free backends don't support multiple candidates well
            translations = [translation] * request.num_candidates
            
            latency = time.time() - start_time
            
            return TranslationResponse(
                translations=translations,
                backend="free",
                model=self.model,
                tokens_used=0,
                cost=0.0,
                latency=latency,
                metadata={"warning": "Free translation may not preserve technical terms and formatting"}
            )
            
        except Exception as e:
            raise RuntimeError(f"Free translation failed: {str(e)}")
    
    def is_available(self) -> bool:
        """Free backend is always available."""
        return True
