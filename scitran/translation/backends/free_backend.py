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
        """Translate synchronously with fallback to cascade backend."""
        start_time = time.time()
        
        source_lang = self._normalize_lang(request.source_lang)
        target_lang = self._normalize_lang(request.target_lang)
        
        # Try Google Translate first
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
            
        except Exception as google_error:
            # Fallback to cascade backend if Google fails
            try:
                from .cascade_backend import CascadeBackend
                cascade = CascadeBackend()
                cascade_response = cascade.translate_sync(request)
                # Update backend name to indicate fallback
                cascade_response.backend = "free(cascade_fallback)"
                return cascade_response
            except Exception as cascade_error:
                # Final fallback: try MyMemory directly
                try:
                    import requests
                    url = "https://api.mymemory.translated.net/get"
                    params = {
                        "q": request.text,
                        "langpair": f"{source_lang}|{target_lang}"
                    }
                    response = requests.get(url, params=params, timeout=5)
                    response.raise_for_status()
                    data = response.json()
                    
                    if data.get("responseStatus") == 200:
                        translation = data.get("responseData", {}).get("translatedText", request.text)
                        latency = time.time() - start_time
                        return TranslationResponse(
                            translations=[translation],
                            backend="free(mymemory_fallback)",
                            model="mymemory",
                            tokens_used=0,
                            cost=0.0,
                            latency=latency
                        )
                except:
                    pass
                
                # If all fallbacks fail, raise original error
                raise RuntimeError(f"Free translation failed: {str(google_error)}")
    
    def is_available(self) -> bool:
        """Free backend is always available."""
        return True
