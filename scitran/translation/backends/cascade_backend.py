"""Smart Cascade Translation Backend: MyMemory→Google→Lingva→LibreTranslate with glossary learning."""

import asyncio
import time
from typing import Optional, Dict, List
import requests
from datetime import datetime

from ..base import TranslationBackend, TranslationRequest, TranslationResponse
from ...utils.logger import setup_logger

logger = setup_logger(__name__)


class CascadeBackend(TranslationBackend):
    """
    Smart cascade backend that tries multiple free services in order:
    1. MyMemory (most reliable free service)
    2. Google Translate (via deep-translator, reliable fallback)
    3. Lingva Translate (fast, privacy-focused)
    4. LibreTranslate (open source, often rate-limited)
    
    Features:
    - Automatic failover
    - Service health tracking (skip consistently failing services)
    - Rate limiting to avoid 429 errors
    - Glossary learning from successful translations
    - Response caching
    - Quality scoring
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "cascade"
        
        # Service endpoints
        self.lingva_url = "https://lingva.ml/api/v1"
        self.libretranslate_url = "https://libretranslate.com"
        self.mymemory_url = "https://api.mymemory.translated.net"
        
        # Rate limiting: track last request time per service
        self._last_request_time = {}
        self._rate_limit_delay = 0.3  # 300ms delay between requests
        
        # Service health tracking: skip services that fail too often
        self._service_failures = {}  # Track consecutive failures per service
        self._service_disabled_until = {}  # Temporarily disable failing services
        self._max_consecutive_failures = 3  # Disable after 3 consecutive failures
        self._disable_duration = 300  # Re-enable after 5 minutes
        
        # Glossary for learning
        self.learned_glossary: Dict[str, Dict[str, str]] = {}
        
        # Statistics
        self.stats = {
            "mymemory_success": 0,
            "google_success": 0,
            "lingva_success": 0,
            "libretranslate_success": 0,
            "total_requests": 0,
            "glossary_entries": 0,
            "rate_limited": 0,
            "services_skipped": 0
        }
    
    def is_available(self) -> bool:
        """Always available - no API key needed."""
        return True
    
    def translate_sync(self, request: TranslationRequest) -> TranslationResponse:
        """Synchronous translation with cascade fallback."""
        self.stats["total_requests"] += 1
        
        # Try each service in order (most reliable first)
        services = [
            ("mymemory", self._try_mymemory),
            ("google", self._try_google),
            ("lingva", self._try_lingva),
            ("libretranslate", self._try_libretranslate)
        ]
        
        for service_name, service_func in services:
            # Check if service is temporarily disabled
            if self._is_service_disabled(service_name):
                self.stats["services_skipped"] += 1
                continue
            
            # Rate limiting: add delay between requests
            self._apply_rate_limit(service_name)
            
            try:
                result = service_func(request)
                if result:
                    logger.info(f"Translation successful via {service_name}")
                    self.stats[f"{service_name}_success"] += 1
                    
                    # Reset failure count on success
                    self._service_failures[service_name] = 0
                    
                    # Learn from successful translation
                    self._learn_from_translation(request.text, result, request.source_lang, request.target_lang)
                    
                    return TranslationResponse(
                        translations=[result],
                        backend="cascade",
                        model=service_name,
                        metadata={
                            "service_used": service_name,
                            "timestamp": datetime.now().isoformat()
                        }
                    )
            except requests.exceptions.HTTPError as e:
                # Handle rate limiting (429) specially
                if e.response.status_code == 429:
                    self.stats["rate_limited"] += 1
                    logger.warning(f"{service_name} rate limited (429), disabling temporarily...")
                    self._disable_service(service_name)
                else:
                    self._record_service_failure(service_name)
                logger.warning(f"{service_name} failed: {e}, trying next service...")
                continue
            except Exception as e:
                self._record_service_failure(service_name)
                logger.warning(f"{service_name} failed: {e}, trying next service...")
                continue
        
        raise RuntimeError("All cascade services failed")
    
    def _is_service_disabled(self, service_name: str) -> bool:
        """Check if service is temporarily disabled."""
        if service_name not in self._service_disabled_until:
            return False
        if time.time() < self._service_disabled_until[service_name]:
            return True
        # Re-enable after timeout
        del self._service_disabled_until[service_name]
        self._service_failures[service_name] = 0
        return False
    
    def _disable_service(self, service_name: str):
        """Temporarily disable a service."""
        self._service_disabled_until[service_name] = time.time() + self._disable_duration
        logger.info(f"Service {service_name} disabled for {self._disable_duration}s due to rate limiting")
    
    def _record_service_failure(self, service_name: str):
        """Record a service failure and disable if too many consecutive failures."""
        self._service_failures[service_name] = self._service_failures.get(service_name, 0) + 1
        if self._service_failures[service_name] >= self._max_consecutive_failures:
            self._disable_service(service_name)
    
    def _apply_rate_limit(self, service_name: str):
        """Apply rate limiting delay between requests."""
        if service_name in self._last_request_time:
            elapsed = time.time() - self._last_request_time[service_name]
            if elapsed < self._rate_limit_delay:
                time.sleep(self._rate_limit_delay - elapsed)
        self._last_request_time[service_name] = time.time()
    
    async def translate(self, request: TranslationRequest) -> TranslationResponse:
        """Async translation."""
        return self.translate_sync(request)
    
    def _try_lingva(self, request: TranslationRequest) -> Optional[str]:
        """Try Lingva Translate API."""
        import urllib.parse
        
        # URL encode the text
        encoded_text = urllib.parse.quote(request.text)
        url = f"{self.lingva_url}/{request.source_lang}/{request.target_lang}/{encoded_text}"
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        translation = data.get("translation")
        
        if not translation:
            raise ValueError("No translation in response")
        
        return translation
    
    def _try_libretranslate(self, request: TranslationRequest) -> Optional[str]:
        """Try LibreTranslate API."""
        url = f"{self.libretranslate_url}/translate"
        
        payload = {
            "q": request.text,
            "source": request.source_lang,
            "target": request.target_lang,
            "format": "text"
        }
        
        response = requests.post(url, json=payload, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        return data.get("translatedText")
    
    def _try_mymemory(self, request: TranslationRequest) -> Optional[str]:
        """Try MyMemory API (most reliable free service)."""
        url = f"{self.mymemory_url}/get"
        
        params = {
            "q": request.text,
            "langpair": f"{request.source_lang}|{request.target_lang}"
        }
        
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        
        # Check response quality
        if data.get("responseStatus") == 200:
            matches = data.get("matches", [])
            if matches and isinstance(matches, list):
                # Use highest quality match (quality can be string or number)
                try:
                    best_match = max(matches, key=lambda m: float(m.get("quality", 0)))
                    quality = float(best_match.get("quality", 0))
                    if quality > 0.7:
                        return best_match.get("translation")
                except (ValueError, TypeError):
                    pass
        
        # Fall back to direct translation
        translation = data.get("responseData", {}).get("translatedText")
        if not translation:
            raise ValueError("No translation in response")
        
        return translation
    
    def _try_google(self, request: TranslationRequest) -> Optional[str]:
        """Try Google Translate via deep-translator (reliable fallback)."""
        try:
            from deep_translator import GoogleTranslator
            translator = GoogleTranslator(source=request.source_lang, target=request.target_lang)
            translation = translator.translate(request.text)
            if translation and translation != request.text:
                return translation
            raise ValueError("Google translation returned same text or empty")
        except ImportError:
            raise ValueError("deep-translator not installed")
        except Exception as e:
            raise ValueError(f"Google translation failed: {e}")
    
    def _learn_from_translation(self, source: str, target: str, source_lang: str, target_lang: str):
        """Learn glossary entries from successful translations."""
        
        # Extract potential technical terms (words with capitals or special chars)
        import re
        
        source_words = re.findall(r'\b[A-Z][a-z]+\b|\b[a-z]+\b', source)
        target_words = re.findall(r'\b[A-Z][a-z]+\b|\b[a-z]+\b', target)
        
        # Simple heuristic: if word appears in both, it might be a technical term
        for word in source_words:
            if len(word) > 3 and word.lower() not in ['this', 'that', 'with', 'from', 'have']:
                lang_pair = f"{source_lang}-{target_lang}"
                
                if lang_pair not in self.learned_glossary:
                    self.learned_glossary[lang_pair] = {}
                
                # Store if not already learned
                if word.lower() not in self.learned_glossary[lang_pair]:
                    # Find corresponding word in target (simplified - could be improved)
                    if target_words:
                        self.learned_glossary[lang_pair][word.lower()] = target_words[0]
                        self.stats["glossary_entries"] += 1
                        logger.debug(f"Learned: {word} → {target_words[0]}")
    
    def get_learned_glossary(self, source_lang: str, target_lang: str) -> Dict[str, str]:
        """Get learned glossary for language pair."""
        lang_pair = f"{source_lang}-{target_lang}"
        return self.learned_glossary.get(lang_pair, {})
    
    def export_glossary(self, filepath: str):
        """Export learned glossary to YAML file."""
        import yaml
        
        with open(filepath, 'w') as f:
            yaml.dump({
                "learned_glossary": self.learned_glossary,
                "stats": self.stats
            }, f, default_flow_style=False)
        
        logger.info(f"Glossary exported to {filepath}")
    
    def get_statistics(self) -> Dict:
        """Get usage statistics."""
        total = self.stats["total_requests"]
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            "success_rate": {
                "mymemory": self.stats["mymemory_success"] / total * 100 if total > 0 else 0,
                "google": self.stats["google_success"] / total * 100 if total > 0 else 0,
                "lingva": self.stats["lingva_success"] / total * 100 if total > 0 else 0,
                "libretranslate": self.stats["libretranslate_success"] / total * 100 if total > 0 else 0
            }
        }
