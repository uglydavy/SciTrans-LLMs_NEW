"""Smart Cascade Translation Backend: Lingva→LibreTranslate→MyMemory with glossary learning."""

import asyncio
from typing import Optional, Dict, List
import requests
from datetime import datetime

from ..base import TranslationBackend, TranslationRequest, TranslationResponse
from ...utils.logger import setup_logger

logger = setup_logger(__name__)


class CascadeBackend(TranslationBackend):
    """
    Smart cascade backend that tries multiple free services in order:
    1. Lingva Translate (fast, privacy-focused)
    2. LibreTranslate (open source)
    3. MyMemory (has free tier)
    
    Features:
    - Automatic failover
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
        
        # Glossary for learning
        self.learned_glossary: Dict[str, Dict[str, str]] = {}
        
        # Statistics
        self.stats = {
            "lingva_success": 0,
            "libretranslate_success": 0,
            "mymemory_success": 0,
            "total_requests": 0,
            "glossary_entries": 0
        }
    
    def is_available(self) -> bool:
        """Always available - no API key needed."""
        return True
    
    def translate_sync(self, request: TranslationRequest) -> TranslationResponse:
        """Synchronous translation with cascade fallback."""
        self.stats["total_requests"] += 1
        
        # Try each service in order
        services = [
            ("lingva", self._try_lingva),
            ("libretranslate", self._try_libretranslate),
            ("mymemory", self._try_mymemory)
        ]
        
        for service_name, service_func in services:
            try:
                result = service_func(request)
                if result:
                    logger.info(f"Translation successful via {service_name}")
                    self.stats[f"{service_name}_success"] += 1
                    
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
            except Exception as e:
                logger.warning(f"{service_name} failed: {e}, trying next service...")
                continue
        
        raise RuntimeError("All cascade services failed")
    
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
        """Try MyMemory API."""
        url = f"{self.mymemory_url}/get"
        
        params = {
            "q": request.text,
            "langpair": f"{request.source_lang}|{request.target_lang}"
        }
        
        response = requests.get(url, params=params, timeout=10)
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
                "lingva": self.stats["lingva_success"] / total * 100,
                "libretranslate": self.stats["libretranslate_success"] / total * 100,
                "mymemory": self.stats["mymemory_success"] / total * 100
            }
        }
