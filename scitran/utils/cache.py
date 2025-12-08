"""Translation caching utilities."""

import hashlib
import json
from pathlib import Path
from typing import Optional, Dict

try:
    import diskcache
    HAS_DISKCACHE = True
except ImportError:
    HAS_DISKCACHE = False


class TranslationCache:
    """Cache for translation results."""
    
    def __init__(self, cache_dir: str = ".cache/translations", use_disk: bool = True):
        """
        Initialize cache.
        
        Args:
            cache_dir: Directory for disk cache
            use_disk: Use disk cache (requires diskcache)
        """
        self.cache_dir = Path(cache_dir)
        self.use_disk = use_disk and HAS_DISKCACHE
        
        if self.use_disk:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.disk_cache = diskcache.Cache(str(self.cache_dir))
        else:
            self.memory_cache = {}
    
    def _make_key(self, text: str, source_lang: str, target_lang: str, backend: str) -> str:
        """Generate cache key from parameters."""
        key_str = f"{text}|{source_lang}|{target_lang}|{backend}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        backend: str
    ) -> Optional[str]:
        """
        Get cached translation.
        
        Args:
            text: Source text
            source_lang: Source language
            target_lang: Target language
            backend: Translation backend
            
        Returns:
            Cached translation or None
        """
        key = self._make_key(text, source_lang, target_lang, backend)
        
        if self.use_disk:
            return self.disk_cache.get(key)
        else:
            return self.memory_cache.get(key)
    
    def set(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        backend: str,
        translation: str
    ):
        """
        Cache translation.
        
        Args:
            text: Source text
            source_lang: Source language
            target_lang: Target language
            backend: Translation backend
            translation: Translated text
        """
        key = self._make_key(text, source_lang, target_lang, backend)
        
        if self.use_disk:
            self.disk_cache.set(key, translation)
        else:
            self.memory_cache[key] = translation
    
    def clear(self):
        """Clear all cache."""
        if self.use_disk:
            self.disk_cache.clear()
        else:
            self.memory_cache.clear()
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        if self.use_disk:
            return {
                "type": "disk",
                "size": len(self.disk_cache),
                "location": str(self.cache_dir)
            }
        else:
            return {
                "type": "memory",
                "size": len(self.memory_cache)
            }
