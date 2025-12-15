"""Translation caching utilities."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from scitran.core.exceptions import CacheError

logger = logging.getLogger(__name__)

try:
    import diskcache
    HAS_DISKCACHE = True
except ImportError:
    HAS_DISKCACHE = False
    logger.debug("diskcache not available, using memory cache only")


class TranslationCache:
    """Cache for translation results with graceful fallback."""
    
    def __init__(
        self, 
        cache_dir: str = ".cache/translations", 
        use_disk: bool = True,
        fallback_to_memory: bool = True
    ):
        """
        Initialize cache with graceful fallback.
        
        Args:
            cache_dir: Directory for disk cache
            use_disk: Use disk cache (requires diskcache)
            fallback_to_memory: Fallback to memory cache if disk cache fails
        """
        self.cache_dir = Path(cache_dir)
        self.use_disk = False
        self.fallback_to_memory = fallback_to_memory
        self.memory_cache: Dict[str, str] = {}
        self._cache_errors: List[str] = []
        
        if use_disk and HAS_DISKCACHE:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                self.disk_cache = diskcache.Cache(str(self.cache_dir))
                self.use_disk = True
                logger.debug(f"Using disk cache at {self.cache_dir}")
            except Exception as e:
                error_msg = f"Failed to initialize disk cache: {e}"
                self._cache_errors.append(error_msg)
                logger.warning(f"{error_msg}. Falling back to memory cache.")
                if not fallback_to_memory:
                    raise CacheError(
                        error_msg,
                        cache_type="disk",
                        operation="init"
                    )
        elif use_disk and not HAS_DISKCACHE:
            logger.debug("diskcache not available, using memory cache")
        
        if not self.use_disk:
            logger.debug("Using memory cache (disk cache unavailable or disabled)")
    
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
        Get cached translation with graceful error handling.
        
        Args:
            text: Source text
            source_lang: Source language
            target_lang: Target language
            backend: Translation backend
            
        Returns:
            Cached translation or None (never raises)
        """
        key = self._make_key(text, source_lang, target_lang, backend)
        
        try:
            if self.use_disk:
                return self.disk_cache.get(key)
            else:
                return self.memory_cache.get(key)
        except Exception as e:
            error_msg = f"Cache get failed: {e}"
            self._cache_errors.append(error_msg)
            logger.warning(f"{error_msg}. Continuing without cache.")
            # Gracefully return None - cache errors are non-fatal
            return None
    
    def set(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        backend: str,
        translation: str
    ) -> None:
        """
        Cache translation with graceful error handling.
        
        Args:
            text: Source text
            source_lang: Source language
            target_lang: Target language
            backend: Translation backend
            translation: Translated text
            
        Note:
            Never raises - cache errors are logged but non-fatal
        """
        key = self._make_key(text, source_lang, target_lang, backend)
        
        try:
            if self.use_disk:
                self.disk_cache.set(key, translation)
            else:
                self.memory_cache[key] = translation
        except Exception as e:
            error_msg = f"Cache set failed: {e}"
            self._cache_errors.append(error_msg)
            logger.warning(f"{error_msg}. Continuing without cache.")
            # Gracefully continue - cache errors are non-fatal
            # If disk cache fails and we have fallback, try memory
            if self.use_disk and self.fallback_to_memory:
                try:
                    self.memory_cache[key] = translation
                    logger.debug("Fell back to memory cache")
                except Exception:
                    pass  # Even memory cache failed, but that's OK
    
    def clear(self) -> None:
        """Clear all cache."""
        if self.use_disk:
            self.disk_cache.clear()
        else:
            self.memory_cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats: Dict[str, Any] = {
            "type": "disk" if self.use_disk else "memory",
            "errors": len(self._cache_errors)
        }
        
        try:
            if self.use_disk:
                stats.update({
                    "size": len(self.disk_cache),
                    "location": str(self.cache_dir)
                })
            else:
                stats.update({
                    "size": len(self.memory_cache)
                })
        except Exception as e:
            stats["error"] = str(e)
        
        if self._cache_errors:
            stats["recent_errors"] = self._cache_errors[-5:]  # Last 5 errors
        
        return stats
