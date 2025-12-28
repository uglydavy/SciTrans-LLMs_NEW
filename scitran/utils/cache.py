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
        fallback_to_memory: bool = True,
        ttl: int = 604800  # Default: 7 days (in seconds)
    ):
        """
        Initialize cache with graceful fallback and expiration.
        
        Args:
            cache_dir: Directory for disk cache
            use_disk: Use disk cache (requires diskcache)
            fallback_to_memory: Fallback to memory cache if disk cache fails
            ttl: Time-to-live in seconds (default: 7 days). Set to None for no expiration.
        """
        self.cache_dir = Path(cache_dir)
        self.use_disk = False
        self.fallback_to_memory = fallback_to_memory
        self.memory_cache: Dict[str, str] = {}
        self._cache_errors: List[str] = []
        self.ttl = ttl  # Cache expiration time
        
        if use_disk and HAS_DISKCACHE:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                # Initialize with eviction policy if diskcache supports it
                self.disk_cache = diskcache.Cache(str(self.cache_dir))
                self.use_disk = True
                logger.debug(f"Using disk cache at {self.cache_dir} (TTL: {ttl}s)")
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
        Get cached translation with graceful error handling and expiration check.
        
        Args:
            text: Source text
            source_lang: Source language
            target_lang: Target language
            backend: Translation backend
            
        Returns:
            Cached translation or None (never raises)
            Returns None if cache entry has expired
        """
        key = self._make_key(text, source_lang, target_lang, backend)
        
        try:
            if self.use_disk:
                # diskcache handles expiration automatically
                result = self.disk_cache.get(key)
                return result
            else:
                # Manual expiration check for memory cache
                entry = self.memory_cache.get(key)
                if entry:
                    if isinstance(entry, dict):
                        # New format with timestamp
                        import time
                        if self.ttl and time.time() - entry['timestamp'] > self.ttl:
                            # Expired - remove it
                            del self.memory_cache[key]
                            return None
                        return entry['value']
                    else:
                        # Old format (backward compatibility)
                        return entry
                return None
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
        Cache translation with graceful error handling and expiration.
        
        Args:
            text: Source text
            source_lang: Source language
            target_lang: Target language
            backend: Translation backend
            translation: Translated text
            
        Note:
            Never raises - cache errors are logged but non-fatal
            Entries expire after self.ttl seconds (default: 7 days)
        """
        key = self._make_key(text, source_lang, target_lang, backend)
        
        try:
            if self.use_disk:
                # Set with expiration if ttl is specified
                if self.ttl:
                    self.disk_cache.set(key, translation, expire=self.ttl)
                else:
                    self.disk_cache.set(key, translation)
            else:
                # Memory cache with timestamp for manual expiration
                import time
                self.memory_cache[key] = {
                    'value': translation,
                    'timestamp': time.time()
                }
        except Exception as e:
            error_msg = f"Cache set failed: {e}"
            self._cache_errors.append(error_msg)
            logger.warning(f"{error_msg}. Continuing without cache.")
            # Gracefully continue - cache errors are non-fatal
            # If disk cache fails and we have fallback, try memory
            if self.use_disk and self.fallback_to_memory:
                try:
                    import time
                    self.memory_cache[key] = {
                        'value': translation,
                        'timestamp': time.time()
                    }
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
