# -*- coding: utf-8 -*-
"""
Batch translation with parallel processing and rate limiting.

This module provides efficient batch translation with:
- Concurrent processing using ThreadPoolExecutor
- Rate limiting to prevent API throttling
- Progress callbacks for real-time updates
"""

import time
import logging
from typing import List, Dict, Optional, Callable, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from threading import Lock
import hashlib

from scitran.core.models import Block, Document
from scitran.translation.base import TranslationRequest, TranslationResponse, TranslationBackend

logger = logging.getLogger(__name__)


@dataclass
class RateLimiter:
    """Rate limiter for API calls."""
    
    requests_per_minute: int = 30
    _last_request_time: float = 0.0
    _request_count: int = 0
    _lock: Lock = None
    
    def __post_init__(self):
        self._lock = Lock()
        self._last_request_time = time.time()
        self._request_count = 0
    
    def wait_if_needed(self):
        """Wait if we've exceeded the rate limit."""
        with self._lock:
            now = time.time()
            elapsed = now - self._last_request_time
            
            # Reset counter every minute
            if elapsed >= 60:
                self._request_count = 0
                self._last_request_time = now
            
            # Check if we need to wait
            if self._request_count >= self.requests_per_minute:
                wait_time = 60 - elapsed
                if wait_time > 0:
                    logger.info(f"Rate limit reached, waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    self._request_count = 0
                    self._last_request_time = time.time()
            
            self._request_count += 1


class TranslationCache:
    """Simple in-memory cache for translations."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._cache: Dict[str, str] = {}
        self._lock = Lock()
        self._hits = 0
        self._misses = 0
    
    def _make_key(self, text: str, source_lang: str, target_lang: str) -> str:
        """Create cache key from text and languages."""
        content = f"{source_lang}:{target_lang}:{text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """Get cached translation."""
        key = self._make_key(text, source_lang, target_lang)
        with self._lock:
            if key in self._cache:
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None
    
    def set(self, text: str, source_lang: str, target_lang: str, translation: str):
        """Cache a translation."""
        key = self._make_key(text, source_lang, target_lang)
        with self._lock:
            # Simple LRU: remove oldest if at capacity
            if len(self._cache) >= self.max_size:
                # Remove first (oldest) item
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            self._cache[key] = translation
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0
        return {
            "size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{hit_rate:.1%}"
        }
    
    def clear(self):
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0


class BatchTranslator:
    """
    Batch translator with parallel processing, caching, and rate limiting.
    """
    
    def __init__(
        self,
        backend: TranslationBackend,
        max_workers: int = 3,
        requests_per_minute: int = 30,
        enable_cache: bool = True,
        cache_size: int = 10000
    ):
        self.backend = backend
        self.max_workers = max_workers
        self.rate_limiter = RateLimiter(requests_per_minute=requests_per_minute)
        self.cache = TranslationCache(max_size=cache_size) if enable_cache else None
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "cached_responses": 0,
            "api_calls": 0,
            "errors": 0,
            "total_time": 0.0
        }
    
    def translate_batch(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> List[str]:
        """
        Translate a batch of texts in parallel.
        
        Args:
            texts: List of texts to translate
            source_lang: Source language code
            target_lang: Target language code
            progress_callback: Optional callback(completed, total, current_text)
            
        Returns:
            List of translations in same order as input
        """
        if not texts:
            return []
        
        start_time = time.time()
        results = [None] * len(texts)
        
        # Check cache first
        uncached_indices = []
        for i, text in enumerate(texts):
            if self.cache:
                cached = self.cache.get(text, source_lang, target_lang)
                if cached:
                    results[i] = cached
                    self.stats["cached_responses"] += 1
                    if progress_callback:
                        completed = sum(1 for r in results if r is not None)
                        progress_callback(completed, len(texts), f"[cached] {text[:30]}...")
                else:
                    uncached_indices.append(i)
            else:
                uncached_indices.append(i)
        
        # Translate uncached texts in parallel
        if uncached_indices:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_index = {}
                for idx in uncached_indices:
                    future = executor.submit(
                        self._translate_single,
                        texts[idx],
                        source_lang,
                        target_lang
                    )
                    future_to_index[future] = idx
                
                # Collect results as they complete
                for future in as_completed(future_to_index):
                    idx = future_to_index[future]
                    try:
                        translation = future.result()
                        results[idx] = translation
                        
                        # Cache the result
                        if self.cache and translation:
                            self.cache.set(texts[idx], source_lang, target_lang, translation)
                        
                        if progress_callback:
                            completed = sum(1 for r in results if r is not None)
                            progress_callback(completed, len(texts), texts[idx][:30] + "...")
                            
                    except Exception as e:
                        logger.error(f"Translation error for index {idx}: {e}")
                        results[idx] = texts[idx]  # Fallback to original
                        self.stats["errors"] += 1
        
        self.stats["total_requests"] += len(texts)
        self.stats["total_time"] += time.time() - start_time
        
        return results
    
    def _translate_single(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate a single text with rate limiting."""
        self.rate_limiter.wait_if_needed()
        
        try:
            request = TranslationRequest(
                text=text,
                source_lang=source_lang,
                target_lang=target_lang
            )
            
            response = self.backend.translate_sync(request)
            self.stats["api_calls"] += 1
            
            if response.translations:
                return response.translations[0]
            return text
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            self.stats["errors"] += 1
            return text
    
    def translate_blocks(
        self,
        blocks: List[Block],
        source_lang: str,
        target_lang: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> List[Block]:
        """
        Translate a list of blocks.
        
        Args:
            blocks: List of blocks to translate
            source_lang: Source language
            target_lang: Target language
            progress_callback: Progress callback
            
        Returns:
            Same blocks with translated_text populated
        """
        # Filter translatable blocks
        translatable = [(i, b) for i, b in enumerate(blocks) if b.is_translatable and b.source_text]
        
        if not translatable:
            return blocks
        
        # Get texts to translate (use masked_text if available)
        texts = [b.masked_text or b.source_text for _, b in translatable]
        
        # Translate batch
        translations = self.translate_batch(
            texts=texts,
            source_lang=source_lang,
            target_lang=target_lang,
            progress_callback=progress_callback
        )
        
        # Apply translations to blocks
        for (idx, block), translation in zip(translatable, translations):
            block.translated_text = translation
        
        return blocks
    
    def get_stats(self) -> Dict[str, Any]:
        """Get translator statistics."""
        stats = dict(self.stats)
        if self.cache:
            stats["cache"] = self.cache.get_stats()
        return stats

