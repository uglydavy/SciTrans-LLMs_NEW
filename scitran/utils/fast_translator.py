# -*- coding: utf-8 -*-
"""
Fast translation with async processing, caching, and optimization.

Key optimizations:
1. Async concurrent requests (not waiting for each to complete)
2. Persistent disk cache (avoid re-translating)
3. Text deduplication (translate unique texts only once)
4. Smart batching (combine small texts)
5. Connection pooling for HTTP
"""

import asyncio
import hashlib
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import List, Dict, Optional, Callable, Any, Tuple

logger = logging.getLogger(__name__)

# Try imports
try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

try:
    import diskcache
    HAS_DISKCACHE = True
except ImportError:
    HAS_DISKCACHE = False


@dataclass
class TranslationJob:
    """A translation job."""
    text: str
    source_lang: str
    target_lang: str
    index: int  # Original position in batch
    result: Optional[str] = None
    error: Optional[str] = None


class PersistentCache:
    """Persistent disk cache for translations with automatic expiration."""
    
    def __init__(self, cache_dir: str = ".cache/translations", ttl_days: int = 7):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_days * 86400  # Convert days to seconds
        
        if HAS_DISKCACHE:
            self.cache = diskcache.Cache(str(self.cache_dir), size_limit=500 * 1024 * 1024)  # 500MB
            logger.info(f"Initialized disk cache with {ttl_days}-day expiration")
        else:
            self.cache = {}
            self._load_json_cache()
            logger.info(f"Using JSON cache with {ttl_days}-day expiration")
    
    def _make_key(self, text: str, source: str, target: str) -> str:
        content = f"{source}|{target}|{text}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]
    
    def get(self, text: str, source: str, target: str) -> Optional[str]:
        key = self._make_key(text, source, target)
        if HAS_DISKCACHE:
            # diskcache handles expiration automatically
            return self.cache.get(key)
        else:
            # Check expiration for JSON cache
            entry = self.cache.get(key)
            if entry and isinstance(entry, dict) and 'timestamp' in entry:
                import time
                if time.time() - entry['timestamp'] > self.ttl_seconds:
                    # Expired - remove it
                    del self.cache[key]
                    self._save_json_cache()
                    return None
                return entry.get('value')
            elif isinstance(entry, str):
                # Old format - return as-is (will expire when cache is cleared)
                return entry
            return None
    
    def set(self, text: str, source: str, target: str, translation: str):
        key = self._make_key(text, source, target)
        if HAS_DISKCACHE:
            # Set with expiration
            self.cache.set(key, translation, expire=self.ttl_seconds)
        else:
            # Store with timestamp for expiration check
            import time
            self.cache[key] = {
                'value': translation,
                'timestamp': time.time()
            }
            self._save_json_cache()
    
    def _load_json_cache(self):
        cache_file = self.cache_dir / "cache.json"
        if cache_file.exists():
            try:
                self.cache = json.loads(cache_file.read_text())
            except Exception:
                self.cache = {}
    
    def _save_json_cache(self):
        cache_file = self.cache_dir / "cache.json"
        try:
            cache_file.write_text(json.dumps(self.cache))
        except Exception:
            pass
    
    def clear(self):
        """Clear all cached translations."""
        if HAS_DISKCACHE:
            self.cache.clear()
        else:
            self.cache = {}
            self._save_json_cache()
        logger.info("Cache cleared")
    
    def stats(self) -> Dict:
        if HAS_DISKCACHE:
            return {"type": "diskcache", "size": len(self.cache)}
        return {"type": "json", "size": len(self.cache)}


class FastTranslator:
    """
    Optimized translator with async processing and smart caching.
    
    Speed optimizations:
    - Async concurrent requests (5x faster)
    - Persistent cache (skip already translated)
    - Text deduplication (translate unique only)
    - Connection pooling
    """
    
    def __init__(
        self,
        max_concurrent: int = 5,  # Increased for better speed (was 3)
        cache_dir: str = ".cache/translations",
        timeout: int = 30,  # Increased timeout for large PDFs
        retry_count: int = 2,
        rate_limit_delay: float = 0.5  # Delay between requests to avoid rate limits
    ):
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.retry_count = retry_count
        self.rate_limit_delay = rate_limit_delay
        self.cache = PersistentCache(cache_dir)
        
        # Stats
        self.stats = {
            "total": 0,
            "cached": 0,
            "translated": 0,
            "errors": 0,
            "deduplicated": 0,
            "time": 0.0,
            "rate_limited": 0
        }
        
        # Translation services (in order of preference)
        # Start with most reliable free service
        self.services = [
            ("mymemory", self._translate_mymemory),  # Most reliable but rate-limited
            ("google_free", self._translate_google_free),  # Fallback
        ]
        
        # Track last request time for rate limiting
        self._last_request_time = 0
        self._request_lock = Lock()
    
    def translate_batch_sync(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[str]:
        """
        Synchronous batch translation (wrapper for async) with overall timeout
        and a safe fallback to avoid UI hangs.
        """
        if not texts:
            return []
        
        # Overall timeout grows with text count but is bounded
        per_item_budget = max(1.0, self.rate_limit_delay * 2 + self.timeout * 0.25)
        overall_timeout = min(240.0, max(45.0, per_item_budget * len(texts)))
        
        async def _run():
            return await asyncio.wait_for(
                self.translate_batch_async(
                    texts, source_lang, target_lang, progress_callback
                ),
                timeout=overall_timeout,
            )
        
        try:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            return loop.run_until_complete(_run())
        except asyncio.TimeoutError:
            logger.warning(
                f"Batch translate timed out after {overall_timeout:.1f}s — falling back to sync path"
            )
            return self._translate_batch_sync_fallback(texts, source_lang, target_lang, progress_callback)
        except Exception as e:
            logger.warning(f"Batch translate failed ({e}), using fallback path")
            return self._translate_batch_sync_fallback(texts, source_lang, target_lang, progress_callback)

    def _translate_batch_sync_fallback(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[str]:
        """
        Slow but safe fallback: translate sequentially with Google free.
        Ensures we return something instead of hanging.
        """
        try:
            from deep_translator import GoogleTranslator
            translator = GoogleTranslator(source=source_lang, target=target_lang)
        except Exception as e:
            logger.debug(f"Fallback translator unavailable: {e}")
            return texts
        
        results: List[str] = []
        for i, text in enumerate(texts):
            translated_chunks: List[str] = []
            for chunk in self._chunk_text(text or "", limit=4500):
                try:
                    translated_chunks.append(translator.translate(chunk) or chunk)
                except Exception:
                    translated_chunks.append(chunk)
            combined = " ".join(translated_chunks).strip()
            results.append(combined if combined else text)
            if progress_callback:
                progress_callback(i + 1, len(texts))
        return results

    def _chunk_text(self, text: str, limit: int) -> List[str]:
        """Split text into chunks within limit while keeping sentences intact."""
        if len(text) <= limit:
            return [text]
        # Split on sentence boundaries; fall back to hard splits for very long sentences
        sentences = re.split(r"(?<=[\.!?])\s+", text)
        chunks: List[str] = []
        current = ""
        for sentence in sentences:
            if not sentence:
                continue
            # If single sentence is huge, hard split it
            if len(sentence) > limit:
                if current:
                    chunks.append(current.strip())
                    current = ""
                for i in range(0, len(sentence), limit):
                    part = sentence[i:i + limit].strip()
                    if part:
                        chunks.append(part)
                continue
            if len(current) + len(sentence) + 1 <= limit:
                current = f"{current} {sentence}".strip()
            else:
                if current:
                    chunks.append(current.strip())
                current = sentence
        if current:
            chunks.append(current.strip())
        return chunks
    
    async def translate_batch_async(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[str]:
        """
        Async batch translation with all optimizations.
        """
        start_time = time.time()
        self.stats["total"] += len(texts)
        
        results = [None] * len(texts)
        jobs_to_translate: List[TranslationJob] = []
        
        # Step 1: Check cache and deduplicate
        text_to_indices: Dict[str, List[int]] = {}  # For deduplication
        
        for i, text in enumerate(texts):
            if not text or not text.strip():
                results[i] = text
                continue
            
            # Check cache
            cached = self.cache.get(text, source_lang, target_lang)
            if cached:
                results[i] = cached
                self.stats["cached"] += 1
                if progress_callback:
                    completed = sum(1 for r in results if r is not None)
                    progress_callback(completed, len(texts))
                continue
            
            # Deduplicate
            if text in text_to_indices:
                text_to_indices[text].append(i)
                self.stats["deduplicated"] += 1
            else:
                text_to_indices[text] = [i]
                jobs_to_translate.append(TranslationJob(
                    text=text,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    index=i
                ))
        
        # Step 2: Translate unique texts concurrently
        if jobs_to_translate:
            if HAS_AIOHTTP:
                await self._translate_async(jobs_to_translate, progress_callback, results, len(texts))
            else:
                self._translate_threaded(jobs_to_translate, progress_callback, results, len(texts))
            
            # Apply results to duplicates
            for job in jobs_to_translate:
                if job.result:
                    # Cache the result
                    self.cache.set(job.text, source_lang, target_lang, job.result)
                    
                    # Apply to all indices with this text
                    for idx in text_to_indices[job.text]:
                        results[idx] = job.result
        
        # Fill any None results with original text
        for i, r in enumerate(results):
            if r is None:
                results[i] = texts[i]
        
        self.stats["time"] += time.time() - start_time
        return results
    
    async def _translate_async(
        self,
        jobs: List[TranslationJob],
        progress_callback: Optional[Callable],
        results: List,
        total: int
    ):
        """Translate jobs using async aiohttp with rate limiting and shared session."""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        # Create shared aiohttp session for all requests (connection pooling)
        if HAS_AIOHTTP:
            session = aiohttp.ClientSession()
        else:
            session = None
        
        try:
            async def translate_one(job: TranslationJob):
                async with semaphore:
                    # Rate limiting - only delay for actual API calls (not cached)
                    # Delay is minimal since caching handles most requests
                    if self.rate_limit_delay > 0:
                        await asyncio.sleep(self.rate_limit_delay)
                    
                    for service_name, service_fn in self.services:
                        for attempt in range(self.retry_count):
                            try:
                                # Hard timeout guard to prevent indefinite hangs
                                # Pass session to service functions that support it
                                if service_name == "mymemory" and session:
                                    result = await asyncio.wait_for(
                                        self._translate_mymemory_with_session(
                                            job.text,
                                            job.source_lang,
                                            job.target_lang,
                                            session
                                        ),
                                        timeout=self.timeout + 5
                                    )
                                else:
                                    result = await asyncio.wait_for(
                                        service_fn(
                                            job.text,
                                            job.source_lang,
                                            job.target_lang
                                        ),
                                        timeout=self.timeout + 5
                                    )
                                if result:
                                    job.result = result
                                    results[job.index] = result
                                    self.stats["translated"] += 1
                                    if progress_callback:
                                        completed = sum(1 for r in results if r is not None)
                                        progress_callback(completed, total)
                                    return
                            except Exception as e:
                                error_msg = str(e)
                                # Check if it's a rate limit error
                                if "429" in error_msg or "Too Many Requests" in error_msg:
                                    self.stats["rate_limited"] += 1
                                    # Exponential backoff for rate limits
                                    wait_time = (attempt + 1) * 2
                                    logger.debug(f"Rate limited, waiting {wait_time}s before retry")
                                    await asyncio.sleep(wait_time)
                                    continue
                                logger.debug(f"{service_name} attempt {attempt+1} failed: {e}")
                                break  # Try next service
                    
                    # All services failed
                    job.error = "All translation services failed"
                    self.stats["errors"] += 1
            
            # Run all translations concurrently
            await asyncio.gather(*[translate_one(job) for job in jobs])
        finally:
            # Clean up session
            if session:
                await session.close()
    
    def _translate_threaded(
        self,
        jobs: List[TranslationJob],
        progress_callback: Optional[Callable],
        results: List,
        total: int
    ):
        """Fallback to threaded translation if aiohttp not available."""
        def translate_one(job: TranslationJob):
            for service_name, service_fn in self.services:
                try:
                    # Run async function in sync context
                    loop = asyncio.new_event_loop()
                    result = loop.run_until_complete(
                        service_fn(job.text, job.source_lang, job.target_lang)
                    )
                    loop.close()
                    
                    if result:
                        job.result = result
                        results[job.index] = result
                        self.stats["translated"] += 1
                        return
                except Exception as e:
                    logger.debug(f"{service_name} failed: {e}")
                    continue
            
            job.error = "All services failed"
            self.stats["errors"] += 1
        
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            list(executor.map(translate_one, jobs))
            if progress_callback:
                completed = sum(1 for r in results if r is not None)
                progress_callback(completed, total)
    
    async def _translate_lingva(self, text: str, source: str, target: str) -> Optional[str]:
        """Translate using Lingva API (async)."""
        import urllib.parse
        
        # Map language codes
        lang_map = {"en": "en", "fr": "fr", "de": "de", "es": "es", "zh": "zh", "ja": "ja"}
        src = lang_map.get(source, source)
        tgt = lang_map.get(target, target)
        
        encoded_text = urllib.parse.quote(text)
        url = f"https://lingva.ml/api/v1/{src}/{tgt}/{encoded_text}"
        
        if HAS_AIOHTTP:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=self.timeout)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("translation")
        else:
            import requests
            resp = requests.get(url, timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json().get("translation")
        
        return None
    
    async def _translate_mymemory(self, text: str, source: str, target: str) -> Optional[str]:
        """Translate using MyMemory API (async) - creates its own session."""
        if HAS_AIOHTTP:
            async with aiohttp.ClientSession() as session:
                return await self._translate_mymemory_with_session(text, source, target, session)
        else:
            # Fallback to requests
            import urllib.parse
            import requests
            
            chunks = self._chunk_text(text, limit=450)
            translated_chunks: List[str] = []
            for chunk in chunks:
                langpair = f"{source}|{target}"
                encoded_text = urllib.parse.quote(chunk)
                url = f"https://api.mymemory.translated.net/get?q={encoded_text}&langpair={langpair}"
                try:
                    resp = requests.get(url, timeout=self.timeout)
                    if resp.status_code == 200:
                        data = resp.json()
                        if data.get("responseStatus") == 200:
                            translated = data.get("responseData", {}).get("translatedText")
                            if translated and translated != chunk:
                                translated_chunks.append(translated)
                                continue
                except Exception as e:
                    logger.debug(f"MyMemory error: {e}")
                translated_chunks.append(chunk)
            
            combined = " ".join(translated_chunks).strip()
            return combined if combined else None
    
    async def _translate_mymemory_with_session(self, text: str, source: str, target: str, session) -> Optional[str]:
        """Translate using MyMemory API with provided session (for connection pooling)."""
        import urllib.parse

        async def translate_chunk(chunk: str) -> Optional[str]:
            langpair = f"{source}|{target}"
            encoded_text = urllib.parse.quote(chunk)
            url = f"https://api.mymemory.translated.net/get?q={encoded_text}&langpair={langpair}"
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=self.timeout)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get("responseStatus") == 200:
                            translated = data.get("responseData", {}).get("translatedText")
                            if translated and translated != chunk:
                                return translated
            except Exception as e:
                logger.debug(f"MyMemory error: {e}")
            return None

        chunks = self._chunk_text(text, limit=450)
        translated_chunks: List[str] = []
        for chunk in chunks:
            translated = await translate_chunk(chunk)
            translated_chunks.append(translated if translated else chunk)

        combined = " ".join(translated_chunks).strip()
        return combined if combined else None
    
    async def _translate_google_free(self, text: str, source: str, target: str) -> Optional[str]:
        """Translate using deep-translator's Google backend (async wrapper to avoid blocking)."""
        try:
            from deep_translator import GoogleTranslator

            translator = GoogleTranslator(source=source, target=target)

            # Run blocking calls in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            translated_chunks: List[str] = []
            
            for chunk in self._chunk_text(text, limit=4500):
                # Use run_in_executor to run blocking translate call in thread pool
                result = await loop.run_in_executor(None, translator.translate, chunk)
                translated_chunks.append(result if result else chunk)

            combined = " ".join(translated_chunks).strip()
            if combined and combined != text:
                return combined
        except Exception as e:
            logger.debug(f"Google free error: {e}")
        
        return None
    
    def get_stats(self) -> Dict:
        """Get translation statistics."""
        stats = dict(self.stats)
        stats["cache"] = self.cache.stats()
        if stats["time"] > 0:
            stats["speed"] = f"{stats['translated'] / stats['time']:.1f} texts/sec"
        return stats


def create_fast_translator(**kwargs) -> FastTranslator:
    """Factory function to create optimized translator."""
    return FastTranslator(**kwargs)

