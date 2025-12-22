"""
Font resolver for non-Latin scripts.

STEP 7: Ensures proper font coverage for target languages.
Downloads and caches fonts as needed.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, List
import urllib.request
import hashlib

logger = logging.getLogger(__name__)


class FontResolver:
    """Resolve and cache fonts for different language scripts."""
    
    # Font URLs for common non-Latin scripts
    FONT_URLS = {
        "arabic": "https://github.com/google/fonts/raw/main/ofl/notosansarabic/NotoSansArabic%5Bwdth%2Cwght%5D.ttf",
        "chinese": "https://github.com/google/fonts/raw/main/ofl/notosanssc/NotoSansSC-Regular.ttf",
        "japanese": "https://github.com/google/fonts/raw/main/ofl/notosansjp/NotoSansJP-Regular.ttf",
        "korean": "https://github.com/google/fonts/raw/main/ofl/notosanskr/NotoSansKR-Regular.ttf",
        "cyrillic": "https://github.com/google/fonts/raw/main/ofl/notosans/NotoSans%5Bwdth%2Cwght%5D.ttf",
        "greek": "https://github.com/google/fonts/raw/main/ofl/notosans/NotoSans%5Bwdth%2Cwght%5D.ttf",
        "hebrew": "https://github.com/google/fonts/raw/main/ofl/notosanshebrew/NotoSansHebrew%5Bwdth%2Cwght%5D.ttf",
        "thai": "https://github.com/google/fonts/raw/main/ofl/notosansthai/NotoSansThai%5Bwdth%2Cwght%5D.ttf",
        "devanagari": "https://github.com/google/fonts/raw/main/ofl/notosansdevanagari/NotoSansDevanagari%5Bwdth%2Cwght%5D.ttf",
    }
    
    # Language to script mapping
    LANG_TO_SCRIPT = {
        "ar": "arabic",
        "zh": "chinese",
        "zh-cn": "chinese",
        "zh-tw": "chinese",
        "ja": "japanese",
        "ko": "korean",
        "ru": "cyrillic",
        "uk": "cyrillic",
        "bg": "cyrillic",
        "el": "greek",
        "he": "hebrew",
        "th": "thai",
        "hi": "devanagari",
        "mr": "devanagari",
        "ne": "devanagari",
    }
    
    def __init__(self, cache_dir: Optional[Path] = None, download_enabled: bool = True):
        """
        Initialize font resolver.
        
        Args:
            cache_dir: Directory to cache fonts (default: ~/.scitrans/fonts)
            download_enabled: Whether to download fonts if missing
        """
        self.cache_dir = cache_dir or Path.home() / ".scitrans" / "fonts"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.download_enabled = download_enabled
        self._font_cache: Dict[str, Path] = {}
    
    def get_font_for_language(self, lang_code: str) -> Optional[Path]:
        """
        Get font path for a language.
        
        Args:
            lang_code: Language code (e.g., 'ar', 'zh', 'ja')
            
        Returns:
            Path to font file, or None if not available
        """
        lang_code = lang_code.lower()
        
        # Check cache
        if lang_code in self._font_cache:
            return self._font_cache[lang_code]
        
        # Map language to script
        script = self.LANG_TO_SCRIPT.get(lang_code)
        if not script:
            # Latin script languages don't need special fonts
            logger.debug(f"Language {lang_code} uses Latin script (no special font needed)")
            return None
        
        # Check if font already downloaded
        font_filename = f"noto-{script}.ttf"
        font_path = self.cache_dir / font_filename
        
        if font_path.exists():
            logger.debug(f"Using cached font for {lang_code}: {font_path}")
            self._font_cache[lang_code] = font_path
            return font_path
        
        # Download if enabled
        if self.download_enabled:
            url = self.FONT_URLS.get(script)
            if url:
                try:
                    logger.info(f"Downloading font for {lang_code} ({script})...")
                    self._download_font(url, font_path)
                    self._font_cache[lang_code] = font_path
                    return font_path
                except Exception as e:
                    logger.error(f"Failed to download font for {lang_code}: {e}")
                    return None
        
        logger.warning(f"Font for {lang_code} not available (download disabled)")
        return None
    
    def _download_font(self, url: str, dest_path: Path):
        """
        Download a font file.
        
        Args:
            url: Font URL
            dest_path: Destination path
        """
        # Download to temp file first
        temp_path = dest_path.with_suffix('.tmp')
        
        try:
            # Download with progress (simple version)
            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read()
            
            # Write to temp file
            with open(temp_path, 'wb') as f:
                f.write(data)
            
            # Verify it's a valid font file (basic check)
            if len(data) < 1000:
                raise ValueError("Downloaded file too small to be a valid font")
            
            # Move to final location
            temp_path.rename(dest_path)
            
            logger.info(f"Downloaded font: {dest_path} ({len(data)} bytes)")
            
        except Exception as e:
            # Cleanup temp file
            if temp_path.exists():
                temp_path.unlink()
            raise RuntimeError(f"Font download failed: {e}")
    
    def list_cached_fonts(self) -> List[str]:
        """
        List all cached fonts.
        
        Returns:
            List of font filenames
        """
        if not self.cache_dir.exists():
            return []
        
        return [f.name for f in self.cache_dir.glob("*.ttf")]
    
    def clear_cache(self):
        """Clear all cached fonts."""
        if self.cache_dir.exists():
            for font_file in self.cache_dir.glob("*.ttf"):
                try:
                    font_file.unlink()
                    logger.info(f"Removed cached font: {font_file.name}")
                except Exception as e:
                    logger.error(f"Failed to remove {font_file.name}: {e}")


# Global font resolver instance
_global_resolver: Optional[FontResolver] = None


def get_font_resolver(download_enabled: bool = True) -> FontResolver:
    """
    Get global font resolver instance.
    
    Args:
        download_enabled: Whether to enable font downloads
        
    Returns:
        FontResolver instance
    """
    global _global_resolver
    if _global_resolver is None:
        _global_resolver = FontResolver(download_enabled=download_enabled)
    return _global_resolver


def resolve_font_for_language(lang_code: str, download_enabled: bool = True) -> Optional[Path]:
    """
    Convenience function to resolve font for a language.
    
    Args:
        lang_code: Language code
        download_enabled: Whether to download if missing
        
    Returns:
        Path to font file, or None
    """
    resolver = get_font_resolver(download_enabled=download_enabled)
    return resolver.get_font_for_language(lang_code)

