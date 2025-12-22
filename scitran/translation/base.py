"""
Base translation backend interface.
All translation engines must inherit from TranslationBackend.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class TranslationRequest:
    """Request for translation."""
    text: str
    source_lang: str
    target_lang: str
    context: List[str] = None
    glossary: Dict[str, str] = None
    system_prompt: Optional[str] = None
    # STEP 3: Temperature=0.0 for deterministic output (prevents placeholder corruption)
    temperature: float = 0.0
    num_candidates: int = 1


@dataclass
class TranslationResponse:
    """Response from translation backend."""
    translations: List[str]
    backend: str
    model: str
    tokens_used: int = 0
    cost: float = 0.0
    latency: float = 0.0
    metadata: Dict = None
    finish_reasons: List[str] = None  # Track truncation: "stop", "length", etc.


class TranslationBackend(ABC):
    """Abstract base class for translation backends."""
    
    # Backend capability flags
    supports_batch_candidates = False  # Can return N candidates in one API call
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key
        self.model = model
        self.name = self.__class__.__name__
    
    @abstractmethod
    async def translate(self, request: TranslationRequest) -> TranslationResponse:
        """
        Translate text asynchronously.
        
        Args:
            request: Translation request with text and parameters
            
        Returns:
            TranslationResponse with translations and metadata
        """
        pass
    
    @abstractmethod
    def translate_sync(self, request: TranslationRequest) -> TranslationResponse:
        """
        Translate text synchronously.
        
        Args:
            request: Translation request with text and parameters
            
        Returns:
            TranslationResponse with translations and metadata
        """
        pass
    
    def is_available(self) -> bool:
        """Check if backend is available and configured."""
        return self.api_key is not None
    
    def get_info(self) -> Dict:
        """Get backend information."""
        return {
            "name": self.name,
            "model": self.model,
            "available": self.is_available()
        }
