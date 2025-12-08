"""Ollama local translation backend."""

import time
from typing import Optional

try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False

from ..base import TranslationBackend, TranslationRequest, TranslationResponse


class OllamaBackend(TranslationBackend):
    """Ollama local LLM translation backend."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "llama3.1"):
        if not HAS_OLLAMA:
            raise ImportError("ollama package not installed. Run: pip install ollama")
        
        super().__init__(api_key, model)
        self.client = ollama.Client()
    
    def _build_prompt(self, request: TranslationRequest):
        """Build prompt for Ollama."""
        parts = []
        
        system = request.system_prompt or f"You are a translator. Translate from {request.source_lang} to {request.target_lang}. Preserve formatting."
        parts.append(system)
        
        if request.context:
            context_text = "\n\n".join(request.context[-2:])
            parts.append(f"\nContext:\n{context_text}")
        
        if request.glossary:
            glossary_text = "\n".join([f"{k} → {v}" for k, v in list(request.glossary.items())[:10]])
            parts.append(f"\nTerminology:\n{glossary_text}")
        
        parts.append(f"\nTranslate:\n{request.text}")
        
        return "\n".join(parts)
    
    async def translate(self, request: TranslationRequest) -> TranslationResponse:
        """Translate asynchronously."""
        return self.translate_sync(request)  # Ollama doesn't have true async
    
    def translate_sync(self, request: TranslationRequest) -> TranslationResponse:
        """Translate synchronously."""
        start_time = time.time()
        prompt = self._build_prompt(request)
        
        try:
            translations = []
            
            for _ in range(request.num_candidates):
                response = self.client.generate(
                    model=self.model,
                    prompt=prompt,
                    options={
                        "temperature": request.temperature,
                        "num_predict": 2048
                    }
                )
                
                translations.append(response['response'].strip())
            
            latency = time.time() - start_time
            
            return TranslationResponse(
                translations=translations,
                backend="ollama",
                model=self.model,
                tokens_used=0,  # Ollama doesn't report tokens
                cost=0.0,  # Local = free
                latency=latency
            )
            
        except Exception as e:
            raise RuntimeError(f"Ollama translation failed: {str(e)}. Make sure Ollama is running and model '{self.model}' is installed.")
    
    def is_available(self) -> bool:
        """Check if Ollama is available."""
        try:
            self.client.list()
            return True
        except:
            return False
