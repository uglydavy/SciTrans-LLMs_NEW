"""Local fast backend: deterministic echo/identity translation for smoke tests."""

import time
from typing import Optional, Dict

from ..base import TranslationBackend, TranslationRequest, TranslationResponse


RULES_EN_FR: Dict[str, str] = {
    "hello": "bonjour",
    "world": "monde",
    "introduction": "introduction",
    "abstract": "résumé",
    "figure": "figure",
    "table": "table",
    "equation": "équation",
    "mesh": "maillage",
    "refinement": "raffinement",
    "3d": "3D",
    "post-processing": "post-traitement",
}


class LocalBackend(TranslationBackend):
    """Deterministic local translator for fast, offline tests (with simple rules)."""

    def __init__(self, api_key: Optional[str] = None, model: str = "local-echo"):
        super().__init__(api_key, model)

    def translate_sync(self, request: TranslationRequest) -> TranslationResponse:
        start = time.time()
        text = request.text or ""
        translation = self._apply_rules(text, request.source_lang, request.target_lang)
        text = request.text or ""
        translation = self._apply_rules(text, request.source_lang, request.target_lang)
        latency = time.time() - start
        return TranslationResponse(
            translations=[translation] * max(1, request.num_candidates),
            backend="local",
            model=self.model,
            tokens_used=0,
            cost=0.0,
            latency=latency,
            metadata={"info": "Local echo translation (no network)"},
        )

    async def translate(self, request: TranslationRequest) -> TranslationResponse:
        return self.translate_sync(request)

    def is_available(self) -> bool:
        return True

    def _apply_rules(self, text: str, source: str, target: str) -> str:
        # Only handle en->fr simple swaps; otherwise echo
        if source.lower().startswith("en") and target.lower().startswith("fr"):
            out = text
            for src, tgt in RULES_EN_FR.items():
                out = out.replace(src, tgt)
                out = out.replace(src.capitalize(), tgt.capitalize())
            return out
        return text

