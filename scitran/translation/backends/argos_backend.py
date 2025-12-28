"""Argos Translate backend (offline, if argostranslate is installed)."""

import time
from typing import Optional

from ..base import TranslationBackend, TranslationRequest, TranslationResponse


class ArgosBackend(TranslationBackend):
    """Offline translation via argostranslate (if installed with language packs)."""

    def __init__(self, api_key: Optional[str] = None, model: str = "argos"):
        super().__init__(api_key, model)
        try:
            import argostranslate.package  # type: ignore
            import argostranslate.translate  # type: ignore
        except Exception as e:  # pragma: no cover - optional dependency
            raise ImportError(
                "argostranslate not installed. Install with: pip install argostranslate"
            ) from e

    def _get_installed_translation(self, source: str, target: str):
        import argostranslate.translate as at  # type: ignore

        installed = at.get_installed_languages()
        src = next((lang for lang in installed if lang.code == source), None)
        tgt = next((lang for lang in installed if lang.code == target), None)
        if not src or not tgt:
            return None
        for trans in src.translations:
            if trans.to_lang.code == target:
                return trans
        return None

    def translate_sync(self, request: TranslationRequest) -> TranslationResponse:
        start = time.time()
        translator = self._get_installed_translation(request.source_lang, request.target_lang)
        if not translator:
            raise RuntimeError(
                f"Argos translation not available for {request.source_lang}->{request.target_lang}. "
                "Install the language pack via argostranslate."
            )
        translation = translator.translate(request.text or "")
        latency = time.time() - start
        return TranslationResponse(
            translations=[translation] * max(1, request.num_candidates),
            backend="argos",
            model=self.model,
            tokens_used=0,
            cost=0.0,
            latency=latency,
            metadata={"note": "Argos offline translation (requires installed language pack)"},
        )

    async def translate(self, request: TranslationRequest) -> TranslationResponse:
        return self.translate_sync(request)

    def is_available(self) -> bool:
        try:
            import argostranslate.translate  # type: ignore
            return True
        except Exception:
            return False














