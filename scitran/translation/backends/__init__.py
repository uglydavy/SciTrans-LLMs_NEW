"""Translation backend implementations."""

from .openai_backend import OpenAIBackend
from .anthropic_backend import AnthropicBackend
from .deepseek_backend import DeepSeekBackend
from .ollama_backend import OllamaBackend
from .free_backend import FreeBackend
from .cascade_backend import CascadeBackend
from .huggingface_backend import HuggingFaceBackend
from .local_backend import LocalBackend
from .libre_backend import LibreTranslateBackend
from .argos_backend import ArgosBackend

__all__ = [
    'OpenAIBackend',
    'AnthropicBackend',
    'DeepSeekBackend',
    'OllamaBackend',
    'FreeBackend',
    'CascadeBackend',
    'HuggingFaceBackend',
    'LocalBackend',
    'LibreTranslateBackend',
    'ArgosBackend'
]
