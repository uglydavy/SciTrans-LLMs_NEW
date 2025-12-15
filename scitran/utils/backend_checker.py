"""
Backend dependency checker and health monitor.

Provides utilities to check backend availability, dependencies, and health status.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING
import logging

from scitran.core.exceptions import DependencyError, BackendError

if TYPE_CHECKING:
    from scitran.translation.backends.base import TranslationBackend

logger = logging.getLogger(__name__)


# Backend dependency mapping
BACKEND_DEPENDENCIES: Dict[str, List[Tuple[str, str, Optional[str]]]] = {
    "openai": [
        ("openai", "OpenAI API client", "pip install openai>=1.0.0"),
    ],
    "anthropic": [
        ("anthropic", "Anthropic API client", "pip install anthropic>=0.25.0"),
    ],
    "deepseek": [
        ("openai", "OpenAI-compatible API client", "pip install openai>=1.0.0"),
    ],
    "ollama": [
        ("ollama", "Ollama Python client", "pip install ollama>=0.1.0"),
    ],
    "free": [
        ("deep_translator", "Deep Translator library", "pip install deep-translator>=1.11.0"),
    ],
    "cascade": [
        ("requests", "HTTP requests library", "pip install requests>=2.31.0"),
    ],
    "libre": [
        ("requests", "HTTP requests library", "pip install requests>=2.31.0"),
    ],
    "argos": [
        ("argostranslate", "Argos Translate library", "pip install argostranslate>=1.9.0"),
    ],
    "huggingface": [
        ("transformers", "HuggingFace Transformers", "pip install transformers>=4.30.0"),
        ("torch", "PyTorch", "pip install torch>=2.0.0"),
    ],
    "local": [
        # No external dependencies
    ],
}


def check_dependency(module_name: str, purpose: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """
    Check if a dependency is available.
    
    Args:
        module_name: Name of the module to check
        purpose: What the dependency is used for
        
    Returns:
        Tuple of (is_available, error_message)
    """
    try:
        __import__(module_name)
        return True, None
    except ImportError as e:
        error_msg = f"Module '{module_name}' not found"
        if purpose:
            error_msg += f" (required for {purpose})"
        return False, error_msg


def check_backend_dependencies(backend: str) -> Tuple[bool, List[DependencyError]]:
    """
    Check if all dependencies for a backend are available.
    
    Args:
        backend: Backend name
        
    Returns:
        Tuple of (all_available, list_of_errors)
    """
    if backend not in BACKEND_DEPENDENCIES:
        logger.warning(f"Unknown backend: {backend}")
        return True, []  # Unknown backends assumed OK
    
    dependencies = BACKEND_DEPENDENCIES.get(backend, [])
    errors = []
    
    for dep_name, purpose, install_cmd in dependencies:
        available, error_msg = check_dependency(dep_name, purpose)
        if not available:
            error = DependencyError(
                dependency=dep_name,
                purpose=purpose,
                install_command=install_cmd
            )
            errors.append(error)
    
    return len(errors) == 0, errors


def get_backend_status(backend: str) -> Dict[str, Any]:
    """
    Get comprehensive status for a backend.
    
    Args:
        backend: Backend name
        
    Returns:
        Status dictionary with availability, dependencies, health
    """
    status = {
        "backend": backend,
        "available": False,
        "dependencies_ok": False,
        "missing_dependencies": [],
        "health": "unknown",
        "error": None
    }
    
    # Check dependencies
    deps_ok, dep_errors = check_backend_dependencies(backend)
    status["dependencies_ok"] = deps_ok
    status["missing_dependencies"] = [
        {
            "name": err.dependency,
            "purpose": err.purpose,
            "install": err.suggestion
        }
        for err in dep_errors
    ]
    
    if not deps_ok:
        status["error"] = f"Missing dependencies: {', '.join([e.dependency for e in dep_errors])}"
        return status
    
    # Try to instantiate backend (basic health check)
    try:
        backend_instance = _create_backend_instance(backend)
        if backend_instance:
            status["available"] = True
            status["health"] = "healthy"
            
            # Check if backend is actually usable
            if hasattr(backend_instance, "is_available"):
                status["health"] = "healthy" if backend_instance.is_available() else "unavailable"
        else:
            status["health"] = "unknown"
    except Exception as e:
        status["error"] = str(e)
        status["health"] = "error"
    
    return status


def _create_backend_instance(backend: str) -> Optional[Any]:
    """Create a backend instance for health checking."""
    try:
        if backend == "openai":
            from scitran.translation.backends.openai_backend import OpenAIBackend
            return OpenAIBackend()
        elif backend == "anthropic":
            from scitran.translation.backends.anthropic_backend import AnthropicBackend
            return AnthropicBackend()
        elif backend == "deepseek":
            from scitran.translation.backends.deepseek_backend import DeepSeekBackend
            return DeepSeekBackend()
        elif backend == "ollama":
            from scitran.translation.backends.ollama_backend import OllamaBackend
            return OllamaBackend()
        elif backend == "free":
            from scitran.translation.backends.free_backend import FreeBackend
            return FreeBackend()
        elif backend == "cascade":
            from scitran.translation.backends.cascade_backend import CascadeBackend
            return CascadeBackend()
        elif backend == "libre":
            from scitran.translation.backends.libre_backend import LibreTranslateBackend
            return LibreTranslateBackend()
        elif backend == "argos":
            from scitran.translation.backends.argos_backend import ArgosBackend
            return ArgosBackend()
        elif backend == "huggingface":
            from scitran.translation.backends.huggingface_backend import HuggingFaceBackend
            return HuggingFaceBackend()
        elif backend == "local":
            from scitran.translation.backends.local_backend import LocalBackend
            return LocalBackend()
    except Exception as e:
        logger.debug(f"Could not create {backend} instance: {e}")
        return None
    
    return None


def get_all_backends_status() -> Dict[str, Dict[str, Any]]:
    """
    Get status for all backends.
    
    Returns:
        Dictionary mapping backend names to their status
    """
    all_backends = list(BACKEND_DEPENDENCIES.keys())
    statuses = {}
    
    for backend in all_backends:
        statuses[backend] = get_backend_status(backend)
    
    return statuses


def validate_backend(backend: str, raise_on_error: bool = True) -> Tuple[bool, Optional[str]]:
    """
    Validate that a backend is available and ready to use.
    
    Args:
        backend: Backend name
        raise_on_error: Whether to raise exception on error
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    deps_ok, dep_errors = check_backend_dependencies(backend)
    
    if not deps_ok:
        error_msg = f"Backend '{backend}' has missing dependencies"
        if raise_on_error and dep_errors:
            raise dep_errors[0]  # Raise first dependency error
        return False, error_msg
    
    return True, None

