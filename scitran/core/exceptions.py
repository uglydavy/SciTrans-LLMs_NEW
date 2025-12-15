"""
Enhanced exception hierarchy for SciTrans-LLMs.

Provides specific exception types for better error handling and user feedback.
"""

from __future__ import annotations
from typing import Optional, Dict, Any, List
from pathlib import Path
import json
from datetime import datetime


class SciTransError(Exception):
    """Base exception for all SciTrans errors."""
    
    def __init__(
        self, 
        message: str, 
        details: Optional[Dict[str, Any]] = None,
        recoverable: bool = False,
        suggestion: Optional[str] = None
    ):
        """
        Initialize error.
        
        Args:
            message: Human-readable error message
            details: Additional error details
            recoverable: Whether error can be recovered from
            suggestion: Suggested fix or workaround
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.recoverable = recoverable
        self.suggestion = suggestion
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for JSON serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
            "recoverable": self.recoverable,
            "suggestion": self.suggestion
        }
    
    def __str__(self) -> str:
        """String representation with suggestion if available."""
        result = self.message
        if self.suggestion:
            result += f"\n💡 Suggestion: {self.suggestion}"
        return result


class TranslationCoverageError(SciTransError):
    """Raised when translation coverage is incomplete in strict mode."""
    
    def __init__(
        self,
        coverage: float,
        missing_blocks: List[str],
        failure_report: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize coverage error.
        
        Args:
            coverage: Coverage ratio (0-1)
            missing_blocks: List of block IDs that failed
            failure_report: Detailed failure report
        """
        message = f"Translation coverage incomplete: {coverage:.1%} ({len(missing_blocks)} blocks missing)"
        details = {
            "coverage": coverage,
            "missing_blocks": missing_blocks,
            "failure_report": failure_report
        }
        suggestion = (
            "Try:\n"
            "1. Enable fallback backend: --enable-fallback-backend true\n"
            "2. Increase retries: --max-translation-retries 5\n"
            "3. Use a stronger backend: --backend openai\n"
            "4. Disable strict mode: --strict-mode false (not recommended)"
        )
        super().__init__(message, details, recoverable=True, suggestion=suggestion)
        self.coverage = coverage
        self.missing_blocks = missing_blocks
        self.failure_report = failure_report
    
    def save_report(self, output_path: Path) -> None:
        """Save failure report to JSON file."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "coverage": self.coverage,
            "missing_blocks": self.missing_blocks,
            "failure_report": self.failure_report,
            "error": self.to_dict()
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)


class BackendError(SciTransError):
    """Raised when a translation backend fails."""
    
    def __init__(
        self,
        backend: str,
        message: str,
        original_error: Optional[Exception] = None,
        missing_dependency: Optional[str] = None
    ):
        """
        Initialize backend error.
        
        Args:
            backend: Backend name
            message: Error message
            original_error: Original exception if any
            missing_dependency: Missing dependency name if any
        """
        full_message = f"Backend '{backend}' failed: {message}"
        details = {
            "backend": backend,
            "original_error": str(original_error) if original_error else None,
            "missing_dependency": missing_dependency
        }
        
        suggestion = None
        if missing_dependency:
            suggestion = f"Install missing dependency: pip install {missing_dependency}"
        elif backend in ["local", "libre", "argos", "cascade"]:
            suggestion = "Some backends require 'requests'. Install with: pip install requests"
        elif backend in ["openai", "anthropic", "deepseek"]:
            suggestion = f"Check API key for {backend}. Set it in Settings or via environment variable."
        
        super().__init__(full_message, details, recoverable=True, suggestion=suggestion)
        self.backend = backend
        self.original_error = original_error
        self.missing_dependency = missing_dependency


class DependencyError(SciTransError):
    """Raised when a required dependency is missing."""
    
    def __init__(
        self,
        dependency: str,
        purpose: Optional[str] = None,
        install_command: Optional[str] = None
    ):
        """
        Initialize dependency error.
        
        Args:
            dependency: Missing dependency name
            purpose: What the dependency is used for
            install_command: Command to install it
        """
        message = f"Missing dependency: {dependency}"
        if purpose:
            message += f" (required for {purpose})"
        
        details = {
            "dependency": dependency,
            "purpose": purpose
        }
        
        suggestion = install_command or f"Install with: pip install {dependency}"
        if not install_command:
            # Try to suggest based on dependency name
            if dependency == "requests":
                suggestion = "pip install requests"
            elif dependency == "diskcache":
                suggestion = "pip install diskcache"
            elif dependency == "aiohttp":
                suggestion = "pip install aiohttp"
            elif dependency == "loguru":
                suggestion = "pip install loguru"
        
        super().__init__(message, details, recoverable=True, suggestion=suggestion)
        self.dependency = dependency
        self.purpose = purpose


class ConfigurationError(SciTransError):
    """Raised when configuration is invalid."""
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        invalid_value: Optional[Any] = None,
        valid_values: Optional[List[Any]] = None
    ):
        """
        Initialize configuration error.
        
        Args:
            message: Error message
            config_key: Configuration key that's invalid
            invalid_value: Invalid value provided
            valid_values: List of valid values
        """
        details = {
            "config_key": config_key,
            "invalid_value": invalid_value,
            "valid_values": valid_values
        }
        
        suggestion = None
        if config_key and valid_values:
            suggestion = f"Valid values for {config_key}: {', '.join(map(str, valid_values))}"
        elif config_key:
            suggestion = f"Check configuration for '{config_key}'"
        
        super().__init__(message, details, recoverable=True, suggestion=suggestion)
        self.config_key = config_key
        self.invalid_value = invalid_value
        self.valid_values = valid_values


class MaskingError(SciTransError):
    """Raised when masking operations fail."""
    
    def __init__(
        self,
        message: str,
        block_id: Optional[str] = None,
        mask_type: Optional[str] = None
    ):
        """
        Initialize masking error.
        
        Args:
            message: Error message
            block_id: Block ID where error occurred
            mask_type: Type of mask that failed
        """
        details = {
            "block_id": block_id,
            "mask_type": mask_type
        }
        suggestion = "Check masking configuration and ensure patterns are valid"
        
        super().__init__(message, details, recoverable=True, suggestion=suggestion)
        self.block_id = block_id
        self.mask_type = mask_type


class RenderingError(SciTransError):
    """Raised when PDF rendering fails."""
    
    def __init__(
        self,
        message: str,
        page: Optional[int] = None,
        block_id: Optional[str] = None
    ):
        """
        Initialize rendering error.
        
        Args:
            message: Error message
            page: Page number where error occurred
            block_id: Block ID where error occurred
        """
        details = {
            "page": page,
            "block_id": block_id
        }
        suggestion = (
            "Try:\n"
            "1. Check font availability\n"
            "2. Verify PDF is not corrupted\n"
            "3. Try with different font settings"
        )
        
        super().__init__(message, details, recoverable=True, suggestion=suggestion)
        self.page = page
        self.block_id = block_id


class CacheError(SciTransError):
    """Raised when cache operations fail."""
    
    def __init__(
        self,
        message: str,
        cache_type: Optional[str] = None,
        operation: Optional[str] = None
    ):
        """
        Initialize cache error.
        
        Args:
            message: Error message
            cache_type: Type of cache (disk/memory)
            operation: Operation that failed (get/set/clear)
        """
        details = {
            "cache_type": cache_type,
            "operation": operation
        }
        suggestion = (
            "Cache errors are non-fatal. The system will continue without caching.\n"
            "To fix: Check disk space and permissions for cache directory."
        )
        
        super().__init__(message, details, recoverable=True, suggestion=suggestion)
        self.cache_type = cache_type
        self.operation = operation
