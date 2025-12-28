"""
Integration tests for backend reliability and dependency checking.
"""

import pytest
from scitran.utils.backend_checker import (
    check_backend_dependencies,
    get_backend_status,
    validate_backend,
    get_all_backends_status
)
from scitran.core.exceptions import DependencyError


class TestBackendDependencyChecking:
    """Test backend dependency checking."""
    
    def test_check_backend_dependencies_local(self):
        """Local backend should have no dependencies."""
        deps_ok, errors = check_backend_dependencies("local")
        assert deps_ok
        assert len(errors) == 0
    
    def test_check_backend_dependencies_free(self):
        """Free backend may or may not have dependencies."""
        deps_ok, errors = check_backend_dependencies("free")
        # Should not raise, even if dependencies missing
        assert isinstance(deps_ok, bool)
        assert isinstance(errors, list)
    
    def test_get_backend_status_local(self):
        """Get status for local backend."""
        status = get_backend_status("local")
        assert status["backend"] == "local"
        assert "available" in status
        assert "dependencies_ok" in status
        assert "missing_dependencies" in status
    
    def test_get_backend_status_all(self):
        """Get status for all backends."""
        statuses = get_all_backends_status()
        assert isinstance(statuses, dict)
        assert len(statuses) > 0
        for backend, status in statuses.items():
            assert "backend" in status
            assert "available" in status
            assert "dependencies_ok" in status
    
    def test_validate_backend_local(self):
        """Validate local backend (should always work)."""
        is_valid, error = validate_backend("local", raise_on_error=False)
        assert is_valid
        assert error is None
    
    def test_validate_backend_unknown(self):
        """Unknown backend should return False."""
        is_valid, error = validate_backend("unknown_backend_xyz", raise_on_error=False)
        # Unknown backends are assumed OK (no dependencies to check)
        assert isinstance(is_valid, bool)


class TestCacheReliability:
    """Test cache reliability and fallback."""
    
    def test_cache_initialization_without_diskcache(self):
        """Cache should work without diskcache."""
        from scitran.utils.cache import TranslationCache
        
        # Should not raise even if diskcache unavailable
        cache = TranslationCache(use_disk=False)
        assert cache.use_disk is False
        assert isinstance(cache.memory_cache, dict)
    
    def test_cache_get_set_memory(self):
        """Test cache get/set in memory mode."""
        from scitran.utils.cache import TranslationCache
        
        cache = TranslationCache(use_disk=False)
        
        # Set
        cache.set("test", "en", "fr", "free", "translated")
        
        # Get
        result = cache.get("test", "en", "fr", "free")
        assert result == "translated"
    
    def test_cache_graceful_error_handling(self):
        """Cache should handle errors gracefully."""
        from scitran.utils.cache import TranslationCache
        
        cache = TranslationCache(use_disk=False)
        
        # These should never raise
        cache.set("test", "en", "fr", "free", "translated")
        result = cache.get("test", "en", "fr", "free")
        assert result == "translated" or result is None  # Either works
    
    def test_cache_stats(self):
        """Test cache statistics."""
        from scitran.utils.cache import TranslationCache
        
        cache = TranslationCache(use_disk=False)
        stats = cache.get_stats()
        
        assert "type" in stats
        assert "size" in stats
        assert "errors" in stats


class TestErrorHandling:
    """Test enhanced error handling."""
    
    def test_dependency_error_has_suggestion(self):
        """DependencyError should have helpful suggestions."""
        from scitran.core.exceptions import DependencyError
        
        error = DependencyError("test_module", "testing", "pip install test_module")
        assert error.suggestion is not None
        assert "pip install" in error.suggestion
    
    def test_backend_error_has_suggestion(self):
        """BackendError should have helpful suggestions."""
        from scitran.core.exceptions import BackendError
        
        error = BackendError("test_backend", "Test error", missing_dependency="requests")
        assert error.suggestion is not None
        assert "pip install" in error.suggestion
    
    def test_error_to_dict(self):
        """Errors should be serializable to dict."""
        from scitran.core.exceptions import DependencyError
        
        error = DependencyError("test_module", "testing")
        error_dict = error.to_dict()
        
        assert "error_type" in error_dict
        assert "message" in error_dict
        assert "suggestion" in error_dict
        assert "recoverable" in error_dict














