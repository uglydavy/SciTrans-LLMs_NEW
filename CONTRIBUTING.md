# Contributing to SciTrans-LLMs

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/SciTrans-LLMs_NEW.git
cd SciTrans-LLMs_NEW

# Install all dependencies including dev tools
make install-all

# Or step by step:
pip install -r requirements-minimal.txt
pip install -r requirements-ml.txt
pip install -r requirements-dev.txt
```

## Code Style

We follow PEP 8 with some modifications:

- Line length: 100 characters
- Use Black for formatting
- Use isort for import sorting
- Use type hints where appropriate

Format your code before committing:

```bash
make format
```

## Testing

All new features should include tests:

```bash
# Run tests
make test

# Run with coverage
make test-all

# Run specific test file
pytest tests/unit/test_masking.py -v
```

### Writing Tests

- Unit tests go in `tests/unit/`
- Integration tests go in `tests/integration/`
- End-to-end tests go in `tests/e2e/`

Example test:

```python
def test_masking():
    """Test LaTeX masking functionality."""
    engine = MaskingEngine()
    block = Block(block_id="1", source_text="Equation: $x=5$")
    
    masked = engine.mask_block(block)
    
    assert "MASK_" in masked.masked_text
    assert len(masked.masks) == 1
```

## Pull Request Process

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Make** your changes
4. **Format** code: `make format`
5. **Test** changes: `make test`
6. **Commit** with clear message: `git commit -m 'Add amazing feature'`
7. **Push** to branch: `git push origin feature/amazing-feature`
8. **Open** a Pull Request

### PR Guidelines

- Keep PRs focused and small
- Include tests for new features
- Update documentation
- Follow existing code style
- Add entry to CHANGELOG.md

## Commit Messages

Use clear, descriptive commit messages:

- `feat: add multi-candidate reranking`
- `fix: correct LaTeX masking for nested brackets`
- `docs: update API documentation`
- `test: add integration tests for pipeline`
- `refactor: simplify masking engine logic`

## Project Structure

```
scitran/
├── core/           # Core models and pipeline
├── translation/    # Translation backends
├── masking/        # Content protection
├── extraction/     # PDF parsing
├── rendering/      # Output generation
├── scoring/        # Quality assessment
└── utils/          # Utilities
```

## Adding a New Translation Backend

1. Create file in `scitran/translation/backends/`
2. Inherit from `TranslationBackend`
3. Implement `translate()` and `translate_sync()`
4. Add to `backends/__init__.py`
5. Add tests in `tests/unit/test_backends.py`
6. Update documentation

Example:

```python
from ..base import TranslationBackend, TranslationRequest, TranslationResponse

class MyBackend(TranslationBackend):
    def translate_sync(self, request: TranslationRequest) -> TranslationResponse:
        # Implement translation logic
        pass
```

## Adding New Masking Patterns

1. Add pattern to `scitran/masking/patterns.py`
2. Update `MaskingEngine` to handle pattern
3. Add validation logic
4. Add tests
5. Update documentation

## Documentation

- Keep docstrings up to date
- Use Google-style docstrings
- Update README.md for major changes
- Add examples for new features

Example docstring:

```python
def translate_document(self, document: Document) -> TranslationResult:
    """
    Translate a complete document.
    
    Args:
        document: Document to translate
        
    Returns:
        TranslationResult with translated document and statistics
        
    Raises:
        ValueError: If document is invalid
        RuntimeError: If translation fails
    """
```

## Reporting Issues

When reporting bugs, include:

- Python version
- OS and version
- Steps to reproduce
- Expected vs actual behavior
- Error messages
- Minimal code example

## Feature Requests

We welcome feature requests! Please:

- Check existing issues first
- Describe the use case
- Explain why it's valuable
- Suggest implementation if possible

## Code Review

All submissions require review. We look for:

- Correct functionality
- Test coverage
- Code quality
- Documentation
- Performance considerations

## Release Process

Maintainers follow this process:

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Run full test suite
4. Create git tag
5. Build and publish to PyPI

## Community

- Be respectful and constructive
- Help others when you can
- Share your use cases
- Contribute to discussions

## Questions?

- Open an issue for bugs
- Start a discussion for questions
- Email maintainers for private matters

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
