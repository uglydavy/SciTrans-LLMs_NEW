# Quick Start Guide

Get started with SciTrans-LLMs in 5 minutes.

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/SciTrans-LLMs_NEW.git
cd SciTrans-LLMs_NEW

# Install dependencies
pip install -r requirements-minimal.txt

# (Optional) Install ML packages
pip install -r requirements-ml.txt
```

## Set API Keys

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export DEEPSEEK_API_KEY="sk-..."
```

Or create `configs/keys.yaml`:
```yaml
api_keys:
  openai: "sk-..."
  anthropic: "sk-ant-..."
  deepseek: "sk-..."
```

## Basic Usage

### 1. Command Line

```bash
# Simple translation
scitran translate paper.pdf -o translated.pdf

# With specific backend
scitran translate paper.pdf -o output.pdf --backend openai --model gpt-4o

# With quality settings
scitran translate paper.pdf --candidates 3 --reranking --quality 0.8
```

### 2. Python API

```python
from scitran.core.pipeline import TranslationPipeline, PipelineConfig
from scitran.extraction.pdf_parser import PDFParser
from scitran.rendering.pdf_renderer import PDFRenderer

# Parse PDF
parser = PDFParser()
document = parser.parse("paper.pdf")

# Configure pipeline
config = PipelineConfig(
    source_lang="en",
    target_lang="fr",
    backend="openai",
    enable_masking=True,
    enable_context=True
)

# Translate
pipeline = TranslationPipeline(config)
result = pipeline.translate_document(document)

# Save
renderer = PDFRenderer()
renderer.render_simple(result.document, "translated.pdf")
```

### 3. GUI Interface

```bash
# Launch web interface
scitran gui

# Or directly:
python gui/app.py
```

Then open http://localhost:7860 in your browser.

## Key Features

### LaTeX Preservation

The system automatically detects and preserves LaTeX equations:

```python
config = PipelineConfig(
    enable_masking=True  # Protects $...$ and \[...\]
)
```

### Multi-Candidate Reranking

Generate multiple translations and select the best:

```python
config = PipelineConfig(
    num_candidates=5,
    enable_reranking=True
)
```

### Context-Aware Translation

Maintain consistency across document sections:

```python
config = PipelineConfig(
    enable_context=True,
    context_window_size=3
)
```

### Custom Glossaries

Enforce domain-specific terminology:

```python
config = PipelineConfig(
    glossary={
        "machine learning": "apprentissage automatique",
        "neural network": "réseau de neurones"
    }
)
```

## Available Backends

- **OpenAI** - GPT-4, GPT-3.5 (best quality)
- **Anthropic** - Claude 3.5 Sonnet (long context)
- **DeepSeek** - Cost-effective option
- **Ollama** - Local/offline translation
- **Free** - Google Translate (no API key needed)

Check backend status:
```bash
scitran backends
```

## Examples

See `examples/basic_usage.py` for more examples:

```bash
# Run example 1: Simple translation
python examples/basic_usage.py 1

# Run example 3: With reranking
python examples/basic_usage.py 3
```

## Testing

```bash
# Run unit tests
make test

# Run with coverage
make test-all

# Run quickstart validation
python quickstart.py
```

## Troubleshooting

### "API key not configured"
Set your API key as environment variable or in `configs/keys.yaml`.

### "PDF not found"
Ensure the file path is correct and file exists.

### "ImportError: No module named..."
Install missing dependencies: `pip install -r requirements-minimal.txt`

### Slow translation
Use faster backend (DeepSeek) or reduce `num_candidates`.

## Next Steps

- Read the [User Guide](user_guide.md) for advanced features
- See [API Documentation](api/index.md) for full API reference
- Check [Thesis Guide](thesis_guide.md) for research experiments

## Support

- Issues: https://github.com/yourusername/SciTrans-LLMs_NEW/issues
- Email: your.email@university.edu
