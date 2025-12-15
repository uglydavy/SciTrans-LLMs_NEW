# Backend Installation Guide

This guide explains how to install and configure local translation backends.

## Available Backends

### Free/Online Backends (No Installation Required)
- **cascade**: Multi-service fallback (mymemory, google, lingva, libretranslate)
- **free**: Google Translate via deep-translator
- **libre**: LibreTranslate (uses public endpoint by default)

### Local Backends (Require Installation)

#### 1. Ollama (Local LLM)
**Installation:**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a translation model (recommended: llama3.1 or qwen2.5)
ollama pull llama3.1

# Install Python package
pip install ollama
```

**Usage:**
- Backend: `ollama`
- Models: `llama3.1`, `llama3.2`, `qwen2.5`, `mistral`, `gemma2`, `llama3.3`
- No API key required
- Runs completely offline

#### 2. Argos Translate (Offline)
**Installation:**
```bash
# Install argostranslate
pip install argostranslate

# Install language packs (example: English to French)
python -m argostranslate.package.update
python -m argostranslate.package.install en fr
```

**Usage:**
- Backend: `argos`
- No API key required
- Completely offline
- Requires language packs for each language pair

**Available Language Packs:**
```bash
# List available packs
python -m argostranslate.package.list

# Install specific packs
python -m argostranslate.package.install en fr  # English to French
python -m argostranslate.package.install fr en  # French to English
```

#### 3. HuggingFace (Local Models)
**Installation:**
```bash
# Install transformers and torch
pip install transformers torch

# Optional: Install specific model
# Models are downloaded automatically on first use
```

**Usage:**
- Backend: `huggingface`
- Models: 
  - `facebook/mbart-large-50-many-to-many-mmt` (multilingual, recommended)
  - `Helsinki-NLP/opus-mt-en-fr` (English-French)
- API key: Optional (for private models or higher rate limits)
  - Set `HUGGINGFACE_API_KEY` environment variable
  - Or configure in GUI Settings

**Models are downloaded automatically on first use** (can be several GB).

#### 4. LibreTranslate (Self-Hosted)
**Installation:**
```bash
# Option 1: Docker (recommended)
docker run -d -p 5000:5000 libretranslate/libretranslate

# Option 2: Python package
pip install libretranslate

# Start server
libretranslate --host 0.0.0.0 --port 5000
```

**Usage:**
- Backend: `libre`
- API key: Optional (for hosted instances)
- Endpoint: Defaults to `https://libretranslate.de`
  - For self-hosted: Set `LIBRETRANSLATE_URL` environment variable
  - Example: `export LIBRETRANSLATE_URL=http://localhost:5000`

#### 5. Local Backend (Testing Only)
**Installation:**
- No installation required
- Built-in for testing

**Usage:**
- Backend: `local`
- Simple rule-based translation (for smoke tests only)
- Not suitable for production

## Configuration

### Environment Variables
```bash
# HuggingFace (optional)
export HUGGINGFACE_API_KEY="your_key_here"

# LibreTranslate (for self-hosted)
export LIBRETRANSLATE_URL="http://localhost:5000"
```

### GUI Configuration
1. Open GUI: `./scitrans gui`
2. Go to **Settings** tab
3. Enter API keys if needed
4. Select backend from dropdown
5. Select model (if applicable)

## Testing Backends

Use the **Testing** tab in the GUI:
1. Select backend from dropdown
2. Enter sample text
3. Click "Test Backend"
4. Check if translation works

## Troubleshooting

### Ollama
- **Error: "ollama package not installed"**
  - Run: `pip install ollama`
- **Error: "Model not found"**
  - Run: `ollama pull llama3.1` (or your chosen model)

### Argos Translate
- **Error: "argostranslate not installed"**
  - Run: `pip install argostranslate`
- **Error: "No translation available"**
  - Install language packs: `python -m argostranslate.package.install en fr`

### HuggingFace
- **Error: "transformers not installed"**
  - Run: `pip install transformers torch`
- **Slow first translation**
  - Models download on first use (can be 1-5 GB)
  - Subsequent uses are faster (cached)

### LibreTranslate
- **Error: Connection refused**
  - Check if server is running: `curl http://localhost:5000/languages`
  - For self-hosted: Set `LIBRETRANSLATE_URL` environment variable

## Performance Notes

- **Ollama**: Medium speed, good quality, completely offline
- **Argos**: Fast, decent quality, completely offline, limited language pairs
- **HuggingFace**: Medium speed, good quality, requires internet for first download
- **LibreTranslate**: Fast (if self-hosted), good quality, requires server

## Recommended Setup

For **offline usage**:
1. Install Ollama + pull model: `ollama pull llama3.1`
2. Or install Argos + language packs

For **best quality**:
1. Use Ollama with `llama3.1` or `qwen2.5`
2. Or use HuggingFace with `facebook/mbart-large-50-many-to-many-mmt`

For **fastest setup**:
1. Use `cascade` or `free` backends (no installation)
2. Or use Argos (lightweight, fast)

