# SciTrans-LLMs Architecture

> Related docs: CLI (docs/CLI.md), GUI (docs/GUI.md), Evaluation (docs/EVALUATION.md), Ablations (docs/ABLATIONS.md), Reproducibility (docs/REPRODUCIBILITY.md)

**Version:** 2.0.0  
**Last Updated:** December 2024

---

## Overview

SciTrans-LLMs is a layout-preserving PDF translation pipeline for scientific documents, implementing three core innovations:

1. **Terminology-Constrained Translation** - Mask sensitive content (LaTeX, code, citations) and enforce glossary terms
2. **Document-Level Context** - Maintain consistency across document sections with sliding context windows
3. **Layout Preservation** - Faithful reproduction of PDF structure, fonts, and formatting

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interfaces                          │
├──────────────────────────────┬──────────────────────────────────┤
│   CLI (Typer)                │   GUI (Gradio)                   │
│   cli/commands/main.py       │   gui/app.py                     │
└──────────────────────────────┴──────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Translation Pipeline                          │
│                 scitran/core/pipeline.py                         │
│                                                                  │
│  TranslationPipeline.translate_document()                       │
│  ├─ Load configuration (PipelineConfig)                         │
│  ├─ Load glossary (if enabled)                                  │
│  ├─ Optimize prompts (if enabled)                               │
│  ├─ Process segments in batches                                 │
│  └─ Return TranslationResult                                    │
└─────────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌──────────────┐   ┌──────────────────┐   ┌──────────────┐
│  Extraction  │   │   Translation    │   │  Rendering   │
│              │   │                  │   │              │
│ pdf_parser   │   │  7 Backends:     │   │ pdf_renderer │
│ layout       │   │  - cascade       │   │              │
│              │   │  - free          │   │ Overlays     │
│              │   │  - ollama        │   │ translated   │
│              │   │  - huggingface   │   │ text on PDF  │
│              │   │  - openai        │   │              │
│              │   │  - anthropic     │   │              │
│              │   │  - deepseek      │   │              │
└──────────────┘   └──────────────────┘   └──────────────┘
         │                    │
         ▼                    ▼
┌──────────────┐   ┌──────────────────┐
│   Masking    │   │   Reranking      │
│              │   │                  │
│ MaskingEngine│   │ AdvancedReranker │
│              │   │                  │
│ Protects:    │   │ Multi-candidate  │
│ - LaTeX      │   │ scoring &        │
│ - URLs       │   │ selection        │
│ - DOIs       │   │                  │
│ - Code       │   │                  │
└──────────────┘   └──────────────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│            Support Modules               │
│                                          │
│  - Glossary Management                   │
│  - Prompt Optimization                   │
│  - Caching (diskcache)                   │
│  - Progress Tracking                     │
│  - Logging (loguru)                      │
└──────────────────────────────────────────┘
```

---

## Core Pipeline Flow

### 1. Document Parsing (`scitran/extraction/pdf_parser.py`)

```python
PDFParser.parse(pdf_path) -> Document
  ├─ Extract text blocks with bounding boxes
  ├─ Detect fonts, sizes, styles
  ├─ Identify structure (headers, body, footers)
  ├─ Segment into translatable units
  └─ Return Document model
```

**Output:** `Document` containing `Segment[]`, each with `Block[]` (text + position + metadata)

### 2. Translation Processing (`scitran/core/pipeline.py`)

```python
TranslationPipeline.translate_document(document) -> TranslationResult
  │
  ├─ [INNOVATION #1] Mask sensitive content
  │   └─ MaskingEngine.mask_block(block)
  │       ├─ Detect LaTeX: $...$ $$...$$ \begin{...}
  │       ├─ Detect URLs: https://...
  │       ├─ Detect DOIs: doi:...
  │       ├─ Detect citations: [1], (Author, 2020)
  │       └─ Replace with placeholders: <MATH_0>, <URL_1>, etc.
  │
  ├─ [INNOVATION #2] Add document context
  │   └─ Build context from surrounding blocks (sliding window)
  │
  ├─ Translate with backend
  │   ├─ Build prompt with glossary terms (if enabled)
  │   ├─ Generate N candidates (if reranking enabled)
  │   └─ Call backend API/service
  │
  ├─ [INNOVATION #2] Rerank candidates (if enabled)
  │   └─ AdvancedReranker.rerank(candidates)
  │       ├─ Fluency score
  │       ├─ Glossary adherence score
  │       ├─ Context consistency score
  │       └─ Select best candidate
  │
  ├─ [INNOVATION #1] Restore masked content
  │   └─ Replace placeholders with original content
  │       └─ Validate: all placeholders restored
  │
  └─ Set block.translated_text
```

**⚠️ KNOWN ISSUE (SPRINT 1):** No validation that ALL blocks have `translated_text` before returning.

### 3. PDF Rendering (`scitran/rendering/pdf_renderer.py`)

```python
PDFRenderer.render_with_layout(source_pdf, document, output_pdf)
  │
  ├─ Open source PDF with PyMuPDF
  │
  ├─ For each page:
  │   ├─ Get translated blocks for this page
  │   ├─ Redact original text at block bounding boxes
  │   │   └─ page.add_redact_annot(bbox, fill=white)
  │   ├─ Insert translated text at same positions
  │   │   └─ page.insert_textbox(bbox, translated_text, font, size)
  │   └─ Preserve images, drawings, layout elements
  │
  └─ Save modified PDF
```

**⚠️ KNOWN ISSUE (SPRINT 1):** If `block.translated_text` is `None`, block is skipped silently → half-translated PDF.

---

## Data Models (`scitran/core/models.py`)

### Core Models

```python
@dataclass
class BoundingBox:
    page: int          # 0-indexed page number
    x0, y0: float      # Top-left corner
    x1, y1: float      # Bottom-right corner

@dataclass
class FontInfo:
    name: str
    size: float
    bold: bool = False
    italic: bool = False

@dataclass
class MaskInfo:
    mask_type: str     # MATH, URL, DOI, CODE, CITATION
    placeholder: str   # <MATH_0>, <URL_1>, etc.
    original: str      # Original masked content

@dataclass
class Block:
    block_id: str
    source_text: str
    translated_text: Optional[str] = None
    bbox: Optional[BoundingBox] = None
    font: Optional[FontInfo] = None
    block_type: BlockType = BlockType.BODY
    masks: List[MaskInfo] = field(default_factory=list)
    masked_text: Optional[str] = None
    
@dataclass
class Segment:
    segment_id: str
    blocks: List[Block]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Document:
    doc_id: str
    segments: List[Segment]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def translatable_blocks(self) -> List[Block]:
        """All blocks that should be translated."""
        return [b for seg in self.segments for b in seg.blocks 
                if b.block_type != BlockType.NON_TEXT]

@dataclass
class TranslationResult:
    document: Document
    blocks_translated: int
    duration: float
    bleu_score: Optional[float] = None
    glossary_adherence: Optional[float] = None
    metadata: TranslationMetadata = field(default_factory=TranslationMetadata)
```

---

## Translation Backends

### Backend Architecture

All backends implement `TranslationBackend` abstract base class:

```python
class TranslationBackend(ABC):
    @abstractmethod
    def translate_sync(self, request: TranslationRequest) -> TranslationResponse:
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        pass
```

### Available Backends

| Backend | Type | Cost | Speed | Quality | Use Case |
|---------|------|------|-------|---------|----------|
| **cascade** | Free | Free | Slow | Medium | Development, testing |
| **free** | Free (Google) | Free | Fast | Good | Quick translations |
| **ollama** | Local | Free | Medium | Good | Offline, privacy |
| **huggingface** | Local | Free | Slow | Medium | Offline, no GPU needed |
| **deepseek** | API | $ | Fast | Good | Cost-effective production |
| **openai** | API | $$$ | Fast | Excellent | Best quality |
| **anthropic** | API | $$$ | Fast | Excellent | Long documents |

### Cascade Backend Strategy

```
CascadeBackend.translate_sync(request):
  1. Try LibreTranslate (if available)
     ├─ Success → return
     └─ Fail → continue
  
  2. Try Google Translate (via deep-translator)
     ├─ Success → return
     └─ Fail → continue
  
  3. Try MyMemory
     ├─ Success → return
     └─ Fail → return error
  
  4. Learn glossary terms from successful translations
```

---

## Configuration System

### PipelineConfig (`scitran/core/pipeline.py`)

```python
@dataclass
class PipelineConfig:
    # Language
    source_lang: str = "en"
    target_lang: str = "fr"
    
    # Backend
    backend: str = "cascade"
    model_name: Optional[str] = None
    api_key: Optional[str] = None
    
    # Innovation toggles
    enable_masking: bool = True
    enable_glossary: bool = True
    enable_reranking: bool = True
    enable_context: bool = True
    
    # Quality settings
    num_candidates: int = 3
    quality_threshold: float = 0.7
    
    # Performance
    batch_size: int = 10
    cache_translations: bool = True
    max_retries: int = 3
    timeout: int = 30
```

### Configuration Loading

1. **Default values** (hardcoded in PipelineConfig)
2. **Config files** (`configs/default.yaml`)
3. **Environment variables** (`OPENAI_API_KEY`, etc.)
4. **Command-line arguments** (highest priority)

---

## Glossary System

### Glossary Architecture

```
Glossary Management
├─ Storage: ~/.scitrans/glossary.json (persistent)
├─ Loader: Load from files, online sources, or inline definitions
├─ Enforcement: Inject into prompts + post-translation validation
└─ Domains: ML/AI, Physics, Biology, Chemistry, CS, Legal, etc.
```

### Glossary Workflow

```python
1. Load glossary terms (e.g., "neural network" → "réseau de neurones")
2. Inject into translation prompt:
   "Use these terms: neural network=réseau de neurones, ..."
3. Translate text
4. Validate: Check if glossary terms were used correctly
5. If missed, retry with stronger enforcement or repair
```

### Current Issues (SPRINT 3)

- Glossaries duplicated in GUI code (700+ lines in `gui/app.py`)
- Separate glossary files in `configs/glossary_*.yaml`
- No centralized glossary manager
- No post-translation validation

**SPRINT 3 Goal:** Refactor into `scitran/translation/glossary/manager.py`

---

## Caching System

### Cache Implementation

```python
# Persistent disk cache (diskcache)
cache_dir = ~/.scitrans/cache/
cache_key = hash(source_text + source_lang + target_lang + backend)
```

### Cache Benefits

- **Speed:** Avoid re-translating identical blocks
- **Cost:** Reduce API calls to paid backends
- **Consistency:** Same input always produces same output

### Cache Invalidation

Cache is invalidated when:
- Backend changes
- Model changes
- Glossary changes
- Manual cache clear

---

## Testing Architecture

### Test Structure

```
tests/
├── conftest.py              # Pytest fixtures
├── unit/                    # Fast, isolated tests
│   ├── test_models.py       # Data model tests
│   ├── test_masking.py      # Masking engine tests
│   └── test_glossary.py     # Glossary tests (TODO: SPRINT 3)
├── integration/             # Multi-component tests
│   ├── test_pipeline.py     # Full pipeline without network
│   └── test_full_pipeline.py
└── e2e/                     # Real-world tests
    ├── test_real_pdfs.py    # Test with actual PDFs
    └── test_real_translation.py
```

### Testing Strategy (SPRINT 2)

1. **Unit tests:** Test each component in isolation
2. **Integration tests:** Test component interactions with mock backends
3. **E2E tests:** Test full pipeline with real backends (marked with `@pytest.mark.slow`)
4. **Deterministic tests:** Use DummyTranslator backend (no network calls)

---

## Deployment Options

### Local Installation

```bash
cd SciTrans-LLMs_NEW
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
scitrans gui
```

### Docker Deployment

```bash
docker build -t scitrans-llms .
docker run -p 7860:7860 -v ~/.scitrans:/root/.scitrans scitrans-llms
```

### CLI Usage

```bash
# Translate PDF
scitrans translate paper.pdf --backend cascade

# Launch GUI
scitrans gui

# Test backends
scitrans test --backend free

# Interactive wizard
scitrans wizard
```

---

## Performance Considerations

### Bottlenecks

1. **PDF parsing** (~1-2s per page)
2. **Translation API calls** (varies by backend)
3. **PDF rendering** (~0.5-1s per page)

### Optimization Strategies

1. **Batching:** Process multiple blocks in parallel
2. **Caching:** Reuse translations from cache
3. **Async processing:** Use async backends where available
4. **Incremental rendering:** Render pages as they're translated

### Expected Performance

| Document Size | Backend | Time (approx) |
|---------------|---------|---------------|
| 10 pages | cascade | 60-120s |
| 10 pages | openai | 30-60s |
| 100 pages | cascade | 10-20 min |
| 100 pages | openai | 5-10 min |

---

## Security & Privacy

### API Key Management

- Keys stored in environment variables or `~/.scitrans/config.json`
- Never committed to git (`.gitignore` enforced)
- Keys passed securely to backends

### Data Privacy

- **Local backends (ollama, huggingface):** No data leaves machine
- **Free backends (cascade, free):** Data sent to third-party services
- **Paid backends (openai, anthropic, deepseek):** Data sent to providers (check their privacy policies)

### Licensing

- **SciTrans-LLMs:** MIT License
- **Dependencies:** Mostly permissive (MIT, Apache, BSD)
- **⚠️ YOLO (if used):** GPL-3.0 (ensure compliance)

---

## Known Issues & Roadmap

### Critical Issues (SPRINT 1)

1. **Half-translated PDFs** - Blocks without translations are silently skipped
   - **Fix:** Add translation coverage guarantee with retries/fallback
   - **Fix:** Fail loudly in STRICT mode if any blocks missing

### Structural Issues (SPRINT 2-3)

2. **Glossary fragmentation** - Duplicated across codebase
   - **Fix:** Centralize in `scitran/translation/glossary/manager.py`

3. **No glossary validation** - Terms may be ignored by backends
   - **Fix:** Add post-translation glossary adherence check

4. **Tests not comprehensive** - Missing coverage for core claims
   - **Fix:** Add deterministic tests for all three innovations

### Feature Gaps (SPRINT 4-5)

5. **No document-level refinement** - Each segment translated independently
   - **Add:** Document-level refinement pass with safety checks

6. **No adaptive escalation** - Always use same backend/model
   - **Add:** Adaptive controller that escalates on quality failures

7. **No evaluation harness** - Can't reproduce thesis results
   - **Add:** Experiment runner with BLEU/chrF/COMET metrics

### Documentation Gaps (SPRINT 6)

8. **Docs are conflicting/outdated**
   - **Fix:** Consolidate into single source of truth

---

## Sprint Plan Summary

- **SPRINT 0:** ✅ Repo hygiene, cleanup, architecture
- **SPRINT 1:** Fix half-translation bug (coverage guarantee)
- **SPRINT 2:** Fix tests, add deterministic unit tests
- **SPRINT 3:** Glossary enforcement + validation
- **SPRINT 4:** Document-level context & refinement (safe + ablatable)
- **SPRINT 5:** Evaluation harness & baseline scripts
- **SPRINT 6:** Documentation consolidation

---

## Contributing

See `docs/DEVELOPMENT.md` for:
- Development setup
- Code style guidelines
- How to add new backends
- How to add new tests
- Debugging tips

---

## References

- **Code:** `/Users/kv.kn/Desktop/Research/SciTrans-LLMs_NEW/`
- **Main pipeline:** `scitran/core/pipeline.py`
- **Rendering:** `scitran/rendering/pdf_renderer.py`
- **Backends:** `scitran/translation/backends/`
- **Tests:** `tests/`
- **Docs:** `docs/`

---

**Last Updated:** December 2024  
**Maintainer:** SciTrans Team

