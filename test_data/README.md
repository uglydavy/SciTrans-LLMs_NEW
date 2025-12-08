# Test Data

Sample files for testing SciTrans-LLMs translation functionality.

## Files

- **sample_en.txt** - English text with LaTeX equations, URLs, citations
- **sample_en.ref.txt** - Reference French translation for quality benchmarking
- **sample.html** - HTML document with technical content

## Usage

### Test Translation

```bash
# Test with text file
./scitrans test --backend cascade --sample "$(cat test_data/sample_en.txt)"

# Test with different backends
./scitrans test --backend free
./scitrans test --backend ollama
```

### Run Benchmarks

```bash
# Quality benchmark (needs reference translations)
python benchmarks/quality_test.py test_data/

# Speed benchmark
python benchmarks/speed_test.py
```

### Interactive Testing

```bash
# Use wizard
./scitrans wizard

# Then select test_data/sample.pdf when it becomes available
```

## Adding More Test Data

1. Add your PDF files to `test_data/`
2. Create reference translations as `filename.ref.txt`
3. Run quality benchmarks to evaluate

## Test Data Guidelines

- Include diverse content: equations, code, URLs, citations
- Provide reference translations for evaluation
- Use real scientific papers (with permission)
- Test different languages and domains
