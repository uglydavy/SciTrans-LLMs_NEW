# Training System Usage Guide

## Quick Start

### 1. Prepare Test Data

Create a directory structure:
```
test_data/
├── source_pdfs/          # Your test PDFs
│   ├── paper1.pdf
│   └── paper2.pdf
└── reference_translations/  # Optional: reference translations
    ├── paper1.json
    └── paper2.json
```

Reference translation format (`paper1.json`):
```json
{
  "blocks": [
    {
      "block_id": "block_1",
      "translation": "Reference translation here..."
    },
    {
      "block_id": "block_2",
      "translation": "Another reference translation..."
    }
  ]
}
```

### 2. Run Evaluation

Evaluate the current system:
```bash
python scripts/evaluate_system.py \
  --pdf-dir test_data/source_pdfs \
  --reference-dir test_data/reference_translations \
  --output-dir evaluation_results \
  --backend deepseek \
  --source-lang en \
  --target-lang fr
```

This will:
- Translate all PDFs
- Compare with reference translations (if provided)
- Compute metrics (BLEU, chrF, masking accuracy, etc.)
- Save results to `evaluation_results/`

### 3. Analyze Failures

```python
from scitran.training.trainer import FailureAnalyzer
from scitran.core.pipeline import TranslationPipeline, PipelineConfig
import json

# Load evaluation results
with open('evaluation_results/evaluation_report.json') as f:
    results = json.load(f)

# Analyze failures
analyzer = FailureAnalyzer()
patterns = analyzer.identify_patterns()

for pattern in patterns:
    print(f"Pattern: {pattern.pattern_type}")
    print(f"Frequency: {pattern.frequency}")
    print(f"Suggestion: {pattern.suggested_fix}")
```

### 4. Run Training Loop

```python
from scitran.training.trainer import TrainingLoop
from scitran.core.pipeline import PipelineConfig
from scripts.evaluate_system import SystemEvaluator
from pathlib import Path

# Setup
config = PipelineConfig(
    backend="deepseek",
    source_lang="en",
    target_lang="fr"
)
evaluator = SystemEvaluator(config)

# Create training loop
loop = TrainingLoop(
    test_pdfs_dir=Path("test_data/source_pdfs"),
    reference_dir=Path("test_data/reference_translations"),
    output_dir=Path("training_results")
)

# Run iterations
for iteration in range(1, 6):  # 5 iterations
    checkpoint = loop.run_iteration(iteration, config, evaluator)
    print(f"Iteration {iteration} complete")
    print(f"Average BLEU: {checkpoint['aggregate_metrics'].get('average_bleu', 0):.3f}")
```

## Metrics Explained

### Translation Quality
- **BLEU**: N-gram overlap with reference (0-1, higher is better)
- **chrF**: Character-level F-score (0-1, higher is better)
- **COMET**: Contextual embedding similarity (optional, requires model)

### Masking Accuracy
- **Preservation Rate**: % of placeholders correctly preserved
- **Loss Rate**: % of placeholders lost in translation
- **Corruption Rate**: % of placeholders where original text appeared

### Coverage
- **Coverage Rate**: % of translatable blocks successfully translated
- **Failed Blocks**: List of block IDs that failed

### Block Detection
- **Bbox Coverage**: % of blocks with bounding boxes
- **Blocks by Type**: Distribution of block types detected

## Improvement Strategies

### 1. Prompt Optimization
When quality failures are detected:
- Analyze common patterns in low-quality translations
- Generate prompt variations emphasizing quality
- Test variations and select best-performing

### 2. Masking Rule Learning
When masking failures occur:
- Identify patterns in lost/corrupted placeholders
- Generate new masking rules for those patterns
- Test and integrate successful rules

### 3. Block Detection Improvement
When detection errors occur:
- Analyze PDF structure patterns
- Improve heuristics (font size, spacing, alignment)
- Update detection logic

### 4. Coverage Improvement
When blocks are skipped:
- Increase retry count
- Enable fallback backend
- Improve error handling

## Example Workflow

```bash
# 1. Initial evaluation
python scripts/evaluate_system.py \
  --pdf-dir test_data/source_pdfs \
  --output-dir baseline_results

# 2. Review results
cat baseline_results/evaluation_report.json | jq '.aggregate_metrics'

# 3. Run training iterations
python -c "
from scitran.training.trainer import TrainingLoop
from scitran.core.pipeline import PipelineConfig
from scripts.evaluate_system import SystemEvaluator
from pathlib import Path

config = PipelineConfig(backend='deepseek')
evaluator = SystemEvaluator(config)
loop = TrainingLoop(
    test_pdfs_dir=Path('test_data/source_pdfs'),
    output_dir=Path('training_results')
)

for i in range(1, 6):
    checkpoint = loop.run_iteration(i, config, evaluator)
    print(f'Iteration {i}: BLEU={checkpoint[\"aggregate_metrics\"].get(\"average_bleu\", 0):.3f}')
"

# 4. Compare improvements
python -c "
import json
baseline = json.load(open('baseline_results/evaluation_report.json'))
final = json.load(open('training_results/checkpoint_5.json'))
print(f'BLEU improvement: {final[\"aggregate_metrics\"][\"average_bleu\"] - baseline[\"aggregate_metrics\"][\"average_bleu\"]:.3f}')
"
```

## Next Steps

1. **Collect more test data**: More diverse PDFs = better training
2. **Create reference translations**: Human-verified translations improve metrics
3. **Iterate**: Run multiple training iterations
4. **Monitor**: Track metrics over time
5. **Deploy**: Use improved prompts/configs in production

