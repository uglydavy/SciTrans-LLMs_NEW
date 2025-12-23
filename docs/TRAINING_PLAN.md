# System Training and Improvement Plan

## Overview
This document outlines the algorithm for training and improving the SciTrans system based on real PDF evaluation metrics.

## Phase 1: Comprehensive Evaluation Framework

### 1.1 Test Dataset Structure
```
test_data/
├── source_pdfs/          # Original PDFs
├── reference_translations/  # Human-verified translations (JSON format)
│   ├── doc1.json         # Block-by-block reference translations
│   └── doc2.json
├── evaluation_results/   # Generated evaluation metrics
└── training_data/        # Collected training examples
```

### 1.2 Metrics Collection
For each PDF translation, collect:
- **Translation Quality**: BLEU, chrF, COMET (if available)
- **Masking Accuracy**: % of placeholders preserved, % lost, % corrupted
- **Block Detection**: Precision, Recall, F1 for block boundaries
- **Style Preservation**: Formality, technicality, domain-specific terms
- **Layout Preservation**: Bbox accuracy, page structure
- **Coverage**: % of blocks translated, % skipped, % failed

### 1.3 Evaluation Script
`scripts/evaluate_system.py`:
- Load test PDFs
- Run translation pipeline
- Compare with reference translations
- Generate comprehensive metrics report
- Save results for training

## Phase 2: Training Algorithm

### 2.1 Failure Analysis
For each failed/mis-translated block:
1. **Identify failure type**:
   - Masking failure (placeholder lost/corrupted)
   - Translation quality failure (low BLEU/chrF)
   - Block detection failure (wrong boundaries)
   - Style mismatch (too formal/informal)
   - Coverage failure (block skipped)

2. **Extract features**:
   - Source text characteristics (length, complexity, domain)
   - Block type (paragraph, table, figure, equation)
   - Context (surrounding blocks)
   - Masking patterns (what was masked)
   - Translation backend used
   - Prompt template used

3. **Store training example**:
   ```json
   {
     "source_text": "...",
     "reference_translation": "...",
     "system_translation": "...",
     "failure_type": "masking|quality|detection|style|coverage",
     "features": {...},
     "metrics": {...}
   }
   ```

### 2.2 Improvement Strategies

#### A. Prompt Optimization
- **Input**: Failed translations with low quality scores
- **Algorithm**: 
  1. Analyze common patterns in failures
  2. Generate prompt variations
  3. Test on validation set
  4. Select best-performing prompt
  5. Update prompt template

#### B. Masking Rule Learning
- **Input**: Masking failures (lost/corrupted placeholders)
- **Algorithm**:
  1. Identify patterns in failed masking
  2. Generate new masking rules
  3. Test on validation set
  4. Add successful rules to masking engine

#### C. Block Detection Improvement
- **Input**: Block detection errors (wrong boundaries)
- **Algorithm**:
  1. Analyze PDF structure patterns
  2. Improve heuristics (font size, spacing, alignment)
  3. Fine-tune YOLO model (if using Doc_YOLO)
  4. Update block detection logic

#### D. Style Adaptation
- **Input**: Style mismatches
- **Algorithm**:
  1. Detect style features (formality, technicality)
  2. Adjust prompts based on detected style
  3. Learn style-specific translations

### 2.3 Training Loop

```python
def training_loop(test_pdfs, iterations=10):
    for iteration in range(iterations):
        # 1. Evaluate current system
        results = evaluate_system(test_pdfs)
        
        # 2. Identify failures
        failures = analyze_failures(results)
        
        # 3. Generate improvements
        improvements = generate_improvements(failures)
        
        # 4. Apply improvements
        apply_improvements(improvements)
        
        # 5. Validate on held-out set
        validation_score = validate_improvements()
        
        # 6. If improved, keep changes; else revert
        if validation_score > previous_score:
            save_checkpoint()
        else:
            revert_changes()
```

## Phase 3: Continuous Learning

### 3.1 Online Learning
- Collect user feedback (corrections, ratings)
- Update training data
- Retrain periodically

### 3.2 A/B Testing
- Test new improvements against baseline
- Use statistical significance testing
- Deploy best-performing version

## Implementation Priority

1. **Week 1**: Evaluation framework + metrics collection
2. **Week 2**: Failure analysis + training data collection
3. **Week 3**: Prompt optimization algorithm
4. **Week 4**: Masking rule learning
5. **Week 5**: Block detection improvement
6. **Week 6**: Style adaptation
7. **Week 7**: Full training loop integration
8. **Week 8**: Testing and refinement

