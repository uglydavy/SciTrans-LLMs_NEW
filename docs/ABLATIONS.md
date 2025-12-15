# Ablations

Toggleable features for research studies and baselines.

## PipelineConfig flags
- `enable_masking` (default True)  
- `enable_glossary` (default True)  
- `enable_context` (default True)  
- `enable_refinement` (default True)  
- `enable_reranking` (default False unless set)  
- `enable_adaptive_escalation` (default True)  
- `strict_mode` (default True for thesis runs)

## CLI examples
Disable refinement:
```bash
./scitrans translate input.pdf --enable-refinement false
```

Disable glossary + masking for baseline:
```bash
./scitrans translate input.pdf --enable-glossary false --enable-masking false
```

Disable adaptive escalation:
```bash
./scitrans translate input.pdf --enable-adaptive-escalation false
```

## Reporting
For each ablation, record:
- Flags used
- Backend/model
- Retry/fallback settings
- Metrics: BLEU/chrF, glossary adherence, numeric consistency, layout proxy

## Safety reminder
For thesis-grade runs keep:
- `strict_mode=true`
- `enable_masking=true`
- `enable_glossary=true`
- `enable_refinement=true`
- `enable_adaptive_escalation=true`

