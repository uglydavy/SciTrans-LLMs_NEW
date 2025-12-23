# Translation Algorithm Consistency Issues

## Summary

This document identifies and fixes inconsistencies that could alter translation results between runs.

## Issues Found and Fixed

### 1. Temperature Inconsistencies ✅ FIXED

**Problem**: Temperature was hardcoded in multiple places with different values:
- Batch candidates: `temperature=0.3` (line 1682)
- Sequential candidates: `temperature=0.3 + i*0.2` (varies: 0.3, 0.5, 0.7...) (line 1691)
- Fallback: `temperature=0.3` (line 1854)
- Refinement: `temperature=0.5` (line 1322)
- Config had no `temperature` field, defaulting to 0.0

**Impact**: Different temperature values cause non-deterministic outputs, especially when temperature > 0.

**Fix**: 
- Added `temperature: float = 0.0` to `PipelineConfig` (defaults to 0.0 for deterministic output)
- All translation calls now use `self.config.temperature` for consistency
- Sequential candidates use same temperature (not varying)

### 2. Async Batch Processing Order ⚠️ TRACKED

**Problem**: `asyncio.gather()` doesn't guarantee completion order. Results dict is populated as tasks complete, which could vary between runs.

**Impact**: If context depends on order, translations might differ.

**Fix**: 
- Added instrumentation to track completion order
- Results are still applied in original block order (line 1484-1487)
- Context is built before batch translation, so order shouldn't affect context

### 3. Reranking Selection ⚠️ TRACKED

**Problem**: Reranking might select different candidates if scores are very close (floating point precision).

**Impact**: Different candidates selected = different translations.

**Fix**: 
- Added instrumentation to track which candidate is selected
- Reranking uses deterministic scoring, but close scores could still cause issues

### 4. Context Window Dependency ⚠️ MONITORED

**Problem**: Context window depends on order of processing. If blocks are processed in different orders, context will differ.

**Impact**: Different context = different translations.

**Fix**: 
- Sequential mode processes blocks in document order (deterministic)
- Batch mode builds context before translation (order-independent)
- Context is built from previous translations, so order matters

### 5. Caching ⚠️ EXPECTED BEHAVIOR

**Problem**: Cache hits vs misses produce different results (cache is populated during first run).

**Impact**: First run vs subsequent runs will differ (expected behavior).

**Fix**: 
- This is expected - cache improves speed
- Consistency test clears cache between runs

## Testing

Use `scripts/test_consistency.py` to test for inconsistencies:

```bash
python scripts/test_consistency.py path/to/test.pdf --num-runs 3 --output consistency_report.json
```

This will:
1. Run translation 3 times on the same PDF
2. Compare translations for each block
3. Check masking consistency
4. Check block order consistency
5. Check coverage consistency
6. Generate a report with all inconsistencies found

## Recommendations

1. **Always use temperature=0.0** for deterministic output (default)
2. **Use sequential mode** for maximum consistency (order-dependent context)
3. **Disable caching** for consistency tests (or clear cache between runs)
4. **Monitor reranking** - if scores are very close, consider deterministic tie-breaking

