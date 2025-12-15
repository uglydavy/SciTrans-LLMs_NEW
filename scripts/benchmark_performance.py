#!/usr/bin/env python3
"""
Performance benchmark script for SciTrans-LLMs.

Measures translation speed, caching effectiveness, and batch processing performance.
"""

import sys
import time
import statistics
from pathlib import Path
from typing import List, Dict, Any
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scitran.core.pipeline import TranslationPipeline, PipelineConfig
from scitran.core.models import Document, Block, Segment
from scitran.utils.cache import TranslationCache


def create_test_document(num_blocks: int = 10) -> Document:
    """Create a test document with specified number of blocks."""
    blocks = []
    for i in range(num_blocks):
        block = Block(
            block_id=f"block_{i}",
            source_text=f"This is test block number {i}. It contains some text to translate."
        )
        blocks.append(block)
    
    segment = Segment(
        segment_id="test_segment",
        segment_type="body",
        blocks=blocks
    )
    
    document = Document(
        document_id="test_doc",
        segments=[segment]
    )
    
    return document


def benchmark_translation_speed(
    backend: str = "local",
    num_blocks: int = 10,
    num_runs: int = 3
) -> Dict[str, Any]:
    """Benchmark translation speed."""
    print(f"\n{'='*60}")
    print(f"Benchmark: Translation Speed ({backend})")
    print(f"{'='*60}")
    print(f"Blocks: {num_blocks}, Runs: {num_runs}")
    
    config = PipelineConfig(
        backend=backend,
        enable_masking=False,  # Disable for speed test
        enable_reranking=False,
        enable_glossary=False,
        enable_context=False,
        cache_translations=False
    )
    
    pipeline = TranslationPipeline(config)
    document = create_test_document(num_blocks)
    
    times = []
    for run in range(num_runs):
        start = time.time()
        result = pipeline.translate_document(document)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Run {run+1}: {elapsed:.2f}s")
    
    avg_time = statistics.mean(times)
    median_time = statistics.median(times)
    min_time = min(times)
    max_time = max(times)
    blocks_per_sec = num_blocks / avg_time
    
    results = {
        "backend": backend,
        "num_blocks": num_blocks,
        "num_runs": num_runs,
        "avg_time": avg_time,
        "median_time": median_time,
        "min_time": min_time,
        "max_time": max_time,
        "blocks_per_sec": blocks_per_sec,
        "times": times
    }
    
    print(f"\nResults:")
    print(f"  Average: {avg_time:.2f}s")
    print(f"  Median: {median_time:.2f}s")
    print(f"  Min: {min_time:.2f}s")
    print(f"  Max: {max_time:.2f}s")
    print(f"  Throughput: {blocks_per_sec:.2f} blocks/sec")
    
    return results


def benchmark_caching(num_blocks: int = 10) -> Dict[str, Any]:
    """Benchmark caching effectiveness."""
    print(f"\n{'='*60}")
    print(f"Benchmark: Caching Effectiveness")
    print(f"{'='*60}")
    
    cache = TranslationCache(use_disk=False)  # Memory cache for speed
    
    # First run (cache miss)
    start = time.time()
    for i in range(num_blocks):
        cache.set(f"text_{i}", "en", "fr", "local", f"translated_{i}")
    set_time = time.time() - start
    
    # Second run (cache hit)
    start = time.time()
    hits = 0
    for i in range(num_blocks):
        result = cache.get(f"text_{i}", "en", "fr", "local")
        if result:
            hits += 1
    get_time = time.time() - start
    
    hit_rate = hits / num_blocks if num_blocks > 0 else 0
    speedup = set_time / get_time if get_time > 0 else float('inf')
    
    results = {
        "num_blocks": num_blocks,
        "set_time": set_time,
        "get_time": get_time,
        "hit_rate": hit_rate,
        "speedup": speedup,
        "set_ops_per_sec": num_blocks / set_time if set_time > 0 else 0,
        "get_ops_per_sec": num_blocks / get_time if get_time > 0 else 0
    }
    
    print(f"Results:")
    print(f"  Set time: {set_time:.4f}s ({results['set_ops_per_sec']:.1f} ops/sec)")
    print(f"  Get time: {get_time:.4f}s ({results['get_ops_per_sec']:.1f} ops/sec)")
    print(f"  Hit rate: {hit_rate:.1%}")
    print(f"  Speedup: {speedup:.2f}x")
    
    return results


def benchmark_batch_processing(
    batch_sizes: List[int] = [1, 5, 10, 20],
    num_blocks: int = 20
) -> Dict[str, Any]:
    """Benchmark batch processing performance."""
    print(f"\n{'='*60}")
    print(f"Benchmark: Batch Processing")
    print(f"{'='*60}")
    
    results = {}
    
    for batch_size in batch_sizes:
        config = PipelineConfig(
            backend="local",
            batch_size=batch_size,
            enable_masking=False,
            enable_reranking=False,
            enable_glossary=False,
            enable_context=False,
            cache_translations=False
        )
        
        pipeline = TranslationPipeline(config)
        document = create_test_document(num_blocks)
        
        start = time.time()
        result = pipeline.translate_document(document)
        elapsed = time.time() - start
        
        blocks_per_sec = num_blocks / elapsed if elapsed > 0 else 0
        
        results[f"batch_{batch_size}"] = {
            "batch_size": batch_size,
            "time": elapsed,
            "blocks_per_sec": blocks_per_sec
        }
        
        print(f"  Batch size {batch_size}: {elapsed:.2f}s ({blocks_per_sec:.2f} blocks/sec)")
    
    return results


def run_all_benchmarks() -> Dict[str, Any]:
    """Run all benchmarks."""
    print("\n" + "="*60)
    print("SciTrans-LLMs Performance Benchmarks")
    print("="*60)
    
    all_results = {
        "timestamp": time.time(),
        "translation_speed": {},
        "caching": {},
        "batch_processing": {}
    }
    
    try:
        # Translation speed
        all_results["translation_speed"] = benchmark_translation_speed(
            backend="local",
            num_blocks=10,
            num_runs=3
        )
    except Exception as e:
        print(f"Translation speed benchmark failed: {e}")
        all_results["translation_speed"] = {"error": str(e)}
    
    try:
        # Caching
        all_results["caching"] = benchmark_caching(num_blocks=100)
    except Exception as e:
        print(f"Caching benchmark failed: {e}")
        all_results["caching"] = {"error": str(e)}
    
    try:
        # Batch processing
        all_results["batch_processing"] = benchmark_batch_processing(
            batch_sizes=[1, 5, 10],
            num_blocks=20
        )
    except Exception as e:
        print(f"Batch processing benchmark failed: {e}")
        all_results["batch_processing"] = {"error": str(e)}
    
    return all_results


if __name__ == "__main__":
    results = run_all_benchmarks()
    
    # Save results
    output_file = Path("benchmarks/performance_results.json")
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Benchmarks complete!")
    print(f"Results saved to: {output_file}")
    print("="*60)


