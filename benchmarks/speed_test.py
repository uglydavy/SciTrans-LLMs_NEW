"""Speed benchmarking."""

import time
from pathlib import Path
from typing import Dict, List
import statistics

from scitran.core.pipeline import TranslationPipeline, PipelineConfig
from scitran.extraction.pdf_parser import PDFParser


class SpeedBenchmark:
    """Benchmark translation speed."""
    
    def __init__(self, test_dir: str = "corpus/test"):
        self.test_dir = Path(test_dir)
    
    def benchmark_backend(self, backend: str, num_runs: int = 3) -> Dict:
        """Benchmark a specific backend."""
        
        config = PipelineConfig(backend=backend)
        pipeline = TranslationPipeline(config)
        parser = PDFParser()
        
        # Get test file
        test_files = list(self.test_dir.glob("*.pdf"))
        if not test_files:
            raise ValueError(f"No test files found in {self.test_dir}")
        
        test_file = test_files[0]
        
        times = []
        
        for run in range(num_runs):
            print(f"  Run {run + 1}/{num_runs}...")
            
            document = parser.parse(str(test_file), max_pages=5)
            
            start = time.time()
            result = pipeline.translate_document(document)
            elapsed = time.time() - start
            
            times.append(elapsed)
        
        return {
            "backend": backend,
            "mean_time": statistics.mean(times),
            "std_time": statistics.stdev(times) if len(times) > 1 else 0,
            "min_time": min(times),
            "max_time": max(times),
            "runs": num_runs
        }
    
    def benchmark_all(self, backends: List[str] = None) -> List[Dict]:
        """Benchmark all backends."""
        
        if backends is None:
            backends = ["openai", "anthropic", "deepseek", "free"]
        
        results = []
        
        for backend in backends:
            print(f"\nBenchmarking {backend}...")
            try:
                result = self.benchmark_backend(backend)
                results.append(result)
                print(f"  Mean: {result['mean_time']:.2f}s")
            except Exception as e:
                print(f"  Error: {str(e)}")
        
        return results
    
    def print_results(self, results: List[Dict]):
        """Print formatted results."""
        
        print("\n" + "=" * 60)
        print("SPEED BENCHMARK RESULTS")
        print("=" * 60)
        
        for result in results:
            print(f"\n{result['backend'].upper()}")
            print(f"  Mean time: {result['mean_time']:.2f}s")
            print(f"  Std dev: {result['std_time']:.2f}s")
            print(f"  Min/Max: {result['min_time']:.2f}s / {result['max_time']:.2f}s")


if __name__ == "__main__":
    benchmark = SpeedBenchmark()
    results = benchmark.benchmark_all()
    benchmark.print_results(results)
