"""Quality benchmarking with BLEU and chrF scores."""

from pathlib import Path
from typing import Dict, List

from scitran.core.pipeline import TranslationPipeline, PipelineConfig
from scitran.extraction.pdf_parser import PDFParser
from scitran.scoring.metrics import compute_bleu, compute_chrf


class QualityBenchmark:
    """Benchmark translation quality."""
    
    def __init__(self, test_dir: str = "corpus/test", reference_dir: str = None):
        self.test_dir = Path(test_dir)
        self.reference_dir = Path(reference_dir) if reference_dir else self.test_dir
    
    def benchmark_backend(self, backend: str) -> Dict:
        """Benchmark quality for a specific backend."""
        
        config = PipelineConfig(backend=backend)
        pipeline = TranslationPipeline(config)
        parser = PDFParser()
        
        test_files = list(self.test_dir.glob("*.pdf"))
        
        total_bleu = 0.0
        total_chrf = 0.0
        count = 0
        
        for test_file in test_files:
            # Check for reference
            ref_file = self.reference_dir / f"{test_file.stem}.ref.txt"
            if not ref_file.exists():
                continue
            
            with open(ref_file) as f:
                reference = f.read()
            
            # Translate
            document = parser.parse(str(test_file))
            result = pipeline.translate_document(document)
            
            # Get hypothesis
            hypothesis = "\n".join([
                b.translated_text or b.source_text 
                for b in result.document.blocks
            ])
            
            # Compute metrics
            bleu = compute_bleu([reference], hypothesis)
            chrf = compute_chrf([reference], hypothesis)
            
            total_bleu += bleu
            total_chrf += chrf
            count += 1
            
            print(f"  {test_file.name}: BLEU={bleu:.2f}, chrF={chrf:.2f}")
        
        return {
            "backend": backend,
            "bleu": total_bleu / count if count > 0 else 0,
            "chrf": total_chrf / count if count > 0 else 0,
            "num_docs": count
        }
    
    def benchmark_all(self, backends: List[str] = None) -> List[Dict]:
        """Benchmark all backends."""
        
        if backends is None:
            backends = ["openai", "anthropic", "deepseek"]
        
        results = []
        
        for backend in backends:
            print(f"\nBenchmarking {backend}...")
            try:
                result = self.benchmark_backend(backend)
                results.append(result)
            except Exception as e:
                print(f"  Error: {str(e)}")
        
        return results
    
    def print_results(self, results: List[Dict]):
        """Print formatted results."""
        
        print("\n" + "=" * 60)
        print("QUALITY BENCHMARK RESULTS")
        print("=" * 60)
        
        for result in results:
            print(f"\n{result['backend'].upper()}")
            print(f"  BLEU: {result['bleu']:.2f}")
            print(f"  chrF: {result['chrf']:.2f}")
            print(f"  Documents: {result['num_docs']}")


if __name__ == "__main__":
    import sys
    
    test_dir = sys.argv[1] if len(sys.argv) > 1 else "corpus/test"
    
    benchmark = QualityBenchmark(test_dir)
    results = benchmark.benchmark_all()
    benchmark.print_results(results)
