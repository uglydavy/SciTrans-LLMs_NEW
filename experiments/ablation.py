"""Ablation study experiments."""

import json
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass, asdict

from scitran.core.pipeline import TranslationPipeline, PipelineConfig
from scitran.extraction.pdf_parser import PDFParser
from scitran.scoring.metrics import compute_bleu, compute_chrf


@dataclass
class AblationResult:
    """Results from an ablation experiment."""
    config_name: str
    bleu: float
    chrf: float
    time: float
    quality: float
    latex_preserved: float


class AblationStudy:
    """Run ablation studies to validate innovations."""
    
    def __init__(self, corpus_dir: str, output_dir: str = "results/ablation"):
        self.corpus_dir = Path(corpus_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run_all(self) -> List[AblationResult]:
        """Run complete ablation study."""
        
        configs = {
            "baseline": PipelineConfig(
                enable_masking=False,
                enable_context=False,
                preserve_layout=False,
                enable_reranking=False
            ),
            "masking_only": PipelineConfig(
                enable_masking=True,
                enable_context=False,
                preserve_layout=False,
                enable_reranking=False
            ),
            "context_only": PipelineConfig(
                enable_masking=False,
                enable_context=True,
                preserve_layout=False,
                enable_reranking=False
            ),
            "layout_only": PipelineConfig(
                enable_masking=False,
                enable_context=False,
                preserve_layout=True,
                enable_reranking=False
            ),
            "masking_context": PipelineConfig(
                enable_masking=True,
                enable_context=True,
                preserve_layout=False,
                enable_reranking=False
            ),
            "masking_layout": PipelineConfig(
                enable_masking=True,
                enable_context=False,
                preserve_layout=True,
                enable_reranking=False
            ),
            "context_layout": PipelineConfig(
                enable_masking=False,
                enable_context=True,
                preserve_layout=True,
                enable_reranking=False
            ),
            "full_system": PipelineConfig(
                enable_masking=True,
                enable_context=True,
                preserve_layout=True,
                enable_reranking=True
            )
        }
        
        results = []
        
        for config_name, config in configs.items():
            print(f"\nRunning: {config_name}")
            result = self.run_experiment(config_name, config)
            results.append(result)
            
            print(f"  BLEU: {result.bleu:.2f}")
            print(f"  chrF: {result.chrf:.2f}")
            print(f"  Time: {result.time:.2f}s")
        
        # Save results
        self.save_results(results)
        
        return results
    
    def run_experiment(self, config_name: str, config: PipelineConfig) -> AblationResult:
        """Run single experiment."""
        
        # Get test files
        test_files = list(self.corpus_dir.glob("*.pdf"))[:5]  # Use first 5 for quick testing
        
        total_bleu = 0.0
        total_chrf = 0.0
        total_time = 0.0
        total_quality = 0.0
        total_latex = 0.0
        
        parser = PDFParser()
        pipeline = TranslationPipeline(config)
        
        for test_file in test_files:
            # Parse document
            document = parser.parse(str(test_file))
            
            # Translate
            result = pipeline.translate_document(document)
            
            # Load reference if exists
            ref_file = test_file.with_suffix(".ref.txt")
            if ref_file.exists():
                with open(ref_file) as f:
                    reference = f.read()
                
                # Compute metrics
                hypothesis = "\n".join([
                    b.translated_text or b.source_text 
                    for b in result.document.blocks
                ])
                
                total_bleu += compute_bleu([reference], hypothesis)
                total_chrf += compute_chrf([reference], hypothesis)
            
            total_time += result.stats.total_time
            total_quality += result.stats.avg_quality
            total_latex += self._compute_latex_preservation(result.document)
        
        n = len(test_files)
        
        return AblationResult(
            config_name=config_name,
            bleu=total_bleu / n,
            chrf=total_chrf / n,
            time=total_time / n,
            quality=total_quality / n,
            latex_preserved=total_latex / n
        )
    
    def _compute_latex_preservation(self, document) -> float:
        """Compute LaTeX preservation rate."""
        total_latex = 0
        preserved_latex = 0
        
        for block in document.blocks:
            source_latex = block.source_text.count("$") + block.source_text.count("\\")
            if block.translated_text:
                trans_latex = block.translated_text.count("$") + block.translated_text.count("\\")
                total_latex += source_latex
                preserved_latex += min(source_latex, trans_latex)
        
        return (preserved_latex / total_latex * 100) if total_latex > 0 else 100.0
    
    def save_results(self, results: List[AblationResult]):
        """Save results to JSON."""
        output_file = self.output_dir / "ablation_results.json"
        
        with open(output_file, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
    
    def generate_latex_table(self, results: List[AblationResult]) -> str:
        """Generate LaTeX table from results."""
        
        latex = [
            "\\begin{table}[h]",
            "\\centering",
            "\\caption{Ablation Study Results}",
            "\\begin{tabular}{lcccc}",
            "\\hline",
            "Configuration & BLEU & chrF & Time (s) & LaTeX (\\%) \\\\",
            "\\hline"
        ]
        
        for result in results:
            latex.append(
                f"{result.config_name.replace('_', ' ').title()} & "
                f"{result.bleu:.2f} & {result.chrf:.2f} & "
                f"{result.time:.2f} & {result.latex_preserved:.1f} \\\\"
            )
        
        latex.extend([
            "\\hline",
            "\\end{tabular}",
            "\\end{table}"
        ])
        
        return "\n".join(latex)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ablation.py <corpus_dir>")
        sys.exit(1)
    
    study = AblationStudy(sys.argv[1])
    results = study.run_all()
    
    print("\n" + "=" * 60)
    print("ABLATION STUDY COMPLETE")
    print("=" * 60)
    
    latex_table = study.generate_latex_table(results)
    print("\nLaTeX Table:")
    print(latex_table)
