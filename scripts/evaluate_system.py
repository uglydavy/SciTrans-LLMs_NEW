#!/usr/bin/env python3
"""
Comprehensive evaluation framework for SciTrans system.

Evaluates translation quality, masking accuracy, block detection, and more
using real PDFs and reference translations.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scitran.core.pipeline import TranslationPipeline, PipelineConfig
from scitran.core.models import Document, Block
from scitran.evaluation.metrics import evaluate_translation, compute_bleu, compute_chrf
from scitran.extraction.pdf_parser import PDFParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SystemEvaluator:
    """Comprehensive system evaluator."""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.pipeline = TranslationPipeline(config=self.config)
        self.parser = PDFParser()
        self.results = []
    
    def evaluate_pdf(
        self,
        pdf_path: Path,
        reference_path: Optional[Path] = None,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Evaluate translation of a single PDF.
        
        Args:
            pdf_path: Path to source PDF
            reference_path: Optional path to reference translation JSON
            output_dir: Directory to save evaluation results
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating PDF: {pdf_path.name}")
        
        # Load reference if available
        reference_data = None
        if reference_path and reference_path.exists():
            with open(reference_path, 'r', encoding='utf-8') as f:
                reference_data = json.load(f)
        
        # Run translation
        try:
            # Parse PDF first
            document = self.parser.parse(str(pdf_path))
            
            # Then translate
            result = self.pipeline.translate_document(document)
            document = result.document  # Get translated document
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return {
                "pdf_name": pdf_path.name,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        
        # Extract metrics
        metrics = self._compute_metrics(document, reference_data)
        
        # Add pipeline stats
        pipeline_stats = self.pipeline.get_statistics()
        metrics.update({
            "pipeline_stats": pipeline_stats,
            "success": result.success,
            "coverage": result.coverage,
            "duration": result.duration
        })
        
        # Save results
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            result_file = output_dir / f"{pdf_path.stem}_evaluation.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved evaluation results to {result_file}")
        
        self.results.append(metrics)
        return metrics
    
    def _compute_metrics(
        self,
        document: Document,
        reference_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Compute comprehensive metrics."""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "total_blocks": len(document.all_blocks),
            "translatable_blocks": len(document.translatable_blocks),
            "translated_blocks": sum(1 for b in document.translatable_blocks if b.translated_text),
            "failed_blocks": sum(1 for b in document.translatable_blocks if not b.translated_text),
        }
        
        # Translation quality metrics (if reference available)
        if reference_data:
            hypotheses = []
            references = []
            sources = []
            
            # Match blocks with reference
            ref_blocks = {b['block_id']: b for b in reference_data.get('blocks', [])}
            
            for block in document.translatable_blocks:
                if block.translated_text and block.block_id in ref_blocks:
                    ref = ref_blocks[block.block_id]
                    hypotheses.append(block.translated_text)
                    references.append(ref.get('translation', ''))
                    sources.append(block.source_text)
            
            if hypotheses and references:
                # BLEU and chrF
                metrics['bleu'] = compute_bleu(hypotheses, references)
                metrics['chrf'] = compute_chrf(hypotheses, references)
                
                # Full evaluation bundle
                eval_metrics = evaluate_translation(
                    hypotheses=hypotheses,
                    references=references,
                    sources=sources,
                    glossary=self.pipeline.glossary,
                    blocks=document.all_blocks,
                    include_comet=False  # COMET requires model download
                )
                metrics.update(eval_metrics)
        
        # Masking accuracy
        masking_metrics = self._compute_masking_metrics(document)
        metrics['masking'] = masking_metrics
        
        # Block detection metrics
        detection_metrics = self._compute_detection_metrics(document)
        metrics['block_detection'] = detection_metrics
        
        # Style metrics
        style_metrics = self._compute_style_metrics(document)
        metrics['style'] = style_metrics
        
        # Coverage metrics
        coverage_metrics = self._compute_coverage_metrics(document)
        metrics['coverage'] = coverage_metrics
        
        return metrics
    
    def _compute_masking_metrics(self, document: Document) -> Dict[str, Any]:
        """Compute masking accuracy metrics."""
        total_masks = 0
        preserved_masks = 0
        lost_masks = 0
        corrupted_masks = 0
        
        for block in document.translatable_blocks:
            if not block.masks:
                continue
            
            total_masks += len(block.masks)
            
            if block.translated_text:
                for mask in block.masks:
                    if mask.placeholder in block.translated_text:
                        preserved_masks += 1
                    elif mask.original in block.translated_text:
                        # Original appeared (should have been masked)
                        corrupted_masks += 1
                    else:
                        lost_masks += 1
        
        return {
            "total_masks": total_masks,
            "preserved": preserved_masks,
            "lost": lost_masks,
            "corrupted": corrupted_masks,
            "preservation_rate": preserved_masks / total_masks if total_masks > 0 else 1.0,
            "loss_rate": lost_masks / total_masks if total_masks > 0 else 0.0,
            "corruption_rate": corrupted_masks / total_masks if total_masks > 0 else 0.0
        }
    
    def _compute_detection_metrics(self, document: Document) -> Dict[str, Any]:
        """Compute block detection metrics."""
        blocks_by_type = {}
        blocks_with_bbox = 0
        blocks_without_bbox = 0
        
        for block in document.all_blocks:
            block_type = block.block_type.name if hasattr(block.block_type, 'name') else str(block.block_type)
            blocks_by_type[block_type] = blocks_by_type.get(block_type, 0) + 1
            
            if block.bbox:
                blocks_with_bbox += 1
            else:
                blocks_without_bbox += 1
        
        return {
            "blocks_by_type": blocks_by_type,
            "blocks_with_bbox": blocks_with_bbox,
            "blocks_without_bbox": blocks_without_bbox,
            "bbox_coverage": blocks_with_bbox / len(document.all_blocks) if document.all_blocks else 0.0
        }
    
    def _compute_style_metrics(self, document: Document) -> Dict[str, Any]:
        """Compute style preservation metrics."""
        # Placeholder for style metrics
        # Could include: formality, technicality, domain-specific terms
        return {
            "note": "Style metrics require reference translations or style classifiers"
        }
    
    def _compute_coverage_metrics(self, document: Document) -> Dict[str, Any]:
        """Compute coverage metrics."""
        translatable = document.translatable_blocks
        translated = [b for b in translatable if b.translated_text]
        failed = [b for b in translatable if not b.translated_text]
        
        return {
            "total_translatable": len(translatable),
            "translated": len(translated),
            "failed": len(failed),
            "coverage_rate": len(translated) / len(translatable) if translatable else 0.0,
            "failed_block_ids": [b.block_id for b in failed]
        }
    
    def generate_report(self, output_path: Path) -> None:
        """Generate comprehensive evaluation report."""
        if not self.results:
            logger.warning("No results to report")
            return
        
        report = {
            "evaluation_date": datetime.now().isoformat(),
            "total_pdfs": len(self.results),
            "config": {
                "backend": self.config.backend,
                "source_lang": self.config.source_lang,
                "target_lang": self.config.target_lang,
                "enable_masking": self.config.enable_masking,
                "enable_reranking": self.config.enable_reranking,
                "enable_context": self.config.enable_context,
            },
            "aggregate_metrics": self._aggregate_metrics(),
            "per_pdf_results": self.results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Generated evaluation report: {output_path}")
    
    def _aggregate_metrics(self) -> Dict[str, Any]:
        """Aggregate metrics across all PDFs."""
        if not self.results:
            return {}
        
        successful = [r for r in self.results if r.get('success', False)]
        
        return {
            "success_rate": len(successful) / len(self.results),
            "average_coverage": sum(r.get('coverage', 0) for r in successful) / len(successful) if successful else 0,
            "average_bleu": sum(r.get('bleu', 0) or 0 for r in successful) / len(successful) if successful else 0,
            "average_chrf": sum(r.get('chrf', 0) or 0 for r in successful) / len(successful) if successful else 0,
            "average_masking_preservation": sum(
                r.get('masking', {}).get('preservation_rate', 0) for r in successful
            ) / len(successful) if successful else 0,
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate SciTrans system on test PDFs")
    parser.add_argument("--pdf-dir", type=Path, required=True, help="Directory containing test PDFs")
    parser.add_argument("--reference-dir", type=Path, help="Directory containing reference translations")
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation_results"), help="Output directory")
    parser.add_argument("--backend", type=str, default="deepseek", help="Translation backend")
    parser.add_argument("--source-lang", type=str, default="en", help="Source language")
    parser.add_argument("--target-lang", type=str, default="fr", help="Target language")
    
    args = parser.parse_args()
    
    # Create config
    config = PipelineConfig(
        backend=args.backend,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        enable_masking=True,
        enable_reranking=True,
        enable_context=True,
        strict_mode=True
    )
    
    # Create evaluator
    evaluator = SystemEvaluator(config)
    
    # Find PDFs
    pdf_files = list(args.pdf_dir.glob("*.pdf"))
    if not pdf_files:
        logger.error(f"No PDFs found in {args.pdf_dir}")
        return
    
    logger.info(f"Found {len(pdf_files)} PDFs to evaluate")
    
    # Evaluate each PDF
    for pdf_path in pdf_files:
        reference_path = None
        if args.reference_dir:
            reference_path = args.reference_dir / f"{pdf_path.stem}.json"
            if not reference_path.exists():
                reference_path = None
        
        evaluator.evaluate_pdf(
            pdf_path=pdf_path,
            reference_path=reference_path,
            output_dir=args.output_dir
        )
    
    # Generate report
    report_path = args.output_dir / "evaluation_report.json"
    evaluator.generate_report(report_path)
    
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()

