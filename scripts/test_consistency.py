#!/usr/bin/env python3
"""
Test translation algorithm for consistency.

Runs the same PDF multiple times and checks for inconsistencies that could
alter translation results.
"""

import json
import hashlib
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Set
from datetime import datetime
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scitran.core.pipeline import TranslationPipeline, PipelineConfig
from scitran.core.models import Document, Block
from scitran.extraction.pdf_parser import PDFParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Debug logging setup
DEBUG_LOG_PATH = Path("/Users/kv.kn/Desktop/Research/SciTrans-LLMs_NEW/.cursor/debug.log")

def debug_log(session_id, run_id, hypothesis_id, location, message, data):
    """Write debug log entry."""
    try:
        log_entry = {
            "sessionId": session_id,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        with open(DEBUG_LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    except Exception as e:
        logger.debug("Failed to write debug log: %s", e)


class ConsistencyTester:
    """Test translation consistency across multiple runs."""
    
    def __init__(self, config, num_runs=3):
        self.config = config
        self.num_runs = num_runs
        self.results = []
        self.inconsistencies = []
    
    def test_pdf(self, pdf_path):
        """Test consistency of translating a single PDF."""
        logger.info(f"Testing consistency for: {pdf_path.name}")
        
        session_id = f"consistency_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Run multiple times
        for run_id in range(1, self.num_runs + 1):
            logger.info(f"Run {run_id}/{self.num_runs}")
            
            # #region agent log
            debug_log(
                session_id=session_id,
                run_id=f"run_{run_id}",
                hypothesis_id="H1",
                location="test_consistency.py:test_pdf",
                message="Starting translation run",
                data={"run_id": run_id, "pdf": str(pdf_path)}
            )
            # #endregion
            
            try:
                # Parse PDF first
                parser = PDFParser()
                document = parser.parse(str(pdf_path))
                
                # Then translate
                pipeline = TranslationPipeline(config=self.config)
                result = pipeline.translate_document(document)
                document = result.document  # Get translated document
                
                # Extract block translations
                block_translations = {}
                block_masks = {}
                block_order = []
                
                for block in document.translatable_blocks:
                    block_id = block.block_id
                    block_order.append(block_id)
                    
                    # #region agent log
                    debug_log(
                        session_id=session_id,
                        run_id=f"run_{run_id}",
                        hypothesis_id="H2",
                        location="test_consistency.py:test_pdf",
                        message="Block processed",
                        data={
                            "block_id": block_id,
                            "has_translation": bool(block.translated_text),
                            "translation_hash": hashlib.md5(block.translated_text.encode() if block.translated_text else b"").hexdigest()[:8],
                            "order": len(block_order)
                        }
                    )
                    # #endregion
                    
                    block_translations[block_id] = block.translated_text or ""
                    
                    # Extract mask information
                    if block.masks:
                        mask_info = []
                        for mask in block.masks:
                            mask_info.append({
                                "type": mask.mask_type.value if hasattr(mask.mask_type, 'value') else str(mask.mask_type),
                                "original": mask.original[:50],  # Truncate for logging
                                "placeholder": mask.placeholder
                            })
                        block_masks[block_id] = mask_info
                    
                    # #region agent log
                    debug_log(
                        session_id=session_id,
                        run_id=f"run_{run_id}",
                        hypothesis_id="H3",
                        location="test_consistency.py:test_pdf",
                        message="Masking info",
                        data={
                            "block_id": block_id,
                            "num_masks": len(block.masks) if block.masks else 0,
                            "masks": block_masks.get(block_id, [])
                        }
                    )
                    # #endregion
                
                # #region agent log
                debug_log(
                    session_id=session_id,
                    run_id=f"run_{run_id}",
                    hypothesis_id="H4",
                    location="test_consistency.py:test_pdf",
                    message="Translation complete",
                    data={
                        "num_blocks": len(block_translations),
                        "coverage": result.coverage,
                        "success": result.success,
                        "block_order": block_order[:5]  # First 5 for logging
                    }
                )
                # #endregion
                
                run_result = {
                    "run_id": run_id,
                    "success": result.success,
                    "coverage": result.coverage,
                    "block_translations": block_translations,
                    "block_masks": block_masks,
                    "block_order": block_order,
                    "pipeline_stats": pipeline.get_statistics()
                }
                
                self.results.append(run_result)
                
            except Exception as e:
                logger.error(f"Run {run_id} failed: {e}")
                # #region agent log
                debug_log(
                    session_id=session_id,
                    run_id=f"run_{run_id}",
                    hypothesis_id="H5",
                    location="test_consistency.py:test_pdf",
                    message="Translation failed",
                    data={"error": str(e)}
                )
                # #endregion
                self.results.append({
                    "run_id": run_id,
                    "success": False,
                    "error": str(e)
                })
        
        # Analyze consistency
        analysis = self._analyze_consistency()
        return analysis
    
    def _analyze_consistency(self):
        """Analyze results for inconsistencies."""
        if len(self.results) < 2:
            return {"error": "Need at least 2 runs to check consistency"}
        
        successful_runs = [r for r in self.results if r.get('success', False)]
        if len(successful_runs) < 2:
            return {
                "consistent": False,
                "reason": "Not enough successful runs",
                "successful_runs": len(successful_runs)
            }
        
        # Check 1: Translation consistency
        translation_inconsistencies = self._check_translation_consistency(successful_runs)
        
        # Check 2: Masking consistency
        masking_inconsistencies = self._check_masking_consistency(successful_runs)
        
        # Check 3: Block order consistency
        order_inconsistencies = self._check_order_consistency(successful_runs)
        
        # Check 4: Coverage consistency
        coverage_inconsistencies = self._check_coverage_consistency(successful_runs)
        
        all_inconsistencies = (
            translation_inconsistencies +
            masking_inconsistencies +
            order_inconsistencies +
            coverage_inconsistencies
        )
        
        return {
            "consistent": len(all_inconsistencies) == 0,
            "total_runs": len(self.results),
            "successful_runs": len(successful_runs),
            "inconsistencies": all_inconsistencies,
            "translation_consistency": {
                "total_blocks": len(translation_inconsistencies),
                "inconsistent_blocks": len([i for i in translation_inconsistencies if i['type'] == 'translation_diff'])
            },
            "masking_consistency": {
                "total_blocks_with_masks": len(masking_inconsistencies),
                "inconsistent_masks": len([i for i in masking_inconsistencies if i['type'] == 'mask_diff'])
            },
            "order_consistency": {
                "consistent": len(order_inconsistencies) == 0,
                "inconsistencies": order_inconsistencies
            },
            "coverage_consistency": {
                "consistent": len(coverage_inconsistencies) == 0,
                "inconsistencies": coverage_inconsistencies
            }
        }
    
    def _check_translation_consistency(self, runs):
        """Check if translations are identical across runs."""
        inconsistencies = []
        
        # Get all block IDs from first run
        first_run = runs[0]
        block_ids = set(first_run.get('block_translations', {}).keys())
        
        for block_id in block_ids:
            translations = []
            for run in runs:
                trans = run.get('block_translations', {}).get(block_id, "")
                translations.append(trans)
            
            # Check if all translations are identical
            if len(set(translations)) > 1:
                inconsistencies.append({
                    "type": "translation_diff",
                    "block_id": block_id,
                    "translations": translations,
                    "runs": [r['run_id'] for r in runs]
                })
        
        return inconsistencies
    
    def _check_masking_consistency(self, runs):
        """Check if masking is consistent across runs."""
        inconsistencies = []
        
        # Get all block IDs with masks from first run
        first_run = runs[0]
        block_ids_with_masks = set(first_run.get('block_masks', {}).keys())
        
        for block_id in block_ids_with_masks:
            mask_sets = []
            for run in runs:
                masks = run.get('block_masks', {}).get(block_id, [])
                # Convert to comparable format
                mask_set = set(
                    (m['type'], m['placeholder']) for m in masks
                )
                mask_sets.append(mask_set)
            
            # Check if all mask sets are identical
            if len(set(tuple(sorted(ms)) for ms in mask_sets)) > 1:
                inconsistencies.append({
                    "type": "mask_diff",
                    "block_id": block_id,
                    "mask_sets": [list(ms) for ms in mask_sets],
                    "runs": [r['run_id'] for r in runs]
                })
        
        return inconsistencies
    
    def _check_order_consistency(self, runs):
        """Check if block processing order is consistent."""
        inconsistencies = []
        
        first_order = runs[0].get('block_order', [])
        
        for i, run in enumerate(runs[1:], 1):
            order = run.get('block_order', [])
            if order != first_order:
                inconsistencies.append({
                    "type": "order_diff",
                    "run_1": first_order,
                    "run_{}".format(i+1): order,
                    "difference": "Order differs"
                })
        
        return inconsistencies
    
    def _check_coverage_consistency(self, runs):
        """Check if coverage is consistent."""
        inconsistencies = []
        
        coverages = [r.get('coverage', 0) for r in runs]
        
        if len(set(coverages)) > 1:
            inconsistencies.append({
                "type": "coverage_diff",
                "coverages": coverages,
                "runs": [r['run_id'] for r in runs]
            })
        
        return inconsistencies
    
    def generate_report(self, output_path):
        """Generate consistency test report."""
        report = {
            "test_date": datetime.now().isoformat(),
            "config": {
                "backend": self.config.backend,
                "source_lang": self.config.source_lang,
                "target_lang": self.config.target_lang,
                "temperature": getattr(self.config, 'temperature', 0.0),
                "enable_reranking": self.config.enable_reranking,
                "enable_context": self.config.enable_context,
                "num_candidates": self.config.num_candidates
            },
            "results": self.results,
            "analysis": self._analyze_consistency() if self.results else {}
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Consistency report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Test translation consistency")
    parser.add_argument("pdf_path", type=Path, help="Path to PDF to test")
    parser.add_argument("--num-runs", type=int, default=3, help="Number of runs (default: 3)")
    parser.add_argument("--output", type=Path, default=Path("consistency_report.json"), help="Output report path")
    parser.add_argument("--backend", type=str, default="deepseek", help="Translation backend")
    parser.add_argument("--source-lang", type=str, default="en", help="Source language")
    parser.add_argument("--target-lang", type=str, default="fr", help="Target language")
    parser.add_argument("--temperature", type=float, help="Override temperature (default: 0.0)")
    
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
    
    if args.temperature is not None:
        config.temperature = args.temperature
    
    # Clear debug log
    if DEBUG_LOG_PATH.exists():
        DEBUG_LOG_PATH.unlink()
    
    # Run test
    tester = ConsistencyTester(config, num_runs=args.num_runs)
    analysis = tester.test_pdf(args.pdf_path)
    
    # Generate report
    tester.generate_report(args.output)
    
    # Print summary
    print("\n" + "="*60)
    print("CONSISTENCY TEST RESULTS")
    print("="*60)
    print(f"Consistent: {analysis.get('consistent', False)}")
    print(f"Successful runs: {analysis.get('successful_runs', 0)}/{analysis.get('total_runs', 0)}")
    
    if not analysis.get('consistent', True):
        print("\nINCONSISTENCIES FOUND:")
        for inc in analysis.get('inconsistencies', []):
            print(f"  - {inc.get('type', 'unknown')}: {inc.get('block_id', 'N/A')}")
    
    print("="*60)
    print(f"\nFull report saved to: {args.output}")
    print(f"Debug logs saved to: {DEBUG_LOG_PATH}")


if __name__ == "__main__":
    main()

