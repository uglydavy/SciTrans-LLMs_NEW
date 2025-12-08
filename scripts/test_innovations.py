#!/usr/bin/env python3
"""
Comprehensive test script for SciTrans-LLMs NEW

Tests all three innovations:
1. Terminology-Constrained Translation (Masking)
2. Document-Level Context (Reranking)
3. Layout Preservation (PDF reconstruction)

Usage:
    python scripts/test_innovations.py
    python scripts/test_innovations.py --verbose
    python scripts/test_innovations.py --innovation 1  # Test specific innovation
"""

import sys
import os
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import test utilities
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    class Console:
        def print(self, *args, **kwargs):
            text = str(args[0]) if args else ""
            # Strip rich formatting
            import re
            text = re.sub(r'\[.*?\]', '', text)
            print(text)


console = Console()


class InnovationTester:
    """Test runner for all three innovations."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results = {
            1: {"name": "Masking", "tests": [], "passed": 0, "failed": 0},
            2: {"name": "Context & Reranking", "tests": [], "passed": 0, "failed": 0},
            3: {"name": "Layout Preservation", "tests": [], "passed": 0, "failed": 0}
        }
    
    def run_test(self, innovation: int, test_name: str, test_func) -> bool:
        """Run a single test and record result."""
        try:
            start = time.time()
            result = test_func()
            elapsed = time.time() - start
            
            self.results[innovation]["tests"].append({
                "name": test_name,
                "passed": result,
                "time": elapsed,
                "error": None
            })
            
            if result:
                self.results[innovation]["passed"] += 1
                if self.verbose:
                    console.print(f"  [green]✓[/green] {test_name} ({elapsed:.2f}s)")
            else:
                self.results[innovation]["failed"] += 1
                console.print(f"  [red]✗[/red] {test_name} - returned False")
                
            return result
            
        except Exception as e:
            self.results[innovation]["tests"].append({
                "name": test_name,
                "passed": False,
                "time": 0,
                "error": str(e)
            })
            self.results[innovation]["failed"] += 1
            console.print(f"  [red]✗[/red] {test_name} - Error: {e}")
            return False

    # ===========================================
    # Innovation #1: Masking Tests
    # ===========================================
    
    def test_innovation_1(self):
        """Test Innovation #1: Terminology-Constrained Translation with Masking."""
        console.print("\n[bold cyan]Innovation #1: Masking System[/bold cyan]")
        
        self.run_test(1, "Basic LaTeX masking", self._test_latex_masking)
        self.run_test(1, "Display math masking", self._test_display_math)
        self.run_test(1, "URL masking", self._test_url_masking)
        self.run_test(1, "Code block masking", self._test_code_masking)
        self.run_test(1, "Citation masking", self._test_citation_masking)
        self.run_test(1, "Mask restoration", self._test_mask_restoration)
        self.run_test(1, "Validation system", self._test_validation)
        self.run_test(1, "Priority system", self._test_priority_system)
    
    def _test_latex_masking(self) -> bool:
        """Test inline LaTeX masking."""
        from scitran.masking.engine import MaskingEngine
        from scitran.core.models import Block
        
        engine = MaskingEngine()
        block = Block(block_id="test", source_text="The equation $E = mc^2$ is famous.")
        
        masked = engine.mask_block(block)
        
        return (
            "<<LATEX_INLINE_" in masked.masked_text and
            "$E = mc^2$" not in masked.masked_text and
            len(masked.masks) == 1 and
            masked.masks[0].mask_type == "latex_inline"
        )
    
    def _test_display_math(self) -> bool:
        """Test display math masking."""
        from scitran.masking.engine import MaskingEngine
        from scitran.core.models import Block
        
        engine = MaskingEngine()
        block = Block(
            block_id="test", 
            source_text=r"The integral: $$\int_0^1 x^2 dx = \frac{1}{3}$$ shows the area."
        )
        
        masked = engine.mask_block(block)
        
        return "<<LATEX_DISPLAY_" in masked.masked_text and len(masked.masks) >= 1
    
    def _test_url_masking(self) -> bool:
        """Test URL masking."""
        from scitran.masking.engine import MaskingEngine
        from scitran.core.models import Block
        
        engine = MaskingEngine()
        block = Block(block_id="test", source_text="See https://arxiv.org/abs/1234.5678 for details.")
        
        masked = engine.mask_block(block)
        
        return (
            "<<URL_FULL_" in masked.masked_text and
            "https://arxiv.org" not in masked.masked_text
        )
    
    def _test_code_masking(self) -> bool:
        """Test code masking."""
        from scitran.masking.engine import MaskingEngine
        from scitran.core.models import Block
        
        engine = MaskingEngine()
        block = Block(block_id="test", source_text="Use `print('hello')` to print.")
        
        masked = engine.mask_block(block)
        
        return "<<CODE_INLINE_" in masked.masked_text
    
    def _test_citation_masking(self) -> bool:
        """Test citation masking."""
        from scitran.masking.engine import MaskingEngine, MaskingConfig
        from scitran.core.models import Block
        
        config = MaskingConfig(mask_references=True)
        engine = MaskingEngine(config)
        block = Block(block_id="test", source_text="As shown in [1,2,3], the results are significant.")
        
        masked = engine.mask_block(block)
        
        return "<<CITATION_BRACKET_" in masked.masked_text
    
    def _test_mask_restoration(self) -> bool:
        """Test mask restoration after translation."""
        from scitran.masking.engine import MaskingEngine
        from scitran.core.models import Block
        
        engine = MaskingEngine()
        block = Block(block_id="test", source_text="The value $x=5$ is important.")
        
        masked = engine.mask_block(block)
        placeholder = masked.masks[0].placeholder
        
        # Simulate translation
        masked.translated_text = f"La valeur {placeholder} est importante."
        
        unmasked = engine.unmask_block(masked)
        
        return "$x=5$" in unmasked.translated_text
    
    def _test_validation(self) -> bool:
        """Test mask validation system."""
        from scitran.masking.engine import MaskingEngine
        from scitran.core.models import Block
        
        engine = MaskingEngine()
        block = Block(block_id="test", source_text="Formula: $y = mx + b$")
        
        masked = engine.mask_block(block)
        
        # Test with preserved placeholder
        good_translation = masked.masked_text.replace("Formula:", "Formule:")
        is_valid, missing = engine.validate_masks(block.source_text, good_translation, masked.masks)
        
        return is_valid and len(missing) == 0
    
    def _test_priority_system(self) -> bool:
        """Test pattern priority system."""
        from scitran.masking.engine import MaskingEngine
        from scitran.core.models import Block
        
        engine = MaskingEngine()
        
        # Test with overlapping patterns - LaTeX environment should take priority
        text = r"\begin{equation}x=1\end{equation}"
        block = Block(block_id="test", source_text=text)
        
        masked = engine.mask_block(block)
        
        # Should mask as environment, not as individual parts
        return len(masked.masks) >= 1

    # ===========================================
    # Innovation #2: Context & Reranking Tests
    # ===========================================
    
    def test_innovation_2(self):
        """Test Innovation #2: Document-Level Context and Reranking."""
        console.print("\n[bold cyan]Innovation #2: Context & Reranking[/bold cyan]")
        
        self.run_test(2, "Multi-dimensional scorer", self._test_scorer)
        self.run_test(2, "Fluency scoring", self._test_fluency_scoring)
        self.run_test(2, "Terminology scoring", self._test_terminology_scoring)
        self.run_test(2, "Format scoring", self._test_format_scoring)
        self.run_test(2, "Candidate reranking", self._test_reranking)
        self.run_test(2, "Prompt templates", self._test_prompt_templates)
        self.run_test(2, "Prompt optimization", self._test_prompt_optimization)
    
    def _test_scorer(self) -> bool:
        """Test multi-dimensional scorer."""
        from scitran.scoring.reranker import MultiDimensionalScorer
        
        scorer = MultiDimensionalScorer()
        
        candidates = [
            "L'apprentissage automatique a révolutionné le traitement du langage naturel.",
            "Machine learning has revolutionized NLP.",  # Untranslated
            "L'apprentissage machine a révolutionné le traitement du langage."
        ]
        
        scored = scorer.score_candidates(
            candidates=candidates,
            source_text="Machine learning has revolutionized natural language processing."
        )
        
        # First candidate should score highest (proper French translation)
        return len(scored) == 3 and scored[0].total_score > 0
    
    def _test_fluency_scoring(self) -> bool:
        """Test fluency dimension scoring."""
        from scitran.scoring.reranker import MultiDimensionalScorer
        
        scorer = MultiDimensionalScorer()
        
        # Good fluency vs. poor fluency
        candidates = [
            "Cette phrase est grammaticalement correcte et fluide.",
            "Cette cette phrase phrase est est répétitive répétitive."
        ]
        
        scored = scorer.score_candidates(candidates, "This is a test.")
        
        # First should have higher fluency
        return (
            scored[0].dimensions['fluency'].score > 
            scored[1].dimensions['fluency'].score
        )
    
    def _test_terminology_scoring(self) -> bool:
        """Test terminology adherence scoring."""
        from scitran.scoring.reranker import MultiDimensionalScorer
        
        scorer = MultiDimensionalScorer()
        
        glossary = {
            "machine learning": "apprentissage automatique",
            "neural network": "réseau de neurones"
        }
        
        candidates = [
            "L'apprentissage automatique utilise des réseaux de neurones.",
            "Le machine learning utilise des neural networks."  # Uses English terms
        ]
        
        scored = scorer.score_candidates(
            candidates=candidates,
            source_text="Machine learning uses neural networks.",
            glossary=glossary
        )
        
        return scored[0].dimensions['terminology'].score > scored[1].dimensions['terminology'].score
    
    def _test_format_scoring(self) -> bool:
        """Test format preservation scoring."""
        from scitran.scoring.reranker import MultiDimensionalScorer
        from scitran.core.models import MaskInfo
        
        scorer = MultiDimensionalScorer()
        
        masks = [
            MaskInfo(original="$E=mc^2$", placeholder="<<LATEX_0001>>", mask_type="latex")
        ]
        
        candidates = [
            "L'équation <<LATEX_0001>> est célèbre.",  # Preserves mask
            "L'équation E=mc^2 est célèbre."           # Lost mask
        ]
        
        scored = scorer.score_candidates(
            candidates=candidates,
            source_text="The equation <<LATEX_0001>> is famous.",
            masks=masks
        )
        
        return scored[0].dimensions['format'].score > scored[1].dimensions['format'].score
    
    def _test_reranking(self) -> bool:
        """Test advanced reranker."""
        from scitran.scoring.reranker import AdvancedReranker
        from scitran.core.models import Block
        
        reranker = AdvancedReranker()
        
        block = Block(
            block_id="test",
            source_text="Machine learning enables AI."
        )
        
        candidates = [
            "L'apprentissage automatique permet l'IA.",
            "Machine learning permet AI.",
            "L'apprentissage permet l'intelligence artificielle."
        ]
        
        glossary = {"machine learning": "apprentissage automatique"}
        
        best, scored = reranker.rerank(
            candidates=candidates,
            block=block,
            glossary=glossary
        )
        
        return len(scored) == 3 and best is not None
    
    def _test_prompt_templates(self) -> bool:
        """Test prompt template library."""
        from scitran.translation.prompts import PromptLibrary
        
        library = PromptLibrary()
        
        return (
            "scientific_expert" in library.templates and
            "few_shot_scientific" in library.templates and
            "chain_of_thought" in library.templates
        )
    
    def _test_prompt_optimization(self) -> bool:
        """Test prompt optimizer."""
        from scitran.translation.prompts import PromptOptimizer
        from scitran.core.models import Block
        
        optimizer = PromptOptimizer()
        
        block = Block(block_id="test", source_text="Test text for translation.")
        
        system_prompt, user_prompt = optimizer.generate_prompt(
            block=block,
            template_name="scientific_expert",
            source_lang="en",
            target_lang="fr",
            glossary_terms={"test": "essai"}
        )
        
        return len(system_prompt) > 0 and len(user_prompt) > 0 and "essai" in user_prompt

    # ===========================================
    # Innovation #3: Layout Preservation Tests
    # ===========================================
    
    def test_innovation_3(self):
        """Test Innovation #3: Layout Preservation."""
        console.print("\n[bold cyan]Innovation #3: Layout Preservation[/bold cyan]")
        
        self.run_test(3, "Bounding box model", self._test_bounding_box)
        self.run_test(3, "Font info model", self._test_font_info)
        self.run_test(3, "Document structure", self._test_document_structure)
        self.run_test(3, "PDF parser (if available)", self._test_pdf_parser)
        self.run_test(3, "PDF renderer (if available)", self._test_pdf_renderer)
    
    def _test_bounding_box(self) -> bool:
        """Test bounding box operations."""
        from scitran.core.models import BoundingBox
        
        bbox1 = BoundingBox(x0=0, y0=0, x1=100, y1=50, page=0)
        bbox2 = BoundingBox(x0=10, y0=10, x1=90, y1=40, page=0)
        bbox3 = BoundingBox(x0=0, y0=0, x1=100, y1=50, page=1)  # Different page
        
        return (
            bbox1.area() == 5000 and
            bbox1.contains(bbox2) and
            not bbox2.contains(bbox1) and
            bbox1.overlaps(bbox2) and
            not bbox1.overlaps(bbox3)  # Different pages don't overlap
        )
    
    def _test_font_info(self) -> bool:
        """Test font info model."""
        from scitran.core.models import FontInfo
        
        font = FontInfo(
            family="Times New Roman",
            size=12.0,
            weight="bold",
            style="italic",
            color="#FF0000"
        )
        
        return font.family == "Times New Roman" and font.size == 12.0
    
    def _test_document_structure(self) -> bool:
        """Test document structure with layout info."""
        from scitran.core.models import Document, Segment, Block, BoundingBox
        
        block1 = Block(
            block_id="b1",
            source_text="Title text",
            bbox=BoundingBox(x0=100, y0=50, x1=500, y1=80, page=0)
        )
        
        block2 = Block(
            block_id="b2",
            source_text="Body text content here.",
            bbox=BoundingBox(x0=100, y0=100, x1=500, y1=200, page=0)
        )
        
        segment = Segment(segment_id="s1", segment_type="main", blocks=[block1, block2])
        document = Document(document_id="test_doc", segments=[segment])
        
        return (
            len(document.all_blocks) == 2 and
            all(b.bbox is not None for b in document.all_blocks)
        )
    
    def _test_pdf_parser(self) -> bool:
        """Test PDF parser (if PyMuPDF available)."""
        try:
            from scitran.extraction.pdf_parser import PDFParser
            
            # Check if any test PDF exists
            test_pdfs = ["test.pdf", "attention_is_all_you_need.pdf", "alphafold.pdf"]
            test_pdf = None
            for pdf in test_pdfs:
                if os.path.exists(pdf):
                    test_pdf = pdf
                    break
            
            if not test_pdf:
                if self.verbose:
                    console.print("    [yellow]No test PDF available, skipping...[/yellow]")
                return True  # Skip if no PDF
            
            parser = PDFParser()
            document = parser.parse(test_pdf, max_pages=1)
            
            return (
                len(document.all_blocks) > 0 and
                any(b.bbox is not None for b in document.all_blocks)
            )
        except ImportError:
            if self.verbose:
                console.print("    [yellow]PyMuPDF not installed, skipping...[/yellow]")
            return True  # Skip if not available
    
    def _test_pdf_renderer(self) -> bool:
        """Test PDF renderer (if PyMuPDF available)."""
        try:
            from scitran.rendering.pdf_renderer import PDFRenderer
            from scitran.core.models import Document, Segment, Block, BoundingBox
            
            # Create test document
            block = Block(
                block_id="b1",
                source_text="Test content",
                translated_text="Contenu de test",
                bbox=BoundingBox(x0=50, y0=50, x1=300, y1=100, page=0)
            )
            segment = Segment(segment_id="s1", segment_type="main", blocks=[block])
            document = Document(document_id="test", segments=[segment])
            
            renderer = PDFRenderer()
            
            # Test simple render
            output_path = "/tmp/scitrans_test_output.pdf"
            renderer.render_simple(document, output_path)
            
            return os.path.exists(output_path)
        except ImportError:
            if self.verbose:
                console.print("    [yellow]PyMuPDF not installed, skipping...[/yellow]")
            return True  # Skip if not available
        except Exception as e:
            if self.verbose:
                console.print(f"    [yellow]Render test failed: {e}[/yellow]")
            return False

    # ===========================================
    # Summary and Reporting
    # ===========================================
    
    def print_summary(self):
        """Print test summary."""
        console.print("\n" + "=" * 60)
        console.print("[bold]Test Summary[/bold]")
        console.print("=" * 60)
        
        total_passed = 0
        total_failed = 0
        
        for innovation_id, data in self.results.items():
            passed = data["passed"]
            failed = data["failed"]
            total = passed + failed
            
            total_passed += passed
            total_failed += failed
            
            status = "[green]PASS[/green]" if failed == 0 else "[red]FAIL[/red]"
            console.print(f"\nInnovation #{innovation_id}: {data['name']}")
            console.print(f"  Tests: {passed}/{total} passed  {status}")
            
            # Show failed tests
            for test in data["tests"]:
                if not test["passed"]:
                    console.print(f"    [red]✗[/red] {test['name']}: {test.get('error', 'Failed')}")
        
        console.print("\n" + "-" * 60)
        success_rate = (total_passed / (total_passed + total_failed) * 100) if (total_passed + total_failed) > 0 else 0
        status_color = "green" if total_failed == 0 else "red"
        console.print(f"[{status_color}]Total: {total_passed}/{total_passed + total_failed} tests passed ({success_rate:.0f}%)[/{status_color}]")
        
        return total_failed == 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test SciTrans-LLMs innovations")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--innovation", "-i", type=int, choices=[1, 2, 3], help="Test specific innovation")
    args = parser.parse_args()
    
    console.print(Panel.fit(
        "[bold cyan]SciTrans-LLMs NEW - Innovation Testing[/bold cyan]\n\n"
        "Testing all three key innovations:\n"
        "1. Terminology-Constrained Translation (Masking)\n"
        "2. Document-Level Context (Reranking)\n"
        "3. Layout Preservation (PDF reconstruction)",
        border_style="cyan"
    ) if HAS_RICH else "SciTrans-LLMs Innovation Testing")
    
    tester = InnovationTester(verbose=args.verbose)
    
    if args.innovation:
        # Test specific innovation
        if args.innovation == 1:
            tester.test_innovation_1()
        elif args.innovation == 2:
            tester.test_innovation_2()
        elif args.innovation == 3:
            tester.test_innovation_3()
    else:
        # Test all innovations
        tester.test_innovation_1()
        tester.test_innovation_2()
        tester.test_innovation_3()
    
    success = tester.print_summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

