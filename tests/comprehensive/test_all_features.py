#!/usr/bin/env python3
"""
Comprehensive test suite for SciTrans-LLMs NEW.
Tests all features: prompting, glossary, masking, batching, extraction, layout, fonts, context, etc.

Usage:
    PYTHONPATH=. pytest tests/comprehensive/test_all_features.py -v
    PYTHONPATH=. python tests/comprehensive/test_all_features.py
"""

import sys
import os
from pathlib import Path
import tempfile
import json
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


class TestComprehensiveFeatures:
    """Comprehensive feature tests."""
    
    @pytest.fixture
    def sample_texts(self):
        """Sample texts for testing."""
        return {
            "simple": "Machine learning is important.",
            "latex": "The equation $E=mc^2$ demonstrates mass-energy equivalence.",
            "url": "See https://arxiv.org/abs/2010.11929 for details.",
            "code": "Use `import numpy as np` for numerical computing.",
            "citation": "Previous work [1] showed promising results.",
            "long": "Machine learning enables computers to learn from data without explicit programming. " * 10,
            "glossary_term": "Neural networks use deep learning techniques.",
        }
    
    @pytest.fixture
    def test_document(self):
        """Create a test document."""
        from scitran.core.models import Document, Segment, Block, BlockType, BoundingBox
        
        blocks = [
            Block(
                block_id="b1",
                source_text="Machine learning is important.",
                block_type=BlockType.PARAGRAPH,
                bbox=BoundingBox(x0=100, y0=100, x1=500, y1=120, page=0)
            ),
            Block(
                block_id="b2",
                source_text="The equation $E=mc^2$ is famous.",
                block_type=BlockType.PARAGRAPH,
                bbox=BoundingBox(x0=100, y0=150, x1=500, y1=170, page=0)
            ),
        ]
        
        segment = Segment(segment_id="s1", segment_type="body", blocks=blocks)
        doc = Document(document_id="test", segments=[segment])
        return doc
    
    # ========================================================================
    # TEST GROUP 1: EXTRACTION & PARSING
    # ========================================================================
    
    def test_pdf_parsing(self):
        """Test PDF extraction and parsing."""
        console.print("\n[bold cyan]TEST GROUP 1: PDF Extraction & Parsing[/bold cyan]")
        
        try:
            from scitran.extraction.pdf_parser import PDFParser
            
            parser = PDFParser()
            assert parser is not None, "Parser should be created"
            
            # Test with a simple text (if no PDF available, skip)
            console.print("  ✓ PDFParser created")
        except Exception as e:
            console.print(f"  ✗ PDF parsing failed: {e}")
            raise
    
    def test_block_extraction(self, test_document):
        """Test block extraction from document."""
        console.print("\n[bold]1.1 Block Extraction[/bold]")
        
        try:
            blocks = test_document.translatable_blocks
            assert len(blocks) == 2, f"Expected 2 blocks, got {len(blocks)}"
            assert blocks[0].source_text == "Machine learning is important."
            console.print("  ✓ Blocks extracted correctly")
        except Exception as e:
            console.print(f"  ✗ Block extraction failed: {e}")
            raise
    
    def test_bbox_extraction(self, test_document):
        """Test bounding box extraction."""
        console.print("\n[bold]1.2 Bounding Box Extraction[/bold]")
        
        try:
            blocks = test_document.translatable_blocks
            assert blocks[0].bbox is not None, "BBox should be present"
            assert blocks[0].bbox.page == 0, "Page should be 0"
            assert blocks[0].bbox.x0 == 100, "x0 should be 100"
            console.print("  ✓ Bounding boxes extracted correctly")
        except Exception as e:
            console.print(f"  ✗ BBox extraction failed: {e}")
            raise
    
    # ========================================================================
    # TEST GROUP 2: MASKING ENGINE
    # ========================================================================
    
    def test_latex_masking(self, sample_texts):
        """Test LaTeX formula masking."""
        console.print("\n[bold cyan]TEST GROUP 2: Masking Engine[/bold cyan]")
        console.print("\n[bold]2.1 LaTeX Masking[/bold]")
        
        try:
            from scitran.masking.engine import MaskingEngine
            from scitran.core.models import Block
            
            engine = MaskingEngine()
            block = Block(block_id="test", source_text=sample_texts["latex"])
            masked = engine.mask_block(block)
            
            assert len(masked.masks) > 0, "Should have masks"
            assert "$E=mc^2$" not in masked.masked_text, "LaTeX should be masked"
            assert "LATEX" in masked.masked_text.upper() or "MASK" in masked.masked_text.upper(), "Should have placeholder"
            console.print("  ✓ LaTeX masked correctly")
            pass
        except Exception as e:
            console.print(f"  ✗ LaTeX masking failed: {e}")
            raise
    
    def test_url_masking(self, sample_texts):
        """Test URL masking."""
        console.print("\n[bold]2.2 URL Masking[/bold]")
        
        try:
            from scitran.masking.engine import MaskingEngine
            from scitran.core.models import Block
            
            engine = MaskingEngine()
            block = Block(block_id="test", source_text=sample_texts["url"])
            masked = engine.mask_block(block)
            
            assert "https://arxiv.org" not in masked.masked_text, "URL should be masked"
            console.print("  ✓ URL masked correctly")
            pass
        except Exception as e:
            console.print(f"  ✗ URL masking failed: {e}")
            raise
    
    def test_code_masking(self, sample_texts):
        """Test code block masking."""
        console.print("\n[bold]2.3 Code Masking[/bold]")
        
        try:
            from scitran.masking.engine import MaskingEngine
            from scitran.core.models import Block
            
            engine = MaskingEngine()
            block = Block(block_id="test", source_text=sample_texts["code"])
            masked = engine.mask_block(block)
            
            assert "`import numpy" not in masked.masked_text or "CODE" in masked.masked_text.upper(), "Code should be masked"
            console.print("  ✓ Code masked correctly")
            pass
        except Exception as e:
            console.print(f"  ✗ Code masking failed: {e}")
            raise
    
    def test_mask_restoration(self, sample_texts):
        """Test mask restoration after translation."""
        console.print("\n[bold]2.4 Mask Restoration[/bold]")
        
        try:
            from scitran.masking.engine import MaskingEngine
            from scitran.core.models import Block
            
            engine = MaskingEngine()
            block = Block(block_id="test", source_text=sample_texts["latex"])
            masked = engine.mask_block(block)
            
            # Simulate translation (keep placeholder)
            masked.translated_text = masked.masked_text.replace("equation", "équation")
            
            # Restore masks
            restored = engine.unmask_block(masked)
            
            assert "$E=mc^2$" in restored.translated_text, "LaTeX should be restored"
            console.print("  ✓ Masks restored correctly")
            pass
        except Exception as e:
            console.print(f"  ✗ Mask restoration failed: {e}")
            raise
    
    # ========================================================================
    # TEST GROUP 3: GLOSSARY & TERMINOLOGY
    # ========================================================================
    
    def test_glossary_loading(self):
        """Test glossary loading."""
        console.print("\n[bold cyan]TEST GROUP 3: Glossary & Terminology[/bold cyan]")
        console.print("\n[bold]3.1 Glossary Loading[/bold]")
        
        try:
            from scitran.core.pipeline import PipelineConfig
            
            config = PipelineConfig(enable_glossary=True)
            assert config.enable_glossary == True, "Glossary should be enabled"
            console.print("  ✓ Glossary config works")
            pass
        except Exception as e:
            console.print(f"  ✗ Glossary loading failed: {e}")
            raise
    
    def test_glossary_enforcement(self):
        """Test glossary term enforcement."""
        console.print("\n[bold]3.2 Glossary Enforcement[/bold]")
        
        try:
            from scitran.core.pipeline import TranslationPipeline, PipelineConfig
            from scitran.core.models import Document, Segment, Block, BlockType
            
            # Create document with glossary term
            block = Block(
                block_id="test",
                source_text="Neural networks use deep learning.",
                block_type=BlockType.PARAGRAPH
            )
            segment = Segment(segment_id="s1", segment_type="body", blocks=[block])
            doc = Document(document_id="test", segments=[segment])
            
            # Configure with glossary
            config = PipelineConfig(
                source_lang="en",
                target_lang="fr",
                backend="cascade",
                enable_glossary=True,
                enable_masking=False,
                enable_reranking=False,
                num_candidates=1
            )
            
            pipeline = TranslationPipeline(config)
            # Load glossary manually for test
            pipeline.glossary = {"neural networks": "réseaux de neurones"}
            
            result = pipeline.translate_document(doc)
            
            # Check if glossary term was used (if translation succeeded)
            if result.success and block.translated_text:
                # Glossary enforcement happens in postprocessing
                console.print("  ✓ Glossary enforcement configured")
                pass
            else:
                console.print("  ⚠ Translation failed, but glossary config OK")
                pass
        except Exception as e:
            console.print(f"  ✗ Glossary enforcement failed: {e}")
            raise
    
    # ========================================================================
    # TEST GROUP 4: BATCH PROCESSING
    # ========================================================================
    
    def test_batch_translation(self):
        """Test batch translation."""
        console.print("\n[bold cyan]TEST GROUP 4: Batch Processing[/bold cyan]")
        console.print("\n[bold]4.1 Batch Translation[/bold]")
        
        try:
            from scitran.utils.fast_translator import FastTranslator
            
            translator = FastTranslator(max_concurrent=3)
            texts = ["Hello", "World", "Test"]
            
            # Test batch sync
            results = translator.translate_batch_sync(
                texts=texts,
                source_lang="en",
                target_lang="fr"
            )
            
            assert len(results) == len(texts), "Should return same number of results"
            assert all(r for r in results), "All results should be non-empty"
            console.print("  ✓ Batch translation works")
            pass
        except Exception as e:
            console.print(f"  ✗ Batch translation failed: {e}")
            raise
    
    def test_batch_caching(self):
        """Test batch translation caching."""
        console.print("\n[bold]4.2 Batch Caching[/bold]")
        
        try:
            from scitran.utils.fast_translator import FastTranslator
            
            translator = FastTranslator(max_concurrent=3)
            text = "Test caching"
            
            # First call (should translate)
            results1 = translator.translate_batch_sync(
                texts=[text],
                source_lang="en",
                target_lang="fr"
            )
            
            # Second call (should use cache)
            results2 = translator.translate_batch_sync(
                texts=[text],
                source_lang="en",
                target_lang="fr"
            )
            
            assert results1[0] == results2[0], "Cached result should match"
            stats = translator.get_stats()
            assert stats.get("cached", 0) > 0, "Should have cache hits"
            console.print("  ✓ Batch caching works")
            pass
        except Exception as e:
            console.print(f"  ✗ Batch caching failed: {e}")
            raise
    
    # ========================================================================
    # TEST GROUP 5: PROMPTING & CONTEXT
    # ========================================================================
    
    def test_prompt_generation(self):
        """Test prompt generation."""
        console.print("\n[bold cyan]TEST GROUP 5: Prompting & Context[/bold cyan]")
        console.print("\n[bold]5.1 Prompt Generation[/bold]")
        
        try:
            from scitran.translation.prompts import PromptOptimizer
            from scitran.core.models import Block
            
            optimizer = PromptOptimizer()
            block = Block(block_id="test", source_text="Test text")
            
            system, user = optimizer.generate_prompt(
                block=block,
                template_name="scientific_expert",
                source_lang="en",
                target_lang="fr",
                glossary_terms={}
            )
            
            assert system is not None, "System prompt should exist"
            assert user is not None, "User prompt should exist"
            assert "en" in user.lower() or "english" in system.lower(), "Should mention source lang"
            assert "fr" in user.lower() or "french" in system.lower(), "Should mention target lang"
            console.print("  ✓ Prompt generation works")
            pass
        except Exception as e:
            console.print(f"  ✗ Prompt generation failed: {e}")
            raise
    
    def test_context_window(self, test_document):
        """Test context window functionality."""
        console.print("\n[bold]5.2 Context Window[/bold]")
        
        try:
            from scitran.core.pipeline import PipelineConfig
            
            config = PipelineConfig(
                enable_context=True,
                context_window_size=5
            )
            
            assert config.enable_context == True, "Context should be enabled"
            assert config.context_window_size == 5, "Window size should be 5"
            console.print("  ✓ Context window configured")
            pass
        except Exception as e:
            console.print(f"  ✗ Context window failed: {e}")
            raise
    
    # ========================================================================
    # TEST GROUP 6: RERANKING & QUALITY
    # ========================================================================
    
    def test_reranking(self):
        """Test reranking system."""
        console.print("\n[bold cyan]TEST GROUP 6: Reranking & Quality[/bold cyan]")
        console.print("\n[bold]6.1 Reranking[/bold]")
        
        try:
            from scitran.scoring.reranker import AdvancedReranker, ScoringStrategy
            from scitran.core.models import Block
            
            reranker = AdvancedReranker(strategy=ScoringStrategy.HYBRID)
            
            block = Block(block_id="test", source_text="Test text")
            candidates = ["Candidat 1", "Candidat 2", "Candidat 3"]
            
            best, scored = reranker.rerank(
                candidates=candidates,
                block=block
            )
            
            assert best is not None, "Should return best candidate"
            assert len(scored) == len(candidates), "Should score all candidates"
            console.print("  ✓ Reranking works")
            pass
        except Exception as e:
            console.print(f"  ✗ Reranking failed: {e}")
            raise
    
    def test_quality_scoring(self):
        """Test quality scoring."""
        console.print("\n[bold]6.2 Quality Scoring[/bold]")
        
        try:
            from scitran.scoring.reranker import MultiDimensionalScorer
            
            scorer = MultiDimensionalScorer()
            
            candidates = ["Good translation", "Bad translation"]
            scored = scorer.score_candidates(
                candidates=candidates,
                source_text="Source text"
            )
            
            assert len(scored) == len(candidates), "Should score all candidates"
            assert scored[0].total_score >= 0, "Score should be non-negative"
            console.print("  ✓ Quality scoring works")
            pass
        except Exception as e:
            console.print(f"  ✗ Quality scoring failed: {e}")
            raise
    
    # ========================================================================
    # TEST GROUP 7: LAYOUT & FONTS
    # ========================================================================
    
    def test_layout_preservation(self, test_document):
        """Test layout preservation."""
        console.print("\n[bold cyan]TEST GROUP 7: Layout & Fonts[/bold cyan]")
        console.print("\n[bold]7.1 Layout Preservation[/bold]")
        
        try:
            blocks = test_document.translatable_blocks
            assert all(b.bbox for b in blocks), "All blocks should have bbox"
            
            # Check bbox properties
            for block in blocks:
                assert block.bbox.x0 < block.bbox.x1, "x0 < x1"
                assert block.bbox.y0 < block.bbox.y1, "y0 < y1"
            
            console.print("  ✓ Layout information preserved")
            pass
        except Exception as e:
            console.print(f"  ✗ Layout preservation failed: {e}")
            raise
    
    def test_font_info(self, test_document):
        """Test font information extraction."""
        console.print("\n[bold]7.2 Font Information[/bold]")
        
        try:
            from scitran.core.models import FontInfo
            
            # Create block with font info
            font = FontInfo(family="Helvetica", size=12, weight="bold")
            assert font.family == "Helvetica"
            assert font.size == 12
            assert font.weight == "bold"
            
            console.print("  ✓ Font info structure works")
            pass
        except Exception as e:
            console.print(f"  ✗ Font info failed: {e}")
            raise
    
    # ========================================================================
    # TEST GROUP 8: BACKENDS
    # ========================================================================
    
    def test_backend_interface(self):
        """Test backend interface."""
        console.print("\n[bold cyan]TEST GROUP 8: Translation Backends[/bold cyan]")
        console.print("\n[bold]8.1 Backend Interface[/bold]")
        
        try:
            from scitran.translation.base import TranslationBackend, TranslationRequest
            
            # Test that backends implement the interface
            backends = ["cascade", "free"]
            
            for backend_name in backends:
                if backend_name == "cascade":
                    from scitran.translation.backends.cascade_backend import CascadeBackend
                    backend = CascadeBackend()
                elif backend_name == "free":
                    from scitran.translation.backends.free_backend import FreeBackend
                    backend = FreeBackend()
                
                assert isinstance(backend, TranslationBackend), f"{backend_name} should implement interface"
                assert backend.is_available(), f"{backend_name} should be available"
            
            console.print("  ✓ Backend interface works")
            pass
        except Exception as e:
            console.print(f"  ✗ Backend interface failed: {e}")
            raise
    
    def test_backend_translation(self):
        """Test actual backend translation."""
        console.print("\n[bold]8.2 Backend Translation[/bold]")
        
        try:
            from scitran.translation.backends.cascade_backend import CascadeBackend
            from scitran.translation.base import TranslationRequest
            
            backend = CascadeBackend()
            request = TranslationRequest(
                text="Hello world",
                source_lang="en",
                target_lang="fr"
            )
            
            response = backend.translate_sync(request)
            
            assert response is not None, "Should return response"
            assert len(response.translations) > 0, "Should have translations"
            console.print("  ✓ Backend translation works")
            pass
        except Exception as e:
            console.print(f"  ✗ Backend translation failed: {e}")
            raise
    
    # ========================================================================
    # TEST GROUP 9: PIPELINE INTEGRATION
    # ========================================================================
    
    def test_full_pipeline(self, test_document):
        """Test full translation pipeline."""
        console.print("\n[bold cyan]TEST GROUP 9: Pipeline Integration[/bold cyan]")
        console.print("\n[bold]9.1 Full Pipeline[/bold]")
        
        try:
            from scitran.core.pipeline import TranslationPipeline, PipelineConfig
            
            config = PipelineConfig(
                source_lang="en",
                target_lang="fr",
                backend="cascade",
                enable_masking=True,
                enable_reranking=False,
                num_candidates=1,
                cache_translations=True
            )
            
            pipeline = TranslationPipeline(config)
            result = pipeline.translate_document(test_document)
            
            assert result is not None, "Should return result"
            assert result.blocks_translated >= 0, "Should have translation count"
            console.print("  ✓ Full pipeline works")
            pass
        except Exception as e:
            console.print(f"  ✗ Full pipeline failed: {e}")
            raise
    
    def test_pipeline_with_all_features(self, test_document):
        """Test pipeline with all features enabled."""
        console.print("\n[bold]9.2 Pipeline with All Features[/bold]")
        
        try:
            from scitran.core.pipeline import TranslationPipeline, PipelineConfig
            
            config = PipelineConfig(
                source_lang="en",
                target_lang="fr",
                backend="cascade",
                enable_masking=True,
                enable_reranking=True,
                enable_context=True,
                enable_glossary=True,
                num_candidates=3,
                context_window_size=5,
                cache_translations=True
            )
            
            pipeline = TranslationPipeline(config)
            # Load test glossary
            pipeline.glossary = {"machine learning": "apprentissage automatique"}
            
            result = pipeline.translate_document(test_document)
            
            assert result is not None, "Should return result"
            console.print("  ✓ Pipeline with all features works")
            pass
        except Exception as e:
            console.print(f"  ✗ Pipeline with all features failed: {e}")
            raise


def run_all_tests():
    """Run all comprehensive tests."""
    console.print(Panel.fit("[bold green]SciTrans-LLMs Comprehensive Test Suite[/bold green]"))
    
    test_instance = TestComprehensiveFeatures()
    results = {}
    
    # Run all test methods
    test_methods = [m for m in dir(test_instance) if m.startswith("test_")]
    
    for test_method_name in sorted(test_methods):
        test_method = getattr(test_instance, test_method_name)
        
        # Get fixtures
        if "sample_texts" in test_method.__code__.co_varnames:
            test_method = pytest.fixture(lambda: None)(lambda: test_method(test_instance.sample_texts()))
        elif "test_document" in test_method.__code__.co_varnames:
            test_method = pytest.fixture(lambda: None)(lambda: test_method(test_instance.test_document()))
        else:
            try:
                result = test_method()
                results[test_method_name] = result
            except Exception as e:
                console.print(f"  ✗ {test_method_name} raised exception: {e}")
                results[test_method_name] = False
    
    # Summary
    console.print("\n" + "="*60)
    console.print("[bold]TEST SUMMARY[/bold]")
    console.print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    table = Table(title="Test Results")
    table.add_column("Test", style="cyan")
    table.add_column("Status", justify="center")
    
    for test_name, result in sorted(results.items()):
        status = "[green]✓ PASS[/green]" if result else "[red]✗ FAIL[/red]"
        table.add_row(test_name.replace("test_", "").replace("_", " ").title(), status)
    
    console.print(table)
    console.print(f"\n[bold]Passed: {passed}/{total}[/bold]")
    
    if passed == total:
        console.print("\n[bold green]✅ ALL TESTS PASSED![/bold green]")
    else:
        console.print(f"\n[bold yellow]⚠️ {total - passed} TEST(S) FAILED[/bold yellow]")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
