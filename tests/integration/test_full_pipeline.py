"""Integration tests for the full translation pipeline."""

import pytest
import os
import tempfile
from pathlib import Path


class TestMaskingIntegration:
    """Test masking integration across components."""
    
    def test_mask_translate_unmask_flow(self):
        """Test complete mask -> translate -> unmask flow."""
        from scitran.masking.engine import MaskingEngine
        from scitran.core.models import Block
        
        engine = MaskingEngine()
        
        # Scientific text with multiple patterns
        source = (
            "The equation $E = mc^2$ proves mass-energy equivalence. "
            "See https://physics.org for details and reference [1]."
        )
        
        block = Block(block_id="test", source_text=source)
        
        # Mask
        masked = engine.mask_block(block)
        assert len(masked.masks) == 3  # Formula, URL, citation
        assert "$E = mc^2$" not in masked.masked_text
        assert "https://physics.org" not in masked.masked_text
        
        # Simulate translation (swap key words)
        translation = masked.masked_text
        translation = translation.replace("equation", "équation")
        translation = translation.replace("proves", "prouve")
        translation = translation.replace("equivalence", "équivalence")
        translation = translation.replace("See", "Voir")
        translation = translation.replace("for details", "pour détails")
        masked.translated_text = translation
        
        # Unmask
        unmasked = engine.unmask_block(masked)
        
        # Verify originals are restored
        assert "$E = mc^2$" in unmasked.translated_text
        assert "https://physics.org" in unmasked.translated_text
        assert "[1]" in unmasked.translated_text
        
        # Verify translation words are present
        assert "équation" in unmasked.translated_text
        assert "prouve" in unmasked.translated_text
    
    def test_nested_latex_handling(self):
        """Test handling of nested LaTeX constructs."""
        from scitran.masking.engine import MaskingEngine
        from scitran.core.models import Block
        
        engine = MaskingEngine()
        
        source = r"Consider $$\int_0^1 \frac{d}{dx}\left(\sum_{n=0}^\infty x^n\right) dx$$"
        block = Block(block_id="test", source_text=source)
        
        masked = engine.mask_block(block)
        
        # Should mask as single display math
        assert len(masked.masks) >= 1
        assert "LATEX" in masked.masked_text
    
    def test_document_level_masking(self):
        """Test masking at document level."""
        from scitran.masking.engine import MaskingEngine
        from scitran.core.models import Document, Segment, Block
        
        engine = MaskingEngine()
        
        blocks = [
            Block(block_id="b1", source_text="Formula: $x = 1$"),
            Block(block_id="b2", source_text="Code: `print(x)`"),
            Block(block_id="b3", source_text="URL: https://test.com"),
        ]
        
        segment = Segment(segment_id="s1", segment_type="main", blocks=blocks)
        document = Document(document_id="test_doc", segments=[segment])
        
        masked_doc = engine.mask_document(document)
        
        assert masked_doc.stats.get('total_masks', 0) >= 3


class TestScoringIntegration:
    """Test scoring and reranking integration."""
    
    def test_scorer_with_glossary(self):
        """Test scorer with glossary enforcement."""
        from scitran.scoring.reranker import MultiDimensionalScorer
        
        scorer = MultiDimensionalScorer()
        
        glossary = {
            "machine learning": "apprentissage automatique",
            "neural network": "réseau de neurones"
        }
        
        candidates = [
            "L'apprentissage automatique utilise des réseaux de neurones.",  # Correct
            "Le machine learning utilise des neural networks.",              # Wrong (untranslated)
            "L'apprentissage profond utilise des réseaux.",                   # Partial
        ]
        
        scored = scorer.score_candidates(
            candidates=candidates,
            source_text="Machine learning uses neural networks.",
            glossary=glossary
        )
        
        # First candidate should rank highest (correct glossary terms)
        assert scored[0].text == candidates[0]
        assert scored[0].dimensions['terminology'].score > scored[1].dimensions['terminology'].score
    
    def test_reranker_with_masks(self):
        """Test reranker enforces mask preservation."""
        from scitran.scoring.reranker import AdvancedReranker
        from scitran.core.models import Block, MaskInfo
        
        reranker = AdvancedReranker()
        
        masks = [
            MaskInfo(original="$x=1$", placeholder="<<LATEX_0001>>", mask_type="latex")
        ]
        
        block = Block(
            block_id="test",
            source_text="The value <<LATEX_0001>> is important.",
            masks=masks
        )
        
        candidates = [
            "La valeur <<LATEX_0001>> est importante.",  # Preserves mask
            "La valeur x=1 est importante.",              # Lost mask
            "La valeur est importante.",                  # Missing value
        ]
        
        best, scored = reranker.rerank(
            candidates=candidates,
            block=block,
            require_all_masks=True
        )
        
        # First candidate should be selected (preserves mask)
        assert best == candidates[0]


class TestPromptIntegration:
    """Test prompt generation and optimization."""
    
    def test_prompt_generation(self):
        """Test prompt generation with context."""
        from scitran.translation.prompts import PromptOptimizer
        from scitran.core.models import Block
        
        optimizer = PromptOptimizer()
        
        block = Block(
            block_id="test",
            source_text="Neural networks learn patterns from data.",
            masked_text="Neural networks learn patterns from data."
        )
        
        context = [
            ("Machine learning is a subset of AI.", "L'apprentissage automatique est un sous-ensemble de l'IA."),
            ("Deep learning uses multiple layers.", "L'apprentissage profond utilise plusieurs couches.")
        ]
        
        glossary = {"neural networks": "réseaux de neurones"}
        
        system, user = optimizer.generate_prompt(
            block=block,
            template_name="scientific_expert",
            source_lang="en",
            target_lang="fr",
            glossary_terms=glossary,
            context=context
        )
        
        # Check prompts contain necessary elements
        assert "réseaux de neurones" in user  # Glossary term
        assert "en" in user or "English" in system  # Source lang
        assert "fr" in user or "French" in system   # Target lang
    
    def test_template_selection(self):
        """Test template selection based on performance."""
        from scitran.translation.prompts import PromptOptimizer
        
        optimizer = PromptOptimizer()
        
        # Record performance for multiple templates
        for _ in range(15):
            optimizer.record_performance("scientific_expert", bleu_score=42.0, chrf_score=68.0)
            optimizer.record_performance("few_shot_scientific", bleu_score=38.0, chrf_score=64.0)
        
        # Scientific expert should be selected (higher scores)
        best = optimizer.select_best_template(min_usage=10)
        assert best == "scientific_expert"


class TestLayoutIntegration:
    """Test layout preservation integration."""
    
    def test_document_coordinates(self):
        """Test document with coordinate information."""
        from scitran.core.models import Document, Segment, Block, BoundingBox
        
        blocks = [
            Block(
                block_id="title",
                source_text="Document Title",
                bbox=BoundingBox(x0=100, y0=50, x1=500, y1=80, page=0),
                translated_text="Titre du Document"
            ),
            Block(
                block_id="para1",
                source_text="First paragraph content here.",
                bbox=BoundingBox(x0=50, y0=100, x1=550, y1=150, page=0),
                translated_text="Contenu du premier paragraphe ici."
            )
        ]
        
        segment = Segment(segment_id="main", segment_type="content", blocks=blocks)
        document = Document(document_id="test", segments=[segment])
        
        # Verify all blocks have layout info
        for block in document.all_blocks:
            assert block.bbox is not None
            assert block.bbox.page == 0
            assert block.bbox.x0 < block.bbox.x1
            assert block.bbox.y0 < block.bbox.y1


class TestBackendIntegration:
    """Test translation backend integration."""
    
    def test_cascade_backend_available(self):
        """Test cascade backend is always available."""
        from scitran.translation.backends.cascade_backend import CascadeBackend
        
        backend = CascadeBackend()
        assert backend.is_available() == True
        assert backend.name == "cascade"
    
    def test_translation_request_response(self):
        """Test request/response structure."""
        from scitran.translation.base import TranslationRequest, TranslationResponse
        
        request = TranslationRequest(
            text="Hello world",
            source_lang="en",
            target_lang="fr",
            glossary={"hello": "bonjour"}
        )
        
        assert request.text == "Hello world"
        assert request.source_lang == "en"
        assert request.glossary == {"hello": "bonjour"}
        
        response = TranslationResponse(
            translations=["Bonjour le monde"],
            backend="test",
            model="test-model"
        )
        
        assert len(response.translations) == 1
        assert response.backend == "test"


class TestEndToEnd:
    """End-to-end tests simulating real usage."""
    
    def test_text_translation_flow(self):
        """Test complete text translation flow."""
        from scitran.masking.engine import MaskingEngine
        from scitran.core.models import Block
        
        # Input: Scientific text
        source = "The transformer model uses attention: $Q K^T / \\sqrt{d_k}$"
        
        # Step 1: Mask
        engine = MaskingEngine()
        block = Block(block_id="test", source_text=source)
        masked = engine.mask_block(block)
        
        # Step 2: Simulate translation
        masked.translated_text = masked.masked_text.replace(
            "transformer model", "modèle transformer"
        ).replace(
            "uses attention", "utilise l'attention"
        )
        
        # Step 3: Unmask
        result = engine.unmask_block(masked)
        
        # Verify
        assert "modèle transformer" in result.translated_text
        assert "utilise l'attention" in result.translated_text
        assert "$Q K^T / \\sqrt{d_k}$" in result.translated_text  # Formula preserved
    
    def test_document_model_serialization(self):
        """Test document serialization and deserialization."""
        from scitran.core.models import Document, Segment, Block, BoundingBox
        
        block = Block(
            block_id="b1",
            source_text="Test text",
            translated_text="Texte de test",
            bbox=BoundingBox(x0=10, y0=20, x1=100, y1=50, page=0)
        )
        
        segment = Segment(segment_id="s1", segment_type="main", blocks=[block])
        document = Document(document_id="doc1", segments=[segment])
        
        # Serialize
        json_str = document.to_json()
        
        # Deserialize
        restored = Document.from_json(json_str)
        
        assert restored.document_id == "doc1"
        assert len(restored.segments) == 1
        assert restored.segments[0].blocks[0].source_text == "Test text"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

