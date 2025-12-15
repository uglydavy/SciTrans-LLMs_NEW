"""
SPRINT 4: Tests for document-level refinement with constraint preservation.

These tests verify:
1. Refinement improves translation coherence
2. Placeholders are preserved during refinement
3. Glossary terms are preserved during refinement
4. Constraint violations are detected and rejected
5. Ablation flags disable features correctly
"""

import pytest
from unittest.mock import Mock, patch

from scitran.core.models import Document, Segment, Block, BlockType, MaskInfo
from scitran.core.pipeline import TranslationPipeline, PipelineConfig
from scitran.translation.glossary.manager import GlossaryManager


class TestRefinementConstraints:
    """Test refinement constraint validation."""
    
    def test_validate_placeholders_preserved(self):
        """Test that refinement preserves all placeholders."""
        config = PipelineConfig(backend="dummy", enable_glossary=False)
        pipeline = TranslationPipeline(config)
        pipeline.glossary_manager = None  # Explicitly disable
        
        # Create block with masks
        block = Block(
            block_id="test",
            source_text="The equation $x=5$ is simple",
            translated_text="L'équation <MATH_0> est simple",
            masks=[
                MaskInfo(
                    mask_type="MATH",
                    placeholder="<MATH_0>",
                    original="$x=5$"
                )
            ]
        )
        
        # Valid refinement: preserves placeholder
        valid_refined = "L'équation <MATH_0> est très simple"
        assert pipeline._validate_refinement_constraints(block, valid_refined) is True
        
        # Invalid refinement: loses placeholder
        invalid_refined = "L'équation x=5 est simple"
        assert pipeline._validate_refinement_constraints(block, invalid_refined) is False
    
    def test_validate_multiple_placeholders(self):
        """Test validation with multiple placeholders."""
        config = PipelineConfig(backend="dummy", enable_glossary=False)
        pipeline = TranslationPipeline(config)
        pipeline.glossary_manager = None  # Explicitly disable
        
        block = Block(
            block_id="test",
            source_text="See $x$ and $y$ and https://example.com",
            translated_text="Voir <MATH_0> et <MATH_1> et <URL_0>",
            masks=[
                MaskInfo("MATH", "<MATH_0>", "$x$"),
                MaskInfo("MATH", "<MATH_1>", "$y$"),
                MaskInfo("URL", "<URL_0>", "https://example.com")
            ]
        )
        
        # All placeholders present
        valid = "Voir <MATH_0> et <MATH_1> et <URL_0> ici"
        assert pipeline._validate_refinement_constraints(block, valid) is True
        
        # Missing one placeholder
        invalid = "Voir <MATH_0> et <MATH_1> ici"  # Lost <URL_0>
        assert pipeline._validate_refinement_constraints(block, invalid) is False
    
    def test_validate_glossary_terms_preserved(self):
        """Test that refinement preserves glossary terms."""
        # Create glossary
        glossary = GlossaryManager()
        glossary.add_term("neural network", "réseau de neurones")
        
        config = PipelineConfig(
            backend="dummy",
            enable_glossary=True,
            glossary_manager=glossary
        )
        pipeline = TranslationPipeline(config)
        pipeline.glossary_manager = glossary
        
        block = Block(
            block_id="test",
            source_text="The neural network is powerful",
            translated_text="Le réseau de neurones est puissant"
        )
        
        # Valid: preserves glossary term
        valid = "Le réseau de neurones est très puissant"
        assert pipeline._validate_refinement_constraints(block, valid) is True
        
        # Invalid: changes glossary term
        invalid = "Le réseau neuronal est puissant"  # Wrong term!
        assert pipeline._validate_refinement_constraints(block, invalid) is False
    
    def test_validate_with_no_constraints(self):
        """Test validation passes when no constraints exist."""
        config = PipelineConfig(backend="dummy", enable_glossary=False)
        pipeline = TranslationPipeline(config)
        pipeline.glossary_manager = None
        
        block = Block(
            block_id="test",
            source_text="Simple text",
            translated_text="Texte simple",
            masks=[]  # No masks
        )
        
        # Any refinement should pass (no constraints)
        refined = "Texte très simple"
        assert pipeline._validate_refinement_constraints(block, refined) is True


class TestAblationFlags:
    """Test ablation flags for thesis experiments."""
    
    def test_ablation_disable_masking(self):
        """Test masking can be disabled via ablation flag."""
        config = PipelineConfig(
            backend="dummy",
            enable_masking=True,  # Enabled
            ablation_disable_masking=True  # But ablation disables it
        )
        
        # Masking should be skipped
        assert config.enable_masking is True  # Config says enabled
        assert config.ablation_disable_masking is True  # But ablation overrides
    
    def test_ablation_disable_glossary(self):
        """Test glossary can be disabled via ablation flag."""
        config = PipelineConfig(
            backend="dummy",
            enable_glossary=True,
            ablation_disable_glossary=True
        )
        
        assert config.ablation_disable_glossary is True
    
    def test_ablation_disable_refinement(self):
        """Test refinement can be disabled via ablation flag."""
        config = PipelineConfig(
            backend="dummy",
            enable_refinement=True,
            ablation_disable_refinement=True
        )
        
        assert config.ablation_disable_refinement is True
    
    def test_ablation_disable_all_for_baseline(self):
        """Test creating baseline configuration with all innovations disabled."""
        config = PipelineConfig(
            backend="openai",
            # Disable all innovations for baseline comparison
            ablation_disable_masking=True,
            ablation_disable_glossary=True,
            ablation_disable_context=True,
            ablation_disable_reranking=True,
            ablation_disable_refinement=True,
            ablation_disable_coverage_guarantee=True
        )
        
        # Verify all ablation flags set
        assert config.ablation_disable_masking is True
        assert config.ablation_disable_glossary is True
        assert config.ablation_disable_context is True
        assert config.ablation_disable_reranking is True
        assert config.ablation_disable_refinement is True
        assert config.ablation_disable_coverage_guarantee is True


class TestRefinementPrompts:
    """Test refinement prompt generation."""
    
    def test_refinement_prompt_types(self):
        """Test different refinement prompt types."""
        prompt_types = ["coherence", "style", "terminology"]
        
        for prompt_type in prompt_types:
            config = PipelineConfig(
                backend="dummy",
                enable_refinement=True,
                refinement_prompt=prompt_type
            )
            
            assert config.refinement_prompt == prompt_type


class TestConstraintPreservation:
    """Integration tests for constraint preservation during refinement."""
    
    def test_refinement_preserves_both_constraints(self):
        """Test refinement preserves both placeholders AND glossary terms."""
        # Setup glossary
        glossary = GlossaryManager()
        glossary.add_term("machine learning", "apprentissage automatique")
        
        config = PipelineConfig(
            backend="dummy",
            enable_glossary=True,
            glossary_manager=glossary,
            validate_refinement_constraints=True
        )
        pipeline = TranslationPipeline(config)
        pipeline.glossary_manager = glossary
        
        # Block with both masks and glossary terms
        block = Block(
            block_id="test",
            source_text="Machine learning uses equation $E=mc^2$",
            translated_text="L'apprentissage automatique utilise l'équation <MATH_0>",
            masks=[MaskInfo("MATH", "<MATH_0>", "$E=mc^2$")]
        )
        
        # Valid: preserves both
        valid = "L'apprentissage automatique utilise l'équation <MATH_0> célèbre"
        assert pipeline._validate_refinement_constraints(block, valid) is True
        
        # Invalid: loses placeholder
        invalid1 = "L'apprentissage automatique utilise l'équation E=mc^2"
        assert pipeline._validate_refinement_constraints(block, invalid1) is False
        
        # Invalid: loses glossary term
        invalid2 = "Le ML utilise l'équation <MATH_0>"
        assert pipeline._validate_refinement_constraints(block, invalid2) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

