"""
SPRINT 3: Tests for glossary enforcement system.

These tests verify:
1. Glossary loading from JSON files
2. Term finding in text
3. Prompt injection formatting
4. Post-translation validation
5. Adherence calculation
6. Multi-domain loading
"""

import pytest
from pathlib import Path
import json
import tempfile

from scitran.translation.glossary.manager import (
    GlossaryManager,
    GlossaryTerm,
    GlossaryStats,
    create_glossary
)


class TestGlossaryManager:
    """Test GlossaryManager class."""
    
    def test_initialization(self):
        """Test manager initialization."""
        manager = GlossaryManager()
        assert len(manager) == 0
        assert manager.domains_loaded == set()
    
    def test_add_term(self):
        """Test adding single terms."""
        manager = GlossaryManager()
        manager.add_term("neural network", "réseau de neurones")
        
        assert len(manager) == 1
        assert "neural network" in manager
        assert manager.get_translation("neural network") == "réseau de neurones"
    
    def test_case_insensitive_lookup(self):
        """Test case-insensitive term lookup."""
        manager = GlossaryManager()
        manager.add_term("Neural Network", "réseau de neurones")
        
        # Should work with different cases
        assert "neural network" in manager
        assert "NEURAL NETWORK" in manager
        assert manager.get_translation("neural network") == "réseau de neurones"
    
    def test_load_from_dict(self):
        """Test loading from dictionary."""
        manager = GlossaryManager()
        terms = {
            "foo": "bar",
            "hello": "bonjour",
            "world": "monde"
        }
        
        count = manager.load_from_dict(terms, domain="test")
        
        assert count == 3
        assert len(manager) == 3
        assert "test" in manager.domains_loaded
    
    def test_load_domain_file(self):
        """Test loading domain from JSON file."""
        manager = GlossaryManager()
        
        # Load ML glossary (should exist after SPRINT 3)
        count = manager.load_domain("ml", "en-fr")
        
        # Should load successfully
        assert count > 0
        assert len(manager) > 0
        assert "ml" in manager.domains_loaded
        
        # Check a known term
        assert "machine learning" in manager
        assert manager.get_translation("machine learning") == "apprentissage automatique"
    
    def test_load_multiple_domains(self):
        """Test loading multiple domains."""
        manager = GlossaryManager()
        
        count1 = manager.load_domain("ml", "en-fr")
        count2 = manager.load_domain("physics", "en-fr")
        
        assert count1 > 0
        assert count2 > 0
        assert len(manager.domains_loaded) == 2
        assert len(manager) == count1 + count2


class TestTermFinding:
    """Test finding terms in text."""
    
    def test_find_terms_in_simple_text(self):
        """Test finding terms in simple text."""
        manager = GlossaryManager()
        manager.add_term("neural network", "réseau de neurones")
        manager.add_term("machine learning", "apprentissage automatique")
        
        text = "The neural network uses machine learning."
        terms = manager.find_terms_in_text(text)
        
        assert len(terms) == 2
        sources = [t.source for t in terms]
        assert "neural network" in sources
        assert "machine learning" in sources
    
    def test_find_terms_case_insensitive(self):
        """Test case-insensitive term finding."""
        manager = GlossaryManager()
        manager.add_term("neural network", "réseau de neurones")
        
        text = "The NEURAL NETWORK is powerful."
        terms = manager.find_terms_in_text(text)
        
        assert len(terms) == 1
    
    def test_find_no_terms(self):
        """Test finding when no terms present."""
        manager = GlossaryManager()
        manager.add_term("neural network", "réseau de neurones")
        
        text = "This text has no relevant terms."
        terms = manager.find_terms_in_text(text)
        
        assert len(terms) == 0


class TestPromptGeneration:
    """Test prompt section generation for LLM injection."""
    
    def test_generate_prompt_section(self):
        """Test prompt section formatting."""
        manager = GlossaryManager()
        manager.add_term("neural network", "réseau de neurones")
        manager.add_term("deep learning", "apprentissage profond")
        
        text = "The neural network uses deep learning."
        prompt = manager.generate_prompt_section(text)
        
        assert "Use these terminology translations:" in prompt
        assert "neural network" in prompt
        assert "réseau de neurones" in prompt
        assert "deep learning" in prompt
        assert "apprentissage profond" in prompt
    
    def test_empty_prompt_when_no_terms(self):
        """Test empty prompt when no relevant terms."""
        manager = GlossaryManager()
        manager.add_term("neural network", "réseau de neurones")
        
        text = "This text has no relevant terms."
        prompt = manager.generate_prompt_section(text)
        
        assert prompt == ""
    
    def test_max_terms_limit(self):
        """Test limiting number of terms in prompt."""
        manager = GlossaryManager()
        
        # Add 50 terms
        for i in range(50):
            manager.add_term(f"term{i}", f"terme{i}")
        
        # All terms in text
        text = " ".join([f"term{i}" for i in range(50)])
        
        # Limit to 10
        prompt = manager.generate_prompt_section(text, max_terms=10)
        
        # Should have exactly 10 terms
        lines = prompt.split("\n")
        term_lines = [l for l in lines if "•" in l]
        assert len(term_lines) == 10


class TestTranslationValidation:
    """Test post-translation glossary validation."""
    
    def test_validate_correct_translation(self):
        """Test validation with correct glossary usage."""
        manager = GlossaryManager()
        manager.add_term("neural network", "réseau de neurones")
        manager.add_term("machine learning", "apprentissage automatique")
        
        source = "The neural network uses machine learning."
        target = "Le réseau de neurones utilise l'apprentissage automatique."
        
        stats = manager.validate_translation(source, target)
        
        assert stats.terms_found == 2
        assert stats.terms_enforced == 2
        assert stats.terms_violated == 0
        assert stats.adherence_rate == 1.0
    
    def test_validate_incorrect_translation(self):
        """Test validation with glossary violations."""
        manager = GlossaryManager()
        manager.add_term("neural network", "réseau de neurones")
        
        source = "The neural network is powerful."
        target = "Le réseau neuronal est puissant."  # Wrong! Should be "réseau de neurones"
        
        stats = manager.validate_translation(source, target)
        
        assert stats.terms_found == 1
        assert stats.terms_enforced == 0  # Not found in translation
        assert stats.terms_violated == 1
        assert stats.adherence_rate == 0.0
    
    def test_validate_no_terms_in_source(self):
        """Test validation when no glossary terms in source."""
        manager = GlossaryManager()
        manager.add_term("neural network", "réseau de neurones")
        
        source = "This is a simple sentence."
        target = "Ceci est une phrase simple."
        
        stats = manager.validate_translation(source, target)
        
        assert stats.terms_found == 0
        assert stats.adherence_rate == 1.0  # 100% when no terms expected
    
    def test_validate_partial_adherence(self):
        """Test partial adherence calculation."""
        manager = GlossaryManager()
        manager.add_term("neural network", "réseau de neurones")
        manager.add_term("deep learning", "apprentissage profond")
        manager.add_term("machine learning", "apprentissage automatique")
        
        source = "Neural networks use deep learning and machine learning."
        target = "Les réseaux de neurones utilisent apprentissage profond et ML."
        # Correct: deep learning → apprentissage profond
        # Wrong: neural networks (plural form not detected)
        # Wrong: machine learning → ML (abbreviation)
        
        stats = manager.validate_translation(source, target)
        
        # Should find all 3 terms
        assert stats.terms_found == 3
        # But may only enforce 1 (deep learning)
        assert 0.0 <= stats.adherence_rate <= 1.0


class TestFileOperations:
    """Test file import/export operations."""
    
    def test_load_from_file(self):
        """Test loading from JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            data = {
                "domain": "test",
                "terms": {
                    "foo": "bar",
                    "hello": "bonjour"
                }
            }
            json.dump(data, f)
            temp_path = f.name
        
        try:
            manager = GlossaryManager()
            count = manager.load_from_file(Path(temp_path))
            
            assert count == 2
            assert "foo" in manager
            assert manager.get_translation("hello") == "bonjour"
        finally:
            Path(temp_path).unlink()
    
    def test_export_to_file(self):
        """Test exporting to JSON file."""
        manager = GlossaryManager()
        manager.add_term("foo", "bar", domain="test1")
        manager.add_term("hello", "bonjour", domain="test2")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "export.json"
            
            # Export all
            manager.export_to_file(output_path)
            
            # Verify file created
            assert output_path.exists()
            
            # Load and verify
            with open(output_path) as f:
                data = json.load(f)
            
            assert data['total_terms'] == 2
            assert "foo" in data['terms']
            assert "hello" in data['terms']
    
    def test_export_single_domain(self):
        """Test exporting single domain."""
        manager = GlossaryManager()
        manager.add_term("foo", "bar", domain="keep")
        manager.add_term("hello", "bonjour", domain="skip")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "export.json"
            
            # Export only "keep" domain
            manager.export_to_file(output_path, domain="keep")
            
            with open(output_path) as f:
                data = json.load(f)
            
            assert data['total_terms'] == 1
            assert "foo" in data['terms']
            assert "hello" not in data['terms']


class TestConvenienceFunction:
    """Test create_glossary convenience function."""
    
    def test_create_glossary_with_domains(self):
        """Test creating glossary with domains."""
        glossary = create_glossary(['ml', 'physics'])
        
        assert len(glossary) > 0
        assert 'ml' in glossary.domains_loaded
        assert 'physics' in glossary.domains_loaded
    
    def test_create_glossary_with_custom_terms(self):
        """Test creating glossary with custom terms."""
        custom = {"foo": "bar", "hello": "bonjour"}
        glossary = create_glossary(custom_terms=custom)
        
        assert "foo" in glossary
        assert glossary.get_translation("hello") == "bonjour"
    
    def test_create_glossary_combined(self):
        """Test creating glossary with domains and custom terms."""
        custom = {"custom_term": "terme_personnalisé"}
        glossary = create_glossary(['ml'], custom_terms=custom)
        
        # Should have both domain terms and custom
        assert "machine learning" in glossary  # From ml domain
        assert "custom_term" in glossary  # From custom


class TestGlossaryStats:
    """Test GlossaryStats dataclass."""
    
    def test_adherence_calculation(self):
        """Test adherence rate calculation."""
        stats = GlossaryStats()
        stats.terms_found = 10
        stats.terms_enforced = 8
        
        rate = stats.calculate_adherence()
        
        assert rate == 0.8
        assert stats.adherence_rate == 0.8
    
    def test_adherence_no_terms_found(self):
        """Test adherence when no terms expected."""
        stats = GlossaryStats()
        stats.terms_found = 0
        stats.terms_enforced = 0
        
        rate = stats.calculate_adherence()
        
        assert rate == 1.0  # 100% when no terms expected


class TestGlossaryTerm:
    """Test GlossaryTerm dataclass."""
    
    def test_term_creation(self):
        """Test creating a glossary term."""
        term = GlossaryTerm(
            source="neural network",
            target="réseau de neurones",
            domain="ml"
        )
        
        assert term.source == "neural network"
        assert term.target == "réseau de neurones"
        assert term.domain == "ml"
        assert term.confidence == 1.0
    
    def test_term_matching_case_insensitive(self):
        """Test term matching."""
        term = GlossaryTerm(
            source="neural network",
            target="réseau de neurones",
            case_sensitive=False
        )
        
        assert term.matches("The neural network is powerful")
        assert term.matches("The NEURAL NETWORK is powerful")
        assert not term.matches("This is unrelated text")
    
    def test_term_matching_case_sensitive(self):
        """Test case-sensitive matching."""
        term = GlossaryTerm(
            source="DNA",
            target="ADN",
            case_sensitive=True
        )
        
        assert term.matches("The DNA sequence")
        assert not term.matches("The dna sequence")  # Wrong case


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


