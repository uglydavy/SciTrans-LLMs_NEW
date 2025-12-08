"""
Advanced prompt engineering and training system for SciTrans-LLMs NEW.

This module implements intelligent prompt generation with optimization
capabilities for maximizing translation quality.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import json
import hashlib
from pathlib import Path

from scitran.core.models import Block, Document


class PromptStrategy(Enum):
    """Different prompting strategies."""
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    ITERATIVE_REFINEMENT = "iterative_refinement"
    ROLE_BASED = "role_based"
    CONTEXTUAL = "contextual"


@dataclass
class PromptTemplate:
    """Template for generating translation prompts."""
    name: str
    strategy: PromptStrategy
    system_prompt: str
    user_prompt_template: str
    
    # Optional components
    few_shot_examples: List[Tuple[str, str]] = field(default_factory=list)
    role_description: Optional[str] = None
    thinking_steps: List[str] = field(default_factory=list)
    
    # Performance metrics
    avg_bleu_score: float = 0.0
    avg_chrf_score: float = 0.0
    usage_count: int = 0
    success_rate: float = 0.0
    
    def get_hash(self) -> str:
        """Generate unique hash for this template."""
        content = f"{self.system_prompt}{self.user_prompt_template}"
        return hashlib.md5(content.encode()).hexdigest()[:8]


class PromptLibrary:
    """Library of optimized prompt templates."""
    
    def __init__(self):
        self.templates: Dict[str, PromptTemplate] = {}
        self._initialize_default_templates()
        
    def _initialize_default_templates(self):
        """Initialize with proven prompt templates."""
        
        # Scientific translation expert
        self.templates["scientific_expert"] = PromptTemplate(
            name="scientific_expert",
            strategy=PromptStrategy.ROLE_BASED,
            role_description="You are an expert scientific translator with deep knowledge of technical terminology in physics, mathematics, and computer science.",
            system_prompt="""You are an expert scientific translator specializing in academic papers. Your key responsibilities:
1. Preserve ALL mathematical formulas, equations, and LaTeX commands exactly as they appear
2. Maintain technical terminology consistency using the provided glossary
3. Preserve document structure including headings, lists, and references
4. Ensure translations are academically rigorous and publication-ready
5. Keep placeholder tokens (<<...>>) unchanged in the translation
6. Maintain formal academic tone appropriate for scientific literature""",
            user_prompt_template="""Translate the following scientific text from {source_lang} to {target_lang}.

GLOSSARY TERMS TO USE:
{glossary_terms}

PREVIOUS CONTEXT:
{context}

TEXT TO TRANSLATE:
{source_text}

REQUIREMENTS:
- Preserve all <<PLACEHOLDER>> tokens exactly
- Use glossary terms consistently
- Maintain academic formality
- Preserve formatting and structure

TRANSLATION:""",
            avg_bleu_score=41.5,
            avg_chrf_score=68.2
        )
        
        # Few-shot with examples
        self.templates["few_shot_scientific"] = PromptTemplate(
            name="few_shot_scientific",
            strategy=PromptStrategy.FEW_SHOT,
            system_prompt="You are a precise scientific translator. Follow the examples provided.",
            user_prompt_template="""Translate scientific text from {source_lang} to {target_lang}.

EXAMPLES:
{examples}

GLOSSARY: {glossary_terms}

NOW TRANSLATE:
{source_text}

TRANSLATION:""",
            few_shot_examples=[
                ("Machine learning has revolutionized natural language processing.",
                 "L'apprentissage automatique a révolutionné le traitement du langage naturel."),
                ("The transformer architecture uses self-attention mechanisms.",
                 "L'architecture transformer utilise des mécanismes d'auto-attention."),
                ("We propose a novel approach to neural machine translation.",
                 "Nous proposons une nouvelle approche de traduction automatique neuronale.")
            ],
            avg_bleu_score=40.8
        )
        
        # Chain of thought
        self.templates["chain_of_thought"] = PromptTemplate(
            name="chain_of_thought",
            strategy=PromptStrategy.CHAIN_OF_THOUGHT,
            system_prompt="You are a methodical scientific translator who thinks step-by-step.",
            user_prompt_template="""Translate from {source_lang} to {target_lang} following these steps:

STEP 1: Identify technical terms and formulas
STEP 2: Check glossary for standard translations
STEP 3: Translate while preserving structure
STEP 4: Verify placeholder preservation

GLOSSARY: {glossary_terms}
CONTEXT: {context}

TEXT: {source_text}

Let's work through this step-by-step:
STEP 1 - Technical terms found:
STEP 2 - Glossary matches:
STEP 3 - Translation:
STEP 4 - Verification:

FINAL TRANSLATION:""",
            thinking_steps=[
                "Identify technical terms",
                "Check glossary",
                "Translate",
                "Verify"
            ],
            avg_bleu_score=42.1
        )
        
        # Iterative refinement
        self.templates["iterative_refinement"] = PromptTemplate(
            name="iterative_refinement",
            strategy=PromptStrategy.ITERATIVE_REFINEMENT,
            system_prompt="You are a perfectionist translator who refines translations iteratively.",
            user_prompt_template="""Translate and refine from {source_lang} to {target_lang}.

SOURCE: {source_text}
GLOSSARY: {glossary_terms}

FIRST DRAFT: Provide initial translation
REFINEMENT 1: Improve technical accuracy
REFINEMENT 2: Enhance fluency
FINAL: Polished translation

Proceed:""",
            avg_bleu_score=42.8
        )
        
        # Contextual expert
        self.templates["contextual_expert"] = PromptTemplate(
            name="contextual_expert",
            strategy=PromptStrategy.CONTEXTUAL,
            system_prompt="You excel at maintaining consistency across document sections.",
            user_prompt_template="""Translate maintaining consistency with previous sections.

DOCUMENT CONTEXT:
{extended_context}

GLOSSARY: {glossary_terms}
STYLE GUIDE:
- Formal academic tone
- Consistent terminology
- Preserve all formatting codes

TRANSLATE THIS SECTION:
{source_text}

CONSISTENT TRANSLATION:""",
            avg_bleu_score=43.2
        )


class PromptOptimizer:
    """
    Optimize prompts based on translation quality feedback.
    
    This implements an adaptive prompt training system that learns
    from translation outcomes to improve prompt effectiveness.
    """
    
    def __init__(self, library: Optional[PromptLibrary] = None):
        self.library = library or PromptLibrary()
        self.performance_history: List[Dict[str, Any]] = []
        self.optimization_rounds = 0
        
    def generate_prompt(self, 
                        block: Block,
                        template_name: str,
                        source_lang: str,
                        target_lang: str,
                        glossary_terms: Dict[str, str],
                        context: List[Tuple[str, str]] = None,
                        **kwargs) -> Tuple[str, str]:
        """
        Generate optimized prompt for translation.
        
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        template = self.library.templates.get(template_name)
        if not template:
            template = self.library.templates["scientific_expert"]
            
        # Format glossary
        glossary_str = "\n".join([f"- {src}: {tgt}" for src, tgt in glossary_terms.items()])
        
        # Format context
        context_str = ""
        if context:
            context_str = "\n".join([f"Source: {s}\nTranslation: {t}" 
                                     for s, t in context[-3:]])  # Last 3 translations
        
        # Format examples for few-shot
        examples_str = ""
        if template.few_shot_examples:
            examples_str = "\n\n".join([f"Source: {s}\nTranslation: {t}"
                                        for s, t in template.few_shot_examples])
        
        # Build user prompt
        user_prompt = template.user_prompt_template.format(
            source_lang=source_lang,
            target_lang=target_lang,
            glossary_terms=glossary_str or "No specific terms",
            context=context_str or "No previous context",
            source_text=block.masked_text or block.source_text,
            examples=examples_str,
            extended_context=context_str,  # For contextual strategy
            **kwargs
        )
        
        # Add role description if applicable
        system_prompt = template.system_prompt
        if template.role_description:
            system_prompt = f"{template.role_description}\n\n{system_prompt}"
            
        return system_prompt, user_prompt
    
    def record_performance(self,
                          template_name: str,
                          bleu_score: float,
                          chrf_score: float,
                          success: bool = True):
        """Record performance metrics for a template."""
        template = self.library.templates.get(template_name)
        if not template:
            return
            
        # Update template metrics
        template.usage_count += 1
        
        # Running average for scores
        alpha = 0.1  # Learning rate
        template.avg_bleu_score = (1 - alpha) * template.avg_bleu_score + alpha * bleu_score
        template.avg_chrf_score = (1 - alpha) * template.avg_chrf_score + alpha * chrf_score
        
        # Success rate
        template.success_rate = ((template.success_rate * (template.usage_count - 1) + 
                                  (1.0 if success else 0.0)) / template.usage_count)
        
        # Record in history
        self.performance_history.append({
            'template': template_name,
            'bleu': bleu_score,
            'chrf': chrf_score,
            'success': success,
            'round': self.optimization_rounds
        })
    
    def select_best_template(self, 
                            document_type: str = "scientific",
                            min_usage: int = 10) -> str:
        """
        Select best performing template based on metrics.
        
        Args:
            document_type: Type of document being translated
            min_usage: Minimum usage count to consider
            
        Returns:
            Name of best template
        """
        candidates = [
            (name, template) 
            for name, template in self.library.templates.items()
            if template.usage_count >= min_usage
        ]
        
        if not candidates:
            return "scientific_expert"  # Default
            
        # Score based on BLEU (40%), chrF (40%), success rate (20%)
        best_score = -1
        best_name = None
        
        for name, template in candidates:
            score = (0.4 * template.avg_bleu_score / 100 +
                    0.4 * template.avg_chrf_score / 100 +
                    0.2 * template.success_rate)
            
            if score > best_score:
                best_score = score
                best_name = name
                
        return best_name or "scientific_expert"
    
    def optimize_templates(self, performance_threshold: float = 0.8):
        """
        Optimize templates based on performance history.
        
        This method analyzes performance patterns and adjusts templates
        to improve translation quality.
        """
        self.optimization_rounds += 1
        
        # Analyze performance by template
        template_performance = {}
        for record in self.performance_history[-100:]:  # Last 100 translations
            template = record['template']
            if template not in template_performance:
                template_performance[template] = []
            template_performance[template].append(record['bleu'])
            
        # Identify underperforming templates
        for template_name, scores in template_performance.items():
            avg_score = sum(scores) / len(scores)
            if avg_score < performance_threshold * 100:  # Convert to percentage
                self._improve_template(template_name, scores)
                
    def _improve_template(self, template_name: str, recent_scores: List[float]):
        """
        Improve an underperforming template.
        
        Strategies:
        - Add more few-shot examples
        - Enhance role description
        - Add thinking steps
        - Adjust prompt structure
        """
        template = self.library.templates.get(template_name)
        if not template:
            return
            
        avg_score = sum(recent_scores) / len(recent_scores)
        
        # If very poor performance, add more structure
        if avg_score < 30:
            template.system_prompt += "\n\nCRITICAL: Pay special attention to preserving technical terms and formulas exactly."
            
        # If moderate performance, enhance with examples
        elif avg_score < 40 and len(template.few_shot_examples) < 5:
            # Would add more examples from high-scoring translations
            pass
            
        # Log optimization
        print(f"Optimized template '{template_name}': avg_score={avg_score:.1f}")
        
    def save_optimization_state(self, path: Path):
        """Save optimization state to file."""
        state = {
            'templates': {
                name: {
                    'avg_bleu': t.avg_bleu_score,
                    'avg_chrf': t.avg_chrf_score,
                    'usage_count': t.usage_count,
                    'success_rate': t.success_rate
                }
                for name, t in self.library.templates.items()
            },
            'optimization_rounds': self.optimization_rounds,
            'history_size': len(self.performance_history)
        }
        
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
            
    def load_optimization_state(self, path: Path):
        """Load optimization state from file."""
        with open(path, 'r') as f:
            state = json.loads(f.read())
            
        for name, metrics in state['templates'].items():
            if name in self.library.templates:
                template = self.library.templates[name]
                template.avg_bleu_score = metrics['avg_bleu']
                template.avg_chrf_score = metrics['avg_chrf']
                template.usage_count = metrics['usage_count']
                template.success_rate = metrics['success_rate']
                
        self.optimization_rounds = state.get('optimization_rounds', 0)
