"""Basic usage examples for SciTrans-LLMs."""

from scitran.core.pipeline import TranslationPipeline, PipelineConfig
from scitran.extraction.pdf_parser import PDFParser
from scitran.rendering.pdf_renderer import PDFRenderer


def example_1_simple_translation():
    """Example 1: Simple document translation."""
    
    print("=" * 60)
    print("Example 1: Simple Translation")
    print("=" * 60)
    
    # Parse PDF
    parser = PDFParser()
    document = parser.parse("examples/sample.pdf")
    
    # Configure pipeline
    config = PipelineConfig(
        source_lang="en",
        target_lang="fr",
        backend="openai",
        model="gpt-4o"
    )
    
    # Translate
    pipeline = TranslationPipeline(config)
    result = pipeline.translate_document(document)
    
    # Save result
    renderer = PDFRenderer()
    renderer.render_simple(result.document, "output_simple.pdf")
    
    print(f"✓ Translated {len(result.document.blocks)} blocks")
    print(f"✓ Output saved to output_simple.pdf")


def example_2_with_masking():
    """Example 2: Translation with LaTeX masking."""
    
    print("\n" + "=" * 60)
    print("Example 2: Translation with Masking")
    print("=" * 60)
    
    parser = PDFParser()
    document = parser.parse("examples/math_paper.pdf")
    
    config = PipelineConfig(
        source_lang="en",
        target_lang="fr",
        backend="openai",
        enable_masking=True,  # Enable LaTeX masking
        enable_context=False
    )
    
    pipeline = TranslationPipeline(config)
    result = pipeline.translate_document(document)
    
    renderer = PDFRenderer()
    renderer.render_simple(result.document, "output_masked.pdf")
    
    print(f"✓ LaTeX preservation: {result.stats.latex_preservation_rate:.1f}%")
    print(f"✓ Output saved to output_masked.pdf")


def example_3_with_reranking():
    """Example 3: Translation with multi-candidate reranking."""
    
    print("\n" + "=" * 60)
    print("Example 3: Translation with Reranking")
    print("=" * 60)
    
    parser = PDFParser()
    document = parser.parse("examples/sample.pdf")
    
    config = PipelineConfig(
        source_lang="en",
        target_lang="fr",
        backend="openai",
        num_candidates=3,  # Generate 3 candidates
        enable_reranking=True,  # Enable reranking
        enable_masking=True,
        enable_context=True
    )
    
    pipeline = TranslationPipeline(config)
    result = pipeline.translate_document(document)
    
    renderer = PDFRenderer()
    renderer.render_simple(result.document, "output_reranked.pdf")
    
    print(f"✓ Average quality: {result.stats.avg_quality:.2f}")
    print(f"✓ Output saved to output_reranked.pdf")


def example_4_context_aware():
    """Example 4: Context-aware translation."""
    
    print("\n" + "=" * 60)
    print("Example 4: Context-Aware Translation")
    print("=" * 60)
    
    parser = PDFParser()
    document = parser.parse("examples/long_paper.pdf")
    
    config = PipelineConfig(
        source_lang="en",
        target_lang="fr",
        backend="anthropic",
        model="claude-3-5-sonnet-20241022",
        enable_context=True,  # Enable context window
        context_window_size=3,
        enable_masking=True
    )
    
    pipeline = TranslationPipeline(config)
    result = pipeline.translate_document(document)
    
    renderer = PDFRenderer()
    renderer.render_with_layout("examples/long_paper.pdf", result.document, "output_context.pdf")
    
    print(f"✓ Translation time: {result.stats.total_time:.1f}s")
    print(f"✓ Output saved with layout preservation")


def example_5_custom_glossary():
    """Example 5: Translation with custom glossary."""
    
    print("\n" + "=" * 60)
    print("Example 5: Custom Glossary")
    print("=" * 60)
    
    # Define custom terminology
    custom_glossary = {
        "neural network": "réseau neuronal",
        "deep learning": "apprentissage profond",
        "transformer": "transformeur",
        "attention mechanism": "mécanisme d'attention"
    }
    
    parser = PDFParser()
    document = parser.parse("examples/ml_paper.pdf")
    
    config = PipelineConfig(
        source_lang="en",
        target_lang="fr",
        backend="deepseek",  # Affordable option
        glossary=custom_glossary
    )
    
    pipeline = TranslationPipeline(config)
    result = pipeline.translate_document(document)
    
    renderer = PDFRenderer()
    renderer.render_markdown(result.document, "output_glossary.md")
    
    print(f"✓ Used {len(custom_glossary)} glossary terms")
    print(f"✓ Cost: ${result.stats.total_cost:.4f}")
    print(f"✓ Output saved as Markdown")


def example_6_batch_translation():
    """Example 6: Batch translation of multiple documents."""
    
    print("\n" + "=" * 60)
    print("Example 6: Batch Translation")
    print("=" * 60)
    
    from pathlib import Path
    
    input_dir = Path("examples/batch")
    output_dir = Path("output/batch")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_files = list(input_dir.glob("*.pdf"))
    
    parser = PDFParser()
    config = PipelineConfig(
        source_lang="en",
        target_lang="fr",
        backend="openai"
    )
    pipeline = TranslationPipeline(config)
    renderer = PDFRenderer()
    
    total_cost = 0.0
    
    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")
        
        document = parser.parse(str(pdf_file))
        result = pipeline.translate_document(document)
        
        output_file = output_dir / f"{pdf_file.stem}_translated.pdf"
        renderer.render_simple(result.document, str(output_file))
        
        total_cost += result.stats.total_cost
        print(f"  ✓ Cost: ${result.stats.total_cost:.4f}")
    
    print(f"\n✓ Processed {len(pdf_files)} documents")
    print(f"✓ Total cost: ${total_cost:.4f}")


if __name__ == "__main__":
    import sys
    
    # Check API key
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set. Some examples will fail.")
        print("Set with: export OPENAI_API_KEY='your-key'")
    
    # Run example based on command line argument
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        examples = {
            "1": example_1_simple_translation,
            "2": example_2_with_masking,
            "3": example_3_with_reranking,
            "4": example_4_context_aware,
            "5": example_5_custom_glossary,
            "6": example_6_batch_translation
        }
        
        if example_num in examples:
            examples[example_num]()
        else:
            print(f"Example {example_num} not found")
    else:
        print("Usage: python basic_usage.py <example_number>")
        print("\nAvailable examples:")
        print("  1 - Simple translation")
        print("  2 - Translation with masking")
        print("  3 - Translation with reranking")
        print("  4 - Context-aware translation")
        print("  5 - Custom glossary")
        print("  6 - Batch translation")
