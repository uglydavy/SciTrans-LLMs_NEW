#!/usr/bin/env python3
"""
Reproducibility harness script for debugging translation issues.

This script:
1. Runs translation on a sample PDF
2. Outputs JSON artifacts (block list + per-block status)
3. Renders output pages to PNG for visual comparison
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import fitz  # PyMuPDF
from scitran.extraction.pdf_parser import PDFParser
from scitran.core.pipeline import TranslationPipeline, PipelineConfig
from scitran.rendering.pdf_renderer import PDFRenderer


def render_pdf_to_png(pdf_path: Path, output_dir: Path, prefix: str = "page"):
    """Render PDF pages to PNG for visual comparison."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    doc = fitz.open(pdf_path)
    png_paths = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(dpi=150)
        png_path = output_dir / f"{prefix}_{page_num + 1:03d}.png"
        pix.save(png_path)
        png_paths.append(png_path)
    
    doc.close()
    return png_paths


def extract_block_status(document) -> list:
    """Extract per-block translation status."""
    blocks_status = []
    
    for segment in document.segments:
        for block in segment.blocks:
            status = {
                "block_id": block.block_id,
                "block_type": str(block.block_type),
                "is_translatable": block.is_translatable,
                "has_source": bool(block.source_text and block.source_text.strip()),
                "has_translation": bool(block.translated_text and block.translated_text.strip()),
                "has_bbox": block.bbox is not None,
                "page": block.bbox.page if block.bbox else None,
                "source_length": len(block.source_text) if block.source_text else 0,
                "translation_length": len(block.translated_text) if block.translated_text else 0,
            }
            
            # Add metadata if available
            if block.metadata:
                status["metadata_status"] = block.metadata.status if hasattr(block.metadata, 'status') else None
                status["failure_reason"] = block.metadata.failure_reason if hasattr(block.metadata, 'failure_reason') else None
                status["retry_count"] = block.metadata.retry_count if hasattr(block.metadata, 'retry_count') else 0
            
            blocks_status.append(status)
    
    return blocks_status


def main():
    parser = argparse.ArgumentParser(description="Translation reproducibility harness")
    parser.add_argument("pdf_path", type=Path, help="Path to input PDF")
    parser.add_argument("--output-dir", type=Path, default=Path("repro_output"), 
                        help="Output directory for artifacts")
    parser.add_argument("--backend", default="deepseek", help="Translation backend")
    parser.add_argument("--source-lang", default="en", help="Source language")
    parser.add_argument("--target-lang", default="fr", help="Target language")
    parser.add_argument("--max-pages", type=int, default=2, help="Max pages to process")
    parser.add_argument("--strict-mode", action="store_true", help="Enable strict mode")
    
    args = parser.parse_args()
    
    # Create output directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"=== SciTrans Translation Repro Harness ===")
    print(f"Input PDF: {args.pdf_path}")
    print(f"Output dir: {output_dir}")
    print(f"Backend: {args.backend}")
    print(f"Language: {args.source_lang} -> {args.target_lang}")
    print(f"Max pages: {args.max_pages}")
    print()
    
    # Step 1: Parse PDF
    print("[1/5] Parsing PDF...")
    parser = PDFParser()
    document = parser.parse(str(args.pdf_path), max_pages=args.max_pages)
    
    num_blocks = len(document.all_blocks)
    num_translatable = len(document.translatable_blocks)
    print(f"  Total blocks: {num_blocks}")
    print(f"  Translatable blocks: {num_translatable}")
    
    # Save extraction artifacts
    extraction_data = {
        "document_id": document.document_id,
        "num_pages": len(set(b.bbox.page for b in document.all_blocks if b.bbox)),
        "num_blocks": num_blocks,
        "num_translatable": num_translatable,
        "blocks": extract_block_status(document)
    }
    
    with open(output_dir / "extraction.json", "w") as f:
        json.dump(extraction_data, f, indent=2)
    
    # Step 2: Translate
    print("\n[2/5] Translating document...")
    config = PipelineConfig(
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        backend=args.backend,
        strict_mode=args.strict_mode,
        enable_masking=True,
        enable_context=True,
        enable_reranking=False,  # Disable for speed in repro
        num_candidates=1,
        generate_artifacts=True,
        artifact_dir=output_dir / "artifacts"
    )
    
    pipeline = TranslationPipeline(config)
    result = pipeline.translate_document(document)
    
    print(f"  Success: {result.success}")
    print(f"  Duration: {result.duration:.2f}s")
    print(f"  Coverage: {result.coverage:.1%}")
    print(f"  Blocks translated: {result.blocks_translated}")
    print(f"  Blocks failed: {result.blocks_failed}")
    
    # Save translation status
    translation_data = {
        "success": result.success,
        "duration": result.duration,
        "coverage": result.coverage,
        "blocks_translated": result.blocks_translated,
        "blocks_failed": result.blocks_failed,
        "error": result.error,
        "blocks": extract_block_status(document)
    }
    
    with open(output_dir / "translation_status.json", "w") as f:
        json.dump(translation_data, f, indent=2)
    
    # Step 3: Render output PDF
    print("\n[3/5] Rendering translated PDF...")
    output_pdf = output_dir / "translated.pdf"
    renderer = PDFRenderer(strict_mode=args.strict_mode)
    renderer.render_pdf(document, str(output_pdf), source_pdf=str(args.pdf_path))
    print(f"  Output PDF: {output_pdf}")
    
    # Step 4: Render source pages to PNG
    print("\n[4/5] Rendering source pages to PNG...")
    source_pngs = render_pdf_to_png(args.pdf_path, output_dir / "source_pages", prefix="source")
    print(f"  Rendered {len(source_pngs)} source pages")
    
    # Step 5: Render translated pages to PNG
    print("\n[5/5] Rendering translated pages to PNG...")
    translated_pngs = render_pdf_to_png(output_pdf, output_dir / "translated_pages", prefix="translated")
    print(f"  Rendered {len(translated_pngs)} translated pages")
    
    # Summary
    print("\n=== Summary ===")
    print(f"Output directory: {output_dir}")
    print(f"- extraction.json: Block extraction data")
    print(f"- translation_status.json: Per-block translation status")
    print(f"- translated.pdf: Output PDF")
    print(f"- source_pages/: Source page PNGs")
    print(f"- translated_pages/: Translated page PNGs")
    
    # Report any issues
    failed_blocks = [b for b in translation_data["blocks"] 
                     if b["is_translatable"] and not b["has_translation"]]
    
    if failed_blocks:
        print(f"\n⚠️  WARNING: {len(failed_blocks)} translatable blocks missing translation:")
        for block in failed_blocks[:5]:  # Show first 5
            print(f"  - Block {block['block_id']} (page {block['page']}, type {block['block_type']})")
        if len(failed_blocks) > 5:
            print(f"  ... and {len(failed_blocks) - 5} more")
    else:
        print("\n✓ All translatable blocks have translations")
    
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())

