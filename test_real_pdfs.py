#!/usr/bin/env python3
"""
Test SciTrans-LLMs with real scientific PDFs from arXiv.

Tests various configurations:
- With/without masking
- With/without reranking
- Different backends
- Different models
"""

import sys
import os
import time
import requests
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel

console = Console()

# Sample arXiv papers (publicly available)
ARXIV_PAPERS = [
    {
        "id": "2104.09864",
        "title": "Attention Is All You Need (Transformer)",
        "url": "https://arxiv.org/pdf/2104.09864.pdf",
        "category": "Machine Learning"
    },
    {
        "id": "1706.03762",
        "title": "Attention Is All You Need",
        "url": "https://arxiv.org/pdf/1706.03762.pdf",
        "category": "NLP"
    },
    {
        "id": "2010.11929",
        "title": "BERT Explained",
        "url": "https://arxiv.org/pdf/2010.11929.pdf",
        "category": "NLP"
    }
]


def download_pdf(url: str, output_path: Path) -> bool:
    """Download a PDF from URL."""
    try:
        console.print(f"[cyan]Downloading {url}...[/cyan]")
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        console.print(f"[green]✓ Downloaded to {output_path}[/green]")
        return True
    except Exception as e:
        console.print(f"[red]✗ Download failed: {e}[/red]")
        return False


def test_configuration(pdf_path: Path, backend: str, enable_masking: bool, enable_reranking: bool, max_pages: int = 2):
    """Test a specific configuration."""
    
    from scitran.core.pipeline import TranslationPipeline, PipelineConfig
    from scitran.extraction.pdf_parser import PDFParser
    
    config_name = f"{backend}"
    if enable_masking:
        config_name += "+mask"
    if enable_reranking:
        config_name += "+rerank"
    
    console.print(f"\n[bold cyan]Testing: {config_name}[/bold cyan]")
    
    try:
        # Parse PDF
        parser = PDFParser()
        document = parser.parse(str(pdf_path), max_pages=max_pages)
        
        # Configure pipeline
        config = PipelineConfig(
            source_lang="en",
            target_lang="fr",
            backend=backend,
            enable_masking=enable_masking,
            enable_reranking=enable_reranking,
            num_candidates=3 if enable_reranking else 1
        )
        
        # Translate
        start = time.time()
        pipeline = TranslationPipeline(config)
        result = pipeline.translate_document(document)
        elapsed = time.time() - start
        
        # Calculate metrics
        total_chars = sum(len(b.source_text) for b in document.all_blocks)
        total_translated = sum(1 for b in result.document.all_blocks if b.translated_text)
        
        # Count LaTeX preservation
        latex_preserved = 0
        latex_total = 0
        for block in result.document.all_blocks:
            if block.source_text:
                source_latex = block.source_text.count('$')
                latex_total += source_latex
                if block.translated_text:
                    trans_latex = block.translated_text.count('$')
                    latex_preserved += min(source_latex, trans_latex)
        
        latex_rate = (latex_preserved / latex_total * 100) if latex_total > 0 else 100
        
        return {
            "config": config_name,
            "status": "✓ Success",
            "blocks": len(document.blocks),
            "translated": total_translated,
            "chars": total_chars,
            "time": f"{elapsed:.2f}s",
            "latex_preserved": f"{latex_rate:.0f}%",
            "avg_quality": f"{result.bleu_score:.2f}" if result.bleu_score else "N/A",
            "error": None
        }
        
    except Exception as e:
        console.print(f"[red]✗ Error: {e}[/red]")
        return {
            "config": config_name,
            "status": "✗ Failed",
            "blocks": 0,
            "translated": 0,
            "chars": 0,
            "time": "0s",
            "latex_preserved": "0%",
            "avg_quality": "N/A",
            "error": str(e)
        }


def run_tests():
    """Run comprehensive tests."""
    
    console.print(Panel.fit(
        "[bold cyan]SciTrans-LLMs Real PDF Testing[/bold cyan]\n\n"
        "Testing with real scientific papers from arXiv",
        border_style="cyan"
    ))
    
    # Create test directory
    test_dir = Path("test_pdfs")
    test_dir.mkdir(exist_ok=True)
    
    # Download PDFs
    console.print("\n[bold]Step 1: Downloading Test PDFs[/bold]")
    pdfs = []
    
    for paper in ARXIV_PAPERS[:2]:  # Test with 2 papers
        pdf_path = test_dir / f"{paper['id']}.pdf"
        
        if not pdf_path.exists():
            if download_pdf(paper['url'], pdf_path):
                pdfs.append((paper, pdf_path))
        else:
            console.print(f"[yellow]Using cached {pdf_path}[/yellow]")
            pdfs.append((paper, pdf_path))
    
    if not pdfs:
        console.print("[red]No PDFs available for testing[/red]")
        return
    
    # Test configurations
    console.print("\n[bold]Step 2: Testing Configurations[/bold]")
    
    configs = [
        # Backend, Masking, Reranking
        ("cascade", False, False),
        ("cascade", True, False),
        ("free", True, False),
        ("huggingface", True, False),
    ]
    
    all_results = []
    
    for paper, pdf_path in pdfs:
        console.print(f"\n[bold yellow]Testing: {paper['title']}[/bold yellow]")
        
        for backend, masking, reranking in configs:
            result = test_configuration(pdf_path, backend, masking, reranking, max_pages=1)
            result["paper"] = paper['title'][:30]
            all_results.append(result)
    
    # Display results
    console.print("\n[bold]Step 3: Results Summary[/bold]")
    
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Paper", style="dim")
    table.add_column("Config")
    table.add_column("Status")
    table.add_column("Blocks", justify="right")
    table.add_column("Time", justify="right")
    table.add_column("LaTeX", justify="right")
    table.add_column("Quality", justify="right")
    
    for result in all_results:
        status_color = "green" if "✓" in result["status"] else "red"
        table.add_row(
            result["paper"],
            result["config"],
            f"[{status_color}]{result['status']}[/{status_color}]",
            str(result["blocks"]),
            result["time"],
            result["latex_preserved"],
            result["avg_quality"]
        )
    
    console.print(table)
    
    # Error summary
    errors = [r for r in all_results if r["error"]]
    if errors:
        console.print("\n[bold red]Errors Encountered:[/bold red]")
        for err in errors:
            console.print(f"  • {err['config']}: {err['error']}")
    
    # Success summary
    successes = [r for r in all_results if "✓" in r["status"]]
    console.print(f"\n[bold green]Success Rate: {len(successes)}/{len(all_results)} ({len(successes)/len(all_results)*100:.0f}%)[/bold green]")
    
    # Best configuration
    if successes:
        console.print("\n[bold]Best Configurations:[/bold]")
        
        # Fastest
        fastest = min(successes, key=lambda x: float(x["time"].replace("s", "")))
        console.print(f"  [cyan]⚡ Fastest:[/cyan] {fastest['config']} ({fastest['time']})")
        
        # Best LaTeX preservation
        best_latex = max(successes, key=lambda x: float(x["latex_preserved"].replace("%", "")))
        console.print(f"  [cyan]📐 Best LaTeX:[/cyan] {best_latex['config']} ({best_latex['latex_preserved']})")


def quick_test():
    """Quick test with a small sample."""
    console.print("[bold cyan]Quick Test Mode[/bold cyan]\n")
    
    from scitran.translation.backends import (
        CascadeBackend, FreeBackend, HuggingFaceBackend
    )
    from scitran.translation.base import TranslationRequest
    
    test_text = "Machine learning is a powerful technique for artificial intelligence. The equation $E = mc^2$ is fundamental."
    
    backends = [
        ("Cascade", CascadeBackend()),
        ("Free (Google)", FreeBackend()),
        ("HuggingFace", HuggingFaceBackend()),
    ]
    
    table = Table(show_header=True)
    table.add_column("Backend", style="cyan")
    table.add_column("Status")
    table.add_column("Result", max_width=60)
    table.add_column("Time", justify="right")
    
    for name, backend in backends:
        try:
            console.print(f"Testing {name}...")
            request = TranslationRequest(
                text=test_text,
                source_lang="en",
                target_lang="fr"
            )
            
            start = time.time()
            result = backend.translate_sync(request)
            elapsed = time.time() - start
            
            table.add_row(
                name,
                "[green]✓ Success[/green]",
                result.translations[0][:60] + "..." if len(result.translations[0]) > 60 else result.translations[0],
                f"{elapsed:.2f}s"
            )
        except Exception as e:
            table.add_row(
                name,
                "[red]✗ Failed[/red]",
                str(e)[:60],
                "N/A"
            )
    
    console.print(table)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        quick_test()
    else:
        run_tests()
