"""Main CLI interface using Typer."""

import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn

from scitran.core.pipeline import TranslationPipeline, PipelineConfig
from scitran.extraction.pdf_parser import PDFParser
from scitran.rendering.pdf_renderer import PDFRenderer
from .interactive import show_welcome, interactive_translate, show_help, show_backend_details

app = typer.Typer(
    name="scitran",
    help="SciTrans-LLMs: Advanced Scientific Document Translation",
    add_completion=False
)

console = Console()


@app.command()
def translate(
    input_file: Path = typer.Argument(..., help="Input PDF file"),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output file path"),
    source_lang: str = typer.Option("en", "-s", "--source", help="Source language"),
    target_lang: str = typer.Option("fr", "-t", "--target", help="Target language"),
    backend: str = typer.Option("cascade", "-b", "--backend", help="Translation backend (cascade/free/huggingface/ollama/deepseek/openai/anthropic)"),
    model: Optional[str] = typer.Option(None, "-m", "--model", help="Model name (e.g., gpt-4o, claude-3-5-sonnet, llama3.1)"),
    candidates: int = typer.Option(1, "-c", "--candidates", help="Number of candidates"),
    enable_masking: bool = typer.Option(True, "--masking/--no-masking", help="Enable LaTeX masking"),
    enable_reranking: bool = typer.Option(False, "--reranking", help="Enable reranking"),
    max_pages: Optional[int] = typer.Option(None, "--max-pages", help="Max pages to translate (from start-page)"),
    start_page: int = typer.Option(0, "--start-page", help="Start page (0-based, inclusive)"),
    end_page: Optional[int] = typer.Option(None, "--end-page", help="End page (0-based, inclusive)"),
    quality_threshold: float = typer.Option(0.5, "--quality", help="Quality threshold"),
    font_dir: Optional[Path] = typer.Option(None, "--font-dir", help="Path to TTF/OTF fonts to embed"),
    font_files: Optional[str] = typer.Option(None, "--font-files", help="Comma-separated list of TTF/OTF font files to embed"),
    mask_custom_macros: bool = typer.Option(True, "--mask-custom-macros/--no-mask-custom-macros", help="Mask LaTeX custom macros (newcommand/DeclareMathOperator/etc.)"),
    mask_apostrophes_in_latex: bool = typer.Option(True, "--mask-apostrophes-in-latex/--no-mask-apostrophes-in-latex", help="Protect apostrophes inside math"),
):
    """Translate a PDF document."""
    
    if not input_file.exists():
        console.print(f"[red]Error: Input file not found: {input_file}[/red]")
        raise typer.Exit(1)
    
    if output is None:
        output = input_file.with_name(f"{input_file.stem}_translated.pdf")
    
    console.print(f"[bold blue]SciTrans-LLMs Translation[/bold blue]")
    console.print(f"Input: {input_file}")
    console.print(f"Output: {output}")
    console.print(f"Translation: {source_lang} → {target_lang}")
    console.print(f"Backend: {backend}\n")
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=30),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=False
        ) as progress:
            # Parse PDF
            parse_task = progress.add_task("[cyan]Parsing PDF...", total=1)
            parser = PDFParser()
            document = parser.parse(
                str(input_file),
                max_pages=max_pages,
                start_page=start_page,
                end_page=end_page,
            )
            total_blocks = sum(len(seg.blocks) for seg in document.segments)
            progress.update(parse_task, completed=1, description=f"[green]✓ Parsed {total_blocks} blocks")
            
            # Configure pipeline
            config_task = progress.add_task("[cyan]Configuring pipeline...", total=1)
            config = PipelineConfig(
                source_lang=source_lang,
                target_lang=target_lang,
                backend=backend,
                model_name=model,  # Fixed: model -> model_name
                num_candidates=candidates,
                enable_masking=enable_masking,
                enable_reranking=enable_reranking,
                quality_threshold=quality_threshold,
                cache_translations=True,
                mask_custom_macros=mask_custom_macros,
                mask_apostrophes_in_latex=mask_apostrophes_in_latex
            )
            
            # Setup progress callback
            translate_task = progress.add_task("[cyan]Translating...", total=100)
            
            def progress_callback(pct: float, message: str):
                progress.update(translate_task, completed=int(pct * 100), description=f"[cyan]{message}")
            
            pipeline = TranslationPipeline(config, progress_callback=progress_callback)
            progress.update(config_task, completed=1, description="[green]✓ Pipeline configured")
            
            # Translate
            result = pipeline.translate_document(document)
            progress.update(translate_task, completed=100, description="[green]✓ Translation complete")
            
            # Render output
            render_task = progress.add_task("[cyan]Rendering output...", total=1)
            renderer = PDFRenderer(
                font_dir=str(font_dir) if font_dir else None,
                font_files=[f.strip() for f in font_files.split(",")] if font_files else None
            )
            
            if output.suffix == ".pdf":
                renderer.render_with_layout(str(input_file), result.document, str(output))
            elif output.suffix == ".txt":
                renderer.render_text(result.document, str(output))
            elif output.suffix == ".md":
                renderer.render_markdown(result.document, str(output))
            else:
                renderer.render_simple(result.document, str(output))
            
            progress.update(render_task, completed=1, description=f"[green]✓ Output saved")
        
        # Show statistics
        console.print("\n[bold green]Translation Complete![/bold green]")
        console.print(f"Blocks translated: {result.blocks_translated}")
        console.print(f"Time taken: {result.duration:.2f}s")
        if result.bleu_score:
            console.print(f"BLEU score: {result.bleu_score:.2f}")
        if result.glossary_adherence:
            console.print(f"Glossary adherence: {result.glossary_adherence:.0%}")
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def info(
    input_file: Path = typer.Argument(..., help="PDF file to analyze")
):
    """Show information about a PDF document."""
    
    if not input_file.exists():
        console.print(f"[red]Error: File not found: {input_file}[/red]")
        raise typer.Exit(1)
    
    try:
        parser = PDFParser()
        metadata = parser.extract_metadata(str(input_file))
        
        console.print(f"\n[bold]PDF Information[/bold]")
        console.print(f"File: {input_file}")
        console.print(f"\n[bold]Metadata:[/bold]")
        for key, value in metadata.items():
            if value:
                console.print(f"  {key}: {value}")
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def backends(detailed: bool = typer.Option(False, "--detailed", "-d", help="Show detailed information")):
    """List available translation backends."""
    
    if detailed:
        show_backend_details()
        return
    
    from scitran.translation.backends import (
        OpenAIBackend, AnthropicBackend, DeepSeekBackend,
        OllamaBackend, FreeBackend, CascadeBackend, HuggingFaceBackend
    )
    
    console.print("\n[bold]Available Translation Backends[/bold]\n")
    
    backends_list = [
        ("cascade", CascadeBackend, "multi-service"),
        ("free", FreeBackend, "google"),
        ("huggingface", HuggingFaceBackend, "facebook/mbart-large-50-many-to-many-mmt"),
        ("ollama", OllamaBackend, "llama3.1"),
        ("deepseek", DeepSeekBackend, "deepseek-chat"),
        ("openai", OpenAIBackend, "gpt-4o"),
        ("anthropic", AnthropicBackend, "claude-3-5-sonnet-20241022")
    ]
    
    for name, backend_class, default_model in backends_list:
        try:
            backend = backend_class(model=default_model)
            status = "✓ Available" if backend.is_available() else "✗ Not configured"
            color = "green" if backend.is_available() else "yellow"
            console.print(f"[{color}]{status}[/{color}] {name} ({default_model})")
        except Exception as e:
            console.print(f"[red]✗ Error[/red] {name}: {str(e)}")
    
    console.print("\n[dim]💡 Use --detailed for more information[/dim]")


@app.command()
def wizard():
    """Interactive translation wizard."""
    params = interactive_translate()
    
    if params is None:
        return
    
    # Execute translation with wizard params
    translate(
        input_file=Path(params["input_file"]),
        output=Path(params["output_file"]),
        source_lang=params["source_lang"],
        target_lang=params["target_lang"],
        backend=params["backend"],
        model=None,
        candidates=3 if params["enable_reranking"] else 1,
        enable_masking=params["enable_masking"],
        enable_reranking=params["enable_reranking"],
        max_pages=None,
        quality_threshold=0.5
    )


@app.command()
def help():
    """Show comprehensive help and usage guide."""
    show_help()


@app.command()
def test(
    backend: str = typer.Option("cascade", "--backend", "-b", help="Backend to test"),
    sample: str = typer.Option("Machine learning enables artificial intelligence.", "--sample", "-s", help="Sample text")
):
    """Test translation with sample text."""
    
    console.print(f"\n[bold cyan]Testing {backend} backend[/bold cyan]\n")
    console.print(f"Sample text: {sample}\n")
    
    try:
        from scitran.core.models import Block
        from scitran.translation.base import TranslationRequest
        
        # Get backend
        from scitran.translation.backends import (
            OpenAIBackend, AnthropicBackend, DeepSeekBackend,
            OllamaBackend, FreeBackend, CascadeBackend, HuggingFaceBackend
        )
        
        backend_map = {
            "openai": OpenAIBackend,
            "anthropic": AnthropicBackend,
            "deepseek": DeepSeekBackend,
            "ollama": OllamaBackend,
            "free": FreeBackend,
            "cascade": CascadeBackend,
            "huggingface": HuggingFaceBackend
        }
        
        if backend not in backend_map:
            console.print(f"[red]Unknown backend: {backend}[/red]")
            raise typer.Exit(1)
        
        backend_instance = backend_map[backend]()
        
        if not backend_instance.is_available():
            console.print(f"[yellow]Warning: {backend} is not configured[/yellow]")
        
        # Translate
        request = TranslationRequest(
            text=sample,
            source_lang="en",
            target_lang="fr"
        )
        
        import time
        start = time.time()
        result = backend_instance.translate_sync(request)
        elapsed = time.time() - start
        
        console.print(f"[green]✓ Translation successful![/green]")
        console.print(f"\nResult: {result.translations[0] if result.translations else 'N/A'}")
        console.print(f"Time: {elapsed:.2f}s")
        console.print(f"Backend: {result.backend}")
        console.print(f"Model: {result.model}")
        
        if result.metadata:
            console.print(f"\nMetadata:")
            for key, value in result.metadata.items():
                console.print(f"  {key}: {value}")
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def glossary(
    action: str = typer.Argument(..., help="Action: list, export, stats"),
    filepath: Optional[Path] = typer.Option(None, "--file", "-f", help="File path for export/import")
):
    """Manage translation glossaries."""
    
    if action == "list":
        console.print("\n[bold]Available Glossaries[/bold]\n")
        from pathlib import Path
        glossary_dir = Path("configs")
        glossaries = list(glossary_dir.glob("glossary_*.yaml"))
        
        for g in glossaries:
            console.print(f"  • {g.name}")
        
        console.print(f"\n[dim]Found {len(glossaries)} glossaries[/dim]")
    
    elif action == "export":
        if not filepath:
            filepath = Path("learned_glossary.yaml")
        
        console.print(f"[yellow]Exporting learned glossary to {filepath}...[/yellow]")
        
        from scitran.translation.backends import CascadeBackend
        backend = CascadeBackend()
        backend.export_glossary(str(filepath))
        
        console.print(f"[green]✓ Glossary exported successfully![/green]")
    
    elif action == "stats":
        from scitran.translation.backends import CascadeBackend
        backend = CascadeBackend()
        stats = backend.get_statistics()
        
        console.print("\n[bold]Cascade Backend Statistics[/bold]\n")
        for key, value in stats.items():
            console.print(f"  {key}: {value}")
    
    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        console.print("Valid actions: list, export, stats")
        raise typer.Exit(1)


@app.command()
def gui():
    """Launch the Gradio GUI."""
    console.print("[bold blue]Launching SciTrans LLMs GUI...[/bold blue]")
    console.print("Opening at http://localhost:7860")
    
    try:
        from gui.app import launch
        launch()
    except ImportError as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        console.print("Install GUI dependencies: pip install gradio")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error launching GUI: {str(e)}[/red]")
        raise typer.Exit(1)


def cli():
    """Main CLI entry point."""
    # Show welcome banner if no args
    import sys
    if len(sys.argv) == 1:
        show_welcome()
        console.print("\n[dim]Type './scitrans --help' for usage information[/dim]")
        console.print("[dim]Or run './scitrans wizard' for interactive mode[/dim]\n")
        return
    
    app()


if __name__ == "__main__":
    cli()
