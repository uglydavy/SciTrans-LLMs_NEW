"""Main CLI interface using Typer."""

import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn

from scitran.core.pipeline import TranslationPipeline, PipelineConfig
from scitran.extraction.pdf_parser import PDFParser
from scitran.rendering.pdf_renderer import PDFRenderer
from scitran.evaluation.block_scorer import BlockScorer, DocumentScoreReport
from scitran.translation.glossary.manager import GlossaryManager
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
    backend: str = typer.Option("deepseek", "-b", "--backend", help="Translation backend (deepseek/cascade/free/huggingface/ollama/openai/anthropic)"),
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
    fast_mode: bool = typer.Option(False, "--fast/--no-fast", help="Fast mode: optimize for speed (single candidate, no reranking)"),
    overflow_strategy: str = typer.Option("shrink", "--overflow-strategy", help="PDF overflow handling: shrink/expand/append_pages/marker+append_pages"),
    debug_mode: bool = typer.Option(False, "--debug/--no-debug", help="Enable debug logging"),
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
            
            # Load API key for the selected backend
            api_key = _load_api_key_for_backend(backend)
            
            config = PipelineConfig(
                source_lang=source_lang,
                target_lang=target_lang,
                backend=backend,
                model_name=model,  # Fixed: model -> model_name
                api_key=api_key,  # Load API key from config/env
                num_candidates=candidates,
                enable_masking=enable_masking,
                enable_reranking=enable_reranking,
                quality_threshold=quality_threshold,
                cache_translations=True,
                mask_custom_macros=mask_custom_macros,
                mask_apostrophes_in_latex=mask_apostrophes_in_latex,
                fast_mode=fast_mode,  # PHASE 1.3
                debug_mode=debug_mode,  # PHASE 4.1
                debug_log_path=Path(".cache/scitrans/debug.jsonl") if debug_mode else None
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
                font_files=[f.strip() for f in font_files.split(",")] if font_files else None,
                overflow_strategy=overflow_strategy  # PHASE 3.2
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
        
        # PHASE 4.2: Show comprehensive run summary
        pipeline.print_run_summary(result)
        
        # Show scoring report if available
        if result.score_report:
            _display_score_report(result.score_report)
        
        # Also show brief stats
        console.print("\n[bold green]Translation Complete![/bold green]")
        console.print(f"Output: {output}")
        if result.glossary_adherence:
            console.print(f"Glossary adherence: {result.glossary_adherence:.0%}")
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def info(
    input_file: Optional[Path] = typer.Argument(None, help="PDF file to analyze (optional, shows system info if omitted)")
):
    """Show system information or PDF document information."""
    
    if input_file is None:
        # Show system info
        show_system_info()
        return
    
    # Show PDF info
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
def score(
    input_file: Path = typer.Argument(..., help="Translated PDF or JSON document file"),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output file for score report (JSON)"),
    glossary_domains: Optional[str] = typer.Option(None, "--glossary-domains", help="Comma-separated glossary domains"),
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Show detailed per-block scores")
):
    """Score and evaluate a translated document."""
    
    if not input_file.exists():
        console.print(f"[red]Error: File not found: {input_file}[/red]")
        raise typer.Exit(1)
    
    try:
        from scitran.core.models import Document
        
        # Load document
        if input_file.suffix == ".json":
            # Load from JSON
            with open(input_file, 'r', encoding='utf-8') as f:
                import json
                doc_data = json.load(f)
                document = Document.from_json(json.dumps(doc_data))
        else:
            # Parse PDF and assume it's already translated (would need translation metadata)
            console.print("[yellow]Warning: PDF scoring requires translation metadata.[/yellow]")
            console.print("[yellow]Please use JSON document format or run scoring after translation.[/yellow]")
            raise typer.Exit(1)
        
        # Load glossary if specified
        glossary_manager = None
        if glossary_domains:
            glossary_manager = GlossaryManager()
            for domain in glossary_domains.split(','):
                glossary_manager.load_domain(domain.strip(), "en-fr")
        
        # Score document
        console.print("\n[cyan]Scoring translated document...[/cyan]")
        scorer = BlockScorer(glossary_manager=glossary_manager)
        score_report = scorer.score_document(document)
        
        # Display report
        _display_score_report(score_report, detailed=detailed)
        
        # Save report if output specified
        if output:
            import json
            from datetime import datetime
            
            report_dict = {
                "document_id": score_report.document_id,
                "timestamp": score_report.timestamp.isoformat(),
                "total_blocks": score_report.total_blocks,
                "translated_blocks": score_report.translated_blocks,
                "failed_blocks": score_report.failed_blocks,
                "average_scores": {
                    "fluency": score_report.avg_fluency,
                    "adequacy": score_report.avg_adequacy,
                    "glossary": score_report.avg_glossary,
                    "numeric": score_report.avg_numeric,
                    "format": score_report.avg_format,
                    "overall": score_report.avg_overall
                },
                "quality_distribution": score_report.score_distribution,
                "issues": score_report.issues,
                "block_scores": [
                    {
                        "block_id": bs.block_id,
                        "overall_score": bs.overall_score,
                        "fluency": bs.fluency_score,
                        "adequacy": bs.adequacy_score,
                        "glossary": bs.glossary_score,
                        "numeric": bs.numeric_consistency,
                        "format": bs.format_preservation
                    }
                    for bs in score_report.block_scores
                ]
            }
            
            with open(output, 'w', encoding='utf-8') as f:
                json.dump(report_dict, f, indent=2, ensure_ascii=False)
            
            console.print(f"\n[green]✓ Score report saved to {output}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        import traceback
        if detailed:
            console.print(traceback.format_exc())
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
    
    # Load API keys from config file
    import os
    from pathlib import Path
    from scitran.utils.config_loader import load_config
    
    config_path = Path.home() / ".scitrans" / "config.yaml"
    api_keys = {}
    if config_path.exists():
        try:
            config = load_config(str(config_path))
            api_keys = config.get("api_keys", {})
        except:
            pass
    
    # Also check environment variables (they take precedence)
    env_mappings = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "huggingface": "HUGGINGFACE_API_KEY"
    }
    
    console.print("\n[bold]Available Translation Backends[/bold]\n")
    
    backends_list = [
        ("cascade", CascadeBackend, "multi-service", None),
        ("free", FreeBackend, "google", None),
        ("huggingface", HuggingFaceBackend, "facebook/mbart-large-50-many-to-many-mmt", "huggingface"),
        ("ollama", OllamaBackend, "llama3.1", None),
        ("deepseek", DeepSeekBackend, "deepseek-chat", "deepseek"),
        ("openai", OpenAIBackend, "gpt-4o", "openai"),
        ("anthropic", AnthropicBackend, "claude-3-5-sonnet-20241022", "anthropic")
    ]
    
    for name, backend_class, default_model, key_name in backends_list:
        try:
            # Get API key from environment (preferred) or config file
            api_key = None
            if key_name:
                # Check environment variable first
                if key_name in env_mappings:
                    api_key = os.getenv(env_mappings[key_name])
                # Fall back to config file
                if not api_key and key_name in api_keys:
                    api_key = api_keys[key_name]
            
            backend = backend_class(api_key=api_key, model=default_model)
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


def show_system_info():
    """Show system information including API key status."""
    import os
    import sys
    import platform
    from pathlib import Path
    from rich.table import Table
    
    console.print("\n[bold cyan]═══ SciTrans-LLMs System Information ═══[/bold cyan]\n")
    
    # System info
    info_table = Table(show_header=False, box=None)
    info_table.add_row("[bold]Python Version:[/bold]", f"{sys.version.split()[0]} ({sys.executable})")
    info_table.add_row("[bold]Platform:[/bold]", f"{platform.system()} {platform.release()}")
    info_table.add_row("[bold]Architecture:[/bold]", platform.machine())
    console.print(info_table)
    
    # API Keys status
    console.print("\n[bold]API Keys Status:[/bold]\n")
    
    from scitran.translation.backends import (
        OpenAIBackend, AnthropicBackend, DeepSeekBackend,
        OllamaBackend, FreeBackend, CascadeBackend, HuggingFaceBackend
    )
    
    api_keys_table = Table(show_header=True, header_style="bold cyan")
    api_keys_table.add_column("Backend", style="cyan")
    api_keys_table.add_column("Status", style="green")
    api_keys_table.add_column("Source", style="dim")
    
    backends_info = [
        ("openai", OpenAIBackend, "OPENAI_API_KEY", "gpt-4o"),
        ("anthropic", AnthropicBackend, "ANTHROPIC_API_KEY", "claude-3-5-sonnet-20241022"),
        ("deepseek", DeepSeekBackend, "DEEPSEEK_API_KEY", "deepseek-chat"),
        ("ollama", OllamaBackend, None, "llama3.1"),
        ("free", FreeBackend, None, "google"),
        ("cascade", CascadeBackend, None, "multi-service"),
        ("huggingface", HuggingFaceBackend, "HUGGINGFACE_API_KEY", "facebook/mbart-large-50-many-to-many-mmt"),
    ]
    
    # Check config file
    config_path = Path.home() / ".scitrans" / "config.yaml"
    config_keys = {}
    if config_path.exists():
        try:
            from scitran.utils.config_loader import load_config
            config = load_config(str(config_path))
            config_keys = config.get("api_keys", {})
        except:
            pass
    
    for name, backend_class, env_var, model in backends_info:
        try:
            backend = backend_class(model=model)
            is_available = backend.is_available()
            
            if is_available:
                status = "✓ Configured"
                status_style = "green"
            else:
                status = "✗ Not configured"
                status_style = "yellow"
            
            # Determine source
            source = "N/A (no key needed)"
            if env_var:
                env_value = os.getenv(env_var)
                if env_value:
                    source = f"Environment ({env_var})"
                elif name in config_keys and config_keys[name]:
                    source = f"Config file (~/.scitrans/config.yaml)"
                else:
                    source = "Not set"
            
            api_keys_table.add_row(name, f"[{status_style}]{status}[/{status_style}]", source)
        except Exception as e:
            api_keys_table.add_row(name, f"[red]Error[/red]", str(e)[:50])
    
    console.print(api_keys_table)
    
    # Configuration paths
    console.print("\n[bold]Configuration:[/bold]\n")
    config_table = Table(show_header=False, box=None)
    config_table.add_row("[bold]Config file:[/bold]", str(config_path))
    config_table.add_row("[bold]Cache directory:[/bold]", str(Path.home() / ".scitrans" / "cache"))
    console.print(config_table)
    
    console.print("\n[dim]💡 Use 'scitrans keys set <backend> <key>' to set API keys[/dim]")
    console.print("[dim]💡 Or set environment variables: OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.[/dim]\n")


@app.command()
def keys(
    action: str = typer.Argument(..., help="Action: list, set, delete, check"),
    backend: Optional[str] = typer.Option(None, "--backend", "-b", help="Backend name (for set/delete)"),
    key: Optional[str] = typer.Option(None, "--key", "-k", help="API key value (for set)")
):
    """Manage API keys for translation backends."""
    import os
    from pathlib import Path
    from scitran.utils.config_loader import load_config, save_config
    
    config_path = Path.home() / ".scitrans" / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing config or create default
    if config_path.exists():
        try:
            config = load_config(str(config_path))
        except:
            from scitran.utils.config_loader import get_default_config
            config = get_default_config()
    else:
        from scitran.utils.config_loader import get_default_config
        config = get_default_config()
    
    if "api_keys" not in config:
        config["api_keys"] = {}
    
    env_mappings = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "huggingface": "HUGGINGFACE_API_KEY"
    }
    
    if action == "list":
        console.print("\n[bold]Configured API Keys:[/bold]\n")
        from rich.table import Table
        keys_table = Table(show_header=True, header_style="bold cyan")
        keys_table.add_column("Backend", style="cyan")
        keys_table.add_column("Status", style="green")
        keys_table.add_column("Source", style="dim")
        
        for backend_name, env_var in env_mappings.items():
            env_value = os.getenv(env_var)
            config_value = config.get("api_keys", {}).get(backend_name, "")
            
            if env_value:
                status = "✓ Set (env)"
                source = f"Environment: {env_var}"
                masked_key = f"{env_value[:8]}...{env_value[-4:]}" if len(env_value) > 12 else "***"
            elif config_value:
                status = "✓ Set (config)"
                source = "Config file"
                masked_key = f"{config_value[:8]}...{config_value[-4:]}" if len(config_value) > 12 else "***"
            else:
                status = "✗ Not set"
                source = "Not configured"
                masked_key = ""
            
            keys_table.add_row(backend_name, status, source)
            if masked_key:
                keys_table.add_row("", f"  Key: {masked_key}", "")
        
        console.print(keys_table)
        console.print(f"\n[dim]Config file: {config_path}[/dim]\n")
    
    elif action == "set":
        if not backend:
            console.print("[red]Error: --backend is required for 'set' action[/red]")
            console.print("Usage: scitrans keys set --backend <name> --key <api_key>")
            raise typer.Exit(1)
        
        if not key:
            console.print("[red]Error: --key is required for 'set' action[/red]")
            console.print("Usage: scitrans keys set --backend <name> --key <api_key>")
            raise typer.Exit(1)
        
        if backend not in env_mappings:
            console.print(f"[yellow]Warning: {backend} is not a known backend that requires API keys[/yellow]")
            console.print(f"Known backends: {', '.join(env_mappings.keys())}")
        
        # Save to config
        config["api_keys"][backend] = key.strip()
        save_config(config, str(config_path))
        
        # Also set environment variable for current session
        if backend in env_mappings:
            os.environ[env_mappings[backend]] = key.strip()
        
        masked_key = f"{key[:8]}...{key[-4:]}" if len(key) > 12 else "***"
        console.print(f"[green]✓ API key saved for {backend}[/green]")
        console.print(f"  Key: {masked_key}")
        console.print(f"  Saved to: {config_path}")
        if backend in env_mappings:
            console.print(f"  Environment variable {env_mappings[backend]} set for current session")
    
    elif action == "delete":
        if not backend:
            console.print("[red]Error: --backend is required for 'delete' action[/red]")
            console.print("Usage: scitrans keys delete --backend <name>")
            raise typer.Exit(1)
        
        if backend in config.get("api_keys", {}):
            del config["api_keys"][backend]
            save_config(config, str(config_path))
            console.print(f"[green]✓ API key deleted for {backend}[/green]")
        else:
            console.print(f"[yellow]No API key found for {backend} in config file[/yellow]")
        
        # Also unset environment variable
        if backend in env_mappings:
            if env_mappings[backend] in os.environ:
                del os.environ[env_mappings[backend]]
                console.print(f"  Environment variable {env_mappings[backend]} unset")
    
    elif action == "check":
        console.print("\n[bold]Checking API Key Availability:[/bold]\n")
        
        from scitran.translation.backends import (
            OpenAIBackend, AnthropicBackend, DeepSeekBackend,
            OllamaBackend, FreeBackend, CascadeBackend, HuggingFaceBackend
        )
        
        backend_map = {
            "openai": (OpenAIBackend, "gpt-4o"),
            "anthropic": (AnthropicBackend, "claude-3-5-sonnet-20241022"),
            "deepseek": (DeepSeekBackend, "deepseek-chat"),
            "ollama": (OllamaBackend, "llama3.1"),
            "free": (FreeBackend, "google"),
            "cascade": (CascadeBackend, "multi-service"),
            "huggingface": (HuggingFaceBackend, "facebook/mbart-large-50-many-to-many-mmt"),
        }
        
        for name, (backend_class, model) in backend_map.items():
            try:
                backend = backend_class(model=model)
                is_available = backend.is_available()
                status = "✓ Available" if is_available else "✗ Not configured"
                color = "green" if is_available else "yellow"
                console.print(f"[{color}]{status}[/{color}] {name}")
            except Exception as e:
                console.print(f"[red]✗ Error[/red] {name}: {str(e)}")
        
        console.print()
    
    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        console.print("Valid actions: list, set, delete, check")
        raise typer.Exit(1)


def _load_api_key_for_backend(backend: str) -> Optional[str]:
    """Load API key for a backend from environment or config file.
    
    Args:
        backend: Backend name
        
    Returns:
        API key if found, None otherwise
    """
    import os
    from pathlib import Path
    
    # Environment variable mappings
    env_mappings = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "huggingface": "HUGGINGFACE_API_KEY"
    }
    
    backend_lower = backend.lower()
    
    # Check environment variable first
    if backend_lower in env_mappings:
        api_key = os.getenv(env_mappings[backend_lower])
        if api_key:
            return api_key
    
    # Check config file
    config_path = Path.home() / ".scitrans" / "config.yaml"
    if config_path.exists():
        try:
            from scitran.utils.config_loader import load_config
            config = load_config(str(config_path))
            api_keys = config.get("api_keys", {})
            if backend_lower in api_keys:
                return api_keys[backend_lower]
        except Exception:
            pass
    
    return None


def _display_score_report(score_report: DocumentScoreReport, detailed: bool = False):
    """Display scoring report in a formatted way."""
    from rich.table import Table
    from rich.panel import Panel
    
    # Summary panel
    summary_table = Table(title="Translation Quality Summary", show_header=True, header_style="bold cyan")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Score", justify="right", style="green")
    
    summary_table.add_row("Total Blocks", str(score_report.total_blocks))
    summary_table.add_row("Translated Blocks", str(score_report.translated_blocks))
    summary_table.add_row("Failed Blocks", str(score_report.failed_blocks))
    summary_table.add_row("", "")  # Separator
    summary_table.add_row("Average Fluency", f"{score_report.avg_fluency:.2%}")
    summary_table.add_row("Average Adequacy", f"{score_report.avg_adequacy:.2%}")
    summary_table.add_row("Average Glossary", f"{score_report.avg_glossary:.2%}")
    summary_table.add_row("Average Numeric", f"{score_report.avg_numeric:.2%}")
    summary_table.add_row("Average Format", f"{score_report.avg_format:.2%}")
    summary_table.add_row("", "")  # Separator
    summary_table.add_row("[bold]Overall Score[/bold]", f"[bold]{score_report.avg_overall:.2%}[/bold]")
    
    console.print("\n")
    console.print(summary_table)
    
    # Quality distribution
    dist_table = Table(title="Quality Distribution", show_header=True, header_style="bold cyan")
    dist_table.add_column("Quality Tier", style="cyan")
    dist_table.add_column("Count", justify="right", style="green")
    dist_table.add_column("Percentage", justify="right", style="yellow")
    
    total_translated = score_report.translated_blocks or 1
    dist_table.add_row(
        "[green]High (≥0.8)[/green]",
        str(score_report.high_quality_blocks),
        f"{score_report.high_quality_blocks / total_translated:.1%}"
    )
    dist_table.add_row(
        "[yellow]Medium (0.5-0.8)[/yellow]",
        str(score_report.medium_quality_blocks),
        f"{score_report.medium_quality_blocks / total_translated:.1%}"
    )
    dist_table.add_row(
        "[red]Low (<0.5)[/red]",
        str(score_report.low_quality_blocks),
        f"{score_report.low_quality_blocks / total_translated:.1%}"
    )
    
    console.print("\n")
    console.print(dist_table)
    
    # Issues
    if score_report.issues:
        issues_table = Table(title="Issues Detected", show_header=True, header_style="bold red")
        issues_table.add_column("Severity", style="red")
        issues_table.add_column("Type", style="yellow")
        issues_table.add_column("Message", style="white")
        
        for issue in score_report.issues[:10]:  # Show first 10 issues
            severity_style = {
                "high": "[bold red]HIGH[/bold red]",
                "medium": "[yellow]MEDIUM[/yellow]",
                "low": "[dim]LOW[/dim]"
            }.get(issue.get("severity", "medium"), "[yellow]MEDIUM[/yellow]")
            
            issues_table.add_row(
                severity_style,
                issue.get("type", "unknown"),
                issue.get("message", "")[:80]
            )
        
        if len(score_report.issues) > 10:
            issues_table.add_row("", "", f"... and {len(score_report.issues) - 10} more issues")
        
        console.print("\n")
        console.print(issues_table)
    
    # Detailed per-block scores
    if detailed and score_report.block_scores:
        blocks_table = Table(title="Per-Block Scores", show_header=True, header_style="bold cyan")
        blocks_table.add_column("Block ID", style="cyan")
        blocks_table.add_column("Overall", justify="right", style="green")
        blocks_table.add_column("Fluency", justify="right")
        blocks_table.add_column("Adequacy", justify="right")
        blocks_table.add_column("Glossary", justify="right")
        blocks_table.add_column("Numeric", justify="right")
        blocks_table.add_column("Format", justify="right")
        
        # Sort by overall score (worst first)
        sorted_blocks = sorted(score_report.block_scores, key=lambda bs: bs.overall_score)
        
        for bs in sorted_blocks[:20]:  # Show worst 20 blocks
            overall_style = (
                "[green]" if bs.overall_score >= 0.8 else
                "[yellow]" if bs.overall_score >= 0.5 else
                "[red]"
            )
            
            blocks_table.add_row(
                bs.block_id[:20],
                f"{overall_style}{bs.overall_score:.2f}[/{overall_style.split('[')[1].split(']')[0]}]",
                f"{bs.fluency_score:.2f}",
                f"{bs.adequacy_score:.2f}",
                f"{bs.glossary_score:.2f}",
                f"{bs.numeric_consistency:.2f}",
                f"{bs.format_preservation:.2f}"
            )
        
        if len(sorted_blocks) > 20:
            blocks_table.add_row("", "", "", "", "", "", f"... and {len(sorted_blocks) - 20} more blocks")
        
        console.print("\n")
        console.print(blocks_table)


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
