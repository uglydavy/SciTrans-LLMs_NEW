"""Interactive CLI features."""

import typer
from pathlib import Path
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.panel import Panel
from typing import Optional

console = Console()


def show_welcome():
    """Show welcome banner."""
    banner = """
[bold blue]╔══════════════════════════════════════════════════════════════╗[/bold blue]
[bold blue]║[/bold blue]          [bold cyan]SciTrans-LLMs[/bold cyan] - Scientific Document Translation    [bold blue]║[/bold blue]
[bold blue]╚══════════════════════════════════════════════════════════════╝[/bold blue]

[yellow]Features:[/yellow]
  ✓ 6 Translation backends (OpenAI, Anthropic, DeepSeek, Ollama, Free, Cascade)
  ✓ LaTeX preservation with advanced masking
  ✓ Multi-candidate translation with reranking
  ✓ Layout-preserving PDF rendering
  ✓ Glossary learning and caching

[green]Quick Start:[/green]
  ./scitrans translate paper.pdf              # Translate PDF
  ./scitrans backends                         # List available backends
  ./scitrans wizard                           # Interactive wizard
  ./scitrans help                             # Show detailed help
"""
    console.print(banner)


def interactive_translate():
    """Interactive translation wizard."""
    show_welcome()
    
    console.print("\n[bold cyan]═══ Interactive Translation Wizard ═══[/bold cyan]\n")
    
    # Get input file
    input_file = Prompt.ask(
        "[yellow]📄 Input PDF file path[/yellow]",
        default="paper.pdf"
    )
    
    if not Path(input_file).exists():
        console.print(f"[red]✗ File not found: {input_file}[/red]")
        return None
    
    # Get languages
    source_lang = Prompt.ask(
        "[yellow]🌍 Source language[/yellow]",
        default="en",
        choices=["en", "fr", "de", "es", "it", "pt", "zh", "ja", "ko"]
    )
    
    target_lang = Prompt.ask(
        "[yellow]🌍 Target language[/yellow]",
        default="fr",
        choices=["en", "fr", "de", "es", "it", "pt", "zh", "ja", "ko"]
    )
    
    # Get backend
    console.print("\n[cyan]Available backends:[/cyan]")
    backends_table = Table(show_header=True)
    backends_table.add_column("Backend", style="cyan")
    backends_table.add_column("Cost", style="yellow")
    backends_table.add_column("Quality", style="green")
    backends_table.add_column("Speed", style="blue")
    
    backends_table.add_row("cascade", "FREE", "Medium", "Fast")
    backends_table.add_row("free", "FREE", "Medium", "Fast")
    backends_table.add_row("huggingface", "FREE", "Good", "Medium")
    backends_table.add_row("ollama", "FREE (local)", "Good", "Medium")
    backends_table.add_row("deepseek", "$$$", "Good", "Fast")
    backends_table.add_row("openai", "$$$$", "Excellent", "Fast")
    backends_table.add_row("anthropic", "$$$$$", "Excellent", "Medium")
    
    console.print(backends_table)
    
    backend = Prompt.ask(
        "\n[yellow]🔧 Translation backend[/yellow]",
        default="cascade",
        choices=["cascade", "free", "huggingface", "ollama", "deepseek", "openai", "anthropic"]
    )
    
    # Advanced options
    enable_masking = Confirm.ask(
        "[yellow]🎭 Enable LaTeX masking?[/yellow]",
        default=True
    )
    
    enable_reranking = Confirm.ask(
        "[yellow]🏆 Enable multi-candidate reranking?[/yellow]",
        default=False
    )
    
    # Output options
    console.print("\n[cyan]Output formats:[/cyan]")
    console.print("  1. PDF (layout preserved)")
    console.print("  2. PDF (simple)")
    console.print("  3. Markdown")
    console.print("  4. Plain text")
    
    format_choice = Prompt.ask(
        "[yellow]📝 Output format[/yellow]",
        default="1",
        choices=["1", "2", "3", "4"]
    )
    
    format_map = {
        "1": ("pdf", "layout"),
        "2": ("pdf", "simple"),
        "3": ("md", "markdown"),
        "4": ("txt", "text")
    }
    
    ext, render_mode = format_map[format_choice]
    output_file = Prompt.ask(
        "[yellow]💾 Output file path[/yellow]",
        default=f"{Path(input_file).stem}_translated.{ext}"
    )
    
    # Summary
    console.print("\n[bold green]═══ Translation Summary ═══[/bold green]")
    summary = Table(show_header=False, box=None)
    summary.add_row("Input:", input_file)
    summary.add_row("Output:", output_file)
    summary.add_row("Languages:", f"{source_lang} → {target_lang}")
    summary.add_row("Backend:", backend)
    summary.add_row("Masking:", "✓" if enable_masking else "✗")
    summary.add_row("Reranking:", "✓" if enable_reranking else "✗")
    summary.add_row("Format:", render_mode)
    console.print(summary)
    
    if not Confirm.ask("\n[yellow]🚀 Start translation?[/yellow]", default=True):
        console.print("[red]Cancelled[/red]")
        return None
    
    return {
        "input_file": input_file,
        "output_file": output_file,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "backend": backend,
        "enable_masking": enable_masking,
        "enable_reranking": enable_reranking,
        "render_mode": render_mode
    }


def show_help():
    """Show comprehensive help."""
    help_text = """
[bold cyan]SciTrans-LLMs - Complete Usage Guide[/bold cyan]

[yellow]COMMANDS:[/yellow]

  [bold]translate[/bold] <file>        Translate a PDF document
    Options:
      -o, --output PATH      Output file path
      -s, --source LANG      Source language (default: en)
      -t, --target LANG      Target language (default: fr)
      -b, --backend NAME     Translation backend
      --masking              Enable LaTeX masking
      --reranking            Enable reranking
      --candidates N         Number of translation candidates
    
    Example:
      ./scitrans translate paper.pdf -o output.pdf -b openai --masking

  [bold]wizard[/bold]                  Interactive translation wizard
    
    Example:
      ./scitrans wizard

  [bold]backends[/bold]                List available translation backends
    
    Example:
      ./scitrans backends

  [bold]info[/bold] <file>             Show PDF document information
    
    Example:
      ./scitrans info paper.pdf

  [bold]test[/bold]                    Test translation quality
    Options:
      --backend NAME         Backend to test
      --sample TEXT          Sample text to translate
    
    Example:
      ./scitrans test --backend cascade --sample "Hello world"

  [bold]glossary[/bold]                Manage translation glossaries
    Subcommands:
      list                   List available glossaries
      export                 Export learned glossary
      import PATH            Import glossary from file
    
    Example:
      ./scitrans glossary export learned.yaml

  [bold]gui[/bold]                     Launch web interface
    
    Example:
      ./scitrans gui

[yellow]BACKENDS:[/yellow]

  [bold cyan]cascade[/bold]   - Smart fallback: Lingva→LibreTranslate→MyMemory (FREE, learns glossary)
  [bold cyan]free[/bold]      - Google Translate via deep-translator (FREE)
  [bold cyan]ollama[/bold]    - Local translation with Ollama (FREE, offline)
  [bold cyan]deepseek[/bold]  - DeepSeek API (affordable, good quality)
  [bold cyan]openai[/bold]    - OpenAI GPT-4 (expensive, best quality)
  [bold cyan]anthropic[/bold] - Anthropic Claude (expensive, long context)

[yellow]FEATURES:[/yellow]

  🎭 [bold]LaTeX Masking[/bold]
     Automatically protects equations, code, URLs, citations
     
  🏆 [bold]Multi-Candidate Reranking[/bold]
     Generates multiple translations, selects best quality
     
  📐 [bold]Layout Preservation[/bold]
     Maintains original PDF layout and formatting
     
  🧠 [bold]Glossary Learning[/bold]
     Cascade backend learns technical terms automatically
     
  💾 [bold]Smart Caching[/bold]
     Saves translations to avoid repeat API calls

[yellow]CONFIGURATION:[/yellow]

  Set API keys (if using paid backends):
    export OPENAI_API_KEY="sk-..."
    export ANTHROPIC_API_KEY="sk-ant-..."
    export DEEPSEEK_API_KEY="sk-..."
  
  Or create configs/keys.yaml:
    api_keys:
      openai: "sk-..."
      anthropic: "sk-ant-..."

[yellow]EXAMPLES:[/yellow]

  # Translate with free cascade backend
  ./scitrans translate paper.pdf -b cascade

  # High-quality translation with OpenAI
  ./scitrans translate paper.pdf -b openai --masking --reranking

  # Offline translation
  ./scitrans translate paper.pdf -b ollama

  # Interactive mode
  ./scitrans wizard

  # Check what backends are configured
  ./scitrans backends

[yellow]SUPPORT:[/yellow]

  Documentation: docs/QUICKSTART.md
  Examples: examples/basic_usage.py
  Issues: GitHub repository
"""
    console.print(Panel(help_text, title="[bold]SciTrans Help[/bold]", border_style="cyan"))


def show_backend_details():
    """Show detailed backend information."""
    console.print("\n[bold cyan]═══ Translation Backends ═══[/bold cyan]\n")
    
    from scitran.translation.backends import (
        OpenAIBackend, AnthropicBackend, DeepSeekBackend,
        OllamaBackend, FreeBackend, CascadeBackend, HuggingFaceBackend
    )
    
    backends = [
        {
            "name": "Cascade (Smart Failover)",
            "class": CascadeBackend,
            "model": "multi-service",
            "cost": "FREE ✓",
            "quality": "⭐⭐⭐",
            "speed": "Fast",
            "features": "Auto-failover, Glossary learning, No API key",
            "best_for": "Free usage, Learning, Testing"
        },
        {
            "name": "Free (Google)",
            "class": FreeBackend,
            "model": "google",
            "cost": "FREE ✓",
            "quality": "⭐⭐⭐",
            "speed": "Fast",
            "features": "Google Translate API, No limits",
            "best_for": "Quick translations, No setup"
        },
        {
            "name": "HuggingFace",
            "class": HuggingFaceBackend,
            "model": "Helsinki-NLP/opus-mt-en-fr",
            "cost": "FREE ✓",
            "quality": "⭐⭐⭐⭐",
            "speed": "Medium",
            "features": "Open source models, Multiple languages",
            "best_for": "Research, Open source, Customization"
        },
        {
            "name": "Ollama (Local)",
            "class": OllamaBackend,
            "model": "llama3.1",
            "cost": "FREE ✓",
            "quality": "⭐⭐⭐⭐",
            "speed": "Medium-Slow",
            "features": "Offline, Privacy, No API costs",
            "best_for": "Offline usage, Privacy concerns"
        },
        {
            "name": "DeepSeek",
            "class": DeepSeekBackend,
            "model": "deepseek-chat",
            "cost": "$0.14/1M tokens",
            "quality": "⭐⭐⭐⭐",
            "speed": "Fast",
            "features": "Affordable, Good quality",
            "best_for": "Cost-effective production"
        },
        {
            "name": "OpenAI GPT-4",
            "class": OpenAIBackend,
            "model": "gpt-4o",
            "cost": "$2.50/1M tokens",
            "quality": "⭐⭐⭐⭐⭐",
            "speed": "Fast",
            "features": "Best quality, Reliable",
            "best_for": "High-quality production"
        },
        {
            "name": "Anthropic Claude",
            "class": AnthropicBackend,
            "model": "claude-3-5-sonnet",
            "cost": "$3.00/1M tokens",
            "quality": "⭐⭐⭐⭐⭐",
            "speed": "Medium",
            "features": "Long context, Detailed",
            "best_for": "Long documents, Context-aware"
        }
    ]
    
    for backend in backends:
        try:
            instance = backend["class"](model=backend["model"])
            status = "✓ READY" if instance.is_available() else "⚠ NOT CONFIGURED"
            status_color = "green" if instance.is_available() else "yellow"
        except:
            status = "✗ ERROR"
            status_color = "red"
        
        console.print(f"\n[bold]{backend['name']}[/bold] [{status_color}]{status}[/{status_color}]")
        console.print(f"  Model:    {backend['model']}")
        console.print(f"  Cost:     {backend['cost']}")
        console.print(f"  Quality:  {backend['quality']}")
        console.print(f"  Speed:    {backend['speed']}")
        console.print(f"  Features: {backend['features']}")
        console.print(f"  Best for: {backend['best_for']}")
