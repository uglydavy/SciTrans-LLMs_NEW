#!/usr/bin/env python3
"""
SciTrans-LLMs NEW - Master Run Script

Launch CLI or GUI interfaces for scientific document translation.

Usage:
    python run.py                    # Interactive mode selection
    python run.py gui                # Launch Gradio GUI
    python run.py cli [args]         # Run CLI commands
    python run.py test               # Run innovation tests
    python run.py translate file.pdf # Quick translate

Examples:
    python run.py gui
    python run.py cli translate paper.pdf --backend cascade
    python run.py test --verbose
    python run.py translate paper.pdf -o output.pdf
"""

import sys
import os
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []
    optional_missing = []
    
    # Core required
    try:
        import yaml
    except ImportError:
        missing.append("pyyaml")
    
    try:
        import requests
    except ImportError:
        missing.append("requests")
    
    # For CLI
    try:
        import typer
    except ImportError:
        optional_missing.append("typer (for CLI)")
    
    try:
        import rich
    except ImportError:
        optional_missing.append("rich (for CLI)")
    
    # For GUI
    try:
        import gradio
    except ImportError:
        optional_missing.append("gradio (for GUI)")
    
    # For scoring
    try:
        import numpy
    except ImportError:
        optional_missing.append("numpy (for reranking)")
    
    # For PDF
    try:
        import fitz
    except ImportError:
        optional_missing.append("PyMuPDF (for PDF processing)")
    
    return missing, optional_missing


def print_banner():
    """Print application banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   ███████╗ ██████╗██╗████████╗██████╗  █████╗ ███╗   ██╗     ║
║   ██╔════╝██╔════╝██║╚══██╔══╝██╔══██╗██╔══██╗████╗  ██║     ║
║   ███████╗██║     ██║   ██║   ██████╔╝███████║██╔██╗ ██║     ║
║   ╚════██║██║     ██║   ██║   ██╔══██╗██╔══██║██║╚██╗██║     ║
║   ███████║╚██████╗██║   ██║   ██║  ██║██║  ██║██║ ╚████║     ║
║   ╚══════╝ ╚═════╝╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝     ║
║                                                               ║
║            Advanced Scientific Document Translation           ║
║                        Version 2.0.0                          ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def launch_gui():
    """Launch the Gradio GUI."""
    try:
        from gui.app import launch_gui
        print("Launching GUI at http://localhost:7860")
        launch_gui()
    except ImportError as e:
        print(f"Error: Could not import GUI module: {e}")
        print("\nTo use the GUI, install required dependencies:")
        print("  pip install gradio plotly pandas")
        sys.exit(1)


def launch_cli(args):
    """Launch the CLI with given arguments."""
    try:
        from cli.commands.main import cli
        sys.argv = ['scitrans'] + args
        cli()
    except ImportError as e:
        print(f"Error: Could not import CLI module: {e}")
        print("\nTo use the CLI, install required dependencies:")
        print("  pip install typer rich")
        sys.exit(1)


def run_tests(args):
    """Run innovation tests."""
    import subprocess
    test_script = PROJECT_ROOT / "scripts" / "test_innovations.py"
    
    if test_script.exists():
        cmd = [sys.executable, str(test_script)] + args
        subprocess.run(cmd)
    else:
        print("Error: Test script not found at scripts/test_innovations.py")
        sys.exit(1)


def quick_translate(filepath, output=None, backend="cascade"):
    """Quick translation using default settings."""
    try:
        from scitran.extraction.pdf_parser import PDFParser
        from scitran.core.pipeline import TranslationPipeline, PipelineConfig
        from scitran.rendering.pdf_renderer import PDFRenderer
        
        if not os.path.exists(filepath):
            print(f"Error: File not found: {filepath}")
            sys.exit(1)
        
        if output is None:
            base = Path(filepath).stem
            output = f"{base}_translated.pdf"
        
        print(f"Translating: {filepath}")
        print(f"Backend: {backend}")
        print(f"Output: {output}")
        print()
        
        # Parse PDF
        print("[1/4] Parsing PDF...")
        parser = PDFParser()
        document = parser.parse(filepath)
        print(f"      Found {len(document.all_blocks)} text blocks")
        
        # Configure pipeline
        print("[2/4] Configuring pipeline...")
        config = PipelineConfig(
            source_lang="en",
            target_lang="fr",
            backend=backend,
            enable_masking=True,
            enable_reranking=False
        )
        pipeline = TranslationPipeline(config)
        
        # Translate
        print("[3/4] Translating...")
        result = pipeline.translate_document(document)
        
        # Render
        print("[4/4] Rendering output...")
        renderer = PDFRenderer()
        renderer.render_simple(result.document, output)
        
        print()
        print(f"✓ Translation complete!")
        print(f"  Output saved to: {output}")
        print(f"  Blocks translated: {result.blocks_translated}")
        print(f"  Duration: {result.duration:.2f}s")
        
    except ImportError as e:
        print(f"Error: Missing dependency: {e}")
        print("\nInstall dependencies with:")
        print("  pip install -r requirements-core.txt")
        sys.exit(1)
    except Exception as e:
        print(f"Error during translation: {e}")
        sys.exit(1)


def show_help():
    """Show help message."""
    help_text = """
SciTrans-LLMs NEW - Scientific Document Translation

USAGE:
    python run.py [command] [options]

COMMANDS:
    gui                     Launch the graphical interface
    cli [args]              Run CLI commands (pass arguments after 'cli')
    test [--verbose]        Run innovation tests
    translate <file.pdf>    Quick translate a PDF file

EXAMPLES:
    python run.py                           # Interactive mode
    python run.py gui                       # Launch GUI
    python run.py cli translate paper.pdf   # CLI translation
    python run.py test -v                   # Run tests with details
    python run.py translate paper.pdf       # Quick translate

CLI COMMANDS (via 'python run.py cli'):
    translate   Translate a PDF document
    info        Show PDF information
    backends    List available backends
    wizard      Interactive translation wizard
    test        Test a backend
    glossary    Manage glossaries
    gui         Launch GUI from CLI

For detailed CLI help:
    python run.py cli --help
    python run.py cli translate --help

DOCUMENTATION:
    README.md               Project overview
    QUICK_START.md          Getting started guide
    docs/INNOVATIONS.md     Technical innovations
    docs/API_KEYS_SETUP.md  API key configuration

THREE INNOVATIONS:
    1. Terminology-Constrained Translation (Masking)
    2. Document-Level Context (Reranking)
    3. Layout Preservation (PDF reconstruction)
"""
    print(help_text)


def interactive_menu():
    """Show interactive menu."""
    print_banner()
    
    # Check dependencies
    missing, optional = check_dependencies()
    
    if missing:
        print("⚠ Missing required dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nInstall with: pip install -r requirements-core.txt")
        print()
    
    if optional:
        print("ℹ Optional dependencies not installed:")
        for dep in optional:
            print(f"  - {dep}")
        print()
    
    print("Select an option:")
    print()
    print("  [1] Launch GUI (Gradio interface)")
    print("  [2] Launch CLI (Command line)")
    print("  [3] Run tests (Innovation verification)")
    print("  [4] Quick translate (Translate a PDF)")
    print("  [5] Show help")
    print("  [q] Quit")
    print()
    
    choice = input("Enter choice: ").strip().lower()
    
    if choice == '1':
        launch_gui()
    elif choice == '2':
        print("\nEntering CLI mode. Type 'exit' to return.")
        print("Example commands:")
        print("  translate paper.pdf --backend cascade")
        print("  backends --detailed")
        print("  test --backend cascade")
        print()
        while True:
            try:
                cmd = input("scitran> ").strip()
                if cmd.lower() in ('exit', 'quit', 'q'):
                    break
                if cmd:
                    launch_cli(cmd.split())
            except KeyboardInterrupt:
                print()
                break
    elif choice == '3':
        run_tests(['-v'])
    elif choice == '4':
        filepath = input("Enter PDF path: ").strip()
        if filepath:
            quick_translate(filepath)
    elif choice == '5':
        show_help()
    elif choice == 'q':
        print("Goodbye!")
        sys.exit(0)
    else:
        print("Invalid choice.")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        interactive_menu()
        return
    
    command = sys.argv[1].lower()
    args = sys.argv[2:]
    
    if command == 'gui':
        launch_gui()
    elif command == 'cli':
        launch_cli(args)
    elif command == 'test':
        run_tests(args)
    elif command == 'translate':
        if not args:
            print("Error: Please specify a PDF file to translate")
            print("Usage: python run.py translate <file.pdf> [-o output.pdf] [--backend cascade]")
            sys.exit(1)
        
        filepath = args[0]
        output = None
        backend = "cascade"
        
        # Parse simple options
        i = 1
        while i < len(args):
            if args[i] in ('-o', '--output') and i + 1 < len(args):
                output = args[i + 1]
                i += 2
            elif args[i] in ('-b', '--backend') and i + 1 < len(args):
                backend = args[i + 1]
                i += 2
            else:
                i += 1
        
        quick_translate(filepath, output, backend)
    elif command in ('-h', '--help', 'help'):
        show_help()
    else:
        print(f"Unknown command: {command}")
        print("Use 'python run.py --help' for usage information.")
        sys.exit(1)


if __name__ == "__main__":
    main()

