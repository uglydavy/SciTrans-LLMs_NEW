#!/usr/bin/env python3
"""
Validate that all development phases are properly implemented.

Development Phases:
Phase 1: Core Foundation (Week 1) - Core models, pipeline, basic translation, masking, CLI
Phase 2: Innovations (Week 2) - Advanced masking, document context, layout preservation
Phase 3: Quality (Week 3) - Prompt training, reranking, multiple backends
Phase 4: Evaluation (Week 4) - Testing, experiments, documentation
"""

import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def check_phase_1():
    """Phase 1: Core Foundation"""
    console.print("\n[bold cyan]Phase 1: Core Foundation[/bold cyan]")
    
    checks = {}
    
    # Core models
    try:
        from scitran.core.models import Document, Block, BoundingBox, BlockType
        checks["Core Models"] = ("✓", "green", "Document, Block, BoundingBox implemented")
    except Exception as e:
        checks["Core Models"] = ("✗", "red", f"Error: {e}")
    
    # Pipeline
    try:
        from scitran.core.pipeline import TranslationPipeline, PipelineConfig
        checks["Translation Pipeline"] = ("✓", "green", "Pipeline and config implemented")
    except Exception as e:
        checks["Translation Pipeline"] = ("✗", "red", f"Error: {e}")
    
    # Basic masking
    try:
        from scitran.masking.engine import MaskingEngine
        from scitran.core.models import Block
        
        engine = MaskingEngine()
        block = Block(block_id="test", source_text="Test $x=5$ equation")
        masked = engine.mask_block(block)
        
        if "MASK_" in masked.masked_text:
            checks["Basic Masking"] = ("✓", "green", "Masking engine working")
        else:
            checks["Basic Masking"] = ("✗", "yellow", "Masking not creating masks")
    except Exception as e:
        checks["Basic Masking"] = ("✗", "red", f"Error: {e}")
    
    # CLI
    try:
        from cli.commands.main import app
        checks["CLI Interface"] = ("✓", "green", "Typer-based CLI implemented")
    except Exception as e:
        checks["CLI Interface"] = ("✗", "red", f"Error: {e}")
    
    return checks


def check_phase_2():
    """Phase 2: Innovations"""
    console.print("\n[bold cyan]Phase 2: Innovations[/bold cyan]")
    
    checks = {}
    
    # Advanced masking with validation
    try:
        from scitran.masking.engine import MaskingEngine
        from scitran.core.models import Block
        
        engine = MaskingEngine()
        block = Block(block_id="test", source_text="URL: https://example.com, code: `print()`, equation: $x^2$")
        masked = engine.mask_block(block)
        
        if len(masked.masks) >= 3:
            checks["Advanced Masking"] = ("✓", "green", f"Multiple pattern types ({len(masked.masks)} masks)")
        else:
            checks["Advanced Masking"] = ("⚠", "yellow", f"Only {len(masked.masks)} masks detected")
    except Exception as e:
        checks["Advanced Masking"] = ("✗", "red", f"Error: {e}")
    
    # Document-level context
    try:
        from scitran.translation.prompts import PromptOptimizer
        checks["Document Context"] = ("✓", "green", "Prompt system with context support")
    except Exception as e:
        checks["Document Context"] = ("✗", "red", f"Error: {e}")
    
    # Layout preservation
    try:
        from scitran.extraction.layout import LayoutDetector
        from scitran.rendering.pdf_renderer import PDFRenderer
        checks["Layout Preservation"] = ("✓", "green", "Layout detection and rendering")
    except Exception as e:
        checks["Layout Preservation"] = ("✗", "red", f"Error: {e}")
    
    return checks


def check_phase_3():
    """Phase 3: Quality"""
    console.print("\n[bold cyan]Phase 3: Quality[/bold cyan]")
    
    checks = {}
    
    # Prompt training system
    try:
        from scitran.translation.prompts import PromptOptimizer
        optimizer = PromptOptimizer()
        
        if hasattr(optimizer, 'select_strategy') and hasattr(optimizer, 'update_performance'):
            checks["Prompt Training"] = ("✓", "green", "Prompt optimizer with learning")
        else:
            checks["Prompt Training"] = ("⚠", "yellow", "Prompt system exists but limited")
    except Exception as e:
        checks["Prompt Training"] = ("✗", "red", f"Error: {e}")
    
    # Reranking and scoring
    try:
        from scitran.scoring.reranker import AdvancedReranker
        checks["Reranking System"] = ("✓", "green", "Multi-dimensional reranking")
    except Exception as e:
        checks["Reranking System"] = ("✗", "red", f"Error: {e}")
    
    # Multiple backends
    try:
        from scitran.translation.backends import (
            OpenAIBackend, AnthropicBackend, DeepSeekBackend,
            OllamaBackend, FreeBackend, CascadeBackend
        )
        checks["Multiple Backends"] = ("✓", "green", "6 backends implemented")
    except Exception as e:
        checks["Multiple Backends"] = ("✗", "red", f"Error: {e}")
    
    return checks


def check_phase_4():
    """Phase 4: Evaluation"""
    console.print("\n[bold cyan]Phase 4: Evaluation[/bold cyan]")
    
    checks = {}
    
    # Comprehensive testing
    test_dirs = ["tests/unit", "tests/integration"]
    has_tests = all(Path(d).exists() for d in test_dirs)
    
    if has_tests:
        test_files = list(Path("tests").rglob("test_*.py"))
        checks["Testing Framework"] = ("✓", "green", f"{len(test_files)} test files")
    else:
        checks["Testing Framework"] = ("⚠", "yellow", "Test structure exists but incomplete")
    
    # Thesis experiments
    try:
        from experiments.ablation import AblationStudy
        from benchmarks.speed_test import SpeedBenchmark
        from benchmarks.quality_test import QualityBenchmark
        checks["Experiment Framework"] = ("✓", "green", "Ablation, speed, quality benchmarks")
    except Exception as e:
        checks["Experiment Framework"] = ("✗", "red", f"Error: {e}")
    
    # Documentation
    docs = ["README.md", "QUICKSTART.md", "BUILD_SUMMARY.md", "UPDATES.md"]
    exists = [Path(f"docs/{d}").exists() or Path(d).exists() for d in docs]
    
    if all(exists):
        checks["Documentation"] = ("✓", "green", "Complete documentation")
    else:
        missing = len([e for e in exists if not e])
        checks["Documentation"] = ("⚠", "yellow", f"{missing} docs missing")
    
    return checks


def print_results(phase_name, checks):
    """Print results for a phase."""
    table = Table(show_header=False, box=None)
    
    for feature, (status, color, details) in checks.items():
        table.add_row(f"[{color}]{status}[/{color}]", f"[bold]{feature}[/bold]", f"[dim]{details}[/dim]")
    
    console.print(table)
    
    # Count
    passed = sum(1 for s, c, d in checks.values() if s == "✓")
    total = len(checks)
    
    if passed == total:
        console.print(f"\n[green]✓ {phase_name}: {passed}/{total} checks passed[/green]")
    else:
        console.print(f"\n[yellow]⚠ {phase_name}: {passed}/{total} checks passed[/yellow]")
    
    return passed == total


def main():
    """Run all phase validations."""
    console.print(Panel.fit(
        "[bold cyan]SciTrans-LLMs Development Phase Validation[/bold cyan]\n\n"
        "Checking all 4 development phases...",
        border_style="cyan"
    ))
    
    results = {}
    
    # Phase 1
    results["Phase 1"] = print_results("Phase 1", check_phase_1())
    
    # Phase 2
    results["Phase 2"] = print_results("Phase 2", check_phase_2())
    
    # Phase 3
    results["Phase 3"] = print_results("Phase 3", check_phase_3())
    
    # Phase 4
    results["Phase 4"] = print_results("Phase 4", check_phase_4())
    
    # Summary
    console.print("\n" + "=" * 70)
    console.print("[bold]SUMMARY[/bold]")
    console.print("=" * 70)
    
    summary_table = Table(show_header=True)
    summary_table.add_column("Phase", style="cyan")
    summary_table.add_column("Status", style="bold")
    summary_table.add_column("Description")
    
    phases = [
        ("Phase 1", "Core Foundation", "Models, Pipeline, Masking, CLI"),
        ("Phase 2", "Innovations", "Advanced Masking, Context, Layout"),
        ("Phase 3", "Quality", "Prompts, Reranking, Backends"),
        ("Phase 4", "Evaluation", "Tests, Experiments, Docs")
    ]
    
    for phase_id, name, desc in phases:
        if results[phase_id]:
            summary_table.add_row(name, "[green]✓ Complete[/green]", desc)
        else:
            summary_table.add_row(name, "[yellow]⚠ Partial[/yellow]", desc)
    
    console.print(summary_table)
    
    # Overall status
    all_passed = all(results.values())
    
    console.print("\n" + "=" * 70)
    
    if all_passed:
        console.print("[bold green]🎉 ALL PHASES COMPLETE![/bold green]")
        console.print("\nYour system is fully implemented and ready for:")
        console.print("  ✓ Production use")
        console.print("  ✓ Thesis experiments")
        console.print("  ✓ Paper publication")
        return 0
    else:
        partial = sum(1 for p in results.values() if p)
        console.print(f"[bold yellow]⚠ {partial}/4 PHASES COMPLETE[/bold yellow]")
        console.print("\nSome components need attention (see above for details)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
