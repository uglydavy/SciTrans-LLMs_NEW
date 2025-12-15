# -*- coding: utf-8 -*-
"""
Progress bar utilities for CLI and GUI.

Provides a unified progress reporting interface that works in:
- Terminal (with rich progress bars)
- GUI (with callback functions)
- Headless environments (with simple logging)
"""

import sys
import time
import logging
from typing import Optional, Callable, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Try to import rich for beautiful progress bars
try:
    from rich.progress import (
        Progress, 
        SpinnerColumn, 
        TextColumn, 
        BarColumn, 
        TaskProgressColumn,
        TimeElapsedColumn,
        TimeRemainingColumn
    )
    from rich.console import Console
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


@dataclass
class ProgressStats:
    """Statistics for progress tracking."""
    total: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    start_time: float = field(default_factory=time.time)
    
    @property
    def percentage(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.completed / self.total) * 100
    
    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time
    
    @property
    def rate(self) -> float:
        """Items per second."""
        if self.elapsed == 0:
            return 0.0
        return self.completed / self.elapsed
    
    @property
    def eta(self) -> float:
        """Estimated time remaining."""
        if self.rate == 0:
            return float('inf')
        remaining = self.total - self.completed
        return remaining / self.rate


class ProgressReporter:
    """
    Unified progress reporter for CLI/GUI.
    
    Usage:
        # CLI with rich progress bar
        with ProgressReporter(total=100, description="Translating") as progress:
            for item in items:
                process(item)
                progress.advance()
        
        # With callback for GUI
        def update_gui(stats):
            gui.update_progress(stats.percentage)
        
        progress = ProgressReporter(total=100, callback=update_gui)
        progress.start()
        for item in items:
            progress.advance()
        progress.finish()
    """
    
    def __init__(
        self,
        total: int,
        description: str = "Processing",
        callback: Optional[Callable[[ProgressStats], None]] = None,
        use_rich: bool = True,
        show_rate: bool = True,
        show_eta: bool = True
    ):
        self.stats = ProgressStats(total=total)
        self.description = description
        self.callback = callback
        self.use_rich = use_rich and HAS_RICH and sys.stdout.isatty()
        self.show_rate = show_rate
        self.show_eta = show_eta
        
        self._progress = None
        self._task_id = None
        self._console = None
    
    def start(self):
        """Start progress tracking."""
        self.stats.start_time = time.time()
        
        if self.use_rich:
            columns = [
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=40),
                TaskProgressColumn(),
            ]
            if self.show_eta:
                columns.extend([
                    TimeElapsedColumn(),
                    TextColumn("•"),
                    TimeRemainingColumn()
                ])
            
            self._console = Console()
            self._progress = Progress(*columns, console=self._console)
            self._progress.start()
            self._task_id = self._progress.add_task(
                self.description, 
                total=self.stats.total
            )
        else:
            self._print_simple(f"{self.description}: 0/{self.stats.total}")
    
    def advance(self, amount: int = 1, status: str = "completed"):
        """Advance progress."""
        if status == "completed":
            self.stats.completed += amount
        elif status == "failed":
            self.stats.failed += amount
            self.stats.completed += amount
        elif status == "skipped":
            self.stats.skipped += amount
            self.stats.completed += amount
        
        if self.use_rich and self._progress:
            self._progress.update(self._task_id, advance=amount)
        elif not self.use_rich:
            self._print_simple_progress()
        
        if self.callback:
            self.callback(self.stats)
    
    def update_description(self, description: str):
        """Update the progress description."""
        self.description = description
        if self.use_rich and self._progress and self._task_id is not None:
            self._progress.update(self._task_id, description=description)
    
    def finish(self, final_message: Optional[str] = None):
        """Finish progress tracking."""
        if self.use_rich and self._progress:
            self._progress.stop()
            if final_message:
                self._console.print(f"[green]✓[/green] {final_message}")
        else:
            msg = final_message or f"{self.description}: Done ({self.stats.completed}/{self.stats.total})"
            self._print_simple(msg, newline=True)
    
    def _print_simple(self, message: str, newline: bool = False):
        """Print simple progress for non-rich environments."""
        end = "\n" if newline else "\r"
        print(f"\r{message}".ljust(80), end=end, flush=True)
    
    def _print_simple_progress(self):
        """Print simple progress bar."""
        pct = self.stats.percentage
        filled = int(pct / 2)
        bar = "█" * filled + "░" * (50 - filled)
        eta_str = ""
        if self.show_eta and self.stats.eta < float('inf'):
            eta_str = f" ETA: {self.stats.eta:.0f}s"
        msg = f"{self.description}: [{bar}] {pct:.1f}%{eta_str}"
        self._print_simple(msg)
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()
        return False


def create_progress_callback(progress_reporter: ProgressReporter) -> Callable[[int, int, str], None]:
    """
    Create a callback function for batch translator progress.
    
    Args:
        progress_reporter: ProgressReporter instance
        
    Returns:
        Callback function (completed, total, message) -> None
    """
    last_completed = [0]  # Mutable to track delta
    
    def callback(completed: int, total: int, message: str):
        delta = completed - last_completed[0]
        if delta > 0:
            progress_reporter.advance(delta)
        last_completed[0] = completed
    
    return callback


class MultiStageProgress:
    """
    Progress tracker for multi-stage operations.
    
    Usage:
        with MultiStageProgress() as progress:
            with progress.stage("Parsing PDF", 10) as stage:
                for page in pages:
                    process_page(page)
                    stage.advance()
            
            with progress.stage("Translating", 100) as stage:
                for block in blocks:
                    translate(block)
                    stage.advance()
    """
    
    def __init__(self, use_rich: bool = True):
        self.use_rich = use_rich and HAS_RICH and sys.stdout.isatty()
        self.stages_completed = 0
        self.current_stage: Optional[ProgressReporter] = None
        self._console = Console() if self.use_rich else None
    
    def stage(self, description: str, total: int) -> ProgressReporter:
        """Create a new progress stage."""
        self.stages_completed += 1
        stage_desc = f"[{self.stages_completed}] {description}"
        return ProgressReporter(
            total=total,
            description=stage_desc,
            use_rich=self.use_rich
        )
    
    def log(self, message: str, style: str = ""):
        """Log a message."""
        if self._console:
            self._console.print(f"[{style}]{message}[/{style}]" if style else message)
        else:
            print(message)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._console:
            self._console.print()
        return False

