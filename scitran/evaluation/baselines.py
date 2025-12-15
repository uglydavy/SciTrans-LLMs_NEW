"""
Lightweight hooks for external baseline comparisons.

Design goals:
- Avoid AGPL or bundled baseline code; call external scripts/binaries.
- Provide a stable interface for experiment runner to ingest baseline outputs.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

BASELINE_TIMEOUT = 3600  # 1 hour default safeguard


def run_external_baseline(
    command: Iterable[str],
    workdir: Optional[str] = None,
    timeout: int = BASELINE_TIMEOUT,
) -> subprocess.CompletedProcess:
    """Run an external baseline command safely.

    Args:
        command: Iterable of command tokens, e.g., ["python", "baseline.py", "--input", "file"]
        workdir: Optional working directory.
        timeout: Seconds before force termination (prevents hanging jobs).

    Returns:
        CompletedProcess with stdout/stderr captured.
    """
    return subprocess.run(
        list(command),
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def load_baseline_results(path: str) -> Dict[str, Any]:
    """Load baseline results from a JSON file produced by external runs."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


