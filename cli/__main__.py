"""CLI entry point for SciTrans-LLMs."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cli.commands.main import cli

if __name__ == "__main__":
    cli()
