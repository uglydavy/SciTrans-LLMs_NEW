#!/bin/bash
# SciTrans LLMs launcher script
# This ensures the correct Python environment is used

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Use project's virtual environment if it exists
if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
elif [ -f ".venv/bin/python3" ]; then
    PYTHON=".venv/bin/python3"
else
    PYTHON="python3"
fi

# Run the CLI module directly (no run.py needed)
exec "$PYTHON" -m cli.commands.main "$@"
