#!/usr/bin/env python
"""
Experiment runner for evaluation metrics.

Reads system outputs, references, and optional sources from text files (one line per sample),
computes metrics, and writes a JSON report to stdout or a file.

Usage:
    python scripts/run_experiment.py \
        --hyp system.txt --ref reference.txt --src source.txt \
        --glossary glossary.json --include-comet \
        --out results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from scitran.evaluation.metrics import evaluate_translation


def _read_lines(path: Optional[str]) -> Optional[List[str]]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def _read_glossary(path: Optional[str]) -> Optional[dict]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation metrics for translation outputs.")
    parser.add_argument("--hyp", required=True, help="Path to hypotheses (system outputs), one per line.")
    parser.add_argument("--ref", help="Path to reference translations, one per line.")
    parser.add_argument("--src", help="Path to source texts, one per line (for glossary/numeric metrics).")
    parser.add_argument("--glossary", help="Glossary JSON path (term -> enforced translation).")
    parser.add_argument("--include-comet", action="store_true", help="Compute COMET if available.")
    parser.add_argument("--comet-model", default="Unbabel/wmt22-comet-da", help="COMET model name.")
    parser.add_argument("--out", help="Output path for metrics JSON. Defaults to stdout.")
    args = parser.parse_args()

    hyps = _read_lines(args.hyp) or []
    refs = _read_lines(args.ref) if args.ref else None
    srcs = _read_lines(args.src) if args.src else None
    glossary = _read_glossary(args.glossary) if args.glossary else None

    metrics = evaluate_translation(
        hypotheses=hyps,
        references=refs,
        sources=srcs,
        glossary=glossary,
        blocks=None,
        include_comet=bool(args.include_comet),
        comet_model=args.comet_model,
    )

    payload = {"metrics": metrics, "counts": {"hypotheses": len(hyps), "references": len(refs or [])}}

    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()


