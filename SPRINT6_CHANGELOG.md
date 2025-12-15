# SPRINT 6 — Documentation Consolidation (Complete)

## Scope & goals
- Produce a single, accurate docs set; remove conflicting/legacy content.
- Ensure README points to authoritative docs and reflects current pipeline defaults.

## Delivered
- Authoritative docs retained/updated:
  - `docs/ARCHITECTURE.md`, `CLI.md`, `GUI.md`, `EVALUATION.md`, `ABLATIONS.md`, `REPRODUCIBILITY.md`, `DEPLOYMENT.md`, `CI_CD.md`.
  - README refreshed: strict/fallback defaults, evaluation runner, CI badge, doc index.
- Legacy/duplicate docs removed; `docs/LEGACY.md` lists archived items; `docs/` now only the authoritative set plus LEGACY listing.
- Trimmed `docs/api`, `docs/ACCURACY_CHECK`, and other redundant guides.

## Tests
- Unit: 70/70 passing (offline; benign PyMuPDF deprecation warnings).***

