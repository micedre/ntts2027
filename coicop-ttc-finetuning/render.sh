#!/usr/bin/env bash
# Full render chain for the abstract. Requires: uv, quarto.
set -euo pipefail
cd "$(dirname "$0")"
uv run python scripts/make_reference_doc.py
if [ -n "${DDC_ENCRYPTION_KEY:-}" ]; then
  uv run python scripts/code_distribution.py
else
  echo "DDC_ENCRYPTION_KEY not set: keeping committed results/code_distribution.*"
fi
uv run python scripts/make_figures.py
quarto render abstract.qmd
uv run python scripts/fix_docx_tables.py abstract.docx
uv run python scripts/apply_template_layout.py abstract.docx
