#!/bin/sh
set -e
if [ ! -d _extensions/grantmcdermott ]; then
  quarto add grantmcdermott/quarto-revealjs-clean --no-prompt
fi

# Generate authors gallery using project venv only.
VENV_PY="../.venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
  echo "Error: .venv Python not found at $VENV_PY"
  exit 1
fi

"$VENV_PY" scripts/generate_authors_gallery.py
