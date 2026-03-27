#!/bin/sh
if [ ! -d _extensions/grantmcdermott ]; then
  quarto add grantmcdermott/quarto-revealjs-clean --no-prompt
fi

# Generate authors gallery (only if _authors.yaml changed)
python scripts/generate_authors_gallery.py
