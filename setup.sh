#!/bin/sh
if [ ! -d _extensions/grantmcdermott ]; then
  quarto add grantmcdermott/quarto-revealjs-clean --no-prompt
fi
