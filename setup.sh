#!/bin/sh
if [ ! -d presentation/_extensions/grantmcdermott ]; then
  cd presentation && quarto add grantmcdermott/quarto-revealjs-clean --no-prompt && cd ..
fi
