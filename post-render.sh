#!/bin/sh
# Move rendered files from public/presentation/ to public/ root
if [ -d public/presentation ]; then
  mv public/presentation/* public/
  rmdir public/presentation
fi
