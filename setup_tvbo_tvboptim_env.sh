#!/bin/bash
# setup_tvbo_env.sh — Set up tvbo+tvboptim Jupyter kernel on EBRAINS
# Usage: bash setup_tvbo_env.sh

set -e

VENV="$HOME/tvbo-env"
LOG="$HOME/setup_tvbo_env.log"

exec > >(tee "$LOG") 2>&1

echo "=== TVBO Environment Setup ==="
echo "    Log: $LOG"

# 1. Create isolated venv (no --system-site-packages to avoid spack conflicts)
echo ""
echo "[1/4] Creating venv at $VENV..."
if [ -d "$VENV" ]; then
    echo "      Venv already exists. Delete it first to recreate: rm -rf $VENV"
else
    python3 -m venv "$VENV"
    echo "      Venv created."
fi

# 2. Upgrade pip (EBRAINS ships pip 23.0 which is extremely slow)
echo ""
echo "[2/4] Upgrading pip..."
"$VENV/bin/python" -m pip install --upgrade pip
echo "      pip upgraded: $("$VENV/bin/pip" --version)"

# 3. Install packages (verbose so you can see progress)
echo ""
echo "[3/4] Installing tvbo, tvboptim, ipykernel..."
"$VENV/bin/pip" install \
    "owlready2<0.48" \
    tvbo \
    tvboptim \
    ipykernel
echo "      Packages installed."

# 4. Register Jupyter kernel (user-level)
echo ""
echo "[4/4] Registering Jupyter kernel..."
"$VENV/bin/python" -m ipykernel install \
    --user \
    --name tvbo-env \
    --display-name "Python (tvbo)"
echo "      Kernel registered."

echo ""
echo "=== Done! Refresh JupyterLab and select the 'Python (tvbo)' kernel. ==="
