#!/bin/bash
# setup_tvbo_tvboptim_env.sh — Set up tvbo+tvboptim Jupyter kernel on EBRAINS
# Usage: bash setup_tvbo_tvboptim_env.sh

set -e

VENV="$HOME/tvboptim-env"
LOG="$HOME/setup_tvbo_env.log"

exec > >(tee "$LOG") 2>&1

echo "=== TVBO Environment Setup ==="

# 1. Install uv
echo "[1/5] Installing uv..."
pip install -q --user uv
export PATH="$HOME/.local/bin:$PATH"
echo "      uv: $(uv --version)"

# 2. Install standalone Python (avoids spack contamination)
echo "[2/5] Installing standalone Python 3.12..."
uv python install 3.12
echo "      Python installed."

# 3. Create venv with standalone Python (remove existing)
echo "[3/5] Creating venv at $VENV..."
rm -rf "$VENV"
uv venv --python-preference only-managed --python 3.12 "$VENV"
echo "      Venv created."

# 4. Register Jupyter kernel
echo "[4/5] Registering Jupyter kernel..."
uv pip install --python "$VENV/bin/python" ipykernel
"$VENV/bin/python" -m ipykernel install --user \
    --name tvboptim-env \
    --display-name "Python (tvb-o-ptim)"
echo "      Kernel registered."

# 5. Install tvbo + tvboptim
echo "[5/5] Installing tvbo, tvboptim..."
uv pip install -U --python "$VENV/bin/python" "owlready2<0.48" tvbo tvboptim
echo "      Packages installed."

echo ""
echo "=== Verifying installation... ==="
"$VENV/bin/python" -c "import pathspec; print(f'  pathspec {pathspec.__version__} from {pathspec.__file__}')"
"$VENV/bin/python" -c "import black; print(f'  black OK')"
"$VENV/bin/python" -c "from tvbo import Dynamics; print('  tvbo OK')"
"$VENV/bin/python" -c "import tvboptim; print('  tvboptim OK')"
echo ""
echo "=== Done! Refresh JupyterLab and select the 'Python (tvb-o-ptim)' kernel. ==="
