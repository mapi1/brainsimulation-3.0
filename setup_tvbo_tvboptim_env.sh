#!/bin/bash
# setup_tvbo_tvboptim_env.sh — Set up tvbo+tvboptim Jupyter kernel on EBRAINS
# Usage: bash setup_tvbo_tvboptim_env.sh

set -e

VENV="$HOME/tvboptim-env"
LOG="$HOME/setup_tvbo_env.log"
KERNEL_DIR="$HOME/.local/share/jupyter/kernels/tvboptim-env"

exec > >(tee "$LOG") 2>&1

echo "=== TVBO Environment Setup ==="

# 1. Install uv (needs spack pip, so do this BEFORE unsetting PYTHONPATH)
echo "[1/6] Installing uv..."
pip install -q --user uv
export PATH="$HOME/.local/bin:$PATH"
echo "      uv: $(uv --version)"

# 2. Neutralize spack — PYTHONPATH injects spack's 3.11 packages into all Pythons
echo "[2/6] Clearing spack PYTHONPATH..."
SPACK_PYTHONPATH="$PYTHONPATH"
unset PYTHONPATH
echo "      PYTHONPATH cleared (was: ${SPACK_PYTHONPATH:-(empty)})"

# 3. Install standalone Python (avoids spack contamination)
echo "[3/6] Installing standalone Python 3.12..."
uv python install 3.12
echo "      Python installed."

# 4. Create venv with standalone Python (remove existing)
echo "[4/6] Creating venv at $VENV..."
rm -rf "$VENV"
uv venv --python-preference only-managed --python 3.12 "$VENV"
echo "      Venv created."

# 5. Install packages + register kernel
echo "[5/6] Installing ipykernel, tvbo, tvboptim..."
uv pip install --python "$VENV/bin/python" ipykernel "owlready2<0.48" tvbo tvboptim
echo "      Packages installed."

echo "[6/6] Registering Jupyter kernel..."
"$VENV/bin/python" -m ipykernel install --user \
    --name tvboptim-env \
    --display-name "Python (tvb-o-ptim)"

# Patch kernel.json to clear PYTHONPATH at runtime (prevents spack leaking into kernel)
"$VENV/bin/python" -c "
import json, pathlib
kf = pathlib.Path('$KERNEL_DIR/kernel.json')
k = json.loads(kf.read_text())
k.setdefault('env', {})
k['env']['PYTHONPATH'] = ''
k['env']['PYTHONNOUSERSITE'] = '1'
kf.write_text(json.dumps(k, indent=1))
print('      kernel.json patched:', kf)
"
echo "      Kernel registered."

echo ""
echo "=== Verifying installation... ==="
"$VENV/bin/python" -c "import pathspec; print(f'  pathspec {pathspec.__version__} from {pathspec.__file__}')"
"$VENV/bin/python" -c "import black; print(f'  black OK')"
"$VENV/bin/python" -c "from tvbo import Dynamics; print('  tvbo OK')"
"$VENV/bin/python" -c "import tvboptim; print('  tvboptim OK')"
echo ""
echo "=== Done! Refresh JupyterLab and select the 'Python (tvb-o-ptim)' kernel. ==="
