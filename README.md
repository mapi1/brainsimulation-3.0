# Lab-Meeting 27.03.2026

## Hands-on session

[Click here to open the notebook in Google Colab](https://colab.research.google.com/github/mapi1/nono.ipynb)

## Local Setup

### Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Python 3.10 or newer (uv installs one for you if needed)

### 1. Clone the repository

```bash
git clone https://github.com/virtual-brain-twins/tvb-o-ptim.git
cd tvb-o-ptim
```

### 2. Create a virtual environment and install dependencies

```bash
uv venv .venv --python 3.12
source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv pip install -e ".[cpu]"         # installs tvbo, tvboptim, bsplot, jax + extras
```

### 3. Register the Jupyter kernel

```bash
python -m ipykernel install --user --name tvb-o-ptim --display-name "Python (tvb-o-ptim)"
```

### 4. Open the notebook

```bash
jupyter notebook notebooks/ReducedWongWang_Optimization_TVBOptim.ipynb
```

Select the **Python (tvb-o-ptim)** kernel and run all cells.

> **Tip – multi-core CPU simulation:** The notebook sets `XLA_FLAGS=--xla_force_host_platform_device_count=8`
> at the top so JAX can parallelise across 8 virtual devices on a single CPU host.
> Adjust `N_devices` in the first code cell to match your machine.