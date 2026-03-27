"""Generate a clean surface-only PDE heat diffusion GIF for the gallery."""
import tempfile, os, yaml
import numpy as np
import nibabel as nib
import templateflow.api as tfa
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.animation import FuncAnimation
from bsplot.surface import plot_surf
from bsplot import style
from tvbo import SimulationExperiment

style.use("tvbo")

# --- Setup cortical surface PDE experiment ---
mesh_gii = str(tfa.get(template="fsLR", density="32k", suffix="midthickness", hemi="R", desc=None))

exp_dict = {
    "label": "Cortical surface diffusion",
    "field_dynamics": {
        "label": "Heat/diffusion equation",
        "mesh": {
            "label": "cortex_rh",
            "element_type": "triangle",
            "mesh_file": mesh_gii,
            "mesh_format": "gifti",
        },
        "parameters": {"D": {"name": "D", "value": 50.0}},
        "state_variables": [{
            "name": "u",
            "label": "u",
            "initial_value": 0.0,
            "boundary_conditions": [{"label": "Zero Dirichlet", "bc_type": "Dirichlet", "value": {"rhs": "0"}}],
            "equation": {"lhs": "u_t", "rhs": "D * laplacian(u)"},
        }],
        "operators": [{"label": "Diffusion", "operator_type": "laplacian", "coefficient": "D"}],
        "solver": {"label": "FEM IE", "discretization": "FEM", "time_integrator": "implicit Euler", "dt": 2.0},
    },
    "integration": {"duration": 60},
}

tmpdir = tempfile.mkdtemp()
yaml_path = os.path.join(tmpdir, "pde_cortex.yaml")
with open(yaml_path, "w") as f:
    yaml.dump(exp_dict, f)

exp = SimulationExperiment.from_file(yaml_path)

# --- Initial condition: Gaussian at occipital pole ---
gi = nib.load(mesh_gii)
vertices = gi.darrays[0].data
seed_idx = np.argmin(vertices[:, 1])
seed = vertices[seed_idx]
dist_sq = np.sum((vertices - seed) ** 2, axis=1)
u0 = np.exp(-dist_sq / 2000)

ts = exp.run("pde", u0=u0)
print(f"PDE result: {ts.data.shape}")

# --- Animate: surface only, no axes ---
n_frames = ts.data.sizes["time"]
vmax = float(ts.data.max())
norm = PowerNorm(gamma=0.3, vmin=0, vmax=vmax)

fig, ax = plt.subplots(1, 1, figsize=(4, 4))
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax.axis("off")

def update(frame):
    ax.clear()
    ax.axis("off")
    overlay = ts.data.isel(time=frame, variable=0).values
    plot_surf(gi, overlay=overlay, hemi="rh", view="lateral", ax=ax, cmap="inferno", norm=norm)

anim = FuncAnimation(fig, update, frames=n_frames, interval=200)
out_path = os.path.join(os.path.dirname(__file__), "pde.gif")
anim.save(out_path, writer="pillow", fps=6, savefig_kwargs={"transparent": True, "facecolor": "white"})
plt.close()
print(f"Saved: {out_path}")
