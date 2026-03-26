"""Generate a 3D parameter space with dummy loss landscape coloring."""
# %%
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

fig = plt.figure(figsize=(5, 8), dpi=200)
ax = fig.add_subplot(111, projection='3d')
fig.subplots_adjust(left=-0.15, right=1.1, bottom=0.05, top=0.95)

# Parameter values (actual coordinates used as cell centers)
G_vals = np.linspace(0.01, 0.5, 8)
sigma_vals = np.logspace(-4, -1, 4)
sigma_log = np.log10(sigma_vals)      # [-4, -3, -2, -1]
stim_vals = np.arange(4)              # [0, 1, 2, 3]

# Build cell edges: one cell per value, centered on the value
def cell_edges(centers):
    """Create n+1 edges so that each center sits in the middle of a cell."""
    c = np.asarray(centers, dtype=float)
    edges = np.empty(len(c) + 1)
    edges[1:-1] = 0.5 * (c[:-1] + c[1:])
    edges[0] = c[0] - (edges[1] - c[0])
    edges[-1] = c[-1] + (c[-1] - edges[-2])
    return edges

x_edges = cell_edges(sigma_log)   # 5 edges → 4 cells
y_edges = cell_edges(G_vals)      # 9 edges → 8 cells
z_edges = cell_edges(stim_vals)   # 5 edges → 4 cells

x0, x1 = x_edges[0], x_edges[-1]
y0, y1 = y_edges[0], y_edges[-1]
z0, z1 = z_edges[0], z_edges[-1]

# --- Dummy 3D loss landscape (4 x 8 x 4) ---
np.random.seed(42)
# Smooth gradient: low loss at mid-G + mid-sigma, higher at edges
G_norm = (G_vals - G_vals.min()) / (G_vals.max() - G_vals.min())
sig_norm = (sigma_log - sigma_log.min()) / (sigma_log.max() - sigma_log.min())
stim_norm = stim_vals / stim_vals.max()

loss_3d = np.zeros((len(sigma_log), len(G_vals), len(stim_vals)))
for i, s in enumerate(sig_norm):
    for j, g in enumerate(G_norm):
        for k, st in enumerate(stim_norm):
            # Bowl shape with minimum near center + some variation per stim
            loss_3d[i, j, k] = (
                0.35 * (2*g - 0.8)**2 +
                0.30 * (2*s - 1.0)**2 +
                0.20 * (st - 0.3)**2 +
                0.05 * np.sin(4*g) * np.cos(3*s) +
                0.015 * np.random.randn()
            )
loss_3d = np.clip(loss_3d, 0, 0.6)

# Colormap
cmap = cm.cividis_r
norm = mcolors.Normalize(vmin=0, vmax=0.6)

edge_color = '#2e3440'
edge_lw = 0.8

def quad(p0, p1, p2, p3):
    return [list(p0), list(p1), list(p2), list(p3)]

# --- Draw outer cells: each cell's exposed faces share the same loss color ---
n_s, n_g, n_st = len(sigma_log), len(G_vals), len(stim_vals)

def add_face(verts, color):
    ax.add_collection3d(Poly3DCollection(
        [verts], facecolor=color, edgecolor=edge_color, linewidth=edge_lw, alpha=1.0))

for i in range(n_s):
    for j in range(n_g):
        for k in range(n_st):
            # Only draw cells that have at least one exposed face
            on_top   = (k == n_st - 1)   # z = z1 (top)
            on_front = (i == n_s - 1)    # x = x1 (front)
            on_right = (j == 0)          # y = y0 (right)

            if not (on_top or on_front or on_right):
                continue

            color = cmap(norm(loss_3d[i, j, k]))
            xi, xi1 = x_edges[i], x_edges[i+1]
            yj, yj1 = y_edges[j], y_edges[j+1]
            zk, zk1 = z_edges[k], z_edges[k+1]

            if on_top:
                add_face(quad([xi,yj,zk1],[xi1,yj,zk1],[xi1,yj1,zk1],[xi,yj1,zk1]), color)
            if on_front:
                add_face(quad([xi1,yj,zk],[xi1,yj1,zk],[xi1,yj1,zk1],[xi1,yj,zk1]), color)
            if on_right:
                add_face(quad([xi,yj,zk],[xi1,yj,zk],[xi1,yj,zk1],[xi,yj,zk1]), color)

# --- Axis labels and ticks (centered in cells) ---
ax.set_xlabel('sigma', fontsize=14, labelpad=12, fontweight='bold')
ax.set_xticks(sigma_log)
ax.set_xticklabels([f'{v:.0e}' for v in sigma_vals], fontsize=7, rotation=15)

ax.set_ylabel('G', fontsize=14, labelpad=12, fontweight='bold')
ax.set_yticks(G_vals)
ax.set_yticklabels([f'{v:.2f}' for v in G_vals], fontsize=6)

ax.set_zlabel('stim', fontsize=14, labelpad=10, fontweight='bold')
ax.set_zticks(stim_vals)
ax.set_zticklabels([f'"{i}"' for i in stim_vals], fontsize=8)

# --- Limits and view ---
ax.set_xlim(x0, x1)
ax.set_ylim(y0, y1)
ax.set_zlim(z0, z1)

ax.view_init(elev=25, azim=-60)
ax.set_box_aspect([0.6, 1.2, 0.7])

# --- Colorbar at the bottom ---
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar_ax = fig.add_axes([0.2, 0.02, 0.6, 0.02])
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Loss', fontsize=12, fontweight='bold')
cbar.ax.tick_params(labelsize=8)

fig.savefig('../figures/param_space_loss.png', dpi=200, bbox_inches='tight',
            transparent=True)
print("Saved param_space_loss.png")
