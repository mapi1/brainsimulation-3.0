"""Generate a 3D parameter space visualization — solid block with grid lines."""
# %%
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

fig = plt.figure(figsize=(5, 8), dpi=200)
ax = fig.add_subplot(111, projection='3d')
fig.subplots_adjust(left=-0.15, right=1.1, bottom=0.0, top=0.95)

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
    # interior edges at midpoints
    edges[1:-1] = 0.5 * (c[:-1] + c[1:])
    # outer edges: mirror the half-step
    edges[0] = c[0] - (edges[1] - c[0])
    edges[-1] = c[-1] + (c[-1] - edges[-2])
    return edges

x_edges = cell_edges(sigma_log)   # 5 edges → 4 cells
y_edges = cell_edges(G_vals)      # 9 edges → 8 cells
z_edges = cell_edges(stim_vals)   # 5 edges → 4 cells

x0, x1 = x_edges[0], x_edges[-1]
y0, y1 = y_edges[0], y_edges[-1]
z0, z1 = z_edges[0], z_edges[-1]

# --- Face colors ---
c_top   = '#5e81ac'  # σ × G (blue)
c_front = '#a3be8c'  # G × stim (green)
c_right = "#d88beb"  # σ × stim (yellow)

edge_color = '#2e3440'
edge_lw = 0.8

def quad(p0, p1, p2, p3):
    return [list(p0), list(p1), list(p2), list(p3)]

# Draw each face as individual grid cells (one per parameter value)

# Top face (z=z1): cells in x (sigma) × y (G)
for i in range(len(sigma_log)):
    for j in range(len(G_vals)):
        cell = quad([x_edges[i],y_edges[j],z1], [x_edges[i+1],y_edges[j],z1],
                    [x_edges[i+1],y_edges[j+1],z1], [x_edges[i],y_edges[j+1],z1])
        ax.add_collection3d(Poly3DCollection(
            [cell], facecolor=c_top, edgecolor=edge_color, linewidth=edge_lw, alpha=1.0))

# Front face (x=x1): cells in y (G) × z (stim)
for j in range(len(G_vals)):
    for k in range(len(stim_vals)):
        cell = quad([x1,y_edges[j],z_edges[k]], [x1,y_edges[j+1],z_edges[k]],
                    [x1,y_edges[j+1],z_edges[k+1]], [x1,y_edges[j],z_edges[k+1]])
        ax.add_collection3d(Poly3DCollection(
            [cell], facecolor=c_front, edgecolor=edge_color, linewidth=edge_lw, alpha=1.0))

# Right face (y=y0): cells in x (sigma) × z (stim)
for i in range(len(sigma_log)):
    for k in range(len(stim_vals)):
        cell = quad([x_edges[i],y0,z_edges[k]], [x_edges[i+1],y0,z_edges[k]],
                    [x_edges[i+1],y0,z_edges[k+1]], [x_edges[i],y0,z_edges[k+1]])
        ax.add_collection3d(Poly3DCollection(
            [cell], facecolor=c_right, edgecolor=edge_color, linewidth=edge_lw, alpha=1.0))

# --- Axis labels and ticks (centered in cells) ---
# x-axis = sigma (log10 coords)
ax.set_xlabel('sigma', fontsize=14, labelpad=12, fontweight='bold')
ax.set_xticks(sigma_log)
ax.set_xticklabels([f'{v:.0e}' for v in sigma_vals], fontsize=7, rotation=15)

# y-axis = G (actual values)
ax.set_ylabel('G', fontsize=14, labelpad=12, fontweight='bold')
ax.set_yticks(G_vals)
ax.set_yticklabels([f'{v:.2f}' for v in G_vals], fontsize=6)

# z-axis = stim (index in quotes)
ax.set_zlabel('stim', fontsize=14, labelpad=10, fontweight='bold')
ax.set_zticks(stim_vals)
ax.set_zticklabels([f'"{i}"' for i in stim_vals], fontsize=8)

# --- Limits and view ---
ax.set_xlim(x0, x1)
ax.set_ylim(y0, y1)
ax.set_zlim(z0, z1)

ax.view_init(elev=25, azim=-60)
ax.set_box_aspect([0.6, 1.2, 0.7])

# ax.set_title('4 × 8 × 4 = 128 sims', fontsize=11, pad=8, fontweight='bold')

fig.savefig('../figures/param_space.png', dpi=200, bbox_inches='tight',
            transparent=True)
print("Saved param_space.png")
