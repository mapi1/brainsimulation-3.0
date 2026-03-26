import jaxley as jx
from jaxley.synapses import IonotropicSynapse
from bsplot.surface import yrotate, zrotate, xrotate, perspective, translate
from bsplot.data.surface import get_surface_geometry
from bsplot.graph import (
    create_network,
    get_centers_from_surface_parc,
    plot_network_on_surface,
)
from matplotlib.patches import ConnectionPatch
from nibabel.freesurfer.io import read_annot
import matplotlib.pyplot as plt

from bsplot import data, style
import numpy as np
import pickle, os
from tvbo import Network, Dynamics
import bsplot

plt.rcParams["figure.dpi"] = 100

_CACHE = os.path.join(os.path.dirname(__file__), ".graph_cache.pkl")

if os.path.exists(_CACHE):
    with open(_CACHE, "rb") as f:
        _c = pickle.load(f)
    G = _c["G"]
    cortical_entries = _c["cortical_entries"]
    node_to_aparc = _c["node_to_aparc"]
    overlay_lh = _c["overlay_lh"]
    overlay_rh = _c["overlay_rh"]
    vertices_lh = _c["vertices_lh"]
    vertices_rh = _c["vertices_rh"]
    labels_lh = _c["labels_lh"]
    labels_rh = _c["labels_rh"]
    print("Loaded graph data from cache.")
else:
    # Load tvbo DK connectivity
    sc = Network.from_db(atlas="DesikanKilliany", rec="avgMatrix")
    W = sc.matrix("weight")

    # Load surface geometry and parcellation
    vertices_lh, faces_lh = get_surface_geometry(
        template="fsaverage", hemi="lh", density="164k"
    )
    vertices_rh, faces_rh = get_surface_geometry(
        template="fsaverage", hemi="rh", density="164k"
    )
    labels_lh, colors_lh, names_lh = read_annot(
        "/Applications/freesurfer/7.4.1/subjects/fsaverage/label/lh.aparc.annot"
    )
    labels_rh, colors_rh, names_rh = read_annot(
        "/Applications/freesurfer/7.4.1/subjects/fsaverage/label/rh.aparc.annot"
    )
    centers_lh = get_centers_from_surface_parc(vertices_lh, labels_lh)
    centers_rh = get_centers_from_surface_parc(vertices_rh, labels_rh)

    # Build aparc name → label index lookup (lh/rh share same label set)
    name_to_idx = {
        (n.decode() if isinstance(n, bytes) else str(n)): idx
        for idx, n in enumerate(names_lh)
    }

    # Map tvbo cortical nodes (ctx-lh-* / ctx-rh-*) to FreeSurfer aparc indices
    cortical_entries = []
    for i, node in enumerate(sc.nodes):
        lbl = node.label
        if lbl.startswith("ctx-lh-"):
            aparc = lbl[7:]
            if aparc in name_to_idx:
                cortical_entries.append((i, "L", aparc, name_to_idx[aparc]))
        elif lbl.startswith("ctx-rh-"):
            aparc = lbl[7:]
            if aparc in name_to_idx:
                cortical_entries.append((i, "R", aparc, name_to_idx[aparc]))

    # Sort: LH first (by aparc label index), then RH
    cortical_entries.sort(key=lambda x: (x[1] != "L", x[3]))
    reorder_idx = np.array([e[0] for e in cortical_entries])

    # Cortical-only SC submatrix
    sc_cortical = W[np.ix_(reorder_idx, reorder_idx)]

    # Surface-based centers for visualization
    centers_src = {"L": centers_lh, "R": centers_rh}
    centers_matched = {
        i: centers_src[hemi][aparc_idx]
        for i, (_, hemi, _, aparc_idx) in enumerate(cortical_entries)
    }
    labels_matched = list(range(len(cortical_entries)))

    G = create_network(
        centers_matched,
        sc_cortical,
        labels=labels_matched,
        threshold_percentile=92,
        directed=False,
    )

    for node in G.nodes():
        G.nodes[node]["strength"] = sum(
            d["weight"] for _, _, d in G.edges(node, data=True)
        )

    connected_nodes = [n for n in G.nodes() if G.degree(n) > 0]
    node_to_aparc = {
        i: (hemi, aparc_idx)
        for i, (_, hemi, _, aparc_idx) in enumerate(cortical_entries)
    }

    # Log-scale node strengths and edge weights
    for node in G.nodes():
        G.nodes[node]["strength"] = np.log1p(G.nodes[node]["strength"])
    for u, v, d in G.edges(data=True):
        d["weight"] = np.log1p(d["weight"])

    overlay_lh = np.full(len(vertices_lh), np.nan)
    overlay_rh = np.full(len(vertices_rh), np.nan)
    for node_id in connected_nodes:
        hemi, aparc_idx = node_to_aparc[node_id]
        strength = G.nodes[node_id]["strength"]
        if hemi == "L":
            overlay_lh[labels_lh == aparc_idx] = strength
        else:
            overlay_rh[labels_rh == aparc_idx] = strength

    with open(_CACHE, "wb") as f:
        pickle.dump(
            {
                "G": G,
                "cortical_entries": cortical_entries,
                "node_to_aparc": node_to_aparc,
                "overlay_lh": overlay_lh,
                "overlay_rh": overlay_rh,
                "vertices_lh": vertices_lh,
                "vertices_rh": vertices_rh,
                "labels_lh": labels_lh,
                "labels_rh": labels_rh,
            },
            f,
        )
    print("Graph data cached.")


def pick_zoom_nodes(nodes_list):
    """Pick 3 spatially spread nodes (top/middle/bottom by y), highest degree per tercile."""
    by_y = sorted(nodes_list, key=lambda n: G.nodes[n]["pos"][1], reverse=True)
    n = len(by_y)
    thirds = [by_y[: n // 3], by_y[n // 3 : 2 * n // 3], by_y[2 * n // 3 :]]
    return [max(t, key=lambda n: G.degree(n)) for t in thirds]


# --- Pick 3 RH nodes (right insets) and 3 LH nodes (left insets) ---
rh_nodes = [n for n in G.nodes() if node_to_aparc[n][0] == "R" and G.degree(n) > 0]
lh_nodes = [n for n in G.nodes() if node_to_aparc[n][0] == "L" and G.degree(n) > 0]
zoom_nodes_rh = pick_zoom_nodes(rh_nodes)
zoom_nodes_lh = pick_zoom_nodes(lh_nodes)
zoom_nodes = zoom_nodes_rh  # keep backward-compat for projection loop below
zoom_colors = ["crimson", "darkorange", "royalblue"]
zoom_seeds_rh = [42, 7, 99]
zoom_seeds_lh = [13, 55, 77]
zoom_freqs = [(10, 3, 25), (8, 5, 20), (12, 4, 30)]  # alpha/theta/beta per node

# keep backward-compat alias used by projection loop
zoom_seeds = zoom_seeds_rh

# --- Project nodes to 2D using cached surface vertices ---
verts = vertices_lh.astype(np.float32)
center = (verts.max(0) + verts.min(0)) / 2
scale = max(verts.max(0) - verts.min(0))

view_angle, xr, zr = 0.0, 0.0, 0.0

MVP = (
    perspective(25, 1, 1, 100)
    @ translate(0, 0, -3)
    @ yrotate(view_angle)
    @ zrotate(zr)
    @ xrotate(xr)
)

proj_verts = (verts - center) / scale
V_all = np.c_[proj_verts, np.ones(len(proj_verts))] @ MVP.T
V_all /= V_all[:, 3:4]
pxmin, pxmax = V_all[:, 0].min(), V_all[:, 0].max()
pymin, pymax = V_all[:, 1].min(), V_all[:, 1].max()

ROT = yrotate(view_angle) @ zrotate(zr) @ xrotate(xr)
V_world = np.c_[verts, np.ones(len(verts))] @ ROT.T
wxmin, wxmax = V_world[:, 0].min(), V_world[:, 0].max()
wymin, wymax = V_world[:, 1].min(), V_world[:, 1].max()
sx = (wxmax - wxmin) / (pxmax - pxmin)
sy = (wymax - wymin) / (pymax - pymin)

# Project each node
node_2d = {}
for zn in zoom_nodes_rh + zoom_nodes_lh:
    pos = np.array(G.nodes[zn]["pos"])
    np_ = (pos - center) / scale
    np4 = np.append(np_, 1) @ MVP.T
    np4 /= np4[3]
    nx = wxmin + (np4[0] - pxmin) * sx
    ny = wymin + (np4[1] - pymin) * sy
    node_2d[zn] = (nx, ny)

# --- Main figure ---
fig = plt.figure(figsize=(12, 6), dpi=100)
ax = fig.add_axes([0.22, 0.0, 0.56, 1.0])

plot_network_on_surface(
    G,
    ax=ax,
    template="fsaverage",
    density="10k",
    hemi="lh",
    view="top",
    surface_alpha=.3,
    node_radius=2,
    node_color="auto",
    node_data_key="strength",
    node_cmap="viridis",
    edge_radius=0.1,
    edge_cmap="viridis",
    edge_data_key="weight",
    edge_scale={"weight": 5, "mode": "log"},
    node_scale={"strength": 2, "mode": "log"},
    overlay=overlay_lh,
    cmap="viridis",
    threshold=0,
    parcellated=True,
)

bsplot.plot_slice(
    bsplot.templates.bigbrain,
    ax=ax,
    view="horizontal",
    slice_mm=10,
    cmap="gray",
    zorder=-1,
    hide_hemi='left'
)
bsplot.plot_slice(
    bsplot.templates.bigbrain,
    ax=ax,
    view="horizontal",
    slice_mm=10,
    cmap="gray",
    zorder=-1,
    alpha=0
)


ax.axis("off")

# --- Shared inset layout ---
inset_width = 0.17
inset_height = 0.25
gap = 0.06
total = 3 * inset_height + 2 * gap
top_offset = (1.0 - total) / 2
inset_bottoms = [top_offset + (2 - i) * (inset_height + gap) for i in range(3)]

# Right insets: tight to brain right edge (0.22 + 0.56 = 0.78)
inset_left_rh = 0.78
# Left insets: tight to brain left edge (0.22)
inset_left_lh = 0.22 - inset_width

# --- Build jaxley small-cell network (for LH bottom inset) ---
_fname = (
    "/Users/leonmartin_bih/work_data/toolboxes/jaxley/docs/tutorials/data/morph.swc"
)
_cell = jx.read_swc(_fname, ncomp=1)
jx_net = jx.Network([_cell] * 5)
jx.connect(jx_net.cell(0).soma.branch(0).comp(0), jx_net[2, 0, 0], IonotropicSynapse())
jx.connect(jx_net.cell(0).soma.branch(0).comp(0), jx_net[3, 0, 0], IonotropicSynapse())
jx.connect(jx_net.cell(0).soma.branch(0).comp(0), jx_net[4, 0, 0], IonotropicSynapse())
jx.connect(jx_net.cell(1).soma.branch(0).comp(0), jx_net[2, 0, 0], IonotropicSynapse())
jx.connect(jx_net.cell(1).soma.branch(0).comp(0), jx_net[3, 0, 0], IonotropicSynapse())
jx.connect(jx_net.cell(1).soma.branch(0).comp(0), jx_net[4, 0, 0], IonotropicSynapse())
jx_net.rotate(-90)
jx_net.cell(0).move(0, 900)
jx_net.cell(1).move(0, 1500)
jx_net.cell(2).move(900, 600)
jx_net.cell(3).move(900, 1200)
jx_net.cell(4).move(900, 1800)

# --- Preload dTOR SC matrix and tractogram path ---
_TCK = "/Users/leonmartin_bih/tools/bsplot/docs/data/dTOR_10K_sample_subsampled_10k.tck"
_dtor_sc = np.log1p(Network.from_db(atlas="DesikanKilliany", rec="dTOR").matrix("weight"))

# --- Preload dynamics simulations (cached, 100ms window, interesting only) ---
_DYN_CACHE = os.path.join(os.path.dirname(__file__), ".dynamics_cache.pkl")
if os.path.exists(_DYN_CACHE):
    with open(_DYN_CACHE, "rb") as f:
        _dyn_traces = pickle.load(f)
    print(f"Loaded dynamics cache ({len(_dyn_traces)} models).")
else:
    _dyn_traces = {}
    for _name in sorted(Dynamics.list_db()):
        _d = Dynamics.from_db(_name)
        _res = _d.run(verbose=0, save=False)
        dt = _res.sample_period  # ms
        n_100ms = int(100.0 / dt) if dt > 0 else 1000
        # Take last 100ms (post-transient)
        _sv0 = _res.data[-n_100ms:, 0, 0, 0].copy()
        rng = _sv0.max() - _sv0.min()
        if rng > 0.01:  # only interesting (non-flat) dynamics
            _dyn_traces[_name] = _sv0
    with open(_DYN_CACHE, "wb") as f:
        pickle.dump(_dyn_traces, f)
    print(f"Simulated and cached {len(_dyn_traces)} interesting dynamics models.")
_DYN_MODELS = sorted(_dyn_traces.keys())

zoom_r = 5
t = np.linspace(0, 2, 1000)


def make_signal(seed, freqs):
    np.random.seed(seed)
    return (
        0.6 * np.sin(2 * np.pi * freqs[0] * t)
        + 0.3 * np.sin(2 * np.pi * freqs[1] * t + 0.5)
        + 0.15 * np.sin(2 * np.pi * freqs[2] * t)
        + 0.1 * np.random.randn(len(t))
    )


def draw_inset(zn, color, seed, freqs, inset_x, connector_side, idx):
    """Draw rectangle on brain, timeseries inset, and connecting lines.
    connector_side: 'left' (inset is to the right of the node) or 'right' (inset is to the left).
    """
    zname = cortical_entries[zn][2]
    nx, ny = node_2d[zn]

    rect = plt.Rectangle(
        (nx - zoom_r, ny - zoom_r),
        2 * zoom_r,
        2 * zoom_r,
        linewidth=2.5,
        edgecolor=color,
        facecolor="none",
        zorder=10,
    )
    ax.add_patch(rect)

    ax_ts = fig.add_axes([inset_x, inset_bottoms[idx], inset_width, inset_height])
    ax_ts.set_facecolor("white")

    if connector_side == "right" and idx == 0:
        # Top-left inset: image (left 30%) + dynamics grid (right 70%)
        _img = plt.imread(os.path.join(os.path.dirname(__file__), "3PopMeanField.png"))
        img_w = inset_width * 0.30
        grid_w = inset_width * 0.70

        # Left: circuit diagram image — use set_aspect to avoid squashing
        ax_img = fig.add_axes([inset_x, inset_bottoms[idx], img_w, inset_height])
        ax_img.set_zorder(ax_ts.get_zorder() - 1)
        ax_img.imshow(_img, aspect="equal")
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        for sp in ax_img.spines.values():
            sp.set_visible(False)

        # Right: grid of interesting dynamics timeseries
        n_models = len(_DYN_MODELS)
        ncols = int(np.ceil(np.sqrt(n_models)))
        nrows = int(np.ceil(n_models / ncols))
        gx0 = inset_x + img_w
        gy0 = inset_bottoms[idx]
        cell_w = grid_w / ncols
        cell_h = inset_height / nrows
        _viridis = plt.cm.viridis
        for gi, _name in enumerate(_DYN_MODELS[:nrows * ncols]):
            row, col = divmod(gi, ncols)
            cx = gx0 + col * cell_w
            cy = gy0 + (nrows - 1 - row) * cell_h  # top-to-bottom
            ax_cell = fig.add_axes([cx, cy, cell_w, cell_h])
            trace = _dyn_traces[_name]
            # Normalize to [0,1]
            tr_min, tr_max = trace.min(), trace.max()
            tr = (trace - tr_min) / (tr_max - tr_min)
            clr = _viridis(gi / max(n_models - 1, 1))
            ax_cell.plot(tr, color=clr, linewidth=0.3)
            ax_cell.set_xlim(0, len(tr))
            ax_cell.set_ylim(-0.05, 1.05)
            ax_cell.axis("off")

        # Use outer ax_ts as visible frame (no background)
        ax_ts.set_xticks([])
        ax_ts.set_yticks([])
        ax_ts.set_facecolor("none")
        ax_ts.patch.set_alpha(0)
        ax_ts.set_title("Dynamical Systems Database", fontsize=8)


    elif connector_side == "right" and idx == 1:
        # Middle-left: tractogram (left 60%) + SC matrix (right 40%) inside single frame
        tract_w = inset_width * 0.6
        sc_w = inset_width * 0.4
        # Draw tractogram in left portion
        ax_tract = fig.add_axes([inset_x, inset_bottoms[idx], tract_w, inset_height])
        ax_tract.set_zorder(ax_ts.get_zorder() - 1)
        bsplot.streamlines.plot_tractogram(
            _TCK, ax=ax_tract, view="sagittal", subsample=2000, alpha=0.6, linewidth=0.1, cmap='viridis'
        )
        ax_tract.set_xticks([])
        ax_tract.set_yticks([])
        for sp in ax_tract.spines.values():
            sp.set_visible(False)

        # Draw SC matrix in right portion
        ax_sc = fig.add_axes([inset_x + tract_w, inset_bottoms[idx], sc_w, inset_height])
        ax_sc.set_zorder(ax_ts.get_zorder() - 1)
        ax_sc.imshow(_dtor_sc, cmap="viridis", interpolation="none")
        ax_sc.set_box_aspect(1)
        ax_sc.set_xticks([])
        ax_sc.set_yticks([])
        for sp in ax_sc.spines.values():
            sp.set_visible(False)

        # Use the outer ax_ts as the single visible frame
        ax_ts.set_xticks([])
        ax_ts.set_yticks([])
        ax_ts.set_facecolor("none")
        ax_ts.patch.set_alpha(0)
        ax_ts.set_title("Network Database", fontsize=8)

    elif connector_side == "right" and idx == 2:
        # Bottom-left inset: jaxley small-cell network
        jx_net.vis(
            ax=ax_ts,
            cell_plot_kwargs=dict(linewidth=0.2),
            synapse_plot_kwargs={"linewidth": 0.5},
            synapse_scatter_kwargs={"s": 2},
        )
        ax_ts.set_title("Multi-Scale Model Support", fontsize=8)
        ax_ts.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    else:
        ax_ts.plot(t, make_signal(seed, freqs), color="grey", linewidth=1)
        ax_ts.set_ylabel("[a.u.]")
        ax_ts.set_title(zname, fontsize=8)
        ax_ts.set_xlim(0, 2)
        if idx == 2:
            ax_ts.set_xlabel("Time (s)")
        else:
            ax_ts.set_xticklabels([])

    for spine in ax_ts.spines.values():
        spine.set_visible(True)
        spine.set_color(color)

    # Connection lines: right-inset connects from node's right edge to inset's left
    # left-inset connects from node's left edge to inset's right
    if connector_side == "left":
        corners = [
            ((nx + zoom_r, ny + zoom_r), (0, 1)),
            ((nx + zoom_r, ny - zoom_r), (0, 0)),
        ]
    else:
        corners = [
            ((nx - zoom_r, ny + zoom_r), (1, 1)),
            ((nx - zoom_r, ny - zoom_r), (1, 0)),
        ]

    for (cx, cy), (ix, iy) in corners:
        fig.add_artist(
            ConnectionPatch(
                xyA=(cx, cy),
                coordsA=ax.transData,
                xyB=(ix, iy),
                coordsB=ax_ts.transAxes,
                color=color,
                linewidth=1.5,
                linestyle="--",
                alpha=0.5,
            )
        )


# --- Right insets (RH nodes) ---
for idx, (zn, color, seed, freqs) in enumerate(
    zip(zoom_nodes_rh, zoom_colors, zoom_seeds_rh, zoom_freqs)
):
    draw_inset(zn, color, seed, freqs, inset_left_rh, connector_side="left", idx=idx)

# --- Left insets (LH nodes) ---
for idx, (zn, color, seed, freqs) in enumerate(
    zip(zoom_nodes_lh, zoom_colors, zoom_seeds_lh, zoom_freqs)
):
    draw_inset(zn, color, seed, freqs, inset_left_lh, connector_side="right", idx=idx)

fig.savefig("/Users/leonmartin_bih/projects/TVB-O/tvb-o-ptim/code/fig/tvbo-network-insets.png", dpi=500)

plt.show()
