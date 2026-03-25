# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Introduction
#
# This tutorial demonstrates how to use TVB-Optim to fit functional connectivity (FC) data from resting-state fMRI. We use the Reduced Wong-Wang (RWW) neural mass model to simulate brain activity, convert it to BOLD signal using a hemodynamic response function, and optimize model parameters to match empirical FC patterns.

# %%
import os

# Detect if running in Google Colab
def is_colab():
    return "google.colab" in str(get_ipython().extension_manager.loaded) if get_ipython() else False

# Detect if running on EBRAINS JupyterLab
def is_ebrains():
    return "EBRAINS" in os.environ.get("LAB_IMAGE_NAME", "")

# Some dependencies are pre-installed on EBRAINS and Colab but may be missing
# in other environments — install them automatically if needed
if is_colab() or is_ebrains():
    print("Running in Colab or EBRAINS - installing dependencies...")
    get_ipython().system("pip install -q 'jax==0.6.1'")
    get_ipython().system("pip install -q --no-warn-conflicts tvboptim")
    print("✓ Dependencies installed!")
# In other environments (e.g. local Jupyter), dependencies are assumed to be
# already installed — run `pip install tvboptim` manually if needed

# %% [markdown]
# The workflow includes:
#
# - Building a whole-brain network with the RWW model
# - Simulating BOLD signal from neural activity
# - Computing functional connectivity from BOLD
# - Optimizing global and region-specific parameters to fit target FC

# %%
# Set up environment
cpu = True
if cpu:
    N_devices = 8
    os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={N_devices}'

# Import all required libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import jax
import jax.numpy as jnp
import copy
import optax
import h5py

# Jax enable x64
jax.config.update("jax_enable_x64", True)

# Import from tvboptim
from tvboptim.types import Parameter, Space, GridAxis
from tvboptim.types.stateutils import show_parameters
from tvboptim.execution import ParallelExecution, SequentialExecution
from tvboptim.optim.optax import OptaxOptimizer
from tvboptim.optim.callbacks import MultiCallback, DefaultPrintCallback, SavingCallback

# Network dynamics imports
from tvboptim.experimental.network_dynamics import Network, solve, prepare
from tvboptim.experimental.network_dynamics.dynamics.tvb import ReducedWongWang
from tvboptim.experimental.network_dynamics.coupling import LinearCoupling, FastLinearCoupling
from tvboptim.experimental.network_dynamics.graph import DenseGraph
from tvboptim.experimental.network_dynamics.solvers import Heun, BoundedSolver
from tvboptim.experimental.network_dynamics.noise import AdditiveNoise

# BOLD monitoring
from tvboptim.observations.tvb_monitors.bold import Bold

# Observation functions
from tvboptim.observations.observation import compute_fc, fc_corr, rmse


# %% [markdown]
# We enable 64-bit precision to get reliable gradient information.

# %%
jax.config.update("jax_enable_x64", True)

# %% [markdown]
# ## Loading Structural Data and Target FC
#
# We load the Lobar8 parcellation (8 lobar regions per hemisphere, 16 total) structural connectivity and empirical functional connectivity from HCP Young Adults dTOR tractography.

# %%
# Load structural connectivity and FC from Lobar8 HDF5 file
h5_path = "../data/tpl-MNI152NLin2009cAsym_rec-avgMatrix_atlas-Lobar8_desc-SCFC_relmat.h5"
with h5py.File(h5_path, "r") as f:
    weights = f["edges/weight/data"][:]
    lengths = f["edges/length/data"][:]
    fc_target = f["edges/fc/data"][:]

# Region labels from the Lobar8 atlas
region_labels = np.array([
    "LH_Frontal", "LH_Parietal", "LH_Temporal", "LH_Occipital",
    "LH_Cingulate", "LH_Insular", "LH_Subcortical", "LH_Cerebellum",
    "RH_Frontal", "RH_Parietal", "RH_Temporal", "RH_Occipital",
    "RH_Cingulate", "RH_Insular", "RH_Subcortical", "RH_Cerebellum",
])

# Normalize weights to [0, 1] range
weights = weights / np.max(weights)
n_nodes = weights.shape[0]

# %%

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.1, 4.05), sharey=True)
im1 = ax1.imshow(weights, cmap="cividis", vmax=0.5)
ax1.set_title("Structural Connectivity")
ax1.set_xlabel("Region")
ax1.set_ylabel("Region")
cbar1 = fig.colorbar(im1, ax=ax1, shrink=0.74, label="Connection Strength [a.u.]", extend='max')

# im2 = ax2.imshow(lengths, cmap="cividis")
# ax2.set_title("Tract Lengths")
# ax2.set_xlabel("Region")
# cbar2 = fig.colorbar(im2, ax=ax2, shrink=0.74, label="Tract Length [mm]")

im2 = ax2.imshow(fc_target, cmap="cividis")
ax2.set_title("Target FC")
ax2.set_xlabel("Region")
cbar2 = fig.colorbar(im2, ax=ax2, shrink=0.74, label="FC Correlation")
plt.tight_layout()

# %% [markdown]
# ## The Reduced Wong-Wang Model
#
# The Reduced Wong-Wang model is a biophysically-based neural mass model that describes the dynamics of NMDA-mediated synaptic gating. It captures the slow dynamics relevant for resting-state fMRI and has been widely used for modeling whole-brain functional connectivity.
#
# The model describes the evolution of synaptic gating variable S:
#
# $$\frac{dS}{dt} = -\frac{S}{\tau_s} + (1-S) \cdot H(x) \cdot \gamma$$
#
# where $x = w \cdot J_N \cdot S + I_o + G \cdot c$ combines local recurrence ($w$), external input ($I_o$), and long-range coupling ($G \cdot c$), and $H(x)$ is a sigmoidal transfer function.
#
# Key parameters:
#
# - `w`: Excitatory recurrence strength (local feedback)
# - `I_o`: External input current
# - `G` (coupling strength): Global scaling of long-range connections
#
# ## Building the Network Model
#
# We combine the RWW dynamics with structural connectivity to create a whole-brain network model.

# %%
# Create network components
graph = DenseGraph(weights, region_labels=region_labels)
dynamics = ReducedWongWang(w=0.5, I_o=0.34, INITIAL_STATE=(0.1,))
coupling = FastLinearCoupling(local_states=["S"], G=0.1)
noise = AdditiveNoise(sigma=0.01, apply_to="S")

# Assemble the network
network = Network(
    dynamics=dynamics,
    coupling={'instant': coupling},
    graph=graph,
    noise=noise
)

# %% [markdown]
# ## Preparing and Running the Simulation
#
# We prepare the network for simulation and run an initial transient to reach a quasi-stationary state.

# %%
# Prepare simulation: compile model and initialize state
t1 = 180_000  # Total simulation duration (ms) - 2 minutes
dt = 4.0      # Integration timestep (ms)
solver = BoundedSolver(Heun(), low=0.0, high=1.0)  # Bound S to [0, 1]
model, state = prepare(network, solver, t1=t1, dt=dt)

# First simulation: run transient to reach quasi-stationary state
result_init = model(state)

# Update network with final state as new initial conditions
network.update_history(result_init)
model, state = prepare(network, solver, t1=t1, dt=dt)

# Second simulation: quasi-stationary dynamics
result = model(state)

# %% [markdown]
# ## Computing BOLD Signal
#
# We convert the neural activity (synaptic gating S) to simulated BOLD signal using a hemodynamic response function. The BOLD monitor downsamples the neural activity and convolves it with a canonical HRF kernel.

# %%
# Create BOLD monitor with standard parameters
bold_monitor = Bold(
    period=1000.0,          # BOLD sampling period (1 TR = 1000 ms)
    downsample_period=4.0,  # Intermediate downsampling matches dt
    voi=0,                  # Monitor first state variable (S)
    history=result_init     # Use initial state as warm start
)

# Apply BOLD monitor to simulation result
bold_result = bold_monitor(result)

# %%

from matplotlib.colors import Normalize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.1, 3.0375))

# Plot raw neural activity (first 1000 ms)
t_max_idx = int(1000 / dt)
time_raw = result.time[:t_max_idx]
data_raw = result.data[:t_max_idx, 0, :]

num_lines = data_raw.shape[1]
cmap = plt.cm.cividis
mean_values = np.mean(data_raw, axis=0)
norm = Normalize(vmin=np.min(mean_values), vmax=np.max(mean_values))
for i in range(num_lines):
    color = cmap(norm(mean_values[i]))
    ax1.plot(time_raw, data_raw[:, i], color=color, linewidth=0.5)

ax1.text(0.95, 0.95, "Raw Neural Activity", transform=ax1.transAxes, fontsize=10,
         ha='right', va='top', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
ax1.set_xlabel("Time [ms]")
ax1.set_ylabel("S [a.u.]")

# Plot BOLD signal (first 60 TRs)
t_bold_max = 60
time_bold = bold_result.time[:t_bold_max]
data_bold = bold_result.data[:t_bold_max, 0, :]

num_lines = data_bold.shape[1]
mean_values = np.mean(data_bold, axis=0)
norm = Normalize(vmin=np.min(mean_values), vmax=np.max(mean_values))
for i in range(num_lines):
    color = cmap(norm(mean_values[i]))
    ax2.plot(time_bold, data_bold[:, i], color=color, linewidth=0.8)

ax2.text(0.95, 0.95, "BOLD Signal", transform=ax2.transAxes, fontsize=10,
         ha='right', va='top', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
ax2.set_xlabel("Time [s]")
ax2.set_ylabel("BOLD [a.u.]")

plt.tight_layout()


# %% [markdown]
# ## Defining Observations and Loss
#
# Functional connectivity (FC) measures the temporal correlation between BOLD signals from different brain regions. We define an observation function that simulates BOLD and computes FC, and a loss function that quantifies the mismatch with empirical FC.

# %%

def observation(state):
    """Compute functional connectivity from simulated BOLD signal."""
    # Run simulation
    result = model(state)
    # Convert to BOLD
    bold = bold_monitor(result)
    # Compute FC, skipping first 20 TRs to avoid transient effects
    fc = compute_fc(bold, skip_t=20)
    return fc

def loss(state):
    """Compute RMSE between simulated and empirical FC."""
    fc = observation(state)
    # return 1- fc_corr(fc, fc_target)
    return rmse(fc, fc_target)


# %%

# Calculate initial FC
fc_initial = np.array(observation(state))

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.1, 3.54375))

# Plot both FC matrices
for ax_current, fc_matrix, title_prefix in zip([ax1, ax2], [fc_target, fc_initial], ["Target FC", "Initial FC"]):
    fc_matrix = np.copy(fc_matrix)
    np.fill_diagonal(fc_matrix, np.nan)  # Set diagonal to NaN
    im = ax_current.imshow(fc_matrix, cmap='cividis', vmax=0.9)

    ax_current.set_xticks([])
    ax_current.set_yticks([])
    ax_current.set_xlabel('')
    ax_current.set_ylabel('')

    # Calculate correlation and RMSE for title
    if title_prefix == "Initial FC":
        corr_value = fc_corr(fc_initial, fc_target)
        rmse_value = rmse(fc_initial, fc_target)
        title = f"{title_prefix}\nr = {corr_value:.3f}, RMSE = {rmse_value:.3f}"
    else:
        title = title_prefix

    # Add title as annotation
    ax_current.annotate(title,
                       xy=(0.25, 0.95),
                       xycoords='axes fraction',
                       ha='left', va='top',
                       fontsize=10, fontweight='bold',
                       color='black',
                       bbox=dict(boxstyle='round,pad=0.3',
                                facecolor='white', alpha=0.9))

plt.tight_layout()

# %% [markdown]
# ## Parameter Exploration
#
# Before optimization, we explore how the model parameters affect FC quality. We systematically vary the excitatory recurrence `w` and global coupling strength `G` across a 2D grid and compute the loss for each combination.

# %%

# Create grid for parameter exploration
n = 32

# Set up parameter axes for exploration
grid_state = copy.deepcopy(state)
grid_state.dynamics.w = GridAxis(0.001, 1.0, n)
grid_state.coupling.instant.G = GridAxis(0.001, 2.0, n)

# Create space (product creates all combinations of w and G)
grid = Space(grid_state, mode="product")

# Parallel execution across 8 processes
exec = ParallelExecution(loss, grid, n_pmap=8)
# Alternative: Sequential execution
# exec = SequentialExecution(loss, grid)
exploration_results = exec.run()

# %%

# Prepare data for visualization
pc = grid.collect()
G_vals = pc.coupling.instant.G.flatten()
w_vals = pc.dynamics.w.flatten()

# Get parameter ranges
G_min, G_max = min(G_vals), max(G_vals)
w_min, w_max = min(w_vals), max(w_vals)

# Create figure and axis
fig, ax = plt.subplots(figsize=(8.1, 4.05))

# Create the heatmap
im = ax.imshow(jnp.stack(exploration_results).reshape(n, n).T,
              cmap='cividis_r',
              extent=[G_min, G_max, w_min, w_max],
              origin='lower',
              aspect='auto',
              interpolation='hamming')

# Add colorbar and labels
cbar = plt.colorbar(im, label="Loss (RMSE)")
# cbar = plt.colorbar(im, label="Loss (Fc Correlation)")
ax.set_xlabel('Global Coupling (G)')
ax.set_ylabel('Excitatory Recurrence (w)')
ax.set_title("Parameter Exploration")

plt.tight_layout()

# %% [markdown]
# ## Gradient-Based Optimization
#
# We use gradient-based optimization to find the best global parameters (same values for all regions) that minimize the FC mismatch. JAX's automatic differentiation computes gradients through the entire simulation pipeline.

# %%
import gc
gc.collect()

# %%
# Mark parameters as optimizable
state.coupling.instant.G = Parameter(state.coupling.instant.G)
state.dynamics.w = Parameter(state.dynamics.w)

# Create and run optimizer
cb = MultiCallback([
    DefaultPrintCallback(every=10),
    SavingCallback(key="state", save_fun=lambda *args: args[1])  # Save updated state
])

opt = OptaxOptimizer(loss, optax.adam(0.01), callback=cb)
fitted_state, fitting_data = opt.run(state, max_steps=200)

# %%

# Prepare data for visualization
pc = grid.collect()
G_vals = pc.coupling.instant.G
w_vals = pc.dynamics.w

# Get parameter ranges
G_min, G_max = min(G_vals), max(G_vals)
w_min, w_max = min(w_vals), max(w_vals)

# Create figure and axis
fig, ax = plt.subplots(figsize=(8.1, 4.725))

# Create the heatmap
im = ax.imshow(jnp.stack(exploration_results).reshape(n, n).T,
               cmap='cividis_r',
               extent=[G_min, G_max, w_min, w_max],
               origin='lower',
               aspect='auto',
               interpolation='hamming')

# Mark initial value
G_init = state.coupling.instant.G.value
w_init = state.dynamics.w.value
ax.scatter(G_init, w_init, color='white', s=100, marker='o',
           edgecolors='k', linewidths=2, zorder=5)

# Add annotation
ax.annotate('Initial', xy=(G_init, w_init),
            xytext=(G_init, w_init+0.05*(w_max-w_min)),
            color='white', fontweight='bold', ha='center', zorder=5,
            path_effects=[path_effects.withStroke(linewidth=3, foreground='black')])

# Add fitted value point
G_fit = fitted_state.coupling.instant.G.value
w_fit = fitted_state.dynamics.w.value
ax.scatter(G_fit, w_fit, color='white', s=100, marker='o',
           edgecolors='k', linewidths=2, zorder=5)

# Add annotation for the fitted value
ax.annotate('Optimized', xy=(G_fit, w_fit),
            xytext=(G_fit, w_fit-0.08*(w_max-w_min)),
            color='white', fontweight='bold', ha='center', zorder=5,
            path_effects=[path_effects.withStroke(linewidth=3, foreground='black')])

# Add optimization path points
G_route = np.array([ds.coupling.instant.G.value for ds in fitting_data["state"].save])
w_route = np.array([ds.dynamics.w.value for ds in fitting_data["state"].save])
ax.scatter(G_route[::2], w_route[::2], color='white', s=15, marker='o',
           linewidths=1, zorder=4, edgecolors='k')

# Remove axes ticks and labels
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('')
ax.set_ylabel('')

plt.tight_layout()

# %% [markdown]
# ## Heterogeneous Optimization
#
# Global parameters (same for all regions) may not capture region-specific variations needed for optimal FC fit. We now make parameters heterogeneous: each brain region gets its own `w` and `I_o` values, while keeping `G` global.

# %%
# Copy already optimized state and make parameters regional
fitted_state_het = copy.deepcopy(fitted_state)

# # Make w regional (one value per node)
fitted_state_het.dynamics.w.shape = (n_nodes,)

# Also make I_o regional and mark as optimizable
fitted_state_het.dynamics.I_o = Parameter(fitted_state_het.dynamics.I_o)
fitted_state_het.dynamics.I_o.shape = (n_nodes,)

# Keep global coupling fixed at optimized value
fitted_state_het.coupling.instant.G = fitted_state_het.coupling.instant.G.value

# fitted_state_het.graph._weights = Parameter(fitted_state_het.graph.weights)
show_parameters(fitted_state_het)


# %%

opt_het = OptaxOptimizer(loss, optax.adam(0.001), callback=cb)
fitted_state_het, fitting_data_het = opt_het.run(fitted_state_het, max_steps=200)

# %% [markdown]
# ## Comparing Global vs Regional Parameters
#
# Let's compare the FC quality from global (homogeneous) vs regional (heterogeneous) parameter fits.

# %%
# Compute FC for both optimization strategies
fc_global = np.array(observation(fitted_state))
fc_regional = np.array(observation(fitted_state_het))

# %%

# Create 2x2 figure: FC heatmaps with Target for visual comparison
fig, axes = plt.subplots(2, 2, figsize=(8.1, 8.1))

fc_list = [fc_target, fc_initial, fc_global, fc_regional]
fc_labels = ["Target FC", "Initial FC", "Global Parameters", "Regional Parameters"]

# Compute shared color range across all FC matrices (excluding diagonal)
all_offdiag = np.concatenate([fc[np.triu_indices_from(fc, k=1)] for fc in fc_list])
vmin_shared, vmax_shared = float(np.min(all_offdiag)), float(np.max(all_offdiag))
vmin_shared, vmax_shared = 0, 0.75

for ax_current, fc_matrix, label in zip(axes.flat, fc_list, fc_labels):
    fc_plot = np.copy(fc_matrix)
    np.fill_diagonal(fc_plot, np.nan)
    im = ax_current.imshow(fc_plot, cmap='cividis', vmin=vmin_shared, vmax=vmax_shared)
    ax_current.set_xticks([])
    ax_current.set_yticks([])

    if label == "Target FC":
        title = label
    else:
        corr_value = fc_corr(fc_matrix, fc_target)
        rmse_value = rmse(fc_matrix, fc_target)
        title = f"{label}\nr = {corr_value:.3f}, RMSE = {rmse_value:.3f}"

    ax_current.set_title(title, fontsize=10, fontweight='bold')

plt.tight_layout()

# %%

# Scatter plots: simulated vs empirical FC
triu_idx = np.triu_indices_from(fc_target, k=1)
fc_target_triu = fc_target[triu_idx]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8.1, 3.24), sharey=True, sharex=True)

for ax_current, fc_matrix, label in zip([ax1, ax2, ax3], [fc_initial, fc_global, fc_regional], ["Initial", "Global Parameters", "Regional Parameters"]):
    fc_triu = fc_matrix[triu_idx]
    corr_value = fc_corr(fc_matrix, fc_target)
    rmse_value = rmse(fc_matrix, fc_target)
    ax_current.scatter(fc_target_triu, fc_triu, alpha=0.3, s=10, color='royalblue', edgecolors='none')
    ax_current.plot([fc_target_triu.min(), fc_target_triu.max()],
                    [fc_target_triu.min(), fc_target_triu.max()],
                    'k--', linewidth=1.5)
    ax_current.set_xlabel('Empirical FC')
    ax_current.set_title(f'{label}\nr = {corr_value:.3f}, RMSE = {rmse_value:.3f}')
    ax_current.grid(True, alpha=0.3)
    ax_current.set_aspect('equal', adjustable='box')

ax1.set_ylabel('Simulated FC')

plt.tight_layout()

# %% [markdown]
# ## Fitted Heterogeneous Parameters
#
# Let's examine the fitted region-specific parameters and their relationship to structural connectivity.

# %%

# Calculate mean incoming connectivity for each region
mean_connectivity = np.mean(weights, axis=1)

# Extract fitted regional parameters
w_fitted = fitted_state_het.dynamics.w.value.flatten()
I_o_fitted = fitted_state_het.dynamics.I_o.value.flatten()

# Get global optimization values for reference
w_global = fitted_state.dynamics.w.value
I_o_global = fitted_state.dynamics.I_o  # Not optimized in global fit, but initial value

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.1, 3.24))

# Plot w vs mean connectivity
ax1.scatter(mean_connectivity, w_fitted, alpha=0.7, s=30, color='royalblue', edgecolors='k', linewidths=0.5)
ax1.axhline(w_global, color='red', linestyle='--', linewidth=2, label=f'Global w = {w_global:.3f}')
ax1.set_xlabel('Mean Incoming Connectivity')
ax1.set_ylabel('Fitted w (Excitatory Recurrence)')
ax1.set_title('Regional Excitatory Recurrence Parameters')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)

# Plot I_o vs mean connectivity
ax2.scatter(mean_connectivity, I_o_fitted, alpha=0.7, s=30, color='royalblue', edgecolors='k', linewidths=0.5)
ax2.axhline(I_o_global, color='red', linestyle='--', linewidth=2, label=f'Initial I_o = {I_o_global:.3f}')
ax2.set_xlabel('Mean Incoming Connectivity')
ax2.set_ylabel('Fitted I_o (External Input)')
ax2.set_title('Regional External Input Parameters')
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
# %%
