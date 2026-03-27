"""
Bayesian prior → posterior mockup using tvbo.datamodel Distribution on Parameters.
Generates a figure showing how parameter distributions update after fitting.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform

from tvbo.datamodel.schema import Distribution, Parameter, Range

# ---------------------------------------------------------------------------
# Define parameters with prior distributions using TVBO's schema classes
# ---------------------------------------------------------------------------

param = Parameter(
    name="G",
    value="2.0",
    description="Global coupling scaling",
    distribution=Distribution(
        name="Gaussian",
        domain=Range(lo="0.5", hi="4.0"),
    ),
)


def sample_prior(param):
    """Sample from TVBO Distribution spec → prior (wide Gaussian)."""
    d = param.distribution
    lo, hi = float(d.domain.lo), float(d.domain.hi)
    mu = (lo + hi) / 2
    sigma = (hi - lo) / 4  # wide prior
    return mu, sigma


def mock_posterior(mu_prior, sigma_prior):
    """Simulate a tighter posterior shifted slightly from the prior."""
    rng = np.random.default_rng(42)
    mu_post = mu_prior + rng.normal(0, sigma_prior * 0.15)
    sigma_post = sigma_prior * 0.35  # narrower
    return mu_post, sigma_post


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(3.2, 2.2), constrained_layout=True)

colors_prior = "#9ab5b8"
colors_post = "#2b6b6b"

mu_pr, sig_pr = sample_prior(param)
mu_po, sig_po = mock_posterior(mu_pr, sig_pr)

lo, hi = float(param.distribution.domain.lo), float(param.distribution.domain.hi)
x = np.linspace(lo - sig_pr, hi + sig_pr, 300)

prior_y = norm.pdf(x, mu_pr, sig_pr)
post_y = norm.pdf(x, mu_po, sig_po)

ax.fill_between(x, prior_y, alpha=0.25, color=colors_prior)
ax.plot(x, prior_y, color=colors_prior, lw=1.5)
ax.fill_between(x, post_y, alpha=0.35, color=colors_post)
ax.plot(x, post_y, color=colors_post, lw=2)

ax.axvline(float(param.value), color="#d46a6a", ls="--", lw=1, alpha=0.7)

ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

fig.savefig(
    __file__.replace(".py", ".png"),
    dpi=200, bbox_inches="tight", transparent=True,
)
plt.show()
print(f"Saved to {__file__.replace('.py', '.png')}")
