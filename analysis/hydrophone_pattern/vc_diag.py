"""Vertical coherence of the ducted rays with and without the hydrophone pattern removed (tilt removed in all),
as in Ducted_VC: tools.jackknife_vertical_coherence on all 28 elements, no zero-fill."""
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from n23_fluctuations_ragland import tools

SP = Path(__file__).parent
VARIANTS = ["none", "phase", "magnitude", "both"]
COLORS = {"none": "k", "phase": "C0", "magnitude": "C2", "both": "C3"}
OFFSET = {"none": -1.5, "phase": -0.5, "magnitude": 0.5, "both": 1.5}  # [m], to separate the error bars


def load(v):
    path = tools.wavefront_values_path(True, v)
    if not path.exists():
        path = SP / f"wavefront_values_tilt_removed_{v}.pkl"  # made by spectra_diag.py
    return pickle.loads(path.read_bytes())


# tools.fit_vertical_coherence starts from A0 = 4e11, ~10x below these MC, and least_squares sometimes stops
# there (z0 = 100 m, A0 = 4e11) and reports success. Start it from the data's own scale instead.
_fit = tools.fit_vertical_coherence
tools.fit_vertical_coherence = lambda dz, MC, L0=100.0, A0_init=None: _fit(
    dz, MC, L0, np.asarray(MC)[1] if A0_init is None else A0_init
)

rays = ["+7", "-7", "+6", "-6"]
vc = {v: {r: tools.jackknife_vertical_coherence(load(v)[r], 9) for r in rays} for v in VARIANTS}

fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
for j, ray_id in enumerate(rays):
    ref = vc["none"][ray_id]
    for v in VARIANTS:
        d = vc[v][ray_id]
        se_z0 = np.sqrt(d["z0_var"])
        axes[0, j].errorbar(
            d["lags"] + OFFSET[v], d["VC"], yerr=np.sqrt(d["VC_var"]), color=COLORS[v],
            marker="o", ms=2.5, lw=1, capsize=1.5, label=f"{v}: z0 = {d['z0']:.0f} ± {se_z0:.0f} m",
        )
        lag = np.linspace(0, d["lags"][-1], 300)
        axes[0, j].plot(lag, np.exp(-0.5 * (lag / d["z0"]) ** 2), color=COLORS[v], lw=0.8, ls="--")
        if v != "none":
            axes[1, j].plot(d["lags"], d["VC"] - ref["VC"], color=COLORS[v], marker="o", ms=2.5, lw=1)
    axes[1, j].fill_between(ref["lags"], -np.sqrt(ref["VC_var"]), np.sqrt(ref["VC_var"]), color="k", alpha=0.12, lw=0)
    axes[1, j].axhline(0, color="k", lw=0.8)
    axes[0, j].set_title(ray_id)
    axes[0, j].legend(fontsize=7, loc="upper right")
    axes[1, j].set_xlabel("lag [m]")
    for ax in axes[:, j]:
        ax.grid(alpha=0.25)
axes[0, 0].set_ylabel("vertical coherence (MC / A0)")
axes[1, 0].set_ylabel("VC − VC(none)\n(grey: ± SE of none)")
fig.suptitle(
    "Ducted vertical coherence, array tilt removed, hydrophone pattern removed or not "
    "(bars: jackknife SE; dashed: Gaussian fit exp(-(dz/z0)^2 / 2), fitted from A0_init = MC at 9 m)",
    fontsize=9,
)
fig.tight_layout()
fig.savefig(SP / "vc_pattern_removal.png", dpi=150)

for ray_id in rays:
    print(ray_id, "  ".join(f"{v}: z0 {vc[v][ray_id]['z0']:.1f} ± {np.sqrt(vc[v][ray_id]['z0_var']):.1f}" for v in VARIANTS))
