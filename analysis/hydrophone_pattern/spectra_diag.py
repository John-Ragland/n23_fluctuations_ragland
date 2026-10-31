"""F_phi and F_chi of the ducted rays with and without the hydrophone pattern removed (tilt removed in all)."""
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from n23_fluctuations_ragland import tools

SP = Path(__file__).parent
VARIANTS = ["none", "phase", "magnitude", "both"]
COLORS = {"none": "k", "phase": "C0", "magnitude": "C2", "both": "C3"}
# variants that coincide (F_phi: none = magnitude, phase = both; F_chi: none ~ phase) stay visible: wide solid
# lines underneath, narrow dashed lines on top
STYLES = {
    "none": {"lw": 3.2, "ls": "-", "alpha": 0.45, "zorder": 2},
    "phase": {"lw": 3.2, "ls": "-", "alpha": 0.45, "zorder": 2},
    "magnitude": {"lw": 1.4, "ls": (0, (4, 2)), "alpha": 1.0, "zorder": 3},
    "both": {"lw": 1.4, "ls": (0, (1.5, 1.5)), "alpha": 1.0, "zorder": 4},
}
real_path = tools.wavefront_values_path

# the phase-only and magnitude-only wavefronts are computed once and cached next to this script
paths = {}
dss = None
for v in VARIANTS:
    if real_path(True, v).exists():
        paths[v] = real_path(True, v)
        continue
    paths[v] = SP / f"wavefront_values_tilt_removed_{v}.pkl"
    if not paths[v].exists():
        dss = dss if dss is not None else tools.open_receptions()
        wv = tools.get_wavefront_values(dss, remove_array_tilt=True, remove_pattern=v)
        paths[v].write_bytes(pickle.dumps(wv))

tools.wavefront_values_path = lambda remove_array_tilt, remove_pattern="none": paths[remove_pattern]
est = {v: tools.estimate_F_phi_chi(remove_array_tilt=True, remove_pattern=v) for v in VARIANTS}

rays = ["+7", "-7", "+6", "-6"]
fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True, sharey="row")
for j, ray_id in enumerate(rays):
    for v in VARIANTS:
        m, _, _, phi_lm, phi_ls, chi_lm, chi_ls = est[v]
        for i, (lm, ls) in enumerate([(phi_lm, phi_ls), (chi_lm, chi_ls)]):
            ax = axes[i, j]
            ax.loglog(m, lm[ray_id], color=COLORS[v], label=v, **STYLES[v])
            ax.fill_between(m, lm[ray_id] / ls[ray_id], lm[ray_id] * ls[ray_id], color=COLORS[v], alpha=0.07, lw=0)
    axes[0, j].set_title(ray_id)
    axes[1, j].set_xlabel("m [cycles/m]")
    for ax in axes[:, j]:
        ax.grid(which="both", alpha=0.25)
axes[0, 0].set_ylabel(r"$F_\phi$ [rad$^2$ m]")
axes[1, 0].set_ylabel(r"$F_\chi$ [m]")
axes[0, 0].legend(title="pattern removed", fontsize=8)
fig.suptitle(
    "Ducted F_phi / F_chi, array tilt removed, hydrophone pattern removed or not "
    "(bands: jackknife SE; 'none' and 'magnitude' zero-fill elements 4, 5, 20 in phase, 'phase'/'both' use all 28)\n"
    "F_phi: none = magnitude and phase = both exactly (the gain does not change the phase); "
    "F_chi: none ~ phase. Wide faint lines underneath, dashed/dotted on top.",
    fontsize=9,
)
fig.tight_layout()
fig.savefig(SP / "spectra_pattern_removal.png", dpi=150)

# ratio to 'none', to read the changes off directly
fig2, axes2 = plt.subplots(2, 4, figsize=(15, 6), sharex=True, sharey="row")
m = est["none"][0]
for j, ray_id in enumerate(rays):
    for v in VARIANTS[1:]:
        for i, k in enumerate([3, 5]):
            axes2[i, j].semilogx(m, est[v][k][ray_id] / est["none"][k][ray_id], color=COLORS[v], label=v)
    axes2[0, j].set_title(ray_id)
    axes2[1, j].set_xlabel("m [cycles/m]")
    for ax in axes2[:, j]:
        ax.axhline(1, color="k", lw=0.8)
        ax.grid(which="both", alpha=0.25)
axes2[0, 0].set_ylabel(r"$F_\phi$ / $F_\phi$(none)")
axes2[1, 0].set_ylabel(r"$F_\chi$ / $F_\chi$(none)")
axes2[0, 0].legend(fontsize=8)
fig2.tight_layout()
fig2.savefig(SP / "spectra_pattern_removal_ratio.png", dpi=150)

for v in VARIANTS[1:]:
    for name, k in (("phi", 3), ("chi", 5)):
        r = [np.median(est[v][k][r] / est["none"][k][r]) for r in rays]
        print(f"{v:9s} F_{name} median ratio to none per ray {rays}: {np.round(r, 2)}")
