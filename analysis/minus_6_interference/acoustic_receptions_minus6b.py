"""Acoustic_Receptions figure with the mean -6b plane-wave arrival line drawn on panel (a).

The line is t(z) = t0 - (z - z0) sin(theta) / c0, the arrival a beam at theta lines up: the steering
delay d = -(z - z0) sin(theta) / c0 used by
``nescafe23.ULA_conventional_beamform`` (and the picker's ``ULA_conventional_beamform_nostack``),
with z the *nominal* depths (what the picker beamformed on), z0 their mean, and c0 = 1500 m/s.
Panel (a) is averaged in *actual* depth, so each element's delay is plotted at that element's
transmission-mean actual depth (the array sits ~76 m below nominal, with ~10 m rather than 9 m
spacing, and its center moves ~50 m rms between transmissions).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from nescafe23.processing import PEAK_LOWER_QC

import nescafe23 as n23
from n23_fluctuations_ragland import figures

C0 = 1500.0
RAY_ID = "-6b"
GOOD_QC = (1, 5)

out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent

peaks = n23.open_peak_locations()
times = np.asarray(peaks["times"][RAY_ID])
angles = np.asarray(peaks["angles"][RAY_ID])
good = np.isin(PEAK_LOWER_QC[RAY_ID], GOOD_QC)

t0, theta = np.nanmean(times[good]), np.nanmean(angles[good])
print(f"{RAY_ID}: {good.sum()}/23 good-QC transmissions")
print(f"  good-QC mean: t0 = {t0:.4f} s, theta = {theta:.2f} deg")
print(f"  all-Tx mean:  t0 = {np.nanmean(times):.4f} s, theta = {np.nanmean(angles):.2f} deg")
print(f"  good-QC std:  t = {np.nanstd(times[good]) * 1e3:.1f} ms, theta = {np.nanstd(angles[good]):.2f} deg")

dss = figures._receptions()
z_nom = dss[0].nominal_depth.values[32:]
z_act = np.mean([ds.actual_depth.values[32:] for ds in dss], axis=0)
print(f"  mean nominal lower depth = {z_nom.mean():.1f} m, mean actual = {z_act.mean():.1f} m")

cfg = figures.PublicationFiguresConfig(usetex=False)
fig = figures.plot_acoustic_receptions(cfg)
ax_pulse = fig.axes[0]

zs = z_act
t_line = t0 - (z_nom - z_nom.mean()) * np.sin(np.radians(theta)) / C0
ax_pulse.plot(t_line, zs, c=figures.whoi_colors[1], lw=1.2, ls="--", alpha=0.85, zorder=100)
ax_pulse.text(
    t_line.max() + 0.01, zs[0] + 15, RAY_ID,
    color=figures.whoi_colors[1], fontsize=9, fontweight="bold", ha="left", va="top", zorder=100,
)

for ext in ("png", "pdf"):
    fn = out_dir / f"Acoustic_Receptions_minus6b.{ext}"
    fig.savefig(fn, dpi=500)
    print(f"saved {fn}")
