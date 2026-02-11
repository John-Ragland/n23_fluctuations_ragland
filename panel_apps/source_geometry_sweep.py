import os
import pickle

import cmps
import matplotlib.pyplot as plt
import numpy as np
import panel as pn
import xarray as xr
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

fn = f"{os.environ['NESCAFE23_DATADIR']}dataproducts/ray_geo_sweep.pkl"

with open(fn, "rb") as f:
    param_sweep = pickle.load(f)

ssource_depths = param_sweep["source_depths"]
rreceiver_ranges = param_sweep["receiver_ranges"]
rays = param_sweep["rays"]
eigs = param_sweep["eigs"]
envi = param_sweep["envi"]

# undo the meshgrid
source_depths = np.unique(ssource_depths)
receiver_ranges = np.unique(rreceiver_ranges)

# Load and pre-compute static data
mean_pulse_lower = xr.open_dataarray(
    f"{os.environ['NESCAFE23_DATADIR']}dataproducts/mean_pulse_lower.nc"
)
mean_pulse_upper = xr.open_dataarray(
    f"{os.environ['NESCAFE23_DATADIR']}dataproducts/mean_pulse_upper.nc"
)
mean_pulse_lower_noise = xr.open_dataarray(
    f"{os.environ['NESCAFE23_DATADIR']}dataproducts/mean_pulse_lower_noise.nc"
)
mean_pulse_upper_noise = xr.open_dataarray(
    f"{os.environ['NESCAFE23_DATADIR']}dataproducts/mean_pulse_upper_noise.nc"
)

static_lower = 20 * np.log10(np.abs(mean_pulse_lower)) - mean_pulse_lower_noise
static_upper = 20 * np.log10(np.abs(mean_pulse_upper)) - mean_pulse_upper_noise

# Create figure once at module level
fig, ax = plt.subplots(figsize=(10, 4))

# Plot the static colormeshes
static_lower.plot(x="time", cmap=cmps.plasma_coolwarm(), vmin=0, ax=ax)
static_upper.plot(
    x="time", cmap=cmps.plasma_coolwarm(), vmin=0, add_colorbar=False, ax=ax
)

# Create empty scatter plot that will be updated
scatter = ax.scatter([], [], c="k", s=5, lw=0)
ax.set_xlim([100, 105])
ax.set_ylim([1500, 0])
ax.set_xlabel("time")
ax.set_ylabel("depth")
title = ax.set_title("")

# Create the matplotlib pane once
mpl_pane = pn.pane.Matplotlib(fig, tight=True)


def update_plot(source_depth_slider, receiver_range_slider):
    source_idx = np.argmin(np.abs(source_depths - source_depth_slider))
    range_idx = np.argmin(np.abs(receiver_ranges - receiver_range_slider * 1e3))
    run_idx = range_idx * len(source_depths) + source_idx

    t_m6 = eigs[run_idx].ts[0][np.where(eigs[run_idx].ray_id[0] == "-6.0")[0][0], -1]
    DT = t_m6 - 100.75

    # Update only scatter data
    scatter.set_offsets(np.c_[rays[run_idx].ts[:, -1] - DT, -rays[run_idx].zs[:, -1]])
    title.set_text(
        f"Source Depth = {ssource_depths[run_idx]} [m], Range = {rreceiver_ranges[run_idx] / 1e3:.1f} [km]\t$\\Delta \\tau$ = {DT:0.3f} [s]"
    )

    # Trigger re-render
    mpl_pane.param.trigger("object")


source_depth_slider = pn.widgets.DiscreteSlider(
    name="Source Depth [m]", options=list(source_depths), value=source_depths[0]
)

receiver_range_slider = pn.widgets.DiscreteSlider(
    name="Receiver Range [km]",
    options=list(receiver_ranges / 1e3),
    value=receiver_ranges[0] / 1e3,
)

# Use pn.depends decorator instead of pn.bind
pn.depends(
    source_depth_slider.param.value, receiver_range_slider.param.value, watch=True
)(update_plot)

app = pn.Column(source_depth_slider, receiver_range_slider, mpl_pane)

app.servable()
