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


def plot_ray_lignup(source_depth_slider, receiver_range_slider):

    source_idx = np.argmin(np.abs(source_depths - source_depth_slider))
    range_idx = np.argmin(np.abs(receiver_ranges - receiver_range_slider * 1e3))

    # Calculate the run index from the meshgrid
    run_idx = range_idx * len(source_depths) + source_idx

    fig = plt.figure(figsize=(10, 4))

    t_m6 = eigs[run_idx].ts[0][np.where(eigs[run_idx].ray_id[0] == "-6.0")[0][0], -1]
    DT = t_m6 - 100.75

    # open mean pulse
    fn1 = f"{os.environ['NESCAFE23_DATADIR']}dataproducts/mean_pulse_lower.nc"
    fn2 = f"{os.environ['NESCAFE23_DATADIR']}dataproducts/mean_pulse_upper.nc"
    fn3 = f"{os.environ['NESCAFE23_DATADIR']}dataproducts/mean_pulse_lower_noise.nc"
    fn4 = f"{os.environ['NESCAFE23_DATADIR']}dataproducts/mean_pulse_upper_noise.nc"

    mean_pulse_lower = xr.open_dataarray(fn1)
    mean_pulse_upper = xr.open_dataarray(fn2)
    mean_pulse_lower_noise = xr.open_dataarray(fn3)
    mean_pulse_upper_noise = xr.open_dataarray(fn4)

    (20 * np.log10(np.abs(mean_pulse_lower)) - mean_pulse_lower_noise).plot(
        x="time", cmap=cmps.plasma_coolwarm(), vmin=0
    )
    (20 * np.log10(np.abs(mean_pulse_upper)) - mean_pulse_upper_noise).plot(
        x="time", cmap=cmps.plasma_coolwarm(), vmin=0, add_colorbar=False
    )

    plt.scatter(
        rays[run_idx].ts[:, -1] - DT, -rays[run_idx].zs[:, -1], c="k", s=5, lw=0
    )
    plt.xlim([100, 105])
    plt.ylim([1500, 0])

    plt.title(
        f"Source Depth = {ssource_depths[run_idx]} [m], Range = {rreceiver_ranges[run_idx]} [km]\t$\\Delta \\tau$ = {DT:0.3f} [s]"
    )
    # plt.hlines(-1072, 90, 110)
    plt.close()
    return pn.pane.Matplotlib(fig, tight=True)


source_depth_slider = pn.widgets.DiscreteSlider(
    name="Source Depth [m]", options=list(source_depths), value=source_depths[0]
)

receiver_range_slider = pn.widgets.DiscreteSlider(
    name="Receiver Range [km]",
    options=list(receiver_ranges / 1e3),
    value=receiver_ranges[0] / 1e3,
)

bound_plot = pn.bind(
    plot_ray_lignup,
    source_depth_slider=source_depth_slider,
    receiver_range_slider=receiver_range_slider,
)
app = pn.Column(source_depth_slider, receiver_range_slider, bound_plot)

app.servable()
