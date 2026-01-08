"""Collection of utility functions for the data analysis of NESCAFE 2023 Acoustic Fluctions"""

# Load environment variables from .env file
import os
import pickle

import fsspec
import h5py
import numpy as np
import scipy
import xarray as xr
from dotenv import find_dotenv, load_dotenv
from tqdm import tqdm

load_dotenv(find_dotenv())

# Define global variables

# angle / time bounding boxes for each arrival (chosen by hand)
PEAK_BBOXS = {
    "+7": (101, 101.1, -13, -4),
    "+6": {
        0: (100.855, 100.885, 7.5, 10),
        1: (100.85, 100.885, 8, 10),
        2: (100.85, 100.885, 8, 11),
        3: (100.84, 100.87, 8, 10.5),
        4: (100.84, 100.885, 8.5, 11),
        5: (100.85, 100.88, 7.5, 11),
        6: (100.84, 100.885, 6, 10),
        7: (100.82, 100.88, 6, 9.5),
        8: (100.82, 100.878, 6, 10),
        9: (100.825, 100.88, 5.5, 9),
        10: (100.82, 100.88, 4, 9),
        11: (100.83, 100.885, 4.5, 8),
        12: (100.83, 100.88, 5.5, 8.5),
        13: (100.83, 100.87, 6.5, 9),
        14: (100.82, 100.88, 8, 10.5),
        15: (100.83, 100.875, 7, 10.5),
        16: (100.83, 100.88, 8.5, 11),
        17: (100.83, 100.89, 8, 10.5),
        18: (100.84, 100.88, 8, 10.5),
        19: (100.83, 100.89, 8, 11),
        20: (100.84, 100.89, 8, 10.5),
        21: (100.84, 100.89, 6, 10.5),
        22: (100.84, 100.87, 6, 9),
    },
    "-7": {
        0: (100.89, 100.93, 5.5, 8.5),
        1: (100.89, 100.93, 6.5, 8.5),
        2: (100.89, 100.93, 7, 10),
        3: (100.88, 100.91, 7, 9.5),
        4: (100.89, 100.93, 6.5, 10),
        5: (100.89, 100.93, 5.5, 10),
        6: (100.89, 100.94, 6.5, 9),
        7: (100.88, 100.92, 5, 8.5),
        8: (100.89, 100.94, 4.5, 9),
        9: (100.89, 100.925, 4, 7),
        10: (100.88, 100.94, 2, 8),
        11: (100.89, 100.94, 3, 7),
        12: (100.88, 100.93, 4, 7),
        13: (100.89, 100.93, 4, 7),
        14: (100.86, 100.93, 6, 8),
        15: (100.88, 100.92, 7, 9.5),
        16: (100.89, 100.94, 7, 9.5),
        17: (100.88, 100.95, 6, 9),
        18: (100.9, 100.94, 6.5, 9),
        19: (100.9, 100.94, 5, 8.5),
        20: (100.89, 100.94, 6, 8),
        21: (100.89, 100.94, 5, 10),
        22: (100.87, 100.91, 4.5, 9),
    },
    "-6": (100.65, 100.8, -16, -8),
    "-5b": (100.4, 100.6, 7, 18),
    "-6b": (100.8, 100.975, -22, -14),
    "+6b": (101.2, 101.4, 12, 20),
    "+7b": (101.6, 101.75, -23, -13),
    "-7b": (101.95, 102.1, 13, 21),
    "-8b": (102.35, 102.55, -22, -15),
    "+8b": (102.75, 103, 14, 22),
    "+9b": (103.25, 103.4, -26, -16),
    "-9b": (103.70, 103.85, 14, 26),
    "-10b": (104.2, 104.4, -27, -15),
    "+10b": (104.7, 104.9, 16, 26),
    "+11b": (105.2, 105.4, -27, -17),
    "-11b": (105.8, 106, 18, 27),
    "-12b": (106.4, 106.6, -28, -20),
    "+12b": (106.9, 107.2, 20, 28),
    "+13b": (107.6, 107.8, -30, -21),
    "-13b": (108.3, 108.4, 22, 30),
    "-14b": (108.9, 109.1, -33, -24),
    "+14b": (109.5, 109.85, 22, 30),
    "+15b": (110.3, 110.5, -34, -26),
    "-15b": (110, 111.4, 20, 30),
}


def convert_to_xarray(fn):
    """
    convert .mat / h5 file structure to useful xarray structure.
    Specifically coded to parse the `dmod` .mat file structure for the processed NESCAFE 2023 data.

    Parameters
    ----------
    fn : string
        file path to demodulated data

    Returns
    -------
    xarray.Dataset
        xarray dataset containing the pilot data
    """

    ds = xr.open_dataset(fn, group="/dm", engine="h5netcdf", phony_dims="sort")
    actual_depth = h5py.File(fn)["hmdepact"][:].flatten()

    ds = ds.squeeze().rename_dims({
        "phony_dim_5": "hydrophone",
        "phony_dim_6": "time",
        "phony_dim_7": "jds_dim",
        "phony_dim_8": "string",
    })

    ds = ds.assign_coords({
        "time": ds["tax"],
        "nominal_depth": ("hydrophone", ds["mrcvz"].values),
        "actual_depth": ("hydrophone", actual_depth),
    }).drop_vars(["tax", "mrcvz"])

    # fix complex data
    dmod_c = xr.DataArray(
        ds["dmod"].values.view(dtype=np.complex128), dims=["hydrophone", "time"]
    )
    ds["dmod"] = dmod_c

    # create attributes
    attrs_vars = [
        "Gs",
        "SL",
        "Tswp",
        "dmrms",
        "dmsnr",
        "fc",
        "fhi",
        "flo",
        "gdla",
        "gdr",
        "rcv",
        "rreflat",
        "rreflon",
        "src",
        "sreflat",
        "sreflon",
        "tofs",
    ]
    attrs = {}

    for var in attrs_vars:
        attrs[var] = ds[var].values

    ds.attrs = attrs
    ds = ds.drop_vars(attrs_vars)
    return ds


def open_receptions(time_slice=(90, 120)):
    """
    Open all receptions from the pilot study, and remove tidal time shift
    """
    fs = fsspec.filesystem("")
    fns = fs.glob(f"{os.environ['NESCAFE23_DATADIR']}dmod/dm*.mat")
    with open(
        f"{os.environ['NESCAFE23_DATADIR']}/dataproducts/T_t_estimated.pkl", "rb"
    ) as f:
        dT_t = pickle.load(f)["dT_t"]

        dss = []

    for k, fn in enumerate(tqdm(fns)):
        ds = convert_to_xarray(fn).sel({"time": slice(time_slice[0], time_slice[1])})
        ds = ds["dmod"].assign_coords({"time": ds.time.values + dT_t[k]})

        dss.append(ds)

    return dss


def ULA_conventional_beamform(da, thetas, c0=1500):
    """

    Parameters
    ----------
    da : xr.DataArray
        signal to beamform. should have dimensions ['depth', 'time']
    thetas : np.array
        array of angles (in degrees) to beamform
    """

    zs = da.depth.values
    nzs = len(zs)
    z0 = np.mean(zs)

    # FFT stuff
    dt = float(da.time[1] - da.time[0])
    nt = len(da.time)
    dw = 2 * np.pi / (nt * dt)

    # Nyquist
    wN = 2 * np.pi / (2 * dt)

    # create frequency vector
    f = scipy.fft.fftfreq(nt, dt)
    ws = 2 * np.pi * f

    w = 2 * np.pi * 250

    delts = -(zs - z0)[:, None] * np.sin(np.radians(thetas[None, :])) / (c0)

    da = da.transpose("depth", "time")
    dmodf = scipy.fft.fft(da.values, axis=1)
    dmodf_ps = dmodf[:, :, None] * np.exp(
        1j * (ws[None, :, None] + w) * delts[:, None, :]
    )

    dmod_ps_t = scipy.fft.ifft(dmodf_ps, axis=1)
    dmod_bf_np = np.sum(dmod_ps_t, axis=0)
    dmod_bf = xr.DataArray(
        dmod_bf_np,
        dims=["time", "angle"],
        coords=({"time": da.time.values, "angle": thetas}),
    )

    # flip angles so that positive is down (to match ray code)
    # dmod_bf = dmod_bf.assign_coords({'angle':np.flip(dmod_bf.angle.values)}).sortby('angle')

    return dmod_bf


def ULA_conventional_beamform_nostack(da, thetas, c0=1500):
    """

    Parameters
    ----------
    da : xr.DataArray
        signal to beamform. should have dimensions ['depth', 'time']
    thetas : np.array
        array of angles (in degrees) to beamform
    """

    zs = da.depth.values
    nzs = len(zs)
    z0 = np.mean(zs)

    # FFT stuff
    dt = float(da.time[1] - da.time[0])
    nt = len(da.time)
    dw = 2 * np.pi / (nt * dt)

    # Nyquist
    wN = 2 * np.pi / (2 * dt)

    # create frequency vector
    f = scipy.fft.fftfreq(nt, dt)
    ws = 2 * np.pi * f

    w = 2 * np.pi * 250

    delts = -(zs - z0)[:, None] * np.sin(np.radians(thetas[None, :])) / (c0)

    da = da.transpose("depth", "time")
    dmodf = scipy.fft.fft(da.values, axis=1)
    dmodf_ps = dmodf[:, :, None] * np.exp(
        1j * (ws[None, :, None] + w) * delts[:, None, :]
    )

    dmod_ps_t = scipy.fft.ifft(dmodf_ps, axis=1)
    dmod_bf = xr.DataArray(
        dmod_ps_t,
        dims=["depth", "time", "angle"],
        coords=({"depth": da.depth.values, "time": da.time.values, "angle": thetas}),
    )

    # flip angles so that positive is down (to match ray code)
    # dmod_bf = dmod_bf.assign_coords({'angle':np.flip(dmod_bf.angle.values)}).sortby('angle')

    return dmod_bf, delts


def get_peak_location(dss, Tx_idxs=np.arange(23)):
    """
    get peak location - using hand derived bounding box (`PEAK_BBOXS`), get time / angle index using peak envelope

    Parameters
    ----------
    dss : list
        list of receptions. created with {mod}`tools.open_receptions`,

    Returns
    -------
    arrival_times : dict
    arrival_angles : dict
    arrival_snr : dict
    """

    arrival_times = {}
    arrival_angles = {}
    arrival_snr = {}

    for ray_id in PEAK_BBOXS:
        arrival_times[ray_id] = np.ones(23) * np.nan
        arrival_angles[ray_id] = np.ones(23) * np.nan
        arrival_snr[ray_id] = np.ones(23) * np.nan

    for Tx_num in tqdm(Tx_idxs, desc="Tx num"):
        thetas = np.linspace(-35, 35, 200)
        lower_Rx = (
            dss[Tx_num]
            .swap_dims({"hydrophone": "nominal_depth"})
            .rename({"nominal_depth": "depth"})[32:]
        )

        bf_unstacked, _ = ULA_conventional_beamform_nostack(lower_Rx, thetas)
        bf = bf_unstacked.sum("depth")

        bf_noise = 20 * np.log10(np.abs(bf.sel({"time": slice(90, 95)}).std("time")))

        for ray_id in PEAK_BBOXS:
            if isinstance(PEAK_BBOXS[ray_id], tuple):
                t1, t2, th1, th2 = PEAK_BBOXS[ray_id]
            elif isinstance(PEAK_BBOXS[ray_id], dict):
                t1, t2, th1, th2 = PEAK_BBOXS[ray_id][Tx_num]
            else:
                raise ValueError("PEAK_BBOXS entry must be tuple or dict")

            ## Get argmax values
            env_slice = 20 * np.log10(
                np.abs(bf.sel({"time": slice(t1, t2), "angle": slice(th1, th2)}))
            )
            flat_idx = env_slice.argmax().item()
            tmax_idx, thmax_idx = np.unravel_index(flat_idx, env_slice.shape)
            tmax = env_slice.time[tmax_idx].values
            thmax = env_slice.angle[thmax_idx].values

            arrival_times[ray_id][Tx_num] = tmax
            arrival_angles[ray_id][Tx_num] = thmax
            arrival_snr[ray_id][Tx_num] = env_slice.max() - bf_noise[thmax_idx]

    return arrival_times, arrival_angles, arrival_snr


def get_wavefront_values(dss, arrival_times, arrival_angles):
    wavefront_amplitudes = {}

    for ray_id in PEAK_BBOXS:
        wavefront_amplitudes[ray_id] = np.ones((23, 28), dtype="complex") * np.nan

    for Tx_num in tqdm(range(len(dss)), desc="Tx num"):
        thetas = np.linspace(-35, 35, 200)
        lower_Rx = (
            dss[Tx_num]
            .swap_dims({"hydrophone": "nominal_depth"})
            .rename({"nominal_depth": "depth"})[32:]
        )

        bf_unstacked, _ = ULA_conventional_beamform_nostack(lower_Rx, thetas)

        for ray_id in PEAK_BBOXS:
            bf_single_peak = (
                bf_unstacked.sel({"angle": arrival_angles[ray_id][Tx_num]})
                .assign_coords({
                    "time": bf_unstacked.time.values - arrival_times[ray_id][Tx_num]
                })
                .sel({"time": slice(-0.15, 0.15)})
            )

            wavefront_amplitude = bf_single_peak.sel({"time": 0}).values
            wavefront_amplitudes[ray_id][Tx_num, :] = wavefront_amplitude.copy()

    return wavefront_amplitudes
