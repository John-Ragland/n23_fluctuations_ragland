"""Collection of utility functions for the data analysis of NESCAFE 2023 Acoustic Fluctions"""

# Load environment variables from .env file
import os
import pickle

import fsspec
import gsw
import h5py
import numpy as np
import pygenray as pr
import scipy
import xarray as xr
import xrsignal as xrs
from dotenv import find_dotenv, load_dotenv
from matplotlib import pyplot as plt
from scipy import io, signal
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

# dictionary containing tuples for matching forward and backward eigenrays. These pairings are hand-picked
# notebook for handing picking: `notebooks/auxiliary/Ray_forward_backward_matching.ipynb`
# (front_idx, back_idx)

EIG_INDEXS = {
    "+7": [(0, 0), (1, 1)],
    "-7": [(0, 2), (1, 1), (2, 0)],
    "-6": [(0, 0)],
    "+6": [(0, 0)],
    "-5b": [(0, 0), (1, 1)],
    "-6b": [(0, 0)],
    "+6b": [(0, 0)],
    "+7b": [(0, 0)],
    "-7b": [(0, 0)],
    "-8b": [(0, 0)],
    "+8b": [(0, 0)],
    "+9b": [(1, 0)],
    "-9b": [(0, 0)],
    "-10b": [(0, 0)],
    "+10b": [(0, 0)],
    "+11b": [(0, 0)],
    "-11b": [(0, 0)],
    "-12b": [(0, 0)],
    "+12b": [(0, 0)],
    "+13b": [(0, 0)],
    "-13b": [(0, 0)],
    "-14b": [(0, 0)],
    "+14b": [(0, 0)],
    "+15b": [(0, 0)],
    "-15b": [(0, 0)],
}


def aog_vsdg(pr, tem, sal):
    """
    Compute sound speed in seawater using the Del Grosso equation.
    Modified from aog_vsdg.m matlab function

    Parameters
    ----------
    pr : array like
        pressure in units dbar
    tem : array like
        in-situ temperature in degrees celsius
    sal : array like
        salinity in parts per thousand

    Returns
    -------
    cs : array like
        sound speed in meters/second

    Reference
    ---------
    New equation for the speed of sound in natural waters
    V. A. Del Grosso,  J.Acoust.Soc.Am., V56,N4,Oct74
    """

    pr = pr / 9.80665  # convert pressure to kg/cm**2
    c0 = 1402.392
    vt = (
        0.501109398873e01 - (0.550946843172e-01 - 0.221535969240e-03 * tem) * tem
    ) * tem

    vs = (0.132952290781e01 + 0.128955756844e-03 * sal) * sal

    vp = (0.156059257041e00 + (0.244998688441e-04 - 0.883392332513e-08 * pr) * pr) * pr
    vstp = (
        -0.127562783426e-01 * tem * sal
        + 0.635191613389e-02 * tem * pr
        + 0.265484716608e-07 * tem * tem * pr * pr
        - 0.159349479045e-05 * tem * pr * pr
        + 0.522116437235e-09 * tem * pr * pr * pr
        - 0.438031096213e-06 * tem * tem * tem * pr
        - 0.161674495909e-08 * sal * sal * pr * pr
        + 0.968403156410e-04 * tem * tem * sal
        + 0.485639620015e-05 * tem * sal * sal * pr
        - 0.340597039004e-03 * tem * sal * pr
    )

    cs = c0 + vt + vs + vp + vstp

    return cs


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


def get_fctd(fn, filt=True, z_smooth=100, r_smooth=10e3, r_shift=0):
    """
    Open, shift in range, and smooth fast ctd data

    Parameters
    ----------
    fn : string
        file path to fctd .mat file
    filt : bool
        whether to spatially filter the profile or not
    z_smooth : float
        spatial corner (in meters) for z low pass filter
    r_smooth : float
        spatial corner (in meters) for r low pass filter
    r_shift : float
        range value to shift fctd by (in meters). Currently only supports values between [0, 47722].
        positive values shifts front to right (towards SION)
    """

    fctd_mat = io.loadmat(fn)

    fctd = xr.DataArray(
        fctd_mat["cc"],
        dims=["range", "depth"],
        coords={
            "range": fctd_mat["r"].flatten() * 1000,
            "depth": fctd_mat["z"].flatten(),
            "latitude": ("range", fctd_mat["lat"].flatten()),
            "longitude": ("range", fctd_mat["lon"].flatten()),
        },
    ).sortby("range")

    # Add 10000 meter
    fctd_pilot = xr.concat(
        (fctd, xr.ones_like(fctd[:, -2:-1]).assign_coords({"depth": [10000]}) * 1640),
        dim="depth",
    )

    # interpolate to Δr=500, Δz = 1
    fctd_grid = fctd_pilot.interp({
        "range": np.arange(0, fctd.range[-1], 500),
        "depth": np.arange(0, 6000),
    })

    if filt:
        bz, az = signal.butter(4, 1 / z_smooth, fs=1)
        br, ar = signal.butter(4, 1 / r_smooth, fs=1 / 500)

        fctd_smooth = xrs.filtfilt(
            xrs.filtfilt(fctd_grid, dim="range", b=br, a=ar), dim="depth", b=bz, a=az
        )
    else:
        fctd_smooth = fctd_grid

    fctd_slice = fctd_smooth.sel({"range": slice(0, 151000)})
    fctd_flip = fctd_slice.assign_coords({
        "range": np.flip(fctd_slice.range.values)
    }).sortby("range")

    return fctd_flip


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


def parse_ray_id(ray_id: str) -> str:
    """
    convert ray_id into pygenray compatible format

    Parameters
    ----------
    ray_id : str
        Ray identifier string (e.g., '-7', '+10b').

    Returns
    -------
    ray_id_pr : str
        Converted ray identifier in pygenray format (e.g., '-7.0', '+10.0b').
    (sign, number, suffix) : tuple
        Parsed components of the ray identifier:
    """
    sign = ray_id[0]  # '+' or '-'
    if ray_id.endswith("b"):
        number_str = ray_id[1:-1]  # digits between sign and 'b'
        suffix = "b"
    else:
        number_str = ray_id[1:]  # digits after sign
        suffix = ""

    number = int(number_str)

    # Convert to pygenray format: negative signs appear, positive don't; number as float with 1 decimal
    if sign == "-":
        ray_id_pr = f"-{number}.0{suffix}"
    else:
        ray_id_pr = f"{number}.0{suffix}"

    return ray_id_pr, (sign, number, suffix)


def compute_ray_path_variables(
    ray_id: str,
    eigs: pr.EigenRays,
    eigs_backward: pr.EigenRays,
    envi: pr.OceanEnvironment2D,
    envi_back: pr.OceanEnvironment2D,
    source_depth: float,
    receiver_depth: float,
    receiver_range: float,
    fc: float,
    N: xr.DataArray,
    dcdz_p: xr.DataArray,
    zeta0: float,
    N0: float,
    jstar: float,
    eig_idx_forward: int = 0,
    eig_idx_backward: int = 0,
    latitude: float = 39.192499999,
    dtheta: float = 1e-5,
):
    """
    Compute variables along ray path that are used in path integral and WFT calculations
        for calculations of fluctuation statistics

    Parameters
    ----------
    rayid : str
        ray identifier (number of ray turning points with sign of launch angle)
    eigs : pr.Eigenrays
        eigenrays object containing all solved eigenrays for the environment
    eigs_backward : pr.EigenRays
        eigenrays object containing all solved eigenrays with backwards propagation (starting at receiver).
    envi : pr.OceanEnvironment2D
        environment object used to solve forward eigenrays
    envi_back : pr.Environment
        environment object used to solve backward eigenrays.
        It should have a perfectly flipped environment (i.e. range coordinate flipped).
    source_depth : float
        source depth [m], used to calculate forward eigenrays
    receiver_depth : float
        receiver depth [m], used to calculate forward eigenrays
    receiver_range : float
        receiver range [m], used to calculate forward eigenrays
    fc : float
        center acoustic frequency [Hz]
    N : xr.DataArray
        range-dependant buoyancy frequency profile. Should have dimensions ['range', 'depth']. **Must have units [rad/s]**
    dcdz_p : xr.DataArray
        potential sound speed gradient. Should have same dimensions and coords as envi.sound_speed, N, ... etc.
        **Must have units [1/s]**
        Calculate as $\\frac{dC}{dz} - \\frac{dC(p_{ref} = 0, \\theta(x), \\S(x))}{dz}_{adiabatic}$, and $\\theta(x)$ is potential temperature.
        See footnote on p. 47 of [@Colosi2016].
    zeta0 : float
        reference displacement of GM spectrum
    N0 : float
        reference buoyancy frequency of GM spectrum. **Must have units [rad/s]**
    jstar : float
        modal bandwidth parameter for GM spectrum
    eig_idx_forward : int
        relative index of eigenray for the case where there are multiple eigenrays for the same rayid (microrays), Default is 0
    eig_idx_backward : int
        relative index of backward eigenray for the case where there are multiple eigenrays for the same rayid (microrays), Default is 0
    latitude : float
        latitude of the environment (for Coriolis parameter calculation), Default is midpoint of pilot moorings
    dtheta : float
        small angle increment for vertical fresnel scale calculation, Default is 1e-5, in units of radians

    Returns
    -------
    ray_variables : dict
        dictionary containing the following variables along the ray path.
        Variables that are along ray path are of length (M,) where the position of the ray is specified by rs (M,) and zs (M,)
        - rs : ray range values
        - zs : ray depth values
        - Cray : ray sound speed
        - Sray : ray sound slowness
        - Nray : ray buoyancy frequency
        - Theta_ray : ray angle
        - dcdz_ray : ray potential sound speed gradient
        - N0B_ray : N0B value around the ray
        - $\\langle \\mu^2 \\rangle$
        - J : maximum mode number
        - j : mode number index
        - N_j :
        - H :
        - $\\langle j \\rangle$
        - $\\langle j^{-1} \\rangle$
        - a
        - L_p : spatial coherence length of internal waves
        - drdt : differential range increment along ray path
        - dzdt : differential depth increment along ray path
        - RT_zeta_1 : Ray tube zeta 1
        - RT_zeta_2 : Ray tube zeta 2
        - $R_f^2$ : fresnel scale squared along ray
        - $\\left \\{ m^2\\right \\}$
        - $m^*$ : $\\frac{\\pi j_\\ast N(s)}{N_0 B}$
        - envi : environment used to solve forward eigenrays
        - receiver_range : receiver range
        - N : buoyancy frequency profile used
        - 'm_low' : minimum vertical wavenumber along ray
        - 'm_high' : maximum vertical wavenumber along ray
        - 'rs_back' : backward ray range values
        - 'zs_back' : backward ray depth values
    """

    ## Parse ray id, get backward ray id, and convert to pr format
    # Parse the ray_id to get components
    ray_id_pr, (sign, number, suffix) = parse_ray_id(ray_id)

    # Determine backward ray_id (flip sign if even number of turning points)
    if number % 2 != 0:
        ray_id_back = ray_id
    else:
        sign_back = "+" if sign == "-" else "-"
        ray_id_back = f"{sign_back}{number}{suffix}"

    # Convert backward ray_id to pr format
    ray_id_pr_back, _ = parse_ray_id(ray_id_back)

    try:
        ray_idx_forward = np.where(eigs.ray_id[0] == ray_id_pr)[0][eig_idx_forward]
    except IndexError:
        ray_idx_forward = np.where(eigs.ray_id[0] == ray_id_pr)[0]

    try:
        ray_idx_backward = np.where(eigs_backward.ray_id[0] == ray_id_pr_back)[0][
            eig_idx_backward
        ]
    except IndexError:
        ray_idx_backward = np.where(eigs_backward.ray_id[0] == ray_id_pr_back)[0]

    # get the range and depth vectors along the eigenray
    rs = eigs.rs[0][ray_idx_forward, :].flatten()
    zs = eigs.zs[0][ray_idx_forward, :].flatten()
    pzs = eigs.ps[0][ray_idx_forward, :].flatten()

    # create xarray with parametric variable s
    rsx = xr.DataArray(rs, dims=["s"])
    zsx = xr.DataArray(zs, dims=["s"])

    # interpolate sound speed and perturbation along eigenray
    Cray = envi.sound_speed.interp({"range": rsx, "depth": -zsx})
    Nray = N.interp({"range": rsx, "depth": -zsx})

    Theta_ray = np.arcsin(pzs * Cray)
    Theta_ray.name = "$\\theta$"
    Theta_ray.attrs["long_name"] = "Ray Angle"
    Theta_ray.attrs["units"] = "rad"

    dcdz_ray = dcdz_p.interp({"range": rsx, "depth": -zsx})

    N0B_ray = N.integrate("depth")
    N0B_ray.name = "$N_0 B$"
    N0B_ray.attrs["units"] = "rad m/s"

    # resample along parametric ray range vector
    N0B_ray = N0B_ray.interp({"range": rsx})

    ## Fractional sound speed variance
    mu2 = zeta0**2 * (N0 / Nray) * (dcdz_ray) ** 2 / Cray**2

    jstar = 3

    # maximum number of internal wave modes along ray path
    J_max = round(float((N0B_ray.max() ** 2) / (6 * np.pi * zeta0**2 * N0**2 * jstar)))

    # number of internal wave modes along ray path
    J = N0B_ray**2 / (6 * np.pi * zeta0**2 * N0**2 * jstar)
    J.name = "number of internal wave modes"
    J.attrs["units"] = "dimensionless"

    j = np.arange(1, J_max + 1)

    # construct H(j), H(j) = 0 for j > J(s)
    H = 1 / (j**2 + jstar**2)
    H = H[:, None] * np.ones((1, len(rs)))  # shape (len(j), len(rs))

    for s_idx in range(len(rs)):
        J_int = np.floor(J[s_idx].values)
        H[j > J_int, s_idx] = 0

    # unit normalization of H(j)
    N_j = np.sum(H, axis=0)
    H = H / N_j

    mean_jn1 = np.sum(H / j[:, None], axis=0)
    mean_j = np.sum(j[:, None] * H, axis=0)

    Omega = 7.292115e-5  # Earth Rotation Speed [rad/s]
    f = 2 * Omega * np.sin(np.radians(latitude))
    a = Nray * np.tan(Theta_ray) / f
    # set invalid a values to zero # NOT SURE IF THIS IS RIGHT
    a = a.where(np.isfinite(a), 0)

    Lp = (
        mean_jn1
        * 4
        / (np.pi**2)
        * N0B_ray
        / f
        * (
            1 / (1 + a**2)
            + (a**2)
            / 2
            * 1
            / (1 + a**2) ** (3 / 2)
            * np.log(((1 + a**2) ** (0.5) + 1) / ((1 + a**2) ** (0.5) - 1))
        )
    )
    # set invalid Lp values to 0
    Lp = Lp.where(np.isfinite(Lp), 0)

    # numerically compute parametric derivatives
    drdt = np.gradient(rs)
    dzdt = np.gradient(zs)

    ## Vertical Fresnel Scale (\zeta_1, \zeta_2, R_f^2)
    forward_launch_angle = eigs.launch_angles[0][ray_idx_forward]
    backward_launch_angle = eigs_backward.launch_angles[0][ray_idx_backward]

    forward_launch_angle_hat = forward_launch_angle + np.degrees(dtheta)
    backward_launch_angle_hat = backward_launch_angle + np.degrees(dtheta)

    source_depth_back = receiver_depth
    num_range_save = len(rs)  # use same as eigenrays passed in

    ray_hat_forward = pr.shoot_ray(
        source_depth, 0, forward_launch_angle_hat, receiver_range, num_range_save, envi
    )
    ray_hat_backward = pr.shoot_ray(
        source_depth_back,
        0,
        backward_launch_angle_hat,
        receiver_range,
        num_range_save,
        envi_back,
    )

    ## Ray Tube Functions (used for calculating the vertical fresnel zone)

    ### adding code to compute the upward and downward d launch angles

    RT_zeta_1 = (eigs.zs[0][ray_idx_forward] - ray_hat_forward.z) / (
        eigs.zs[0][ray_idx_forward, -1] - ray_hat_forward.z[-1]
    )
    RT_zeta_2 = (eigs_backward.zs[0][ray_idx_backward] - ray_hat_backward.z) / (
        eigs_backward.zs[0][ray_idx_backward, -1] - ray_hat_backward.z[-1]
    )
    # flip RT_zeta_2 so that it has the same range coords as RT_zeta_1
    RT_zeta_2 = np.flip(RT_zeta_2)
    r1 = ray_hat_forward.r
    dzeta1 = np.gradient(RT_zeta_1, r1)
    dzeta2 = np.gradient(RT_zeta_2, r1)

    # calculate average slowness along ray
    Sray = (1 / envi.sound_speed).interp({"range": rsx, "depth": -zsx})
    Sray.name = "S"
    Sray.attrs["units"] = "s/m"
    Sray.attrs["long_name"] = "Sound Slowness"

    Sbar = Sray.mean()
    Cbar = 1 / Sbar

    acoustic_wavelength = float(Cbar / fc)
    Rf2 = (acoustic_wavelength * RT_zeta_1 * RT_zeta_2) / (
        (RT_zeta_1 * dzeta2) - (RT_zeta_2 * dzeta1)
    )
    Rf2 = np.abs(Rf2)

    # convert to xarray DataArray
    Rf2 = xr.DataArray(Rf2, dims=["s"], coords={"s": Cray.coords["s"]})
    Rf2.name = "$R_f^2$"
    Rf2.attrs["units"] = "$m^2$"

    # calculate {m^2}
    brack_m2 = (np.pi * Nray / N0B_ray) ** 2 * mean_j / mean_jn1
    # set invalid values to zero
    brack_m2 = brack_m2.where(np.isfinite(brack_m2), 0)

    m_star = np.pi * jstar * Nray / N0B_ray

    # Calculate vertical wavenumber bounds (as a function of ray path)
    m_low = np.pi * Nray / N0B_ray  # [rad / m]

    m_high = J * np.pi * Nray / N0B_ray  # [rad / m]

    m_lo_max = float(m_low.max() / (2 * np.pi))  # [cycles / m]
    m_hi_max = float(m_high.max() / (2 * np.pi))  # [cycles / m]

    # construct output dictionary
    ray_variables = {
        "rs": rs,
        "zs": zs,
        "Cray": Cray,
        "Sray": Sray,
        "Nray": Nray,
        "Theta_ray": Theta_ray,
        "dcdz_ray": dcdz_ray,
        "N0B_ray": N0B_ray,
        "mu2": mu2,
        "J": J,
        "J_max": J_max,
        "j": j,
        "N_j": N_j,
        "H": H,
        "mean_j": mean_j,
        "mean_jn1": mean_jn1,
        "a": a,
        "L_p": Lp,
        "drdt": drdt,
        "dzdt": dzdt,
        "RT_zeta_1": RT_zeta_1,
        "RT_zeta_2": RT_zeta_2,
        "Rf2": Rf2,
        "brack_m2": brack_m2,
        "m_star": m_star,
        "envi": envi,
        "receiver_range": receiver_range,
        "N": N,
        "m_lo_max": m_lo_max,
        "m_hi_max": m_hi_max,
        "rh_forward": ray_hat_forward,
        "rh_backward": ray_hat_backward,
    }

    return ray_variables


def compute_n23_ray_variables(
    fc: float = 250,
    jstar: float = 3,
    f: float = 9.216181213401561e-05,  # [rad/s]
    zeta0: float = 7.3,  # m 1GM energy
    N0: float = 3 * (2 * np.pi) / 3600,  # [rad/s], 3 cph
    ray_ids: list | None = None,
):
    """
    Compute the ray path variables for the nescafe23 environment.
    Parameters
    ----------
    fc : float
        center acoustic frequency [Hz], Default is 250 Hz
    jstar : float
        modal bandwidth parameter for GM spectrum, Default is 3
    f : float
        Coriolis parameter [rad/s], Default is value at pilot mooring latitude
    zeta0 : float
        reference displacement of GM spectrum, Default is 7.3 m (1GM energy level)
    N0 : float
        reference buoyancy frequency of GM spectrum [rad/s], Default is 3 cph
    ray_ids : list
        list of ray identifier strings (e.g., '-7', '+10b'). If None, use all ray IDs in EIG_INDEXS.

    Returns
    -------
    ray_variables : dict
        dictionary containing the following variables along the ray path.
        Variables that are along ray path are of length (M,) where the position of the ray is specified by rs (M,) and zs (M,)
        - rs : ray range values
        - zs : ray depth values
        - Cray : ray sound speed
        - Sray : ray sound slowness
        - Nray : ray buoyancy frequency
        - Theta_ray : ray angle
        - dcdz_ray : ray potential sound speed gradient
        - N0B_ray : N0B value around the ray
        - $\\langle \\mu^2 \\rangle$
        - J : maximum mode number
        - j : mode number index
        - N_j :
        - H :
        - $\\langle j \\rangle$
        - $\\langle j^{-1} \\rangle$
        - a
        - L_p : spatial coherence length of internal waves
        - drdt : differential range increment along ray path
        - dzdt : differential depth increment along ray path
        - RT_zeta_1 : Ray tube zeta 1
        - RT_zeta_2 : Ray tube zeta 2
        - $R_f^2$ : fresnel scale squared along ray
        - $\\left \\{ m^2\\right \\}$
        - $m^*$ : $\\frac{\\pi j_\\ast N(s)}{N_0 B}$
        - envi : environment used to solve forward eigenrays
        - receiver_range : receiver range
        - N : buoyancy frequency profile used
        - 'm_low' : minimum vertical wavenumber along ray
        - 'm_high' : maximum vertical wavenumber along ray
    """

    # open environments
    fn1 = f"{os.environ['NESCAFE23_DATADIR']}environment_data/dataproducts/full_field_nlm.nc"

    envi_field = xr.open_dataset(fn1)

    # shift front location
    front_location = 35e3

    envi_field = envi_field.assign_coords({
        "range": (envi_field.range.values + front_location)
    })

    c = envi_field["c"]
    dcdz_p = envi_field["dcdz_p"]

    N = envi_field["N"]

    # remove negative buoancy frequency
    N = N.where(N > 0, 0)

    N_rad = N * (2 * np.pi) / 3600
    N_rad.attrs["units"] = "rad/s"

    bathy = xr.DataArray(
        np.linspace(4908, 4908, 100),
        dims=["range"],
        coords={"range": np.linspace(0, 200e3, 100)},
    )

    envi = pr.OceanEnvironment2D(sound_speed=c, bathymetry=bathy)

    fn = f"{os.environ['NESCAFE23_DATADIR']}dataproducts/ray_run_forward_back.pkl"

    with open(fn, "rb") as f:
        ray_run = pickle.load(f)

    eigs = ray_run["eigs"]
    eigs_back = ray_run["eigs_back"]
    envi = ray_run["envi"]
    envi_back = ray_run["envi_back"]

    source_depth = ray_run["geo"]["source_depth"]
    receiver_range = ray_run["geo"]["receiver_range"]
    receiver_depths = ray_run["geo"]["receiver_depths"]

    rvs = {}

    if ray_ids is None:
        ray_ids = list(EIG_INDEXS.keys())

    if len(ray_ids) == 0:
        tqdm_disable = False
    else:
        tqdm_disable = True

    for k, ray_id in tqdm(enumerate(ray_ids), disable=tqdm_disable):
        for n in range(len(EIG_INDEXS[ray_id])):
            if n == 0:
                rvs[ray_id] = []

            rvs[ray_id].append(
                compute_ray_path_variables(
                    ray_id,
                    eigs,
                    eigs_back,
                    envi,
                    envi_back,
                    source_depth,
                    receiver_depths[0],
                    receiver_range,
                    fc,
                    N_rad,
                    dcdz_p,
                    zeta0,
                    N0,
                    jstar,
                    eig_idx_forward=EIG_INDEXS[ray_id][n][0],
                    eig_idx_backward=EIG_INDEXS[ray_id][n][1],
                )
            )

    return rvs


def plot_ray_variables(rvs: dict, ray_ids: list, pair_idx: int = 0):
    """
    Given ray variables calculated with `compute_ray_path_variables`, generate plot summarizing them.

    Parameters
    ----------
    rvs : dict
        dictionary containing ray variables as output from `compute_ray_path_variables`
    ray_ids : list
        ray identifier (number of ray turning points with sign of launch angle)
    pair_idx : int, optional
        index of ray pair to plot the ray variables for
    Returns
    -------
    fig : matplotlib.figure.Figure
        figure containing the ray variable plots
    """
    fig, axs = plt.subplots(3, 4, figsize=(13, 10))

    fig.suptitle(f"Ray IDs: {', '.join(ray_ids)}", fontweight="bold")
    axs = axs.flatten()

    for i, ray_id in enumerate(ray_ids):
        rv = rvs[ray_id][pair_idx]
        ## Sound Speed
        plt.sca(axs[0])
        if i == 0:
            rv["envi"].plot()
            plt.xlim([0, rv["receiver_range"]])
        else:
            pass

        ## Buoyancy Frequency
        plt.sca(axs[1])
        if i == 0:
            rv["N"].plot(x="range", vmin=0, vmax=0.03)
        else:
            pass

        plt.xlim([0, rv["receiver_range"]])
        plt.ylim([5000, 0])

        ## Ray Path
        plt.sca(axs[2])
        plt.plot(rv["rs"] / 1000, -rv["zs"], label=ray_id)
        plt.ylim([5000, 0])
        plt.title("Eigenray Path")
        plt.xlabel("range [km]")
        plt.ylabel("depth [m]")
        plt.legend()

        ## N0B
        plt.sca(axs[3])
        plt.plot(rv["N0B_ray"].range / 1000, rv["N0B_ray"])
        plt.xlabel("range [km]")
        plt.ylabel("[rad m/s]")
        plt.title("$N_0 B(r)$")
        if np.any(np.isnan(rv["N0B_ray"])):
            plt.text(
                0.95,
                0.95,
                "nan",
                horizontalalignment="center",
                verticalalignment="center",
                transform=plt.gca().transAxes,
            )

        ## <mu^2>
        plt.sca(axs[4])
        plt.plot(rv["mu2"].range / 1000, rv["mu2"])
        plt.title("$\\langle \\mu^2(z) \\rangle$")
        plt.xlabel("range [km]")
        if np.any(np.isnan(rv["mu2"])):
            plt.text(
                0.95,
                0.95,
                "nan",
                horizontalalignment="center",
                verticalalignment="center",
                transform=plt.gca().transAxes,
            )

        ## a
        plt.sca(axs[5])
        plt.plot(rv["rs"] / 1000, rv["a"])
        plt.xlabel("range [km]")
        plt.title("$a$")

        if np.any(np.isnan(rv["a"])):
            plt.text(
                0.95,
                0.95,
                "nan",
                horizontalalignment="center",
                verticalalignment="center",
                transform=plt.gca().transAxes,
            )

        ## Lp
        plt.sca(axs[6])
        plt.plot(rv["rs"] / 1000, rv["L_p"].values)
        plt.yscale("log")
        plt.xlabel("range [km]")
        plt.ylabel("$L_p$ [m]")
        plt.title("$L_p$ [m]")

        if np.any(np.isnan(rv["L_p"].values)):
            plt.text(
                0.95,
                0.95,
                "nan",
                horizontalalignment="center",
                verticalalignment="center",
                transform=plt.gca().transAxes,
            )

        ## RT_zeta_1 and RT_zeta_2
        plt.sca(axs[7])
        plt.plot(rv["rs"] / 1000, rv["RT_zeta_1"], label="$\\zeta_1$")
        # plt.plot(rv["rs"] / 1000, rv["RT_zeta_2"], label="$\\zeta_2$")
        plt.xlabel("range [km]")
        plt.title("$\\zeta_1$ and $\\zeta_2$")
        plt.legend()
        plt.grid()

        if np.any(np.isnan(rv["RT_zeta_1"])) and not np.any(np.isnan(rv["RT_zeta_2"])):
            plt.text(
                0.95,
                0.95,
                "nan",
                horizontalalignment="center",
                verticalalignment="center",
                transform=plt.gca().transAxes,
            )

        ## Rf^2
        plt.sca(axs[8])
        plt.plot(rv["rs"] / 1000, rv["Rf2"])
        plt.xlabel("range [km]")
        plt.title("$R_f^2$ [$m^2$]")
        plt.ylabel("$R_f^2$ [$m^2$]")

        if np.any(np.isnan(rv["Rf2"])):
            plt.text(
                0.95,
                0.95,
                "nan",
                horizontalalignment="center",
                verticalalignment="center",
                transform=plt.gca().transAxes,
            )

        ## {m^2}
        plt.sca(axs[9])
        plt.plot(rv["rs"] / 1000, rv["brack_m2"])
        plt.xlabel("range [km]")
        plt.title(r"{ $m^2(z)$ }")

        if np.any(np.isnan(rv["brack_m2"])):
            plt.text(
                0.95,
                0.95,
                "nan",
                horizontalalignment="center",
                verticalalignment="center",
                transform=plt.gca().transAxes,
            )

        ## $ cos(m(j)) $\Delta z(s)$
        plt.sca(axs[10])
        mj = (
            np.pi
            * rv["j"][None, :, None]
            * rv["Nray"].values[:, None, None]
            / rv["N0B_ray"].values[:, None, None]
        )  # (s,j,1)
        summand = rv["H"].T[:, :, None] * (
            1
            - np.cos(mj * rv["RT_zeta_1"][:, None, None] * 250) / rv["j"][None, :, None]
        )  # (s,j,dzr)
        # set any nan values to zero
        summand = np.where(np.isfinite(summand), summand, 0)

        # plt.plot(rv['rs'] / 1000, summand[:,:,0], alpha=0.1, lw=1, c='k')

        if i == 0:
            axtwin = plt.gca().twinx()
        plt.sca(axtwin)

        plt.plot(rv["rs"] / 1000, np.nansum(summand[:, :, 0], axis=1), lw=2)

        plt.title(
            "$\\cos(m(j) \\zeta_1(s) \\Delta z_r(s)$\nfor $j=1,2,...,M$ and $\\Delta z_r = 250$ [ m ]"
        )

        plt.sca(axs[11])
        omegac = 2 * np.pi * 250

        deltaz_r = np.linspace(0, 500, 100)

        ## Compute vertical coherence at deltaz_r locations
        mj = (
            np.pi
            * rv["j"][None, :, None]
            * rv["Nray"].values[:, None, None]
            / rv["N0B_ray"].values[:, None, None]
        )  # (s,j,1)
        dz_s = rv["RT_zeta_1"][:, None, None] * deltaz_r[None, None, :]  # (s,1,dzr)
        summand = rv["H"].T[:, :, None] * (
            (1 - np.cos(mj * dz_s)) / rv["j"][None, :, None]
        )  # (s,j,dzr)
        # set any nan values to zero
        summand = np.where(np.isfinite(summand), summand, 0)

        integrand1 = (
            2 * omegac**2 * rv["mu2"] * rv["L_p"] / (rv["Cray"] ** 2 + rv["mean_jn1"])
        ).values
        integrand = (
            integrand1[:, None]
            * np.sum(summand, axis=1)
            * np.sqrt(rv["drdt"] ** 2 + rv["dzdt"] ** 2)[:, None]
        )  # (s,dzr)

        # set any nan values to zero
        integrand = np.where(np.isfinite(integrand), integrand, 0)

        plt.plot(rv["rs"] / 1000, integrand[:, 50], label=ray_id)

    plt.tight_layout(pad=0)

    plt.subplots_adjust(top=0.93)
    # plt.show()
    plt.close()

    return fig


def compute_F_phi_chi(
    rv: dict,
    m: np.array,
    fc: float = 250,
    jstar: float = 3,
    f: float = 9.216181213401561e-05,  # [rad/s]
):
    r"""
    Given the dictionary of ray variables, computed with `compute_ray_path_variables`, compute power spectral density of phase and log-amplitude ($F_\phi$ and $F_\chi$)

    Parameters
    ----------
    rv : dict
        dictionary of ray variables, computed with `compute_ray_path_variables`
    fc : float
        acoustic center frequency [1/s]
    jstar : float
        modal bandwidth parameter for GM spectrum.
    f : float
        Coriolis parameter [rad/s]
    m : np.array
        array of vertical wavenumbers [rad/m] at which to compute $F_\phi$ and $F_\chi$

    Returns
    -------
    F_phi : np.array
        array of phase power spectral density, $F_\phi$ [rad m], at each vertical wavenumber in m
    F_chi : np.array
        array of log-amplitude power spectral density, $F_\chi$ [rad m] at each vertical wavenumber in m
    """

    wc = 2 * np.pi * fc
    q0 = wc / 1500

    mstar = jstar * np.pi * rv["Nray"] / rv["N0B_ray"]
    sigma_L2 = f**2 + rv["Nray"] ** 2 * np.tan(rv["Theta_ray"]) ** 2

    W = 0.5 * (
        np.sqrt((1 - (sigma_L2 / rv["Nray"] ** 2)) * (1 - (f**2 / rv["Nray"] ** 2)))
        + ((np.sqrt(sigma_L2) / f) - (f / np.sqrt(sigma_L2)))
        * np.log(
            (
                np.sqrt(1 - (f**2 / rv["Nray"] ** 2))
                + np.sqrt((f**2 / sigma_L2) - (f**2 / rv["Nray"] ** 2))
            )
            / (np.sqrt(1 - (f**2 / sigma_L2)))
        )
    )

    F_phi_integrand = (
        8
        * q0**2
        / (np.pi**2)
        * rv["mu2"].values[:, None]
        * (mstar.values[:, None] / (m * (m**2 + mstar.values[:, None] ** 2)))
        * rv["Nray"].values[:, None]
        * f
        / sigma_L2.values[:, None]
        * W.values[:, None]
        * (1 + np.cos(m**2 * rv["Rf2"].values[:, None] / (2 * np.pi)))
        * np.sqrt(rv["drdt"][:, None] ** 2 + rv["dzdt"][:, None] ** 2)
    )
    # set nan values of integrand to zero
    F_phi_integrand[np.isnan(F_phi_integrand)] = 0

    F_phi = scipy.integrate.simpson(F_phi_integrand, axis=0)

    F_chi_integrand = (
        8
        * q0**2
        / (np.pi**2)
        * rv["mu2"].values[:, None]
        * (mstar.values[:, None] / (m * (m**2 + mstar.values[:, None] ** 2)))
        * rv["Nray"].values[:, None]
        * f
        / sigma_L2.values[:, None]
        * W.values[:, None]
        * (1 - np.cos(m**2 * rv["Rf2"].values[:, None] / (2 * np.pi)))
        * np.sqrt(rv["drdt"][:, None] ** 2 + rv["dzdt"][:, None] ** 2)
    )
    F_chi_integrand[np.isnan(F_chi_integrand)] = 0
    F_chi = scipy.integrate.simpson(F_chi_integrand, axis=0)

    valid_mask = (m[0] / (2 * np.pi) > rv["m_lo_max"]) & (
        m[0] / (2 * np.pi) < rv["m_hi_max"]
    )

    # set invalid F_phi and F_chi values to nan
    F_phi[~valid_mask] = np.nan
    F_chi[~valid_mask] = np.nan

    return F_phi, F_chi


def compute_lambda_phi(
    rv: dict,
    fc: float = 250,
):
    wc = 2 * np.pi * fc

    phi2_integrand = (
        wc**2
        * rv["mu2"].values[:, None]
        * rv["L_p"].values[:, None]
        / (rv["Cray"].values[:, None] ** 2)
        * np.sqrt(rv["drdt"][:, None] ** 2 + rv["dzdt"][:, None] ** 2)
    )
    Phi2 = scipy.integrate.simpson(phi2_integrand, axis=0)

    lam_integrand = (
        wc**2
        / Phi2
        * rv["mu2"].values[:, None]
        * rv["L_p"].values[:, None]
        / (rv["Cray"].values[:, None] ** 2)
        * rv["brack_m2"].values[:, None]
        * rv["Rf2"].values[:, None]
        / (2 * np.pi)
        * np.sqrt(rv["drdt"][:, None] ** 2 + rv["dzdt"][:, None] ** 2)
    )
    Lamda = scipy.integrate.simpson(lam_integrand, axis=0)

    return Lamda, Phi2


def compute_phase_structure_function(
    rv: dict,
    deltaz_r: np.array,
    fc: float = 250,
):
    r"""
    Given the ray variables, calculate the phase structure function as a function of vertical seperation $\delta z_r$.

    Parameters
    ----------
    rv : dict
        ray variables dictionary, computed with `compute_ray_path_variables`
    deltaz_r : np.array
        array of vertical separations [m] at which to compute phase structure function
    fc : float
        acoustic center frequency [1/s]

    Returns
    -------
    D_dzr : np.array
        array of phase structure function values at each vertical separation in deltaz_r
    """

    omegac = 2 * np.pi * fc

    ## Compute phase structure function at deltaz_r locations
    mj = (
        np.pi
        * rv["j"][None, :, None]
        * rv["Nray"].values[:, None, None]
        / rv["N0B_ray"].values[:, None, None]
    )  # (s,j,1)
    dz_s = rv["RT_zeta_1"][:, None, None] * deltaz_r[None, None, :]  # (s,1,dzr)
    summand = rv["H"].T[:, :, None] * (
        (1 - np.cos(mj * dz_s)) / rv["j"][None, :, None]
    )  # (s,j,dzr)
    # set any nan values to zero
    summand = np.where(np.isfinite(summand), summand, 0)

    integrand1 = (
        2 * omegac**2 * rv["mu2"] * rv["L_p"] / (rv["Cray"] ** 2 + rv["mean_jn1"])
    ).values
    integrand = (
        integrand1[:, None]
        * np.sum(summand, axis=1)
        * np.sqrt(rv["drdt"] ** 2 + rv["dzdt"] ** 2)[:, None]
    )  # (s,dzr)

    # set any nan values to zero
    integrand = np.where(np.isfinite(integrand), integrand, 0)

    D_dzr = scipy.integrate.simpson(integrand, axis=0)

    return D_dzr


def estimate_mutual_coherence(X, T):
    r"""
    Given array X of shape (M,N) containing M snapshots of vector $\\in \\mathbb{R}^N$
        where the vector contains equally spaced samples in dimension that mutual coherence is to be estimated.

    The mutual coherence is estimated with an unbiased complex cross-correlation of the values of X along dim 1,
        and averaged along all common lags, and all snapshots.

    Parameters
    ----------
    X : np.array
        Array of shape (M,N), containing M snapshots of vector $\\in \\mathbb{R}^N$
    T : sample spacing of samples in dim 1of X. Units arbitrary


    Returns
    -------
    MC : np.array
        Array of shape (N,). Normalized mutual coherence, normalized by the value at the smallest
        non-zero lag (to remove bias from noise at lag=0).
    t : np.array
        Array containing the lags of MC in units of T
    """

    Rxxs = []

    M, N = X.shape

    lags = np.arange(-N + 1, N)
    denom = N - np.abs(lags)
    for k in range(M):
        x = X[k]

        Rxx = signal.correlate(x, x)
        rxx_unbiased = Rxx / denom  # divide by (N - |lag|)
        Rxxs.append(rxx_unbiased)

    # average over all snapshots
    MC_full = np.mean(np.array(Rxxs), axis=0)

    # remove negative lag and normalize
    MC = MC_full.real[N:] / MC_full.real[N]

    t = np.arange(T, N * T - np.finfo(float).eps, T)

    # variance of estimate
    MC_var = (1 / (M * denom))[N:]

    return t, MC, MC_var


def create_2D_ocean_slice():
    """
    Create 2D ocean slice using FCTD, and two full water CTD casts.

    Returns
    -------
    front_full : xarray.Dataset
        xarray Dataset containing 2D ocean slice that is stitched together from FCTD and two CTD casts
    front_full_nlm : xarray.Dataset
        xarray Dataset containing 2D ocean slice that is stitched together from FCTD and two CTD casts.
        Local maximum near -6 upper turning point in SION CTD is manually removed.
    """
    coords = {
        "SION": [39 + 37.194 / 60, -63 - 39.427 / 60],
        "SIOE": [38 + 45.906 / 60, -62 - 15.626 / 60],
    }

    fn = f"{os.environ['NESCAFE23_DATADIR']}environment_data/SION_CTD.mat"
    SION_mat = io.loadmat(fn)

    fn = f"{os.environ['NESCAFE23_DATADIR']}environment_data/SIOE_CTD.mat"
    SIOE_mat = io.loadmat(fn)

    fn = f"{os.environ['NESCAFE23_DATADIR']}environment_data/FCTD_SoundSpeed_Colosi_v2.mat"
    fctd_mat = io.loadmat(fn)

    fctds = {}
    fctds_pilot = {}

    deep_value = {"tt": 2.3, "ss": 34.885}  # , 'cc':1640}

    for k, var in enumerate(["tt", "ss"]):
        fctds[var] = xr.DataArray(
            fctd_mat[var],
            dims=["range", "depth"],
            coords={
                "range": fctd_mat["r"].flatten() * 1000,
                "depth": fctd_mat["z"].flatten(),
                "latitude": ("range", fctd_mat["lat"].flatten()),
                "longitude": ("range", fctd_mat["lon"].flatten()),
            },
        ).sortby("range")

        # Add 10000 meter
        fctds_pilot[var] = xr.concat(
            (
                fctds[var],
                xr.ones_like(fctds[var][:, -2:-1]).assign_coords({"depth": [10000]})
                * deep_value[var],
            ),
            dim="depth",
        )

    fctds_pilot = xr.Dataset(fctds_pilot)

    # interpolate to Δr=200m, Δz = 3km
    fctd_grid = fctds_pilot.interp({
        "range": np.arange(0, fctds_pilot.range[-1], 500),
        "depth": np.arange(0, 6000),
    })

    r_smooth = 3000
    z_smooth = 200

    bz, az = signal.butter(4, 1 / z_smooth, fs=1)
    br, ar = signal.butter(4, 1 / r_smooth, fs=1 / 500)

    front = {}
    for k, var in enumerate(["tt", "ss"]):
        fctd_smooth = xrs.filtfilt(
            xrs.filtfilt(fctd_grid[var], dim="range", b=br, a=ar),
            dim="depth",
            b=bz,
            a=az,
        )

        front[var] = fctd_smooth.sel({"range": slice(60000, 170000)})
        front[var] = front[var].assign_coords({
            "range": front[var].range.values - 115000
        })
        front[var] = (
            front[var]
            .assign_coords({"range": np.flip(front[var].range.values)})
            .sortby("range")
        )

        front[var] = front[var].drop_vars(["latitude", "longitude"])

    front = xr.Dataset(front)

    full_field_range_coord = np.arange(-155000, 155000, 500)

    mask = xr.DataArray(
        np.ones((6000, len(full_field_range_coord))),
        dims=["depth", "range"],
        coords={"depth": front.depth, "range": full_field_range_coord},
    )

    front_rep = xr.concat(
        (
            front.isel({"range": 0}).assign_coords({"range": -155000}),
            front,
            front.isel({"range": -1}).assign_coords({"range": 155000}),
        ),
        dim="range",
    )

    front_full = front_rep.interp_like(
        mask, kwargs={"bounds_error": False, "fill_value": "extrapolate"}
    )

    SION = {
        "tt": xr.DataArray(
            SION_mat["t5"].flatten(),
            dims=["depth"],
            coords={"depth": SION_mat["z5"].flatten()},
        ),
        "ss": xr.DataArray(
            SION_mat["s5"].flatten(),
            dims=["depth"],
            coords={"depth": SION_mat["z5"].flatten()},
        ),
    }

    SIOE = {
        "tt": xr.DataArray(
            SIOE_mat["t8"].flatten(),
            dims=["depth"],
            coords={"depth": SIOE_mat["z8"].flatten()},
        ),
        "ss": xr.DataArray(
            SIOE_mat["s8"].flatten(),
            dims=["depth"],
            coords={"depth": SIOE_mat["z8"].flatten()},
        ),
    }

    for var in SION.keys():
        # Add 10000 meter
        SION[var] = xr.concat(
            (
                SION[var],
                xr.ones_like(SION[var][-2:-1]).assign_coords({"depth": [10000]})
                * deep_value[var],
            ),
            dim="depth",
        )

        # Add 10000 meter
        SIOE[var] = xr.concat(
            (
                SIOE[var],
                xr.ones_like(SIOE[var][-2:-1]).assign_coords({"depth": [10000]})
                * deep_value[var],
            ),
            dim="depth",
        )

        SION[var] = xrs.filtfilt(SION[var], dim="depth", b=bz, a=az)
        SIOE[var] = xrs.filtfilt(SIOE[var], dim="depth", b=bz, a=az)

    SION = xr.Dataset(SION)
    SIOE = xr.Dataset(SIOE)

    SION_full = SION.interp_like(
        mask, kwargs={"bounds_error": False, "fill_value": "extrapolate"}
    ).broadcast_like(mask)

    SIOE_full = SIOE.interp_like(
        mask, kwargs={"bounds_error": False, "fill_value": "extrapolate"}
    ).broadcast_like(mask)

    SION_full_nlm = SION_full.copy()
    SION_full_nlm = SION_full_nlm.where(
        (SION_full_nlm.depth > 350) | (SION_full_nlm.depth < 25), np.nan
    )
    SION_full_nlm = SION_full_nlm.interpolate_na(dim="depth")

    # compute C
    p = gsw.p_from_z(-SION_full_nlm.depth, lat=coords["SION"][0])

    mask1 = mask.where(
        (mask.depth < 800) & (mask.range > -50000) & (mask.range < 50000), 0
    )
    mask2 = (-mask1 + 1).where(mask.range > -30000, 0)
    mask3 = (-mask1 + 1).where(mask.range <= -30000, 0)

    z = np.linspace(-np.pi, np.pi, 600)
    x = np.linspace(-np.pi, np.pi, 120)

    kernel_x = (np.cos(x) + 1) / 2
    kernel_z = (np.cos(z) + 1) / 2

    kernel = kernel_z[:, None] * kernel_x[None, :]
    kernel = kernel / np.sum(kernel)

    mask1_smooth = xr.DataArray(
        _fftconvolve_mirror(mask1.values, kernel, mode="same"),
        dims=["depth", "range"],
        coords={"depth": mask.depth, "range": mask.range},
    )

    mask2_smooth = xr.DataArray(
        _fftconvolve_mirror(mask2.values, kernel, mode="same"),
        dims=["depth", "range"],
        coords={"depth": mask.depth, "range": mask.range},
    )

    mask3_smooth = xr.DataArray(
        _fftconvolve_mirror(mask3.values, kernel, mode="same"),
        dims=["depth", "range"],
        coords={"depth": mask.depth, "range": mask.range},
    )

    full_field = (
        front_full * mask1_smooth + SION_full * mask2_smooth + SIOE_full * mask3_smooth
    )

    full_field["tt"].attrs["units"] = "deg C"
    full_field["tt"].name = "in-situ temperature"

    full_field["ss"].attrs["units"] = "psu"
    full_field["ss"].name = "pracitical salinity"

    full_field_nlm = (
        front_full * mask1_smooth
        + SION_full_nlm * mask2_smooth
        + SIOE_full * mask3_smooth
    )

    full_field_nlm["tt"].attrs["units"] = "deg C"
    full_field_nlm["tt"].name = "in-situ temperature"

    full_field_nlm["ss"].attrs["units"] = "psu"
    full_field_nlm["ss"].name = "pracitical salinity"

    # ## Calculate all oceanographic variables
    p = gsw.p_from_z(-full_field.depth, coords["SION"][0])

    ## Salinity
    SA = gsw.SA_from_SP(full_field["ss"], p, coords["SION"][1], coords["SION"][0])
    SA.attrs["units"] = "g/kg"
    SA.name = "Absolute Salinity"

    SA_nlm = gsw.SA_from_SP(
        full_field_nlm["ss"], p, coords["SION"][1], coords["SION"][0]
    )
    SA_nlm.attrs["units"] = "g/kg"
    SA_nlm.name = "Absolute Salinity"

    ## Conservative Temperature
    CT = gsw.CT_from_t(SA, full_field["tt"], p)
    CT.attrs["units"] = "deg C"
    CT.name = "Conservative Temperature"

    CT_nlm = gsw.CT_from_t(SA_nlm, full_field_nlm["tt"], p)
    CT_nlm.attrs["units"] = "deg C"
    CT_nlm.name = "Conservative Temperature"

    ## Sound Spped
    c = aog_vsdg(p, full_field["tt"], full_field["ss"])
    c.attrs["units"] = "m/s"
    c.name = "Sound Speed"

    c_nlm = aog_vsdg(p, full_field_nlm["tt"], full_field_nlm["ss"])
    c_nlm.attrs["units"] = "m/s"
    c_nlm.name = "Sound Speed"

    ## Buoyancy Frequency
    N2, pmid = gsw.Nsquared(SA, CT, p.values[:, None], lat=coords["SION"][0], axis=0)
    z_mid = gsw.z_from_p(pmid, coords["SION"][0])[:, 0]
    N2x = xr.DataArray(
        N2, dims=["depth", "range"], coords={"depth": -z_mid, "range": full_field.range}
    )
    N = (
        np.sqrt(
            N2x.interp(
                {"depth": full_field.depth},
                kwargs={"bounds_error": False, "fill_value": "extrapolate"},
            )
        )
        / (2 * np.pi)
        * 3600
    )
    N.attrs["units"] = "cph"
    N.name = "Buoyancy Frequency"

    N2_nlm, pmid = gsw.Nsquared(
        SA_nlm, CT_nlm, p.values[:, None], lat=coords["SION"][0], axis=0
    )
    z_mid = gsw.z_from_p(pmid, coords["SION"][0])[:, 0]
    N2x_nlm = xr.DataArray(
        N2_nlm,
        dims=["depth", "range"],
        coords={"depth": -z_mid, "range": full_field.range},
    )
    N_nlm = (
        np.sqrt(
            N2x_nlm.interp(
                {"depth": full_field.depth},
                kwargs={"bounds_error": False, "fill_value": "extrapolate"},
            )
        )
        / (2 * np.pi)
        * 3600
    )
    N_nlm.attrs["units"] = "cph"
    N_nlm.name = "Buoyancy Frequency"

    ## Potential Temperature
    PT = gsw.pt_from_t(SA, full_field["tt"], p, 0)
    PT.attrs["units"] = "deg C"
    PT.name = "potential temperature"

    PT_nlm = gsw.pt_from_t(SA_nlm, full_field_nlm["tt"], p, 0)
    PT_nlm.attrs["units"] = "deg C"
    PT_nlm.name = "potential temperature"

    ## Adiabatic Sound Speed Gradient
    C_adiabatic = aog_vsdg(0, PT, full_field["ss"])
    C_adiabatic.attrs["units"] = "deg C"
    C_adiabatic.name = "adiabatic sound speed"

    C_adiabatic_nlm = aog_vsdg(0, PT_nlm, full_field_nlm["ss"])
    C_adiabatic_nlm.attrs["units"] = "deg C"
    C_adiabatic_nlm.name = "adiabatic sound speed"

    dcdz_p = -C_adiabatic.differentiate("depth")
    dcdz_p.attrs["units"] = "$m/s^2$"
    dcdz_p.name = "adiabatic vertical sound speed gradient"

    dcdz_p_nlm = -C_adiabatic_nlm.differentiate("depth")
    dcdz_p_nlm.attrs["units"] = "$m/s^2$"
    dcdz_p_nlm.name = "adiabatic vertical sound speed gradient"

    full_field_full = xr.Dataset({
        "t": full_field["tt"],
        "SP": full_field["ss"],
        "c": c,
        "N": N,
        "SA": SA,
        "dcdz_p": dcdz_p,
    })
    full_field_full_nlm = xr.Dataset({
        "t": full_field_nlm["tt"],
        "SP": full_field_nlm["ss"],
        "c": c_nlm,
        "N": N_nlm,
        "SA": SA_nlm,
        "dcdz_p": dcdz_p_nlm,
    })

    return full_field_full, full_field_full_nlm


def _fftconvolve_mirror(input_array, kernel, mode="same"):
    """
    FFT convolution with mirror boundary conditions.

    Parameters:
    -----------
    input_array : array_like
        Input array to convolve
    kernel : array_like
        Convolution kernel
    mode : str
        'same', 'valid', or 'full'
    """
    input_shape = np.array(input_array.shape)
    kernel_shape = np.array(kernel.shape)

    # Calculate padding needed (kernel_size - 1 on each side)
    pad_width = [(k - 1, k - 1) for k in kernel_shape]

    # Pad with mirror/reflection
    padded = np.pad(input_array, pad_width, mode="reflect")

    # Perform FFT convolution with mode='valid'
    result = signal.fftconvolve(padded, kernel, mode="valid")

    if mode == "same":
        # Center the result by cropping from the correct starting position
        # The valid convolution on padded input produces an output that needs to be
        # cropped by (k-1)//2 from the start to align with scipy's 'same' mode
        start = [(k - 1) // 2 for k in kernel_shape]
        slices = tuple(
            slice(s, s + inp_s) for s, inp_s in zip(start, input_shape, strict=False)
        )
        return result[slices]
    elif mode == "full":
        return result
    elif mode == "valid":
        # Calculate valid region size
        valid_shape = input_shape - kernel_shape + 1
        # Center the valid region using the same offset as 'same' mode
        start = [(k - 1) // 2 for k in kernel_shape]
        slices = tuple(
            slice(s, s + v) for s, v in zip(start, valid_shape, strict=False)
        )
        return result[slices]

    return result


def run_rays():
    """
    Solve for the ray fan and eigen rays for the forward and backward propagation environment.
    """

    ## Open Environment
    fn = f"{os.environ['NESCAFE23_DATADIR']}environment_data/dataproducts/full_field_nlm.nc"
    envi_field = xr.open_dataset(fn)

    # shift front location
    front_location = 35e3  # m

    envi_field = envi_field.assign_coords({
        "range": (envi_field.range.values + front_location)
    })

    c = envi_field["c"]

    bathy = xr.DataArray(
        np.linspace(4908, 4908, 100),  # flat bottom
        dims=["range"],
        coords={"range": np.linspace(0, 200e3, 100)},
    )

    envi = pr.OceanEnvironment2D(sound_speed=c, bathymetry=bathy)

    ## Source / Receiver Geometry
    # TODO do correct source localization
    source_depth = 2361.50872806
    source_range = 0
    launch_angles = np.linspace(-30, 30, 8000)
    receiver_range = 150404.34909747
    num_range_save = 1000
    receiver_depths = [999.37979522]

    ## Forward Rays
    rays = pr.shoot_rays(
        source_depth, source_range, launch_angles, receiver_range, num_range_save, envi
    )
    eigs = pr.find_eigenrays(
        rays,
        receiver_depths,
        source_depth,
        source_range,
        receiver_range,
        num_range_save,
        envi,
        ztol=10,
        max_iter=500,
    )

    ## Backward Rays
    r_back = -1 * (c.range.values - receiver_range)
    c_back = c.assign_coords({"range": r_back}).sortby("range")
    envi_back = pr.OceanEnvironment2D(sound_speed=c_back, bathymetry=bathy)

    receiver_depth_back = [source_depth]
    source_depth_back = receiver_depths[0]

    rays_back = pr.shoot_rays(
        source_depth_back,
        source_range,
        launch_angles,
        receiver_range,
        num_range_save,
        envi_back,
    )
    eigs_back = pr.find_eigenrays(
        rays_back,
        receiver_depth_back,
        source_depth_back,
        source_range,
        receiver_range,
        num_range_save,
        envi_back,
        ztol=10,
        max_iter=500,
    )

    geo = {
        "source_depth": source_depth,
        "source_range": source_range,
        "launch_angles": launch_angles,
        "receiver_range": receiver_range,
        "receiver_depths": receiver_depths,
        "num_range_save": num_range_save,
    }

    return rays, eigs, rays_back, eigs_back, envi, envi_back, geo
