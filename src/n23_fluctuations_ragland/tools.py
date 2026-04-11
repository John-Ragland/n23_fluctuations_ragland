"""Collection of utility functions for the data analysis of NESCAFE 2023 Acoustic Fluctions"""

import os
import pickle
import warnings

import numpy as np
import pygenray as pr
import scipy
import xarray as xr
import xrsignal as xrs
from matplotlib import pyplot as plt
from nescafe23 import compute_filtered_fctd_grid, open_receptions, parse_ray_id
from nescafe23.data import get_data_dir
from scipy import io, optimize, signal
from tqdm import tqdm

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
    "-8": [(0, 0)],
    "+7": [(0, 0)],
    "-7": [(1, 0)],
    "-6": [(0, 2), (1, 1), (2, 0)],
    "+6": [(0, 0)],
    "-5b": [(0, 0), (1, 3), (2, 2)],
    "-6b": [(0, 0)],
    "+6b": [(0, 0)],
    "+7b": [(1, 0)],
    "-7b": [(0, 0)],
    "-8b": [(0, 0)],
    "+8b": [(0, 0)],
    "+9b": [(0, 0)],
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


# TODO remove this function
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
            flat_idx = np.argmax(env_slice.data)
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

    ray_idx_forward = np.where(eigs.ray_id[0] == ray_id_pr)[0][eig_idx_forward]
    ray_idx_backward = np.where(eigs_backward.ray_id[0] == ray_id_pr_back)[0][
        eig_idx_backward
    ]

    # Removing exception handling
    # except IndexError:
    #    ray_idx_forward = np.where(eigs.ray_id[0] == ray_id_pr)[0][0]

    # except IndexError:
    #    ray_idx_backward = np.where(eigs_backward.ray_id[0] == ray_id_pr_back)[0]

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

    wronk = (RT_zeta_1 * dzeta2) - (RT_zeta_2 * dzeta1)

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
        "wronk": wronk,
    }

    return ray_variables


def compute_n23_ray_variables(
    fc: float = 250,
    jstar: float = 3,
    f: float = 9.216181213401561e-05,  # [rad/s]
    zeta0: float = 7.3,  # m 1GM energy
    N0: float = 3 * (2 * np.pi) / 3600,  # [rad/s], 3 cph
    ray_ids: list | None = None,
    envi_field: xr.Dataset | None = None,
    ray_run: dict | None = None,
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
    envi_field : xr.Dataset, optional
        xarray dataset of the environment, generated with `create_2D_ocean_slice`. If None, load from pre-computed data (generated from CLI)
    ray_run : dict, optional
        dictionary containing ray run data, generated with `run_rays`. If None, load from pre-computed data (generated from CLI)

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

    envi_field = compute_filtered_fctd_grid(z_smooth=200, r_smooth=15000)

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

    if ray_run is None:
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


def plot_ray_variables(
    rvs: dict,
    ray_ids: list,
    pair_idxs: dict | None = None,
):
    """
    Given ray variables calculated with `compute_ray_path_variables`, generate plot summarizing them.

    Parameters
    ----------
    rvs : dict
        dictionary containing ray variables as output from `compute_ray_path_variables`
    ray_ids : list
        ray identifier (number of ray turning points with sign of launch angle)
    pair_idxs : dict, optional
        indices of ray pairs to plot the ray variables for. keys are ray_ids. If not provided, defaults to first pair for each ray_id.
    Returns
    -------
    fig : matplotlib.figure.Figure
        figure containing the ray variable plots
    """
    if pair_idxs is None:
        pair_idxs = dict(
            zip(list(EIG_INDEXS.keys()), [[0]] * len(EIG_INDEXS), strict=False)
        )

    fig, axs = plt.subplots(3, 4, figsize=(16, 10))

    fig.suptitle(f"Ray IDs: {', '.join(ray_ids)}", fontweight="bold")
    axs = axs.flatten()

    for i, ray_id in enumerate(ray_ids):
        for n, pair_idx in enumerate(pair_idxs[ray_id]):
            rv = rvs[ray_id][pair_idx]
            ## Sound Speed
            plt.sca(axs[0])
            if (i == 0) & (n == 0):
                rv["envi"].plot()
                plt.xlim([0, rv["receiver_range"]])
            else:
                pass

            plt.plot(rv["rs"], -rv["zs"], label=ray_id)
            plt.ylim([5000, 0])
            plt.title("Eigenray Path")
            plt.xlabel("range [km]")
            plt.ylabel("depth [m]")
            # plt.legend()

            ## Buoyancy Frequency
            plt.sca(axs[1])
            if (i == 0) & (n == 0):
                rv["N"].plot(x="range", vmin=0, vmax=0.03)
            else:
                pass

            plt.xlim([0, rv["receiver_range"]])
            plt.ylim([5000, 0])

            plt.plot(rv["rs"], -rv["zs"], label=ray_id)
            plt.ylim([5000, 0])
            plt.title("Eigenray Path")
            plt.xlabel("range [km]")
            plt.ylabel("depth [m]")
            # plt.legend()

            ## Ray Paths
            # plt.sca(axs[2])
            # plt.plot(rv["rs"] / 1000, -rv["zs"], label=ray_id)
            # plt.ylim([5000, 0])
            # plt.title("Eigenray Path")
            # plt.xlabel("range [km]")
            # plt.ylabel("depth [m]")
            # plt.legend()

            ## Wronksian
            plt.sca(axs[2])
            plt.plot(rv["rs"] / 1000, np.abs(rv["wronk"]))
            plt.xlabel("range [km]")
            plt.title("$\\zeta_1 \\partial_x \\zeta_2 - \\zeta_2 \\partial_x \\zeta_1$")

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
            plt.legend(ncols=4)
            plt.grid()

            if np.any(np.isnan(rv["RT_zeta_1"])) and not np.any(
                np.isnan(rv["RT_zeta_2"])
            ):
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
                - np.cos(mj * rv["RT_zeta_1"][:, None, None] * 250)
                / rv["j"][None, :, None]
            )  # (s,j,dzr)
            # set any nan values to zero
            summand = np.where(np.isfinite(summand), summand, 0)

            # plt.plot(rv['rs'] / 1000, summand[:,:,0], alpha=0.1, lw=1, c='k')

            # if i == 0:
            #    axtwin = plt.gca().twinx()
            # plt.sca(axtwin)

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
                2
                * omegac**2
                * rv["mu2"]
                * rv["L_p"]
                / (rv["Cray"] ** 2 + rv["mean_jn1"])
            ).values
            integrand = (
                integrand1[:, None]
                * np.sum(summand, axis=1)
                * np.sqrt(rv["drdt"] ** 2 + rv["dzdt"] ** 2)[:, None]
            )  # (s,dzr)

            # set any nan values to zero
            integrand = np.where(np.isfinite(integrand), integrand, 0)

            plt.plot(rv["rs"] / 1000, integrand[:, 50], label=ray_id)
            plt.title("Integrad for D")

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
    Given the dictionary of ray variables, computed with `compute_ray_path_variables`, compute power spectral density of phase and log-amplitude ($F_\\phi$ and $F_\\chi$)

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
        array of vertical wavenumbers [rad/m] at which to compute $F_\\phi$ and $F_\\chi$

    Returns
    -------
    F_phi : np.array
        array of phase power spectral density, $F_\\phi$ [rad m], at each vertical wavenumber in m
    F_chi : np.array
        array of log-amplitude power spectral density, $F_\\chi$ [rad m] at each vertical wavenumber in m
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


def compute_F_phi_chi_n23(GM_energy):
    """
    Compute the theoretical phase and log-amplitude PSDs for the nescafe23 variables specifically

    Parameters
    ----------
    GM_energy : float
        energy level of Garrett-Munk spectrum, X corresponds to zeta0 = $\\sqrt(X) \\times$ 7.3 [m]

    Returns
    -------
    m : np.array
        array of vertical wavenumbers [rad/m] at which F_phi and F_chi are computed
    F_phis : dict
        dictionary of phase power spectral densities for each ray_id.
    F_chis : dict
        dictionary of log-amplitude power spectral densities for each ray_id.
    """

    zeta0 = 7.3 * np.sqrt(GM_energy)
    rvs = compute_n23_ray_variables(zeta0=zeta0)

    m = 2 * np.pi * np.logspace(-5, 0, 500)[None, :]  # [rad / m]

    F_phis = {}
    F_chis = {}

    # compute only for ducted paths
    for ray_id in ["+7", "-7", "+6", "-6"]:
        F_phis[ray_id] = []
        F_chis[ray_id] = []

        for n in range(len(rvs[ray_id])):
            F_phi, F_chi = compute_F_phi_chi(rvs[ray_id][n], m)

            F_phis[ray_id].append(F_phi)
            F_chis[ray_id].append(F_chi)

    return m, F_phis, F_chis


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


def run_rays(
    num_rays: int | None = None,
    source_depth: float | None = None,
    receiver_range: float | None = None,
    receiver_depths: list | None = None,
):
    """
    Solve for the ray fan and eigen rays for the forward and backward propagation environment.

    Default values for source/receiver geometry are from source localization using stitched environment.
    TODO
    Parameters
    ----------
    envi_field : xarray.Dataset, optional
        xarray Dataset containing 2D ocean slice. If None, will open pre-computed environment
    num_rays : int
        Number of rays to shoot in the ray fan. Default is 8000.
    source_depth : float
        Source depth in meters. Default is 2361.50872806 m.
    receiver_range : float
        Receiver range in meters. Default is 150404.34909747 m.
    receiver_depths : list of float
        List of receiver depths in meters. Default is [999.37979522] m.

    Returns
    -------
    rays : pr.RayFan
        Forward ray fan
    eigs : pr.Eigenrays
        Forward eigenrays
    rays_back : pr.RayFan
        Backward ray fan
    eigs_back : pr.Eigenrays
        Backward eigenrays
    envi : pr.OceanEnvironment2D
        Forward ocean environment
    envi_back : pr.OceanEnvironment2D
        Backward ocean environment
    geo : dict
        Dictionary containing source/receiver geometry
    """

    # create default values for geometry
    if num_rays is None:
        num_rays = 8000
    if source_depth is None:
        source_depth = 2300
    if receiver_range is None:
        receiver_range = 153000
    if receiver_depths is None:
        # receiver_depths = [1072.3225807360088]  # from pressure sensors
        receiver_depths = [
            1139.04728384,
            378.59703093,
        ]  # from 12th transmission pressure sensors
    # z_smooth is set to 200 [m]
    # r_smooth is set to 15 [km]
    fctd = compute_filtered_fctd_grid(z_smooth=200, r_smooth=15000)

    # shift front location
    front_location = 35e3  # m

    fctd = fctd.assign_coords({"range": (fctd.range.values + front_location)})

    c = fctd["c"]

    bathy = xr.DataArray(
        np.linspace(5000, 5000, 100),  # flat bottom
        dims=["range"],
        coords={"range": np.linspace(0, 200e3, 100)},
    )

    envi = pr.OceanEnvironment2D(sound_speed=c, bathymetry=bathy)

    ## Source / Receiver Geometry

    # TODO do correct source localization
    source_range = 0
    launch_angles = np.linspace(-30, 30, num_rays)
    num_range_save = 1000

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
        ztol=1,
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
        ztol=1,
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


def compute_mean_pulse():
    """
    Compute the mean pulse and mean pulse noise for the 23 receptions
    """
    dss = open_receptions()

    dss_interp = []
    dss_upper_interp = []
    for ds in tqdm(dss):
        dss_interp.append(
            np.abs(
                ds.swap_dims({"hydrophone": "actual_depth"})
                .rename({"actual_depth": "depth"})[32:]
                .interp({
                    "depth": np.linspace(800, 1400, 50),
                    "time": dss[0].time.values,
                })
                .drop_vars(["nominal_depth"])
            )
        )

        dss_upper_interp.append(
            np.abs(
                ds.swap_dims({"hydrophone": "actual_depth"})
                .rename({"actual_depth": "depth"})[:32]
                .interp({
                    "depth": np.linspace(100, 600, 50),
                    "time": dss[0].time.values,
                })
                .drop_vars(["nominal_depth"])
            )
        )

    ds_interp = xr.concat(dss_interp, dim="Tx")
    ds_interp_upper = xr.concat(dss_upper_interp, dim="Tx")

    mean_pulse_upper = ds_interp_upper.mean(dim="Tx")
    mean_pulse_lower = ds_interp.mean(dim="Tx")

    mean_pulse_upper_noise = 20 * np.log10(
        np.sqrt((mean_pulse_upper.sel({"time": slice(90, 95)}) ** 2).mean(dim="time"))
    )
    mean_pulse_lower_noise = 20 * np.log10(
        np.sqrt((mean_pulse_lower.sel({"time": slice(90, 95)}) ** 2).mean(dim="time"))
    )

    return (
        mean_pulse_lower,
        mean_pulse_upper,
        mean_pulse_lower_noise,
        mean_pulse_upper_noise,
    )


def estimate_F_phi_chi():
    r"""
    Estimate observed $F_{\phi}$ and $F_{\chi}$ for the acoustic receptions.

    Periodograms are computed for each of the receptions and the periodograms are averaged together in the log domain

    Returns
    -------
    m_data : np.array
        array of vertical wavenumbers [1/m] at which $F_\\phi$ and $F_\\chi$ are estimated
    F_phi_data : dict
        dictionary containing arrays of phase power spectral density, $F_\\phi$ [rad m]
    F_chi_data : dict
        dictionary containing arrays of log-amplitude power spectral density, $F_\\chi$ [rad m]
    F_phi_lm : dict
        dictionary containing arrays of log-mean phase power spectral density, $F_\\phi$ [rad m]
    F_phi_ls : dict
        dictionary containing arrays of log-standard deviation of phase power spectral density, $F_\\phi$ [rad m]
    F_chi_lm : dict
        dictionary containing arrays of log-mean log-amplitude power spectral density, $F_\\chi$ [rad m]
    F_chi_ls : dict
        dictionary containing arrays of log-standard deviation of log-amplitude power spectral density, $F_\\chi$ [rad m]
    """

    fn = get_data_dir() / "dataproducts" / "wavefront_values.pkl"

    with open(fn, "rb") as f:
        wavefront_values = pickle.load(f)

    ray_ids = ["+7", "-7", "-6", "+6"]

    F_chi_data = {}
    F_phi_data = {}

    F_chi_lm = {}
    F_chi_ls = {}
    F_phi_lm = {}
    F_phi_ls = {}

    for ray_id in ray_ids:
        # extract amplitude, intensity, and phase from wavefront values
        amp = np.abs(wavefront_values[ray_id])
        lamp = np.log(amp)
        phase = np.unwrap(np.angle(wavefront_values[ray_id]), axis=1)

        # remove outliers from phase
        phase[np.s_[:, np.r_[4:6, 20]]] = np.nan
        phase = phase - np.nanmean(phase, axis=1)[:, None]
        phase[np.isnan(phase)] = 0

        # m_data [1/m]
        # F_phi_data, F_chi_data [rad m]
        _, F_phi_data[ray_id] = signal.periodogram(phase, fs=1 / 9, axis=1)
        m_data, F_chi_data[ray_id] = signal.periodogram(lamp, fs=1 / 9, axis=1)

        # remove DC comp
        m_data = m_data[1:]
        F_chi_data[ray_id] = F_chi_data[ray_id][:, 1:]
        F_phi_data[ray_id] = F_phi_data[ray_id][:, 1:]

        # Take mean and standard deviation in log space
        F_chi_lm[ray_id] = 10 ** np.mean(np.log10(F_chi_data[ray_id]), axis=0)
        F_chi_ls[ray_id] = 10 ** np.std(np.log10(F_chi_data[ray_id]), axis=0)

        F_phi_lm[ray_id] = 10 ** np.mean(np.log10(F_phi_data[ray_id]), axis=0)
        F_phi_ls[ray_id] = 10 ** np.std(np.log10(F_phi_data[ray_id]), axis=0)

    return m_data, F_phi_data, F_chi_data, F_phi_lm, F_phi_ls, F_chi_lm, F_chi_ls


def phi_chi_spectrum_model(m, GM, phi_noise, chi_noise, ray_id, all_paths=False):
    """
    phi_chi_spectrum_model - compute model of F_phi and F_chi, GM internal waves, WFT, and white observational noise

    Parameters
    ----------
    m : np.array
        array of vertical wavenumbers [rad/m] at which to compute $F_\\phi$ and $F_\\chi$
    GM : float
        energy level of Garrett-Munk spectrum, X corresponds to zeta0 = $\\sqrt(X) \\times$ 7.3 [m]
    phi_noise : float
        observational noise level for phase power spectral density, $F_\\phi$ [rad m]
    chi_noise : float
        observational noise level for log-amplitude power spectral density, $F_\\chi$ [rad m]
    ray_id : str
        ray id for which to compute the model ('+7', '-7', '+6', '-6')
    all_paths : bool, optional
        whether to compute the model for all paths (default: False)
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        zeta0 = 7.3 * np.sqrt(GM)
        rvs = compute_n23_ray_variables(zeta0=zeta0, ray_ids=[ray_id])

        # using first ray pair for now TODO
        if all_paths:
            F_phis = []
            F_chis = []
            F_phis_noise = []
            F_chis_noise = []
            for k in range(len(rvs[ray_id])):
                F_phi, F_chi = compute_F_phi_chi(rvs[ray_id][k], m)
                F_phis.append(F_phi)
                F_chis.append(F_chi)

                F_phis_noise.append(F_phi + phi_noise)
                F_chis_noise.append(F_chi + chi_noise)

            return F_phis_noise, F_chis_noise, F_phis, F_chis
        else:
            F_phi, F_chi = compute_F_phi_chi(rvs[ray_id][0], m)

            F_phi_noise = F_phi + phi_noise
            F_chi_noise = F_chi + chi_noise

            return F_phi_noise, F_chi_noise, F_phi, F_chi


def fit_F_phi_chis(m_data, F_phi_lm, F_chi_lm, F_phi_ls, F_chi_ls):
    """
    given observation and theory (assuming GM), fit GM energy and
        observational noise levels to the observations

    Parameters
    ----------
    m_data : np.array
        array of vertical wavenumbers [1/m] at which $F_\\phi$ and $F_\\chi$ are estimated
        computed with `estimate_F_phi_chi`
    F_phi_lm : dict
        dictionary containing arrays of log-mean phase power spectral density, $F_\\phi$ [rad m]
        computed with `estimate_F_phi_chi`
    F_chi_lm : dict
        dictionary containing arrays of log-mean log-amplitude power spectral density, $F_\\chi$ [rad m]
        computed with `estimate_F_phi_chi`
    F_phi_ls : dict
        dictionary containing arrays of log-standard deviation of phase power spectral density, $F_\\phi$
        computed with `estimate_F_phi_chi`
    F_chi_ls : dict
        dictionary containing arrays of log-standard deviation of log-amplitude power spectral density, $F_\\chi$
        computed with `estimate_F_phi_chi`
    """
    m_data_radm = m_data * 2 * np.pi  # convert to [rad/m]

    def residuals(params, ray_id):

        GM, phi_noise, chi_noise = params

        P_phi = np.log10(F_phi_ls[ray_id]) ** 2
        P_chi = np.log10(F_chi_ls[ray_id]) ** 2

        F_phi_model, F_chi_model, _, _ = phi_chi_spectrum_model(
            m_data_radm, GM, phi_noise, chi_noise, ray_id
        )

        y_phi = np.log10(F_phi_model)
        f_phi = np.log10(F_phi_lm[ray_id])
        W_phi = np.diag(1 / P_phi)

        y_chi = np.log10(F_chi_model)
        f_chi = np.log10(F_chi_lm[ray_id])
        W_chi = np.diag(1 / P_chi)

        L_phi = (y_phi - f_phi).T @ W_phi @ (y_phi - f_phi)
        L_chi = (y_chi - f_chi).T @ W_chi @ (y_chi - f_chi)

        L = L_phi + L_chi

        return L

    # Fit for each ray id

    # Initial guess for [GM, phi_noise, chi_noise]
    results = {}
    for ray_id in tqdm(["+7", "-7", "+6", "-6"]):
        initial_params = np.array([1, 1e0, 2e-1])

        # Minimize
        results[ray_id] = optimize.minimize(
            residuals, initial_params, args=(ray_id,), method="Nelder-Mead"
        )

    return results
