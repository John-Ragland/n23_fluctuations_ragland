"""Collection of utility functions for the data analysis of NESCAFE 2023 Acoustic Fluctions"""

import pickle
import warnings

import nescafe23 as n23
import numpy as np
import pygenray as pr
import scipy
import xarray as xr
import xrsignal as xrs
from matplotlib import pyplot as plt
from nescafe23 import compute_filtered_fctd_grid, open_receptions, parse_ray_id
from nescafe23.data import get_data_dir

## Define global variables
from nescafe23.processing import PEAK_BOXES_LOWER
from scipy import io, optimize, signal, stats
from tqdm import tqdm

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
    fctd_grid = fctd_pilot.interp(
        {
            "range": np.arange(0, fctd.range[-1], 500),
            "depth": np.arange(0, 6000),
        }
    )

    if filt:
        bz, az = signal.butter(4, 1 / z_smooth, fs=1)
        br, ar = signal.butter(4, 1 / r_smooth, fs=1 / 500)

        fctd_smooth = xrs.filtfilt(
            xrs.filtfilt(fctd_grid, dim="range", b=br, a=ar), dim="depth", b=bz, a=az
        )
    else:
        fctd_smooth = fctd_grid

    fctd_slice = fctd_smooth.sel({"range": slice(0, 151000)})
    fctd_flip = fctd_slice.assign_coords(
        {"range": np.flip(fctd_slice.range.values)}
    ).sortby("range")

    return fctd_flip


ARRAY_TILT_OMIT = ("-15b", "-14b", "-6b", "-13b")


def estimate_array_tilt(peak_locations, omit=ARRAY_TILT_OMIT):
    """
    Estimate the lower-array tilt from the per-transmission scatter common to all arrival angles.

    Each ray's picked angle is demeaned over transmissions, and the tilt is the mean of those
    anomalies across rays (`notebooks/publication_figures/receptions.ipynb`, "Array tilt
    estimation and removal").

    Parameters
    ----------
    peak_locations : dict
        picks from `nescafe23.open_peak_locations()`; only ``angles`` is used
    omit : tuple of str
        ray ids left out of the tilt average. The default drops the four rays with the fewest
        good QC flags.

    Returns
    -------
    array_tilt : dict
        ``array_tilt``: (23,) tilt [deg]; ``array_tilt_se``: (23,) its jackknife standard error
        [deg] (`array_tilt_jackknife_se`); ``mean_angles``: ray_id → transmission-averaged angle
        [deg]; ``angles_plus_tilt``: ray_id → (23,) ``mean_angles[ray_id] + array_tilt`` [deg],
        for every ray including the omitted ones; ``omit``; ``notes``.
    """
    angles = peak_locations["angles"]
    mean_angles = {ray_id: np.nanmean(angles[ray_id]) for ray_id in angles}

    anomalies = _array_tilt_anomalies(angles, omit)
    array_tilt = np.nanmean(anomalies, axis=0)

    angles_plus_tilt = {
        ray_id: mean_angles[ray_id] + array_tilt for ray_id in mean_angles
    }

    return {
        "array_tilt": array_tilt,
        "array_tilt_se": _leave_one_ray_out_se(anomalies),
        "mean_angles": mean_angles,
        "angles_plus_tilt": angles_plus_tilt,
        "omit": tuple(omit),
        "notes": "array tilt is the mean over rays of the demeaned peak_locations arrival angles",
    }


def _array_tilt_anomalies(angles, omit):
    """(rays, 23) picked angles of the rays not in `omit`, each demeaned over transmissions."""
    return np.array(
        [
            angles[ray_id] - np.nanmean(angles[ray_id])
            for ray_id in angles
            if ray_id not in omit
        ]
    )


def _leave_one_ray_out_se(anomalies):
    """Leave-one-ray-out jackknife standard error of the mean over rays (axis 0) of `anomalies`."""
    return _jackknife_se(
        np.array(
            [
                np.nanmean(np.delete(anomalies, k, axis=0), axis=0)
                for k in range(anomalies.shape[0])
            ]
        )
    )


def array_tilt_jackknife_se(peak_locations, omit=ARRAY_TILT_OMIT):
    """
    Leave-one-ray-out jackknife standard error of the array tilt of `estimate_array_tilt`.

    The tilt is a mean over rays, so the rays are the jackknife units. Each ray is left out of
    the average in turn, with every ray's mean angle kept at its full-data value.

    Parameters
    ----------
    peak_locations : dict
        picks from `nescafe23.open_peak_locations()`; only ``angles`` is used
    omit : tuple of str
        ray ids left out of the tilt average, as in `estimate_array_tilt`

    Returns
    -------
    array_tilt_se : np.ndarray
        (23,) standard error [deg]
    """
    return _leave_one_ray_out_se(_array_tilt_anomalies(peak_locations["angles"], omit))


ARRIVAL_TIME_WINDOW = (100, 115)
"""Reception time window [s] cross-correlated by the arrival time offset estimate."""

ARRIVAL_TIME_ARRAYS = {
    "upper": (slice(0, 32), np.linspace(125, 540, 50)),
    "lower": (slice(32, None), np.linspace(885, 1315, 50)),
}
"""Hydrophone rows and actual-depth grid [m] of each array in the arrival time offset estimate, as in
``cli.compute_arrival_time_offset``. The receptions have 60 hydrophones, 32 upper and 28 lower."""


def _interp_weights(depths, grid):
    """(len(grid), len(depths)) linear interpolation matrix onto `grid`, zero outside `depths`."""
    order = np.argsort(depths)
    eye = np.eye(len(depths))
    W = np.zeros((len(grid), len(depths)))
    for k, h in enumerate(order):
        W[:, h] = np.interp(grid, depths[order], eye[k], left=0, right=0)
    return W


def _arrival_time_envelope(x, depths, grid):
    """|reception - its mean| on the depth grid, NaN set to zero, as in ``cli.compute_arrival_time_offset``."""
    y = _interp_weights(depths, grid) @ x
    y = y - y.mean()
    y[np.isnan(y)] = 0
    return np.abs(y)


def _peak_lag(taus, r, subsample):
    """Lag of the maximum of `r`, refined by a parabola through it and its neighbours if `subsample`."""
    i = int(np.argmax(r))
    if not subsample or i in (0, len(r) - 1):
        return taus[i]
    y0, y1, y2 = r[i - 1], r[i], r[i + 1]
    den = y0 - 2 * y1 + y2
    return taus[i] + (0.5 * (y0 - y2) / den if den != 0 else 0.0) * (taus[1] - taus[0])


def arrival_time_shifts(receptions, array, drop=None, subsample=True):
    """
    Shift of each transmission's reception envelope relative to the first, for one array.

    Each reception is interpolated onto the array's depth grid (`ARRIVAL_TIME_ARRAYS`), demeaned, and its
    magnitude cross-correlated in time with the first transmission's, depth by depth. The shift is the lag of
    the maximum of the correlation summed over depth. With ``subsample=False`` this is the estimate of
    ``cli.compute_arrival_time_offset``.

    Parameters
    ----------
    receptions : list of xr.DataArray
        complex demodulated receptions, dims ``hydrophone`` (60, with an ``actual_depth`` coordinate) and
        ``time`` [s], one per transmission (``convert_to_xarray(fn)["dmod"]``)
    array : str
        ``"upper"`` or ``"lower"``
    drop : int, optional
        hydrophone of the array (counted from the array's first) left out, for the jackknife
    subsample : bool
        refine the peak with a parabola, so the shift is not quantized to the sample interval. Default True.

    Returns
    -------
    shifts : np.ndarray
        (n_transmissions,) [s]; the first is zero
    """
    rows, grid = ARRIVAL_TIME_ARRAYS[array]
    envelopes = []
    for da in receptions:
        d = da.isel(hydrophone=rows).sel(time=slice(*ARRIVAL_TIME_WINDOW))
        depths = np.asarray(d.actual_depth.values, dtype=float)
        x = d.transpose("hydrophone", "time").values
        if drop is not None:
            keep = np.arange(len(depths)) != drop
            depths, x = depths[keep], x[keep]
        envelopes.append(_arrival_time_envelope(x, depths, grid))

    ts = float(receptions[0].time.diff("time")[0])
    taus = np.arange(-15 / 2, 15 / 2, ts)[1:]
    ref = envelopes[0]
    return np.array(
        [
            _peak_lag(
                taus,
                signal.fftconvolve(ref, np.flip(e, axis=1), mode="same", axes=1).sum(0),
                subsample,
            )
            for e in envelopes
        ]
    )


def _jackknife_se(loo):
    n = loo.shape[0]
    return np.sqrt((n - 1) / n * np.sum((loo - loo.mean(axis=0)) ** 2, axis=0))


def jackknife_arrival_time_offset(receptions, progress=None):
    """
    Common arrival time offset with leave-one-hydrophone-out jackknife standard errors.

    The offset is the mean of the upper- and lower-array shifts of `arrival_time_shifts`, with the peak refined
    to sub-sample lags; without that the jackknife of a quantized argmax is mostly zero. The refined offset
    differs from the stored ``T_t_estimated.pkl`` by at most ~0.2 ms on the real data.

    The jackknife units are the hydrophones (60: 32 upper, 28 lower): each is left out in turn and its array's
    shift recomputed. The per-array SEs leave out only that array's hydrophones. The first transmission is the reference, so its
    SE is zero by construction.

    Parameters
    ----------
    receptions : list of xr.DataArray
        see `arrival_time_shifts`
    progress : callable, optional
        called with no arguments after each leave-one-out estimate, one per hydrophone

    Returns
    -------
    offset : dict
        ``dT_t``, ``dT_t_upper``, ``dT_t_lower`` [s] and their ``*_se`` [s], each (n_transmissions,);
        ``notes``
    """
    full = {a: arrival_time_shifts(receptions, a) for a in ARRIVAL_TIME_ARRAYS}
    loo = {}
    for a, (rows, _) in ARRIVAL_TIME_ARRAYS.items():
        n = receptions[0].isel(hydrophone=rows).sizes["hydrophone"]
        estimates = []
        for h in range(n):
            estimates.append(arrival_time_shifts(receptions, a, drop=h))
            if progress is not None:
                progress()
        loo[a] = np.array(estimates)

    loo_avg = np.concatenate(
        [(loo["upper"] + full["lower"]) / 2, (full["upper"] + loo["lower"]) / 2]
    )
    return {
        "dT_t": (full["upper"] + full["lower"]) / 2,
        "dT_t_upper": full["upper"],
        "dT_t_lower": full["lower"],
        "dT_t_se": _jackknife_se(loo_avg),
        "dT_t_upper_se": _jackknife_se(loo["upper"]),
        "dT_t_lower_se": _jackknife_se(loo["lower"]),
        "notes": (
            "arrival time offset (cross-correlated reception envelope, parabolic sub-sample peak) with "
            "leave-one-hydrophone-out jackknife standard errors; dT_t_se jackknifes over every hydrophone of both arrays"
        ),
    }


def open_array_tilt():
    """
    Open the array tilt estimate, ``dataproducts/array_tilt.pkl``.

    The file is produced by running ``n23_fluctuation_analysis`` and selecting
    **estimate array tilt**.

    Returns
    -------
    array_tilt : dict
        see `estimate_array_tilt`
    """
    path = get_data_dir() / "dataproducts" / "array_tilt.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Array tilt file not found at {path}.\n"
            "Generate it by running 'n23_fluctuation_analysis' and selecting "
            "'estimate array tilt'."
        )
    with path.open("rb") as f:
        return pickle.load(f)


# which part of the hydrophone pattern (`estimate_hydrophone_pattern`) is removed before beamforming
REMOVE_PATTERN_CHOICES = ("none", "phase", "magnitude", "both")
_REMOVE_PATTERN_SUFFIXES = {
    "none": "",
    "phase": "_phase_removed",
    "magnitude": "_magnitude_removed",
    "both": "_phase_magnitude_removed",
}


def _check_remove_pattern(remove_pattern):
    if remove_pattern not in REMOVE_PATTERN_CHOICES:
        msg = f"remove_pattern: expected one of {list(REMOVE_PATTERN_CHOICES)}, got {remove_pattern!r}"
        raise ValueError(msg)


def _product_suffix(remove_array_tilt, remove_pattern):
    _check_remove_pattern(remove_pattern)
    tilt = "_tilt_removed" if remove_array_tilt else ""
    return tilt + _REMOVE_PATTERN_SUFFIXES[remove_pattern]


def wavefront_values_path(remove_array_tilt, remove_pattern="none"):
    """
    Path of the cached wavefront values,
    ``dataproducts/wavefront_values[_tilt_removed][_phase_removed|_magnitude_removed|_phase_magnitude_removed].pkl``.

    Parameters
    ----------
    remove_array_tilt : bool
        If True, the path of the wavefront values with the array tilt removed.
    remove_pattern : str
        hydrophone pattern removed before beamforming, one of `REMOVE_PATTERN_CHOICES`. Default is ``"none"``.
    """
    fn = f"wavefront_values{_product_suffix(remove_array_tilt, remove_pattern)}.pkl"
    return get_data_dir() / "dataproducts" / fn


SPECTRAL_WINDOWS = ("hann", "tukey", "rectangular")
"""Taper windows of `estimate_F_phi_chi`. ``"hann"`` is the default."""

TUKEY_ALPHA = 0.3
"""Fraction of the record inside the cosine tapers of the ``"tukey"`` window (0.15 at each end)."""


def _spectral_method_suffix(window="hann", first_difference=False):
    """File-name suffix of a spectral estimation method of `estimate_F_phi_chi`. The default (Hann, no first
    difference) has none, so its products keep their old names."""
    _check_spectral_window(window)
    return ("_first_difference" if first_difference else "") + (
        "" if window == "hann" else f"_{window}"
    )


def _check_spectral_window(window):
    if window not in SPECTRAL_WINDOWS:
        msg = f"window must be one of {SPECTRAL_WINDOWS}, got {window!r}"
        raise ValueError(msg)


def phi_chi_LS_fit_path(
    remove_array_tilt, remove_pattern="none", window="hann", first_difference=False
):
    """
    Path of the cached ɸ χ GM least squares fit,
    ``dataproducts/phi_chi_LS_fit[_tilt_removed][...][_first_difference][_rectangular|_tukey].pkl``, with
    the suffixes of `wavefront_values_path` and then of the spectral estimation method.

    Parameters
    ----------
    remove_array_tilt : bool
        If True, the path of the fit to the wavefront values with the array tilt removed.
    remove_pattern : str
        hydrophone pattern removed before beamforming, one of `REMOVE_PATTERN_CHOICES`. Default is ``"none"``.
    window : str
        taper of the spectra, one of `SPECTRAL_WINDOWS`: ``"hann"`` (default, no suffix), ``"tukey"`` (adds
        ``_tukey``) or ``"rectangular"`` (adds ``_rectangular``)
    first_difference : bool
        spectra estimated by first differencing and postcoloring; True adds ``_first_difference``
    """
    fn = (
        f"phi_chi_LS_fit{_product_suffix(remove_array_tilt, remove_pattern)}"
        f"{_spectral_method_suffix(window, first_difference)}.pkl"
    )
    return get_data_dir() / "dataproducts" / fn


PAPER_SETTINGS = {
    "remove_array_tilt": True,
    "remove_pattern": "both",
    "window": "hann",
    "first_difference": True,
}
"""Processing choices of the published analysis (the manuscript's Figs 8, 9, 11 and Table II): wavefront values
with the lower-array tilt and the hydrophone phase and magnitude pattern removed, and spectra estimated with a
Hann taper after first differencing. `cli.reproduce_paper` and the publication figures config defaults use it."""


def hydrophone_pattern_path():
    """Path of the hydrophone pattern estimate, ``dataproducts/hydrophone_pattern.pkl``."""
    return get_data_dir() / "dataproducts" / "hydrophone_pattern.pkl"


def open_hydrophone_pattern():
    """
    Open the hydrophone pattern estimate, ``dataproducts/hydrophone_pattern.pkl``.

    The file is produced by running ``n23_fluctuation_analysis`` and selecting
    **estimate hydrophone pattern**.

    Returns
    -------
    pattern : dict
        see `estimate_hydrophone_pattern`
    """
    path = hydrophone_pattern_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Hydrophone pattern file not found at {path}.\n"
            "Generate it by running 'n23_fluctuation_analysis' and selecting "
            "'estimate hydrophone pattern'."
        )
    with path.open("rb") as f:
        return pickle.load(f)


def _linear_detrend(x):
    """Remove the least-squares line along the last axis (the element index)."""
    x = np.asarray(x, dtype=float)
    x2 = np.atleast_2d(x)
    V = np.vander(np.arange(x2.shape[1], dtype=float), 2)
    coef, *_ = np.linalg.lstsq(V, x2.T, rcond=None)
    return (x2 - (V @ coef).T).reshape(x.shape)


def _phase_pattern(X):
    """
    Phase [rad] along the array common to the records `X` (R, n), linearly detrended.

    The mean step between neighbouring elements is the angle of the sum of the unit phasor products
    Z(n) Z*(n-1), which needs no unwrapping (as `estimation.circular_common_phase`). Its cumulative sum is
    removed from each record, and the mean of the linearly detrended, unwrapped residuals is added back.
    """
    Z = X / np.abs(X)
    steps = np.angle(np.sum(Z[:, 1:] * np.conj(Z[:, :-1]), axis=0))
    prealigned = np.concatenate(([0.0], np.cumsum(steps)))
    residual = np.unwrap(np.angle(Z * np.exp(-1j * prealigned)), axis=1)
    return _linear_detrend(prealigned + np.mean(_linear_detrend(residual), axis=0))


def _gain_pattern(X):
    """Amplitude along the array common to the records `X` (R, n): the mean of each record's |X| over its
    array mean, normalized to unit mean."""
    A = np.abs(X)
    g = np.mean(A / np.mean(A, axis=1, keepdims=True), axis=0)
    return g / np.mean(g)


def _phase_residual_rms(X, phase):
    """rms over the array of each record's phase after removing `phase` and the record's own line."""
    residual = np.unwrap(np.angle(X * np.exp(-1j * phase)), axis=1)
    return np.sqrt(np.mean(_linear_detrend(residual) ** 2, axis=1))


def _gain_residual_rms(X, gain):
    A = np.abs(X)
    return np.sqrt(
        np.mean((A / np.mean(A, axis=1, keepdims=True) / gain - 1) ** 2, axis=1)
    )


def _pooled(records, arrivals):
    """The records (R, n) and transmission index of each, of `arrivals`."""
    X = np.concatenate([records[a][1] for a in arrivals], axis=0)
    tx = np.concatenate([records[a][0] for a in arrivals])
    return X, tx


def _jackknifed_pattern(pattern_fn, X, tx):
    """`pattern_fn` of all records, its leave-one-transmission-out jackknife variance, and the number of
    transmissions."""
    pattern = pattern_fn(X)
    loo = [pattern_fn(X[tx != k]) for k in np.unique(tx)]
    return pattern, _jackknife_var(loo), len(loo)


def _pattern_difference_p_value(p_a, v_a, m_a, p_o, v_o, m_o, n_fit):
    """
    p-value of the difference of two patterns with jackknife variances `v_a`, `v_o` from `m_a`, `m_o`
    transmissions. Each element's difference is a Welch t statistic with Welch-Satterthwaite degrees of
    freedom, mapped to the normal score of the same two-sided p-value, and the sum of squared scores is
    compared with chi^2 on n - `n_fit` degrees of freedom. Using the t statistics directly as normal scores
    would overstate the significance, because a jackknife variance from ~20 transmissions is itself noisy.
    """
    v = v_a + v_o
    nu = v**2 / (v_a**2 / (m_a - 1) + v_o**2 / (m_o - 1))
    t = np.abs(p_a - p_o) / np.sqrt(v)
    z = stats.norm.isf(stats.t.sf(t, nu))
    return stats.chi2.sf(np.sum(z**2), len(p_a) - n_fit)


def _reject_arrivals(records, pattern_fn, n_fit, alpha):
    """
    Iterated test of each arrival's pattern against the pattern of the other arrivals pooled, from the
    difference of the two patterns and their leave-one-transmission-out jackknife variances
    (`_pattern_difference_p_value`). The covariance across elements is not used: with fewer transmissions than
    elements the jackknife covariance is singular. The arrival with the smallest p-value is dropped while it is
    below alpha / (number of arrivals), and the test is repeated on the rest.

    Returns
    -------
    kept : list of str
    rejected : dict
        ray_id → p-value when it was dropped
    """
    kept = list(records)
    rejected = {}
    while len(kept) > 2:
        p_values = {}
        for a in kept:
            own = _jackknifed_pattern(pattern_fn, *_pooled(records, [a]))
            others = _jackknifed_pattern(
                pattern_fn, *_pooled(records, [b for b in kept if b != a])
            )
            p_values[a] = _pattern_difference_p_value(*own, *others, n_fit)
        worst = min(p_values, key=p_values.get)
        if p_values[worst] >= alpha / len(kept):
            break
        kept.remove(worst)
        rejected[worst] = float(p_values[worst])
    return kept, rejected


def _reject_records(records, arrivals, residual_rms, z_max):
    """
    Records of `arrivals` whose rms residual from the pattern has a robust z-score, (rms - median) / (1.4826 MAD)
    over all the records, above `z_max`.

    Returns
    -------
    keep : dict
        ray_id → boolean mask over the arrival's records
    rejected : dict
        ray_id → transmission indices rejected (only arrivals with any)
    """
    rms = {a: residual_rms(records[a][1]) for a in arrivals}
    pooled = np.concatenate(list(rms.values()))
    med = np.median(pooled)
    mad = 1.4826 * np.median(np.abs(pooled - med))
    keep = {a: (rms[a] - med) / mad <= z_max for a in arrivals}
    rejected = {
        a: records[a][0][~keep[a]].tolist() for a in arrivals if not np.all(keep[a])
    }
    return keep, rejected


def estimate_hydrophone_pattern(
    wavefront_values,
    omit=ARRAY_TILT_OMIT,
    qc_only=False,
    alpha=0.01,
    z_max=3.5,
    min_records=5,
    f0=250.0,
):
    """
    Estimate the phase and gain of each lower-array hydrophone, the pattern along the array that is common to
    every transmission and arrival.

    The pattern is estimated from the wavefront values (`get_wavefront_values`, without pattern removal) of
    every lower-array arrival except `omit`, ducted and bottom-bounced alike, pooled. The phase pattern is
    linearly detrended along the array, since a line is a beam angle (array tilt) error and not a property
    of the hydrophones (`_phase_pattern`). The gain is the mean of each record's |X| over its array mean,
    normalized to unit mean (`_gain_pattern`). The phase is reported as the delay phase / (2π f0) as well,
    the form in which `remove_hydrophone_pattern` removes it.

    Outliers are rejected in two steps, separately for the phase and the gain:

    1. arrivals: each arrival's pattern is tested against the pattern of the others, with their jackknife
       variances, and the worst is dropped while its p-value is below alpha / (number of arrivals)
       (`_reject_arrivals`);
    2. records: the remaining (transmission, arrival) records whose rms residual from the pattern has a
       robust z-score above `z_max` are dropped (`_reject_records`).

    The standard errors are leave-one-transmission-out jackknife errors of the final pattern. A transmission
    is left out of every arrival at once, because arrivals of one transmission share its tilt and tide.

    Parameters
    ----------
    wavefront_values : dict
        ray_id → complex wavefront values (23, n)
    omit : tuple of str
        arrivals left out. Default is `ARRAY_TILT_OMIT`, the arrivals left out of the array tilt.
    qc_only : bool
        if True, only picks with QC flag 1 or 5 (`nescafe23.processing.PEAK_LOWER_QC`) are used; a short QC
        row counts as unreviewed. Default is False.
    alpha : float
        family-wise significance of the arrival test. Default is 0.01.
    z_max : float
        largest robust z-score of a record's rms residual. Default is 3.5.
    min_records : int
        arrivals with fewer usable transmissions are left out. Default is 5.
    f0 : float
        carrier frequency [Hz] at which the phase is converted to a delay. Default is 250.

    Returns
    -------
    pattern : dict
        ``phase``, ``phase_se`` [rad]; ``delay``, ``delay_se`` [s]; ``gain``, ``gain_se``: (n,) arrays;
        ``per_arrival``: {"phase", "gain"} → ray_id → (n,) pattern of every candidate arrival;
        ``arrivals``: {"phase", "gain"} → arrivals kept; ``rejected_arrivals``: {"phase", "gain"} → ray_id →
        p-value; ``rejected_records``: {"phase", "gain"} → ray_id → rejected transmissions;
        ``n_records``: {"phase", "gain"} → number of records in the final pattern; ``omit``, ``f0``,
        ``notes``.
    """
    records = {}
    for ray_id, X in wavefront_values.items():
        if ray_id in omit:
            continue
        ok = np.all(np.isfinite(X), axis=1) & np.all(X != 0, axis=1)
        if qc_only:
            qc = np.asarray(n23.processing.PEAK_LOWER_QC.get(ray_id, []))
            flagged_good = np.zeros(len(X), dtype=bool)
            n_qc = min(len(qc), len(X))
            flagged_good[:n_qc] = np.isin(qc[:n_qc], (1, 5))
            ok &= flagged_good
        if ok.sum() >= min_records:
            records[ray_id] = (np.flatnonzero(ok), X[ok])

    pattern = {
        "per_arrival": {},
        "arrivals": {},
        "rejected_arrivals": {},
        "rejected_records": {},
        "n_records": {},
        "omit": tuple(omit),
        "f0": f0,
    }
    for key, pattern_fn, n_fit, residual_rms in (
        ("phase", _phase_pattern, 2, _phase_residual_rms),
        ("gain", _gain_pattern, 1, _gain_residual_rms),
    ):
        pattern["per_arrival"][key] = {a: pattern_fn(records[a][1]) for a in records}
        kept, pattern["rejected_arrivals"][key] = _reject_arrivals(
            records, pattern_fn, n_fit, alpha
        )
        p = pattern_fn(_pooled(records, kept)[0])
        keep, pattern["rejected_records"][key] = _reject_records(
            records, kept, lambda X, p=p, f=residual_rms: f(X, p), z_max
        )
        cleaned = {a: (records[a][0][keep[a]], records[a][1][keep[a]]) for a in kept}
        X, tx = _pooled(cleaned, kept)
        p, var, _ = _jackknifed_pattern(pattern_fn, X, tx)
        pattern[key] = p
        pattern[f"{key}_se"] = np.sqrt(var)
        pattern["arrivals"][key] = kept
        pattern["n_records"][key] = len(X)

    pattern["delay"] = pattern["phase"] / (2 * np.pi * f0)
    pattern["delay_se"] = pattern["phase_se"] / (2 * np.pi * f0)
    pattern["notes"] = (
        "phase (linearly detrended) and gain along the lower array common to the transmissions and "
        "arrivals, estimated from the tilt-removed wavefront values with outliers rejected; "
        "delay = phase / (2 pi f0)"
    )
    return pattern


def remove_hydrophone_pattern(da, pattern, remove_pattern):
    """
    Remove the hydrophone pattern from the receptions of the lower array, in the time domain.

    The phase is removed as a time delay, applied as in `nescafe23.ULA_conventional_beamform_nostack`: each
    element is multiplied in the frequency domain by exp(i(ω + ω0) d) with d = -``delay``, which at the carrier
    is exp(-i ``phase``). The gain is removed by dividing each element by ``gain``.

    Parameters
    ----------
    da : xr.DataArray
        complex demodulated receptions of the lower array, dims ``depth`` (in the order of the wavefront
        values) and ``time``
    pattern : dict
        `estimate_hydrophone_pattern`
    remove_pattern : str
        one of `REMOVE_PATTERN_CHOICES`

    Returns
    -------
    da : xr.DataArray
        dims ``depth``, ``time``
    """
    _check_remove_pattern(remove_pattern)
    if remove_pattern == "none":
        return da
    da = da.transpose("depth", "time")
    x = da.values
    if len(pattern["delay"]) != x.shape[0]:
        msg = f"hydrophone pattern has {len(pattern['delay'])} elements, receptions have {x.shape[0]}"
        raise ValueError(msg)

    if remove_pattern in ("phase", "both"):
        dt = float(da.time[1] - da.time[0])
        ws = 2 * np.pi * scipy.fft.fftfreq(x.shape[1], dt)
        w0 = 2 * np.pi * pattern["f0"]
        d = -np.asarray(pattern["delay"])
        x = scipy.fft.ifft(
            scipy.fft.fft(x, axis=1) * np.exp(1j * (ws[None, :] + w0) * d[:, None]),
            axis=1,
        )
    if remove_pattern in ("magnitude", "both"):
        x = x / np.asarray(pattern["gain"])[:, None]
    return da.copy(data=x)


def get_wavefront_values(
    dss, remove_array_tilt=False, remove_pattern="none", half_window=1.0
):
    """
    Unstacked beamformer output of the lower array at each ray's arrival time and angle.

    Each ray is beamformed at its own (arbitrary, off-grid) angle over a short window around
    its arrival, so no angle grid is involved. Arrival times always come from
    `nescafe23.open_peak_locations()`.

    Parameters
    ----------
    dss : list
    remove_array_tilt : bool
        If False, beamform at the angle picked on each transmission (``peak_locations["angles"]``).
        If True, beamform at the ray's transmission-averaged angle plus the array tilt,
        ``open_array_tilt()["angles_plus_tilt"]``, so the per-transmission angle scatter is only
        the tilt common to all rays. Default is False.
    remove_pattern : str
        hydrophone pattern removed from the receptions before beamforming (`remove_hydrophone_pattern`, with
        the pattern of `open_hydrophone_pattern`): ``"none"`` (default), ``"phase"`` (as a time delay),
        ``"magnitude"`` (as a gain), or ``"both"``.
    half_window : float
        half width [s] of the time window beamformed around each arrival. Must comfortably exceed
        the largest plane-wave delay across the array (~46 ms at 35°) so the FFT wrap-around stays
        away from the arrival. Default is 1.0.

    Returns
    -------
    wavefront_amplitudes : dict
        ray_id → complex array (23, 28). Rows with a NaN arrival time or angle are left NaN.
    """
    peak_locations = n23.open_peak_locations()
    arrival_times = peak_locations["times"]
    if remove_array_tilt:
        arrival_angles = open_array_tilt()["angles_plus_tilt"]
    else:
        arrival_angles = peak_locations["angles"]
    _check_remove_pattern(remove_pattern)
    pattern = None if remove_pattern == "none" else open_hydrophone_pattern()

    wavefront_amplitudes = {}

    for ray_id in PEAK_BOXES_LOWER:
        wavefront_amplitudes[ray_id] = np.ones((23, 28), dtype="complex") * np.nan

    for Tx_num in tqdm(range(len(dss)), desc="Tx num"):
        lower_Rx = (
            dss[Tx_num]
            .swap_dims({"hydrophone": "nominal_depth"})
            .rename({"nominal_depth": "depth"})[32:]
        )

        for ray_id in PEAK_BOXES_LOWER:
            t_k = arrival_times[ray_id][Tx_num]
            theta_k = arrival_angles[ray_id][Tx_num]
            if np.isnan(t_k) or np.isnan(theta_k):
                continue

            window = lower_Rx.sel({"time": slice(t_k - half_window, t_k + half_window)})
            if pattern is not None:
                window = remove_hydrophone_pattern(window, pattern, remove_pattern)
            bf, _ = n23.ULA_conventional_beamform_nostack(window, np.array([theta_k]))

            wavefront_amplitudes[ray_id][Tx_num, :] = (
                bf.isel({"angle": 0}).sel({"time": t_k}, method="nearest").values
            )

    return wavefront_amplitudes


def fctd_environment(
    z_smooth: float = 200, r_smooth: float = 15000, front_location: float = 35e3
) -> xr.Dataset:
    """
    The smoothed FCTD environment used for the NESCAFE23 ray runs, shifted so that the front sits
    ``front_location`` meters further from the source.

    Parameters
    ----------
    z_smooth : float
        vertical smoothing scale [m], passed to `compute_filtered_fctd_grid`. Default is 200 m.
    r_smooth : float
        horizontal smoothing scale [m], passed to `compute_filtered_fctd_grid`. Default is 15 km.
    front_location : float
        range shift [m] applied to the FCTD range coordinate. Default is 35 km.

    Returns
    -------
    envi_field : xr.Dataset
        environment with variables ``c`` [m/s], ``N`` [cph], ``dcdz_p`` [1/s], ``t``, ``SP``, ``SA`` on
        ``(depth, range)``
    """
    envi_field = compute_filtered_fctd_grid(z_smooth=z_smooth, r_smooth=r_smooth)

    return envi_field.assign_coords(
        {"range": (envi_field.range.values + front_location)}
    )


def pair_eigenrays(
    ray_id: str,
    eigs: pr.EigenRays,
    eigs_backward: pr.EigenRays,
    ridx: int = 0,
    tol: float = 1.0,
):
    """
    Pair forward and backward eigenrays of one ray id by reciprocity.

    A backward eigenray (launched at the receiver) that retraces a forward eigenray has launch angle
    equal to minus the forward received angle, and received angle equal to minus the forward launch
    angle. Each forward eigenray is assigned the backward eigenray minimizing the larger of the two
    mismatches (one-to-one, `scipy.optimize.linear_sum_assignment`). On the FCTD ray run this
    reproduces the hand-picked `EIG_INDEXS` for the ducted rays.

    Parameters
    ----------
    ray_id : str
        NESCAFE23 ray identifier (e.g. ``'+7'``)
    eigs : pr.EigenRays
        forward eigenrays
    eigs_backward : pr.EigenRays
        backward eigenrays, traced in the range-flipped environment
    ridx : int
        receiver depth index of ``eigs``. Default is 0.
    tol : float
        largest accepted angle mismatch [deg]. Pairs above it are dropped with a warning.

    Returns
    -------
    pairs : list of tuple of int
        ``(eig_idx_forward, eig_idx_backward)`` relative indices, as in `EIG_INDEXS`, sorted by the
        forward index. Empty if the ray id has no forward or backward eigenray.
    """
    ray_id_pr, (sign, number, suffix) = parse_ray_id(ray_id)
    if number % 2 != 0:
        ray_id_back = ray_id
    else:
        ray_id_back = f"{'+' if sign == '-' else '-'}{number}{suffix}"
    ray_id_pr_back, _ = parse_ray_id(ray_id_back)

    fwd = np.where(eigs.ray_id[ridx] == ray_id_pr)[0]
    bck = np.where(eigs_backward.ray_id[0] == ray_id_pr_back)[0]
    if len(fwd) == 0 or len(bck) == 0:
        return []

    fwd_launch = np.asarray(eigs.launch_angles[ridx])[fwd]
    fwd_recv = np.asarray(eigs.received_angles[ridx])[fwd]
    bck_launch = np.asarray(eigs_backward.launch_angles[0])[bck]
    bck_recv = np.asarray(eigs_backward.received_angles[0])[bck]

    cost = np.maximum(
        np.abs(bck_launch[None, :] + fwd_recv[:, None]),
        np.abs(bck_recv[None, :] + fwd_launch[:, None]),
    )
    rows, cols = optimize.linear_sum_assignment(cost)

    pairs = []
    for i, k in zip(rows, cols, strict=True):
        if cost[i, k] > tol:
            warnings.warn(
                f"{ray_id}: forward eigenray {i} has no reciprocal backward eigenray "
                f"(best mismatch {cost[i, k]:.2f} deg > {tol} deg); dropped",
                stacklevel=2,
            )
            continue
        pairs.append((int(i), int(k)))

    unpaired = sorted(set(range(len(fwd))) - set(rows.tolist()))
    if unpaired:
        warnings.warn(
            f"{ray_id}: forward eigenrays {unpaired} outnumber the backward eigenrays; dropped",
            stacklevel=2,
        )

    return sorted(pairs)


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
    f: float | None = None,
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
    f : float, optional
        Coriolis parameter [rad/s]. If None (default), computed from `latitude`.

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

    # integrate N only down to the seafloor of the traced environment (clamped past its ends)
    seafloor = xr.DataArray(
        np.interp(N.range, envi.bathymetry.range, envi.bathymetry),
        dims=["range"],
        coords={"range": N.range},
    )
    N0B_ray = N.where(N.depth <= seafloor, 0).integrate("depth")
    N0B_ray.name = "$N_0 B$"
    N0B_ray.attrs["units"] = "rad m/s"

    # resample along parametric ray range vector
    N0B_ray = N0B_ray.interp({"range": rsx})

    ## Fractional sound speed variance
    mu2 = zeta0**2 * (N0 / Nray) * (dcdz_ray) ** 2 / Cray**2

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

    if f is None:
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
        "m_low": m_low,
        "m_high": m_high,
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
    eig_indexs: dict | str | None = None,
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
        xarray dataset of the environment with variables ``c``, ``N`` [cph] and ``dcdz_p`` [1/s], on the range
        coordinates of the ray run (i.e. already shifted to the front location). If None, it is computed with
        `compute_filtered_fctd_grid(z_smooth=200, r_smooth=15000)` and shifted by 35 km.
    ray_run : dict, optional
        dictionary containing ray run data, generated with `run_rays`. If None, load from pre-computed data (generated from CLI)
    eig_indexs : dict or "reciprocity", optional
        forward/backward eigenray pairs per ray id. If None (default), the hand-picked `EIG_INDEXS`, which
        are only valid for the cached FCTD ray run. ``"reciprocity"`` pairs them with `pair_eigenrays`, for
        any environment; ray ids without eigenrays are then skipped with a warning.

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

    if envi_field is None:
        envi_field = fctd_environment()

    dcdz_p = envi_field["dcdz_p"]

    N = envi_field["N"]

    # remove negative buoancy frequency
    N = N.where(N > 0, 0)

    N_rad = N * (2 * np.pi) / 3600
    N_rad.attrs["units"] = "rad/s"

    fn = get_data_dir() / "dataproducts" / "ray_run_forward_back.pkl"

    if ray_run is None:
        with fn.open("rb") as file:
            ray_run = pickle.load(file)

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

    if eig_indexs is None:
        eig_indexs = EIG_INDEXS
    elif eig_indexs == "reciprocity":
        eig_indexs = {
            ray_id: pair_eigenrays(ray_id, eigs, eigs_back) for ray_id in ray_ids
        }
        for ray_id in ray_ids:
            if not eig_indexs[ray_id]:
                warnings.warn(
                    f"{ray_id}: no forward/backward eigenray pair; skipped",
                    stacklevel=2,
                )

    tqdm_disable = len(ray_ids) == 1

    for _k, ray_id in tqdm(enumerate(ray_ids), disable=tqdm_disable):
        if not eig_indexs.get(ray_id):
            continue
        rvs[ray_id] = []
        for n in range(len(eig_indexs[ray_id])):
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
                    eig_idx_forward=eig_indexs[ray_id][n][0],
                    eig_idx_backward=eig_indexs[ray_id][n][1],
                    f=f,
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
                / (rv["Cray"] ** 2 * rv["mean_jn1"])
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
    fresnel: bool = True,
    per_point_band: bool = False,
):
    r"""
    Given the dictionary of ray variables, computed with `compute_ray_path_variables`, compute power spectral density of phase and log-amplitude ($F_\\phi$ and $F_\\chi$)

    The spectra are Colosi (2016) Eq. 6.52, which is a one-sided density per unit wavenumber in rad/m
    (m = π j N / N0B, j ≥ 1). They are returned as a one-sided density per cycle/m, i.e. multiplied by the
    Jacobian 2π, so they can be compared directly with the periodograms of `estimate_F_phi_chi` (sampled at
    9 m, so per cycle/m) and with the spectra plotted per cpm in Colosi (2016), Figs. 6.1 and 6.10.

    By default (``per_point_band=False``) every point of the ray is integrated at every m, and the result is
    masked after the integral to the receiver band ``rv["m_lo_max"] < m / 2π < rv["m_hi_max"]`` (the largest
    mode-1 and mode-J wavenumbers along the ray, in cycles/m).

    With ``per_point_band=True`` the band is applied inside the integral: each point s of the ray contributes
    only over its own internal-wave band, mode 1 to mode J(s): $\pi N/N_0B \le m \le J \pi N/N_0B$, with $m$
    the local wavenumber. The edges are the modes themselves (not half-integer bins), so $\int F_\phi dm$ in
    the geometric limit is about 0.6 $\Phi^2$ of `compute_lambda_phi`, which sums the discrete modes
    (Eq. 6.56): the continuous integral from mode 1 undercounts the mode-1-dominated sum.

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
    fresnel : bool
        whether to include the Fresnel filter, default is True, which is consistent with Colosi (2016) Eq. 6.51
        This is only done for the phase, log-amplitude remains unchanged
    per_point_band : bool
        whether to limit each point of the ray to its own band, mode 1 to mode J(s), inside the integral.
        Default is False, which masks the integrated spectra to the receiver band [m_lo_max, m_hi_max].

    Returns
    -------
    F_phi : np.array
        array of one-sided phase power spectral density per cycle/m, $F_\\phi$ [1/cpm], at each vertical
        wavenumber in m. NaN outside the band.
    F_chi : np.array
        array of one-sided log-amplitude power spectral density per cycle/m, $F_\\chi$ [1/cpm], at each vertical
        wavenumber in m. NaN outside the band.
    """

    if per_point_band:
        # band of each point along the ray, mode 1 to mode J(s), in local wavenumber
        m_low = (np.pi * rv["Nray"] / rv["N0B_ray"]).values[:, None]  # (s, 1)
        m_high = rv["J"].values[:, None] * m_low
        band = (m >= m_low) & (m <= m_high)  # (s, M)

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

    if fresnel:
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
    else:
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
            * 2
            * np.sqrt(rv["drdt"][:, None] ** 2 + rv["dzdt"][:, None] ** 2)
        )
    # set nan values of integrand, and points outside their band, to zero
    F_phi_integrand[np.isnan(F_phi_integrand)] = 0
    if per_point_band:
        F_phi_integrand = np.where(band, F_phi_integrand, 0)

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
    if per_point_band:
        F_chi_integrand = np.where(band, F_chi_integrand, 0)
    F_chi = scipy.integrate.simpson(F_chi_integrand, axis=0)

    # Eq. 6.52 is a density per rad/m; convert to a density per cycle/m
    F_phi = 2 * np.pi * F_phi
    F_chi = 2 * np.pi * F_chi

    if per_point_band:
        # wavenumbers no point of the ray contributes to
        valid_mask = band.any(axis=0)
    else:
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
        dictionary of phase power spectral densities per cycle/m for each ray_id.
    F_chis : dict
        dictionary of log-amplitude power spectral densities per cycle/m for each ray_id.
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


def compute_wft_structure_function(
    rv: dict,
    deltaz_r: np.array,
    m: np.array,
    fc: float = 250,
    jstar: float = 3,
    f: float = 9.216181213401561e-05,  # [rad/s]
    fresnel: bool = True,
    per_point_band: bool = False,
):
    r"""
    Phase and log-amplitude structure functions of weak fluctuation theory, from the continuum spectra of
    `compute_F_phi_chi` (Colosi 2016 Eq. 6.52), $D(\Delta z) = 2\int_0^\infty F(m)[1 - \cos(2\pi m \Delta z)]dm$
    with $m$ in cycles/m. Wavenumbers outside the ray's modal band (NaN in `compute_F_phi_chi`) count as zero.

    Like Eq. 6.52 this is the local straight-ray result: there is no $\xi_1$ dilation of the separation.
    See `reference/phase_structure_function_wft_vs_path_integral.md`.

    Parameters
    ----------
    rv : dict
        ray variables dictionary, computed with `compute_ray_path_variables`
    deltaz_r : np.array
        vertical separations [m]
    m : np.array
        vertical wavenumbers [cycles/m] of the integration grid, e.g. ``np.logspace(-5, 0, 4000)``
    fc : float
        acoustic center frequency [1/s]
    jstar : float
        modal bandwidth parameter for GM spectrum
    f : float
        Coriolis parameter [rad/s]
    fresnel : bool
        whether to include the Fresnel filter, default is True, which is consistent with Colosi (2016) Eq. 6.51
    per_point_band : bool
        passed to `compute_F_phi_chi`: whether each point of the ray is limited to its own band inside the
        integral, rather than masking the result to the receiver band. Default is False.

    Returns
    -------
    D_phi : np.array
        phase structure function [rad$^2$] at each separation in ``deltaz_r``
    D_chi : np.array
        log-amplitude structure function at each separation in ``deltaz_r``
    """
    F_phi, F_chi = compute_F_phi_chi(
        rv,
        2 * np.pi * m[None, :],
        fc=fc,
        jstar=jstar,
        f=f,
        fresnel=fresnel,
        per_point_band=per_point_band,
    )

    kernel = 1 - np.cos(2 * np.pi * m[None, :] * deltaz_r[:, None])  # (dzr, m)
    D_phi = 2 * scipy.integrate.trapezoid(
        np.nan_to_num(F_phi)[None, :] * kernel, m, axis=1
    )
    D_chi = 2 * scipy.integrate.trapezoid(
        np.nan_to_num(F_chi)[None, :] * kernel, m, axis=1
    )

    return D_phi, D_chi


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
    # set non-finite values (e.g. N = 0) to zero
    phi2_integrand[~np.isfinite(phi2_integrand)] = 0
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
    lam_integrand[~np.isfinite(lam_integrand)] = 0
    Lamda = scipy.integrate.simpson(lam_integrand, axis=0)

    return Lamda, Phi2


def compute_phase_structure_integrand(
    rv: dict,
    deltaz_r: np.array,
    fc: float = 250,
):
    r"""
    Given the ray variables, calculate the phase structure function integrand as a function of vertical seperation $\delta z_r$ and range.

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
    integrand : np.array
        array of phase structure function integrand values at each vertical separation in deltaz_r and range value
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
        2 * omegac**2 * rv["mu2"] * rv["L_p"] / (rv["Cray"] ** 2 * rv["mean_jn1"])
    ).values
    integrand = (
        integrand1[:, None]
        * np.sum(summand, axis=1)
        * np.sqrt(rv["drdt"] ** 2 + rv["dzdt"] ** 2)[:, None]
    )  # (s,dzr)

    # set any nan values to zero
    integrand = np.where(np.isfinite(integrand), integrand, 0)

    return integrand


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

    integrand = compute_phase_structure_integrand(rv, deltaz_r, fc)
    D_dzr = scipy.integrate.simpson(integrand, axis=0)

    return D_dzr, integrand


def compute_brack_sigma2(
    rv: dict,
    f: float = 9.216181213401561e-05,  # [rad/s]
):
    r"""
    Given the ray variables, compute the spectrally averaged squared internal-wave frequency $\{\sigma^2\}$
    along the ray, under the perpendicular wavenumber constraint (Colosi 2016, Appendix C, Eqs. C.1 and C.2).

    Near ray turning points $\sigma_L \to f$ and the closed form suffers catastrophic cancellation, so the
    axial-ray limit $\{\sigma^2\} = 2 N^2 f^2 \ln(N/f) / (N^2 - f^2)$ (Colosi 2016, §7.4.4) is used there.

    Parameters
    ----------
    rv : dict
        ray variables dictionary, computed with `compute_ray_path_variables`
    f : float
        Coriolis parameter [rad/s]

    Returns
    -------
    brack_sigma2 : xr.DataArray
        $\{\sigma^2\}$ [rad$^2$/s$^2$] along the ray path. Set to zero where undefined (e.g. $N = 0$, or
        $\sigma_L \geq N$ so that no internal waves satisfy the resonance condition)
    """

    N = rv["Nray"]
    sigma_L2 = f**2 + N**2 * np.tan(rv["Theta_ray"]) ** 2
    sigma_L = np.sqrt(sigma_L2)

    # undefined values (N = 0, sigma_L >= N, turning points) are expected and set to zero below
    with np.errstate(divide="ignore", invalid="ignore"):
        # normalization, Eq. C.2 (same quantity as W in `compute_F_phi_chi`)
        M_sigma_inv = 0.5 * (
            np.sqrt((1 - (sigma_L2 / N**2)) * (1 - (f**2 / N**2)))
            + ((sigma_L / f) - (f / sigma_L))
            * np.log(
                (
                    np.sqrt(1 - (f**2 / N**2))
                    + np.sqrt((f**2 / sigma_L2) - (f**2 / N**2))
                )
                / (np.sqrt(1 - (f**2 / sigma_L2)))
            )
        )

        # Eq. C.1
        A = np.sqrt(N**2 - sigma_L2) * np.sqrt(N**2 - f**2)
        brack_sigma2 = (
            sigma_L2
            / (2 * M_sigma_inv)
            * (
                np.log((2 * N**2 - sigma_L2 - f**2 + 2 * A) / (sigma_L2 - f**2))
                + (f / sigma_L)
                * np.log(
                    (
                        N**2 * (sigma_L2 + f**2)
                        - 2 * f**2 * sigma_L2
                        - 2 * f * sigma_L * A
                    )
                    / (N**2 * (sigma_L2 - f**2))
                )
            )
        )

        # axial-ray limit, where Eq. C.1 is numerically unstable
        brack_sigma2_axial = 2 * N**2 * f**2 * np.log(N / f) / (N**2 - f**2)
    brack_sigma2 = brack_sigma2.where(
        (sigma_L2 - f**2) / f**2 > 1e-4, brack_sigma2_axial
    )

    # set invalid values to zero
    brack_sigma2 = brack_sigma2.where(np.isfinite(brack_sigma2), 0)
    brack_sigma2.name = "$\\{\\sigma^2\\}$"
    brack_sigma2.attrs["units"] = "rad$^2$/s$^2$"

    return brack_sigma2


def compute_vertical_coherence_length(
    rv: dict,
    fc: float = 250,
):
    r"""
    Given the ray variables, compute the vertical coherence length $z_0$ (Colosi 2016, Eq. 7.52, with
    $\{m^2\}$ from Eq. 7.53). $z_0$ is the receiver separation at which the coherence decays to $e^{-1/2}$,
    from the quadratic (small separation) approximation of the phase structure function.

    Parameters
    ----------
    rv : dict
        ray variables dictionary, computed with `compute_ray_path_variables`
    fc : float
        acoustic center frequency [1/s]

    Returns
    -------
    z0 : float
        vertical coherence length [m]
    integrand : np.array
        integrand of $z_0^{-2}$ along the ray path
    """

    wc = 2 * np.pi * fc

    integrand = (
        wc**2
        * rv["mu2"].values
        * rv["L_p"].values
        / (rv["Cray"].values ** 2)
        * rv["brack_m2"].values
        * rv["RT_zeta_1"] ** 2
        * np.sqrt(rv["drdt"] ** 2 + rv["dzdt"] ** 2)
    )
    # set any nan values to zero
    integrand = np.where(np.isfinite(integrand), integrand, 0)

    z0 = 1 / np.sqrt(scipy.integrate.simpson(integrand, axis=0))

    return z0, integrand


def compute_time_coherence_length(
    rv: dict,
    fc: float = 250,
    f: float = 9.216181213401561e-05,  # [rad/s]
):
    r"""
    Given the ray variables, compute the coherence time $t_0$ (Colosi 2016, Eq. 7.55, with $\{\sigma^2\}$
    from Appendix C, Eq. C.1). $t_0$ is the time separation at which the coherence decays to $e^{-1/2}$,
    from the quadratic (small separation) approximation of the phase structure function.

    Parameters
    ----------
    rv : dict
        ray variables dictionary, computed with `compute_ray_path_variables`
    fc : float
        acoustic center frequency [1/s]
    f : float
        Coriolis parameter [rad/s]

    Returns
    -------
    t0 : float
        coherence time [s]
    integrand : np.array
        integrand of $t_0^{-2}$ along the ray path
    """

    wc = 2 * np.pi * fc

    brack_sigma2 = compute_brack_sigma2(rv, f)

    with np.errstate(invalid="ignore"):
        integrand = (
            wc**2
            * rv["mu2"].values
            * rv["L_p"].values
            / (rv["Cray"].values ** 2)
            * brack_sigma2.values
            * np.sqrt(rv["drdt"] ** 2 + rv["dzdt"] ** 2)
        )
    # set any nan values to zero
    integrand = np.where(np.isfinite(integrand), integrand, 0)

    t0 = 1 / np.sqrt(scipy.integrate.simpson(integrand, axis=0))

    return t0, integrand


def effective_sample_size(
    M: int, dt: float | None = None, t0: float | None = None
) -> float:
    r"""
    Effective number of independent snapshots $M_\mathrm{eff}$ of $M$ snapshots spaced $\Delta t$ apart,
    with Gaussian time coherence $\rho_t(\tau) = \exp(-\tau^2 / 2 t_0^2)$ (Bayley & Hammersley 1946):

    $$M_\mathrm{eff} = \frac{M^2}{\sum_{k,k'} \rho_t^2((k-k')\Delta t)}
        = \frac{M}{\sum_{|j|<M} (1 - |j|/M)\, \rho_t^2(j \Delta t)}$$

    $M_\mathrm{eff} = M$ for $t_0 \to 0$, and $M_\mathrm{eff} \approx M \Delta t / (\sqrt{\pi} t_0)$ for
    $M \Delta t \gg t_0$. $\rho_t^2$ (not $\rho_t$) appears because this is the effective sample size for a
    second-order statistic (a correlation estimate), not for a mean.

    Parameters
    ----------
    M : int
        number of snapshots
    dt : float
        time between snapshots, same units as `t0`
    t0 : float
        coherence time, same units as `dt`. If None or 0, the snapshots are independent and M is returned.

    Returns
    -------
    M_eff : float
        effective number of independent snapshots, $1 \le M_\mathrm{eff} \le M$
    """
    if t0 is None or t0 == 0:
        return float(M)
    if dt is None:
        msg = "dt is required when t0 is given"
        raise ValueError(msg)

    j = np.arange(-M + 1, M)
    rho_t2 = np.exp(-((j * dt / t0) ** 2))
    return float(M / np.sum((1 - np.abs(j) / M) * rho_t2))


def mutual_coherence_covariance(R, M_eff: float, gamma: float = 0.5):
    r"""
    Covariance between lags of the (real part of the) unbiased mutual coherence estimate
    $\hat{C}(l) = \frac{1}{M} \sum_k \frac{1}{N_l} \sum_{n<N_l} x_k[n+l] x_k^*[n]$, $N_l = N - |l|$
    (Bartlett's formula). For a zero-mean Gaussian field with correlation $R(d)$ in the sample dimension,
    Isserlis' theorem gives

    $$\mathrm{Cov}[\hat{C}(l), \hat{C}(m)] = \frac{\gamma}{M_\mathrm{eff} N_l N_m}
        \sum_{n<N_l} \sum_{n'<N_m} \left[ R(n-n'+l-m) R(n-n') + R(n-n'+l) R(n-n'-m) \right]$$

    where $\gamma = 1/2$ for the real part of a circular complex field and $\gamma = 1$ for a real field.
    Every argument of $R$ in the sum has $|d| \le N-1$.

    Parameters
    ----------
    R : np.array
        Array of shape (N,), real correlation at lags 0, ..., N-1 (symmetric, $R(-d) = R(d)$)
    M_eff : float
        effective number of independent snapshots, see `effective_sample_size`
    gamma : float
        1/2 for complex data, 1 for real data

    Returns
    -------
    cov : np.array
        Array of shape (N,N), covariance of $\hat{C}$ between lags 0, ..., N-1, in units of R^2
    """
    R = np.asarray(R, dtype=float)
    N = len(R)
    # R_sym[d + N - 1] = R(d) for d in -(N-1), ..., N-1
    R_sym = np.concatenate([R[:0:-1], R])

    def R_at(d):
        return R_sym[d + N - 1]

    cov = np.empty((N, N))
    for lag_l in range(N):
        for lag_m in range(lag_l, N):
            d = np.subtract.outer(np.arange(N - lag_l), np.arange(N - lag_m))
            s = np.sum(
                R_at(d + lag_l - lag_m) * R_at(d) + R_at(d + lag_l) * R_at(d - lag_m)
            )
            cov[lag_l, lag_m] = cov[lag_m, lag_l] = (
                gamma * s / (M_eff * (N - lag_l) * (N - lag_m))
            )

    return cov


def estimate_mutual_coherence(
    X,
    T,
    omit_zero_lag: bool = True,
    norm: bool = True,
    dt: float | None = None,
    t0: float | None = None,
):
    r"""
    Given array X of shape (M,N) containing M snapshots of vector $\\in \\mathbb{R}^N$
        where the vector contains equally spaced samples in dimension that mutual coherence is to be estimated.

    The mutual coherence is estimated with an unbiased complex cross-correlation of the values of X along dim 1,
        and averaged along all common lags, and all snapshots.

    The variance of the estimate is from Bartlett's formula (`mutual_coherence_covariance`), with the
    estimated (unnormalized, real part) correlation $\hat{R}(d)$ at all lags plugged in for the true $R(d)$.
    No shape is assumed for the coherence along dim 1. $\hat{R}(0)$ includes the noise, which correctly adds
    to the variance of the zero-lag products. The snapshots are correlated with Gaussian coherence time `t0`,
    which reduces M to `effective_sample_size(M, dt, t0)`. When normalized by the reference lag $r$, the
    variance of the ratio is from the delta method,

    $$\mathrm{Var}\left[\frac{\hat{C}(l)}{\hat{C}(r)}\right] \approx \frac{\mathrm{Var}\,\hat{C}(l)
        - 2\, MC(l)\, \mathrm{Cov}[\hat{C}(l), \hat{C}(r)] + MC(l)^2\, \mathrm{Var}\,\hat{C}(r)}{\hat{C}(r)^2}$$

    which is 0 at $l = r$.

    Assumptions: zero mean, Gaussian, stationary along dim 1, and separable in dim 1 and snapshots. Real
    input (e.g. `np.abs(X)`) uses $\gamma = 1$, but its nonzero mean is not removed, so its variance is only
    approximate. $\hat{R}$ at the largest lags is from few samples and noisy; it is not truncated.

    Parameters
    ----------
    X : np.array
        Array of shape (M,N), containing M snapshots of vector $\\in \\mathbb{R}^N$
    T : sample spacing of samples in dim 1of X. Units arbitrary
    omit_zero_lag : bool
        If True, omit the zero lag from the mutual coherence estimate, normalized to next smallest lag.
        If False, zero lag is included and normalized by this. Default is True.
    norm : bool
        If True, normalize the mutual coherence by the value at the smallest non-zero lag (to remove bias from noise at lag=0). Default is True.
    dt : float
        time between snapshots, same units as `t0`. Required if `t0` is given.
    t0 : float
        coherence time between snapshots. If None (default), the snapshots are independent.

    Returns
    -------
    t : np.array
        Array containing the lags of MC in units of T
    MC : np.array
        mutual coherence at lags `t`, normalized by the value at the reference lag if `norm`
    MC_var : np.array
        variance of MC at lags `t`, in the units of MC squared
    MC_noise_ratio : float
        ratio of the lag 0 to lag 1 mutual coherence
    denom : np.array
        number of samples N - |lag| at lags `t`, per snapshot
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
    if not omit_zero_lag:
        if norm:
            MC = MC_full.real[N - 1 :] / MC_full.real[N - 1]
        else:
            MC = MC_full.real[N - 1 :]
    else:
        MC = MC_full.real[N:] / MC_full.real[N] if norm else MC_full.real[N:]

    MC_noise_ratio = MC_full.real[N - 1] / MC_full.real[N]

    if not omit_zero_lag:
        t = np.arange(0, N * T - np.finfo(float).eps, T)
        start = N - 1
    else:
        t = np.arange(T, N * T - np.finfo(float).eps, T)
        start = N

    # variance of estimate (Bartlett), on lags 0, ..., N-1
    R = MC_full.real[N - 1 :]
    gamma = 0.5 if np.iscomplexobj(X) else 1.0
    cov = mutual_coherence_covariance(R, effective_sample_size(M, dt, t0), gamma)

    # restrict to the same lags as MC
    lag_idx = np.arange(start - (N - 1), N)
    if norm:
        r = lag_idx[0]  # reference lag
        MC_var = (
            np.diag(cov)[lag_idx] - 2 * MC * cov[lag_idx, r] + MC**2 * cov[r, r]
        ) / R[r] ** 2
    else:
        MC_var = np.diag(cov)[lag_idx]

    return t, MC, MC_var, MC_noise_ratio, denom[start:]


def gaussian_coherence_model(delta, A0, L):
    r"""
    Gaussian coherence model $A_0 \exp(-\frac{1}{2} (\Delta / L)^2)$. The coherence decays to
    $A_0 e^{-1/2}$ at $\Delta = L$, the same definition as `compute_vertical_coherence_length` and
    `compute_time_coherence_length`.
    """
    return A0 * np.exp(-0.5 * (delta / L) ** 2)


def fit_coherence_length(lags, MC, L0: float):
    r"""
    Least squares fit of `gaussian_coherence_model` to an estimated mutual coherence.

    Parameters
    ----------
    lags : np.array
        lags of the mutual coherence estimate
    MC : np.array
        mutual coherence estimate at `lags`
    L0 : float
        initial guess of the coherence length, in units of `lags`

    Returns
    -------
    A0 : float
        fitted amplitude, NaN if the fit fails
    L : float
        fitted coherence length in units of `lags`, NaN if the fit fails
    """
    lags = np.asarray(lags, dtype=float)
    MC = np.asarray(MC, dtype=float)
    valid = np.isfinite(lags) & np.isfinite(MC)

    try:
        popt, _ = optimize.curve_fit(
            gaussian_coherence_model,
            lags[valid],
            MC[valid],
            p0=(1.0, L0),
            bounds=(0, np.inf),
        )
    except (RuntimeError, ValueError, TypeError):
        return np.nan, np.nan

    return float(popt[0]), float(popt[1])


def fit_vertical_coherence(dz, MC, L0: float = 100.0, A0_init: float | None = None):
    r"""
    Least squares fit of `gaussian_coherence_model` to the nonzero lags of an unnormalized mutual
    coherence. The zero lag is biased by noise and is omitted from the fit.

    The fit is made in scaled parameters, $z_0 / L_0$ and $A_0 / A_{0,\mathrm{init}}$, both starting at 1,
    with the residuals divided by $A_{0,\mathrm{init}}$. Unscaled, $A_0 \sim 10^{12}$ and $z_0 \sim 10^2$,
    and ``least_squares`` (even with ``x_scale="jac"``) sometimes stopped on its ``xtol`` test at the
    initial guess or short of the minimum while still reporting success. That made 12 of the 23
    leave-one-out refits of ``+7`` (tilt and pattern removed) return the initial guess, which blew up
    the `jackknife_vertical_coherence` error bars of ``Ducted_VC``. Checked on 2026-11-02 against a
    20-start grid on every ray, product and leave-one-out subset: the scaled fit reaches the best
    minimum on all of them except ``-6b``, whose coherence is already negative at the first lag.

    Parameters
    ----------
    dz : np.array
        lags of the mutual coherence estimate, starting at lag 0
    MC : np.array
        unnormalized mutual coherence estimate at `dz`
    L0 : float
        initial guess of the coherence length, in units of `dz`
    A0_init : float, optional
        initial guess of the amplitude, in units of `MC`, and the scale of the amplitude and the
        residuals. Default is ``MC[1]``, the coherence at the first nonzero lag.

    Returns
    -------
    A0 : float
        fitted amplitude
    z0 : float
        fitted coherence length in units of `dz`

    Raises
    ------
    ValueError
        if `A0_init` (by default ``MC[1]``) is not positive and finite, or the fit fails: the optimizer
        reports failure, or either parameter ends on its lower bound of zero.
    """
    dz = np.asarray(dz, dtype=float)[1:]
    MC = np.asarray(MC, dtype=float)[1:]
    valid = np.isfinite(dz) & np.isfinite(MC)

    if A0_init is None:
        A0_init = float(MC[0])
    if not (np.isfinite(A0_init) and A0_init > 0):
        msg = (
            f"fit_vertical_coherence: A0_init must be positive and finite, got {A0_init:.3g} "
            "(by default the coherence at the first nonzero lag, which is not positive when the "
            "coherence decorrelates within one lag)"
        )
        raise ValueError(msg)

    def residuals(params):
        z0, A0 = params
        return (
            gaussian_coherence_model(dz[valid], A0 * A0_init, z0 * L0) - MC[valid]
        ) / A0_init

    result = optimize.least_squares(
        residuals, x0=[1.0, 1.0], bounds=([0, 0], [np.inf, np.inf])
    )
    z0, A0 = result.x * [L0, A0_init]
    if not result.success or np.any(result.x <= 0):
        msg = (
            f"fit_vertical_coherence: the Gaussian fit failed ({result.message}), "
            f"z0 = {z0:.3g}, A0 = {A0:.3g}"
        )
        raise ValueError(msg)

    return float(A0), float(z0)


def _jackknife_var(theta):
    """
    Jackknife variance, (M-1)/M sum_k (theta_k - mean(theta))^2, of leave-one-out estimates `theta`
    of shape (M, ...).
    """
    theta = np.asarray(theta, dtype=float)
    M = theta.shape[0]
    return (M - 1) / M * np.sum((theta - np.mean(theta, axis=0)) ** 2, axis=0)


def jackknife_vertical_coherence(X, dz: float = 9.0):
    r"""
    Vertical coherence of one ray, normalized by the amplitude of a Gaussian fit, with leave-one-out
    jackknife variances over transmissions.

    The unnormalized mutual coherence $\hat{C}$ (`estimate_mutual_coherence`, zero lag kept) is fitted by
    `fit_vertical_coherence` to give $A_0$ and $z_0$, and the vertical coherence is $\hat{C} / A_0$. For
    each transmission $k$, $\hat{C}_k$, $A_{0,k}$ and $z_{0,k}$ are recomputed without it, and the
    variance of the vertical coherence is the jackknife variance of the ratio $\hat{C}_k / A_{0,k}$, so
    it includes the uncertainty of $A_0$.

    Parameters
    ----------
    X : np.array
        wavefront values of shape (transmission, hydrophone)
    dz : float
        hydrophone spacing [m]

    Returns
    -------
    vc : dict
        - lags : lags [m], starting at 0
        - MC : unnormalized mutual coherence at `lags`
        - A0, z0 : Gaussian fit amplitude and coherence length [m] of `MC`
        - VC : vertical coherence, MC / A0
        - VC_var, A0_var, z0_var : jackknife variances of VC (per lag), A0 and z0
        - MC_jk, A0_jk, z0_jk : leave-one-out estimates, with leading dimension transmission
    """
    X = np.asarray(X)
    M = X.shape[0]

    lags, MC, _, _, _ = estimate_mutual_coherence(
        X, dz, omit_zero_lag=False, norm=False
    )
    A0, z0 = fit_vertical_coherence(lags, MC)

    MC_jk = []
    A0_jk = []
    z0_jk = []
    for k in range(M):
        _, MC_k, _, _, _ = estimate_mutual_coherence(
            np.delete(X, k, axis=0), dz, omit_zero_lag=False, norm=False
        )
        A0_k, z0_k = fit_vertical_coherence(lags, MC_k)
        MC_jk.append(MC_k)
        A0_jk.append(A0_k)
        z0_jk.append(z0_k)

    MC_jk = np.array(MC_jk)
    A0_jk = np.array(A0_jk)
    z0_jk = np.array(z0_jk)

    return {
        "lags": lags,
        "MC": MC,
        "A0": A0,
        "z0": z0,
        "VC": MC / A0,
        "VC_var": _jackknife_var(MC_jk / A0_jk[:, None]),
        "A0_var": float(_jackknife_var(A0_jk)),
        "z0_var": float(_jackknife_var(z0_jk)),
        "MC_jk": MC_jk,
        "A0_jk": A0_jk,
        "z0_jk": z0_jk,
    }


def estimate_observed_coherence_lengths(
    wavefront_values: dict, dz: float = 9.0, dt: float = 3600.0
):
    r"""
    Estimate the observed vertical coherence length $z_0$ and coherence time $t_0$ for each ray by fitting
    `gaussian_coherence_model` to the mutual coherence of the wavefront values.

    $z_0$ is from `jackknife_vertical_coherence`: a fit to the nonzero lags of the unnormalized depth
    mutual coherence (the zero lag is biased by noise), with a leave-one-transmission-out jackknife
    standard deviation. The time mutual coherence keeps the zero lag, using the hydrophones as snapshots
    and the transmissions as time samples.

    Parameters
    ----------
    wavefront_values : dict
        wavefront values for each ray ID, of shape (transmission, hydrophone), from `get_wavefront_values`
    dz : float
        hydrophone spacing [m], default is the 9 m design spacing
    dt : float
        time between transmissions [s], default is 1 hour

    Returns
    -------
    coherence_lengths : dict
        for each ray ID, a dict with `z0` [m], the Gaussian fit amplitude `A0`, their jackknife standard
        deviations `z0_std` [m] and `A0_std`, and `t0` [s]
    """
    coherence_lengths = {}
    for ray_id, X in wavefront_values.items():
        vc = jackknife_vertical_coherence(X, dz)

        dts, MC_t, _, _, _ = estimate_mutual_coherence(X.T, dt, omit_zero_lag=False)
        _, t0 = fit_coherence_length(dts, MC_t, L0=dt)

        coherence_lengths[ray_id] = {
            "z0": vc["z0"],
            "z0_std": float(np.sqrt(vc["z0_var"])),
            "A0": float(vc["A0"]),
            "A0_std": float(np.sqrt(vc["A0_var"])),
            "t0": t0,
        }

    return coherence_lengths


def compute_theoretical_coherence_lengths(fc: float = 250, zeta0: float = 7.3):
    r"""
    Compute the GM path integral vertical coherence length $z_0$ and coherence time $t_0$ for every ray ID
    and eigenray pair in `EIG_INDEXS`.

    Parameters
    ----------
    fc : float
        acoustic center frequency [Hz]
    zeta0 : float
        reference displacement of GM spectrum [m], Default is 7.3 m (1GM energy level)

    Returns
    -------
    coherence_lengths : dict
        for each ray ID, a list (one entry per eigenray pair) of dicts with `z0` [m] and `t0` [s].
        Ray IDs whose ray variables could not be computed are omitted.
    """
    try:
        rvs = compute_n23_ray_variables(fc=fc, zeta0=zeta0)
    except Exception:
        # one failing ray fails the whole call, so compute them one at a time
        with (get_data_dir() / "dataproducts" / "ray_run_forward_back.pkl").open(
            "rb"
        ) as f:
            ray_run = pickle.load(f)

        rvs = {}
        for ray_id in EIG_INDEXS:
            try:
                rvs.update(
                    compute_n23_ray_variables(
                        fc=fc, zeta0=zeta0, ray_ids=[ray_id], ray_run=ray_run
                    )
                )
            except Exception as e:
                warnings.warn(
                    f"could not compute ray variables for {ray_id}: {e}", stacklevel=2
                )

    coherence_lengths = {}
    for ray_id, rv_pairs in rvs.items():
        coherence_lengths[ray_id] = [
            {
                "z0": float(compute_vertical_coherence_length(rv, fc)[0]),
                "t0": float(compute_time_coherence_length(rv, fc)[0]),
            }
            for rv in rv_pairs
        ]

    return coherence_lengths


# rows of the coherence lengths table: each observed (ducted) ray and the predicted ray IDs it is compared to.
# The +7 observation includes the unresolved -8 arrival, as in the Ducted_VC figure.
COHERENCE_TABLE_ROWS = {"+7": ["-8", "+7"], "-7": ["-7"], "+6": ["+6"], "-6": ["-6"]}


def coherence_lengths_latex_table(
    observed: dict, theory: dict, rows: dict | None = None
) -> str:
    r"""
    Format observed and theoretical vertical coherence lengths as a LaTeX table.

    Each observed ray spans (``\multirow``) the rows of the predictions it is compared to: one row per
    predicted ray ID and eigenray pair. A ray ID with several eigenray pairs is labelled with a subscript
    per path (``$-6_1$``, ``$-6_2$``, ...). The table needs ``\usepackage{multirow}``.

    Parameters
    ----------
    observed : dict
        output of `estimate_observed_coherence_lengths`
    theory : dict
        output of `compute_theoretical_coherence_lengths`
    rows : dict, optional
        observed ray ID -> list of predicted ray IDs, in table order. Default is `COHERENCE_TABLE_ROWS`.

    Returns
    -------
    table : str
        LaTeX table with columns Ray ID, $z_0$ obs [m], $z_0$ pred [m], $A_0$ obs. Missing values are `--`.
    """
    if rows is None:
        rows = COHERENCE_TABLE_ROWS

    def fmt(value):
        if value is None or not np.isfinite(value):
            return "--"
        return f"{value:.1f}"

    def fmt_std(value, std):
        if std is None or not np.isfinite(std) or fmt(value) == "--":
            return fmt(value)
        return f"{value:.1f} $\\pm$ {std:.1f}"

    def fmt_sci(value, std):
        if value is None or not np.isfinite(value):
            return "--"
        exponent = int(np.floor(np.log10(abs(value)))) if value != 0 else 0
        mantissa = value / 10**exponent
        if std is None or not np.isfinite(std):
            return f"${mantissa:.2f} \\times 10^{{{exponent}}}$"
        return f"$({mantissa:.2f} \\pm {std / 10**exponent:.2f}) \\times 10^{{{exponent}}}$"

    def multirow(n, text):
        return text if n == 1 else f"\\multirow{{{n}}}{{*}}{{{text}}}"

    body = []
    for obs_id, pred_ids in rows.items():
        obs = observed.get(obs_id, {})
        preds = []
        for pred_id in pred_ids:
            pairs = theory.get(pred_id, [{}])
            for n, gm in enumerate(pairs):
                label = f"{pred_id}_{n + 1}" if len(pairs) > 1 else pred_id
                preds.append((label, gm))

        z0_obs = multirow(len(preds), fmt_std(obs.get("z0"), obs.get("z0_std")))
        A0_obs = multirow(len(preds), fmt_sci(obs.get("A0"), obs.get("A0_std")))
        for n, (label, gm) in enumerate(preds):
            first = n == 0
            body.append(
                f"${label}$ & {z0_obs if first else ''} & {fmt(gm.get('z0'))} & "
                f"{A0_obs if first else ''} \\\\"
            )

    lines = [
        r"\begin{table}",
        r"\centering",
        r"\begin{tabular}{lccc}",
        r"\hline\hline",
        r"Ray ID & $z_0$ obs [m] & $z_0$ pred [m] & $A_0$ obs \\",
        r"\hline",
        *body,
        r"\hline\hline",
        r"\end{tabular}",
        r"\caption{Observed and predicted (1GM path integral) vertical coherence length $z_0$, and the "
        r"observed coherence amplitude $A_0$. Subscripts denote separate eigenray paths.}",
        r"\label{tab:coherence_lengths}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def structure_function_from_coherence(vc: dict, max_lag: float = 126.0):
    r"""
    Observed wave structure function $D(\Delta z) = -2 \ln \mathrm{VC}(\Delta z)$ of one ray, with
    leave-one-transmission-out jackknife variances.

    VC is the vertical coherence $\hat{C} / A_0$ of `jackknife_vertical_coherence`, and each jackknife sample
    is $-2 \ln (\hat{C}_k / A_{0,k})$, so the variance includes the uncertainty of $A_0$. Since
    $\langle p_1 p_2^* \rangle = \exp(-D/2)$, D is the structure function of the whole wave, $D_\phi + D_\chi$ in
    WFT, which is what the path integral `compute_phase_structure_function` predicts.

    Only the lags $0 < \Delta z \le$ `max_lag` where VC and every jackknife sample are positive are kept. The
    default 126 m is half the 28-element aperture: beyond it few element pairs are left and VC falls toward
    zero, where the logarithm is unstable.

    Parameters
    ----------
    vc : dict
        output of `jackknife_vertical_coherence`
    max_lag : float
        largest lag [m]

    Returns
    -------
    sf : dict
        - lags : lags [m]
        - D : structure function [rad$^2$] at `lags`
        - D_var : jackknife variance of D
        - D_jk : leave-one-out estimates, of shape (transmission, lag)
    """
    lags = np.asarray(vc["lags"], dtype=float)
    ratio_jk = vc["MC_jk"] / vc["A0_jk"][:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        keep = (
            (lags > 0)
            & (lags <= max_lag)
            & (vc["VC"] > 0)
            & np.all(ratio_jk > 0, axis=0)
        )
        D = -2 * np.log(vc["VC"][keep])
        D_jk = -2 * np.log(ratio_jk[:, keep])

    return {"lags": lags[keep], "D": D, "D_var": _jackknife_var(D_jk), "D_jk": D_jk}


def fit_GM_structure_function(sf: dict, model, GM_bounds=(1e-2, 1e2)):
    r"""
    Weighted least squares fit of a GM energy level to an observed structure function.

    The cost is $\sum_k (D_\mathrm{model}(\Delta z_k; \mathrm{GM}) - D_k)^2 / \mathrm{Var}[D_k]$, minimized over
    $\log_{10} \mathrm{GM}$. Lags with zero jackknife variance are left out. The model is evaluated at every
    step, because the GM level also sets the mode cut-off J, so D is only nearly linear in GM.

    The standard error is a leave-one-transmission-out jackknife of the fit linearized at its optimum,
    $\mathrm{GM}_k = \widehat{\mathrm{GM}} + \sum w g (D_k - D) / \sum w g^2$ with
    $g = D_\mathrm{model} / \widehat{\mathrm{GM}}$, which avoids refitting the model 23 times. It is exact
    when D is linear in GM.

    Parameters
    ----------
    sf : dict
        output of `structure_function_from_coherence`
    model : callable
        ``model(GM)`` returns the model structure function [rad$^2$] at ``sf["lags"]``
    GM_bounds : tuple
        bounds of the GM level

    Returns
    -------
    fit : dict
        - GM : fitted GM energy level
        - GM_std : its jackknife standard error
        - cost : the weighted sum of squares at the optimum
        - n_lags : number of lags in the fit
        - D_model : the model at the fitted level, at ``sf["lags"]``
    """
    valid = np.isfinite(sf["D"]) & np.isfinite(sf["D_var"]) & (sf["D_var"] > 0)
    w = 1 / sf["D_var"][valid]

    def cost(log_GM):
        return float(np.sum(w * (model(10**log_GM)[valid] - sf["D"][valid]) ** 2))

    result = optimize.minimize_scalar(
        cost,
        bounds=np.log10(GM_bounds),
        method="bounded",
        options={"xatol": 1e-4},
    )
    GM = float(10**result.x)

    D_model = np.asarray(model(GM), dtype=float)
    g = D_model[valid] / GM
    GM_jk = GM + (sf["D_jk"][:, valid] - sf["D"][valid]) @ (w * g) / np.sum(w * g**2)

    return {
        "GM": GM,
        "GM_std": float(np.sqrt(_jackknife_var(GM_jk))),
        "cost": float(result.fun),
        "n_lags": int(np.sum(valid)),
        "D_model": D_model,
    }


def fit_GM_vertical_coherence(
    wavefront_values: dict,
    ray_ids: list | None = None,
    dz: float = 9.0,
    max_lag: float = 126.0,
    envi_field: xr.Dataset | None = None,
    ray_run: dict | None = None,
):
    r"""
    Fit the GM energy level of the path integral structure function to the observed vertical coherence of
    each ray, the coherence counterpart of the spectral fit `fit_F_phi_chis`.

    The observed $D = -2 \ln \mathrm{VC}$ is from `structure_function_from_coherence`, and the model is
    `compute_phase_structure_function` on the first eigenray pair with $\zeta_0 = 7.3 \sqrt{\mathrm{GM}}$ m, as in `phi_chi_spectrum_model`. The fit is
    `fit_GM_structure_function`.

    Parameters
    ----------
    wavefront_values : dict
        wavefront values for each ray ID, of shape (transmission, hydrophone), from `get_wavefront_values`
    ray_ids : list
        ray IDs to fit. Default is `DUCTED_RAY_IDS`.
    dz : float
        hydrophone spacing [m]
    max_lag : float
        largest lag [m] in the fit, see `structure_function_from_coherence`
    envi_field : xr.Dataset, optional
        environment passed to `compute_n23_ray_variables`. Default is `fctd_environment()`, computed once.
    ray_run : dict, optional
        ray run passed to `compute_n23_ray_variables`. Default is the cached ``ray_run_forward_back.pkl``,
        loaded once.

    Returns
    -------
    fits : dict
        for each ray ID, the output of `fit_GM_structure_function` plus the observed ``lags``, ``D`` and
        ``D_var``
    """
    if ray_ids is None:
        ray_ids = DUCTED_RAY_IDS
    if envi_field is None:
        envi_field = fctd_environment()
    if ray_run is None:
        with (get_data_dir() / "dataproducts" / "ray_run_forward_back.pkl").open(
            "rb"
        ) as f:
            ray_run = pickle.load(f)

    fits = {}
    for ray_id in ray_ids:
        sf = structure_function_from_coherence(
            jackknife_vertical_coherence(wavefront_values[ray_id], dz), max_lag
        )

        def model(GM, ray_id=ray_id, lags=sf["lags"]):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rv = compute_n23_ray_variables(
                    zeta0=7.3 * np.sqrt(GM),
                    ray_ids=[ray_id],
                    envi_field=envi_field,
                    ray_run=ray_run,
                )[ray_id][0]
                return compute_phase_structure_function(rv, lags)[0]

        fit = fit_GM_structure_function(sf, model)
        fits[ray_id] = {**fit, "lags": sf["lags"], "D": sf["D"], "D_var": sf["D_var"]}

    return fits


def GM_levels_latex_table(spectral: dict | None, coherence: dict | None) -> str:
    r"""
    Format the GM energy levels fitted to the $F_\phi$, $F_\chi$ spectra (WFT) and to the vertical coherence
    (path integral) as a LaTeX (booktabs) table.

    Parameters
    ----------
    spectral : dict or None
        output of `jackknife_F_phi_chi_fit`, or None if the fit has not been made
    coherence : dict or None
        output of `fit_GM_vertical_coherence`, or None

    Returns
    -------
    table : str
        LaTeX table. Levels are reported $\pm$ their jackknife standard error. Missing values are `--`.
    """

    def fmt(fits, ray_id):
        if fits is None or ray_id not in fits or not np.isfinite(fits[ray_id]["GM"]):
            return "--"
        return f"{fits[ray_id]['GM']:.2f} $\\pm$ {fits[ray_id]['GM_std']:.2f}"

    ray_ids = list(DUCTED_RAY_IDS)
    for fits in (spectral, coherence):
        ray_ids += [r for r in (fits or {}) if r not in ray_ids]

    rows = [
        f"${ray_id}$ & {fmt(spectral, ray_id)} & {fmt(coherence, ray_id)} \\\\"
        for ray_id in ray_ids
    ]

    lines = [
        r"\begin{table}",
        r"\centering",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Ray ID & $F_\phi$, $F_\chi$ (WFT) & VC (path integral) \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        (
            r"\caption{GM energy levels fitted to the observed phase and log-amplitude spectra (WFT, Colosi"
            r" 2016 Eq.~6.52, local straight-ray) and to the vertical coherence ($D = -2 \ln \mathrm{VC}$,"
            r" path integral, Eq.~7.50, with the ray-tube separation $\xi_1$), $\pm$ their"
            r" leave-one-transmission-out jackknife standard errors.}"
        ),
        r"\label{tab:GM_levels}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def run_rays(
    num_rays: int | None = None,
    source_depth: float | None = None,
    receiver_range: float | None = None,
    receiver_depths: list | None = None,
    envi_field: xr.Dataset | None = None,
    bathymetry: xr.DataArray | None = None,
):
    """
    Solve for the ray fan and eigen rays for the forward and backward propagation environment.

    Default values for source/receiver geometry are from source localization using stitched environment.
    TODO
    Parameters
    ----------
    envi_field : xarray.Dataset, optional
        environment with sound speed ``c`` on ``(depth, range)``, source at range 0. If None, the shifted
        FCTD of `fctd_environment`.
    bathymetry : xarray.DataArray, optional
        bottom depth [m, positive down] over ``range``. If None, a flat 5000 m bottom over 0-200 km.
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
    if envi_field is None:
        envi_field = fctd_environment()

    c = envi_field["c"]

    if bathymetry is None:
        bathy = xr.DataArray(
            np.linspace(5000, 5000, 100),  # flat bottom
            dims=["range"],
            coords={"range": np.linspace(0, 200e3, 100)},
        )
    else:
        bathy = bathymetry

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


def source_geometry_sweep(
    lower_array_depth,
    source_depths=(2000, 2050, 2100, 2150, 2200, 2250, 2300, 2350, 2400),
    receiver_ranges=(148e3, 149e3, 150e3, 151e3, 152e3, 153e3, 154e3, 155e3),
):
    """
    Sweep the source depth and receiver range, tracing rays through the FCTD environment, to match the observed
    ducted section of the wavefront (exploratory; read by ``panel_apps/source_geometry_sweep.py``).

    Parameters
    ----------
    lower_array_depth : float
        receiver depth [m] of the eigenrays, e.g. the mean actual depth of the lower array
    source_depths, receiver_ranges : sequence of float
        grid of source depths [m] and receiver ranges [m]

    Returns
    -------
    sweep : dict
        ``source_depths``, ``receiver_ranges`` (flattened grid), ``rays``, ``eigs`` and ``envi``, the contents of
        ``ray_geo_sweep.pkl``
    """
    fctd = compute_filtered_fctd_grid(z_smooth=500, r_smooth=10000)
    front_location = 35e3  # m
    fctd = fctd.assign_coords({"range": (fctd.range.to_numpy() + front_location)})

    bathy = xr.DataArray(
        np.linspace(4908, 4908, 100),
        dims=["range"],
        coords={"range": np.linspace(0, 200e3, 100)},
    )
    envi = pr.OceanEnvironment2D(sound_speed=fctd["c"], bathymetry=bathy)

    ssource_depths, rreceiver_ranges = np.meshgrid(
        np.asarray(source_depths, dtype=float), np.asarray(receiver_ranges, dtype=float)
    )
    ssource_depths = ssource_depths.flatten()
    rreceiver_ranges = rreceiver_ranges.flatten()

    rays = []
    eigs = []
    for k, (source_depth, receiver_range) in enumerate(
        zip(ssource_depths, rreceiver_ranges, strict=True)
    ):
        print(
            f"Run {k + 1}/{len(ssource_depths)}, zs = {source_depth}[m], r={receiver_range / 1000} [km]..."
        )
        rays.append(
            pr.shoot_rays(
                source_depth, 0, np.linspace(-35, 35, 1001), receiver_range, 1000, envi
            )
        )
        eigs.append(
            pr.find_eigenrays(
                rays[k],
                [float(lower_array_depth)],
                source_depth,
                0,
                receiver_range,
                1000,
                envi,
            )
        )

    return {
        "source_depths": ssource_depths,
        "receiver_ranges": rreceiver_ranges,
        "rays": rays,
        "eigs": eigs,
        "envi": envi,
    }


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
                .interp(
                    {
                        "depth": np.linspace(800, 1400, 50),
                        "time": dss[0].time.values,
                    }
                )
                .drop_vars(["nominal_depth"])
            )
        )

        dss_upper_interp.append(
            np.abs(
                ds.swap_dims({"hydrophone": "actual_depth"})
                .rename({"actual_depth": "depth"})[:32]
                .interp(
                    {
                        "depth": np.linspace(100, 600, 50),
                        "time": dss[0].time.values,
                    }
                )
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


def _log_periodogram_bias(dof):
    r"""
    Bias of the log10 of a periodogram, $E[\log_{10}(\chi^2_\nu / \nu)] = (\psi(\nu/2) - \ln(\nu/2)) / \ln 10$,
    for a periodogram bin with $\nu$ degrees of freedom: $-\gamma / \ln 10 = -0.2507$ for $\nu = 2$ and
    $-(\gamma + \ln 2) / \ln 10 = -0.5517$ for $\nu = 1$.

    Parameters
    ----------
    dof : float or np.array
        degrees of freedom of each periodogram bin

    Returns
    -------
    bias : float or np.array
        expected value of log10(periodogram / true spectrum)
    """
    dof = np.asarray(dof, dtype=float)
    return (scipy.special.digamma(dof / 2) - np.log(dof / 2)) / np.log(10)


def _periodogram_dof(n_samples: int, n_bins: int):
    """
    Degrees of freedom of the bins of a one-sided periodogram of `n_samples` real samples, with the DC bin
    removed: 2 for every bin except the Nyquist bin (the last one, when `n_samples` is even), which has 1.

    Parameters
    ----------
    n_samples : int
        number of samples the periodogram was computed from
    n_bins : int
        number of bins after removing the DC bin

    Returns
    -------
    dof : np.array
        degrees of freedom of each bin, shape (n_bins,)
    """
    dof = np.full(n_bins, 2.0)
    if n_samples % 2 == 0:
        dof[-1] = 1.0
    return dof


def _one_sided_periodogram(x, fs: float):
    """
    One-sided periodogram along axis 1 of `x`, as a power spectral density at every bin except DC.

    `scipy.signal.periodogram` doubles every bin of the one-sided spectrum except DC and, for an even record,
    the Nyquist bin, so that the bins sum to the variance. The Nyquist bin is then half the power spectral
    density there; it is doubled here so that every bin estimates the same one-sided density.

    Parameters
    ----------
    x : np.array
        real samples, shape (n_records, n_samples)
    fs : float
        sampling frequency

    Returns
    -------
    freqs : np.array
        frequencies of the bins, shape (n_bins,)
    P : np.array
        one-sided power spectral density, shape (n_records, n_bins)
    dof : np.array
        degrees of freedom of each bin, see `_periodogram_dof`
    """
    freqs, P = signal.periodogram(x, fs=fs, axis=1)
    freqs, P = freqs[1:], P[:, 1:]
    n_samples = x.shape[1]
    if n_samples % 2 == 0:
        P[:, -1] *= 2
    return freqs, P, _periodogram_dof(n_samples, len(freqs))


def _jackknife_log_mean(F, dof):
    r"""
    Bias-corrected log-mean of periodograms over transmissions, with its jackknife standard error.

    The log-mean, mean(log10(F)), of periodograms with $\nu$ degrees of freedom is biased low by
    `_log_periodogram_bias` (a factor of $e^{-\gamma} = 0.56$ for $\nu = 2$), which is removed. The
    standard error is the leave-one-transmission-out jackknife (Efron & Tibshirani 1993, Ch. 11), which
    for a mean equals std(log10(F), ddof=1) / sqrt(M). The bias correction is a constant and does not
    change it.

    Parameters
    ----------
    F : np.array
        periodograms, shape (n_transmissions, n_wavenumbers)
    dof : np.array
        degrees of freedom of each wavenumber bin, shape (n_wavenumbers,), see `_periodogram_dof`

    Returns
    -------
    lm : np.array
        bias-corrected log-mean, 10 ** (mean(log10(F)) - bias), shape (n_wavenumbers,)
    ls : np.array
        jackknife standard error of log10(lm) as a multiplicative factor, 10 ** sigma_jk,
        shape (n_wavenumbers,)
    """
    M = F.shape[0]
    log_F = np.log10(F)
    # leave out one transmission at a time
    mean_jk = np.stack([np.mean(np.delete(log_F, j, axis=0), axis=0) for j in range(M)])
    lm = 10 ** (np.mean(log_F, axis=0) - _log_periodogram_bias(dof))
    return lm, 10 ** np.sqrt(_jackknife_var(mean_jk))


def _jackknife_linear_mean(F):
    r"""
    Arithmetic mean of periodograms over transmissions, with the jackknife standard error of its log10.

    The mean of periodograms is an unbiased estimate of the expected periodogram, whatever the distribution
    of each bin, and when the spectrum varies between transmissions it is the mean spectrum, not the
    geometric mean that the log-mean (`_jackknife_log_mean`) estimates. The standard error is the
    leave-one-transmission-out jackknife (Efron & Tibshirani 1993, Ch. 11) of log10 of the mean.

    Parameters
    ----------
    F : np.array
        periodograms, shape (n_transmissions, n_wavenumbers)

    Returns
    -------
    lm : np.array
        mean of the periodograms, shape (n_wavenumbers,)
    ls : np.array
        jackknife standard error of log10(lm) as a multiplicative factor, 10 ** sigma_jk,
        shape (n_wavenumbers,)
    """
    M = F.shape[0]
    # leave out one transmission at a time
    mean_jk = np.stack([np.mean(np.delete(F, j, axis=0), axis=0) for j in range(M)])
    return np.mean(F, axis=0), 10 ** np.sqrt(_jackknife_var(np.log10(mean_jk)))


SPECTRUM_AVERAGES = ("mean", "log_mean")


def _average_periodograms(F, dof, average: str):
    """Average of periodograms over transmissions: ``mean`` (`_jackknife_linear_mean`) or ``log_mean``
    (`_jackknife_log_mean`)."""
    if average == "mean":
        return _jackknife_linear_mean(F)
    if average == "log_mean":
        return _jackknife_log_mean(F, dof)
    msg = f"average must be one of {SPECTRUM_AVERAGES}, got {average!r}"
    raise ValueError(msg)


def _taper_window(n: int, window: str = "hann"):
    """
    Periodic taper of length `n`, one of `SPECTRAL_WINDOWS`, scaled to unit mean square so that it leaves the
    expected periodogram of white noise, and so its total power, unchanged. ``"tukey"`` has
    ``alpha = TUKEY_ALPHA``; ``"rectangular"`` is all ones.
    """
    _check_spectral_window(window)
    if window == "hann":
        taper = signal.windows.hann(n, sym=False)
    elif window == "tukey":
        taper = signal.windows.tukey(n, TUKEY_ALPHA, sym=False)
    else:
        taper = np.ones(n)
    return taper / np.sqrt(np.mean(taper**2))


def _hann_window(n: int):
    """Periodic Hann window of length `n`, scaled to unit mean square (`_taper_window`)."""
    return _taper_window(n, "hann")


def _interpolate_nan_columns(x):
    """Replace the NaN columns of `x` (n_records, n_samples) by linear interpolation along axis 1 between the
    finite columns (constant past the ends)."""
    x = x.copy()
    nan = np.any(np.isnan(x), axis=0)
    if nan.any():
        z = np.arange(x.shape[1])
        for row in x:
            row[nan] = np.interp(z[nan], z[~nan], row[~nan])
    return x


def _first_difference_postcolor(m, dz: float = 9):
    r"""
    Postcoloring factor of a first difference, $1 / |2 \sin(\pi m \Delta z)|^2$, that turns the spectrum of a
    first-differenced record back into the spectrum of the record, at wavenumbers `m` [cycles/m].
    """
    return 1 / (2 * np.sin(np.pi * m * dz)) ** 2


def estimate_F_phi_chi(
    remove_array_tilt: bool = False,
    window: str = "hann",
    remove_pattern: str = "none",
    first_difference: bool = False,
    average: str = "mean",
):
    r"""
    Estimate observed $F_{\phi}$ and $F_{\chi}$ for the acoustic receptions.

    Periodograms are computed for each of the receptions and averaged over transmissions. By default
    (``average="mean"``) the average is the arithmetic mean (`_jackknife_linear_mean`). With
    ``average="log_mean"`` it is the log-mean with its $\chi^2_\nu$ bias removed (`_jackknife_log_mean`),
    which estimates the geometric mean over transmissions. That is lower than the mean when the spectrum varies
    between transmissions, and it is exactly bias-corrected only where every bin is $\chi^2_\nu$ (not at the
    record's lowest and highest bins, nor for the untapered red phase spectrum). The uncertainty of either is
    the leave-one-transmission-out jackknife standard error of its log10.

    Each depth record has its mean removed and is tapered (`_taper_window`) before the periodogram, by default
    with a Hann window. Without the taper the very red phase spectrum leaks into every bin and sets a floor that
    hides the spectrum at high wavenumbers. The window has unit mean square, so white noise keeps its level, but
    variance at scales longer than the array, which shows up as a slope or curvature concentrated at the array
    ends, is down-weighted. The Tukey window (``alpha = TUKEY_ALPHA = 0.3``) is flat over the middle 70% of
    the record and tapers only the ends, so it down-weights the array ends less than Hann, at the cost of
    more leakage.

    With `first_difference`, each record is first-differenced along the array (prewhitened) before the
    optional taper, and the periodogram is postcolored by $1 / |2 \sin(\pi m \Delta z)|^2$
    (`_first_difference_postcolor`). The order is mean removal, first difference, Hann window, periodogram,
    postcoloring, as in `estimation.Estimator` (``prewhiten``) and ``analysis/spectral_estimation_methods``.
    The differenced record is one element shorter (27 of 28), so its wavenumbers are k / (27 x 9 m), with no
    Nyquist bin. Each of the `SPECTRAL_WINDOWS` combines with or without first differencing.

    Elements 4, 5 and 20 carry a constant phase offset, so they are dropped from the phase, unless the phase of
    the hydrophone pattern was removed (`remove_pattern` ``"phase"`` or ``"both"``), which corrects them, and all
    28 elements are used. Without first differencing they are zero-filled after the mean removal. With it they
    are filled by linear interpolation along the array, because a zero inside a red record differences to two
    large steps that postcoloring would put into the lowest bins.

    Parameters
    ----------
    remove_array_tilt : bool
        If True, wavefront values with array tilt removed are used.
    window : str
        taper, one of `SPECTRAL_WINDOWS`: ``"hann"`` (default), ``"tukey"`` (``alpha = TUKEY_ALPHA``) or
        ``"rectangular"`` (the untapered periodogram)
    remove_pattern : str
        hydrophone pattern removed from the wavefront values used (`wavefront_values_path`), one of
        `REMOVE_PATTERN_CHOICES`. Default is ``"none"``.
    first_difference : bool
        If True, first-difference each record before the taper and postcolor the periodogram. Default is False.
    average : str
        average of the periodograms over transmissions, one of `SPECTRUM_AVERAGES`: ``"mean"`` (default) or
        ``"log_mean"``

    Returns
    -------
    m_data : np.array
        array of vertical wavenumbers [cycles/m] at which $F_\\phi$ and $F_\\chi$ are estimated
    F_phi_data : dict
        dictionary containing arrays of phase power spectral density, $F_\\phi$ [rad$^2$ m], one-sided per cycle/m
    F_chi_data : dict
        dictionary containing arrays of log-amplitude power spectral density, $F_\\chi$ [m], one-sided per cycle/m
    F_phi_lm : dict
        dictionary containing arrays of the average (`average`) over transmissions of the phase power spectral
        density, $F_\\phi$ [rad$^2$ m], one-sided per cycle/m
    F_phi_ls : dict
        dictionary containing arrays of the jackknife standard error of log10 of `F_phi_lm`, as a
        multiplicative factor
    F_chi_lm : dict
        dictionary containing arrays of the average (`average`) over transmissions of the log-amplitude power
        spectral density, $F_\\chi$ [m], one-sided per cycle/m
    F_chi_ls : dict
        dictionary containing arrays of the jackknife standard error of log10 of `F_chi_lm`, as a
        multiplicative factor
    """

    _check_spectral_window(window)
    with wavefront_values_path(remove_array_tilt, remove_pattern).open("rb") as f:
        wavefront_values = pickle.load(f)
    phase_removed = remove_pattern in ("phase", "both")

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

        # remove outliers from phase, unless the hydrophone phase pattern already corrected them
        if not phase_removed:
            phase[np.s_[:, np.r_[4:6, 20]]] = np.nan
        phase = phase - np.nanmean(phase, axis=1)[:, None]
        if first_difference:
            phase = _interpolate_nan_columns(phase)
        else:
            phase[np.isnan(phase)] = 0

        lamp = lamp - np.mean(lamp, axis=1)[:, None]

        # prewhiten
        if first_difference:
            phase = np.diff(phase, axis=1)
            lamp = np.diff(lamp, axis=1)

        # taper in depth
        if window != "rectangular":
            taper = _taper_window(phase.shape[1], window)
            phase = phase * taper
            lamp = lamp * taper

        # m_data [1/m]
        # F_phi_data [rad^2 m], F_chi_data [m], one-sided per cycle/m
        _, F_phi_data[ray_id], dof = _one_sided_periodogram(phase, fs=1 / 9)
        m_data, F_chi_data[ray_id], dof = _one_sided_periodogram(lamp, fs=1 / 9)

        # postcolor
        if first_difference:
            F_phi_data[ray_id] = F_phi_data[ray_id] * _first_difference_postcolor(
                m_data
            )
            F_chi_data[ray_id] = F_chi_data[ray_id] * _first_difference_postcolor(
                m_data
            )

        # average over transmissions and its jackknife standard error
        F_chi_lm[ray_id], F_chi_ls[ray_id] = _average_periodograms(
            F_chi_data[ray_id], dof, average
        )
        F_phi_lm[ray_id], F_phi_ls[ray_id] = _average_periodograms(
            F_phi_data[ray_id], dof, average
        )

    return m_data, F_phi_data, F_chi_data, F_phi_lm, F_phi_ls, F_chi_lm, F_chi_ls


DUCTED_RAY_IDS = ["+7", "-7", "+6", "-6"]


def _lag1_autocorrelation(samples):
    """Lag-1 autocorrelation along axis 0 of `samples`, shape (M, K), for each column."""
    x = samples - np.mean(samples, axis=0)
    return np.sum(x[1:] * x[:-1], axis=0) / np.sum(x**2, axis=0)


def transmission_independence_test(samples, n_perm: int = 2000, seed: int = 0):
    r"""
    Test that per-transmission estimates are uncorrelated from one transmission to the next, the assumption
    of the leave-one-transmission-out jackknife.

    The lag-1 autocorrelation $r_1$ of each column is compared to the $\pm 2 / \sqrt{M}$ band expected for
    uncorrelated samples, whose mean is $-1/M$. The one-sided p-value of the mean $r_1$ is from permuting the
    order of the transmissions, with the same permutation for every column so that correlations between
    columns are kept.

    Parameters
    ----------
    samples : np.array
        per-transmission estimates, shape (M, K), with the transmissions (rows) in time order and equally
        spaced
    n_perm : int
        number of permutations
    seed : int
        seed of the permutations

    Returns
    -------
    result : dict
        - r1 : lag-1 autocorrelation of each column, shape (K,)
        - mean_r1 : mean of r1
        - band : 2 / sqrt(M)
        - null_mean : -1 / M, the expected mean of r1 for uncorrelated samples
        - n_outside : number of columns with |r1| > band
        - p : one-sided permutation p-value of mean_r1 (small if the transmissions are positively correlated)
    """
    samples = np.asarray(samples, dtype=float)
    M = samples.shape[0]

    r1 = _lag1_autocorrelation(samples)
    rng = np.random.default_rng(seed)
    null = np.array(
        [
            np.mean(_lag1_autocorrelation(samples[rng.permutation(M)]))
            for _ in range(n_perm)
        ]
    )
    band = 2 / np.sqrt(M)

    return {
        "r1": r1,
        "mean_r1": float(np.mean(r1)),
        "band": float(band),
        "null_mean": -1 / M,
        "n_outside": int(np.sum(np.abs(r1) > band)),
        "p": float(np.mean(null >= np.mean(r1))),
    }


def _independence_test_per_ray(samples_per_ray: dict, **kwargs):
    """Pooled `transmission_independence_test` of the columns of all rays, and the mean r1 of each ray."""
    pooled = transmission_independence_test(
        np.hstack(list(samples_per_ray.values())), **kwargs
    )
    per_ray = {
        ray_id: float(np.mean(_lag1_autocorrelation(samples)))
        for ray_id, samples in samples_per_ray.items()
    }
    return {"pooled": pooled, "per_ray": per_ray}


def independence_test_F_phi_chi(
    remove_array_tilt: bool = False,
    remove_pattern: str = "none",
    window: str = "hann",
    first_difference: bool = False,
    **kwargs,
):
    """
    `transmission_independence_test` of the per-transmission log-periodograms, log10 F_phi and log10 F_chi
    (`estimate_F_phi_chi`), at every wavenumber of the ducted rays.

    Parameters
    ----------
    remove_array_tilt : bool
        If True, wavefront values with array tilt removed are used.
    remove_pattern : str
        hydrophone pattern removed from the wavefront values used, one of `REMOVE_PATTERN_CHOICES`
    window : str
        taper, one of `SPECTRAL_WINDOWS`, see `estimate_F_phi_chi`
    first_difference : bool
        spectral estimation method, see `estimate_F_phi_chi`
    **kwargs
        passed to `transmission_independence_test`

    Returns
    -------
    result : dict
        for "phi" and "chi", a dict with the "pooled" `transmission_independence_test` over the wavenumbers of
        all ducted rays, and the mean r1 of each ray ("per_ray")
    """
    _, F_phi_data, F_chi_data, _, _, _, _ = estimate_F_phi_chi(
        remove_array_tilt=remove_array_tilt,
        remove_pattern=remove_pattern,
        window=window,
        first_difference=first_difference,
    )
    return {
        name: _independence_test_per_ray(
            {ray_id: np.log10(F[ray_id]) for ray_id in DUCTED_RAY_IDS}, **kwargs
        )
        for name, F in [("phi", F_phi_data), ("chi", F_chi_data)]
    }


def independence_test_vertical_coherence(
    wavefront_values: dict, dz: float = 9.0, max_lag: float = 180.0, **kwargs
):
    r"""
    `transmission_independence_test` of the per-transmission lagged products $R_k(\Delta z)$, whose mean over
    transmissions is the mutual coherence of `jackknife_vertical_coherence`, at every lag up to `max_lag` of
    the ducted rays.

    Neighbouring lags are strongly correlated with each other, so `n_outside` is not a count of independent
    tests; report `mean_r1`.

    Parameters
    ----------
    wavefront_values : dict
        wavefront values for each ray ID, of shape (transmission, hydrophone), from `get_wavefront_values`
    dz : float
        hydrophone spacing [m]
    max_lag : float
        largest lag tested [m]. The largest lags have few hydrophone pairs and are excluded.
    **kwargs
        passed to `transmission_independence_test`

    Returns
    -------
    result : dict
        the "pooled" `transmission_independence_test` over the lags of all ducted rays, and the mean r1 of each
        ray ("per_ray")
    """
    samples = {}
    for ray_id in DUCTED_RAY_IDS:
        X = wavefront_values[ray_id]
        R = []
        for k in range(X.shape[0]):
            lags, R_k, _, _, _ = estimate_mutual_coherence(
                X[k : k + 1], dz, omit_zero_lag=False, norm=False
            )
            R.append(R_k)
        samples[ray_id] = np.array(R)[:, lags <= max_lag]
    return _independence_test_per_ray(samples, **kwargs)


def phi_chi_spectrum_model(
    m, GM, phi_noise, chi_noise, ray_id, all_paths=False, fresnel=True
):
    """
    phi_chi_spectrum_model - compute model of F_phi and F_chi, GM internal waves, WFT, and white observational noise

    The wavenumbers are in rad/m but the spectra and noise levels are one-sided densities per cycle/m, the
    units of `estimate_F_phi_chi` (see `compute_F_phi_chi`).

    Parameters
    ----------
    m : np.array
        array of vertical wavenumbers [rad/m] at which to compute $F_\\phi$ and $F_\\chi$
    GM : float
        energy level of Garrett-Munk spectrum, X corresponds to zeta0 = $\\sqrt(X) \\times$ 7.3 [m]
    phi_noise : float
        observational noise level for phase power spectral density, $F_\\phi$ [rad$^2$ m], one-sided per cycle/m
    chi_noise : float
        observational noise level for log-amplitude power spectral density, $F_\\chi$ [m], one-sided per cycle/m
    ray_id : str
        ray id for which to compute the model ('+7', '-7', '+6', '-6')
    all_paths : bool, optional
        whether to compute the model for all paths (default: False)
    fresnel : bool, optional
        whether to include the Fresnel filter, passed to `compute_F_phi_chi` (default: True)
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
                F_phi, F_chi = compute_F_phi_chi(rvs[ray_id][k], m, fresnel=fresnel)
                F_phis.append(F_phi)
                F_chis.append(F_chi)

                F_phis_noise.append(F_phi + phi_noise)
                F_chis_noise.append(F_chi + chi_noise)

            return F_phis_noise, F_chis_noise, F_phis, F_chis
        else:
            F_phi, F_chi = compute_F_phi_chi(rvs[ray_id][0], m, fresnel=fresnel)

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
        array of vertical wavenumbers [cycles/m] at which $F_\\phi$ and $F_\\chi$ are estimated
        computed with `estimate_F_phi_chi`
    F_phi_lm : dict
        dictionary containing arrays of log-mean phase power spectral density, $F_\\phi$ [rad$^2$ m], one-sided per cycle/m
        computed with `estimate_F_phi_chi`
    F_chi_lm : dict
        dictionary containing arrays of log-mean log-amplitude power spectral density, $F_\\chi$ [m], one-sided per cycle/m
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

        # Minimize. GM must stay positive (zeta0 = 7.3 sqrt(GM)) and the noise levels non-negative,
        # otherwise the simplex can step to an undefined model when the optimum GM is below 1.
        results[ray_id] = optimize.minimize(
            residuals,
            initial_params,
            args=(ray_id,),
            method="Nelder-Mead",
            bounds=[(1e-3, None), (0, None), (0, None)],
        )

    return results


def compute_phi_chi_LS_fit(
    remove_array_tilt=True, remove_pattern="none", window="hann", first_difference=False
):
    """
    Fit the GM energy and observational noise levels to the observed ɸ and χ spectra (`estimate_F_phi_chi`,
    `fit_F_phi_chis`) and write the fit to `phi_chi_LS_fit_path` of the same settings.

    Not part of the published analysis (the paper draws the 1x GM model only); this is what the publication
    figures config's ``spectra_model: fit`` reads.

    Returns
    -------
    results : dict
        output of `fit_F_phi_chis`
    """
    m_data, _, _, F_phi_lm, F_phi_ls, F_chi_lm, F_chi_ls = estimate_F_phi_chi(
        remove_array_tilt=remove_array_tilt,
        remove_pattern=remove_pattern,
        window=window,
        first_difference=first_difference,
    )
    results = fit_F_phi_chis(m_data, F_phi_lm, F_chi_lm, F_phi_ls, F_chi_ls)

    path = phi_chi_LS_fit_path(
        remove_array_tilt, remove_pattern, window, first_difference
    )
    with path.open("wb") as f:
        pickle.dump(results, f)
    return results


def jackknife_F_phi_chi_fit(
    results: dict,
    m_data: np.array,
    F_phi_data: dict,
    F_chi_data: dict,
    F_phi_ls: dict,
    F_chi_ls: dict,
    model=None,
    rel_step: float = 1e-2,
):
    r"""
    Leave-one-transmission-out jackknife standard errors of the GM and noise levels of `fit_F_phi_chis`.

    Refitting the model 23 times per ray is slow, since every model evaluation re-traces the rays, so the
    fit is linearized at its optimum $\hat\theta$ (Gauss-Newton). Leaving out transmission k changes the log-mean data by
    $\Delta y_k = \overline{\log_{10} F}_{(k)} - \overline{\log_{10} F}$ (the bias correction is a constant and
    cancels), and the parameters by $\Delta\theta_k = (J^T W J)^{-1} J^T W \Delta y_k$, with
    $J = \partial \log_{10} F_\mathrm{model} / \partial \theta$ and W the fit weights $1 / \log_{10}^2$ of the
    jackknife SE factors, stacked over $F_\phi$ and $F_\chi$. The GM column of J is a central difference of
    the model, and the noise columns are analytic. A parameter on its bound (a noise level of 0, or GM at
    the lower bound 1e-3) is held fixed and gets a NaN standard error.

    Parameters
    ----------
    results : dict
        output of `fit_F_phi_chis`, for each ray ID a result with ``x = [GM, phi_noise, chi_noise]``
    m_data : np.array
        wavenumbers [cycles/m] of the spectra, from `estimate_F_phi_chi`
    F_phi_data, F_chi_data : dict
        per-transmission periodograms, shape (transmission, wavenumber), from `estimate_F_phi_chi`
    F_phi_ls, F_chi_ls : dict
        jackknife SE factors used as the fit weights, from `estimate_F_phi_chi`
    model : callable, optional
        model spectra with the signature of `phi_chi_spectrum_model` (default)
    rel_step : float
        relative step in GM of the central difference

    Returns
    -------
    jackknife : dict
        for each ray ID, a dict with ``GM``, ``GM_std``, ``phi_noise``, ``phi_noise_std``, ``chi_noise``,
        ``chi_noise_std`` and the leave-one-out ``params_jk`` of shape (transmission, 3)
    """
    if model is None:
        model = phi_chi_spectrum_model
    m_radm = m_data * 2 * np.pi
    n = len(m_data)

    jackknife = {}
    for ray_id, result in results.items():
        GM, phi_noise, chi_noise = (float(p) for p in result.x)

        def log_model(GM, ray_id=ray_id, phi_noise=phi_noise, chi_noise=chi_noise):
            F_phi, F_chi, _, _ = model(m_radm, GM, phi_noise, chi_noise, ray_id)
            return np.log10(np.concatenate([F_phi, F_chi]))

        log_F = np.concatenate(
            [np.log10(F_phi_data[ray_id]), np.log10(F_chi_data[ray_id])], axis=1
        )  # (M, 2n)
        M = log_F.shape[0]
        dy = (np.sum(log_F, axis=0) - log_F) / (M - 1) - np.mean(log_F, axis=0)
        sigma = np.concatenate([np.log10(F_phi_ls[ray_id]), np.log10(F_chi_ls[ray_id])])
        w = 1 / sigma**2

        F_model = 10 ** log_model(GM)
        J_GM = (log_model(GM * (1 + rel_step)) - log_model(GM * (1 - rel_step))) / (
            2 * GM * rel_step
        )
        J_phi = np.r_[1 / (np.log(10) * F_model[:n]), np.zeros(n)]
        J_chi = np.r_[np.zeros(n), 1 / (np.log(10) * F_model[n:])]
        J = np.stack([J_GM, J_phi, J_chi], axis=1)

        free = np.array([GM > 1e-3, phi_noise > 0, chi_noise > 0])
        rows = (
            np.all(np.isfinite(J), axis=1)
            & np.isfinite(w)
            & np.all(np.isfinite(dy), axis=0)
        )
        Jf = J[rows][:, free]
        Wdy = w[rows, None] * dy[:, rows].T  # (rows, M)
        d_free = np.linalg.solve(
            Jf.T @ (w[rows, None] * Jf), Jf.T @ Wdy
        ).T  # (M, n_free)

        d_params = np.zeros((M, 3))
        d_params[:, free] = d_free
        params_jk = np.array([GM, phi_noise, chi_noise]) + d_params
        std = np.where(free, np.sqrt(_jackknife_var(params_jk)), np.nan)

        jackknife[ray_id] = {
            "GM": GM,
            "GM_std": float(std[0]),
            "phi_noise": phi_noise,
            "phi_noise_std": float(std[1]),
            "chi_noise": chi_noise,
            "chi_noise_std": float(std[2]),
            "params_jk": params_jk,
        }

    return jackknife
