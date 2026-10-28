"""Experiments on the design decisions in the estimation of $F_\\phi$ and $F_\\chi$ from the wavefront values.

`tools.estimate_F_phi_chi` is one fixed set of choices. Here each choice is an option of an
`Estimator`, and an experiment runs a list of estimators on

- the observed wavefront values, fitting the GM level and the noise levels of each and estimating their
  leave-one-transmission-out jackknife standard errors, and
- synthetic wavefront values drawn from a known GM + white-noise spectrum, so that the bias of each
  estimator in the spectrum and in the fitted parameters can be measured.

The estimation of one ray, for phase and for log-amplitude, is

1. phase is unwrapped over the good elements only (bad elements are NaN), after any common phase has been
   removed as phasors (`circular_common_phase`), so that constant element offsets cannot cause 2π slips;
2. each transmission is detrended along the array (``detrend``: mean or linear fit);
3. a pattern common to transmissions (``common_phase``/``common_log_amplitude``: per ray, pooled over the
   rays, or pooled over all lower-array arrivals except ``common_omit`` and bad QC picks) is subtracted;
4. the bad elements are filled (``bad_element_fill``: zero, i.e. the trend line, or linear interpolation);
5. optionally the records are first-differenced (``prewhiten``) and the spectra postcolored;
6. one-sided spectral densities per cycle/m are computed with the taper(s) of ``window``;
7. the per-transmission spectra are averaged (``average``: bias-corrected log-mean, or arithmetic mean).

The GM model is exactly linear in the GM level (only $\\langle\\mu^2\\rangle \\propto \\zeta_0^2$ depends on it),
so the model is computed once at GM = 1 for each ray and wavenumber grid, and the fit of
`tools.fit_F_phi_chis` (log10 weighted least squares, Nelder-Mead) is repeated cheaply. Unlike
`tools.compute_F_phi_chi`, the upper wavenumber mask (m_high ∝ J ∝ 1/GM) is not moved with the GM
level. It only matters for GM above ~5, where m_high enters the data band.

Use the CLI ("create estimation experiments config", "run estimation experiments") or
`run_experiment(load_config(path))`.
"""

from __future__ import annotations

import itertools
import pickle
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from importlib import resources
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import yaml
from matplotlib.figure import Figure
from nescafe23.processing import PEAK_LOWER_QC
from scipy import optimize, signal

from n23_fluctuations_ragland import tools

DZ = 9.0  # hydrophone spacing of the lower array [m]
WAVEFRONT_SOURCES = ("tilt_removed", "picked_angle")
DETRENDS = ("constant", "linear")
COMMON_MODES = ("none", "per_ray", "all_rays", "all_arrivals")
FILLS = ("zero", "linear")
WINDOWS = ("rectangular", "hann", "tukey", "multitaper")
AVERAGES = ("log_mean", "mean")
FIGURE_FORMATS = ("pdf", "png", "svg")
TEMPLATE_FILE = "estimation_experiments.yaml"


class ConfigError(ValueError):
    """The estimation experiments config is invalid."""


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Estimator:
    """One set of design decisions for the estimation of F_phi and F_chi. The defaults are
    `tools.estimate_F_phi_chi(remove_array_tilt=True)`."""

    name: str = "current"
    wavefronts: str = "tilt_removed"
    bad_elements: tuple[int, ...] = (4, 5, 20)
    bad_element_fill: str = "zero"
    mask_bad_elements_log_amplitude: bool = False
    detrend: str = "constant"
    common_phase: str = "none"
    common_log_amplitude: str = "none"
    common_omit: tuple[str, ...] = tools.ARRAY_TILT_OMIT
    common_qc_only: bool = True
    correct_common_removal: bool = True
    prewhiten: bool = False
    window: str = "hann"
    tukey_alpha: float = 0.5
    multitaper_nw: float = 2.0
    multitaper_k: int = 3
    average: str = "log_mean"

    def __post_init__(self):
        choices = {
            "wavefronts": WAVEFRONT_SOURCES,
            "bad_element_fill": FILLS,
            "detrend": DETRENDS,
            "common_phase": COMMON_MODES,
            "common_log_amplitude": COMMON_MODES,
            "window": WINDOWS,
            "average": AVERAGES,
        }
        for key, allowed in choices.items():
            if getattr(self, key) not in allowed:
                msg = f"{self.name}.{key}: expected one of {allowed}, got {getattr(self, key)!r}"
                raise ConfigError(msg)
        if not 0 <= self.tukey_alpha <= 1:
            msg = f"{self.name}.tukey_alpha: expected a number in [0, 1], got {self.tukey_alpha}"
            raise ConfigError(msg)
        if self.multitaper_k < 1 or self.multitaper_k > 2 * self.multitaper_nw:
            msg = (
                f"{self.name}.multitaper_k: expected 1 <= K <= 2 NW = {2 * self.multitaper_nw}, "
                f"got {self.multitaper_k}"
            )
            raise ConfigError(msg)

    @classmethod
    def option_names(cls) -> tuple[str, ...]:
        return tuple(f.name for f in fields(cls) if f.name != "name")

    def with_options(self, options: Mapping[str, Any], path: str) -> Estimator:
        """A copy with `options` (from the config) replacing the defaults."""
        _check_keys(options, ("name", *self.option_names()), path)
        typed = {}
        for key, value in options.items():
            default = getattr(self, key)
            if key in ("bad_elements", "common_omit"):
                if not isinstance(value, Sequence) or isinstance(value, str):
                    msg = f"{path}.{key}: expected a list, got {value!r}"
                    raise ConfigError(msg)
                typed[key] = tuple(
                    int(v) if key == "bad_elements" else str(v) for v in value
                )
            elif isinstance(default, bool):
                if not isinstance(value, bool):
                    msg = f"{path}.{key}: expected true or false, got {value!r}"
                    raise ConfigError(msg)
                typed[key] = value
            elif isinstance(default, int | float):
                if isinstance(value, bool) or not isinstance(value, int | float):
                    msg = f"{path}.{key}: expected a number, got {value!r}"
                    raise ConfigError(msg)
                typed[key] = type(default)(value)
            else:
                typed[key] = str(value)
        return replace(self, **typed)

    def tapers(self, n: int) -> np.ndarray:
        """Tapers of length `n`, shape (K, n), each with unit energy (sum of squares 1)."""
        if self.window == "rectangular":
            tapers = np.ones((1, n))
        elif self.window == "hann":
            tapers = signal.windows.hann(n, sym=False)[None, :]
        elif self.window == "tukey":
            tapers = signal.windows.tukey(n, self.tukey_alpha, sym=False)[None, :]
        else:
            tapers = np.atleast_2d(
                signal.windows.dpss(n, self.multitaper_nw, Kmax=self.multitaper_k)
            )
        return tapers / np.sqrt(np.sum(tapers**2, axis=1, keepdims=True))


# ---------------------------------------------------------------------------
# Spectral estimation
# ---------------------------------------------------------------------------


@dataclass
class SpectralEstimate:
    """Spectra of one ray from one `Estimator`. Densities are one-sided per cycle/m."""

    m: np.ndarray  # wavenumbers [cycles/m]
    phi_lm: np.ndarray  # average F_phi [rad^2 m]
    phi_ls: np.ndarray  # jackknife standard error of log10(phi_lm), as a factor
    chi_lm: np.ndarray  # average F_chi [m]
    chi_ls: np.ndarray
    phi_S: np.ndarray  # per-transmission F_phi, (M, n_bins)
    chi_S: np.ndarray
    dof: np.ndarray
    tx: np.ndarray  # transmission index of each row


def load_wavefronts(source: str) -> dict[str, np.ndarray]:
    """Wavefront values (`get_wavefront_values`) for ``tilt_removed`` or ``picked_angle``."""
    with tools.wavefront_values_path(source == "tilt_removed").open("rb") as f:
        return pickle.load(f)


def _bad_mask(n: int, bad_elements: Sequence[int]) -> np.ndarray:
    bad = np.zeros(n, dtype=bool)
    bad[list(bad_elements)] = True
    return bad


def wavefront_records(X: np.ndarray, est: Estimator):
    """
    Phase [rad] and log-amplitude [Np] of the wavefront values of one ray, before detrending.

    Phase is unwrapped along the array over the good elements only, so a bad element cannot put a 2π
    step into the rest of the record. Bad elements are NaN in phase, and in log-amplitude if
    ``mask_bad_elements_log_amplitude``. Transmissions with a missing good element are dropped.

    Returns
    -------
    phase, lamp : np.ndarray
        shape (M, n)
    tx : np.ndarray
        transmission index of each row
    """
    bad = _bad_mask(X.shape[1], est.bad_elements)
    tx = np.flatnonzero(np.all(np.isfinite(X[:, ~bad]), axis=1))
    X = X[tx]

    phase = np.full(X.shape, np.nan)
    phase[:, ~bad] = np.unwrap(np.angle(X[:, ~bad]), axis=1)

    lamp = np.log(np.abs(X))
    if est.mask_bad_elements_log_amplitude:
        lamp[:, bad] = np.nan
    return phase, lamp, tx


def detrend(x: np.ndarray, kind: str) -> np.ndarray:
    """Remove the mean (``constant``) or least-squares line (``linear``) along axis 1, fitted to the columns
    without NaN."""
    z = np.arange(x.shape[1], dtype=float)
    cols = np.all(np.isfinite(x), axis=0)
    V = np.vander(z, 1 if kind == "constant" else 2)
    coef, *_ = np.linalg.lstsq(V[cols], x[:, cols].T, rcond=None)
    return x - (V @ coef).T


def common_pattern_rows(
    wavefronts: Mapping[str, np.ndarray],
    est: Estimator,
    ray_ids: Sequence[str],
    mode: str,
    drop_tx: int | None = None,
) -> dict[str, np.ndarray]:
    """
    The arrivals and transmissions that form the pattern common to the transmissions.

    ``per_ray`` and ``all_rays`` use the rays being estimated. ``all_arrivals`` uses every lower-array arrival
    in `wavefronts` except ``common_omit`` (by default `tools.ARRAY_TILT_OMIT`, the arrivals also left out of
    the array tilt). The offsets are hardware: every arrival kept, ducted or bottom-bounced, from 7° to 29°
    and up- or down-going, sees the same offsets on elements 4, 5 and 20. With ``common_qc_only``, only
    transmissions whose pick has QC flag 1 or 5 (`nescafe23.processing.PEAK_LOWER_QC`) are used, and a short
    QC row counts as unreviewed. Rows with a missing good element and the jackknifed `drop_tx` are always left out.

    Returns
    -------
    rows : dict
        ray_id → boolean mask over the transmissions (rows of `wavefronts[ray_id]`); empty for ``none``
    """
    if mode == "none":
        return {}
    if mode == "all_arrivals":
        rays = [r for r in wavefronts if r not in est.common_omit]
    else:
        rays = list(ray_ids)

    rows = {}
    for ray_id in rays:
        X = wavefronts[ray_id]
        good = ~_bad_mask(X.shape[1], est.bad_elements)
        ok = np.all(np.isfinite(X[:, good]), axis=1)
        if est.common_qc_only:
            qc = np.asarray(PEAK_LOWER_QC.get(ray_id, []))
            flagged_good = np.zeros(len(X), dtype=bool)
            n_qc = min(len(qc), len(X))
            flagged_good[:n_qc] = np.isin(qc[:n_qc], (1, 5))
            ok &= flagged_good
        if drop_tx is not None and drop_tx < len(ok):
            ok[drop_tx] = False
        rows[ray_id] = ok
    return rows


def common_patterns(
    records: Mapping[str, np.ndarray],
    tx: Mapping[str, np.ndarray],
    rows: Mapping[str, np.ndarray],
    mode: str,
    ray_ids: Sequence[str],
):
    """
    Pattern along the array common to the transmissions: the mean of the detrended records over the
    transmissions of each ray (``per_ray``), or over the transmissions and the rays (``all_rays``) or
    arrivals (``all_arrivals``) of `rows` (`common_pattern_rows`).

    Parameters
    ----------
    records : dict
        ray_id → detrended records (M, n)
    tx : dict
        ray_id → transmission index of each record
    rows : dict
        ray_id → transmissions that form the pattern
    mode : str
    ray_ids : list of str
        rays the patterns are returned for

    Returns
    -------
    patterns : dict
        ray_id → pattern (n,), or None for ``none``
    n_mean : dict
        ray_id → number of records in the mean
    """
    if mode == "none":
        return dict.fromkeys(ray_ids), dict.fromkeys(ray_ids, np.inf)
    used = {r: records[r][rows[r][tx[r]]] for r in rows}
    if mode == "per_ray":
        return (
            {r: np.mean(used[r], axis=0) for r in ray_ids},
            {r: len(used[r]) for r in ray_ids},
        )
    pooled = np.concatenate(list(used.values()), axis=0)
    return dict.fromkeys(ray_ids, np.mean(pooled, axis=0)), dict.fromkeys(
        ray_ids, pooled.shape[0]
    )


def fill_bad_elements(x: np.ndarray, how: str) -> np.ndarray:
    """Replace NaN columns by zero (the trend line, after detrending) or by linear interpolation along
    the array (constant past the ends)."""
    x = x.copy()
    nan = ~np.all(np.isfinite(x), axis=0)
    if not nan.any():
        return x
    if how == "zero":
        x[:, nan] = 0
        return x
    z = np.arange(x.shape[1])
    for row in x:
        row[nan] = np.interp(z[nan], z[~nan], row[~nan])
    return x


def tapered_spectra(x: np.ndarray, tapers: np.ndarray, dz: float = DZ):
    """
    One-sided spectral density per cycle/m of each row of `x`, averaged over unit-energy tapers, at every
    bin except DC. With one rectangular taper this is `tools._one_sided_periodogram`: every bin, including
    the Nyquist bin, estimates the same one-sided density.

    Returns
    -------
    m : np.ndarray
        wavenumbers [cycles/m], shape (n_bins,)
    S : np.ndarray
        spectral densities, shape (n_rows, n_bins)
    dof : np.ndarray
        degrees of freedom of each bin, 2K (K at the Nyquist bin of an even record)
    """
    n = x.shape[1]
    X = np.fft.rfft(x[:, None, :] * tapers[None, :, :], axis=-1)
    S = 2 * dz * np.mean(np.abs(X) ** 2, axis=1)[:, 1:]
    m = np.fft.rfftfreq(n, dz)[1:]
    dof = np.full(len(m), 2.0 * tapers.shape[0])
    if n % 2 == 0:
        dof[-1] /= 2
    return m, S, dof


def average_spectra(S: np.ndarray, dof: np.ndarray, how: str):
    """
    Average of per-transmission spectra over transmissions, with the leave-one-out jackknife standard
    error of its log10 as a multiplicative factor. ``log_mean`` is `tools._jackknife_log_mean`
    (bias-corrected), ``mean`` the arithmetic mean.
    """
    if how == "log_mean":
        return tools._jackknife_log_mean(S, dof)
    M = S.shape[0]
    mean_jk = np.stack([np.mean(np.delete(S, j, axis=0), axis=0) for j in range(M)])
    return np.mean(S, axis=0), 10 ** np.sqrt(tools._jackknife_var(np.log10(mean_jk)))


def _spectra_of(x: np.ndarray, est: Estimator):
    if est.prewhiten:
        x = np.diff(x, axis=1)
    m, S, dof = tapered_spectra(x, est.tapers(x.shape[1]))
    if est.prewhiten:
        S = S / (2 * np.sin(np.pi * m * DZ)) ** 2
    return m, S, dof


def estimate_spectra(
    wavefronts: Mapping[str, np.ndarray],
    est: Estimator,
    ray_ids: Sequence[str],
    drop_tx: int | None = None,
    return_patterns: bool = False,
):
    """
    Estimate F_phi and F_chi of each ray with `est`.

    Parameters
    ----------
    wavefronts : dict
        ray_id → complex wavefront values (M, n). For ``all_arrivals`` it must hold the other arrivals too.
    est : Estimator
    ray_ids : list of str
        rays to estimate
    drop_tx : int, optional
        transmission left out (for the jackknife of the fits), dropped before the common pattern is formed
    return_patterns : bool
        also return the common phase and log-amplitude patterns

    Returns
    -------
    estimates : dict
        ray_id → `SpectralEstimate`
    """
    phase_rows = common_pattern_rows(
        wavefronts, est, ray_ids, est.common_phase, drop_tx
    )
    lamp_rows = common_pattern_rows(
        wavefronts, est, ray_ids, est.common_log_amplitude, drop_tx
    )
    prealigned = circular_common_phase(wavefronts, est, ray_ids, phase_rows)

    phases, lamps, txs = {}, {}, {}
    for ray_id in dict.fromkeys([*ray_ids, *phase_rows, *lamp_rows]):
        X = wavefronts[ray_id] * np.exp(-1j * prealigned[ray_id])
        phase, lamp, txs[ray_id] = wavefront_records(X, est)
        phases[ray_id] = detrend(phase, est.detrend)
        lamps[ray_id] = detrend(lamp, est.detrend)

    phase_patterns, phase_n = common_patterns(
        phases, txs, phase_rows, est.common_phase, ray_ids
    )
    lamp_patterns, lamp_n = common_patterns(
        lamps, txs, lamp_rows, est.common_log_amplitude, ray_ids
    )

    estimates = {}
    for ray_id in ray_ids:
        keep = txs[ray_id] != drop_tx
        fields_out = {}
        for key, records, patterns, n_mean in (
            ("phi", phases, phase_patterns, phase_n),
            ("chi", lamps, lamp_patterns, lamp_n),
        ):
            x = records[ray_id][keep]
            if patterns[ray_id] is not None:
                x = x - patterns[ray_id]
            x = fill_bad_elements(x, est.bad_element_fill)
            m, S, dof = _spectra_of(x, est)
            # removing a mean of n records also removes 1/n of the variance of each one
            if est.correct_common_removal and np.isfinite(n_mean[ray_id]):
                S = S * n_mean[ray_id] / (n_mean[ray_id] - 1)
            lm, ls = average_spectra(S, dof, est.average)
            fields_out[key] = (lm, ls, S)
        estimates[ray_id] = SpectralEstimate(
            m=m,
            phi_lm=fields_out["phi"][0],
            phi_ls=fields_out["phi"][1],
            chi_lm=fields_out["chi"][0],
            chi_ls=fields_out["chi"][1],
            phi_S=fields_out["phi"][2],
            chi_S=fields_out["chi"][2],
            dof=dof,
            tx=txs[ray_id][keep],
        )
    if return_patterns:
        # the common phase is what was removed before unwrapping plus the residual mean after it
        phase_total = {
            ray_id: None if p is None else prealigned[ray_id] + p
            for ray_id, p in phase_patterns.items()
        }
        return estimates, {"phase": phase_total, "log_amplitude": lamp_patterns}
    return estimates


def circular_common_phase(
    wavefronts: Mapping[str, np.ndarray],
    est: Estimator,
    ray_ids: Sequence[str],
    rows: Mapping[str, np.ndarray],
):
    """
    Phase pattern common to the transmissions, estimated from the wavefront values before unwrapping, so it
    can be removed first and the unwrapping only has to follow the small residuals.

    A constant phase offset on one element (elements 4, 5 and 20 sit about 1.1 rad off their neighbours on
    every transmission and arrival) can make the raw phase step between neighbours approach π, where
    unwrapping becomes ambiguous. Here the mean step between neighbouring good elements is the angle of the
    sum, over the transmissions `rows` (`common_pattern_rows`) of each ray, or of all of its rays or arrivals,
    of the unit phasor products Z(n) Z*(n-1). It needs no unwrapping. The pattern is its cumulative sum,
    detrended like the records (``detrend``). The arithmetic mean of the unwrapped records (`common_patterns`)
    then removes whatever is left.

    Returns
    -------
    patterns : dict
        ray_id → common phase (n,) [rad] for the rays in `ray_ids` and `rows`, zero at bad elements and for
        ``common_phase: none``
    """
    n = next(iter(wavefronts.values())).shape[1]
    targets = list(dict.fromkeys([*ray_ids, *wavefronts]))
    if est.common_phase == "none":
        return {ray_id: np.zeros(n) for ray_id in targets}

    good = ~_bad_mask(n, est.bad_elements)
    sums = {}
    for ray_id, ok in rows.items():
        Z = wavefronts[ray_id][ok][:, good]
        Z = Z / np.abs(Z)
        sums[ray_id] = np.sum(Z[:, 1:] * np.conj(Z[:, :-1]), axis=0)
    if est.common_phase != "per_ray":
        sums = dict.fromkeys(targets, sum(sums.values()))

    patterns = {}
    for ray_id in targets:
        if ray_id not in sums:
            patterns[ray_id] = np.zeros(n)
            continue
        pattern = np.full(n, np.nan)
        pattern[good] = np.concatenate(([0.0], np.cumsum(np.angle(sums[ray_id]))))
        # detrend against the element positions, skipping the bad elements
        pattern = detrend(pattern[None, :], est.detrend)[0]
        pattern[~good] = 0.0
        patterns[ray_id] = pattern
    return patterns


# ---------------------------------------------------------------------------
# GM model and fit
# ---------------------------------------------------------------------------


class GMModel:
    """
    The F_phi and F_chi model of `tools.phi_chi_spectrum_model` at GM = 1 (first eigenray of each ray),
    cached per wavenumber grid. The model at GM level X is X times it.
    """

    def __init__(self, ray_ids: Sequence[str]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rvs = tools.compute_n23_ray_variables(zeta0=7.3, ray_ids=list(ray_ids))
        self.rvs = {ray_id: rvs[ray_id][0] for ray_id in ray_ids}
        self._cache: dict = {}

    def __call__(self, ray_id: str, m: np.ndarray):
        """F_phi, F_chi at GM = 1, one-sided per cycle/m, at wavenumbers `m` [cycles/m]. NaN outside the
        wavenumber band of the ray."""
        key = (ray_id, m.tobytes())
        if key not in self._cache:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._cache[key] = tools.compute_F_phi_chi(
                    self.rvs[ray_id], 2 * np.pi * m[None, :]
                )
        return self._cache[key]


@dataclass
class FitResult:
    GM: float
    phi_noise: float
    chi_noise: float
    success: bool
    cost: float
    GM_se: float = np.nan
    phi_noise_se: float = np.nan
    chi_noise_se: float = np.nan


def fit_gm(F1_phi, F1_chi, phi_lm, phi_ls, chi_lm, chi_ls, x0=(1.0, 1.0, 0.2)):
    """
    Fit GM level and white noise levels by the weighted least squares of `tools.fit_F_phi_chis`: the
    squared log10 misfit weighted by the squared jackknife standard error of log10 of the observed
    spectrum, summed over phase and log-amplitude.
    """
    w_phi = 1 / np.maximum(np.log10(phi_ls) ** 2, 1e-12)
    w_chi = 1 / np.maximum(np.log10(chi_ls) ** 2, 1e-12)
    f_phi, f_chi = np.log10(phi_lm), np.log10(chi_lm)
    use = np.isfinite(F1_phi) & np.isfinite(F1_chi)

    def cost(params):
        GM, phi_noise, chi_noise = params
        r_phi = np.log10(GM * F1_phi[use] + phi_noise) - f_phi[use]
        r_chi = np.log10(GM * F1_chi[use] + chi_noise) - f_chi[use]
        return np.sum(w_phi[use] * r_phi**2) + np.sum(w_chi[use] * r_chi**2)

    res = optimize.minimize(
        cost,
        np.asarray(x0, dtype=float),
        method="Nelder-Mead",
        bounds=[(1e-3, None), (0, None), (0, None)],
    )
    return FitResult(
        GM=float(res.x[0]),
        phi_noise=float(res.x[1]),
        chi_noise=float(res.x[2]),
        success=bool(res.success),
        cost=float(res.fun),
    )


def fit_estimates(estimates: Mapping[str, SpectralEstimate], model: GMModel):
    """`fit_gm` of each ray."""
    fits = {}
    for ray_id, e in estimates.items():
        F1_phi, F1_chi = model(ray_id, e.m)
        fits[ray_id] = fit_gm(F1_phi, F1_chi, e.phi_lm, e.phi_ls, e.chi_lm, e.chi_ls)
    return fits


def jackknife_fits(wavefronts, est: Estimator, ray_ids, model: GMModel, fits):
    """Add leave-one-transmission-out jackknife standard errors to `fits`, repeating the whole estimation
    (including the common pattern) without each transmission."""
    n_tx = max(len(w) for w in wavefronts.values())
    samples = {ray_id: [] for ray_id in ray_ids}
    for j in range(n_tx):
        for ray_id, fit in fit_estimates(
            estimate_spectra(wavefronts, est, ray_ids, drop_tx=j), model
        ).items():
            samples[ray_id].append((fit.GM, fit.phi_noise, fit.chi_noise))
    for ray_id in ray_ids:
        se = np.sqrt(tools._jackknife_var(np.array(samples[ray_id])))
        fits[ray_id].GM_se, fits[ray_id].phi_noise_se, fits[ray_id].chi_noise_se = se
    return fits


# ---------------------------------------------------------------------------
# Synthetic wavefront values
# ---------------------------------------------------------------------------


def synthesize_records(S, df, n_records, n, rng):
    """
    Real Gaussian records of length `n` with one-sided spectral density `S` at wavenumbers ``k df``,
    k = 1 … len(S), cut at a random offset from a periodic record of length 2 len(S). The long record
    carries the variance at scales longer than `n`, which is what leaks into the short-record estimates.

    Returns
    -------
    x : np.ndarray
        shape (n_records, n)
    """
    L = 2 * len(S)
    coef = np.zeros((n_records, len(S) + 1), dtype=complex)
    # each wavenumber contributes S df to the variance; the Nyquist term is real and not doubled by irfft
    scale = np.sqrt(np.asarray(S) * df / 4)
    coef[:, 1:] = scale * (
        rng.standard_normal((n_records, len(S)))
        + 1j * rng.standard_normal((n_records, len(S)))
    )
    coef[:, -1] = 2 * coef[:, -1].real
    x = np.fft.irfft(coef, n=L, axis=1) * L
    start = rng.integers(0, L, n_records)
    idx = (start[:, None] + np.arange(n)[None, :]) % L
    return np.take_along_axis(x, idx, axis=1)


@dataclass(frozen=True)
class SyntheticConfig:
    """The ``synthetic`` section of the config."""

    enabled: bool = True
    n_realizations: int = 200
    seed: int = 0
    GM: float | None = None
    phi_noise: float | None = None
    chi_noise: float | None = None
    record_length: int = 4096
    low_wavenumber: str = "zero"
    common_pattern: str = "none"
    angle_jitter_deg: float = 0.0
    fc: float = 250.0
    c0: float = 1500.0


def synthetic_truth(model: GMModel, ray_id, params, m):
    """True F_phi, F_chi of the synthetic wavefronts at wavenumbers `m` [cycles/m]: GM model plus white
    noise."""
    F1_phi, F1_chi = model(ray_id, m)
    return params["GM"] * F1_phi + params["phi_noise"], params["GM"] * F1_chi + params[
        "chi_noise"
    ]


def synthesize_wavefronts(
    model: GMModel,
    params: Mapping[str, Mapping[str, float]],
    cfg: SyntheticConfig,
    rng,
    n_tx: int = 23,
    n: int = 28,
    patterns: Mapping[str, Mapping[str, np.ndarray]] | None = None,
):
    """
    Complex wavefront values of each ray: phase and log-amplitude are independent Gaussian records with the
    true spectra (`synthetic_truth`), plus a random phase constant per transmission, an optional random
    plane-wave angle error per transmission, and an optional fixed pattern along the array.

    Below the lowest wavenumber of the GM band (mode 1, where the model is NaN) the spectra are zero
    (``low_wavenumber: zero``) or continue at their value at the band edge (``flat``).
    """
    n_long = cfg.record_length // 2
    df = 1 / (cfg.record_length * DZ)
    m_long = df * np.arange(1, n_long + 1)
    z = (np.arange(n) - (n - 1) / 2) * DZ
    k0 = 2 * np.pi * cfg.fc / cfg.c0

    wavefronts = {}
    for ray_id, p in params.items():
        S_phi, S_chi = synthetic_truth(model, ray_id, p, m_long)
        below = np.isnan(S_phi)
        if cfg.low_wavenumber == "flat":
            edge = np.argmax(~below)
            S_phi[below], S_chi[below] = S_phi[edge], S_chi[edge]
        else:
            S_phi[below], S_chi[below] = 0.0, 0.0

        phase = synthesize_records(S_phi, df, n_tx, n, rng)
        lamp = synthesize_records(S_chi, df, n_tx, n, rng)
        phase += rng.uniform(-np.pi, np.pi, (n_tx, 1))
        if cfg.angle_jitter_deg > 0:
            dtheta = np.radians(cfg.angle_jitter_deg) * rng.standard_normal((n_tx, 1))
            phase += k0 * z[None, :] * dtheta
        if patterns is not None:
            phase += np.nan_to_num(patterns["phase"][ray_id])
            lamp += np.nan_to_num(patterns["log_amplitude"][ray_id])
        wavefronts[ray_id] = np.exp(lamp + 1j * phase)
    return wavefronts


def run_synthetic(
    estimators: Sequence[Estimator],
    model: GMModel,
    params,
    cfg: SyntheticConfig,
    ray_ids,
    patterns=None,
    progress=None,
):
    """
    Monte Carlo of every estimator on the same synthetic experiments (23 transmissions each).

    Returns
    -------
    result : dict
        ``bias[name][ray_id]``: (n_realizations, n_bins) log10(estimate / truth) for ``phi`` and ``chi``;
        ``m[name]``; ``fits[name][ray_id]``: (n_realizations, 3) fitted GM, phi_noise, chi_noise;
        ``params``.
    """
    rng = np.random.default_rng(cfg.seed)
    bias = {e.name: {r: {"phi": [], "chi": []} for r in ray_ids} for e in estimators}
    fits = {e.name: {r: [] for r in ray_ids} for e in estimators}
    m_of = {}
    for k in range(cfg.n_realizations):
        if progress is not None:
            progress(k)
        wavefronts = synthesize_wavefronts(model, params, cfg, rng, patterns=patterns)
        for est in estimators:
            estimates = estimate_spectra(wavefronts, est, ray_ids)
            for ray_id, e in estimates.items():
                m_of[est.name] = e.m
                t_phi, t_chi = synthetic_truth(model, ray_id, params[ray_id], e.m)
                bias[est.name][ray_id]["phi"].append(np.log10(e.phi_lm / t_phi))
                bias[est.name][ray_id]["chi"].append(np.log10(e.chi_lm / t_chi))
                F1_phi, F1_chi = model(ray_id, e.m)
                fit = fit_gm(F1_phi, F1_chi, e.phi_lm, e.phi_ls, e.chi_lm, e.chi_ls)
                fits[est.name][ray_id].append((fit.GM, fit.phi_noise, fit.chi_noise))
    for name in bias:
        for ray_id in ray_ids:
            for key in ("phi", "chi"):
                bias[name][ray_id][key] = np.array(bias[name][ray_id][key])
            fits[name][ray_id] = np.array(fits[name][ray_id])
    return {"bias": bias, "m": m_of, "fits": fits, "params": params}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _require_mapping(data: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        msg = f"{path}: expected a mapping, got {type(data).__name__}"
        raise ConfigError(msg)
    return data


def _check_keys(data: Mapping[str, Any], allowed: Sequence[str], path: str) -> None:
    unknown = sorted(set(data) - set(allowed))
    if unknown:
        msg = f"{path}: unknown key(s) {unknown}; allowed: {sorted(allowed)}"
        raise ConfigError(msg)


@dataclass(frozen=True)
class ExperimentConfig:
    """Contents of an estimation experiments ``config.yaml``."""

    output_dir: Path = Path("estimation_experiments")
    formats: tuple[str, ...] = ("pdf",)
    usetex: bool = False
    ray_ids: tuple[str, ...] = tuple(tools.DUCTED_RAY_IDS)
    base: Estimator = field(default_factory=Estimator)
    estimators: tuple[Estimator, ...] = ()
    fit: bool = True
    fit_jackknife: bool = True
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)


def _expand_sweep(base: Estimator, sweep: Mapping[str, Any]) -> list[Estimator]:
    sweep = _require_mapping(sweep, "sweep")
    _check_keys(sweep, Estimator.option_names(), "sweep")
    keys = list(sweep)
    for key in keys:
        if not isinstance(sweep[key], Sequence) or isinstance(sweep[key], str):
            msg = f"sweep.{key}: expected a list of values, got {sweep[key]!r}"
            raise ConfigError(msg)
    estimators = []
    for values in itertools.product(*(sweep[k] for k in keys)):
        options = dict(zip(keys, values, strict=True))
        name = ", ".join(f"{k}={v}" for k, v in options.items())
        estimators.append(base.with_options({"name": name, **options}, "sweep"))
    return estimators


def _synthetic_config(data: Any) -> SyntheticConfig:
    data = _require_mapping(data, "synthetic")
    allowed = tuple(f.name for f in fields(SyntheticConfig))
    _check_keys(data, allowed, "synthetic")
    cfg = replace(SyntheticConfig(), **data)
    if cfg.low_wavenumber not in ("zero", "flat"):
        msg = f"synthetic.low_wavenumber: expected zero or flat, got {cfg.low_wavenumber!r}"
        raise ConfigError(msg)
    if cfg.common_pattern not in ("none", "observed"):
        msg = f"synthetic.common_pattern: expected none or observed, got {cfg.common_pattern!r}"
        raise ConfigError(msg)
    if cfg.record_length < 64 or cfg.record_length % 2:
        msg = f"synthetic.record_length: expected an even number >= 64, got {cfg.record_length}"
        raise ConfigError(msg)
    return cfg


def load_config(path: str | Path) -> ExperimentConfig:
    """Read and validate an estimation experiments config. Relative ``output_dir`` is relative to the
    config file."""
    path = Path(path).expanduser()
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    data = _require_mapping(data, str(path))
    _check_keys(
        data,
        (
            "output_dir",
            "formats",
            "usetex",
            "ray_ids",
            "base",
            "variants",
            "sweep",
            "fit",
            "fit_jackknife",
            "synthetic",
        ),
        str(path),
    )

    base = Estimator().with_options(
        {"name": "base", **_require_mapping(data.get("base") or {}, "base")}, "base"
    )
    estimators = []
    for k, variant in enumerate(data.get("variants") or []):
        variant = _require_mapping(variant, f"variants[{k}]")
        if "name" not in variant:
            msg = f"variants[{k}]: every variant needs a name"
            raise ConfigError(msg)
        estimators.append(base.with_options(variant, f"variants[{k}]"))
    if data.get("sweep"):
        estimators += _expand_sweep(base, data["sweep"])
    if not estimators:
        estimators = [replace(base, name="base")]
    names = [e.name for e in estimators]
    if len(set(names)) != len(names):
        msg = f"variant names must be unique, got {names}"
        raise ConfigError(msg)

    formats = tuple(data.get("formats") or ("pdf",))
    bad = [f for f in formats if f not in FIGURE_FORMATS]
    if bad:
        msg = f"formats: unknown {bad}; choose from {FIGURE_FORMATS}"
        raise ConfigError(msg)

    output_dir = Path(data.get("output_dir") or "estimation_experiments").expanduser()
    if not output_dir.is_absolute():
        output_dir = path.parent / output_dir

    return ExperimentConfig(
        output_dir=output_dir,
        formats=formats,
        usetex=bool(data.get("usetex", False)),
        ray_ids=tuple(data.get("ray_ids") or tools.DUCTED_RAY_IDS),
        base=base,
        estimators=tuple(estimators),
        fit=bool(data.get("fit", True)),
        fit_jackknife=bool(data.get("fit_jackknife", True)),
        synthetic=_synthetic_config(data.get("synthetic") or {}),
    )


def template_path() -> Path:
    return Path(
        str(resources.files("n23_fluctuations_ragland") / "templates" / TEMPLATE_FILE)
    )


def write_config_template(dest: str | Path, *, overwrite: bool = False) -> Path:
    """Copy the commented config template to `dest`."""
    dest = Path(dest).expanduser()
    if dest.exists() and not overwrite:
        msg = f"{dest} already exists"
        raise FileExistsError(msg)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(template_path().read_text())
    return dest


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


@dataclass
class ExperimentResult:
    config: ExperimentConfig
    estimates: dict = field(default_factory=dict)  # name → ray_id → SpectralEstimate
    fits: dict = field(default_factory=dict)  # name → ray_id → FitResult
    patterns: dict = field(
        default_factory=dict
    )  # wavefronts → common patterns of the base detrend
    synthetic: dict | None = None
    written: list[Path] = field(default_factory=list)
    model: GMModel | None = None


def observed_patterns(wavefronts, est: Estimator, ray_ids):
    """
    Common phase and log-amplitude patterns of the observed records after `est`'s unwrapping and detrending:
    per ray (``per_ray``, with standard errors ``per_ray_se``), pooled over the rays (``all_rays``) and over
    the arrivals (``all_arrivals``), and of each arrival used for ``all_arrivals`` (``per_arrival``).
    """
    out = {}
    for mode in ("per_ray", "all_rays", "all_arrivals"):
        probe = replace(est, common_phase=mode, common_log_amplitude=mode)
        _, patterns = estimate_spectra(wavefronts, probe, ray_ids, return_patterns=True)
        out[mode] = patterns
    arrivals = [r for r in wavefronts if r not in est.common_omit]
    probe = replace(est, common_phase="per_ray", common_log_amplitude="per_ray")
    _, out["per_arrival"] = estimate_spectra(
        wavefronts, probe, arrivals, return_patterns=True
    )

    # standard errors of the per-ray patterns, unwrapped after removing the common phase as phasors
    out["per_ray_se"] = {"phase": {}, "log_amplitude": {}}
    rows = common_pattern_rows(wavefronts, probe, ray_ids, "per_ray")
    prealigned = circular_common_phase(wavefronts, probe, ray_ids, rows)
    for ray_id in ray_ids:
        phase, lamp, tx = wavefront_records(
            wavefronts[ray_id] * np.exp(-1j * prealigned[ray_id]), est
        )
        for key, x in (("phase", phase), ("log_amplitude", lamp)):
            x = detrend(x, est.detrend)[rows[ray_id][tx]]
            out["per_ray_se"][key][ray_id] = np.std(x, axis=0, ddof=1) / np.sqrt(len(x))
    return out


def run_experiment(cfg: ExperimentConfig, progress=print) -> ExperimentResult:
    """Run every estimator on the observed wavefront values (and the synthetic ones), save the results,
    tables and figures to ``cfg.output_dir``."""
    result = ExperimentResult(config=cfg)
    ray_ids = list(cfg.ray_ids)
    sources = {e.wavefronts for e in cfg.estimators} | {cfg.base.wavefronts}
    wavefronts = {s: load_wavefronts(s) for s in sources}

    model = GMModel(ray_ids) if cfg.fit or cfg.synthetic.enabled else None
    result.model = model

    for est in cfg.estimators:
        progress(f"estimating: {est.name}")
        result.estimates[est.name] = estimate_spectra(
            wavefronts[est.wavefronts], est, ray_ids
        )
        if cfg.fit:
            fits = fit_estimates(result.estimates[est.name], model)
            if cfg.fit_jackknife:
                fits = jackknife_fits(
                    wavefronts[est.wavefronts], est, ray_ids, model, fits
                )
            result.fits[est.name] = fits

    for source in sources:
        result.patterns[source] = observed_patterns(
            wavefronts[source], replace(cfg.base, wavefronts=source), ray_ids
        )

    if cfg.synthetic.enabled:
        params = _synthetic_params(cfg, model, wavefronts, ray_ids)
        patterns = None
        if cfg.synthetic.common_pattern == "observed":
            patterns = result.patterns[cfg.base.wavefronts]["per_ray"]
        progress(
            f"synthetic: {cfg.synthetic.n_realizations} realizations x {len(cfg.estimators)} estimators"
        )
        result.synthetic = run_synthetic(
            cfg.estimators,
            model,
            params,
            cfg.synthetic,
            ray_ids,
            patterns=patterns,
            progress=lambda k: (
                progress(f"  realization {k + 1}") if (k + 1) % 50 == 0 else None
            ),
        )

    save_results(result)
    result.written += make_figures(result)
    return result


def _synthetic_params(cfg, model, wavefronts, ray_ids):
    """True GM and noise levels of each ray: from the config, or (null) from the fit of the base estimator
    to the observations."""
    s = cfg.synthetic
    base_fits = fit_estimates(
        estimate_spectra(wavefronts[cfg.base.wavefronts], cfg.base, ray_ids), model
    )
    return {
        ray_id: {
            "GM": s.GM if s.GM is not None else base_fits[ray_id].GM,
            "phi_noise": s.phi_noise
            if s.phi_noise is not None
            else base_fits[ray_id].phi_noise,
            "chi_noise": s.chi_noise
            if s.chi_noise is not None
            else base_fits[ray_id].chi_noise,
        }
        for ray_id in ray_ids
    }


def fit_table(result: ExperimentResult) -> pd.DataFrame:
    rows = []
    for name, fits in result.fits.items():
        for ray_id, fit in fits.items():
            rows.append({"estimator": name, "ray_id": ray_id, **fit.__dict__})
    return pd.DataFrame(rows)


def synthetic_table(result: ExperimentResult) -> pd.DataFrame:
    """Median and 16-84 % range over realizations of the fitted / true parameters, and the rms over
    wavenumber of the median spectral bias [decades]."""
    syn = result.synthetic
    rows = []
    for name in syn["fits"]:
        for ray_id, samples in syn["fits"][name].items():
            p = syn["params"][ray_id]
            truth = np.array([p["GM"], p["phi_noise"], p["chi_noise"]])
            ratio = samples / truth
            row = {"estimator": name, "ray_id": ray_id}
            for k, key in enumerate(("GM", "phi_noise", "chi_noise")):
                q16, q50, q84 = np.nanpercentile(ratio[:, k], [16, 50, 84])
                row |= {
                    f"{key}_ratio_median": q50,
                    f"{key}_ratio_p16": q16,
                    f"{key}_ratio_p84": q84,
                }
            for key in ("phi", "chi"):
                med = np.nanmedian(syn["bias"][name][ray_id][key], axis=0)
                row[f"{key}_bias_rms_decades"] = float(np.sqrt(np.nanmean(med**2)))
                row[f"{key}_bias_max_decades"] = float(np.nanmax(np.abs(med)))
            rows.append(row)
    return pd.DataFrame(rows)


def save_results(result: ExperimentResult) -> None:
    out = result.config.output_dir
    out.mkdir(parents=True, exist_ok=True)
    with (out / "results.pkl").open("wb") as f:
        pickle.dump(
            {k: v for k, v in result.__dict__.items() if k not in ("written", "model")},
            f,
        )
    result.written.append(out / "results.pkl")
    if result.fits:
        fit_table(result).to_csv(out / "fit_summary.csv", index=False)
        result.written.append(out / "fit_summary.csv")
    if result.synthetic is not None:
        synthetic_table(result).to_csv(out / "synthetic_summary.csv", index=False)
        result.written.append(out / "synthetic_summary.csv")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _colors(n):
    cmap = plt.get_cmap("tab10" if n <= 10 else "tab20")
    return [cmap(k % cmap.N) for k in range(n)]


def _legend_strip(fig, n_labels):
    """Fraction of the figure height to leave at the bottom for `_legend_below`."""
    rows = -(-n_labels // 3)
    return (0.1 + 0.16 * rows) / fig.get_figheight()


def _legend_below(fig, handles, labels):
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncols=min(len(labels), 3),
        fontsize=7,
        frameon=False,
    )


def plot_spectra(result: ExperimentResult) -> Figure:
    """Observed F_phi and F_chi of every estimator (± jackknife SE for up to 6 estimators), with the fitted
    model (dashed)."""
    cfg = result.config
    ray_ids = list(cfg.ray_ids)
    names = list(result.estimates)
    colors = _colors(len(names))
    model = result.model if result.fits else None

    fig, axes = plt.subplots(
        len(ray_ids), 2, figsize=(7, 2 * len(ray_ids) + 1), sharex=True, squeeze=False
    )
    for i, ray_id in enumerate(ray_ids):
        for name, color in zip(names, colors, strict=True):
            e = result.estimates[name][ray_id]
            for j, (lm, ls) in enumerate(((e.phi_lm, e.phi_ls), (e.chi_lm, e.chi_ls))):
                ax = axes[i, j]
                ax.loglog(e.m, lm, color=color, lw=1.2, label=name)
                if len(names) <= 6:
                    ax.fill_between(
                        e.m, lm / ls, lm * ls, color=color, alpha=0.12, lw=0
                    )
                if model is not None:
                    fit = result.fits[name][ray_id]
                    F1 = model(ray_id, e.m)[j]
                    noise = (fit.phi_noise, fit.chi_noise)[j]
                    ax.loglog(e.m, fit.GM * F1 + noise, color=color, lw=0.8, ls="--")
        axes[i, 0].set_ylabel(f"{ray_id}\n$F_\\phi$ [rad$^2$ m]")
        axes[i, 1].set_ylabel("$F_\\chi$ [m]")
    for ax in axes[-1]:
        ax.set_xlabel("$m$ [cpm]")
    for ax in axes.flat:
        ax.grid(which="both", lw=0.3, alpha=0.5)
    axes[0, 0].set_title("phase", loc="left")
    axes[0, 1].set_title("log-amplitude", loc="left")
    fig.tight_layout(rect=(0, _legend_strip(fig, len(names)), 1, 1))
    _legend_below(fig, *axes[0, 0].get_legend_handles_labels())
    return fig


def plot_fit_summary(result: ExperimentResult) -> Figure:
    """Fitted GM level and noise levels of every estimator and ray, ± leave-one-transmission-out jackknife
    SE."""
    table = fit_table(result)
    names = list(result.fits)
    ray_ids = list(result.config.ray_ids)
    colors = _colors(len(ray_ids))
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 1.2 + 0.32 * len(names)), sharey=True)
    y = np.arange(len(names))
    for k, ray_id in enumerate(ray_ids):
        rows = table[table.ray_id == ray_id].set_index("estimator").loc[names]
        offset = (k - (len(ray_ids) - 1) / 2) * 0.12
        for ax, key, label in zip(
            axes,
            ("GM", "phi_noise", "chi_noise"),
            ("GM level", "$F_\\phi$ noise [rad$^2$ m]", "$F_\\chi$ noise [m]"),
            strict=True,
        ):
            ax.errorbar(
                rows[key],
                y + offset,
                xerr=rows[f"{key}_se"],
                fmt="o",
                ms=3,
                color=colors[k],
                label=ray_id,
                lw=0.8,
            )
            ax.set_xlabel(label)
    axes[0].set_yticks(y, names)
    axes[0].invert_yaxis()
    for ax in axes:
        ax.grid(lw=0.3, alpha=0.5)
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    return fig


def plot_common_patterns(result: ExperimentResult) -> Figure:
    """Common phase and log-amplitude patterns (mean over transmissions of the detrended records) of each
    ray ± SE, pooled over the rays and over the arrivals (dashed), and of each arrival (grey), for each
    wavefront source."""
    base_source = result.config.base.wavefronts
    sources = sorted(result.patterns, key=lambda source: source != base_source)
    ray_ids = list(result.config.ray_ids)
    colors = _colors(len(ray_ids))
    bad = list(result.config.base.bad_elements)
    fig, axes = plt.subplots(
        len(sources), 2, figsize=(7, 2.4 * len(sources) + 0.4), squeeze=False
    )
    for i, source in enumerate(sources):
        p = result.patterns[source]
        for j, key in enumerate(("phase", "log_amplitude")):
            ax = axes[i, j]
            for k, ray_id in enumerate(ray_ids):
                pattern = p["per_ray"][key][ray_id]
                se = p["per_ray_se"][key][ray_id]
                el = np.arange(len(pattern))
                ax.plot(el, pattern, color=colors[k], lw=1, label=ray_id)
                ax.fill_between(
                    el, pattern - se, pattern + se, color=colors[k], alpha=0.15, lw=0
                )
            for k, pattern in enumerate(p["per_arrival"][key].values()):
                ax.plot(
                    el,
                    pattern,
                    color="0.75",
                    lw=0.5,
                    zorder=0,
                    label="each arrival" if k == 0 else None,
                )
            ax.plot(
                el, p["all_rays"][key][ray_ids[0]], color="k", lw=1.5, label="all rays"
            )
            ax.plot(
                el,
                p["all_arrivals"][key][ray_ids[0]],
                color="k",
                lw=1.5,
                ls="--",
                label="all arrivals",
            )
            for b in bad:
                ax.axvline(b, color="0.6", lw=0.5, ls=":")
            ax.set_ylabel(("phase [rad]", "log-amplitude [Np]")[j])
            ax.set_title(f"{source}", loc="left", fontsize=8)
            ax.grid(lw=0.3, alpha=0.5)
    for ax in axes[-1]:
        ax.set_xlabel("element (lower array, 9 m spacing)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.3 / fig.get_figheight(), 1, 1))
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncols=len(labels),
        fontsize=7,
        frameon=False,
    )
    return fig


def plot_synthetic_bias(result: ExperimentResult) -> Figure:
    """Median over realizations of log10(estimate / truth) of every estimator, with the 16-84 % range for up
    to 6 estimators."""
    syn = result.synthetic
    ray_ids = list(result.config.ray_ids)
    names = list(syn["bias"])
    colors = _colors(len(names))
    fig, axes = plt.subplots(
        len(ray_ids),
        2,
        figsize=(7, 2 * len(ray_ids) + 1),
        sharex=True,
        sharey="row",
        squeeze=False,
    )
    for i, ray_id in enumerate(ray_ids):
        for name, color in zip(names, colors, strict=True):
            m = syn["m"][name]
            for j, key in enumerate(("phi", "chi")):
                b = syn["bias"][name][ray_id][key]
                ax = axes[i, j]
                ax.semilogx(m, np.nanmedian(b, axis=0), color=color, lw=1.2, label=name)
                if len(names) <= 6:
                    q16, q84 = np.nanpercentile(b, [16, 84], axis=0)
                    ax.fill_between(m, q16, q84, color=color, alpha=0.12, lw=0)
        axes[i, 0].set_ylabel(f"{ray_id}\nlog$_{{10}}$(estimate / truth)")
    for ax in axes.flat:
        ax.axhline(0, color="k", lw=0.6)
        ax.grid(which="both", lw=0.3, alpha=0.5)
    for ax in axes[-1]:
        ax.set_xlabel("$m$ [cpm]")
    axes[0, 0].set_title("phase", loc="left")
    axes[0, 1].set_title("log-amplitude", loc="left")
    fig.tight_layout(rect=(0, _legend_strip(fig, len(names)), 1, 1))
    _legend_below(fig, *axes[0, 0].get_legend_handles_labels())
    return fig


def plot_synthetic_fits(result: ExperimentResult) -> Figure:
    """Fitted / true GM level and noise levels over the synthetic realizations, pooled over rays: median,
    interquartile box and 5-95 % whiskers."""
    syn = result.synthetic
    names = list(syn["fits"])
    ray_ids = list(result.config.ray_ids)
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 1.2 + 0.32 * len(names)), sharey=True)
    for k, (key, label) in enumerate(
        (
            ("GM", "GM level"),
            ("phi_noise", "$F_\\phi$ noise"),
            ("chi_noise", "$F_\\chi$ noise"),
        )
    ):
        data = []
        for name in names:
            ratios = [
                syn["fits"][name][r][:, k] / syn["params"][r][key] for r in ray_ids
            ]
            data.append(np.concatenate(ratios))
        ax = axes[k]
        ax.boxplot(data, vert=False, whis=(5, 95), showfliers=False, widths=0.6)
        ax.axvline(1, color="k", lw=0.6)
        ax.set_xscale("log")
        lo, hi = ax.get_xlim()
        ax.set_xlim(min(lo, 0.7), max(hi, 1 / 0.7))
        ax.xaxis.set_major_locator(
            mticker.FixedLocator([0.1, 0.2, 0.5, 0.7, 1, 1.4, 2, 5, 10])
        )
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_xlabel(f"fitted / true {label}")
        ax.grid(which="both", lw=0.3, alpha=0.5)
    axes[0].set_yticks(np.arange(1, len(names) + 1), names)
    axes[0].invert_yaxis()
    fig.tight_layout()
    return fig


def make_figures(result: ExperimentResult) -> list[Path]:
    cfg = result.config
    figures = {"spectra": plot_spectra, "common_patterns": plot_common_patterns}
    if result.fits:
        figures["fit_summary"] = plot_fit_summary
    if result.synthetic is not None:
        figures["synthetic_bias"] = plot_synthetic_bias
        figures["synthetic_fits"] = plot_synthetic_fits
    written = []
    for name, func in figures.items():
        with mpl.rc_context({"font.size": 8, "text.usetex": cfg.usetex}):
            fig = func(result)
            for fmt in cfg.formats:
                path = cfg.output_dir / f"{name}.{fmt}"
                fig.savefig(path, dpi=200, bbox_inches="tight")
                written.append(path)
            plt.close(fig)
    return written
