"""Fluctuation theory (WFT and path integral) for the NESCAFE23 geometry in different ocean environments.

The theory functions of `tools` take only the ray variables of `tools.compute_ray_path_variables`, so the
environment enters through the ray run and through ``N`` and ``dcdz_p``. Here each environment is an
`xr.Dataset` on ``(depth, range)`` (depth positive down [m], range [m] with the source at 0) with

- ``c``: sound speed [m/s],
- ``N``: buoyancy frequency [cph],
- ``dcdz_p``: potential sound speed gradient, z positive up [1/s] (Colosi 2016 Eq. 2.18),

the same contract as `tools.fctd_environment`. `run_theory` traces it with `tools.run_rays` and computes the
same quantities as the analysis (`tools.compute_n23_ray_variables`, `compute_F_phi_chi`,
`compute_phase_structure_function`, `compute_wft_structure_function`, `compute_vertical_coherence_length`,
`compute_lambda_phi`), so differences between environments are differences of environment, not of code.

The environments of `ENVIRONMENTS`, all traced with the same geometry, flat 5000 m bottom and GM parameters:

- ``munk``: canonical Munk (1974), range independent and analytic (`munk_environment`).
- ``fctd``: the smoothed, shifted FCTD of the analysis, with the cached ray run and the hand-picked
  `tools.EIG_INDEXS`, i.e. exactly the current ray variables.
- ``fctd_range_avg``: the FCTD temperature and salinity averaged over the path, re-derived
  (`fctd_range_averaged`).
- ``nesma_beta0``: the SSH-conditioned mean of the `nesma_param` ocean model (β = 0) on the SIOE-SION
  geodesic on the transmission day (`nesma_beta0_environment`). Needs the optional ``nesma`` extra.

Ray tracing uses multiprocessing, so call `run_theory` from a notebook or under
``if __name__ == "__main__":``.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import xarray as xr
from nescafe23 import acoustic_fields_from_TS
from nescafe23.data import get_data_dir

from n23_fluctuations_ragland import tools

#: NESCAFE23 moorings, ``(latitude, longitude)``. SIOE is the source (range 0), SION the receiver. Same
#: positions as `n23_inversion_ragland.transect.SIOE_COORD`/`SION_COORD`.
SIOE_COORD = (38.76510, -62.26043)
SION_COORD = (39.61990, -63.65712)

#: Coriolis parameter at the midpoint of the moorings [rad/s]
F_N23 = 2 * 7.292115e-5 * np.sin(np.radians(39.192499999))

#: range grid on which the range-independent environments are tiled [m]
RANGE_INDEPENDENT_RANGES = np.linspace(-300e3, 300e3, 13)


def _tile_range(profile: xr.Dataset, ranges: np.ndarray) -> xr.Dataset:
    """Repeat a profile Dataset (dimension ``depth``) over ``range``."""
    ones = xr.DataArray(np.ones(len(ranges)), dims=["range"], coords={"range": ranges})
    return (profile * ones).transpose("depth", "range")


def munk_environment(
    B: float = 1300.0,
    z_axis: float = 1300.0,
    gamma_a: float = 1.134e-5,
    mu: float = 24.5,
    c1: float = 1500.0,
    g: float = 9.81,
    depth: np.ndarray | None = None,
    ranges: np.ndarray | None = None,
) -> xr.Dataset:
    r"""
    Canonical Munk (1974) environment, range independent. Sound speed, stratification and potential sound
    speed gradient are mutually consistent: with $\eta = 2(z - z_a)/B$ and $\epsilon = \gamma_a B / 2$,

    - $c = c_1[1 + \epsilon(\eta + e^{-\eta} - 1)]$,
    - $N^2 = (g\gamma_a/\mu)\, e^{-\eta}$, i.e. $N = N_0 e^{-z/B}$,
    - $\frac{1}{c}\frac{dc_p}{dz} = \frac{\mu}{g} N^2$ (Colosi 2016 Eq. 2.20, z positive up),

    so that the total gradient is the potential part plus the adiabatic gradient $\gamma_a c_1$ (exactly,
    with $c_1$ in place of $c$ in the last relation; Colosi's relation with $c$ is used). The defaults
    give $\epsilon = 0.00737$ (as `pygenray.environment.munk_ssp`) and $N$ = 3.3 cph at the surface.

    Parameters
    ----------
    B : float
        thermocline scale depth [m]
    z_axis : float
        sound channel axis depth [m]
    gamma_a : float
        adiabatic fractional sound speed gradient [1/m]
    mu : float
        $\mu$ of $(1/c)(dc_p/dz) = (\mu/g) N^2$, dimensionless
    c1 : float
        sound speed at the axis [m/s]
    g : float
        gravitational acceleration [m/s²]
    depth : np.ndarray, optional
        depth grid [m]. Default is 0-5999 m at 1 m.
    ranges : np.ndarray, optional
        range grid [m] on which the profile is repeated. Default is `RANGE_INDEPENDENT_RANGES`.

    Returns
    -------
    envi_field : xr.Dataset
        ``c`` [m/s], ``N`` [cph], ``dcdz_p`` [1/s] on ``(depth, range)``
    """
    if depth is None:
        depth = np.arange(0, 6000.0)
    if ranges is None:
        ranges = RANGE_INDEPENDENT_RANGES

    eps = gamma_a * B / 2
    eta = 2 * (depth - z_axis) / B

    c = c1 * (1 + eps * (eta + np.exp(-eta) - 1))
    N2 = g * gamma_a / mu * np.exp(-eta)  # [rad²/s²]
    N = np.sqrt(N2) / (2 * np.pi) * 3600  # [cph]
    dcdz_p = c * mu * N2 / g  # [1/s], z positive up

    profile = xr.Dataset(
        {
            "c": ("depth", c, {"units": "m/s"}),
            "N": ("depth", N, {"units": "cph"}),
            "dcdz_p": ("depth", dcdz_p, {"units": "1/s"}),
        },
        coords={"depth": depth},
    )

    return _tile_range(profile, ranges)


def fctd_range_averaged(
    envi_field: xr.Dataset | None = None,
    r_min: float = 0.0,
    r_max: float = 153e3,
    ranges: np.ndarray | None = None,
) -> xr.Dataset:
    """
    The FCTD environment with in-situ temperature and practical salinity averaged over the path, then
    re-derived with `nescafe23.acoustic_fields_from_TS` (the derivation of the FCTD itself) and repeated
    over range.

    Parameters
    ----------
    envi_field : xr.Dataset, optional
        environment with ``t`` and ``SP``. Default is `tools.fctd_environment`.
    r_min, r_max : float
        range interval averaged over [m]. Default is source to receiver.
    ranges : np.ndarray, optional
        range grid [m] on which the profile is repeated. Default is `RANGE_INDEPENDENT_RANGES`.

    Returns
    -------
    envi_field : xr.Dataset
        ``t``, ``SP``, ``c``, ``N`` [cph], ``SA``, ``dcdz_p`` [1/s] on ``(depth, range)``
    """
    if envi_field is None:
        envi_field = tools.fctd_environment()
    if ranges is None:
        ranges = RANGE_INDEPENDENT_RANGES

    path = envi_field[["t", "SP"]].sel(range=slice(r_min, r_max)).mean("range")
    tiled = _tile_range(path, ranges)

    return acoustic_fields_from_TS(tiled["t"], tiled["SP"])


def transect_points(
    origin: tuple[float, float], target: tuple[float, float], x: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample the geodesic from ``origin`` towards ``target`` at distances ``x`` [m], past ``target`` if
    needed (WGS84). Mirrors `n23_inversion_ragland.transect.transect_points`.

    Returns
    -------
    lats, lons : np.ndarray
        latitude and longitude of each point [deg]
    """
    from pyproj import Geod

    geod = Geod(ellps="WGS84")
    bearing, _, _ = geod.inv(origin[1], origin[0], target[1], target[0])
    lons, lats, _ = geod.fwd(
        np.full(len(x), origin[1]),
        np.full(len(x), origin[0]),
        np.full(len(x), bearing),
        x,
    )
    return np.asarray(lats), np.asarray(lons)


def nesma_beta0_environment(
    time: str = "2023-04-30",
    N_eof: int = 10,
    x_max: float = 200e3,
    N_x: int = 501,
    argo_eof_cache: str | Path | None = None,
    pad_ranges: tuple[float, float] = (-300e3, 300e3),
) -> xr.Dataset:
    r"""
    The `nesma_param` ocean model at β = 0 on the SIOE-SION geodesic: the mean of the Argo EOF
    coefficients conditioned on the satellite SSH at each range,

    $T = \mu_T + \Psi_T\, \mu_\alpha(\zeta(x))$, $S = \mu_S + \Psi_S\, \mu_\alpha(\zeta(x))$,

    with $\mu_T, \mu_S$ the regional WOA23 mean. This is `gen_models.VerticalHorizontalModel.forward` at
    β = 0, which does not depend on the horizontal basis, built with the parameters of the
    `n23_inversion_ragland` observation runs (10 EOFs, EOFs to 1800 m, 6000 depth points).
    The acoustic fields are then derived with `nescafe23.acoustic_fields_from_TS`, as for the FCTD (so
    pressure is at the SION latitude, not the latitude 38° of `forward_c`; < 0.05 m/s difference).

    Needs the optional ``nesma`` extra and the ``WOA2023_PATH``, ``ARGO_DATA_DIR`` and
    ``COPERNICUS_SATELLITE_ALTIMETRY_PATH`` environment variables.

    Parameters
    ----------
    time : str
        date of the SSH field. Default is the transmission day (day of year 120).
    N_eof : int
        number of vertical EOF modes
    x_max : float
        range of the last transect point [m]
    N_x : int
        number of transect points
    argo_eof_cache : str or Path, optional
        cache file of the Argo EOFs. Default is ``dataproducts/argo_EOF_fluct.npz`` under
        ``NESCAFE23_DATADIR``, separate from the inversion's cache, which a different grid would overwrite.
    pad_ranges : tuple of float
        the edge profiles are repeated out to these ranges [m], so that the backward ray run is covered

    Returns
    -------
    envi_field : xr.Dataset
        ``t``, ``SP``, ``c``, ``N`` [cph], ``SA``, ``dcdz_p`` [1/s] on ``(depth, range)``, with the
        coordinates ``latitude``, ``longitude`` and ``ssh`` [m] over range
    """
    try:
        import pandas as pd
        from nesma_param import gen_models, open_data
        from nesma_param.cross_stream import ssh_at_points
        from nesma_param.jax_helpers import interp1d_linear
    except ImportError as err:
        msg = (
            "nesma_beta0_environment needs nesma_param, which is not a dependency of "
            "n23_fluctuations_ragland (it is not public yet). Within the full NESCAFE23 "
            "workspace, install it with `uv sync --all-packages`."
        )
        raise ImportError(msg) from err

    if argo_eof_cache is None:
        argo_eof_cache = get_data_dir() / "dataproducts" / "argo_EOF_fluct.npz"

    zeta_grid = np.linspace(-0.5, 1.5, 500)
    z_grid = np.linspace(0, 6000, 6000)
    x_grid = np.linspace(0, x_max, N_x)

    woa = open_data.open_woa(z_grid)
    Psi_T, Psi_S, _, mu_alpha, *_ = gen_models.get_vertical_basis(
        str(argo_eof_cache),
        N_eof,
        zeta_grid,
        z_grid,
        max_EOF_depth=1800,
        min_EOF_depth=5,
    )

    lats, lons = transect_points(SIOE_COORD, SION_COORD, x_grid)
    ssh = np.asarray(ssh_at_points(pd.Timestamp(time), lats, lons))

    alpha = np.asarray(
        interp1d_linear(ssh, zeta_grid, mu_alpha, fill_value="boundary", axis=0)
    )  # (N_x, N)
    T = woa["T"].to_numpy()[:, None] + np.asarray(Psi_T) @ alpha.T  # (N_z, N_x)
    S = woa["S"].to_numpy()[:, None] + np.asarray(Psi_S) @ alpha.T

    coords = {
        "depth": z_grid,
        "range": x_grid,
        "latitude": ("range", lats),
        "longitude": ("range", lons),
        "ssh": ("range", ssh),
    }
    TS = xr.Dataset(
        {"t": (("depth", "range"), T), "SP": (("depth", "range"), S)}, coords=coords
    ).interp(depth=np.arange(0, 6000.0))

    # repeat the edge profiles out to the padding ranges
    TS = xr.concat(
        (
            TS.isel(range=0).assign_coords(range=pad_ranges[0]),
            TS,
            TS.isel(range=-1).assign_coords(range=pad_ranges[1]),
        ),
        dim="range",
    )

    return acoustic_fields_from_TS(TS["t"], TS["SP"])


#: environments compared by `run_theory`: a builder, the eigenray pairing, and whether to use the cached
#: ray run of the analysis (`ray_run_forward_back.pkl`) instead of tracing
ENVIRONMENTS = {
    "munk": {
        "build": munk_environment,
        "eig_indexs": "reciprocity",
        "cached_ray_run": False,
    },
    "fctd": {
        "build": tools.fctd_environment,
        "eig_indexs": None,
        "cached_ray_run": True,
    },
    "fctd_range_avg": {
        "build": fctd_range_averaged,
        "eig_indexs": "reciprocity",
        "cached_ray_run": False,
    },
    "nesma_beta0": {
        "build": nesma_beta0_environment,
        "eig_indexs": "reciprocity",
        "cached_ray_run": False,
    },
}


def theory_path(name: str) -> Path:
    """Path of the cached `run_theory` output of environment ``name``."""
    return get_data_dir() / "dataproducts" / "theory_envs" / f"{name}.pkl"


def run_theory(
    name: str,
    envi_field: xr.Dataset | None = None,
    *,
    ray_ids: list | None = None,
    eig_indexs: dict | str | None = "reciprocity",
    ray_run: dict | None = None,
    deltaz_r: np.ndarray | None = None,
    m: np.ndarray | None = None,
    fc: float = 250,
    jstar: float = 3,
    zeta0: float = 7.3,
    N0: float = 3 * (2 * np.pi) / 3600,
    f: float = F_N23,
    num_rays: int | None = None,
    force_overwrite: bool = False,
    save: bool = True,
) -> dict:
    r"""
    Trace an environment and compute the WFT and path-integral statistics of its eigenrays.

    For a name in `ENVIRONMENTS`, ``envi_field``, ``eig_indexs`` and ``ray_run`` default to its entry. The
    result is cached at `theory_path` and loaded from there unless ``force_overwrite``.

    Parameters
    ----------
    name : str
        environment name, used for the cache file
    envi_field : xr.Dataset, optional
        environment (``c``, ``N`` [cph], ``dcdz_p`` [1/s] on ``(depth, range)``)
    ray_ids : list, optional
        NESCAFE23 ray ids. Default is `tools.DUCTED_RAY_IDS`.
    eig_indexs : dict or "reciprocity" or None
        eigenray pairing, see `tools.compute_n23_ray_variables`
    ray_run : dict, optional
        ray run as written by the CLI; if None, traced with `tools.run_rays` (default geometry and bottom)
    deltaz_r : np.ndarray, optional
        vertical separations [m]. Default is ``linspace(0, 250, 200)``.
    m : np.ndarray, optional
        vertical wavenumbers [cycles/m]. Default is ``logspace(-5, 0, 4000)``.
    fc, jstar, zeta0, N0, f
        acoustic frequency [Hz] and GM parameters (N0 and f in rad/s)
    num_rays : int, optional
        size of the ray fan, passed to `tools.run_rays`
    force_overwrite : bool
        recompute even if the cache exists
    save : bool
        write the cache

    Returns
    -------
    theory : dict
        ``name``, ``envi_field``, ``ray_run`` (eigenrays, environments and geometry; no ray fans),
        ``rvs`` and ``eig_indexs``, ``deltaz_r``, ``m`` [cpm], and per ray id a list over micro-rays of
        ``F_phi``, ``F_chi`` (per cycle/m), ``D_PI``, ``D_wft_phi``, ``D_wft_chi``, ``z0``, ``Phi2``,
        ``Lambda``, plus ``params``
    """
    fn = theory_path(name)
    if fn.exists() and not force_overwrite:
        with fn.open("rb") as file:
            return pickle.load(file)

    entry = ENVIRONMENTS.get(name)
    if entry is not None:
        if envi_field is None:
            envi_field = entry["build"]()
        eig_indexs = entry["eig_indexs"]
        if ray_run is None and entry["cached_ray_run"]:
            with (get_data_dir() / "dataproducts" / "ray_run_forward_back.pkl").open(
                "rb"
            ) as file:
                ray_run = pickle.load(file)
    if envi_field is None:
        msg = f"unknown environment {name!r}: pass envi_field"
        raise ValueError(msg)

    if ray_ids is None:
        ray_ids = list(tools.DUCTED_RAY_IDS)
    if deltaz_r is None:
        deltaz_r = np.linspace(0, 250, 200)
    if m is None:
        m = np.logspace(-5, 0, 4000)

    if ray_run is None:
        _, eigs, _, eigs_back, envi, envi_back, geo = tools.run_rays(
            num_rays=num_rays, envi_field=envi_field
        )
    else:
        eigs, eigs_back = ray_run["eigs"], ray_run["eigs_back"]
        envi, envi_back, geo = ray_run["envi"], ray_run["envi_back"], ray_run["geo"]
    ray_run = {
        "eigs": eigs,
        "eigs_back": eigs_back,
        "envi": envi,
        "envi_back": envi_back,
        "geo": geo,
    }

    if eig_indexs == "reciprocity":
        eig_indexs = {
            ray_id: tools.pair_eigenrays(ray_id, eigs, eigs_back) for ray_id in ray_ids
        }
    elif eig_indexs is None:
        eig_indexs = {ray_id: tools.EIG_INDEXS[ray_id] for ray_id in ray_ids}

    rvs = tools.compute_n23_ray_variables(
        fc=fc,
        jstar=jstar,
        f=f,
        zeta0=zeta0,
        N0=N0,
        ray_ids=ray_ids,
        envi_field=envi_field,
        ray_run=ray_run,
        eig_indexs=eig_indexs,
    )

    keys = ["F_phi", "F_chi", "D_PI", "D_wft_phi", "D_wft_chi", "z0", "Phi2", "Lambda"]
    theory = {key: {} for key in keys}
    for ray_id, rv_list in rvs.items():
        for key in keys:
            theory[key][ray_id] = []
        for rv in rv_list:
            F_phi, F_chi = tools.compute_F_phi_chi(
                rv, 2 * np.pi * m[None, :], fc=fc, jstar=jstar, f=f
            )
            D_PI, _ = tools.compute_phase_structure_function(rv, deltaz_r, fc=fc)
            D_wft_phi, D_wft_chi = tools.compute_wft_structure_function(
                rv, deltaz_r, m, fc=fc, jstar=jstar, f=f
            )
            z0, _ = tools.compute_vertical_coherence_length(rv, fc=fc)
            Lambda, Phi2 = tools.compute_lambda_phi(rv, fc=fc)

            for key, value in zip(
                keys,
                [F_phi, F_chi, D_PI, D_wft_phi, D_wft_chi, z0, Phi2, Lambda],
                strict=True,
            ):
                theory[key][ray_id].append(np.squeeze(value))

    theory.update(
        {
            "name": name,
            "envi_field": envi_field,
            "ray_run": ray_run,
            "rvs": rvs,
            "eig_indexs": {ray_id: eig_indexs[ray_id] for ray_id in rvs},
            "deltaz_r": deltaz_r,
            "m": m,
            "params": {"fc": fc, "jstar": jstar, "zeta0": zeta0, "N0": N0, "f": f},
        }
    )

    if save:
        fn.parent.mkdir(parents=True, exist_ok=True)
        with fn.open("wb") as file:
            pickle.dump(theory, file)

    return theory
