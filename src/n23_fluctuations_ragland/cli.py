import pickle
from functools import wraps
from pathlib import Path

import fsspec
import matplotlib.pyplot as plt
import nescafe23 as n23
import numpy as np
import pandas as pd
import pygenray as pr
import questionary
import xarray as xr
from nescafe23 import convert_to_xarray
from nescafe23.data import get_data_dir
from scipy import signal

from n23_fluctuations_ragland import estimation, figures, tools


def check_output_exists(*output_paths):
    """
    Decorator to check if output file(s) exist and handle overwrite logic.

    The decorated function must have a `force_overwrite` parameter (bool).
    The decorator will check if the output file(s) exist and either:
    - Skip the function if any file exists and user doesn't want to overwrite
    - Run the function if no files exist or user wants to overwrite

    Parameters
    ----------
    *output_paths : str or callable
        One or more paths to output files to check, or callables that take
        the same arguments as the decorated function and return paths.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, force_overwrite=False, **kwargs):
            filepaths = []
            for output_path in output_paths:
                if callable(output_path):
                    filepath = output_path(*args, **kwargs)
                else:
                    filepath = output_path
                filepaths.append(Path(filepath))

            existing_files = [fp for fp in filepaths if fp.exists()]

            if not force_overwrite and existing_files:
                files_list = "\n".join(str(fp) for fp in existing_files)
                prompt_message = f"Output file(s) already exist:\n{files_list}\nDo you want to overwrite them?"
                should_proceed = questionary.confirm(
                    prompt_message, default=False
                ).ask()

                if not should_proceed:
                    print(f"Skipping {func.__name__}. Output already exists.")
                    return None

            return func(*args, **kwargs)

        return wrapper

    return decorator


def main():
    """Interactive CLI menu."""
    try:
        get_data_dir()
    except OSError:
        print("\nNESCAFE23_DATADIR is not set.")
        print("Run 'nescafe23-init' to configure the data directory, then try again.")
        return

    choice = questionary.select(
        "Which script would you like to run?",
        choices=[
            "run all processing",
            "compute arrival time offset",
            "compute mean pulse",
            "estimate peak locations",
            "estimate array tilt",
            "compute wavefront values",
            "estimate hydrophone pattern",
            "compute rays",
            "source geometry sweep",
            "ɸ χ GM least squares fit",
            "coherence lengths (observed vs. GM)",
            "GM levels (spectra vs. coherence)",
            "transmission independence test",
            "create estimation experiments config",
            "run estimation experiments",
            "create publication figures config",
            "generate publication figures",
            "exit",
        ],
    ).ask()

    if choice == "run all processing":
        run_all_processing()
    elif choice == "compute arrival time offset":
        compute_arrival_time_offset()
    elif choice == "compute mean pulse":
        compute_mean_pulse()
    elif choice == "estimate peak locations":
        estimate_peak_locations()
    elif choice == "estimate array tilt":
        estimate_array_tilt()
    elif choice == "compute wavefront values":
        get_wavefront_values()
    elif choice == "estimate hydrophone pattern":
        estimate_hydrophone_pattern()
    elif choice == "compute rays":
        compute_rays()
    elif choice == "source geometry sweep":
        source_geometry_sweep()
    elif choice == "ɸ χ GM least squares fit":
        phi_chi_LS_fit()
    elif choice == "coherence lengths (observed vs. GM)":
        coherence_lengths()
    elif choice == "GM levels (spectra vs. coherence)":
        gm_levels()
    elif choice == "transmission independence test":
        independence_test()
    elif choice == "create estimation experiments config":
        create_estimation_experiments_config()
    elif choice == "run estimation experiments":
        run_estimation_experiments()
    elif choice == "create publication figures config":
        create_publication_figures_config()
    elif choice == "generate publication figures":
        generate_publication_figures()
    elif choice == "exit":
        print("Goodbye!")
        return


def run_all_processing():
    """Run all processing steps in sequence."""
    print("=" * 60)
    print("Running all processing steps...")
    print("=" * 60)

    overwrite_all = questionary.confirm(
        "Overwrite existing data products if they exist?", default=False
    ).ask()

    processing_steps = [
        ("Step 1: Computing arrival time offset", compute_arrival_time_offset, {}),
        ("Step 2: Estimating peak locations", estimate_peak_locations, {}),
        ("Step 3: Estimating array tilt", estimate_array_tilt, {}),
        (
            "Step 4: Computing wavefront values",
            get_wavefront_values,
            {"remove_array_tilt": False, "remove_pattern": "none"},
        ),
        (
            "Step 5: Computing wavefront values (array tilt removed)",
            get_wavefront_values,
            {"remove_array_tilt": True, "remove_pattern": "none"},
        ),
        ("Step 6: Computing mean pulse", compute_mean_pulse, {}),
        ("Step 7: Computing rays", compute_rays, {}),
        ("Step 8: Source geometry sweep", source_geometry_sweep, {}),
        (
            "Step 9: ɸ χ GM least squares fit",
            phi_chi_LS_fit,
            {"remove_array_tilt": False, "remove_pattern": "none"},
        ),
        (
            "Step 10: ɸ χ GM least squares fit (array tilt removed)",
            phi_chi_LS_fit,
            {"remove_array_tilt": True, "remove_pattern": "none"},
        ),
    ]

    for step_name, step_func, step_kwargs in processing_steps:
        print(f"\n{'=' * 60}")
        print(f"{step_name}")
        print(f"{'=' * 60}")
        try:
            step_func(force_overwrite=overwrite_all, **step_kwargs)
            print(f"✓ {step_name} completed successfully")
        except Exception as e:
            print(f"✗ Error in {step_name}: {e!s}")
            if not questionary.confirm(
                "Continue with remaining steps?", default=True
            ).ask():
                raise


@check_output_exists(lambda: get_data_dir() / "dataproducts" / "T_t_estimated.pkl")
def compute_arrival_time_offset():
    """Compute arrival time offset between transmissions."""
    print("Estimating arrival time offset using cross-correlations...")

    data_dir = get_data_dir()
    T_t_path = data_dir / "dataproducts" / "T_t_estimated.pkl"

    fs = fsspec.filesystem("")
    fns = fs.glob(str(data_dir / "dmod" / "dm*.mat"))

    dss = []
    for fn in fns:
        dss.append(convert_to_xarray(fn))

    depth_grid_up = np.linspace(125, 540, 50)
    depth_grid_down = np.linspace(885, 1315, 50)

    dss_interp_up = []
    dss_interp_down = []

    for ds in dss:
        dss_interp_up.append(
            ds["dmod"][:32]
            .swap_dims({"hydrophone": "actual_depth"})
            .drop_vars(["nominal_depth"])
            .interp(
                {"actual_depth": depth_grid_up},
                kwargs={"bounds_error": False, "fill_value": 0},
            )
            .sel({"time": slice(100, 115)})
        )
        dss_interp_down.append(
            ds["dmod"][32:]
            .swap_dims({"hydrophone": "actual_depth"})
            .drop_vars(["nominal_depth"])
            .interp(
                {"actual_depth": depth_grid_down},
                kwargs={"bounds_error": False, "fill_value": 0},
            )
            .sel({"time": slice(100, 115)})
        )

    Ts = float(dss_interp_up[0].time.diff("time")[0])
    Rxx_taus = np.arange(-15 / 2, 15 / 2, Ts)[1:]

    Rxxs = []
    for k in range(23):
        interp1 = dss_interp_up[0].to_numpy() - dss_interp_up[0].mean().to_numpy()
        interp2 = dss_interp_up[k].to_numpy() - dss_interp_up[k].mean().to_numpy()

        interp1[np.isnan(interp1)] = 0
        interp2[np.isnan(interp2)] = 0

        Rxx = signal.fftconvolve(
            np.abs(interp1), np.flip(np.abs(interp2), axis=1), mode="same", axes=1
        )
        Rxxs.append(Rxx)

    dT_t_upper = []
    for _k, Rxx in enumerate(Rxxs):
        Rxx_sum = np.sum(Rxx, axis=0)
        idx = np.argmax(Rxx_sum)
        dT_t_upper.append(Rxx_taus[idx])
    dT_t_upper = np.array(dT_t_upper)

    Rxxs = []
    for k in range(23):
        interp1 = dss_interp_down[0].to_numpy() - dss_interp_down[0].mean().to_numpy()
        interp2 = dss_interp_down[k].to_numpy() - dss_interp_down[k].mean().to_numpy()

        interp1[np.isnan(interp1)] = 0
        interp2[np.isnan(interp2)] = 0

        Rxx = signal.fftconvolve(
            np.abs(interp1), np.flip(np.abs(interp2), axis=1), mode="same", axes=1
        )
        Rxxs.append(Rxx)

    dT_t_lower = []
    for Rxx in Rxxs:
        Rxx_sum = np.sum(Rxx, axis=0)
        idx = np.argmax(Rxx_sum)
        dT_t_lower.append(Rxx_taus[idx])
    dT_t_lower = np.array(dT_t_lower)

    dT_avg = (dT_t_upper + dT_t_lower) / 2

    with Path(T_t_path).open("wb") as f:
        pickle.dump(
            {
                "dT_t": dT_avg,
                "dT_t_upper": dT_t_upper,
                "dT_t_lower": dT_t_lower,
                "notes": "tidal arrival time offset estimated with cross-correlations in time of the reception envelope",
            },
            f,
        )


@check_output_exists(lambda: get_data_dir() / "dataproducts" / "peak_locations.pkl")
def estimate_peak_locations():
    """
    Using the location of maximum intensity, and hand chosen angle/time bounding boxes,
    find the location in angle and time for each transmission and peak.
    """
    data_dir = get_data_dir()
    dss = tools.open_receptions()
    times, angles, snrs, upper_times, upper_angles, upper_snrs = n23.get_peak_location(
        dss
    )
    peak_locations = {
        "times": times,
        "angles": angles,
        "snrs": snrs,
        "upper_times": upper_times,
        "upper_angles": upper_angles,
        "upper_snrs": upper_snrs,
    }

    with (data_dir / "dataproducts" / "peak_locations.pkl").open("wb") as f:
        pickle.dump(peak_locations, f)


def _ask_remove_array_tilt():
    return questionary.confirm("Remove array tilt?", default=True).ask()


@check_output_exists(lambda: get_data_dir() / "dataproducts" / "array_tilt.pkl")
def estimate_array_tilt():
    """
    Estimate the lower-array tilt from the estimated peak locations, see `tools.estimate_array_tilt`.
    """
    data_dir = get_data_dir()
    array_tilt = tools.estimate_array_tilt(n23.open_peak_locations())

    with (data_dir / "dataproducts" / "array_tilt.pkl").open("wb") as f:
        pickle.dump(array_tilt, f)


def _ask_remove_pattern():
    return questionary.select(
        "Remove hydrophone pattern?",
        choices=list(tools.REMOVE_PATTERN_CHOICES),
        default="none",
    ).ask()


def _wavefront_values_path(remove_array_tilt, remove_pattern="none"):
    return tools.wavefront_values_path(remove_array_tilt, remove_pattern)


def get_wavefront_values(
    remove_array_tilt=None, remove_pattern=None, force_overwrite=False
):
    """
    Given the estimated peak locations, extract the complex pressure values of the wavefront at these time locations.

    With the array tilt removed, each ray is beamformed at its mean angle plus the array tilt and the
    result is written to ``wavefront_values_tilt_removed.pkl`` instead of ``wavefront_values.pkl``.
    With a hydrophone pattern removed (``phase``, ``magnitude`` or ``both``, from "estimate hydrophone
    pattern"), it is removed from the receptions before beamforming and the file name gets
    ``_phase_removed``, ``_magnitude_removed`` or ``_phase_magnitude_removed``. The user is asked whether
    to remove the tilt and the pattern when `remove_array_tilt` and `remove_pattern` are None.
    """
    if remove_array_tilt is None:
        remove_array_tilt = _ask_remove_array_tilt()
    if remove_pattern is None:
        remove_pattern = _ask_remove_pattern()
    _compute_wavefront_values(
        remove_array_tilt=remove_array_tilt,
        remove_pattern=remove_pattern,
        force_overwrite=force_overwrite,
    )


@check_output_exists(
    lambda remove_array_tilt, remove_pattern: _wavefront_values_path(
        remove_array_tilt, remove_pattern
    )
)
def _compute_wavefront_values(remove_array_tilt, remove_pattern):
    dss = tools.open_receptions()
    wavefront_values = tools.get_wavefront_values(
        dss, remove_array_tilt=remove_array_tilt, remove_pattern=remove_pattern
    )

    with _wavefront_values_path(remove_array_tilt, remove_pattern).open("wb") as f:
        pickle.dump(wavefront_values, f)


@check_output_exists(tools.hydrophone_pattern_path)
def estimate_hydrophone_pattern():
    """
    Estimate the phase and gain of each lower-array hydrophone from the wavefront values with the array tilt
    removed and no pattern removed (`tools.estimate_hydrophone_pattern`), write it to
    ``hydrophone_pattern.pkl``, print it, and save a diagnostic plot to ``hydrophone_pattern.png``.
    """
    wavefront_values = _load_wavefront_values(True)
    pattern = tools.estimate_hydrophone_pattern(wavefront_values)

    path = tools.hydrophone_pattern_path()
    with path.open("wb") as f:
        pickle.dump(pattern, f)

    print("=" * 60)
    print(hydrophone_pattern_report(pattern))
    print("=" * 60)

    fig = plot_hydrophone_pattern(pattern)
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"Wrote {path} and {path.with_suffix('.png')}")
    return pattern


def hydrophone_pattern_report(pattern):
    """Per-element table and the rejected arrivals and records of `tools.estimate_hydrophone_pattern`."""
    lines = [
        f"{'element':>7} {'phase [rad]':>16} {'delay [ms]':>16} {'gain':>16}",
    ]
    for n in range(len(pattern["phase"])):
        lines.append(
            f"{n:>7} {pattern['phase'][n]:+8.3f} ± {pattern['phase_se'][n]:5.3f}"
            f" {1e3 * pattern['delay'][n]:+8.3f} ± {1e3 * pattern['delay_se'][n]:5.3f}"
            f" {pattern['gain'][n]:8.3f} ± {pattern['gain_se'][n]:5.3f}"
        )
    for key in ("phase", "gain"):
        rejected = pattern["rejected_arrivals"][key]
        arrivals = ", ".join(f"{a} (p = {p:.1e})" for a, p in rejected.items())
        records = ", ".join(
            f"{a}: {txs}" for a, txs in pattern["rejected_records"][key].items()
        )
        lines += [
            f"{key}: {len(pattern['arrivals'][key])} arrivals, {pattern['n_records'][key]} records",
            f"  rejected arrivals: {arrivals or 'none'}",
            f"  rejected records (transmissions): {records or 'none'}",
        ]
    lines.append(f"omitted arrivals: {', '.join(pattern['omit']) or 'none'}")
    return "\n".join(lines)


def plot_hydrophone_pattern(pattern):
    """Phase (and delay) and gain along the array with their jackknife standard errors, over each arrival's
    own pattern (rejected arrivals dashed)."""
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    for ax, key, label in (
        (axes[0], "phase", "phase [rad]"),
        (axes[1], "gain", "gain"),
    ):
        el = np.arange(len(pattern[key]))
        for ray_id, p in pattern["per_arrival"][key].items():
            rejected = ray_id in pattern["rejected_arrivals"][key]
            ax.plot(
                el,
                p,
                color="C3" if rejected else "0.6",
                ls="--" if rejected else "-",
                lw=0.8,
                alpha=0.8 if rejected else 0.4,
            )
        ax.errorbar(
            el,
            pattern[key],
            yerr=pattern[f"{key}_se"],
            color="k",
            marker="o",
            ms=3,
            capsize=2,
        )
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    delay_axis = axes[0].secondary_yaxis(
        "right",
        functions=(
            lambda phi: 1e3 * phi / (2 * np.pi * pattern["f0"]),
            lambda tau: 2 * np.pi * pattern["f0"] * tau / 1e3,
        ),
    )
    delay_axis.set_ylabel("delay [ms]")
    axes[1].set_xlabel("element")
    axes[0].set_title(
        "hydrophone pattern (black: pooled ± jackknife SE; dashed: rejected arrivals)"
    )
    return fig


@check_output_exists(
    lambda: get_data_dir() / "dataproducts" / "ray_run_forward_back.pkl"
)
def compute_rays():
    """
    Compute the forward and backward rays for NESCAFE23 geometry and environment.
    """
    data_dir = get_data_dir()
    rays, eigs, rays_back, eigs_back, envi, envi_back, geo = tools.run_rays()

    ray_data = {
        "rays": rays,
        "eigs": eigs,
        "rays_back": rays_back,
        "eigs_back": eigs_back,
        "envi": envi,
        "envi_back": envi_back,
        "geo": geo,
    }

    with (data_dir / "dataproducts" / "ray_run_forward_back.pkl").open("wb") as f:
        pickle.dump(ray_data, f)


@check_output_exists(lambda: get_data_dir() / "dataproducts" / "ray_geo_sweep.pkl")
def source_geometry_sweep():
    """
    Do a sweep of source depth and range to match the observed ducted section of the wavefront to rays through the FCTD.
    """
    data_dir = get_data_dir()
    dss = tools.open_receptions()
    lower_array_depth = [float(dss[k].actual_depth[32:].mean()) for k in range(23)]
    Z_lower_array = np.mean(lower_array_depth)

    fctd = tools.compute_filtered_fctd_grid(z_smooth=500, r_smooth=10000)
    front_location = 35e3  # m
    fctd = fctd.assign_coords({"range": (fctd.range.to_numpy() + front_location)})

    c = fctd["c"]
    bathy = xr.DataArray(
        np.linspace(4908, 4908, 100),
        dims=["range"],
        coords={"range": np.linspace(0, 200e3, 100)},
    )
    envi = pr.OceanEnvironment2D(sound_speed=c, bathymetry=bathy)

    source_depths = np.array(
        [
            2000,
            2050,
            2100,
            2150,
            2200,
            2250,
            2300,
            2350,
            2400,
        ]
    ).astype(float)
    receiver_ranges = np.array(
        [
            148e3,
            149e3,
            150e3,
            151e3,
            152e3,
            153e3,
            154e3,
            155e3,
        ]
    ).astype(float)

    ssource_depths, rreceiver_ranges = np.meshgrid(source_depths, receiver_ranges)
    ssource_depths = ssource_depths.flatten()
    rreceiver_ranges = rreceiver_ranges.flatten()

    rays = []
    eigs = []

    for k in range(len(ssource_depths)):
        source_depth = ssource_depths[k]
        receiver_range = rreceiver_ranges[k]
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
                [float(Z_lower_array)],
                source_depth,
                0,
                receiver_range,
                1000,
                envi,
            )
        )

    with (data_dir / "dataproducts" / "ray_geo_sweep.pkl").open("wb") as f:
        pickle.dump(
            {
                "source_depths": ssource_depths,
                "receiver_ranges": rreceiver_ranges,
                "rays": rays,
                "eigs": eigs,
                "envi": envi,
            },
            f,
        )


def serve_panel_apps():
    """Serve Panel apps from the panel_apps directory."""
    import subprocess
    import sys

    panel_apps_dir = Path(__file__).parent.parent.parent / "panel_apps"
    app_files = list(panel_apps_dir.glob("*.py"))

    cmd = [
        sys.executable,
        "-m",
        "panel",
        "serve",
        *[str(f) for f in app_files],
        "--show",
    ]
    subprocess.run(cmd, check=False)


@check_output_exists(
    lambda: get_data_dir() / "dataproducts" / "mean_pulse_lower.nc",
    lambda: get_data_dir() / "dataproducts" / "mean_pulse_upper.nc",
    lambda: get_data_dir() / "dataproducts" / "mean_pulse_lower_noise.nc",
    lambda: get_data_dir() / "dataproducts" / "mean_pulse_upper_noise.nc",
)
def compute_mean_pulse():
    data_dir = get_data_dir()
    (
        mean_pulse_lower,
        mean_pulse_upper,
        mean_pulse_lower_noise,
        mean_pulse_upper_noise,
    ) = tools.compute_mean_pulse()

    mean_pulse_lower.to_netcdf(data_dir / "dataproducts" / "mean_pulse_lower.nc")
    mean_pulse_upper.to_netcdf(data_dir / "dataproducts" / "mean_pulse_upper.nc")
    mean_pulse_lower_noise.to_netcdf(
        data_dir / "dataproducts" / "mean_pulse_lower_noise.nc"
    )
    mean_pulse_upper_noise.to_netcdf(
        data_dir / "dataproducts" / "mean_pulse_upper_noise.nc"
    )


def phi_chi_LS_fit(remove_array_tilt=None, remove_pattern=None, force_overwrite=False):
    """
    Fit the GM energy and observational noise levels to the observed ɸ and χ spectra.

    The spectra are estimated from ``wavefront_values_tilt_removed.pkl`` with the array tilt removed
    and the fit is written to ``phi_chi_LS_fit_tilt_removed.pkl``, otherwise from
    ``wavefront_values.pkl`` to ``phi_chi_LS_fit.pkl``. A removed hydrophone pattern adds the same
    suffix to both (`tools.wavefront_values_path`). The user is asked whether to remove the tilt and
    the pattern when `remove_array_tilt` and `remove_pattern` are None.
    """
    if remove_array_tilt is None:
        remove_array_tilt = _ask_remove_array_tilt()
    if remove_pattern is None:
        remove_pattern = _ask_remove_pattern()
    _compute_phi_chi_LS_fit(
        remove_array_tilt=remove_array_tilt,
        remove_pattern=remove_pattern,
        force_overwrite=force_overwrite,
    )


@check_output_exists(
    lambda remove_array_tilt, remove_pattern: tools.phi_chi_LS_fit_path(
        remove_array_tilt, remove_pattern
    )
)
def _compute_phi_chi_LS_fit(remove_array_tilt, remove_pattern):
    m_data, _, _, F_phi_lm, F_phi_ls, F_chi_lm, F_chi_ls = tools.estimate_F_phi_chi(
        remove_array_tilt=remove_array_tilt, remove_pattern=remove_pattern
    )
    results = tools.fit_F_phi_chis(m_data, F_phi_lm, F_chi_lm, F_phi_ls, F_chi_ls)

    with tools.phi_chi_LS_fit_path(remove_array_tilt, remove_pattern).open("wb") as f:
        pickle.dump(results, f)


def coherence_lengths(remove_array_tilt=None, remove_pattern=None):
    """
    Print a LaTeX table of the observed (Gaussian fit to the mutual coherence) and theoretical (1GM path
    integral) vertical coherence lengths and coherence times for each ray.

    The observed z0 is fitted to the nonzero lags of the unnormalized mutual coherence, as in the
    Ducted_VC figure, and reported ± its leave-one-transmission-out jackknife standard deviation
    (`tools.jackknife_vertical_coherence`).

    The user is asked whether to use the wavefront values with the array tilt and the hydrophone pattern
    removed when `remove_array_tilt` and `remove_pattern` are None.
    """
    if remove_array_tilt is None:
        remove_array_tilt = _ask_remove_array_tilt()
    if remove_pattern is None:
        remove_pattern = _ask_remove_pattern()

    wavefront_values = _load_wavefront_values(remove_array_tilt, remove_pattern)

    print("Estimating observed coherence lengths...")
    observed = tools.estimate_observed_coherence_lengths(wavefront_values)
    print("Computing theoretical (1GM) coherence lengths...")
    theory = tools.compute_theoretical_coherence_lengths()

    table = tools.coherence_lengths_latex_table(observed, theory)
    print("=" * 60)
    print(table)
    print("=" * 60)
    return table


def gm_levels(remove_array_tilt=None, remove_pattern=None):
    """
    Print a LaTeX table of the GM energy levels fitted to the observed ɸ and χ spectra (WFT) and to the
    vertical coherence (path integral), ± their jackknife standard errors.

    The spectral level is read from ``phi_chi_LS_fit[_tilt_removed].pkl`` of the "ɸ χ GM least
    squares fit" step (shown as -- if it has not been made), and its standard error is the linearized
    jackknife `tools.jackknife_F_phi_chi_fit`. The coherence level is fitted here
    (`tools.fit_GM_vertical_coherence`), to D = -2 ln VC of the same wavefront values.

    The user is asked whether to use the wavefront values with the array tilt and the hydrophone pattern
    removed when `remove_array_tilt` and `remove_pattern` are None.
    """
    if remove_array_tilt is None:
        remove_array_tilt = _ask_remove_array_tilt()
    if remove_pattern is None:
        remove_pattern = _ask_remove_pattern()

    wavefront_values = _load_wavefront_values(remove_array_tilt, remove_pattern)

    path = tools.phi_chi_LS_fit_path(remove_array_tilt, remove_pattern)
    if path.exists():
        with path.open("rb") as f:
            results = pickle.load(f)
        print("Jackknifing the ɸ χ GM fit...")
        m_data, F_phi_data, F_chi_data, _, F_phi_ls, _, F_chi_ls = (
            tools.estimate_F_phi_chi(
                remove_array_tilt=remove_array_tilt, remove_pattern=remove_pattern
            )
        )
        spectral = tools.jackknife_F_phi_chi_fit(
            results, m_data, F_phi_data, F_chi_data, F_phi_ls, F_chi_ls
        )
    else:
        print(
            f"{path.name} not found; run 'ɸ χ GM least squares fit' to fill that column."
        )
        spectral = None

    print("Fitting GM to the vertical coherence...")
    coherence = tools.fit_GM_vertical_coherence(wavefront_values)

    table = tools.GM_levels_latex_table(spectral, coherence)
    print("=" * 60)
    print(table)
    print("=" * 60)
    return table


def _load_wavefront_values(remove_array_tilt, remove_pattern="none"):
    wavefront_values_path = tools.wavefront_values_path(
        remove_array_tilt, remove_pattern
    )
    if not wavefront_values_path.exists():
        removed = ["array tilt"] if remove_array_tilt else []
        if remove_pattern != "none":
            removed.append(f"hydrophone {remove_pattern} pattern")
        mode = f" ({', '.join(removed)} removed)" if removed else ""
        raise FileNotFoundError(
            f"Wavefront values{mode} not found at {wavefront_values_path}. "
            "Please run 'compute wavefront values' first."
        )
    with wavefront_values_path.open("rb") as f:
        return pickle.load(f)


def independence_test(remove_array_tilt=None, remove_pattern=None):
    """
    Print the transmission independence test (lag-1 autocorrelation of the per-transmission estimates over the
    hourly transmissions) for the ɸ and χ spectra and for the vertical coherence, the assumption of their
    jackknife error bars.

    The user is asked whether to use the wavefront values with the array tilt and the hydrophone pattern
    removed when `remove_array_tilt` and `remove_pattern` are None.
    """
    if remove_array_tilt is None:
        remove_array_tilt = _ask_remove_array_tilt()
    if remove_pattern is None:
        remove_pattern = _ask_remove_pattern()

    wavefront_values = _load_wavefront_values(remove_array_tilt, remove_pattern)

    print("Testing the ɸ and χ spectra...")
    F_results = tools.independence_test_F_phi_chi(
        remove_array_tilt=remove_array_tilt, remove_pattern=remove_pattern
    )
    print("Testing the vertical coherence...")
    vc_result = tools.independence_test_vertical_coherence(wavefront_values)

    sections = [
        ("F_phi: log10 periodogram, all wavenumbers", F_results["phi"]),
        ("F_chi: log10 periodogram, all wavenumbers", F_results["chi"]),
        ("vertical coherence: R_k(dz), dz <= 180 m", vc_result),
    ]
    lines = []
    for title, result in sections:
        pooled = result["pooled"]
        per_ray = ", ".join(
            f"{ray_id} {r1:+.2f}" for ray_id, r1 in result["per_ray"].items()
        )
        lines += [
            title,
            f"  mean lag-1 autocorrelation per ray: {per_ray}",
            (
                f"  pooled mean r1 = {pooled['mean_r1']:+.3f} (uncorrelated: {pooled['null_mean']:+.3f}), "
                f"|r1| > {pooled['band']:.2f}: {pooled['n_outside']}/{len(pooled['r1'])}, "
                f"permutation p = {pooled['p']:.2f}"
            ),
        ]

    report = "\n".join(lines)
    print("=" * 60)
    print(report)
    print("=" * 60)
    return report


def create_estimation_experiments_config():
    """Write the estimation experiments config template to a path chosen by the user."""
    answer = questionary.path(
        "Where should the config be written?", default="estimation_experiments.yaml"
    ).ask()
    if not answer:
        return None

    dest = Path(answer).expanduser()
    if dest.is_dir():
        dest = dest / "estimation_experiments.yaml"
    if (
        dest.exists()
        and not questionary.confirm(
            f"{dest} already exists. Overwrite it?", default=False
        ).ask()
    ):
        print(f"Kept existing {dest}.")
        return dest

    estimation.write_config_template(dest, overwrite=True)
    print(f"Wrote {dest}. Edit it, then run 'run estimation experiments'.")
    return dest


def run_estimation_experiments():
    """Run the F_phi / F_chi estimation experiments described by a config file."""
    answer = questionary.path(
        "Estimation experiments config:", default="estimation_experiments.yaml"
    ).ask()
    if not answer:
        return None

    try:
        cfg = estimation.load_config(answer)
    except (estimation.ConfigError, FileNotFoundError) as e:
        print(f"Configuration error: {e}")
        return None

    print(f"Writing results to {cfg.output_dir}")
    result = estimation.run_experiment(cfg, progress=lambda msg: print(f"  {msg}"))

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        if result.fits:
            print(
                estimation.fit_table(result)[
                    ["estimator", "ray_id", "GM", "GM_se", "phi_noise", "chi_noise"]
                ].to_string(index=False, float_format="%.3g")
            )
        if result.synthetic is not None:
            print(
                estimation.synthetic_table(result)[
                    [
                        "estimator",
                        "ray_id",
                        "GM_ratio_median",
                        "phi_bias_rms_decades",
                        "chi_bias_rms_decades",
                    ]
                ].to_string(index=False, float_format="%.3g")
            )
    for path in result.written:
        print(f"✓ wrote {path}")
    return result


def create_publication_figures_config():
    """Write the publication figures config template to a path chosen by the user."""
    answer = questionary.path(
        "Where should the config be written?", default="config.yaml"
    ).ask()
    if not answer:
        return None

    dest = Path(answer).expanduser()
    if dest.is_dir():
        dest = dest / "config.yaml"
    if (
        dest.exists()
        and not questionary.confirm(
            f"{dest} already exists. Overwrite it?", default=False
        ).ask()
    ):
        print(f"Kept existing {dest}.")
        return dest

    figures.write_config_template(dest, overwrite=True)
    print(f"Wrote {dest}. Edit it, then run 'generate publication figures'.")
    return dest


def generate_publication_figures():
    """Generate the publication figures described by a config file."""
    answer = questionary.path(
        "Publication figures config:", default="config.yaml"
    ).ask()
    if not answer:
        return None

    try:
        cfg = figures.load_config(answer)
    except (figures.ConfigError, FileNotFoundError) as e:
        print(f"Configuration error: {e}")
        return None

    print(f"Writing figures to {cfg.output_dir}")
    build = figures.generate_publication_figures(
        cfg, progress=lambda name: print(f"  building {name} ...")
    )

    for path in build.written:
        print(f"✓ wrote {path}")
    for name, tb in build.failed.items():
        print(f"\n✗ {name} failed:\n{tb}")
    if build.failed:
        print(f"{len(build.failed)} of {len(cfg.which)} figure(s) failed.")
    return build
