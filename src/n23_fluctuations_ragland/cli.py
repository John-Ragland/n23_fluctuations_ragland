import pickle
from functools import wraps
from pathlib import Path

import fsspec
import nescafe23 as n23
import numpy as np
import pygenray as pr
import questionary
import xarray as xr
from nescafe23 import convert_to_xarray
from nescafe23.data import get_data_dir
from scipy import signal

from n23_fluctuations_ragland import figures, tools


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
            "compute rays",
            "source geometry sweep",
            "ɸ χ GM least squares fit",
            "coherence lengths (observed vs. GM)",
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
    elif choice == "compute rays":
        compute_rays()
    elif choice == "source geometry sweep":
        source_geometry_sweep()
    elif choice == "ɸ χ GM least squares fit":
        phi_chi_LS_fit()
    elif choice == "coherence lengths (observed vs. GM)":
        coherence_lengths()
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
            {"remove_array_tilt": False},
        ),
        (
            "Step 5: Computing wavefront values (array tilt removed)",
            get_wavefront_values,
            {"remove_array_tilt": True},
        ),
        ("Step 6: Computing mean pulse", compute_mean_pulse, {}),
        ("Step 7: Computing rays", compute_rays, {}),
        ("Step 8: Source geometry sweep", source_geometry_sweep, {}),
        (
            "Step 9: ɸ χ GM least squares fit",
            phi_chi_LS_fit,
            {"remove_array_tilt": False},
        ),
        (
            "Step 10: ɸ χ GM least squares fit (array tilt removed)",
            phi_chi_LS_fit,
            {"remove_array_tilt": True},
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
        interp1 = dss_interp_up[0].values - dss_interp_up[0].mean().values
        interp2 = dss_interp_up[k].values - dss_interp_up[k].mean().values

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
        interp1 = dss_interp_down[0].values - dss_interp_down[0].mean().values
        interp2 = dss_interp_down[k].values - dss_interp_down[k].mean().values

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


def _wavefront_values_path(remove_array_tilt):
    return tools.wavefront_values_path(remove_array_tilt)


def get_wavefront_values(remove_array_tilt=None, force_overwrite=False):
    """
    Given the estimated peak locations, extract the complex pressure values of the wavefront at these time locations.

    With the array tilt removed, each ray is beamformed at its mean angle plus the array tilt and the
    result is written to ``wavefront_values_tilt_removed.pkl`` instead of ``wavefront_values.pkl``.
    The user is asked whether to remove the tilt when `remove_array_tilt` is None.
    """
    if remove_array_tilt is None:
        remove_array_tilt = _ask_remove_array_tilt()
    _compute_wavefront_values(
        remove_array_tilt=remove_array_tilt, force_overwrite=force_overwrite
    )


@check_output_exists(
    lambda remove_array_tilt: _wavefront_values_path(remove_array_tilt)
)
def _compute_wavefront_values(remove_array_tilt):
    dss = tools.open_receptions()
    wavefront_values = tools.get_wavefront_values(
        dss, remove_array_tilt=remove_array_tilt
    )

    with _wavefront_values_path(remove_array_tilt).open("wb") as f:
        pickle.dump(wavefront_values, f)


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
    fctd = fctd.assign_coords({"range": (fctd.range.values + front_location)})

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


def phi_chi_LS_fit(remove_array_tilt=None, force_overwrite=False):
    """
    Fit the GM energy and observational noise levels to the observed ɸ and χ spectra.

    The spectra are estimated from ``wavefront_values_tilt_removed.pkl`` with the array tilt removed
    and the fit is written to ``phi_chi_LS_fit_tilt_removed.pkl``, otherwise from
    ``wavefront_values.pkl`` to ``phi_chi_LS_fit.pkl``. The user is asked whether to remove the tilt
    when `remove_array_tilt` is None.
    """
    if remove_array_tilt is None:
        remove_array_tilt = _ask_remove_array_tilt()
    _compute_phi_chi_LS_fit(
        remove_array_tilt=remove_array_tilt, force_overwrite=force_overwrite
    )


@check_output_exists(
    lambda remove_array_tilt: tools.phi_chi_LS_fit_path(remove_array_tilt)
)
def _compute_phi_chi_LS_fit(remove_array_tilt):
    m_data, _, _, F_phi_lm, F_phi_ls, F_chi_lm, F_chi_ls = tools.estimate_F_phi_chi(
        remove_array_tilt=remove_array_tilt
    )
    results = tools.fit_F_phi_chis(m_data, F_phi_lm, F_chi_lm, F_phi_ls, F_chi_ls)

    with tools.phi_chi_LS_fit_path(remove_array_tilt).open("wb") as f:
        pickle.dump(results, f)


def coherence_lengths(remove_array_tilt=None):
    """
    Print a LaTeX table of the observed (Gaussian fit to the mutual coherence) and theoretical (1GM path
    integral) vertical coherence lengths and coherence times for each ray.

    The user is asked whether to use the wavefront values with the array tilt removed when
    `remove_array_tilt` is None.
    """
    if remove_array_tilt is None:
        remove_array_tilt = _ask_remove_array_tilt()

    wavefront_values_path = tools.wavefront_values_path(remove_array_tilt)
    if not wavefront_values_path.exists():
        mode = " (array tilt removed)" if remove_array_tilt else ""
        raise FileNotFoundError(
            f"Wavefront values{mode} not found at {wavefront_values_path}. "
            "Please run 'compute wavefront values' first."
        )

    with wavefront_values_path.open("rb") as f:
        wavefront_values = pickle.load(f)

    print("Estimating observed coherence lengths...")
    observed = tools.estimate_observed_coherence_lengths(wavefront_values)
    print("Computing theoretical (1GM) coherence lengths...")
    theory = tools.compute_theoretical_coherence_lengths()

    table = tools.coherence_lengths_latex_table(observed, theory)
    print("=" * 60)
    print(table)
    print("=" * 60)
    return table


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
