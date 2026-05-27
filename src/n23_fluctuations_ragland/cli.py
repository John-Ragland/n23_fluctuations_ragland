import pickle
from functools import wraps
from pathlib import Path

import fsspec
import matplotlib.pyplot as plt
import nescafe23 as n23
import numpy as np
import pygenray as pr
import questionary
import xarray as xr
from nescafe23 import convert_to_xarray
from nescafe23.data import get_data_dir
from scipy import signal

from n23_fluctuations_ragland import tools


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
            "compute wavefront values",
            "compute rays",
            "source geometry sweep",
            "ɸ χ GM least squares fit",
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
    elif choice == "compute wavefront values":
        get_wavefront_values()
    elif choice == "compute rays":
        compute_rays()
    elif choice == "source geometry sweep":
        source_geometry_sweep()
    elif choice == "ɸ χ GM least squares fit":
        phi_chi_LS_fit()
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
        ("Step 1: Computing arrival time offset", compute_arrival_time_offset),
        ("Step 2: Estimating peak locations", estimate_peak_locations),
        ("Step 3: Computing wavefront values", get_wavefront_values),
        ("Step 4: Computing mean pulse", compute_mean_pulse),
        ("Step 5: Computing rays", compute_rays),
        ("Step 6: Source geometry sweep", source_geometry_sweep),
        ("Step 7: ɸ χ GM least squares fit", phi_chi_LS_fit),
    ]

    for step_name, step_func in processing_steps:
        print(f"\n{'=' * 60}")
        print(f"{step_name}")
        print(f"{'=' * 60}")
        try:
            step_func(force_overwrite=overwrite_all)
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

    plt.rcParams.update({"font.size": 8})
    plt.rcParams.update({"text.usetex": True})

    fig = plt.figure(figsize=(3.37, 2))
    plt.plot(
        np.arange(1, 24), dT_t_upper, label="upper array", c="k", linestyle="--", lw=1
    )
    plt.plot(
        np.arange(1, 24), dT_t_lower, label="lower array", c="k", linestyle="-.", lw=1
    )
    plt.plot(np.arange(1, 24), dT_avg, c="k", lw=1, label="average")
    plt.ylabel("$\\Delta \\tau$ [s]")
    plt.legend()
    plt.xlabel("transmission number")
    plt.tight_layout()

    figure_path = (
        Path(__file__).parent.parent.parent
        / "figures"
        / "common_arrival_time_shift_estimation.pdf"
    )
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, bbox_inches="tight")

    with Path(T_t_path).open("wb") as f:
        pickle.dump(
            {
                "dT_t": dT_avg,
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
    peak_locations = {"times": times, "angles": angles, "snrs": snrs}

    with (data_dir / "dataproducts" / "peak_locations.pkl").open("wb") as f:
        pickle.dump(peak_locations, f)


@check_output_exists(lambda: get_data_dir() / "dataproducts" / "wavefront_values.pkl")
def get_wavefront_values():
    """
    Given the estimated peak locations, extract the complex pressure values of the wavefront at these time locations.
    """
    data_dir = get_data_dir()
    dss = tools.open_receptions()

    peak_locations_path = data_dir / "dataproducts" / "peak_locations.pkl"
    if not peak_locations_path.exists():
        raise FileNotFoundError(
            "Peak locations not found. Please run 'estimate peak locations' first."
        )

    with peak_locations_path.open("rb") as f:
        peak_locations = pickle.load(f)

    times = peak_locations["times"]
    angles = peak_locations["angles"]

    wavefront_values = tools.get_wavefront_values(dss, times, angles)

    with (data_dir / "dataproducts" / "wavefront_values.pkl").open("wb") as f:
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


@check_output_exists(lambda: get_data_dir() / "dataproducts" / "phi_chi_LS_fit.pkl")
def phi_chi_LS_fit():
    data_dir = get_data_dir()
    m_data, _, _, F_phi_lm, F_phi_ls, F_chi_lm, F_chi_ls = tools.estimate_F_phi_chi()
    results = tools.fit_F_phi_chis(m_data, F_phi_lm, F_chi_lm, F_phi_ls, F_chi_ls)

    with (data_dir / "dataproducts" / "phi_chi_LS_fit.pkl").open("wb") as f:
        pickle.dump(results, f)
