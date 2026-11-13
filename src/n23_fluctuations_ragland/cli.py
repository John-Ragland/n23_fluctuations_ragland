"""
``n23_fluctuation_analysis``: the analysis of the NESCAFE23 acoustic fluctuations paper.

The menu holds only the analysis that is in the paper. Its first entry, "reproduce paper", runs every
processing step, writes Table II and builds every publication figure in one go
(also ``n23_fluctuation_analysis --reproduce-paper``). The remaining entries run the steps one at a time.

The processing starts from the demodulated receptions (``$NESCAFE23_DATADIR/dmod/dm*.mat``), which are not
public: they are available from the corresponding author on request. `check_paper_inputs` lists every input
the reproduction needs.

Exploratory analysis that is not in the paper (the GM + noise fit of the spectra, the GM levels table, the
transmission independence test, the estimation experiments, the source geometry sweep, the theory
environments) is not in this menu. Its code is in `tools`, `estimation` and `theory_environments`.
"""

import argparse
import pickle
import shutil
import subprocess
import time
import traceback
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path

import fsspec
import matplotlib.pyplot as plt
import nescafe23 as n23
import numpy as np
import questionary
from nescafe23 import convert_to_xarray
from nescafe23.data import get_data_dir
from scipy import signal
from tqdm import tqdm

from n23_fluctuations_ragland import __version__, figures, tools

N_TRANSMISSIONS = 23

DATA_ON_REQUEST = (
    "The demodulated receptions (dmod/dm*.mat) are not public. They are available from the corresponding "
    "author on reasonable request (see the paper's Data Availability statement)."
)


def check_output_exists(*output_paths):
    """
    Decorator to check if output file(s) exist and handle overwrite logic.

    The decorated function gets two keyword arguments:

    - ``force_overwrite`` (default False): run even if the output exists;
    - ``skip_existing`` (default False): if any output exists, reuse it and skip the function without asking.

    With neither set and an existing output, the user is asked whether to overwrite it.

    Parameters
    ----------
    *output_paths : str or callable
        One or more paths to output files to check, or callables that take
        the same arguments as the decorated function and return paths.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, force_overwrite=False, skip_existing=False, **kwargs):
            filepaths = []
            for output_path in output_paths:
                if callable(output_path):
                    filepath = output_path(*args, **kwargs)
                else:
                    filepath = output_path
                filepaths.append(Path(filepath))

            existing_files = [fp for fp in filepaths if fp.exists()]

            if not force_overwrite and existing_files:
                if skip_existing:
                    print(f"Skipping {func.__name__}: reusing existing output.")
                    return None

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


MENU_REPRODUCE = "reproduce paper (all analysis, Table II and figures)"
MENU_CHOICES = (
    MENU_REPRODUCE,
    "compute arrival time offset",
    "estimate arrival time offset error",
    "compute mean pulse",
    "estimate peak locations",
    "estimate array tilt",
    "compute wavefront values and hydrophone pattern",
    "compute rays",
    "Table II: vertical coherence lengths",
    "create publication figures config",
    "generate publication figures",
    "exit",
)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="n23_fluctuation_analysis",
        description=(
            "Analysis of the NESCAFE23 acoustic fluctuations paper. Without arguments, an interactive menu."
        ),
    )
    parser.add_argument(
        "--reproduce-paper",
        action="store_true",
        help="run all the paper's analysis, write Table II and build every figure, without prompts",
    )
    parser.add_argument(
        "--output",
        default="paper_output",
        help="directory for Table II, the figures and the report (default: paper_output)",
    )
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="reuse existing data products instead of recomputing them",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Interactive CLI menu, or ``--reproduce-paper`` without prompts. Returns the exit code."""
    args = _parse_args(argv)
    if args.reproduce_paper:
        return 0 if reproduce_paper(args.output, reuse=args.reuse) else 1

    try:
        get_data_dir()
    except OSError:
        print("\nNESCAFE23_DATADIR is not set.")
        print("Run 'nescafe23-init' to configure the data directory, then try again.")
        return 1

    choice = questionary.select(
        "Which script would you like to run?", choices=list(MENU_CHOICES)
    ).ask()

    if choice == MENU_REPRODUCE:
        output_dir = questionary.path(
            "Output directory for Table II, the figures and the report:",
            default="paper_output",
        ).ask()
        if not output_dir:
            return 1
        reuse = (
            questionary.select(
                "Existing data products:",
                choices=["recompute everything", "reuse existing data products"],
                default="recompute everything",
            ).ask()
            == "reuse existing data products"
        )
        return 0 if reproduce_paper(output_dir, reuse=reuse) else 1
    if choice == "compute arrival time offset":
        compute_arrival_time_offset()
    elif choice == "estimate arrival time offset error":
        estimate_arrival_time_offset_error()
    elif choice == "compute mean pulse":
        compute_mean_pulse()
    elif choice == "estimate peak locations":
        estimate_peak_locations()
    elif choice == "estimate array tilt":
        estimate_array_tilt()
    elif choice == "compute wavefront values and hydrophone pattern":
        compute_wavefront_values_and_pattern()
    elif choice == "compute rays":
        compute_rays()
    elif choice == "Table II: vertical coherence lengths":
        coherence_lengths()
    elif choice == "create publication figures config":
        create_publication_figures_config()
    elif choice == "generate publication figures":
        generate_publication_figures()
    elif choice == "exit":
        print("Goodbye!")
    return 0


# ---------------------------------------------------------------------------
# One-click reproduction
# ---------------------------------------------------------------------------


@dataclass
class PaperInputs:
    """Outcome of `check_paper_inputs`: missing required inputs, and warnings that don't stop the run."""

    missing: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    has_latex: bool = True

    @property
    def ok(self):
        return not self.missing


def check_paper_inputs(cfg=None):
    """
    Check that every input of the paper's analysis is present, before anything is computed.

    Required: ``NESCAFE23_DATADIR``; the 23 receptions ``dmod/dm*.mat``; under ``environment_data/`` the FCTD
    section, the SION and SIOE CTD casts, and the map bathymetry and SSH files (`figures.map_input_paths`).
    Warnings: a missing map font (the default font is used), and no ``latex`` on the PATH (the figures are then
    built with ``usetex: false``).

    Parameters
    ----------
    cfg : figures.PublicationFiguresConfig, optional
        config whose map inputs are checked. Default is the default (paper) config.
    """
    result = PaperInputs()
    try:
        data_dir = get_data_dir()
    except OSError:
        result.missing.append(
            "NESCAFE23_DATADIR is not set (set it in the workspace .env, e.g. with `nescafe23`)."
        )
        return result
    if not data_dir.is_dir():
        result.missing.append(f"NESCAFE23_DATADIR {data_dir} is not a directory.")
        return result

    dmod = sorted((data_dir / "dmod").glob("dm*.mat"))
    if len(dmod) != N_TRANSMISSIONS:
        result.missing.append(
            f"{data_dir / 'dmod' / 'dm*.mat'}: expected {N_TRANSMISSIONS} receptions, found {len(dmod)}. "
            + DATA_ON_REQUEST
        )

    cfg = cfg or figures.PublicationFiguresConfig()
    environment = data_dir / "environment_data"
    map_paths = figures.map_input_paths(cfg)
    required = {
        "FCTD section": environment / "FCTD_SoundSpeed_Colosi_v2.mat",
        "SION CTD cast": environment / "SION_CTD_new.mat",
        "SIOE CTD cast": environment / "SIOE_CTD.mat",
        "map bathymetry": map_paths["bathymetry"],
        "sea surface height": map_paths["ssh"],
    }
    for name, path in required.items():
        if not path.exists():
            result.missing.append(f"{name}: {path}")

    if not map_paths["font"].exists():
        result.warnings.append(
            f"map font {map_paths['font']} not found; the Map uses the default font."
        )
    result.has_latex = shutil.which("latex") is not None
    if not result.has_latex:
        result.warnings.append(
            "latex not found on the PATH; the figures are built with usetex: false."
        )
    return result


def _git_commit():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            check=True,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip() + (" (uncommitted changes)" if dirty else "")


def _load_dataproduct(name):
    path = get_data_dir() / "dataproducts" / name
    if not path.exists():
        return None
    with path.open("rb") as f:
        return pickle.load(f)


def paper_steps(output_dir):
    """The paper's analysis, in order, as ``(name, function)``. Each function takes the overwrite keywords of
    `check_output_exists` (``force_overwrite``, ``skip_existing``); Table II and the figures always rerun."""
    output_dir = Path(output_dir)

    def table_ii(**_):
        coherence_lengths(output_path=output_dir / "table_II.tex")

    def publication_figures(**_):
        build_paper_figures(output_dir / "figures")

    return [
        ("arrival time offset", compute_arrival_time_offset),
        ("arrival time offset error (jackknife)", estimate_arrival_time_offset_error),
        ("mean pulse", compute_mean_pulse),
        ("peak locations", estimate_peak_locations),
        ("array tilt", estimate_array_tilt),
        (
            "wavefront values and hydrophone pattern",
            compute_wavefront_values_and_pattern,
        ),
        ("rays", compute_rays),
        ("Table II", table_ii),
        ("publication figures", publication_figures),
    ]


def build_paper_figures(output_dir, usetex=None):
    """Build every publication figure with the default (paper) config into ``output_dir``. Raises if any figure
    fails. ``usetex`` defaults to whether ``latex`` is on the PATH."""
    if usetex is None:
        usetex = shutil.which("latex") is not None
    cfg = figures.PublicationFiguresConfig(output_dir=Path(output_dir), usetex=usetex)
    figures.clear_cache()
    build = figures.generate_publication_figures(
        cfg, progress=lambda name: print(f"  building {name} ...")
    )
    for path in build.written:
        print(f"✓ wrote {path}")
    if build.failed:
        details = "\n".join(f"{name}:\n{tb}" for name, tb in build.failed.items())
        msg = f"{len(build.failed)} of {len(cfg.which)} figure(s) failed:\n{details}"
        raise RuntimeError(msg)
    return build


def reproduce_paper(output_dir="paper_output", reuse=False, steps=None):
    """
    Reproduce the paper: every processing step, Table II and every publication figure, without prompts.

    The inputs are checked first (`check_paper_inputs`); if any is missing, nothing is run. The steps
    (`paper_steps`) then run in order and stop at the first failure, since each one needs the products of
    the previous ones. The data products go to ``$NESCAFE23_DATADIR/dataproducts`` as usual; Table II
    (``table_II.tex``), the figures (``figures/``) and a report (``reproduction_report.txt``) go to
    ``output_dir``.

    Parameters
    ----------
    output_dir : str or Path
        directory for Table II, the figures and the report
    reuse : bool
        if True, existing data products are reused; otherwise every product is recomputed
    steps : list of (str, callable), optional
        the steps to run. Default is `paper_steps` (for tests).

    Returns
    -------
    success : bool
    """
    output_dir = Path(output_dir).expanduser()
    print("=" * 60)
    print("Reproducing the NESCAFE23 acoustic fluctuations paper")
    print("=" * 60)

    inputs = check_paper_inputs()
    for warning in inputs.warnings:
        print(f"! {warning}")
    if not inputs.ok:
        print("\nCannot reproduce the paper; these inputs are missing:")
        for item in inputs.missing:
            print(f"  ✗ {item}")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    if steps is None:
        steps = paper_steps(output_dir)
    overwrite = {"skip_existing": True} if reuse else {"force_overwrite": True}

    status = []
    success = True
    for k, (name, func) in enumerate(steps, start=1):
        print(f"\n{'=' * 60}\nStep {k}/{len(steps)}: {name}\n{'=' * 60}")
        start = time.perf_counter()
        try:
            func(**overwrite)
        except Exception:
            elapsed = time.perf_counter() - start
            tb = traceback.format_exc()
            print(f"✗ {name} failed after {elapsed:.0f} s:\n{tb}")
            status.append((name, "FAILED", elapsed, tb))
            success = False
            break
        elapsed = time.perf_counter() - start
        print(f"✓ {name} ({elapsed:.0f} s)")
        status.append((name, "ok", elapsed, None))

    report = reproduction_report(status, steps, reuse, inputs)
    report_path = output_dir / "reproduction_report.txt"
    report_path.write_text(report)
    print(f"\n{report}\nWrote {report_path}")
    return success


def reproduction_report(status, steps, reuse, inputs):
    """Text report of a `reproduce_paper` run: versions, settings, step status and times, and the numbers the
    paper quotes from the data products (the Fig. 5 caption's arrival time offset errors)."""
    lines = [
        "NESCAFE23 acoustic fluctuations: paper reproduction report",
        f"date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"n23_fluctuations_ragland {__version__}, commit {_git_commit()}",
        f"NESCAFE23_DATADIR: {get_data_dir()}",
        f"data products: {'reused where present' if reuse else 'recomputed'}",
        f"settings (tools.PAPER_SETTINGS): {tools.PAPER_SETTINGS}",
        f"figures usetex: {inputs.has_latex}",
        *(f"warning: {w}" for w in inputs.warnings),
        "",
        "steps:",
    ]
    for name, state, elapsed, _ in status:
        lines.append(f"  {state:>6}  {name} ({elapsed:.0f} s)")
    for name, _ in steps[len(status) :]:
        lines.append(f"  {'-':>6}  {name} (not run)")
    for name, state, _, tb in status:
        if state == "FAILED":
            lines += ["", f"{name} failed:", tb]

    offset = _load_dataproduct("T_t_jackknife.pkl")
    if offset is not None:
        lines += [
            "",
            "Fig. 5 caption:",
            "  " + arrival_time_offset_error_report(offset),
        ]
    pattern = _load_dataproduct("hydrophone_pattern.pkl")
    if pattern is not None:
        lines += [
            "",
            "hydrophone pattern (Fig. 5b):",
            hydrophone_pattern_report(pattern),
        ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Processing steps
# ---------------------------------------------------------------------------


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


@check_output_exists(lambda: get_data_dir() / "dataproducts" / "T_t_jackknife.pkl")
def estimate_arrival_time_offset_error():
    """
    Leave-one-hydrophone-out jackknife standard error of the arrival time offset
    (`tools.jackknife_arrival_time_offset`), written to ``T_t_jackknife.pkl``.

    The offset there is refined to sub-sample lags, so it differs slightly (~0.2 ms) from the one in
    ``T_t_estimated.pkl``, which is left unchanged.
    """
    data_dir = get_data_dir()
    fns = sorted(fsspec.filesystem("").glob(str(data_dir / "dmod" / "dm*.mat")))
    receptions = [convert_to_xarray(fn)["dmod"] for fn in fns]

    with tqdm(
        total=receptions[0].sizes["hydrophone"], desc="leave-one-hydrophone-out"
    ) as bar:
        offset = tools.jackknife_arrival_time_offset(receptions, progress=bar.update)

    path = data_dir / "dataproducts" / "T_t_jackknife.pkl"
    with path.open("wb") as f:
        pickle.dump(offset, f)

    print(arrival_time_offset_error_report(offset))
    print(f"Wrote {path}")
    return offset


def arrival_time_offset_error_report(offset):
    """One-line summary of the ``dT_t`` jackknife SEs of `tools.jackknife_arrival_time_offset`."""
    se = 1e3 * np.asarray(offset["dT_t_se"])
    return (
        f"dT_t jackknife SE over {len(se)} transmissions: mean {se.mean():.2f} ms, "
        f"median {np.median(se):.2f} ms, max {se.max():.2f} ms "
        f"(transmission {int(np.argmax(se)) + 1})"
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


@check_output_exists(lambda: get_data_dir() / "dataproducts" / "array_tilt.pkl")
def estimate_array_tilt():
    """
    Estimate the lower-array tilt from the estimated peak locations, see `tools.estimate_array_tilt`.
    """
    data_dir = get_data_dir()
    array_tilt = tools.estimate_array_tilt(n23.open_peak_locations())

    with (data_dir / "dataproducts" / "array_tilt.pkl").open("wb") as f:
        pickle.dump(array_tilt, f)


def compute_wavefront_values_and_pattern(force_overwrite=False, skip_existing=False):
    """
    The paper's wavefront values, in the order they depend on each other:

    1. wavefront values with the array tilt removed (``wavefront_values_tilt_removed.pkl``), from which
    2. the hydrophone pattern is estimated (``hydrophone_pattern.pkl``), which is removed (phase and
       magnitude) from the receptions before beamforming the
    3. wavefront values the paper analyzes (`tools.PAPER_SETTINGS`,
       ``wavefront_values_tilt_removed_phase_magnitude_removed.pkl``).
    """
    overwrite = {"force_overwrite": force_overwrite, "skip_existing": skip_existing}
    _compute_wavefront_values(
        remove_array_tilt=True, remove_pattern="none", **overwrite
    )
    estimate_hydrophone_pattern(**overwrite)
    _compute_wavefront_values(
        remove_array_tilt=tools.PAPER_SETTINGS["remove_array_tilt"],
        remove_pattern=tools.PAPER_SETTINGS["remove_pattern"],
        **overwrite,
    )


@check_output_exists(
    lambda remove_array_tilt, remove_pattern: tools.wavefront_values_path(
        remove_array_tilt, remove_pattern
    )
)
def _compute_wavefront_values(remove_array_tilt, remove_pattern):
    """Extract the complex pressure of each arrival's wavefront at its picked time (`tools.get_wavefront_values`)."""
    dss = tools.open_receptions()
    wavefront_values = tools.get_wavefront_values(
        dss, remove_array_tilt=remove_array_tilt, remove_pattern=remove_pattern
    )

    with tools.wavefront_values_path(remove_array_tilt, remove_pattern).open("wb") as f:
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


def coherence_lengths(output_path=None):
    """
    Table II: print a LaTeX table of the observed (Gaussian fit to the mutual coherence) and predicted (1GM path
    integral) vertical coherence lengths, and the observed fit amplitude A0, for the ducted rays
    (`tools.COHERENCE_TABLE_ROWS`: the +7 observation spans the -8 and +7 predictions, and the -6
    observation its three eigenray paths), from the paper's wavefront values (`tools.PAPER_SETTINGS`).

    The observed z0 is fitted to the nonzero lags of the unnormalized mutual coherence, as in the
    Ducted_VC figure, and reported ± its leave-one-transmission-out jackknife standard deviation
    (`tools.jackknife_vertical_coherence`). Only the ducted rays (`tools.DUCTED_RAY_IDS`) are fitted: the
    fit raises on rays such as ``-6b`` whose coherence is negative at the first nonzero lag.

    Parameters
    ----------
    output_path : str or Path, optional
        if given, the table is also written there
    """
    wavefront_values = _load_wavefront_values(
        tools.PAPER_SETTINGS["remove_array_tilt"],
        tools.PAPER_SETTINGS["remove_pattern"],
    )

    print("Estimating observed coherence lengths...")
    observed = tools.estimate_observed_coherence_lengths(
        {ray_id: wavefront_values[ray_id] for ray_id in tools.DUCTED_RAY_IDS}
    )
    print("Computing theoretical (1GM) coherence lengths...")
    theory = tools.compute_theoretical_coherence_lengths()

    table = tools.coherence_lengths_latex_table(observed, theory)
    print("=" * 60)
    print(table)
    print("=" * 60)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(table + "\n")
        print(f"Wrote {output_path}")
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
            "Please run 'compute wavefront values and hydrophone pattern' first."
        )
    with wavefront_values_path.open("rb") as f:
        return pickle.load(f)


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
