# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Overview

`n23_fluctuations_ragland` analyzes NESCAFE23 acoustic fluctuations: Garrett-Munk (GM) internal-wave
fluctuation theory, ray-path variable computation, phase/log-amplitude spectral density theory and model
fitting, and wavefront/mean-pulse extraction from ray-traced eigenrays.

This repo is a **git submodule** of the NESCAFE23 workspace, checked out at `projects/n23_fluctuations_ragland/`
there. See the workspace root `CLAUDE.md` for the full cross-repo dependency graph.

**Depends on:** `nescafe23` only (verified by repo-wide grep — no `acoustic_inversion` or `nesma_param` usage
anywhere). Also depends directly on external `pygenray` (ray tracing, GitHub install) and `cmps` (colormap
package, GitHub install, used by the panel app).

## Commands

```bash
uv sync
uv run pytest                    # test_cli.py, test_package.py — lightweight, avoid heavy computation
uv run n23_fluctuation_analysis   # interactive CLI: processing pipeline + publication figures
uv run n23_panel                  # (or nescafe23_panel) serves panel_apps/ dashboards
```

## Architecture

- **`src/n23_fluctuations_ragland/tools.py`** — the scientific core:
  - GM internal-wave fluctuation theory; ray-path variable computation (`compute_ray_path_variables`,
    `compute_n23_ray_variables`); phase/log-amplitude spectral density theory (`compute_F_phi_chi`,
    `compute_F_phi_chi_n23`); observational spectra estimation (`estimate_F_phi_chi`); model fitting
    (`fit_F_phi_chis`, `phi_chi_spectrum_model`); ray-fan/eigenray solving (`run_rays`, via `pr.shoot_ray(s)`,
    `pr.find_eigenrays`); wavefront amplitude extraction (`get_wavefront_values`); mean pulse computation;
    mutual coherence estimation.
  - `get_wavefront_values(dss, remove_array_tilt=False)` loads its own picks from
    `nescafe23.open_peak_locations()`. With `remove_array_tilt=True`, each ray is beamformed at
    its transmission-averaged angle plus the lower-array tilt (`estimate_array_tilt` /
    `open_array_tilt`, `array_tilt.pkl`) instead of the per-transmission picked angle. The tilt is
    the mean over rays of the demeaned picked angles. It omits `ARRAY_TILT_OMIT`, the rays with the
    fewest good QC flags.
  - `EIG_INDEXS` — a large hand-picked dict mapping ray IDs to forward/backward eigenray pair indices, built
    via `notebooks/auxiliary/Ray_forward_backward_matching.ipynb`. **Don't regenerate this programmatically**
    without understanding that provenance — it encodes manual matching decisions, not a derivable rule.
- **`src/n23_fluctuations_ragland/cli.py`** — `main()`: a `questionary` menu wrapping the processing pipeline
  (arrival time offset, peak locations, array tilt, wavefront values, mean pulse, ray computation, source geometry sweep,
  φ/χ least-squares fit), each step decorated with `check_output_exists` to skip/overwrite cached outputs.
  Wavefront values, the φ/χ fit and the coherence-lengths table each ask "Remove array tilt?"
  (default yes) and read/write the `*_tilt_removed.pkl` products when it's yes; `run_all_processing()`
  runs both modes of the wavefront values and the fit.
  `run_all_processing()` chains all steps; `serve_panel_apps()` shells out to `panel serve` on `panel_apps/`.
  Two more menu entries drive the publication figures: "create publication figures config" copies
  the template to a `config.yaml`, and "generate publication figures" builds from one.
- **`src/n23_fluctuations_ragland/figures.py`** — the publication figures, one `plot_*` function per
  figure, registered in `FIGURES` in manuscript order. The keys are the output filenames (`Map`,
  `CTD_FCTD`, `2D_slice`, …). Bodies are copied from the notebook cells listed in each docstring.
  Keep them close to the notebook code rather than refactoring. `generate_publication_figures(config)`
  saves each figure to the config's `output_dir`, which is relative to the config file. A failing
  figure is recorded and the rest still build. The config is validated against
  `templates/publication_figures.yaml`, and an unknown key is an error.
  - `Map` and the ellipses/labels on `Acoustic_Receptions` were hand-annotated in the published
    versions. They are recreated in code with positions measured off the reference images, so expect
    them to need nudging if the data or layout changes.
  - Multi-pane figures end with `label_panels(axes, locs)`, which letters the panes `(a)`, `(b)`, …
    The default spot is the upper right. Each figure overrides that where the corner would cover data.
    `Ducted_Intensity_Distribution` is deliberately unlettered.
  - The config's `remove_array_tilt` (default `true`) selects the wavefront values
    (`wavefront_values_tilt_removed.pkl` vs `wavefront_values.pkl`) for `F_phi_chi_ducted`,
    `Ducted_Intensity_Distribution` and `Ducted_VC`, and the matching φ/χ fit
    (`phi_chi_LS_fit_tilt_removed.pkl` vs `phi_chi_LS_fit.pkl`) for `F_phi_chi_ducted`.
    The notebooks still use `remove_array_tilt=False` and `phi_chi_LS_fit.pkl`.
  - `common_arrival_time_shift_estimation` reads `dT_t_upper`/`dT_t_lower` from `T_t_estimated.pkl`.
    Only this project's "compute arrival time offset" writes those keys. The base `nescafe23` CLI
    writes the same file without them.
- **`src/archive.py`** — dead/archived code, excluded from linting; not maintained.
- **`panel_apps/source_geometry_sweep.py`** — standalone Panel dashboard visualizing the source-geometry
  grid-search ray-tracing results; not part of the installable package.

### Notebooks (`notebooks/`)
- `auxiliary/` — exploratory/debugging notebooks (ray matching, FCTD spectra, phase outlier removal).
- `presentation_figures/` and `publication_figures/` — **intentionally near-duplicate notebook pairs** at
  different polish levels (talk vs. paper figures) — not accidental copies, don't try to de-duplicate them.
- `presentation_figures/animations/` — rendered PNG frames + GIFs, committed as binary artifacts.
- Top-level `dev.ipynb` / `exploring_pm6.ipynb` — scratch space.

## Non-obvious conventions

- **Pipeline ordering matters**: the processing steps must be run before `figures.py` or the
  `publication_figures`/`presentation_figures` notebooks, which only load cached pickled/NetCDF data
  products. The notebooks are the original figure source and may drift from the current `tools.py`. For example,
  `compute_phase_structure_function` now returns `(D_dzr, integrand)`, which `Vertical_Coherence.ipynb`
  predates. `figures.py` is the maintained path.
- **Caching**: outputs are guarded by `check_output_exists`. Changing `tools.py` logic has *no effect* on
  downstream notebooks unless you pass `force_overwrite=True` or delete the stale pickle under
  `NESCAFE23_DATADIR/dataproducts/`.
- Data products live under `NESCAFE23_DATADIR/dataproducts/`: pickles (`ray_run_forward_back.pkl`,
  `wavefront_values.pkl`, `wavefront_values_tilt_removed.pkl`, `array_tilt.pkl`, `peak_locations.pkl`, `ray_geo_sweep.pkl`, `phi_chi_LS_fit.pkl`, `phi_chi_LS_fit_tilt_removed.pkl`,
  `T_t_estimated.pkl`) and NetCDF (`mean_pulse_*.nc`). Raw transmissions come from `.mat` files under
  `dmod/`. Figures are saved as PDFs to `figures/`.
- Notebooks are stripped of outputs via `nbstripout` as a git filter — run `nbstripout --install` locally.
- ruff excludes `notebooks/`, `dev`, `panel_apps`, and `src/archive.py`.
- **Known unresolved issues** (per README TODOs): source localization code is intentionally omitted from this
  public repo; backward ray tracing fails to find eigenrays for `+7b`; hardcoded geometry constants in
  `tools.py`/`cli.py` are marked "best guess" and should be treated as provisional, not final truth.
- `cli.py` must import `convert_to_xarray` from `nescafe23` directly, not via `tools` —
  `test_convert_to_xarray_not_on_tools` in `test_cli.py` encodes a real historical bug where this was wrong.
