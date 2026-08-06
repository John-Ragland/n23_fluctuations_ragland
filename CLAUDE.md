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
uv run nescafe23-processing       # interactive CLI: runs the processing pipeline steps
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
  - `EIG_INDEXS` — a large hand-picked dict mapping ray IDs to forward/backward eigenray pair indices, built
    via `notebooks/auxiliary/Ray_forward_backward_matching.ipynb`. **Don't regenerate this programmatically**
    without understanding that provenance — it encodes manual matching decisions, not a derivable rule.
- **`src/n23_fluctuations_ragland/cli.py`** — `main()`: a `questionary` menu wrapping the processing pipeline
  (arrival time offset, peak locations, wavefront values, mean pulse, ray computation, source geometry sweep,
  φ/χ least-squares fit), each step decorated with `check_output_exists` to skip/overwrite cached outputs.
  `run_all_processing()` chains all steps; `serve_panel_apps()` shells out to `panel serve` on `panel_apps/`.
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

- **Pipeline ordering matters**: the `nescafe23-processing` CLI must be run before the `publication_figures`/
  `presentation_figures` notebooks, which only load cached pickled/NetCDF data products — the notebooks are
  not self-contained.
- **Caching**: outputs are guarded by `check_output_exists`. Changing `tools.py` logic has *no effect* on
  downstream notebooks unless you pass `force_overwrite=True` or delete the stale pickle under
  `NESCAFE23_DATADIR/dataproducts/`.
- Data products live under `NESCAFE23_DATADIR/dataproducts/`: pickles (`ray_run_forward_back.pkl`,
  `wavefront_values.pkl`, `peak_locations.pkl`, `ray_geo_sweep.pkl`, `phi_chi_LS_fit.pkl`,
  `T_t_estimated.pkl`) and NetCDF (`mean_pulse_*.nc`). Raw transmissions come from `.mat` files under
  `dmod/`. Figures are saved as PDFs to `figures/`.
- Notebooks are stripped of outputs via `nbstripout` as a git filter — run `nbstripout --install` locally.
- ruff excludes `notebooks/`, `dev`, `panel_apps`, and `src/archive.py`.
- **Known unresolved issues** (per README TODOs): source localization code is intentionally omitted from this
  public repo; backward ray tracing fails to find eigenrays for `+7b`; hardcoded geometry constants in
  `tools.py`/`cli.py` are marked "best guess" and should be treated as provisional, not final truth.
- `cli.py` must import `convert_to_xarray` from `nescafe23` directly, not via `tools` —
  `test_convert_to_xarray_not_on_tools` in `test_cli.py` encodes a real historical bug where this was wrong.
