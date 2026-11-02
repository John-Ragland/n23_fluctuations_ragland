# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Overview

`n23_fluctuations_ragland` analyzes NESCAFE23 acoustic fluctuations: Garrett-Munk (GM) internal-wave
fluctuation theory, ray-path variable computation, phase/log-amplitude spectral density theory and model
fitting, and wavefront/mean-pulse extraction from ray-traced eigenrays.

This repo is a **git submodule** of the NESCAFE23 workspace, checked out at `projects/n23_fluctuations_ragland/`
there. See the workspace root `CLAUDE.md` for the full cross-repo dependency graph.

**Depends on:** `nescafe23` (no `acoustic_inversion` usage). `nesma_param` is an **optional** extra
(`[nesma]`), imported lazily only by `theory_environments.nesma_beta0_environment`. Also depends directly on external `pygenray` (ray tracing, GitHub install) and `cmps` (colormap
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
  - `get_wavefront_values(..., remove_pattern=...)` also takes `"none"`, `"phase"`, `"magnitude"` or `"both"`
    (`REMOVE_PATTERN_CHOICES`). This removes the per-hydrophone pattern (`open_hydrophone_pattern`,
    `hydrophone_pattern.pkl`) from each receptions window before beamforming (`remove_hydrophone_pattern`).
    The phase is removed as a **time delay** τ_n = φ_n/(2π·250 Hz), applied like the beamformer's steering
    delays (exp(i(ω+ω₀)d) with d = −τ_n), so it is e^{−iφ_n} only at the carrier. The magnitude is removed
    by dividing by the gain g_n. `wavefront_values_path(tilt, pattern)` and `phi_chi_LS_fit_path` add
    `_phase_removed` / `_magnitude_removed` / `_phase_magnitude_removed`. `"none"` keeps the old names.
    With the phase removed, `estimate_F_phi_chi(remove_pattern=...)` uses all 28 elements and does not
    zero-fill 4, 5 and 20.
  - `estimate_hydrophone_pattern(wavefront_values)` estimates the pattern. It reads the tilt-removed,
    unpatterned wavefront values of every lower arrival except `ARRAY_TILT_OMIT`, ducted and bottom-bounced,
    pooled. Phase uses the phasor-step method of `estimation.circular_common_phase` and is then linearly
    detrended, since a line is a beam-angle error. Gain is the mean of |X| over its record's array mean,
    normalized to unit mean. SEs are a leave-one-transmission-out jackknife. Outliers are rejected
    automatically, in place of visual QC, separately for phase and gain:
    (1) an iterated per-arrival test against the pooled pattern of the other arrivals, using
    per-element Welch t statistics mapped to normal scores, summed as χ², Bonferroni α = 0.01
    (`_pattern_difference_p_value`). Plain d²/var on the noisy jackknife variances rejected on 10–19% of
    clean synthetic runs. The calibrated test rejects on 2–4%.
    (2) robust z (MAD) > 3.5 on each record's rms residual.
    On the real data (2026-10-30) elements 4, 5 and 20 come out at −0.87/−0.59/−1.11 ± 0.06 rad.
    The phase test rejects `+7`, `-6`, `+7b`, `+9b`, `+13b` and `+15b`, which mostly differ by a smooth
    bowl (curvature) along the array, not at single elements.
  - `jackknife_vertical_coherence(X, dz)` is the observed vertical coherence used by `Ducted_VC`
    and the CLI coherence-lengths table. It takes the unnormalized mutual coherence, divides it by the
    amplitude A0 of a Gaussian fit to the nonzero lags (`fit_vertical_coherence`), and adds
    leave-one-transmission-out jackknife variances. The error bars jackknife the *ratio* MC_k/A0_k,
    refitting A0 in each sample. That deliberately differs from `Vertical_Coherence.ipynb` cell 9,
    which divides Var[MC_k] by the full-data A0². Because MC_k and A0_k share the per-transmission
    intensity fluctuation, the ratio's bars are several times smaller at short lags.
    Since 2026-11-02 `fit_vertical_coherence` starts from `A0_init = MC[1]` and fits in scaled parameters
    (z0/L0, A0/A0_init), and it **raises** `ValueError` instead of returning NaN or a silently bad fit.
    The old fixed start (A0 = 4e11) plus `x_scale="jac"` made `least_squares` stop on `xtol` at the initial
    guess while reporting success. That happened on 12/23 leave-one-out refits of `+7` with the pattern
    removed, and it sent the `Ducted_VC` (a) error bars to ±27. Other fits stopped short of the minimum.
    Rays whose MC[1] ≤ 0 (`-6b`, every product) can't be fitted, so the CLI coherence-lengths table fits
    only `DUCTED_RAY_IDS`.
  - `estimate_F_phi_chi` returns `F_*_lm`, the average over transmissions of the per-transmission
    periodograms. Since 2026-11-02 the default (`average="mean"`, `_jackknife_linear_mean`) is the
    **arithmetic mean**. `average="log_mean"` gives the old log-mean with its χ²_ν bias removed
    (`_jackknife_log_mean`; a raw log-mean of χ²₂ bins is low by γ/ln10 = 0.251 decades, 0.552 at the
    1-dof Nyquist bin). The log-mean estimates the *geometric* mean over transmissions, and its correction
    is exact only where every bin is χ²_ν around one common spectrum. On the real data the arithmetic mean
    is higher, by 0.1–0.7 decades for χ (largest for `-7` with Hann) and by up to ~0.3 decades in the
    low φ bins, because the spectrum varies between transmissions. `estimation.Estimator.average`
    defaults to `"mean"` to match. It also
    returns `F_*_ls`, the leave-one-transmission-out jackknife standard error of log10 of that mean (Efron &
    Tibshirani 1993, Ch. 11), as a multiplicative factor. The fit weights (`fit_F_phi_chis`) and the
    `F_phi_chi_ducted` bands both use it. The bands are the SE, not the single-periodogram spread,
    because the spread is ~0.56 decades everywhere and only reflects χ²₂ sampling noise. **`phi_chi_LS_fit*.pkl`
    files made before the bias correction are biased low**, and files made before 2026-11-02 are fits to
    the log-mean (the file names don't encode the average). Regenerate both with `force_overwrite=True`.
  - `estimate_F_phi_chi` tapers each depth record with a periodic window scaled to unit mean square
    (`_taper_window`) after the mean removal and the zero-fill of elements 4, 5, 20. `window` is one of
    `SPECTRAL_WINDOWS`: `"hann"` (default), `"tukey"` (α = `TUKEY_ALPHA` = 0.3, fixed: flat over the
    middle 70%, so it down-weights the array ends less than Hann but leaks more) or `"rectangular"` (the
    old untapered periodogram). Before 2026-11-02 this was a boolean `hann`. Untapered, the very red phase spectrum leaks into every bin and sets a
    floor near 1–3 rad² m that hides the spectrum at high wavenumbers. The taper keeps white-noise levels
    (`tests/test_tools.py`), but down-weights variance at scales longer than the array, which sits at the
    array ends, so total power drops (to about 45–70% for ducted φ). The fit still compares against the
    analytic model, not the model passed through the same taper; that was a deliberate choice, because with
    the taper the two differ little in the resolved bins. Fits made before 2026-10-23 used the untapered
    spectra.
  - `estimate_F_phi_chi(first_difference=True)` prewhitens: mean removal → first difference → taper
    (`window`) → periodogram → postcolor ÷ (2 sin πmΔz)², the order of `estimation.Estimator(prewhiten=True)`
    and `analysis/spectral_estimation_methods/` (`test_first_difference_matches_prewhitened_estimator`). The
    differenced record has 27 samples, so `m_data` is 13 bins at k/243 m with no Nyquist bin. With
    differencing, phase elements 4, 5, 20 are **linearly interpolated** instead of zero-filled (a zero
    in a red record differences to two big steps that postcoloring pushes into the lowest bins); the
    undifferenced methods still zero-fill. The six methods are selected by `window` × `first_difference`
    in the CLI fit / GM-levels / independence-test entries (asked, "Taper window?" is a select;
    `run_all_processing` uses the default Hann, no differencing) and in the publication config.
    `phi_chi_LS_fit_path(..., window, first_difference)` adds `_first_difference` and/or
    `_rectangular` / `_tukey`; the default keeps the old names.
  - `compute_F_phi_chi` (Colosi 2016 Eq. 6.52) takes m in rad/m but returns a one-sided density **per
    cycle/m**: Eq. 6.52 is a density per rad/m, so it is multiplied by the 2π Jacobian to match the 9 m
    periodograms of `estimate_F_phi_chi` and Colosi's per-cpm figures. `phi_chi_spectrum_model`,
    `fit_F_phi_chis` and `F_phi_chi_ducted` inherit that, so only the x-axis is converted (`m / 2π`) when
    plotting. Before 2026-10-23 the Jacobian was missing: the WFT curves were 2π too low and the fitted GM
    levels about 2π too high. `tests/test_tools.py` pins the convention. The spectra (Eq. 6.26/6.52, local
    straight-ray, no wavenumber dilation) and `compute_phase_structure_function` (Eq. 7.50,
    with the ξ₁ dilation) are different approximations, so they need not agree on one GM level.
  - `compute_F_phi_chi` has two band treatments. The default (`per_point_band=False`) integrates every
    point of the ray at every m and then masks the result to the receiver band `[m_lo_max, m_hi_max]`
    (the largest mode-1 / mode-J wavenumbers along the ray). With `per_point_band=True` the band is applied
    inside the integral: each point s contributes only over its own band, mode 1 to mode J(s)
    (πN/N₀B ≤ m ≤ JπN/N₀B, in local m), and NaN marks only where no point contributes.
    `compute_wft_structure_function` passes `per_point_band` through. Every other caller (fits, figures,
    `theory_environments`, `estimation`) uses the default. With the per-point band the edges are the modes, **deliberately not half-integer bins**
    (the user's choice, the literal reading of Eq. 6.52). Because of that, in the geometric limit
    ∫F_φ dm ≈ 0.61 Φ² of `compute_lambda_phi` on every ducted ray, and the WFT structure function saturates
    at ≈0.61 × 2Φ². Φ² is the discrete mode sum of Eq. 6.56, and the continuous integral from mode 1
    undercounts its mode-1-dominated sum (edge factor 0.68 × H normalization 0.894). A ratio far from
    0.61, or one that varies between rays, would indicate a bug.
  - `fit_GM_vertical_coherence(wavefront_values)` is the coherence counterpart of `fit_F_phi_chis`. It
    fits one GM level to D = −2 ln VC (`structure_function_from_coherence`, lags 9–126 m, jackknife weights),
    using the path-integral D of `compute_phase_structure_function` (with the ξ₁ dilation) on the first eigenray pair.
    The model is re-evaluated at each GM because J(s) depends on GM, though D/GM varies by <1% from 1 to 4 GM.
    The SE is a jackknife of the fit linearized at the optimum. D from the coherence is the *wave* structure
    function (φ + χ), so it can't see the WFT φ/χ split that the spectral fit depends on.
  - `jackknife_F_phi_chi_fit(results, ...)` gives jackknife SEs for a `fit_F_phi_chis` result. The fit is
    linearized at the optimum (Gauss-Newton), so the 23 leave-one-out refits are skipped, since each model
    evaluation re-traces the rays. The weights are held at their full-data values. A parameter on its bound is held fixed and
    gets a NaN SE. `test_jackknife_F_phi_chi_fit_matches_refit` checks it against exact refits.
  - `GM_levels_latex_table` puts the WFT spectral-fit and path-integral coherence-fit levels side by side, each ±
    its jackknife SE. The table backs the CLI's "GM levels (spectra vs. coherence)" entry, which asks "Remove
    array tilt?" and reads the existing `phi_chi_LS_fit[_tilt_removed].pkl` (`--` if missing). It does the
    coherence fit itself and takes under a minute. On the tilt-removed data the coherence gives
    3.47 ± 0.65 / 0.74 ± 0.21 / 2.71 ± 1.25 / 3.32 ± 0.98 (+7 / −7 / +6 / −6), and the spectral fit gave
    0.56 / 0.19 / 0.45 / 0.84 (SEs not yet re-run since the dilation option was removed on 2026-10-29).
  - `transmission_independence_test(samples)` supports the jackknife's assumption that the hourly
    transmissions are independent. It compares the lag-1 autocorrelation of per-transmission estimates
    with ±2/√M and gives a permutation p-value. `independence_test_F_phi_chi` (log-periodograms) and
    `independence_test_vertical_coherence` (lagged products R_k(Δz), Δz ≤ 180 m) apply it to the ducted
    rays. The CLI's "transmission independence test" entry prints both. The test assumes rows are in time
    order: `open_receptions` globs `dm<day><hour>1000.mat`, and those names sort chronologically at 1 h spacing.
  - `EIG_INDEXS` — a large hand-picked dict mapping ray IDs to forward/backward eigenray pair indices, built
    via `notebooks/auxiliary/Ray_forward_backward_matching.ipynb`. **Don't regenerate this programmatically**
    without understanding that provenance — it encodes manual matching decisions, not a derivable rule.
- **`src/n23_fluctuations_ragland/cli.py`** — `main()`: a `questionary` menu wrapping the processing pipeline
  (arrival time offset, peak locations, array tilt, wavefront values, mean pulse, ray computation, source geometry sweep,
  φ/χ least-squares fit), each step decorated with `check_output_exists` to skip/overwrite cached outputs.
  Wavefront values, the φ/χ fit, the coherence-lengths table and the independence test each ask "Remove array tilt?"
  (default yes) and read/write the `*_tilt_removed.pkl` products when it's yes; `run_all_processing()`
  runs both modes of the wavefront values and the fit. The same entries also ask "Remove hydrophone
  pattern?" (none/phase/magnitude/both, default none). `run_all_processing()` always uses `none`.
  "estimate hydrophone pattern" reads `wavefront_values_tilt_removed.pkl` and writes `hydrophone_pattern.pkl`
  and a diagnostic `hydrophone_pattern.png` (`hydrophone_pattern_report`, `plot_hydrophone_pattern`). It has
  to run before any pattern-removed wavefront values.
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
  - The config's `remove_pattern` (`none` default, `phase`, `magnitude`, `both`) picks the
    pattern-removed variants of the same files (`tools.wavefront_values_path`).
  - The config's `window` (`hann` default, `tukey`, `rectangular`) and `first_difference` (default `false`)
    pick the spectral
    estimation method of `F_phi_chi_ducted`, and with `spectra_model: fit` the matching fit file.
  - The config's `spectra_model` picks the `F_phi_chi_ducted` model curves: `fit` (default, the notebook) draws
    the least-squares GM level with model + noise dashed and the fit text box; `gm_levels` draws 1× GM (solid)
    and 2× GM (dashed) without noise or the fit, with a shared GM legend strip as in `Ducted_VC` (−8 overlay:
    dash-dot 1×, dotted 2×). Both write the same `F_phi_chi_ducted` name.
    `gm_levels` (default `[1, 2]`) sets those levels: one or two positive numbers, solid then dashed, with
    the legend labels built from them; `[1]` draws 1× GM only. It is ignored with `spectra_model: fit`.
  - `common_arrival_time_shift_estimation` reads `dT_t_upper`/`dT_t_lower` from `T_t_estimated.pkl`.
    Only this project's "compute arrival time offset" writes those keys. The base `nescafe23` CLI
    writes the same file without them.
- **`src/n23_fluctuations_ragland/estimation.py`** — experiments on the design decisions of the F_φ/F_χ
  estimate (CLI: "create estimation experiments config" / "run estimation experiments", template
  `templates/estimation_experiments.yaml`). An `Estimator` holds each choice: wavefront source, bad-element
  handling, detrend, removal of the pattern common to transmissions (per ray, over the ducted rays, or over
  all arrivals), prewhitening,
  taper (rectangular/Hann/Tukey/DPSS multitaper), log-mean vs mean. Its defaults reproduce
  `tools.estimate_F_phi_chi(remove_array_tilt=True)` exactly (`tests/test_estimation.py`), except that phase is
  unwrapped with the bad elements already removed. The template's base instead keeps all 28 elements:
  elements 4, 5 (about -1.1 and -0.85 rad) and 20 (about -1.1 rad) carry a constant phase offset that is the
  same on every transmission and on every arrival from 7° to 29°, up- and down-going, so it is electronic,
  not an element-position error, which would scale with sin θ. It is removed as the common phase
  (`common_phase: all_arrivals`), pooled over every lower-array arrival except `common_omit` (default
  `tools.ARRAY_TILT_OMIT`, as for the array tilt) and over picks with QC flags 1 or 5 (`common_qc_only`).
  The common phase is removed as phasors before unwrapping (`circular_common_phase`), because the raw
  neighbour steps around those elements reach ~3 rad. Each variant is fitted with its own jackknife SEs, and a
  Monte Carlo runs every variant on synthetic wavefronts with a known GM + white-noise spectrum. The fit uses
  F(GM) = GM × F(GM=1): the model is linear in the GM level, so the slow `compute_n23_ray_variables` runs once.
  The synthetic results depend on the GM model being right, including the zero (or flat) spectrum below
  mode 1. Only the rays in `ray_ids` are synthesized, so `all_arrivals` pools just those there. The wavefront
  source never affects the synthetic runs, so variants that differ only in it give
  identical synthetic results. The observed common pattern enters them only with
  `synthetic.common_pattern: observed`. The sources `tilt_removed_phase`, `tilt_removed_magnitude` and
  `tilt_removed_both` (`WAVEFRONT_FILES`) read the wavefront values with the hydrophone pattern removed
  in the time domain, for comparison with the phasor-domain `common_phase`.
- **`src/n23_fluctuations_ragland/theory_environments.py`** — runs the analysis' own theory code on other
  environments, to tell environment effects from code effects in the WFT vs path-integral gap. An environment
  is a Dataset `c` [m/s], `N` [cph], `dcdz_p` [1/s] on `(depth, range)`, source at range 0 (the contract of
  `tools.fctd_environment`). `run_theory(name)` traces it with `tools.run_rays(envi_field=...)` (same geometry,
  flat 5000 m bottom) and returns ray variables, F_φ/F_χ, D_PI, D_wft φ/χ (`tools.compute_wft_structure_function`),
  z0, Φ², Λ, cached at `dataproducts/theory_envs/<name>.pkl`. `ENVIRONMENTS`: `munk` (canonical, analytic,
  self-consistent c/N/dc_p), `fctd` (the analysis' cached ray run + `EIG_INDEXS`, so it is exactly the current
  product), `fctd_range_avg` (FCTD t, SP averaged over 0–153 km, re-derived), `nesma_beta0` (nesma_param β = 0:
  WOA mean + Argo EOFs at the SSH-conditioned mean coefficients, SSH of 2023-04-30 on the SIOE→SION geodesic;
  Argo EOF cache `dataproducts/argo_EOF_fluct.npz`, kept apart from the inversion's). All T,S-based
  environments go through `nescafe23.acoustic_fields_from_TS`, the derivation of `compute_filtered_fctd_grid`.
  Comparison notebook: `notebooks/theory_environment_comparison.ipynb`.
  - In canonical Munk at this geometry only `-6` (and `-5`) reach the 1139 m receiver; `+7`, `-7`, `+6` have no
    eigenray there, so Munk compares on `-6` only.
- **Eigenray pairing outside the FCTD run.** `tools.pair_eigenrays` pairs forward/backward eigenrays by
  reciprocity (backward launch = −forward received angle, backward received = −forward launch;
  `linear_sum_assignment` on the larger mismatch, 1° tolerance). On the cached FCTD run it reproduces
  `EIG_INDEXS` for `+7, -7, +6, -6`, but **not** for `+7b` (reciprocity gives (1, 2), hand-picked (1, 0)) or
  `-5b`. `compute_n23_ray_variables(eig_indexs="reciprocity")` uses it; the default (`None`) is still
  `EIG_INDEXS`. `run_rays` and `compute_n23_ray_variables` both default to `tools.fctd_environment()`.
- **`analysis/`** holds one-off diagnostic studies, one directory each, with the scripts, their figures and a
  README of findings. It is not part of the package and is excluded from ruff. Large cached inputs are
  gitignored and regenerated by the scripts. `analysis/hydrophone_pattern/` covers the hydrophone pattern
  estimate and the effect of its removal on the ducted spectra and vertical coherence.
  `analysis/spectral_estimation_methods/` compares taper/prewhitening choices on synthetic `-7` GM wavefronts.
  `analysis/minus_6_interference/` covers how the bottom-bounced `-6b` overlaps the ducted `+6`/`+7` wavefronts.
  Its plane-wave line is t_n = t₀ − (z_n − z̄)·sinθ/c: the beamformer convention puts negative angles
  later with depth. The picks are in nominal depth, which is ~100 m shallower than the actual depth the mean pulse uses.
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
  `hydrophone_pattern.pkl`, the `*_phase_removed` / `*_magnitude_removed` / `*_phase_magnitude_removed` variants of the
  wavefront values and fits,
  `T_t_estimated.pkl`) and NetCDF (`mean_pulse_*.nc`). Raw transmissions come from `.mat` files under
  `dmod/`. Figures are saved as PDFs to `figures/`.
- Notebooks are stripped of outputs via `nbstripout` as a git filter — run `nbstripout --install` locally.
- ruff excludes `notebooks/`, `dev`, `panel_apps`, and `src/archive.py`.
- **Known unresolved issues** (per README TODOs): source localization code is intentionally omitted from this
  public repo; backward ray tracing fails to find eigenrays for `+7b`; hardcoded geometry constants in
  `tools.py`/`cli.py` are marked "best guess" and should be treated as provisional, not final truth.
- `cli.py` must import `convert_to_xarray` from `nescafe23` directly, not via `tools` —
  `test_convert_to_xarray_not_on_tools` in `test_cli.py` encodes a real historical bug where this was wrong.
