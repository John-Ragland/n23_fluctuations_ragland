# Hydrophone pattern diagnostics

Diagnostics for the per-hydrophone phase and gain pattern of the lower array
(`tools.estimate_hydrophone_pattern`) and for its removal before beamforming
(`get_wavefront_values(remove_pattern=...)`). All were made on 2026-10-30 with the array tilt removed.

Run from anywhere with the workspace environment, e.g. `uv run python analysis/hydrophone_pattern/vc_diag.py`.
Each script reads `$NESCAFE23_DATADIR/dataproducts` and writes its figure next to itself.

| script | figure | what it shows |
|---|---|---|
| `estimate_pattern_real_data.py` | `hydrophone_pattern.png` | the pattern with jackknife SEs and each arrival's own pattern, rejected arrivals dashed (same plot as the CLI's "estimate hydrophone pattern") |
| `arrival_test_false_rejection.py` | (prints) | false-rejection rate of the outlier tests on 100 clean synthetic runs |
| `spectra_diag.py` | `spectra_pattern_removal.png`, `spectra_pattern_removal_ratio.png` | ducted F_φ, F_χ with no / phase / magnitude / both patterns removed, and their ratio to none |
| `vc_diag.py` | `vc_pattern_removal.png` | ducted vertical coherence (as in `Ducted_VC`) for the same four variants |

`spectra_diag.py` needs `wavefront_values_tilt_removed[_phase_magnitude_removed].pkl` in `dataproducts`
("compute wavefront values"). It computes the phase-only and magnitude-only wavefront values itself, about 25 s
each, and caches them here as `wavefront_values_tilt_removed_{phase,magnitude}.pkl`. Those files are gitignored.
`vc_diag.py` reads the same cache.

## Findings

- **Pattern on the real data.** Elements 4, 5 and 20 come out at −0.87, −0.59 and −1.11 ± 0.06 rad. As delays
  that is −0.55, −0.38 and −0.70 ms. The gain runs from 0.72 at element 0 to 1.18.
- **Outlier test.**
  - The arrival test rejects `+7`, `-6`, `+7b`, `+9b`, `+13b` and `+15b` from the phase. They differ from the
    pooled pattern mainly by a smooth bowl (curvature) along the array, not at single elements. That looks like
    physics, not hardware. The pattern is only linearly detrended, so it still holds the average curvature of
    the arrivals it keeps.
  - The first version of the test, d²/var with the noisy jackknife variances, rejected an arrival on 10–19% of
    clean synthetic runs. The calibrated Welch-t version rejects on 2–4% (`arrival_test_false_rejection.py`).
- **Spectra.**
  - Removing the phase lowers F_φ by 20–55% at high wavenumbers; the median ratio is 0.80–0.90 per ray.
  - It also raises the lowest bins of `+7` and `-6` by ×1.5–1.75. These are the rays rejected for curvature.
    The comparison mixes two changes: "phase" and "both" also switch from zero-filling 4, 5, 20 to all 28
    elements.
  - The phase delay leaves F_χ within a few percent. The gain removal lowers F_χ above about 0.015 cpm,
    typically by 20–60% and in a few bins up to 70%. The lowest bins barely change.
- **Vertical coherence.**
  - z0 does not change within SE. In metres:

    | ray | none | phase | magnitude | both |
    |---|---|---|---|---|
    | `+7` | 59 ± 6 | 59 ± 6 | 59 ± 5 | 58 ± 5 |
    | `-7` | 133 ± 27 | 132 ± 25 | 145 ± 42 | 143 ± 37 |
    | `+6` | 111 ± 29 | 111 ± 27 | 117 ± 40 | 116 ± 35 |
    | `-6` | 86 ± 18 | 82 ± 20 | 86 ± 22 | 83 ± 23 |

  - Removing the phase lowers the zero-lag excess by 0.09–0.16 on every ray. The element offsets acted like
    uncorrelated noise.
- **Fit problem found here, not fixed in the library.**
  - `tools.fit_vertical_coherence` starts from A0 = 4e11, while these mutual coherences are about 4e12. Some
    fits stop at the starting point (z0 = 100 m) and still report success. With the pattern removed this blew
    up `+7` (z0 = 100 ± 89 m).
  - `vc_diag.py` starts the fit from the mutual coherence at 9 m instead.

## Open questions

- Should the pattern also remove a quadratic, so only element-scale structure is treated as hardware?
- Should `fit_vertical_coherence` start from the data's own scale? `Ducted_VC.pdf` built with
  `remove_pattern: both` before the fix has a bad `+7` SE.
