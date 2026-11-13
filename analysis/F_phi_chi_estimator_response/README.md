# F_φ / F_χ estimator response (−7)

What each spectral estimator does to the `-7` GM theory spectrum of `tools.compute_F_phi_chi`, on the 28-element,
9 m lower array. Made on 2026-11-02. Extends `analysis/spectral_estimation_methods/` with the Tukey taper
(α = 0.3, as `tools.TUKEY_ALPHA`) and first differencing combined with every taper, and adds an exact expectation
next to the Monte Carlo.

Run with `uv run python analysis/F_phi_chi_estimator_response/expected_spectra.py` (about 4 s) and
`.../monte_carlo_spectra.py` (about 6 s). They need only the cached ray run that `estimation.GMModel` reads.

| script | figure | what it shows |
|---|---|---|
| `expected_spectra.py` | `expected_spectra.png` | the exact expected value of each estimator applied to the theory, and log10(expected / theory). Lines are on the theory's wavenumber grid (the DTFT of the tapered record, i.e. infinite zero padding); markers are the estimator's own DFT bins, which lie on the lines |
| `monte_carlo_spectra.py` | `monte_carlo_spectra.png` | the same estimators on 500 synthetic 23-transmission experiments, as the pipeline averages them (bias-corrected log-mean), with the jackknife SE of one experiment as bands and the exact expectation as thin black lines |
| `common.py` | | theory spectrum, the six `estimation.Estimator`s, plot styling |

## Setup

- Truth: `-7` GM theory at 1× GM, no noise (`estimation.GMModel`, first eigenray). It is zero outside the ray's
  band (`low_wavenumber: zero`), so zero below mode 1 (3.8×10⁻⁴ cpm). It is given at k/(4096 × 9 m) up to the
  array Nyquist (1/18 cpm), the grid of `estimation.synthesize_wavefronts`. Aliasing from above Nyquist is
  therefore not included.
- Estimators: {rectangular, Hann, Tukey α = 0.3} × {without, with first difference + postcolor}. Each is an
  `estimation.Estimator` with all 28 elements and mean removal. The order is mean removal, first difference,
  taper, periodogram, postcolor ÷ (2 sin πmΔz)². The differenced records have 27 samples, so their 13 bins are at
  k/243 m and have no Nyquist bin.
- Exact expectation: the estimators are linear in the record up to the squared modulus, so
  E[S_k] = 2Δz b_kᴴ C b_k / P_k. Here b_k = Aᵀ(w e_k), A is the mean removal (and difference), w the unit-energy
  taper, P_k the postcolor, and C the covariance of the 28 samples from the cosine transform of the truth. This is
  the mean of the periodograms. It has no sampling noise and needs no Monte Carlo.

## Findings (log10 bias, exact expectation)

| method | F_φ | F_χ |
|---|---|---|
| rectangular | +0.50 at bin 1 rising to +2.0 at Nyquist: leakage floor | +0.07 rising to +0.66 |
| Hann | +0.18 / +0.42 in bins 1–2, then within ±0.08 | −0.13 / +0.18 in bins 1–2, then within ±0.08 |
| Tukey (α = 0.3) | +0.3 to +0.57 up to 0.03 cpm, within ±0.09 above 0.035 cpm | +0.04 to +0.13 up to 0.025 cpm, then within ±0.08 |
| first difference, rectangular | within +0.05 below 0.03 cpm, +0.09 to +0.13 in the top bins | within ±0.05 |
| first difference, Hann | within ±0.07 | −0.09 at bin 1, within ±0.05 above |
| first difference, Tukey (α = 0.3) | within ±0.06 | −0.05 at bin 1, within ±0.05 above |

- **Tukey alone does not stop the F_φ leakage.** Its flat 70% middle has the sidelobes of a rectangular window,
  so the red phase spectrum leaks into the mid bins (+0.55 decades at 0.02–0.024 cpm). It falls back to the
  theory only above ~0.035 cpm, where its tapered ends start to roll the sidelobes off. It is better than
  rectangular everywhere and worse than Hann below 0.03 cpm. For F_χ, which is much less red, Tukey is
  within +0.13 and has no low first bin. Hann reads −0.13 there.
- **After first differencing the taper barely matters.** The differenced record is close to white, so leakage
  is small for any taper. Differencing with Tukey is the best of the six overall: within ±0.06 for F_φ, and
  its F_χ bin 1 (−0.05) is closer than Hann's (−0.09). Differencing with a rectangular window keeps a +0.1
  leakage residual in the top F_φ bins.
- **Between and below the bins** (the continuous lines):
  - The rectangular and Tukey F_φ responses oscillate between bins (sidelobe scalloping). The rectangular
    bins fall on the peaks of that ripple. Hann's response is smooth.
  - Below the first bin (1/252 m), mean removal sends every undifferenced response toward a constant as m → 0,
    so they fall 1–3 decades below the theory.
  - Below the first bin, the differenced responses instead rise as m⁻², because postcoloring divides by
    (2 sin πmΔz)² → 0. The record after differencing still has a nonzero mean, x₂₈ − x₁. This happens to
    follow the red GM F_φ to within 0.2 decades, but over-reads F_χ by up to 2 decades below its peak.
  - Nothing is estimated there in practice, since no bin falls below 1/L. It shows that the bins at and
    above 1/L are the only meaningful output.
- The Monte Carlo agrees with the exact expectation to within ±0.05 for all tapered and differenced
  estimators. For rectangular, and for Tukey's first F_φ bin, the Monte Carlo reads lower than the exact mean
  (up to 0.2 decades for rectangular F_φ). It averages the bias-corrected log-mean, and that correction
  assumes χ²₂ bins. Bins dominated by power leaked from a few long-scale components are not χ²₂, and their
  geometric mean is below their arithmetic mean.
- The jackknife SE of one experiment is calibrated against the spread between experiments to within 0.01
  decades: about 0.12 decades per bin, 0.20 at the 1-dof Nyquist bin.
- The theory curve is jagged above ~0.01 cpm, which puts a few hundredths of a decade of scatter into the
  per-bin biases.
