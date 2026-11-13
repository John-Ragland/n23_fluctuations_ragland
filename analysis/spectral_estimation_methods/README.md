# Spectral estimation methods

How the choice of F_φ / F_χ estimator biases the spectrum of a very red wavefront, tested on synthetic data with
a known spectrum. Made on 2026-10-31.

Run with `uv run python analysis/spectral_estimation_methods/spectral_methods.py` (about 15 s). It needs no data
products beyond the cached ray run that `estimation.GMModel` reads.

| script | figure | what it shows |
|---|---|---|
| `spectral_methods.py` | `spectral_methods.png` | F_φ (left) and F_χ (right) of `-7` at 1× GM, no noise, estimated by a rectangular window, a Hann window (current), first-differencing + rectangular + postcoloring, and first-differencing + Hann + postcoloring; prints log10(expected / truth), the jackknife SE and the spread between experiments per bin |
| | `spectral_methods_noise.png` | the same with white noise at the levels fitted to the observed `-7` (φ 1.56 rad² m, χ 0.279 m; tilt removed, 28 elements, common phase over all arrivals, fitted GM 0.19) |

## Setup

- The truth is the `-7` GM spectrum of `estimation.GMModel` (first eigenray, GM = 1, no white noise). It is zero
  below mode 1 (3.8×10⁻⁴ cpm, ten times lower than the array's first bin).
- Phase and log-amplitude are drawn as independent Gaussian records by `estimation.synthesize_wavefronts`. The
  28 × 9 m array is cut from a 4096-element periodic record, so variance at scales longer than the array is
  present and can leak.
- The estimates go through `estimation.estimate_spectra` with all 28 elements (`bad_elements=()`) and mean
  removal ("no detrend"), so they average with the same bias-corrected log-mean as the real pipeline.
- Lines show the expected value: the mean, over 500 independent 23-transmission experiments, of the log-mean.
  Bands show the error bar one 23-transmission experiment carries: the RMS over the 500 experiments of each
  one's own leave-one-transmission-out jackknife SE (log10). The script also prints the standard deviation of
  the log-mean between experiments. The two agree to within 0.01 decades in every bin and method (about 0.12
  decades, 0.20 at the 1-dof Nyquist bin), so the jackknife SE of a single experiment is calibrated.
- An earlier version plotted one experiment (seed 0) as a solid line. Its second F_χ bin for first difference
  + Hann read −0.38 decades, 3.6σ below the expected +0.02 and lower than all 500 Monte Carlo experiments.
  That was a sampling fluke, not a bias: the single-transmission periodograms in that bin are exactly χ²₂ when
  pooled over many experiments.

## Findings (log10 bias of the expected value)

No noise:

| method | F_φ | F_χ |
|---|---|---|
| rectangular | +0.45 at the first bin, rising to +2.0 at Nyquist: leakage floor | +0.06 to +0.65, rising with m |
| Hann (current) | +0.42 at bin 2, +0.16 at bin 1, within ±0.08 above | −0.13 / +0.17 in bins 1–2, then within ±0.07 |
| first difference, rectangular, postcolored | within +0.08 below 0.04 cpm, +0.10 to +0.13 in the top 3 bins | within ±0.05 everywhere |
| first difference, Hann, postcolored | within ±0.06 everywhere | −0.10 at bin 1, within ±0.05 above |

With the observed −7 noise levels, the noise covers the high-wavenumber leakage. Rectangular still reads +0.3 to
+0.6 for F_φ and +0.1 to +0.17 for F_χ. Hann is within ±0.01 above 0.02 cpm but keeps its low-bin errors
(F_φ +0.16 / +0.38, F_χ −0.13 / +0.17). Both differenced estimators stay within ±0.06, apart from −0.09 at the
first F_χ bin of the Hann version.

### Why Hann reads high, and sometimes low, at the lowest bins

The mean is removed before tapering, so the DFT of the untapered record is zero at DC. Hann's bin k is
(X_k − ½X_{k−1} − ½X_{k+1})/2, scaled by √(8/3) to unit mean square. Bin 1 is therefore
√(8/3)(½X₁ − ¼X₂). With uncorrelated bins its expectation is (2/3)S₁ + (1/6)S₂. That is 0.70 S₁, −0.16 decades,
for the −7 spectrum. This is the lost large-separation weight.

That only holds if the record has no variance at scales longer than the array. The variance it does have there
leaks into the first bins, with a positive sign. Which effect wins depends on the spectrum below 1/L
(L = 252 m). Tested by setting the GM truth to zero below a cutoff (no noise, 300 experiments):

| truth zero below | Hann F_φ bin 1 | Hann F_χ bin 1 | diff + Hann F_φ / F_χ bin 1 |
|---|---|---|---|
| mode 1 (full GM) | +0.16 | −0.15 | +0.06 / −0.11 |
| 0.5/L | −0.11 | −0.20 | −0.11 / −0.13 |
| 0.9/L | −0.52 | −0.43 | −0.37 / −0.25 |

The GM F_φ still rises by two decades below 1/L, so its leakage outweighs the loss. The GM F_χ peaks at
1.6×10⁻³ cpm and falls below it, so the loss wins. The −0.13 in the F_χ figure is this effect. The +0.42 at
Hann's F_φ bin 2 is the leakage alone.

On the real data, the first bins read low whenever the record holds less variance at scales longer than the array
than GM predicts. That happens when the spectrum really is flatter there, or when processing removes it: a
per-transmission linear detrend gives −0.33 / −0.37 (F_φ / F_χ, table below). Steering each transmission at its
own picked angle should act like a linear detrend of the phase. This was not tested here.

Extra methods checked with the same setup, no noise (not in the figures):

| method | F_φ | F_χ |
|---|---|---|
| linear detrend + Hann | −0.33 / +0.30 in bins 1–2, then as Hann | −0.37 / +0.14 in bins 1–2, then as Hann |
| DPSS multitaper, NW = 2, K = 3 | +0.4 to +0.96 above bin 1: the higher-order tapers leak | +0.05 to +0.29 |

A linear detrend before first-differencing changes nothing: the line differences to a constant, which falls
in the DC bin, and the DC bin is dropped.

The theory curve itself is jagged above ~0.01 cpm (it comes from `compute_F_phi_chi` on the traced ray). That
jaggedness puts a few hundredths of a decade of scatter into the per-bin biases.
