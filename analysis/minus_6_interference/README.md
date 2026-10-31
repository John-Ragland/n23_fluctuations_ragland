# -6b interference diagnostics

Diagnostics for how the bottom-bounced `-6b` arrival overlaps the ducted `+6`/`+7` wavefronts on the lower array.
Made on 2026-10-30.

Run from anywhere with the workspace environment, e.g.
`uv run python analysis/minus_6_interference/acoustic_receptions_minus6b.py`. It reads
`$NESCAFE23_DATADIR/dataproducts` and writes its figure next to itself.

| script | figure | what it shows |
|---|---|---|
| `acoustic_receptions_minus6b.py` | `Acoustic_Receptions_minus6b.{png,pdf}` | the `Acoustic_Receptions` publication figure with the mean `-6b` plane-wave arrival drawn on the mean pulse, panel (a) |

## Method

- The `-6b` time and angle are the mean of the `peak_locations.pkl` picks over the 13 transmissions with good QC
  flags (1 or 5): t₀ = 100.920 s (28 ms spread), θ = −17.2° (1.0° spread). The mean over all 23 transmissions is
  100.914 s, −17.6°.
- The line is t_n = t₀ − (z_nom,n − z̄_nom)·sinθ / 1500. That is the arrival a beam at θ lines up, given the
  steering delay d_n = −(z_n − z₀)·sinθ/c of `ULA_conventional_beamform`. Note the minus sign: negative angles
  arrive later with depth.
- The picks were beamformed on **nominal** depths, but panel (a) is averaged in **actual** depth. So each element's
  delay is plotted at its transmission-mean actual depth. The actual depths average about 100 m below nominal
  (1072 vs 972 m), the spacing is about 10 m rather than 9 m, and the array's center moves between 1006 and
  1157 m from one transmission to the next.

## Findings

- `-6b` runs through the late side of the `+6`/`+7` band and tilts the opposite way. The two overlap mainly in the
  middle of the array, about 1050–1150 m. At the top, `-6b` is just behind `+7`. At the bottom, it falls in the
  gap before `-7/+8`.
- The ~150 m spread in array depth blurs the mean pulse in depth. A per-transmission version would show the
  overlap more sharply.
