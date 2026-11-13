"""Shared setup for the -7 estimator-response figures: the GM theory spectrum of `tools.compute_F_phi_chi`, the
six estimators, and plot styling."""

import matplotlib.pyplot as plt
import numpy as np

from n23_fluctuations_ragland import estimation as E

RAY = "-7"
GM = 1.0
N_EL = 28  # lower-array elements, all used (no bad-element fill)
DZ = E.DZ
TUKEY_ALPHA = 0.3  # as tools.TUKEY_ALPHA
# the discretization of `estimation.synthesize_wavefronts`: a periodic record of RECORD_LENGTH elements, so the
# spectrum is given at k df, k = 1 ... RECORD_LENGTH / 2, up to the 1/18 cpm Nyquist of the array
RECORD_LENGTH = 4096
DF = 1 / (RECORD_LENGTH * DZ)
M_LONG = DF * np.arange(1, RECORD_LENGTH // 2 + 1)


def _est(name, window, prewhiten):
    return E.Estimator(
        name=name,
        bad_elements=(),
        window=window,
        tukey_alpha=TUKEY_ALPHA,
        prewhiten=prewhiten,
        # the bias-corrected log-mean that monte_carlo_spectra.png was made with (the Estimator default became
        # "mean" on 2026-11-02); expected_spectra.py ignores it
        average="log_mean",
    )


# label → (Estimator, color, linestyle, marker). The order is mean removal, (first difference), taper,
# periodogram, (postcolor), as in `estimation._spectra_of` and `tools.estimate_F_phi_chi`.
METHODS = {
    "rectangular": (_est("rect", "rectangular", False), "C3", "-", "o"),
    "Hann": (_est("hann", "hann", False), "C0", "-", "o"),
    f"Tukey (α = {TUKEY_ALPHA:g})": (_est("tukey", "tukey", False), "C2", "-", "o"),
    "first difference, rectangular": (_est("fd_rect", "rectangular", True), "C3", "--", "s"),
    "first difference, Hann": (_est("fd_hann", "hann", True), "C0", "--", "s"),
    f"first difference, Tukey (α = {TUKEY_ALPHA:g})": (_est("fd_tukey", "tukey", True), "C2", "--", "s"),
}

model = E.GMModel([RAY])


def theory(m):
    """GM theory F_phi, F_chi of -7 at GM level `GM` (`tools.compute_F_phi_chi`, per cycle/m), at `m` [cpm].
    NaN outside the ray's band."""
    F_phi, F_chi = model(RAY, np.asarray(m, dtype=float))
    return GM * F_phi, GM * F_chi


def truth_long():
    """The truth on `M_LONG`, zero outside the band (``low_wavenumber: zero``)."""
    return tuple(np.nan_to_num(F, nan=0.0) for F in theory(M_LONG))


def style_spectrum_axes(axes, xlim=(3e-4, 0.06)):
    for ax in np.ravel(axes):
        ax.set_xscale("log")
        ax.axvline(1 / (N_EL * DZ), color="0.6", lw=0.8, ls=":")
        ax.grid(True, which="major", alpha=0.3)
        ax.set_xlim(*xlim)


def legend_below(fig, axes, extra=()):
    h, l = axes.flat[0].get_legend_handles_labels()
    for handle, label in extra:
        h.append(handle)
        l.append(label)
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=9)


plt.rcParams["figure.dpi"] = 100
