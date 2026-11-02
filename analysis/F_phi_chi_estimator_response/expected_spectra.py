"""Exact expected value of each estimator applied to the -7 GM theory spectrum (`tools.compute_F_phi_chi`, 1x GM,
no noise, zero outside the ray's band), on the 28-element, 9 m array.

Each estimator is linear in the record before the squared modulus: bin k of one record x is
S_k = 2 dz |sum_j w_j e^{-2 pi i k j / n'} (A x)_j|^2 / P_k, with A the mean removal (and first difference), w the
unit-energy taper and P_k the postcoloring (1 for no difference). So E[S_k] = 2 dz b_k^H C b_k / P_k, with
b_k = A^T (w e_k) and C the covariance of x, C_ij = R((i - j) dz), R(tau) = sum_k S(m_k) df cos(2 pi m_k tau).
The spectrum is taken on the grid of `estimation.synthesize_wavefronts` (4096-element record, up to the Nyquist of
the array), so this is exactly the mean the Monte Carlo of `monte_carlo_spectra.py` converges to (for the
arithmetic mean of the periodograms; its bias-corrected log-mean differs from it only where bins are not chi^2).

Writes `expected_spectra.png` and prints log10(expected / truth) per bin.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from common import (
    DF,
    DZ,
    GM,
    M_LONG,
    METHODS,
    N_EL,
    RAY,
    legend_below,
    style_spectrum_axes,
    theory,
    truth_long,
)

SP = Path(__file__).parent


def covariance(S, n=N_EL):
    """Covariance matrix of n samples at spacing DZ of a stationary record with one-sided spectrum S on M_LONG."""
    tau = DZ * np.arange(n)
    R = (S * DF) @ np.cos(2 * np.pi * np.outer(M_LONG, tau))
    lag = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    return R[lag]


def expected_spectrum(est, C, m=None):
    """
    Wavenumbers [cpm] and the exact expected spectrum of estimator `est` for a record with covariance C.

    By default at the DFT bins of the (differenced) record, the bins the estimator returns. With `m`, at those
    wavenumbers instead: the discrete-time Fourier transform of the tapered record (the limit of infinite zero
    padding), which passes through the bin values and shows the estimator's response between them.
    """
    n = C.shape[0]
    A = np.eye(n) - 1 / n  # mean removal
    if est.prewhiten:
        A = np.diff(A, axis=0)
    n2 = A.shape[0]
    w = est.tapers(n2)[0]
    if m is None:
        m = np.arange(1, n2 // 2 + 1) / (n2 * DZ)
    m = np.asarray(m, dtype=float)
    B = (np.exp(-2j * np.pi * DZ * np.outer(m, np.arange(n2))) * w) @ A  # (wavenumbers, n)
    S = 2 * DZ * np.real(np.einsum("ki,ij,kj->k", B, C, B.conj()))
    if est.prewhiten:
        S = S / (2 * np.sin(np.pi * m * DZ)) ** 2
    return m, S


def main():
    m_fine = np.logspace(np.log10(3e-4), np.log10(1 / (2 * DZ)), 600)
    F_fine = dict(zip(("phi", "chi"), theory(m_fine)))

    S_long = dict(zip(("phi", "chi"), truth_long()))
    expected, expected_fine = {}, {}
    for q in ("phi", "chi"):
        C = covariance(S_long[q])
        for label, (est, *_) in METHODS.items():
            expected[label, q] = expected_spectrum(est, C)
            expected_fine[label, q] = expected_spectrum(est, C, m_fine)[1]

    fig, axes = plt.subplots(
        2, 2, figsize=(11, 7.2), sharex=True, gridspec_kw={"height_ratios": (2.2, 1)}
    )
    for col, (q, ylabel) in enumerate(
        (("phi", r"$F_\phi$ [rad$^2$ m]"), ("chi", r"$F_\chi$ [m]"))
    ):
        ax, axr = axes[0, col], axes[1, col]
        ax.loglog(
            m_fine, F_fine[q], "k", lw=2.2, zorder=5,
            label=f"GM theory (compute_F_phi_chi), {RAY}, {GM:g}× GM",
        )
        for label, (_, c, ls, mk) in METHODS.items():
            m, S = expected[label, q]
            truth = theory(m)[0 if q == "phi" else 1]
            S_f = expected_fine[label, q]
            mfc = c if ls == "-" else "none"
            # continuous response on the theory grid, with the estimator's own bins as markers on it
            ax.loglog(m_fine, S_f, color=c, ls=ls, lw=1.2, label=label)
            ax.loglog(m, S, color=c, ls="none", marker=mk, ms=4, mfc=mfc)
            axr.semilogx(m_fine, np.log10(S_f / F_fine[q]), color=c, ls=ls, lw=1.2)
            axr.semilogx(m, np.log10(S / truth), color=c, ls="none", marker=mk, ms=4, mfc=mfc)
        axr.axhline(0, color="k", lw=1)
        ax.set_ylabel(ylabel)
        axr.set_ylabel("log10(expected / theory)")
        axr.set_xlabel("vertical wavenumber [cycles/m]")
    style_spectrum_axes(axes)
    axes[0, 0].set_title("phase")
    axes[0, 1].set_title("log-amplitude")
    legend_below(fig, axes)
    fig.suptitle(
        f"Exact expected value of each estimator, {RAY} GM theory, {N_EL} elements × {DZ:g} m, no noise, "
        "zero below mode 1\n(lines: expected value on the theory grid (DTFT of the tapered record); markers: the "
        "estimator's own bins; dotted vertical: 1/array length)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(SP / "expected_spectra.png", dpi=200)

    var_fraction = {
        q: np.sum(S_long[q] * DF) for q in ("phi", "chi")
    }
    print(f"variance of the record (truth up to Nyquist): phi {var_fraction['phi']:.3g} rad², "
          f"chi {var_fraction['chi']:.3g}")
    for q in ("phi", "chi"):
        print(f"\n{q}: log10(expected / theory) per bin")
        for label, (est, *_) in METHODS.items():
            m, S = expected[label, q]
            truth = theory(m)[0 if q == "phi" else 1]
            print(f"{'m [cpm]':34s} " + " ".join(f"{x:6.4f}" for x in m))
            print(f"{label[:34]:34s} " + " ".join(f"{x:6.2f}" for x in np.log10(S / truth)))


if __name__ == "__main__":
    main()
