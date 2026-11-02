"""Monte Carlo of each estimator on synthetic -7 wavefronts drawn from the GM theory spectrum
(`tools.compute_F_phi_chi`, 1x GM, no noise, zero below mode 1), 28 elements x 9 m, 23 transmissions per
experiment, N_MC experiments.

The synthesis and estimation are `estimation.synthesize_wavefronts` and `estimation.estimate_spectra` (all 28
elements, mean removal, bias-corrected log-mean over transmissions), as in `analysis/spectral_estimation_methods`.
Lines: the expected value, the mean over the N_MC experiments of log10 of the log-mean. Bands: the error bar one
23-transmission experiment carries, the RMS over experiments of each one's own leave-one-transmission-out
jackknife SE (log10). Thin black lines in the bias panels: the exact expectation of `expected_spectra.py`.

Writes `monte_carlo_spectra.png` and prints log10(expected / theory), the jackknife SE and the spread between
experiments per bin.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from common import (
    DZ,
    GM,
    METHODS,
    N_EL,
    RAY,
    RECORD_LENGTH,
    legend_below,
    model,
    style_spectrum_axes,
    theory,
    truth_long,
)
from expected_spectra import covariance, expected_spectrum

from n23_fluctuations_ragland import estimation as E

SP = Path(__file__).parent
N_TX = 23
N_MC = 500
SEED = 0


def main():
    params = {RAY: {"GM": GM, "phi_noise": 0.0, "chi_noise": 0.0}}
    cfg = E.SyntheticConfig(record_length=RECORD_LENGTH, low_wavenumber="zero")
    rng = np.random.default_rng(SEED)

    log_mean = {(k, q): [] for k in METHODS for q in ("phi", "chi")}
    log_se = {(k, q): [] for k in METHODS for q in ("phi", "chi")}
    m_of = {}
    for _ in range(N_MC):
        w = E.synthesize_wavefronts(model, params, cfg, rng, n_tx=N_TX, n=N_EL)
        for k, (est, *_) in METHODS.items():
            e = E.estimate_spectra(w, est, [RAY])[RAY]
            m_of[k] = e.m
            for q in ("phi", "chi"):
                log_mean[k, q].append(np.log10(getattr(e, f"{q}_lm")))
                log_se[k, q].append(np.log10(getattr(e, f"{q}_ls")))
    mc_mean = {key: 10 ** np.mean(v, axis=0) for key, v in log_mean.items()}
    se_jk = {key: np.sqrt(np.mean(np.square(v), axis=0)) for key, v in log_se.items()}
    se_mc = {key: np.std(v, axis=0) for key, v in log_mean.items()}

    # exact expectation, for the check in the bias panels
    S_long = dict(zip(("phi", "chi"), truth_long()))
    exact = {}
    for q in ("phi", "chi"):
        C = covariance(S_long[q])
        for k, (est, *_) in METHODS.items():
            exact[k, q] = expected_spectrum(est, C)[1]

    m_fine = np.logspace(np.log10(3e-4), np.log10(1 / (2 * DZ)), 600)
    F_fine = dict(zip(("phi", "chi"), theory(m_fine)))

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
        for k, (_, c, ls, mk) in METHODS.items():
            m = m_of[k]
            lm, fac = mc_mean[k, q], 10 ** se_jk[k, q]
            truth = theory(m)[0 if q == "phi" else 1]
            mfc = c if ls == "-" else "none"
            ax.loglog(m, lm, color=c, ls=ls, lw=1.3, marker=mk, ms=4, mfc=mfc, label=k)
            ax.fill_between(m, lm / fac, lm * fac, color=c, alpha=0.1, lw=0)
            bias = np.log10(lm / truth)
            axr.semilogx(m, bias, color=c, ls=ls, lw=1.3, marker=mk, ms=4, mfc=mfc)
            axr.fill_between(m, bias - se_jk[k, q], bias + se_jk[k, q], color=c, alpha=0.08, lw=0)
            axr.semilogx(m, np.log10(exact[k, q] / truth), color="k", lw=0.6, ls=ls, alpha=0.6)
        axr.axhline(0, color="k", lw=1)
        ax.set_ylabel(ylabel)
        axr.set_ylabel("log10(MC mean / theory)")
        axr.set_xlabel("vertical wavenumber [cycles/m]")
    style_spectrum_axes(axes)
    axes[0, 0].set_title("phase")
    axes[0, 1].set_title("log-amplitude")
    legend_below(
        fig,
        axes,
        extra=(
            (plt.matplotlib.patches.Patch(color="0.3", alpha=0.15, lw=0),
             f"±1 jackknife SE of one {N_TX}-transmission experiment"),
            (plt.matplotlib.lines.Line2D([], [], color="k", lw=0.6, alpha=0.6),
             "exact expectation (expected_spectra.py)"),
        ),
    )
    fig.suptitle(
        f"Monte Carlo, synthetic {RAY} GM theory, {N_EL} × {DZ:g} m, {N_TX} transmissions per estimate, "
        f"mean of {N_MC} experiments, no noise\n(log-mean over transmissions; dotted vertical: 1/array length)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(SP / "monte_carlo_spectra.png", dpi=200)

    for q in ("phi", "chi"):
        print(f"\n{q}: log10(MC mean / theory), log10(exact / theory), jackknife SE (RMS), SD between "
              "experiments [decades]")
        for k in METHODS:
            m = m_of[k]
            truth = theory(m)[0 if q == "phi" else 1]
            print(f"{'m [cpm]':34s} " + " ".join(f"{x:6.4f}" for x in m))
            print(f"{k[:34]:34s} " + " ".join(f"{x:6.2f}" for x in np.log10(mc_mean[k, q] / truth)))
            print(f"{'  exact':34s} " + " ".join(f"{x:6.2f}" for x in np.log10(exact[k, q] / truth)))
            print(f"{'  SE jk':34s} " + " ".join(f"{x:6.2f}" for x in se_jk[k, q]))
            print(f"{'  SD MC':34s} " + " ".join(f"{x:6.2f}" for x in se_mc[k, q]))


if __name__ == "__main__":
    main()
