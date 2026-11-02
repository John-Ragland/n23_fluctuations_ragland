"""F_phi and F_chi of synthetic -7 wavefronts (GM theory at GM = 1) estimated by several methods on a 28-element,
9 m array with 23 transmissions, against the spectrum they were drawn from.

Two figures: no noise (`spectral_methods.png`), and white noise at the levels fitted to the observed -7
(`spectral_methods_noise.png`).

Lines: the expected value, the mean over N_MC independent 23-transmission experiments of the bias-corrected
log-mean. Bands: the typical leave-one-transmission-out jackknife SE of one 23-transmission experiment (RMS over
the N_MC experiments of each experiment's own jackknife SE, in log10), i.e. the error bar one experiment carries.
The spread of the log-mean between experiments is printed next to it as a check on the jackknife.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from n23_fluctuations_ragland import estimation as E

SP = Path(__file__).parent
RAY = "-7"
GM = 1.0
N_EL = 28
N_TX = 23
N_MC = 500
SEED = 0
# white-noise levels of the fit of the observed -7 (tilt removed, all 28 elements, common phase over all
# arrivals; fitted GM 0.19): phi 1.56 rad^2 m, chi 0.279 m
NOISE = {"phi_noise": 1.56, "chi_noise": 0.279}

# "no detrending" is the mean removal of the current stack (detrend: constant). average="log_mean" pins the
# bias-corrected log-mean these results were made with (the Estimator default became "mean" on 2026-11-02).
METHODS = {
    "rectangular": E.Estimator(name="rectangular", bad_elements=(), window="rectangular", average="log_mean"),
    "Hann (current)": E.Estimator(name="hann", bad_elements=(), window="hann", average="log_mean"),
    "first difference, rectangular, postcolored": E.Estimator(
        name="prewhiten", bad_elements=(), window="rectangular", prewhiten=True, average="log_mean"
    ),
    "first difference, Hann, postcolored": E.Estimator(
        name="prewhiten_hann", bad_elements=(), window="hann", prewhiten=True, average="log_mean"
    ),
}
COLORS = ["C3", "C0", "C2", "C1", "C4", "C5"]

model = E.GMModel([RAY])
cfg = E.SyntheticConfig(record_length=4096, low_wavenumber="zero")


def run(noise, fname):
    params = {RAY: {"GM": GM, "phi_noise": 0.0, "chi_noise": 0.0, **noise}}
    rng = np.random.default_rng(SEED)

    def synth():
        return E.synthesize_wavefronts(model, params, cfg, rng, n_tx=N_TX, n=N_EL)

    # N_MC independent 23-transmission experiments: the log-mean of each and its own jackknife SE
    log_mean = {k: {"phi": [], "chi": []} for k in METHODS}
    log_se = {k: {"phi": [], "chi": []} for k in METHODS}
    m_of = {}
    for _ in range(N_MC):
        w = synth()
        for k, est in METHODS.items():
            e = E.estimate_spectra(w, est, [RAY])[RAY]
            m_of[k] = e.m
            for q in ("phi", "chi"):
                log_mean[k][q].append(np.log10(getattr(e, f"{q}_lm")))
                log_se[k][q].append(np.log10(getattr(e, f"{q}_ls")))
    expected = {k: {q: 10 ** np.mean(v[q], axis=0) for q in v} for k, v in log_mean.items()}
    # error bar of one experiment: RMS of the jackknife SE (decades), and the actual spread between experiments
    se_jk = {k: {q: np.sqrt(np.mean(np.square(v[q]), axis=0)) for q in v} for k, v in log_se.items()}
    se_mc = {k: {q: np.std(v[q], axis=0) for q in v} for k, v in log_mean.items()}

    # theory on a fine grid
    m_fine = np.logspace(np.log10(3e-4), np.log10(1 / (2 * E.DZ)), 400)
    F_gm = dict(zip(("phi", "chi"), (GM * F for F in model(RAY, m_fine))))
    has_noise = any(v > 0 for v in noise.values())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True)
    for ax, q, label in (
        (axes[0], "phi", r"$F_\phi$ [rad$^2$ m]"),
        (axes[1], "chi", r"$F_\chi$ [m]"),
    ):
        n_q = params[RAY][f"{q}_noise"]
        if has_noise:
            ax.loglog(m_fine, F_gm[q] + n_q, "k", lw=2.2, label="truth: GM theory + noise", zorder=5)
            ax.loglog(m_fine, F_gm[q], "k", lw=1.2, ls=":", label=f"GM theory, {RAY}, {GM:g}× GM", zorder=5)
            ax.axhline(n_q, color="0.5", lw=1.0, ls="-.", label="white noise", zorder=1)
        else:
            ax.loglog(m_fine, F_gm[q], "k", lw=2.2, label=f"GM theory, {RAY}, {GM:g}× GM", zorder=5)
        for k, c in zip(METHODS, COLORS):
            m, lm, ls = m_of[k], expected[k][q], 10 ** se_jk[k][q]
            ax.loglog(m, lm, color=c, lw=1.4, marker=".", label=k)
            ax.fill_between(m, lm / ls, lm * ls, color=c, alpha=0.15, lw=0)
        ax.axvline(1 / (N_EL * E.DZ), color="0.6", lw=0.8, ls=":")
        ax.set_xlabel("vertical wavenumber [cycles/m]")
        ax.set_ylabel(label)
        ax.grid(True, which="major", alpha=0.3)
        ax.set_xlim(3e-4, 0.06)
    axes[0].set_title("phase")
    axes[1].set_title("log-amplitude")
    h, l = axes[0].get_legend_handles_labels()
    h.append(plt.matplotlib.patches.Patch(color="0.3", alpha=0.15, lw=0))
    l.append(f"±1 jackknife SE of one {N_TX}-transmission experiment")
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=9)
    title = f"Synthetic {RAY}: {N_EL} elements × {E.DZ:g} m, {N_TX} transmissions per estimate, mean of {N_MC}"
    if has_noise:
        title += f", white noise φ {noise['phi_noise']:g} rad² m, χ {noise['chi_noise']:g} m"
    fig.suptitle(title + " (dotted vertical: 1/array length)", fontsize=10)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(SP / fname, dpi=200)

    # bias of the expected value, log10(E / truth), per bin (the differenced record is one shorter, so its bins
    # differ)
    print(f"\n===== {fname} =====")
    for q in ("phi", "chi"):
        print(f"\n{q}: log10(expected / truth), then jackknife SE (RMS) and SD between experiments [decades]")
        for k in METHODS:
            m = m_of[k]
            truth = GM * model(RAY, m)[0 if q == "phi" else 1] + params[RAY][f"{q}_noise"]
            print(f"{'m [cpm]':12s} " + " ".join(f"{x:6.4f}" for x in m))
            print(f"{k[:12]:12s} " + " ".join(f"{x:6.2f}" for x in np.log10(expected[k][q] / truth)))
            print(f"{'  SE jk':12s} " + " ".join(f"{x:6.2f}" for x in se_jk[k][q]))
            print(f"{'  SD MC':12s} " + " ".join(f"{x:6.2f}" for x in se_mc[k][q]))


run({}, "spectral_methods.png")
run(NOISE, "spectral_methods_noise.png")
