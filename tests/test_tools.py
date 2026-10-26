"""Lightweight tests for the weak fluctuation theory in tools (no data needed)."""

from __future__ import annotations

import numpy as np
import scipy.integrate
import xarray as xr

from n23_fluctuations_ragland import tools


def test_F_phi_chi_is_a_density_per_cycle_per_meter():
    """
    Integrated over cycles/m, `compute_F_phi_chi` must give the integral of Colosi (2016) Eq. 6.52 over
    rad/m, i.e. it returns Eq. 6.52 times the 2π Jacobian.

    The ray is straight with constant properties and no Fresnel filtering (R_f = 0), so F_phi = 2 x the
    Eq. 6.52 kernel and F_chi = 0, and the m-integral has the closed form ln(1 + m*^2/m_1^2) / (2 m*).
    """
    n_s, ds = 11, 100.0
    N, N0B, theta, mu2, f = 5e-3, 2.5, 0.1, 1e-8, 9.216181213401561e-05
    q0 = 2 * np.pi * 250 / 1500
    m1 = np.pi * N / N0B  # mode 1 [rad/m]
    mstar = 3 * m1  # jstar = 3

    def along_s(value):
        return xr.DataArray(np.full(n_s, value), dims=["s"])

    rv = {
        "Nray": along_s(N),
        "N0B_ray": along_s(N0B),
        "Theta_ray": along_s(theta),
        "mu2": along_s(mu2),
        "Rf2": along_s(0.0),
        "drdt": np.full(n_s, ds),
        "dzdt": np.zeros(n_s),
        "m_lo_max": 0.999 * m1 / (2 * np.pi),
        "m_hi_max": 1e6,
    }

    # W(N, sigma_L, f) from its defining integral (Eq. 6.53), with sigma^2 = sigma_L^2 + u^2
    sigma_L2 = f**2 + N**2 * np.tan(theta) ** 2

    def W_integrand(u):
        sigma2 = sigma_L2 + u**2
        return sigma_L2 * np.sqrt(sigma2 - f**2) / sigma2**2

    W, _ = scipy.integrate.quad(W_integrand, 0, np.sqrt(N**2 - sigma_L2))

    path_length = ds * (n_s - 1)
    expected = (
        2  # 1 + cos(0)
        * 8
        * q0**2
        / np.pi**2
        * mu2
        * N
        * f
        / sigma_L2
        * W
        * path_length
        * np.log(1 + mstar**2 / m1**2)
        / (2 * mstar)
    )

    m_cpm = np.logspace(
        np.log10(m1 / (2 * np.pi)), np.log10(1e5 * mstar / (2 * np.pi)), 40000
    )
    F_phi, F_chi = tools.compute_F_phi_chi(rv, 2 * np.pi * m_cpm[None, :])

    np.testing.assert_allclose(
        scipy.integrate.trapezoid(F_phi, m_cpm), expected, rtol=1e-3
    )
    np.testing.assert_allclose(F_chi, 0, atol=1e-12 * F_phi.max())


def _expected_white_periodogram(operator, sigma2: float, dz: float = 9.0):
    """Expected one-sided periodogram (DC dropped, Nyquist doubled) of white noise mapped by `operator`."""
    n = operator.shape[0]
    E = np.fft.fft(np.eye(n), axis=0) @ operator
    S = 2 * dz / n * sigma2 * np.sum(np.abs(E) ** 2, axis=1)
    return S[1 : n // 2 + 1]


def test_estimate_F_phi_chi_hann_is_unbiased_for_white_noise(tmp_path, monkeypatch):
    """
    With the Hann taper, the estimated spectra of white phase and log-amplitude match the exact expected
    periodogram of the estimator (mask/zero-fill of elements 4, 5, 20, demean, unit-mean-square Hann).
    """
    import pickle

    n_tx, n_el, s_phi, s_chi = 400, 28, 0.3, 0.2
    rng = np.random.default_rng(0)
    wavefront_values = {
        ray_id: np.exp(
            rng.normal(0, s_chi, (n_tx, n_el)) + 1j * rng.normal(0, s_phi, (n_tx, n_el))
        )
        for ray_id in ["+7", "-7", "-6", "+6"]
    }
    path = tmp_path / "wavefront_values.pkl"
    path.write_bytes(pickle.dumps(wavefront_values))
    monkeypatch.setattr(tools, "wavefront_values_path", lambda remove_array_tilt: path)

    m, _, _, F_phi_lm, _, F_chi_lm, _ = tools.estimate_F_phi_chi(hann=True)

    window = tools._hann_window(n_el)
    np.testing.assert_allclose(np.mean(window**2), 1)

    keep = np.ones(n_el)
    keep[[4, 5, 20]] = 0
    demean_masked = np.diag(keep) @ (
        np.eye(n_el) - np.outer(np.ones(n_el), keep) / keep.sum()
    )
    demean = np.eye(n_el) - 1 / n_el
    S_phi = _expected_white_periodogram(window[:, None] * demean_masked, s_phi**2)
    S_chi = _expected_white_periodogram(window[:, None] * demean, s_chi**2)

    for ray_id in F_phi_lm:
        # geometric mean of observed / expected over the bins
        np.testing.assert_allclose(
            np.exp(np.mean(np.log(F_phi_lm[ray_id] / S_phi))), 1, rtol=0.05
        )
        np.testing.assert_allclose(
            np.exp(np.mean(np.log(F_chi_lm[ray_id] / S_chi))), 1, rtol=0.05
        )
    # and white noise keeps its level, 2 sigma^2 dz, away from the lowest bins
    np.testing.assert_allclose(
        np.median(S_chi[2:]) / (2 * s_chi**2 * 9.0), 1, rtol=0.05
    )
