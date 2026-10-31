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
        "J": along_s(1e6),  # band reaches past the top of the grid
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
    monkeypatch.setattr(
        tools,
        "wavefront_values_path",
        lambda remove_array_tilt, remove_pattern="none": path,
    )

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


# ---------------------------------------------------------------------------
# hydrophone pattern


def _true_phase_pattern(n_el=28):
    phase = np.zeros(n_el)
    phase[[4, 5, 20]] = [-1.1, -0.85, -1.1]
    return tools._linear_detrend(phase)


def _true_gain_pattern(n_el=28, seed=3):
    g = 1 + 0.1 * np.random.default_rng(seed).standard_normal(n_el)
    return g / g.mean()


def _pattern_wavefronts(rng, n_arrivals=10, n_tx=23, n_el=28, s_phi=0.1, s_chi=0.1):
    """Wavefront values with a known phase and gain pattern, plus a random amplitude, phase, tilt and noise per
    record."""
    phase, gain = _true_phase_pattern(n_el), _true_gain_pattern(n_el)
    el = np.arange(n_el)
    wavefronts = {}
    for a in range(n_arrivals):
        amp = rng.uniform(0.5, 2, (n_tx, 1))
        offset = rng.uniform(-np.pi, np.pi, (n_tx, 1))
        tilt = rng.normal(0, 0.05, (n_tx, 1)) * el
        noise = rng.normal(0, s_phi, (n_tx, n_el))
        lamp = rng.normal(0, s_chi, (n_tx, n_el))
        wavefronts[f"a{a}"] = (
            amp * gain * np.exp(lamp) * np.exp(1j * (phase + offset + tilt + noise))
        )
    return wavefronts


def test_hydrophone_pattern_recovers_known_pattern():
    wavefronts = _pattern_wavefronts(np.random.default_rng(0))
    wavefronts["-6b"] = wavefronts["a0"] * 10  # in ARRAY_TILT_OMIT, so never used
    pattern = tools.estimate_hydrophone_pattern(wavefronts)

    for key, true in (
        ("phase", _true_phase_pattern()),
        ("gain", _true_gain_pattern()),
    ):
        err = pattern[key] - true
        se = pattern[f"{key}_se"]
        assert np.all(se > 0)
        assert np.all(np.abs(err) < 5 * se + 0.01), key
        assert pattern["rejected_arrivals"][key] == {}
        assert "-6b" not in pattern["per_arrival"][key]
    np.testing.assert_allclose(pattern["delay"], pattern["phase"] / (2 * np.pi * 250))


def test_hydrophone_pattern_rejects_outliers():
    rng = np.random.default_rng(1)
    wavefronts = _pattern_wavefronts(rng)
    # one arrival with a different phase and gain at element 10
    wavefronts["a3"][:, 10] *= 1.5 * np.exp(0.5j)
    # one record with a much noisier phase and amplitude
    wavefronts["a5"][7] *= np.exp(rng.normal(0, 0.8, 28) + 1j * rng.normal(0, 1.0, 28))

    pattern = tools.estimate_hydrophone_pattern(wavefronts)
    for key in ("phase", "gain"):
        assert list(pattern["rejected_arrivals"][key]) == ["a3"]
        assert "a3" not in pattern["arrivals"][key]
        assert 7 in pattern["rejected_records"][key].get("a5", [])
    assert np.abs(pattern["phase"][10] - _true_phase_pattern()[10]) < 0.05


def _pulse(t, tau=0.0, f0=250.0, width=0.01):
    """Complex baseband Gaussian pulse delayed by tau, with the carrier phase of the delay."""
    return np.exp(-(((t - tau) / width) ** 2)) * np.exp(-2j * np.pi * f0 * tau)


def test_remove_hydrophone_pattern_is_a_time_delay():
    """The phase is removed as a delay: a pulse delayed by -delay_n on element n comes back aligned at its
    envelope and its carrier phase, and the gain is divided out."""
    n_el = 28
    t = np.arange(-1, 1, 1 / 2000)
    phase = _true_phase_pattern(n_el)
    gain = _true_gain_pattern(n_el)
    pattern = {
        "phase": phase,
        "delay": phase / (2 * np.pi * 250),
        "gain": gain,
        "f0": 250.0,
    }

    # at the carrier the pulse delayed by -delay has phase +phase
    x = gain[:, None] * np.stack([_pulse(t, -d) for d in pattern["delay"]])
    da = xr.DataArray(
        x, dims=["depth", "time"], coords={"time": t, "depth": 9.0 * np.arange(n_el)}
    )

    clean = _pulse(t)[None, :] * np.ones((n_el, 1))
    for remove_pattern, expected in (
        ("none", x),
        ("magnitude", x / gain[:, None]),
        ("phase", gain[:, None] * clean),
        ("both", clean),
    ):
        out = tools.remove_hydrophone_pattern(da, pattern, remove_pattern)
        np.testing.assert_allclose(
            out.values, expected, atol=1e-8, err_msg=remove_pattern
        )

    # a constant phase rotation would leave the envelope where it was
    rotated = x * np.exp(-1j * phase)[:, None]
    assert np.max(np.abs(rotated / gain[:, None] - clean)) > 1e-3


def test_estimate_F_phi_chi_keeps_bad_elements_when_phase_removed(
    tmp_path, monkeypatch
):
    import pickle

    rng = np.random.default_rng(2)
    n_tx, n_el = 23, 28
    wavefront_values = {
        ray_id: np.exp(
            rng.normal(0, 0.1, (n_tx, n_el)) + 1j * rng.normal(0, 0.3, (n_tx, n_el))
        )
        for ray_id in ["+7", "-7", "-6", "+6"]
    }
    path = tmp_path / "wavefront_values.pkl"
    path.write_bytes(pickle.dumps(wavefront_values))
    monkeypatch.setattr(
        tools,
        "wavefront_values_path",
        lambda remove_array_tilt, remove_pattern="none": path,
    )

    _, F_none, _, _, _, _, _ = tools.estimate_F_phi_chi()
    _, F_phase, _, _, _, _, _ = tools.estimate_F_phi_chi(remove_pattern="phase")

    phase = np.unwrap(np.angle(wavefront_values["+7"]), axis=1)
    phase = (phase - phase.mean(axis=1)[:, None]) * tools._hann_window(n_el)
    _, F_all, _ = tools._one_sided_periodogram(phase, fs=1 / 9)
    np.testing.assert_allclose(F_phase["+7"], F_all)
    assert not np.allclose(F_none["+7"], F_all)
