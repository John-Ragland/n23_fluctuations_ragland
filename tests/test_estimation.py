"""Lightweight tests for the F_phi / F_chi estimation experiments (no data needed)."""

from __future__ import annotations

import pickle
from dataclasses import replace

import numpy as np
import pytest

from n23_fluctuations_ragland import estimation, tools


def _random_wavefronts(rng, n_tx=23, n=28, ray_ids=tools.DUCTED_RAY_IDS):
    """Wavefront values with a smooth random phase, so that unwrapping is unambiguous."""
    out = {}
    for ray_id in ray_ids:
        phase = np.cumsum(0.4 * rng.standard_normal((n_tx, n)), axis=1)
        lamp = 0.3 * rng.standard_normal((n_tx, n))
        out[ray_id] = np.exp(lamp + 1j * phase)
    return out


@pytest.mark.parametrize("window", tools.SPECTRAL_WINDOWS)
def test_default_estimator_matches_estimate_F_phi_chi(tmp_path, monkeypatch, window):
    """The default `Estimator` is `tools.estimate_F_phi_chi` (Hann taper, mean removed, bad elements
    4, 5, 20 zero-filled in phase, bias-corrected log-mean), and so is each other taper."""
    wavefronts = _random_wavefronts(np.random.default_rng(1))
    path = tmp_path / "wavefront_values_tilt_removed.pkl"
    with path.open("wb") as f:
        pickle.dump(wavefronts, f)
    monkeypatch.setattr(
        tools,
        "wavefront_values_path",
        lambda remove_array_tilt, remove_pattern="none": path,
    )

    m, _, _, F_phi_lm, F_phi_ls, F_chi_lm, F_chi_ls = tools.estimate_F_phi_chi(
        remove_array_tilt=True, window=window
    )
    estimates = estimation.estimate_spectra(
        wavefronts,
        estimation.Estimator(window=window, tukey_alpha=tools.TUKEY_ALPHA),
        tools.DUCTED_RAY_IDS,
    )
    for ray_id, e in estimates.items():
        np.testing.assert_allclose(e.m, m)
        np.testing.assert_allclose(e.phi_lm, F_phi_lm[ray_id], rtol=1e-10)
        np.testing.assert_allclose(e.phi_ls, F_phi_ls[ray_id], rtol=1e-10)
        np.testing.assert_allclose(e.chi_lm, F_chi_lm[ray_id], rtol=1e-10)
        np.testing.assert_allclose(e.chi_ls, F_chi_ls[ray_id], rtol=1e-10)


@pytest.mark.parametrize("window", tools.SPECTRAL_WINDOWS)
@pytest.mark.parametrize("remove_pattern", ["none", "phase"])
def test_first_difference_matches_prewhitened_estimator(
    tmp_path, monkeypatch, window, remove_pattern
):
    """`tools.estimate_F_phi_chi(first_difference=True)` is the prewhitened `Estimator`: first difference, then
    the taper, then postcoloring, with the bad elements 4, 5, 20 linearly interpolated in phase (or kept when
    the phase pattern was removed)."""
    wavefronts = _random_wavefronts(np.random.default_rng(3))
    path = tmp_path / "wavefront_values_tilt_removed.pkl"
    with path.open("wb") as f:
        pickle.dump(wavefronts, f)
    monkeypatch.setattr(
        tools,
        "wavefront_values_path",
        lambda remove_array_tilt, remove_pattern="none": path,
    )

    m, _, _, F_phi_lm, F_phi_ls, F_chi_lm, F_chi_ls = tools.estimate_F_phi_chi(
        remove_array_tilt=True,
        window=window,
        remove_pattern=remove_pattern,
        first_difference=True,
    )
    assert len(m) == 13  # 27 differenced samples
    est = estimation.Estimator(
        prewhiten=True,
        window=window,
        tukey_alpha=tools.TUKEY_ALPHA,
        bad_element_fill="linear",
        bad_elements=() if remove_pattern == "phase" else (4, 5, 20),
    )
    estimates = estimation.estimate_spectra(wavefronts, est, tools.DUCTED_RAY_IDS)
    for ray_id, e in estimates.items():
        np.testing.assert_allclose(e.m, m)
        np.testing.assert_allclose(e.phi_lm, F_phi_lm[ray_id], rtol=1e-10)
        np.testing.assert_allclose(e.phi_ls, F_phi_ls[ray_id], rtol=1e-10)
        np.testing.assert_allclose(e.chi_lm, F_chi_lm[ray_id], rtol=1e-10)
        np.testing.assert_allclose(e.chi_ls, F_chi_ls[ray_id], rtol=1e-10)


@pytest.mark.parametrize("window", estimation.WINDOWS)
def test_tapered_spectra_white_noise_level(window):
    """Every taper keeps the one-sided density of white noise, 2 sigma^2 dz, in every bin."""
    rng = np.random.default_rng(0)
    sigma, n = 0.7, 28
    x = sigma * rng.standard_normal((20000, n))
    tapers = estimation.Estimator(window=window).tapers(n)
    _, S, dof = estimation.tapered_spectra(x, tapers)
    np.testing.assert_allclose(S.mean(axis=0), 2 * sigma**2 * estimation.DZ, rtol=0.05)
    assert dof[0] == 2 * tapers.shape[0]


def test_prewhitening_recovers_random_walk_spectrum():
    """A random walk differences to white noise, so prewhitening + postcoloring gives its spectrum,
    2 sigma^2 dz / |2 sin(pi m dz)|^2, without leakage."""
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.standard_normal((20000, 28)), axis=1)
    est = estimation.Estimator(window="rectangular", prewhiten=True)
    m, S, _ = estimation._spectra_of(x, est)
    assert len(m) == 13  # 27 differenced samples
    expected = 2 * estimation.DZ / (2 * np.sin(np.pi * m * estimation.DZ)) ** 2
    np.testing.assert_allclose(S.mean(axis=0), expected, rtol=0.05)


def test_synthesized_records_have_the_requested_spectrum():
    """Variance is the integral of the spectrum, and white input gives a flat periodogram at its level."""
    rng = np.random.default_rng(0)
    n_long, df = 256, 1 / (512 * estimation.DZ)
    S = np.full(n_long, 3.0)
    x = estimation.synthesize_records(S, df, 4000, 28, rng)
    np.testing.assert_allclose(x.var(), S.sum() * df, rtol=0.03)
    _, P, _ = estimation.tapered_spectra(
        x, estimation.Estimator(window="rectangular").tapers(28)
    )
    np.testing.assert_allclose(P.mean(axis=0), 3.0, rtol=0.06)


def test_common_pattern_removal_recovers_pattern():
    rng = np.random.default_rng(2)
    wavefronts = _random_wavefronts(rng)
    pattern = 0.5 * np.sin(np.arange(28) / 3)
    pattern -= pattern.mean()
    with_pattern = {r: X * np.exp(1j * pattern) for r, X in wavefronts.items()}

    est = estimation.Estimator(common_phase="per_ray")
    _, patterns = estimation.estimate_spectra(
        with_pattern, est, ["+7"], return_patterns=True
    )
    _, patterns0 = estimation.estimate_spectra(
        wavefronts, est, ["+7"], return_patterns=True
    )
    good = ~estimation._bad_mask(28, est.bad_elements)
    diff = patterns["phase"]["+7"][good] - patterns0["phase"]["+7"][good]
    np.testing.assert_allclose(
        diff - diff.mean(), pattern[good] - pattern[good].mean(), atol=1e-10
    )


@pytest.mark.parametrize("mode", ["per_ray", "all_rays"])
def test_constant_element_offsets_are_removed_without_unwrapping_slips(mode):
    """Offsets near pi on elements 4, 5 and 20, the same on every transmission, would make raw unwrapping
    slip. With the common phase removed (as phasors, then as a mean) the spectra are those without the
    offsets."""
    rng = np.random.default_rng(3)
    wavefronts = _random_wavefronts(rng)
    offset = np.zeros(28)
    offset[[4, 5]], offset[20] = 2.9, -2.9
    with_offsets = {r: X * np.exp(1j * offset) for r, X in wavefronts.items()}

    est = estimation.Estimator(bad_elements=(), common_phase=mode)
    clean = estimation.estimate_spectra(wavefronts, est, tools.DUCTED_RAY_IDS)
    offsets = estimation.estimate_spectra(with_offsets, est, tools.DUCTED_RAY_IDS)
    for ray_id in tools.DUCTED_RAY_IDS:
        np.testing.assert_allclose(
            offsets[ray_id].phi_S, clean[ray_id].phi_S, rtol=1e-8
        )


def test_all_arrivals_pattern_pools_arrivals_except_omitted():
    """``all_arrivals`` forms the common pattern from every arrival except ``common_omit``, so an omitted
    arrival cannot change the estimates and a kept one does, and the constant offsets are still removed."""
    rng = np.random.default_rng(4)
    ray_ids = list(tools.DUCTED_RAY_IDS)
    wavefronts = _random_wavefronts(rng, ray_ids=[*ray_ids, "+8b", "-6b"])
    offset = np.zeros(28)
    offset[[4, 5]], offset[20] = -1.1, -1.1
    with_offsets = {r: X * np.exp(1j * offset) for r, X in wavefronts.items()}

    est = estimation.Estimator(bad_elements=(), common_phase="all_arrivals")
    assert "-6b" in est.common_omit
    clean = estimation.estimate_spectra(wavefronts, est, ray_ids)
    offsets = estimation.estimate_spectra(with_offsets, est, ray_ids)
    np.testing.assert_allclose(offsets["+7"].phi_S, clean["+7"].phi_S, rtol=1e-8)

    rows = estimation.common_pattern_rows(wavefronts, est, ray_ids, "all_arrivals")
    assert set(rows) == {*ray_ids, "+8b"}

    def changed(ray_id):
        other = dict(wavefronts)
        other[ray_id] = _random_wavefronts(rng, ray_ids=[ray_id])[ray_id]
        return estimation.estimate_spectra(other, est, ray_ids)["+7"].phi_S

    np.testing.assert_allclose(changed("-6b"), clean["+7"].phi_S, rtol=1e-12)
    assert not np.allclose(changed("+8b"), clean["+7"].phi_S)


def test_fit_gm_recovers_parameters():
    m = np.fft.rfftfreq(28, estimation.DZ)[1:]
    F1_phi, F1_chi = 0.01 * m**-2, 0.002 * m**-1.5
    GM, phi_noise, chi_noise = 0.6, 1.5, 0.3
    ls = np.full(len(m), 1.2)
    fit = estimation.fit_gm(
        F1_phi, F1_chi, GM * F1_phi + phi_noise, ls, GM * F1_chi + chi_noise, ls
    )
    np.testing.assert_allclose(
        [fit.GM, fit.phi_noise, fit.chi_noise], [GM, phi_noise, chi_noise], rtol=1e-3
    )


def test_template_config_loads(tmp_path):
    path = estimation.write_config_template(tmp_path / "config.yaml")
    cfg = estimation.load_config(path)
    assert cfg.output_dir == tmp_path / "estimation_experiments"
    assert cfg.base.bad_elements == ()
    assert cfg.base.common_phase == "all_arrivals"
    assert cfg.base.common_omit == tools.ARRAY_TILT_OMIT
    reference = next(e for e in cfg.estimators if e.name.startswith("reference"))
    assert replace(reference, name="current") == estimation.Estimator()
    assert {e.window for e in cfg.estimators} == set(estimation.WINDOWS)


def test_config_sweep_and_errors(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("sweep:\n  window: [hann, tukey]\n  detrend: [constant, linear]\n")
    cfg = estimation.load_config(path)
    assert len(cfg.estimators) == 4
    assert cfg.estimators[-1].name == "window=tukey, detrend=linear"

    path.write_text("variants:\n  - name: a\n    windw: hann\n")
    with pytest.raises(estimation.ConfigError, match="windw"):
        estimation.load_config(path)

    path.write_text("variants:\n  - name: a\n    window: kaiser\n")
    with pytest.raises(estimation.ConfigError, match="window"):
        estimation.load_config(path)


def test_menu_has_estimation_experiment_options():
    import inspect

    import n23_fluctuations_ragland.cli as cli_mod

    source = inspect.getsource(cli_mod.main)
    for choice in (
        "create estimation experiments config",
        "run estimation experiments",
    ):
        assert source.count(f'"{choice}"') == 2, (
            f"menu is missing or not wired: {choice}"
        )
