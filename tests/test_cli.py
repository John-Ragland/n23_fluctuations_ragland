"""Lightweight tests for the n23_fluctuation_analysis CLI.

These tests verify wiring, imports, and decorator logic without running any
heavy scientific computation.
"""

from __future__ import annotations

import pytest


def test_cli_imports():
    """All CLI functions can be imported without error."""
    from n23_fluctuations_ragland import cli

    for name in [
        "compute_arrival_time_offset",
        "compute_mean_pulse",
        "compute_rays",
        "estimate_peak_locations",
        "estimate_array_tilt",
        "get_wavefront_values",
        "phi_chi_LS_fit",
        "run_all_processing",
        "source_geometry_sweep",
        "create_publication_figures_config",
        "generate_publication_figures",
        "coherence_lengths",
        "independence_test",
    ]:
        assert hasattr(cli, name), f"cli.{name} is missing"


def test_tools_has_required_functions():
    """Tools module exposes every function the CLI calls."""
    from n23_fluctuations_ragland import tools

    required = [
        "open_receptions",
        "get_wavefront_values",
        "estimate_array_tilt",
        "open_array_tilt",
        "run_rays",
        "compute_mean_pulse",
        "estimate_F_phi_chi",
        "fit_F_phi_chis",
        "compute_filtered_fctd_grid",
        "fit_vertical_coherence",
        "jackknife_vertical_coherence",
        "transmission_independence_test",
        "independence_test_F_phi_chi",
        "independence_test_vertical_coherence",
        "estimate_observed_coherence_lengths",
        "compute_theoretical_coherence_lengths",
        "coherence_lengths_latex_table",
    ]
    for name in required:
        assert hasattr(tools, name), f"tools.{name} is missing"


def test_convert_to_xarray_not_on_tools():
    """
    cli.py must import convert_to_xarray from nescafe23, not tools.
    This was a real bug where tools.convert_to_xarray was called and
    would raise AttributeError at runtime.
    """
    from n23_fluctuations_ragland import tools

    assert not hasattr(tools, "convert_to_xarray"), (
        "convert_to_xarray should be imported from nescafe23, not accessed via tools"
    )

    from nescafe23 import convert_to_xarray

    assert convert_to_xarray is not None


def test_check_output_exists_runs_when_no_file(tmp_path):
    """Decorated function runs when output file does not exist."""
    from n23_fluctuations_ragland.cli import check_output_exists

    output_file = tmp_path / "out.pkl"
    calls = []

    @check_output_exists(output_file)
    def my_func():
        calls.append(1)

    my_func()
    assert calls == [1]


def test_check_output_exists_force_overwrite(tmp_path):
    """force_overwrite=True bypasses the existence check."""
    from n23_fluctuations_ragland.cli import check_output_exists

    output_file = tmp_path / "out.pkl"
    output_file.touch()
    calls = []

    @check_output_exists(output_file)
    def my_func():
        calls.append(1)

    my_func(force_overwrite=True)
    assert calls == [1]


def test_check_output_exists_skips_existing(tmp_path, monkeypatch):
    """Decorated function is skipped when file exists and user declines overwrite."""
    from n23_fluctuations_ragland.cli import check_output_exists

    output_file = tmp_path / "out.pkl"
    output_file.touch()
    calls = []

    # Simulate the user answering "no" to the questionary prompt
    monkeypatch.setattr(
        "n23_fluctuations_ragland.cli.questionary.confirm",
        lambda *a, **kw: type("Q", (), {"ask": staticmethod(lambda: False)})(),
    )

    @check_output_exists(output_file)
    def my_func():
        calls.append(1)

    my_func()
    assert calls == []


def test_run_all_processing_has_all_steps():
    """run_all_processing includes every processing step."""
    import inspect

    from n23_fluctuations_ragland import cli

    source = inspect.getsource(cli.run_all_processing)

    expected_funcs = [
        "compute_arrival_time_offset",
        "estimate_peak_locations",
        "estimate_array_tilt",
        "get_wavefront_values",
        "compute_mean_pulse",
        "compute_rays",
        "source_geometry_sweep",
        "phi_chi_LS_fit",
    ]
    for func_name in expected_funcs:
        assert func_name in source, f"run_all_processing is missing step: {func_name}"


def test_arrival_time_offset_saves_per_array_offsets():
    """compute_arrival_time_offset pickles the upper/lower offsets the publication figure plots."""
    import inspect

    import n23_fluctuations_ragland.cli as cli_mod

    source = inspect.getsource(cli_mod.compute_arrival_time_offset)
    for key in ('"dT_t"', '"dT_t_upper"', '"dT_t_lower"'):
        assert key in source, f"T_t_estimated.pkl must contain {key}"


def test_menu_has_publication_figure_options():
    """The interactive menu offers creating a config and generating the figures."""
    import inspect

    import n23_fluctuations_ragland.cli as cli_mod

    source = inspect.getsource(cli_mod.main)
    for choice in ("create publication figures config", "generate publication figures"):
        assert source.count(f'"{choice}"') == 2, (
            f"menu is missing or not wired: {choice}"
        )


def test_estimate_F_phi_chi_uses_get_data_dir():
    """estimate_F_phi_chi must use get_data_dir() not os.environ directly."""
    import inspect

    from n23_fluctuations_ragland import tools

    source = inspect.getsource(tools.estimate_F_phi_chi)
    assert "os.environ['NESCAFE23_DATADIR']" not in source, (
        "estimate_F_phi_chi must use get_data_dir() instead of "
        "os.environ['NESCAFE23_DATADIR'] to avoid missing path separator"
    )
    assert "wavefront_values_path" in source


def test_fit_coherence_length_recovers_gaussian():
    """The coherence length fit recovers A0 and L, with and without the zero lag."""
    import numpy as np

    from n23_fluctuations_ragland import tools

    for lags in [np.arange(0, 250, 9.0), np.arange(9, 250, 9.0)]:
        MC = tools.gaussian_coherence_model(lags, 0.8, 120.0)
        A0, L = tools.fit_coherence_length(lags, MC, L0=100.0)
        assert np.isclose(A0, 0.8)
        assert np.isclose(L, 120.0)


def test_coherence_lengths_latex_table():
    """One row per eigenray pair, missing values as --."""
    from n23_fluctuations_ragland import tools

    observed = {"-6": {"z0": 100.0, "t0": 3600.0}, "+7": {"z0": 50.0, "t0": 7200.0}}
    theory = {
        "-6": [{"z0": 90.0, "t0": 1800.0}, {"z0": 95.0, "t0": 1900.0}],
        "-8": [{"z0": 80.0, "t0": 1200.0}],
    }
    table = tools.coherence_lengths_latex_table(observed, theory)

    assert "\\begin{tabular}" in table
    rows = [line for line in table.splitlines() if line.startswith("$")]
    assert len(rows) == 4
    assert rows[0] == "$-8$ & 0 & -- & 80.0 & -- & 20.0 \\\\"
    assert rows[1] == "$+7$ & -- & 50.0 & -- & 120.0 & -- \\\\"
    assert rows[2] == "$-6$ & 0 & 100.0 & 90.0 & 60.0 & 30.0 \\\\"
    assert rows[3].startswith("$-6$ & 1 & -- & 95.0")

    observed["+7"]["z0_std"] = 4.25
    table = tools.coherence_lengths_latex_table(observed, theory)
    rows = [line for line in table.splitlines() if line.startswith("$")]
    assert rows[1] == "$+7$ & -- & 50.0 $\\pm$ 4.2 & -- & 120.0 & -- \\\\"


def test_fit_vertical_coherence_recovers_gaussian():
    """The fit to the unnormalized coherence ignores the noise-biased zero lag."""
    import numpy as np

    from n23_fluctuations_ragland import tools

    dz = np.arange(0, 32 * 9.0, 9.0)
    MC = tools.gaussian_coherence_model(dz, 4e11, 120.0)
    MC[0] *= 1.5  # noise at lag 0
    A0, z0 = tools.fit_vertical_coherence(dz, MC)
    assert np.isclose(A0, 4e11)
    assert np.isclose(z0, 120.0)


def test_estimate_array_tilt_recovers_tilt():
    """The tilt common to all rays is recovered, omitted rays are excluded but still get angles."""
    import numpy as np

    from n23_fluctuations_ragland import tools

    rng = np.random.default_rng(0)
    tilt = rng.normal(0, 1, 23)
    angles = {
        "+7": 5.0 + tilt,
        "-7": -5.0 + tilt,
        "+10b": 12.0 + tilt,
        "-15b": -20.0 + rng.normal(0, 3, 23),
    }

    result = tools.estimate_array_tilt({"angles": angles}, omit=("-15b",))

    assert np.allclose(result["array_tilt"], tilt - tilt.mean())
    assert result["omit"] == ("-15b",)
    assert set(result["angles_plus_tilt"]) == set(angles)
    for ray_id, angle in angles.items():
        assert np.isclose(result["mean_angles"][ray_id], angle.mean())
        assert np.allclose(
            result["angles_plus_tilt"][ray_id], angle.mean() + result["array_tilt"]
        )


def test_get_wavefront_values_signature():
    """get_wavefront_values reads its own picks and selects the angles with remove_array_tilt."""
    import inspect

    from n23_fluctuations_ragland import tools

    params = inspect.signature(tools.get_wavefront_values).parameters
    assert params["remove_array_tilt"].default is False
    assert "arrival_times" not in params
    assert "arrival_angles" not in params


def test_wavefront_values_paths_per_mode(tmp_path, monkeypatch):
    """Tilt-removed wavefront values are written to their own file."""
    import n23_fluctuations_ragland.cli as cli_mod

    monkeypatch.setattr(cli_mod.tools, "get_data_dir", lambda: tmp_path)
    assert cli_mod._wavefront_values_path(False).name == "wavefront_values.pkl"
    assert (
        cli_mod._wavefront_values_path(True).name == "wavefront_values_tilt_removed.pkl"
    )


def test_phi_chi_LS_fit_paths_per_mode(tmp_path, monkeypatch):
    """Tilt-removed ɸ χ fits are written to their own file."""
    from n23_fluctuations_ragland import tools

    monkeypatch.setattr(tools, "get_data_dir", lambda: tmp_path)
    assert (
        tools.phi_chi_LS_fit_path(False)
        == tmp_path / "dataproducts" / "phi_chi_LS_fit.pkl"
    )
    assert (
        tools.phi_chi_LS_fit_path(True)
        == tmp_path / "dataproducts" / "phi_chi_LS_fit_tilt_removed.pkl"
    )


@pytest.mark.parametrize("remove_array_tilt", [False, True])
def test_phi_chi_LS_fit_uses_matching_wavefront_values(
    tmp_path, monkeypatch, remove_array_tilt
):
    """The fit estimates the spectra in the requested mode and saves to that mode's file."""
    import pickle

    import n23_fluctuations_ragland.cli as cli_mod

    (tmp_path / "dataproducts").mkdir()
    monkeypatch.setattr(cli_mod.tools, "get_data_dir", lambda: tmp_path)

    calls = []

    def fake_estimate(remove_array_tilt=False):
        calls.append(remove_array_tilt)
        return "m", None, None, "phi_lm", "phi_ls", "chi_lm", "chi_ls"

    monkeypatch.setattr(cli_mod.tools, "estimate_F_phi_chi", fake_estimate)
    monkeypatch.setattr(cli_mod.tools, "fit_F_phi_chis", lambda *args: {"args": args})

    cli_mod.phi_chi_LS_fit(remove_array_tilt=remove_array_tilt, force_overwrite=True)

    assert calls == [remove_array_tilt]
    with cli_mod.tools.phi_chi_LS_fit_path(remove_array_tilt).open("rb") as f:
        assert pickle.load(f) == {"args": ("m", "phi_lm", "chi_lm", "phi_ls", "chi_ls")}
    assert not cli_mod.tools.phi_chi_LS_fit_path(not remove_array_tilt).exists()


def test_run_all_processing_fits_both_modes():
    """run_all_processing fits the ɸ χ spectra with and without the array tilt removed."""
    import inspect

    from n23_fluctuations_ragland import cli

    source = inspect.getsource(cli.run_all_processing)
    assert source.count("phi_chi_LS_fit,") == 2
    assert source.count('{"remove_array_tilt": True}') == 2


def test_peak_locations_saves_upper_array():
    """peak_locations.pkl keeps the upper-array keys that n23_inversion_ragland reads."""
    import inspect

    import n23_fluctuations_ragland.cli as cli_mod

    source = inspect.getsource(cli_mod.estimate_peak_locations)
    for key in ('"upper_times"', '"upper_angles"', '"upper_snrs"'):
        assert key in source, f"peak_locations.pkl must contain {key}"


def test_effective_sample_size():
    """M_eff is M for independent snapshots and M dt / (sqrt(pi) t0) for a long record."""
    import numpy as np

    from n23_fluctuations_ragland import tools

    assert tools.effective_sample_size(23) == 23
    assert tools.effective_sample_size(23, 3600.0, 0) == 23
    assert np.isclose(
        tools.effective_sample_size(2000, 1.0, 10.0),
        2000 / (np.sqrt(np.pi) * 10.0),
        rtol=0.01,
    )
    assert 1 <= tools.effective_sample_size(23, 3600.0, 3 * 3600.0) <= 23


def _simulate_snapshots(rng, M, N, z0, t0, noise_std):
    """
    Circular complex Gaussian snapshots, Gaussian coherence in both dims plus noise that is white in
    depth. The noise has the same time coherence as the signal, so the field is separable.
    """
    import numpy as np

    def chol(n, L):
        d = np.subtract.outer(np.arange(n), np.arange(n))
        return np.linalg.cholesky(np.exp(-0.5 * (d / L) ** 2) + 1e-8 * np.eye(n))

    Lz, Lt = chol(N, z0), chol(M, t0)
    W = (rng.standard_normal((M, N)) + 1j * rng.standard_normal((M, N))) / np.sqrt(2)
    noise = (rng.standard_normal((M, N)) + 1j * rng.standard_normal((M, N))) / np.sqrt(
        2
    )
    return Lt @ (W @ Lz.T + noise_std * noise)


def test_mutual_coherence_variance_monte_carlo():
    """The Bartlett variance matches the Monte Carlo variance, with and without time correlation."""
    import numpy as np

    from n23_fluctuations_ragland import tools

    rng = np.random.default_rng(0)
    M, N, z0, n_real = 40, 16, 3.0, 400

    for t0 in [None, 3.0]:
        for norm, omit_zero_lag in [(False, False), (True, True)]:
            MCs, MC_vars = [], []
            for _ in range(n_real):
                X = _simulate_snapshots(rng, M, N, z0, t0 or 1e-6, noise_std=0.5)
                _, MC, MC_var, _, _ = tools.estimate_mutual_coherence(
                    X, 1.0, omit_zero_lag=omit_zero_lag, norm=norm, dt=1.0, t0=t0
                )
                MCs.append(MC)
                MC_vars.append(MC_var)

            empirical = np.var(MCs, axis=0)
            predicted = np.mean(MC_vars, axis=0)
            # the reference lag has zero variance when normalized
            check = slice(1, N // 2) if norm else slice(0, N // 2)
            if norm:
                assert np.allclose(predicted[0], 0)
            np.testing.assert_allclose(predicted[check], empirical[check], rtol=0.25)


def test_jackknife_vertical_coherence():
    """Leave-one-transmission-out jackknife of the vertical coherence and its Gaussian fit."""
    import numpy as np

    from n23_fluctuations_ragland import tools

    rng = np.random.default_rng(0)
    M, N = 23, 32
    # unit-variance field with z0 = 10 samples, scaled to the ~4e11 power of the wavefront values
    X = 6e5 * _simulate_snapshots(rng, M, N, z0=10.0, t0=1e-6, noise_std=0.3)

    vc = tools.jackknife_vertical_coherence(X, 9.0)

    assert vc["lags"].shape == vc["MC"].shape == vc["VC"].shape == vc["VC_var"].shape
    assert vc["MC_jk"].shape == (M, N)
    assert vc["A0_jk"].shape == vc["z0_jk"].shape == (M,)
    np.testing.assert_allclose(vc["VC"], vc["MC"] / vc["A0"])
    assert np.isclose(vc["z0"], 90.0, rtol=0.3)
    assert np.all(vc["VC_var"] >= 0)
    assert np.isfinite(vc["z0_var"])
    assert vc["z0_var"] > 0

    # MC is a mean over transmissions, for which the jackknife variance is exactly var / M
    per_transmission = np.array(
        [
            tools.estimate_mutual_coherence(
                X[k : k + 1], 9.0, omit_zero_lag=False, norm=False
            )[1]
            for k in range(M)
        ]
    )
    np.testing.assert_allclose(
        tools._jackknife_var(vc["MC_jk"]),
        np.var(per_transmission, axis=0, ddof=1) / M,
    )


def test_observed_coherence_lengths_report_jackknife_std():
    """The observed z0 comes from the jackknifed fit, with its standard deviation."""
    import numpy as np

    from n23_fluctuations_ragland import tools

    rng = np.random.default_rng(1)
    X = 6e5 * _simulate_snapshots(rng, 23, 32, z0=10.0, t0=1e-6, noise_std=0.3)

    observed = tools.estimate_observed_coherence_lengths({"-6": X})
    vc = tools.jackknife_vertical_coherence(X, 9.0)
    assert observed["-6"]["z0"] == vc["z0"]
    assert np.isclose(observed["-6"]["z0_std"], np.sqrt(vc["z0_var"]))


def test_log_periodogram_bias_and_dof():
    """Bias of log10 of a chi-square periodogram bin, and the 1-dof Nyquist bin of an even record."""
    import numpy as np

    from n23_fluctuations_ragland import tools

    assert np.isclose(tools._log_periodogram_bias(2), -np.euler_gamma / np.log(10))
    assert np.isclose(
        tools._log_periodogram_bias(1), -(np.euler_gamma + np.log(2)) / np.log(10)
    )
    np.testing.assert_array_equal(tools._periodogram_dof(28, 14), [2.0] * 13 + [1.0])
    np.testing.assert_array_equal(tools._periodogram_dof(27, 13), [2.0] * 13)


def test_jackknife_log_mean_is_unbiased_for_white_noise():
    """The corrected log-mean recovers a white spectrum; the uncorrected one is low by exp(-gamma)."""
    import numpy as np

    from n23_fluctuations_ragland import tools

    rng = np.random.default_rng(0)
    M, N, sigma, fs = 23, 28, 2.0, 1 / 9
    true_level = 2 * sigma**2 / fs  # one-sided PSD of white noise

    lms, raw = [], []
    for _ in range(400):
        x = sigma * rng.standard_normal((M, N))
        _, F, dof = tools._one_sided_periodogram(x, fs)
        lm, ls = tools._jackknife_log_mean(F, dof)
        lms.append(np.log10(lm))
        raw.append(np.mean(np.log10(F), axis=0))
        # the jackknife standard error of a mean is the standard error
        np.testing.assert_allclose(
            np.log10(ls), np.std(np.log10(F), axis=0, ddof=1) / np.sqrt(M)
        )

    # every bin, including the 1-dof Nyquist bin, estimates the one-sided level
    true = np.log10(np.full(14, true_level))
    np.testing.assert_allclose(np.mean(lms, axis=0), true, atol=0.02)
    np.testing.assert_allclose(
        np.mean(raw, axis=0)[:-1] - true[:-1], -np.euler_gamma / np.log(10), atol=0.02
    )


def test_transmission_independence_test():
    """Uncorrelated rows sit at the null; strongly autocorrelated rows are detected."""
    import numpy as np

    from n23_fluctuations_ragland import tools

    rng = np.random.default_rng(0)
    M, K = 23, 20

    iid = tools.transmission_independence_test(rng.standard_normal((M, K)), n_perm=500)
    assert iid["r1"].shape == (K,)
    assert np.isclose(iid["band"], 2 / np.sqrt(M))
    assert np.isclose(iid["null_mean"], -1 / M)
    assert abs(iid["mean_r1"] - iid["null_mean"]) < 0.15
    assert iid["p"] > 0.05

    ar = np.zeros((M, K))
    ar[0] = rng.standard_normal(K)
    for k in range(1, M):
        ar[k] = 0.8 * ar[k - 1] + rng.standard_normal(K)
    corr = tools.transmission_independence_test(ar, n_perm=500)
    assert corr["mean_r1"] > 0.4
    assert corr["p"] < 0.01
    assert corr["n_outside"] > K // 2


def test_independence_test_vertical_coherence_shapes():
    """One column per lag up to max_lag for every ducted ray."""
    import numpy as np

    from n23_fluctuations_ragland import tools

    rng = np.random.default_rng(0)
    wavefront_values = {
        ray_id: _simulate_snapshots(rng, 23, 28, z0=10.0, t0=1e-6, noise_std=0.3)
        for ray_id in tools.DUCTED_RAY_IDS
    }
    result = tools.independence_test_vertical_coherence(
        wavefront_values, dz=9.0, max_lag=180.0, n_perm=100
    )
    assert result["pooled"]["r1"].shape == (4 * 21,)
    assert set(result["per_ray"]) == set(tools.DUCTED_RAY_IDS)
