"""Lightweight tests for the n23_fluctuation_analysis CLI.

These tests verify wiring, imports, and decorator logic without running any
heavy scientific computation.
"""

from __future__ import annotations


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
    assert "get_data_dir" in source


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

    monkeypatch.setattr(cli_mod, "get_data_dir", lambda: tmp_path)
    assert cli_mod._wavefront_values_path(False).name == "wavefront_values.pkl"
    assert (
        cli_mod._wavefront_values_path(True).name == "wavefront_values_tilt_removed.pkl"
    )


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
