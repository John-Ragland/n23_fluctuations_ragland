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
        "estimate_arrival_time_offset_error",
        "estimate_hydrophone_pattern",
        "compute_wavefront_values_and_pattern",
        "create_publication_figures_config",
        "generate_publication_figures",
        "coherence_lengths",
        "reproduce_paper",
        "check_paper_inputs",
    ]:
        assert hasattr(cli, name), f"cli.{name} is missing"


def test_exploratory_entries_are_not_in_the_cli():
    """Analysis that is not in the paper has no CLI entry; its code stays in tools / estimation."""
    from n23_fluctuations_ragland import cli, tools

    for name in [
        "run_all_processing",
        "source_geometry_sweep",
        "phi_chi_LS_fit",
        "gm_levels",
        "independence_test",
        "run_estimation_experiments",
    ]:
        assert not hasattr(cli, name), f"cli.{name} should not be in the paper CLI"
    for name in ["source_geometry_sweep", "compute_phi_chi_LS_fit", "fit_F_phi_chis"]:
        assert hasattr(tools, name), f"tools.{name} is missing"


def test_menu_starts_with_reproduce_paper():
    import inspect

    from n23_fluctuations_ragland import cli

    assert cli.MENU_CHOICES[0] == cli.MENU_REPRODUCE
    assert cli.MENU_CHOICES[-1] == "exit"
    source = inspect.getsource(cli.main)
    for choice in cli.MENU_CHOICES[1:]:
        assert f'"{choice}"' in source, f"menu choice not wired: {choice}"


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
        "structure_function_from_coherence",
        "fit_GM_structure_function",
        "fit_GM_vertical_coherence",
        "GM_levels_latex_table",
        "jackknife_F_phi_chi_fit",
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
        assert choice in cli_mod.MENU_CHOICES
        assert source.count(f'"{choice}"') == 1, f"menu choice not wired: {choice}"


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


def test_structure_function_from_coherence():
    """D = -2 ln VC on the positive lags up to max_lag, jackknifed through MC_k / A0_k."""
    import numpy as np

    from n23_fluctuations_ragland import tools

    rng = np.random.default_rng(2)
    X = 6e5 * _simulate_snapshots(rng, 23, 28, z0=10.0, t0=1e-6, noise_std=0.3)
    vc = tools.jackknife_vertical_coherence(X, 9.0)

    sf = tools.structure_function_from_coherence(vc, max_lag=126.0)

    assert sf["lags"][0] == 9.0
    assert sf["lags"][-1] <= 126.0
    keep = np.isin(vc["lags"], sf["lags"])
    np.testing.assert_allclose(sf["D"], -2 * np.log(vc["VC"][keep]))
    np.testing.assert_allclose(
        sf["D_var"],
        tools._jackknife_var(-2 * np.log(vc["MC_jk"][:, keep] / vc["A0_jk"][:, None])),
    )
    assert sf["D_jk"].shape == (23, len(sf["lags"]))


def test_fit_GM_structure_function_recovers_linear_level():
    """With D linear in GM, the fit recovers GM and the linearized jackknife is the exact refit jackknife."""
    import numpy as np

    from n23_fluctuations_ragland import tools

    rng = np.random.default_rng(3)
    lags = 9.0 * np.arange(1, 15)
    D1 = (lags / 100) ** 2  # model at GM = 1
    D_jk = 2.5 * D1 + 0.01 * rng.standard_normal((23, len(lags)))
    D = D_jk.mean(axis=0)
    sf = {"lags": lags, "D": D, "D_var": tools._jackknife_var(D_jk), "D_jk": D_jk}

    fit = tools.fit_GM_structure_function(sf, lambda GM: GM * D1)

    w = 1 / sf["D_var"]
    np.testing.assert_allclose(
        fit["GM"], np.sum(w * D * D1) / np.sum(w * D1**2), rtol=1e-4
    )
    assert np.isclose(fit["GM"], 2.5, rtol=0.05)
    GM_jk = D_jk @ (w * D1) / np.sum(w * D1**2)
    np.testing.assert_allclose(
        fit["GM_std"], np.sqrt(tools._jackknife_var(GM_jk)), rtol=1e-3
    )
    assert fit["n_lags"] == len(lags)


def test_GM_levels_latex_table():
    """WFT and coherence GM levels per ray with their standard errors, missing fits as --."""
    from n23_fluctuations_ragland import tools

    spectral = {"-7": {"GM": 1.933, "GM_std": 0.497}}
    coherence = {"-7": {"GM": 0.744, "GM_std": 0.206}}
    table = tools.GM_levels_latex_table(spectral, coherence)

    rows = [line for line in table.splitlines() if line.startswith("$")]
    assert [row.split(" & ")[0] for row in rows] == ["$+7$", "$-7$", "$+6$", "$-6$"]
    assert rows[1] == "$-7$ & 1.93 $\\pm$ 0.50 & 0.74 $\\pm$ 0.21 \\\\"
    assert rows[0] == "$+7$ & -- & -- \\\\"

    rows = [
        line
        for line in tools.GM_levels_latex_table(None, coherence).splitlines()
        if line.startswith("$-7$")
    ]
    assert rows == ["$-7$ & -- & 0.74 $\\pm$ 0.21 \\\\"]


def test_jackknife_F_phi_chi_fit_matches_refit():
    """The linearized jackknife of the spectral fit is close to refitting without each transmission."""
    from types import SimpleNamespace

    import numpy as np
    from scipy import optimize

    from n23_fluctuations_ragland import tools

    rng = np.random.default_rng(4)
    m_data = np.arange(1, 15) / (28 * 9)
    A = 0.02 * m_data**-1.5  # F_phi at GM = 1, cheap stand-in for the WFT model
    B = 0.002 * m_data**-0.5

    def model(m, GM, phi_noise, chi_noise, ray_id):
        return GM * A + phi_noise, GM * B + chi_noise, GM * A, GM * B

    M = 23
    dof = tools._periodogram_dof(28, len(m_data))
    F_phi = (2.0 * A + 1.0) * rng.chisquare(2, (M, len(m_data))) / 2
    F_chi = (2.0 * B + 0.1) * rng.chisquare(2, (M, len(m_data))) / 2

    def fit(F_phi, F_chi):
        phi_lm, phi_ls = tools._jackknife_log_mean(F_phi, dof)
        chi_lm, chi_ls = tools._jackknife_log_mean(F_chi, dof)

        def cost(p):
            Fp, Fc, _, _ = model(m_data, *p, "-7")
            return np.sum(
                (np.log10(Fp) - np.log10(phi_lm)) ** 2 / np.log10(phi_ls) ** 2
            ) + np.sum((np.log10(Fc) - np.log10(chi_lm)) ** 2 / np.log10(chi_ls) ** 2)

        return (
            optimize.minimize(
                cost,
                [1, 1, 0.1],
                method="Nelder-Mead",
                options={"xatol": 1e-8, "fatol": 1e-10},
            ),
            phi_ls,
            chi_ls,
        )

    result, phi_ls, chi_ls = fit(F_phi, F_chi)
    jk = tools.jackknife_F_phi_chi_fit(
        {"-7": SimpleNamespace(x=result.x)},
        m_data,
        {"-7": F_phi},
        {"-7": F_chi},
        {"-7": phi_ls},
        {"-7": chi_ls},
        model=model,
    )["-7"]

    # exact refit jackknife, with the weights held at the full-data values as in the linearization
    def fit_fixed_weights(F_phi_k, F_chi_k):
        phi_lm = tools._jackknife_log_mean(F_phi_k, dof)[0]
        chi_lm = tools._jackknife_log_mean(F_chi_k, dof)[0]

        def cost(p):
            Fp, Fc, _, _ = model(m_data, *p, "-7")
            return np.sum(
                (np.log10(Fp) - np.log10(phi_lm)) ** 2 / np.log10(phi_ls) ** 2
            ) + np.sum((np.log10(Fc) - np.log10(chi_lm)) ** 2 / np.log10(chi_ls) ** 2)

        return optimize.minimize(
            cost,
            result.x,
            method="Nelder-Mead",
            options={"xatol": 1e-8, "fatol": 1e-10},
        ).x

    GM_refit = np.array(
        [
            fit_fixed_weights(np.delete(F_phi, k, 0), np.delete(F_chi, k, 0))[0]
            for k in range(M)
        ]
    )
    assert jk["params_jk"].shape == (M, 3)
    assert np.isclose(jk["GM"], result.x[0])
    np.testing.assert_allclose(jk["params_jk"][:, 0], GM_refit, rtol=0.02)
    assert np.isclose(jk["GM_std"], np.sqrt(tools._jackknife_var(GM_refit)), rtol=0.1)


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


def test_fit_vertical_coherence_converges_at_large_amplitude():
    """A0 ~ 5e12 and a short z0 (the +7 case that stopped at the old fixed initial guess) are recovered,
    whatever the initial coherence length."""
    import numpy as np

    from n23_fluctuations_ragland import tools

    dz = np.arange(0, 28 * 9.0, 9.0)
    MC = tools.gaussian_coherence_model(dz, 4.84e12, 58.0)
    for L0 in (20.0, 100.0, 400.0):
        A0, z0 = tools.fit_vertical_coherence(dz, MC, L0=L0)
        assert np.isclose(A0, 4.84e12, rtol=1e-6)
        assert np.isclose(z0, 58.0, rtol=1e-6)


def test_fit_vertical_coherence_raises_on_failure():
    """A coherence that is not positive at the first nonzero lag can't be fitted, and raises instead of
    returning a value."""
    import numpy as np
    import pytest

    from n23_fluctuations_ragland import tools

    dz = np.arange(0, 28 * 9.0, 9.0)
    MC = np.full(dz.shape, -1e10)
    MC[0] = 3e11
    with pytest.raises(ValueError, match="A0_init"):
        tools.fit_vertical_coherence(dz, MC)


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


def test_array_tilt_jackknife_se_matches_leave_one_ray_out():
    """The tilt SE is the leave-one-ray-out jackknife of the tilt, NaN picks skipped."""
    import numpy as np

    from n23_fluctuations_ragland import tools

    rng = np.random.default_rng(1)
    tilt = rng.normal(0, 1, 23)
    angles = {f"+{k}": 3.0 * k + tilt + rng.normal(0, 0.2, 23) for k in range(6)}
    angles["+0"][4] = np.nan
    angles["-15b"] = rng.normal(0, 3, 23)
    omit = ("-15b",)

    se = tools.array_tilt_jackknife_se({"angles": angles}, omit=omit)
    assert np.allclose(
        tools.estimate_array_tilt({"angles": angles}, omit=omit)["array_tilt_se"], se
    )

    rays = [r for r in angles if r not in omit]
    loo = np.array(
        [
            tools.estimate_array_tilt(
                {"angles": {r: angles[r] for r in rays if r != left_out}}, omit=()
            )["array_tilt"]
            for left_out in rays
        ]
    )
    n = len(rays)
    expected = np.sqrt((n - 1) / n * np.sum((loo - loo.mean(0)) ** 2, axis=0))
    assert np.allclose(se, expected)
    assert np.all(se > 0)


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
    assert cli_mod.tools.wavefront_values_path(False).name == "wavefront_values.pkl"
    assert (
        cli_mod.tools.wavefront_values_path(True).name
        == "wavefront_values_tilt_removed.pkl"
    )


def test_wavefront_values_paths_per_pattern(tmp_path, monkeypatch):
    """Each removed hydrophone pattern gets its own wavefront values and fit file; none keeps the old names."""
    from n23_fluctuations_ragland import tools

    monkeypatch.setattr(tools, "get_data_dir", lambda: tmp_path)
    names = {
        (tilt, pattern): tools.wavefront_values_path(tilt, pattern).name
        for tilt in (False, True)
        for pattern in tools.REMOVE_PATTERN_CHOICES
    }
    assert names[False, "none"] == "wavefront_values.pkl"
    assert names[True, "none"] == "wavefront_values_tilt_removed.pkl"
    assert names[True, "phase"] == "wavefront_values_tilt_removed_phase_removed.pkl"
    assert names[False, "magnitude"] == "wavefront_values_magnitude_removed.pkl"
    assert (
        names[True, "both"]
        == "wavefront_values_tilt_removed_phase_magnitude_removed.pkl"
    )
    assert len(set(names.values())) == len(names)
    assert (
        tools.phi_chi_LS_fit_path(True, "both").name
        == "phi_chi_LS_fit_tilt_removed_phase_magnitude_removed.pkl"
    )
    with pytest.raises(ValueError, match="remove_pattern"):
        tools.wavefront_values_path(True, "gain")


def test_get_wavefront_values_pattern_option():
    import inspect

    from n23_fluctuations_ragland import tools

    params = inspect.signature(tools.get_wavefront_values).parameters
    assert params["remove_pattern"].default == "none"


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


def test_phi_chi_LS_fit_paths_per_spectral_method(tmp_path, monkeypatch):
    """Each spectral estimation method gets its own fit file; the default (Hann, no first difference) keeps
    the old name."""
    from n23_fluctuations_ragland import tools

    monkeypatch.setattr(tools, "get_data_dir", lambda: tmp_path)
    names = {
        (window, fd): tools.phi_chi_LS_fit_path(True, "none", window, fd).name
        for window in tools.SPECTRAL_WINDOWS
        for fd in (False, True)
    }
    assert names["hann", False] == "phi_chi_LS_fit_tilt_removed.pkl"
    assert names["rectangular", False] == "phi_chi_LS_fit_tilt_removed_rectangular.pkl"
    assert names["tukey", False] == "phi_chi_LS_fit_tilt_removed_tukey.pkl"
    assert names["hann", True] == "phi_chi_LS_fit_tilt_removed_first_difference.pkl"
    assert (
        names["rectangular", True]
        == "phi_chi_LS_fit_tilt_removed_first_difference_rectangular.pkl"
    )
    assert (
        names["tukey", True] == "phi_chi_LS_fit_tilt_removed_first_difference_tukey.pkl"
    )
    assert (
        tools.phi_chi_LS_fit_path(False, "phase", "hann", True).name
        == "phi_chi_LS_fit_phase_removed_first_difference.pkl"
    )
    with pytest.raises(ValueError, match="window"):
        tools.phi_chi_LS_fit_path(True, "none", "hamming")


@pytest.mark.parametrize("remove_array_tilt", [False, True])
@pytest.mark.parametrize("remove_pattern", ["none", "phase", "both"])
@pytest.mark.parametrize(
    ("window", "first_difference"),
    [("hann", False), ("rectangular", True), ("tukey", False), ("tukey", True)],
)
def test_phi_chi_LS_fit_uses_matching_wavefront_values(
    tmp_path, monkeypatch, remove_array_tilt, remove_pattern, window, first_difference
):
    """The fit estimates the spectra in the requested mode and saves to that mode's file."""
    import pickle

    from n23_fluctuations_ragland import tools

    (tmp_path / "dataproducts").mkdir()
    monkeypatch.setattr(tools, "get_data_dir", lambda: tmp_path)

    calls = []

    def fake_estimate(
        remove_array_tilt=False,
        remove_pattern="none",
        window="hann",
        first_difference=False,
    ):
        calls.append((remove_array_tilt, remove_pattern, window, first_difference))
        return "m", None, None, "phi_lm", "phi_ls", "chi_lm", "chi_ls"

    monkeypatch.setattr(tools, "estimate_F_phi_chi", fake_estimate)
    monkeypatch.setattr(
        tools,
        "fit_F_phi_chis",
        lambda *args, **kwargs: {"args": args, "kwargs": kwargs},
    )

    tools.compute_phi_chi_LS_fit(
        remove_array_tilt=remove_array_tilt,
        remove_pattern=remove_pattern,
        window=window,
        first_difference=first_difference,
    )

    assert calls == [(remove_array_tilt, remove_pattern, window, first_difference)]
    path = tools.phi_chi_LS_fit_path(
        remove_array_tilt, remove_pattern, window, first_difference
    )
    with path.open("rb") as f:
        assert pickle.load(f) == {
            "args": ("m", "phi_lm", "chi_lm", "phi_ls", "chi_ls"),
            "kwargs": {},
        }
    written = list((tmp_path / "dataproducts").iterdir())
    assert written == [path]


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
    # over seeds the fitted z0 has median 89 m and sd 20 m; this seed gives 119 m (92nd percentile). It
    # passed rtol=0.3 before only because the old unscaled fit stopped early near its 100 m initial guess.
    assert np.isclose(vc["z0"], 90.0, rtol=0.4)
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


def test_jackknife_linear_mean_is_unbiased_for_white_noise():
    """The arithmetic mean of periodograms recovers a white spectrum in every bin, and its jackknife SE of
    log10 is the delta-method SE, std(F) / (sqrt(M) mean(F) ln 10)."""
    import numpy as np

    from n23_fluctuations_ragland import tools

    rng = np.random.default_rng(0)
    M, N, sigma, fs = 23, 28, 2.0, 1 / 9
    true_level = 2 * sigma**2 / fs  # one-sided PSD of white noise

    lms, ratios = [], []
    for _ in range(400):
        x = sigma * rng.standard_normal((M, N))
        _, F, dof = tools._one_sided_periodogram(x, fs)
        lm, ls = tools._jackknife_linear_mean(F)
        np.testing.assert_array_equal(lm, np.mean(F, axis=0))
        lms.append(lm)
        # log10 is nonlinear, so a single outlying transmission pulls the jackknife away from the delta method
        delta = np.std(F, axis=0, ddof=1) / (
            np.sqrt(M) * np.mean(F, axis=0) * np.log(10)
        )
        ratios.append(np.log10(ls) / delta)
        # dispatch
        np.testing.assert_array_equal(
            tools._average_periodograms(F, dof, "mean")[0], lm
        )

    # unbiased in linear space, including the 1-dof Nyquist bin
    np.testing.assert_allclose(np.mean(lms, axis=0) / true_level, 1, atol=0.05)
    np.testing.assert_allclose(np.median(ratios), 1, atol=0.05)


def test_average_periodograms_rejects_unknown_average():
    import numpy as np
    import pytest

    from n23_fluctuations_ragland import tools

    with pytest.raises(ValueError, match="average"):
        tools._average_periodograms(np.ones((3, 4)), np.full(4, 2.0), "median")


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


def _synthetic_receptions(delays, noise=0.0, seed=0):
    """Receptions with 32 upper and 28 lower hydrophones, a Gaussian pulse delayed by `delays` [s]."""
    import numpy as np
    import xarray as xr

    rng = np.random.default_rng(seed)
    ts = 0.07  # 214 samples in the 100-115 s window, the length of the lag grid
    time = 100.05 + ts * (np.arange(501) - 200)
    depth = np.concatenate([np.linspace(110, 560, 32), np.linspace(870, 1330, 28)])
    out = []
    for d in delays:
        pulse = np.exp(-0.5 * ((time - 107.0 - d) / 0.4) ** 2)
        x = np.tile(pulse, (60, 1)) * (1 + noise * rng.normal(size=(60, 1)))
        x = x + noise * rng.normal(size=x.shape)
        out.append(
            xr.DataArray(
                x.astype(complex),
                dims=("hydrophone", "time"),
                coords={"time": time, "actual_depth": ("hydrophone", depth)},
            )
        )
    return out


def test_arrival_time_shifts_recover_delay():
    """Sub-sample delays are recovered (shift = -delay); the argmax is quantized to the sample interval.

    Shifts are compared relative to the first transmission: the lag grid of the estimate is offset from the
    zero lag of ``fftconvolve(mode="same")`` by a constant (0.06 s here, 0.29 ms on the real data).
    """
    import numpy as np

    from n23_fluctuations_ragland import tools

    delays = np.array([0.0, 0.31, -0.52])
    rx = _synthetic_receptions(delays)
    for array in ("upper", "lower"):
        shifts = tools.arrival_time_shifts(rx, array)
        assert np.allclose(shifts - shifts[0], -delays, atol=0.005)
        coarse = tools.arrival_time_shifts(rx, array, subsample=False)
        assert np.allclose(coarse - coarse[0], -delays, atol=0.07)


def test_jackknife_arrival_time_offset_leaves_out_every_hydrophone():
    import numpy as np

    from n23_fluctuations_ragland import tools

    delays = np.array([0.0, 0.2, -0.4, 0.13])
    rx = _synthetic_receptions(delays, noise=0.05)
    calls = []
    result = tools.jackknife_arrival_time_offset(rx, progress=lambda: calls.append(1))

    assert len(calls) == 60  # 32 upper + 28 lower, each left out once
    assert np.allclose(
        result["dT_t"], (result["dT_t_upper"] + result["dT_t_lower"]) / 2
    )
    assert np.allclose(result["dT_t"] - result["dT_t"][0], -delays, atol=0.01)
    for key in ("dT_t_se", "dT_t_upper_se", "dT_t_lower_se"):
        se = result[key]
        assert se[0] < 1e-12  # the first transmission is the reference
        assert np.all(np.isfinite(se))
        assert np.all(se[1:] > 0)
        assert np.all(se < 0.01)


def test_check_output_exists_skip_existing_reuses_without_prompt(tmp_path, monkeypatch):
    """skip_existing=True reuses an existing output without asking, and still runs when it is missing."""
    from n23_fluctuations_ragland.cli import check_output_exists

    def no_prompt(*a, **kw):
        raise AssertionError("must not prompt")

    monkeypatch.setattr("n23_fluctuations_ragland.cli.questionary.confirm", no_prompt)
    output_file = tmp_path / "out.pkl"
    calls = []

    @check_output_exists(output_file)
    def my_func():
        calls.append(1)

    my_func(skip_existing=True)
    assert calls == [1]
    output_file.touch()
    my_func(skip_existing=True)
    assert calls == [1]


def _paper_data_dir(tmp_path, n_dmod=23, environment=True):
    from n23_fluctuations_ragland import figures

    (tmp_path / "dmod").mkdir(parents=True)
    (tmp_path / "dataproducts").mkdir()
    for k in range(n_dmod):
        (tmp_path / "dmod" / f"dm{k:03d}.mat").touch()
    if environment:
        env = tmp_path / "environment_data"
        env.mkdir()
        for name in (
            "FCTD_SoundSpeed_Colosi_v2.mat",
            "SION_CTD_new.mat",
            "SIOE_CTD.mat",
            "map_bathy.nc",
            figures.SSH_FILENAME,
        ):
            (env / name).touch()
    return tmp_path


@pytest.fixture
def paper_data_dir(tmp_path, monkeypatch):
    from n23_fluctuations_ragland import cli, figures

    data_dir = _paper_data_dir(tmp_path / "data")
    for mod in (cli, figures):
        monkeypatch.setattr(mod, "get_data_dir", lambda: data_dir)
    return data_dir


def test_check_paper_inputs_complete(paper_data_dir):
    from n23_fluctuations_ragland import cli

    inputs = cli.check_paper_inputs()
    assert inputs.ok, inputs.missing
    # the map font is optional
    assert any("font" in w for w in inputs.warnings)


def test_check_paper_inputs_lists_every_missing_input(tmp_path, monkeypatch):
    from n23_fluctuations_ragland import cli, figures

    data_dir = _paper_data_dir(tmp_path, n_dmod=0, environment=False)
    for mod in (cli, figures):
        monkeypatch.setattr(mod, "get_data_dir", lambda: data_dir)
    inputs = cli.check_paper_inputs()
    assert not inputs.ok
    text = "\n".join(inputs.missing)
    assert "expected 23 receptions, found 0" in text
    assert "on reasonable request" in text
    for name in ("FCTD_SoundSpeed_Colosi_v2.mat", "SION_CTD_new.mat", "SIOE_CTD.mat"):
        assert name in text
    assert len(inputs.missing) == 6


def test_check_paper_inputs_without_datadir(monkeypatch):
    from n23_fluctuations_ragland import cli

    def unset():
        raise OSError("unset")

    monkeypatch.setattr(cli, "get_data_dir", unset)
    inputs = cli.check_paper_inputs()
    assert not inputs.ok
    assert "NESCAFE23_DATADIR" in inputs.missing[0]


def test_paper_steps_order():
    """The steps run in the order their products depend on each other, ending with Table II and the figures."""
    from n23_fluctuations_ragland import cli

    names = [name for name, _ in cli.paper_steps("out")]
    assert names == [
        "arrival time offset",
        "arrival time offset error (jackknife)",
        "mean pulse",
        "peak locations",
        "array tilt",
        "wavefront values and hydrophone pattern",
        "rays",
        "Table II",
        "publication figures",
    ]


def test_wavefront_values_and_pattern_order(monkeypatch):
    """The pattern is estimated from the unpatterned values before the paper's values are made."""
    from n23_fluctuations_ragland import cli

    calls = []
    monkeypatch.setattr(
        cli,
        "_compute_wavefront_values",
        lambda **kw: calls.append(
            ("values", kw["remove_array_tilt"], kw["remove_pattern"])
        ),
    )
    monkeypatch.setattr(
        cli, "estimate_hydrophone_pattern", lambda **kw: calls.append(("pattern",))
    )
    cli.compute_wavefront_values_and_pattern(skip_existing=True)
    assert calls == [("values", True, "none"), ("pattern",), ("values", True, "both")]


def test_reproduce_paper_runs_steps_in_order(paper_data_dir, tmp_path):
    from n23_fluctuations_ragland import cli

    calls = []
    steps = [
        (name, lambda name=name, **kw: calls.append((name, kw)))
        for name in ("a", "b", "c")
    ]
    out = tmp_path / "out"
    assert cli.reproduce_paper(out, steps=steps)
    assert calls == [(n, {"force_overwrite": True}) for n in "abc"]

    calls.clear()
    assert cli.reproduce_paper(out, reuse=True, steps=steps)
    assert calls == [(n, {"skip_existing": True}) for n in "abc"]
    report = (out / "reproduction_report.txt").read_text()
    assert "PAPER_SETTINGS" in report
    assert "ok  a" in report


def test_reproduce_paper_stops_at_the_first_failure(paper_data_dir, tmp_path):
    from n23_fluctuations_ragland import cli

    calls = []

    def fail(**kw):
        raise RuntimeError("boom")

    steps = [
        ("a", lambda **kw: calls.append("a")),
        ("b", fail),
        ("c", lambda **kw: calls.append("c")),
    ]
    out = tmp_path / "out"
    assert not cli.reproduce_paper(out, steps=steps)
    assert calls == ["a"]
    report = (out / "reproduction_report.txt").read_text()
    assert "FAILED  b" in report
    assert "c (not run)" in report
    assert "boom" in report


def test_reproduce_paper_checks_inputs_first(tmp_path, monkeypatch):
    from n23_fluctuations_ragland import cli, figures

    data_dir = _paper_data_dir(tmp_path / "data", n_dmod=5)
    for mod in (cli, figures):
        monkeypatch.setattr(mod, "get_data_dir", lambda: data_dir)
    calls = []
    out = tmp_path / "out"
    assert not cli.reproduce_paper(out, steps=[("a", lambda **kw: calls.append("a"))])
    assert calls == []
    assert not out.exists()


def test_main_reproduce_paper_flag(monkeypatch):
    from n23_fluctuations_ragland import cli

    seen = {}

    def fake_reproduce(output_dir, reuse=False):
        seen.update(output_dir=output_dir, reuse=reuse)
        return False

    monkeypatch.setattr(cli, "reproduce_paper", fake_reproduce)
    assert cli.main(["--reproduce-paper", "--output", "x", "--reuse"]) == 1
    assert seen == {"output_dir": "x", "reuse": True}


def test_coherence_lengths_uses_paper_settings_and_writes_table(tmp_path, monkeypatch):
    from n23_fluctuations_ragland import cli

    loaded = []

    def fake_load(remove_array_tilt, remove_pattern="none"):
        loaded.append((remove_array_tilt, remove_pattern))
        return {ray_id: ray_id for ray_id in cli.tools.DUCTED_RAY_IDS}

    monkeypatch.setattr(cli, "_load_wavefront_values", fake_load)
    monkeypatch.setattr(cli.tools, "estimate_observed_coherence_lengths", lambda wv: wv)
    monkeypatch.setattr(cli.tools, "compute_theoretical_coherence_lengths", dict)
    monkeypatch.setattr(
        cli.tools, "coherence_lengths_latex_table", lambda o, t: "TABLE"
    )
    path = tmp_path / "t" / "table_II.tex"
    assert cli.coherence_lengths(output_path=path) == "TABLE"
    assert loaded == [(True, "both")]
    assert path.read_text() == "TABLE\n"
