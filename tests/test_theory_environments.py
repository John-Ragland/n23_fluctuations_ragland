"""Lightweight tests for the theory environments and eigenray pairing (no data, no ray tracing)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from n23_fluctuations_ragland import theory_environments as te
from n23_fluctuations_ragland import tools


def test_munk_environment_is_canonical_and_consistent():
    B, z_axis, gamma_a, mu, g = 1300.0, 1300.0, 1.134e-5, 24.5, 9.81
    env = te.munk_environment(B=B, z_axis=z_axis, gamma_a=gamma_a, mu=mu, g=g)

    assert env["c"].dims == ("depth", "range")
    # range independent
    np.testing.assert_array_equal(env["c"].isel(range=0), env["c"].isel(range=-1))

    # sound channel axis and epsilon of pygenray's munk_ssp
    c = env["c"].isel(range=0)
    assert float(c.idxmin("depth")) == z_axis
    assert gamma_a * B / 2 == pytest.approx(0.00737, abs=1e-5)

    # N = N0 exp(-z/B)
    N = env["N"].isel(range=0)
    slope = np.polyfit(N.depth, np.log(N), 1)[0]
    assert slope == pytest.approx(-1 / B, rel=1e-10)
    assert float(N.isel(depth=0)) == pytest.approx(3.3, abs=0.1)  # cph

    # (1/c) dc_p/dz = (mu/g) N^2, and dc_p/dz + gamma_a c is the total gradient (z up)
    N_rad = env["N"] * 2 * np.pi / 3600
    np.testing.assert_allclose(env["dcdz_p"] / env["c"], mu / g * N_rad**2, rtol=1e-12)
    # total gradient (z up) = potential + adiabatic, with Munk's c1 in place of c (equal to O(epsilon))
    c1 = 1500.0
    dcdz_total = -c.differentiate("depth")
    expected = (env["dcdz_p"] * c1 / env["c"]).isel(range=0) - gamma_a * c1
    np.testing.assert_allclose(
        dcdz_total.isel(depth=slice(1, -1)),
        expected.isel(depth=slice(1, -1)),
        atol=1e-6,
    )


def _eigs(ray_ids, launch, received):
    return SimpleNamespace(
        ray_id={0: np.array(ray_ids)},
        launch_angles={0: np.array(launch)},
        received_angles={0: np.array(received)},
    )


def test_pair_eigenrays_by_reciprocity():
    # three '-6' micro-rays; the backward ones are listed in reverse order (as in the FCTD run)
    fwd = _eigs(
        ["-6.0"] * 3 + ["7.0"], [-5.5, -5.06, -4.93, 3.66], [-9.01, -8.63, -8.57, -7.09]
    )
    bck = _eigs(
        ["6.0"] * 3 + ["7.0"], [8.34, 8.40, 8.78, 6.90], [4.90, 5.02, 5.46, -3.63]
    )

    assert tools.pair_eigenrays("-6", fwd, bck) == [(0, 2), (1, 1), (2, 0)]
    assert tools.pair_eigenrays("+7", fwd, bck) == [(0, 0)]
    assert tools.pair_eigenrays("+8", fwd, bck) == []


def test_pair_eigenrays_drops_non_reciprocal():
    fwd = _eigs(["7.0", "7.0"], [3.66, 9.0], [-7.09, -12.0])
    bck = _eigs(["7.0"], [6.90], [-3.63])

    with pytest.warns(UserWarning, match="outnumber"):
        assert tools.pair_eigenrays("+7", fwd, bck) == [(0, 0)]

    far = _eigs(["7.0"], [20.0], [-15.0])
    with pytest.warns(UserWarning, match="no reciprocal"):
        assert tools.pair_eigenrays("+7", bck, far) == []


def test_wft_structure_function_without_fresnel_filter():
    """With R_f = 0, D_chi = 0, D_phi(0) = 0 and D_phi -> 2 x the variance at large separation."""
    n_s = 11

    def along_s(value):
        return xr.DataArray(np.full(n_s, value), dims=["s"])

    N, N0B = 5e-3, 2.5
    m1 = np.pi * N / N0B
    rv = {
        "Nray": along_s(N),
        "N0B_ray": along_s(N0B),
        "Theta_ray": along_s(0.1),
        "mu2": along_s(1e-8),
        "Rf2": along_s(0.0),
        "drdt": np.full(n_s, 100.0),
        "dzdt": np.zeros(n_s),
        "J": along_s(1e6),  # band reaches past the top of the grid
        "m_lo_max": 0.999 * m1 / (2 * np.pi),
        "m_hi_max": 1e6,
    }
    m = np.logspace(-5, 1, 20000)
    dz = np.array([0.0, 1e5])

    D_phi, D_chi = tools.compute_wft_structure_function(rv, dz, m)
    F_phi, _ = tools.compute_F_phi_chi(rv, 2 * np.pi * m[None, :])
    variance = np.trapezoid(np.nan_to_num(F_phi), m)

    np.testing.assert_array_equal(D_chi, 0)
    assert D_phi[0] == 0
    assert D_phi[1] == pytest.approx(2 * variance, rel=1e-2)
