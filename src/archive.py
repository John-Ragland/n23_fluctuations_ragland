"""
Functions that were written, but not used in the final analysis
"""

import os

import gsw
import numpy as np
import xarray as xr
import xrsignal as xrs
from dotenv import find_dotenv, load_dotenv
from scipy import io, signal

from n23_fluctuations_ragland.tools import aog_vsdg

load_dotenv(find_dotenv())


def create_2D_ocean_slice(
    z_smooth: float = 200,
    r_smooth: float = 3000,
):
    """
    Create 2D ocean slice using FCTD, and two full water CTD casts.

    Parameters
    ----------
    z_smooth : float
        vertical smoothing scale in meters. Default is 200 m.
    r_smooth : float
        horizontal smoothing scale in meters. Default is 3000 m.

    Returns
    -------
    full_field_full : xarray.Dataset
        xarray Dataset containing 2D ocean slice that is stitched together from FCTD and two CTD casts
    """
    coords = {
        "SION": [39 + 37.194 / 60, -63 - 39.427 / 60],
        "SIOE": [38 + 45.906 / 60, -62 - 15.626 / 60],
    }

    fn = f"{os.environ['NESCAFE23_DATADIR']}environment_data/SION_CTD.mat"
    SION_mat = io.loadmat(fn)

    fn = f"{os.environ['NESCAFE23_DATADIR']}environment_data/SIOE_CTD.mat"
    SIOE_mat = io.loadmat(fn)

    fn = f"{os.environ['NESCAFE23_DATADIR']}environment_data/FCTD_SoundSpeed_Colosi_v2.mat"
    fctd_mat = io.loadmat(fn)

    fctds = {}
    fctds_pilot = {}

    deep_value = {"tt": 2.3, "ss": 34.885}  # , 'cc':1640}

    for k, var in enumerate(["tt", "ss"]):
        fctds[var] = xr.DataArray(
            fctd_mat[var],
            dims=["range", "depth"],
            coords={
                "range": fctd_mat["r"].flatten() * 1000,
                "depth": fctd_mat["z"].flatten(),
                "latitude": ("range", fctd_mat["lat"].flatten()),
                "longitude": ("range", fctd_mat["lon"].flatten()),
            },
        ).sortby("range")

        # Add 10000 meter
        fctds_pilot[var] = xr.concat(
            (
                fctds[var],
                xr.ones_like(fctds[var][:, -2:-1]).assign_coords({"depth": [10000]})
                * deep_value[var],
            ),
            dim="depth",
        )

    fctds_pilot = xr.Dataset(fctds_pilot)

    # interpolate to Δz=1m, Δr = 500
    fctd_grid = fctds_pilot.interp({
        "range": np.arange(0, fctds_pilot.range[-1], 500),
        "depth": np.arange(0, 6000),
    })

    bz, az = signal.butter(4, 1 / z_smooth, fs=1)
    br, ar = signal.butter(4, 1 / r_smooth, fs=1 / 500)

    front = {}
    for k, var in enumerate(["tt", "ss"]):
        fctd_smooth = xrs.filtfilt(
            xrs.filtfilt(fctd_grid[var], dim="range", b=br, a=ar),
            dim="depth",
            b=bz,
            a=az,
        )

        front[var] = fctd_smooth.sel({"range": slice(60000, 170000)})
        front[var] = front[var].assign_coords({
            "range": front[var].range.values - 115000
        })
        front[var] = (
            front[var]
            .assign_coords({"range": np.flip(front[var].range.values)})
            .sortby("range")
        )

        front[var] = front[var].drop_vars(["latitude", "longitude"])

    front = xr.Dataset(front)

    full_field_range_coord = np.arange(-155000, 155000, 500)

    mask = xr.DataArray(
        np.ones((6000, len(full_field_range_coord))),
        dims=["depth", "range"],
        coords={"depth": front.depth, "range": full_field_range_coord},
    )

    front_rep = xr.concat(
        (
            front.isel({"range": 0}).assign_coords({"range": -155000}),
            front,
            front.isel({"range": -1}).assign_coords({"range": 155000}),
        ),
        dim="range",
    )

    front_full = front_rep.interp_like(
        mask, kwargs={"bounds_error": False, "fill_value": "extrapolate"}
    )

    SION = {
        "tt": xr.DataArray(
            SION_mat["t5"].flatten(),
            dims=["depth"],
            coords={"depth": SION_mat["z5"].flatten()},
        ),
        "ss": xr.DataArray(
            SION_mat["s5"].flatten(),
            dims=["depth"],
            coords={"depth": SION_mat["z5"].flatten()},
        ),
    }

    SIOE = {
        "tt": xr.DataArray(
            SIOE_mat["t8"].flatten(),
            dims=["depth"],
            coords={"depth": SIOE_mat["z8"].flatten()},
        ),
        "ss": xr.DataArray(
            SIOE_mat["s8"].flatten(),
            dims=["depth"],
            coords={"depth": SIOE_mat["z8"].flatten()},
        ),
    }

    for var in SION.keys():
        # Add 10000 meter
        SION[var] = xr.concat(
            (
                SION[var],
                xr.ones_like(SION[var][-2:-1]).assign_coords({"depth": [10000]})
                * deep_value[var],
            ),
            dim="depth",
        )

        # Add 10000 meter
        SIOE[var] = xr.concat(
            (
                SIOE[var],
                xr.ones_like(SIOE[var][-2:-1]).assign_coords({"depth": [10000]})
                * deep_value[var],
            ),
            dim="depth",
        )

        SION[var] = xrs.filtfilt(SION[var], dim="depth", b=bz, a=az)
        SIOE[var] = xrs.filtfilt(SIOE[var], dim="depth", b=bz, a=az)

    SION = xr.Dataset(SION)
    SIOE = xr.Dataset(SIOE)

    SION_full = SION.interp_like(
        mask, kwargs={"bounds_error": False, "fill_value": "extrapolate"}
    ).broadcast_like(mask)

    SIOE_full = SIOE.interp_like(
        mask, kwargs={"bounds_error": False, "fill_value": "extrapolate"}
    ).broadcast_like(mask)

    SION_full_nlm = SION_full.copy()
    SION_full_nlm = SION_full_nlm.where(
        (SION_full_nlm.depth > 350) | (SION_full_nlm.depth < 25), np.nan
    )
    SION_full_nlm = SION_full_nlm.interpolate_na(dim="depth")

    # compute C
    p = gsw.p_from_z(-SION_full_nlm.depth, lat=coords["SION"][0])

    mask1 = mask.where(
        (mask.depth < 800) & (mask.range > -50000) & (mask.range < 50000), 0
    )
    mask2 = (-mask1 + 1).where(mask.range > -30000, 0)
    mask3 = (-mask1 + 1).where(mask.range <= -30000, 0)

    z = np.linspace(-np.pi, np.pi, 600)
    x = np.linspace(-np.pi, np.pi, 120)

    kernel_x = (np.cos(x) + 1) / 2
    kernel_z = (np.cos(z) + 1) / 2

    kernel = kernel_z[:, None] * kernel_x[None, :]
    kernel = kernel / np.sum(kernel)

    mask1_smooth = xr.DataArray(
        _fftconvolve_mirror(mask1.values, kernel, mode="same"),
        dims=["depth", "range"],
        coords={"depth": mask.depth, "range": mask.range},
    )

    mask2_smooth = xr.DataArray(
        _fftconvolve_mirror(mask2.values, kernel, mode="same"),
        dims=["depth", "range"],
        coords={"depth": mask.depth, "range": mask.range},
    )

    mask3_smooth = xr.DataArray(
        _fftconvolve_mirror(mask3.values, kernel, mode="same"),
        dims=["depth", "range"],
        coords={"depth": mask.depth, "range": mask.range},
    )

    full_field = (
        front_full * mask1_smooth + SION_full * mask2_smooth + SIOE_full * mask3_smooth
    )

    full_field["tt"].attrs["units"] = "deg C"
    full_field["tt"].name = "in-situ temperature"

    full_field["ss"].attrs["units"] = "psu"
    full_field["ss"].name = "pracitical salinity"

    full_field_nlm = (
        front_full * mask1_smooth
        + SION_full_nlm * mask2_smooth
        + SIOE_full * mask3_smooth
    )

    full_field_nlm["tt"].attrs["units"] = "deg C"
    full_field_nlm["tt"].name = "in-situ temperature"

    full_field_nlm["ss"].attrs["units"] = "psu"
    full_field_nlm["ss"].name = "pracitical salinity"

    # ## Calculate all oceanographic variables
    p = gsw.p_from_z(-full_field.depth, coords["SION"][0])

    ## Salinity
    SA = gsw.SA_from_SP(full_field["ss"], p, coords["SION"][1], coords["SION"][0])
    SA.attrs["units"] = "g/kg"
    SA.name = "Absolute Salinity"

    ## Conservative Temperature
    CT = gsw.CT_from_t(SA, full_field["tt"], p)
    CT.attrs["units"] = "deg C"
    CT.name = "Conservative Temperature"

    ## Sound Spped
    c = aog_vsdg(p, full_field["tt"], full_field["ss"])
    c.attrs["units"] = "m/s"
    c.name = "Sound Speed"

    ## Buoyancy Frequency
    N2, pmid = gsw.Nsquared(SA, CT, p.values[:, None], lat=coords["SION"][0], axis=0)
    z_mid = gsw.z_from_p(pmid, coords["SION"][0])[:, 0]
    N2x = xr.DataArray(
        N2, dims=["depth", "range"], coords={"depth": -z_mid, "range": full_field.range}
    )
    N = (
        np.sqrt(
            N2x.interp(
                {"depth": full_field.depth},
                kwargs={"bounds_error": False, "fill_value": "extrapolate"},
            )
        )
        / (2 * np.pi)
        * 3600
    )
    N.attrs["units"] = "cph"
    N.name = "Buoyancy Frequency"

    ## Potential Temperature
    PT = gsw.pt_from_t(SA, full_field["tt"], p, 0)
    PT.attrs["units"] = "deg C"
    PT.name = "potential temperature"

    ## Adiabatic Sound Speed Gradient
    C_adiabatic = aog_vsdg(0, PT, full_field["ss"])
    C_adiabatic.attrs["units"] = "deg C"
    C_adiabatic.name = "adiabatic sound speed"

    dcdz_p = -C_adiabatic.differentiate("depth")
    dcdz_p.attrs["units"] = "$m/s^2$"
    dcdz_p.name = "adiabatic vertical sound speed gradient"

    full_field_full = xr.Dataset({
        "t": full_field["tt"],
        "SP": full_field["ss"],
        "c": c,
        "N": N,
        "SA": SA,
        "dcdz_p": dcdz_p,
    })

    return full_field_full


def create_2D_ocean_slice_nlm(
    z_smooth: float = 200,
    r_smooth: float = 3000,
):
    """
    Create 2D ocean slice using FCTD, and two full water CTD casts.
    Local maximum near -6 upper turning point in SION CTD is manually removed.

    Parameters
    ----------
    z_smooth : float
        vertical smoothing scale in meters. Default is 200 m.
    r_smooth : float
        horizontal smoothing scale in meters. Default is 3000 m.

    Returns
    -------
    full_field_full_nlm : xarray.Dataset
        xarray Dataset containing 2D ocean slice that is stitched together from FCTD and two CTD casts.
        Local maximum near -6 upper turning point in SION CTD is manually removed.
    """
    coords = {
        "SION": [39 + 37.194 / 60, -63 - 39.427 / 60],
        "SIOE": [38 + 45.906 / 60, -62 - 15.626 / 60],
    }

    fn = f"{os.environ['NESCAFE23_DATADIR']}environment_data/SION_CTD.mat"
    SION_mat = io.loadmat(fn)

    fn = f"{os.environ['NESCAFE23_DATADIR']}environment_data/SIOE_CTD.mat"
    SIOE_mat = io.loadmat(fn)

    fn = f"{os.environ['NESCAFE23_DATADIR']}environment_data/FCTD_SoundSpeed_Colosi_v2.mat"
    fctd_mat = io.loadmat(fn)

    fctds = {}
    fctds_pilot = {}

    deep_value = {"tt": 2.3, "ss": 34.885}  # , 'cc':1640}

    for k, var in enumerate(["tt", "ss"]):
        fctds[var] = xr.DataArray(
            fctd_mat[var],
            dims=["range", "depth"],
            coords={
                "range": fctd_mat["r"].flatten() * 1000,
                "depth": fctd_mat["z"].flatten(),
                "latitude": ("range", fctd_mat["lat"].flatten()),
                "longitude": ("range", fctd_mat["lon"].flatten()),
            },
        ).sortby("range")

        # Add 10000 meter
        fctds_pilot[var] = xr.concat(
            (
                fctds[var],
                xr.ones_like(fctds[var][:, -2:-1]).assign_coords({"depth": [10000]})
                * deep_value[var],
            ),
            dim="depth",
        )

    fctds_pilot = xr.Dataset(fctds_pilot)

    # interpolate to Δz=1m, Δr = 500
    fctd_grid = fctds_pilot.interp({
        "range": np.arange(0, fctds_pilot.range[-1], 500),
        "depth": np.arange(0, 6000),
    })

    bz, az = signal.butter(4, 1 / z_smooth, fs=1)
    br, ar = signal.butter(4, 1 / r_smooth, fs=1 / 500)

    front = {}
    for k, var in enumerate(["tt", "ss"]):
        fctd_smooth = xrs.filtfilt(
            xrs.filtfilt(fctd_grid[var], dim="range", b=br, a=ar),
            dim="depth",
            b=bz,
            a=az,
        )

        front[var] = fctd_smooth.sel({"range": slice(60000, 170000)})
        front[var] = front[var].assign_coords({
            "range": front[var].range.values - 115000
        })
        front[var] = (
            front[var]
            .assign_coords({"range": np.flip(front[var].range.values)})
            .sortby("range")
        )

        front[var] = front[var].drop_vars(["latitude", "longitude"])

    front = xr.Dataset(front)

    full_field_range_coord = np.arange(-155000, 155000, 500)

    mask = xr.DataArray(
        np.ones((6000, len(full_field_range_coord))),
        dims=["depth", "range"],
        coords={"depth": front.depth, "range": full_field_range_coord},
    )

    front_rep = xr.concat(
        (
            front.isel({"range": 0}).assign_coords({"range": -155000}),
            front,
            front.isel({"range": -1}).assign_coords({"range": 155000}),
        ),
        dim="range",
    )

    front_full = front_rep.interp_like(
        mask, kwargs={"bounds_error": False, "fill_value": "extrapolate"}
    )

    SION = {
        "tt": xr.DataArray(
            SION_mat["t5"].flatten(),
            dims=["depth"],
            coords={"depth": SION_mat["z5"].flatten()},
        ),
        "ss": xr.DataArray(
            SION_mat["s5"].flatten(),
            dims=["depth"],
            coords={"depth": SION_mat["z5"].flatten()},
        ),
    }

    SIOE = {
        "tt": xr.DataArray(
            SIOE_mat["t8"].flatten(),
            dims=["depth"],
            coords={"depth": SIOE_mat["z8"].flatten()},
        ),
        "ss": xr.DataArray(
            SIOE_mat["s8"].flatten(),
            dims=["depth"],
            coords={"depth": SIOE_mat["z8"].flatten()},
        ),
    }

    for var in SION.keys():
        # Add 10000 meter
        SION[var] = xr.concat(
            (
                SION[var],
                xr.ones_like(SION[var][-2:-1]).assign_coords({"depth": [10000]})
                * deep_value[var],
            ),
            dim="depth",
        )

        # Add 10000 meter
        SIOE[var] = xr.concat(
            (
                SIOE[var],
                xr.ones_like(SIOE[var][-2:-1]).assign_coords({"depth": [10000]})
                * deep_value[var],
            ),
            dim="depth",
        )

        SION[var] = xrs.filtfilt(SION[var], dim="depth", b=bz, a=az)
        SIOE[var] = xrs.filtfilt(SIOE[var], dim="depth", b=bz, a=az)

    SION = xr.Dataset(SION)
    SIOE = xr.Dataset(SIOE)

    SION_full = SION.interp_like(
        mask, kwargs={"bounds_error": False, "fill_value": "extrapolate"}
    ).broadcast_like(mask)

    SIOE_full = SIOE.interp_like(
        mask, kwargs={"bounds_error": False, "fill_value": "extrapolate"}
    ).broadcast_like(mask)

    SION_full_nlm = SION_full.copy()
    SION_full_nlm = SION_full_nlm.where(
        (SION_full_nlm.depth > 350) | (SION_full_nlm.depth < 25), np.nan
    )
    SION_full_nlm = SION_full_nlm.interpolate_na(dim="depth")

    # compute C
    p = gsw.p_from_z(-SION_full_nlm.depth, lat=coords["SION"][0])

    mask1 = mask.where(
        (mask.depth < 800) & (mask.range > -50000) & (mask.range < 50000), 0
    )
    mask2 = (-mask1 + 1).where(mask.range > -30000, 0)
    mask3 = (-mask1 + 1).where(mask.range <= -30000, 0)

    z = np.linspace(-np.pi, np.pi, 600)
    x = np.linspace(-np.pi, np.pi, 120)

    kernel_x = (np.cos(x) + 1) / 2
    kernel_z = (np.cos(z) + 1) / 2

    kernel = kernel_z[:, None] * kernel_x[None, :]
    kernel = kernel / np.sum(kernel)

    mask1_smooth = xr.DataArray(
        _fftconvolve_mirror(mask1.values, kernel, mode="same"),
        dims=["depth", "range"],
        coords={"depth": mask.depth, "range": mask.range},
    )

    mask2_smooth = xr.DataArray(
        _fftconvolve_mirror(mask2.values, kernel, mode="same"),
        dims=["depth", "range"],
        coords={"depth": mask.depth, "range": mask.range},
    )

    mask3_smooth = xr.DataArray(
        _fftconvolve_mirror(mask3.values, kernel, mode="same"),
        dims=["depth", "range"],
        coords={"depth": mask.depth, "range": mask.range},
    )

    full_field_nlm = (
        front_full * mask1_smooth
        + SION_full_nlm * mask2_smooth
        + SIOE_full * mask3_smooth
    )

    full_field_nlm["tt"].attrs["units"] = "deg C"
    full_field_nlm["tt"].name = "in-situ temperature"

    full_field_nlm["ss"].attrs["units"] = "psu"
    full_field_nlm["ss"].name = "pracitical salinity"

    # ## Calculate all oceanographic variables
    p = gsw.p_from_z(-full_field_nlm.depth, coords["SION"][0])

    ## Salinity
    SA_nlm = gsw.SA_from_SP(
        full_field_nlm["ss"], p, coords["SION"][1], coords["SION"][0]
    )
    SA_nlm.attrs["units"] = "g/kg"
    SA_nlm.name = "Absolute Salinity"

    ## Conservative Temperature
    CT_nlm = gsw.CT_from_t(SA_nlm, full_field_nlm["tt"], p)
    CT_nlm.attrs["units"] = "deg C"
    CT_nlm.name = "Conservative Temperature"

    ## Sound Spped
    c_nlm = aog_vsdg(p, full_field_nlm["tt"], full_field_nlm["ss"])
    c_nlm.attrs["units"] = "m/s"
    c_nlm.name = "Sound Speed"

    ## Buoyancy Frequency
    N2_nlm, pmid = gsw.Nsquared(
        SA_nlm, CT_nlm, p.values[:, None], lat=coords["SION"][0], axis=0
    )
    z_mid = gsw.z_from_p(pmid, coords["SION"][0])[:, 0]
    N2x_nlm = xr.DataArray(
        N2_nlm,
        dims=["depth", "range"],
        coords={"depth": -z_mid, "range": full_field_nlm.range},
    )
    N_nlm = (
        np.sqrt(
            N2x_nlm.interp(
                {"depth": full_field_nlm.depth},
                kwargs={"bounds_error": False, "fill_value": "extrapolate"},
            )
        )
        / (2 * np.pi)
        * 3600
    )
    N_nlm.attrs["units"] = "cph"
    N_nlm.name = "Buoyancy Frequency"

    ## Potential Temperature
    PT_nlm = gsw.pt_from_t(SA_nlm, full_field_nlm["tt"], p, 0)
    PT_nlm.attrs["units"] = "deg C"
    PT_nlm.name = "potential temperature"

    ## Adiabatic Sound Speed Gradient
    C_adiabatic_nlm = aog_vsdg(0, PT_nlm, full_field_nlm["ss"])
    C_adiabatic_nlm.attrs["units"] = "deg C"
    C_adiabatic_nlm.name = "adiabatic sound speed"

    dcdz_p_nlm = -C_adiabatic_nlm.differentiate("depth")
    dcdz_p_nlm.attrs["units"] = "$m/s^2$"
    dcdz_p_nlm.name = "adiabatic vertical sound speed gradient"

    full_field_full_nlm = xr.Dataset({
        "t": full_field_nlm["tt"],
        "SP": full_field_nlm["ss"],
        "c": c_nlm,
        "N": N_nlm,
        "SA": SA_nlm,
        "dcdz_p": dcdz_p_nlm,
    })

    return full_field_full_nlm


def _fftconvolve_mirror(input_array, kernel, mode="same"):
    """
    FFT convolution with mirror boundary conditions.

    Parameters:
    -----------
    input_array : array_like
        Input array to convolve
    kernel : array_like
        Convolution kernel
    mode : str
        'same', 'valid', or 'full'
    """
    input_shape = np.array(input_array.shape)
    kernel_shape = np.array(kernel.shape)

    # Calculate padding needed (kernel_size - 1 on each side)
    pad_width = [(k - 1, k - 1) for k in kernel_shape]

    # Pad with mirror/reflection
    padded = np.pad(input_array, pad_width, mode="reflect")

    # Perform FFT convolution with mode='valid'
    result = signal.fftconvolve(padded, kernel, mode="valid")

    if mode == "same":
        # Center the result by cropping from the correct starting position
        # The valid convolution on padded input produces an output that needs to be
        # cropped by (k-1)//2 from the start to align with scipy's 'same' mode
        start = [(k - 1) // 2 for k in kernel_shape]
        slices = tuple(
            slice(s, s + inp_s) for s, inp_s in zip(start, input_shape, strict=False)
        )
        return result[slices]
    elif mode == "full":
        return result
    elif mode == "valid":
        # Calculate valid region size
        valid_shape = input_shape - kernel_shape + 1
        # Center the valid region using the same offset as 'same' mode
        start = [(k - 1) // 2 for k in kernel_shape]
        slices = tuple(
            slice(s, s + v) for s, v in zip(start, valid_shape, strict=False)
        )
        return result[slices]

    return result
