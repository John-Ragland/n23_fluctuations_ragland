"""Publication figures for the NESCAFE23 acoustic fluctuations paper.

Each ``plot_*`` function builds one figure and returns it. The bodies are the
corresponding cells of ``notebooks/publication_figures/*.ipynb`` (and, for the map,
``notebooks/presentation_figures/Map_and_Environment.ipynb``), kept as close to the
notebook code as possible. The only systematic edits are:

- data paths use ``get_data_dir()`` instead of ``os.environ['NESCAFE23_DATADIR']``;
- ``savefig`` is left to :func:`generate_publication_figures`;
- ``text.usetex`` follows the config instead of being hard-coded to ``True``;
- the random jitter in the intensity-distribution scatter is seeded.

The figures only read cached data products, so the processing pipeline
(``n23_fluctuation_analysis`` → "run all processing") must have been run first.

Usage
-----
Write a config with :func:`write_config_template`, edit it, then call
:func:`generate_publication_figures` (or use the CLI options "create publication
figures config" and "generate publication figures").
"""

from __future__ import annotations

import colorsys
import pickle
import traceback
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from importlib import resources
from itertools import cycle
from pathlib import Path
from typing import Any

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean
import cmps
import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import nescafe23 as n23
import numpy as np
import pandas as pd
import pyproj
import scipy
import seaborn as sns
import xarray as xr
import yaml
from matplotlib import gridspec, patheffects
from matplotlib import lines as mlines
from matplotlib.figure import Figure
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from nescafe23.data import get_data_dir

from n23_fluctuations_ragland import tools

whoi_colors = ["#00B7BD", "#EE5340", "#0069b1", "#041e42"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FIGURE_FORMATS = ("pdf", "png", "svg")
TEMPLATE_FILE = "publication_figures.yaml"


class ConfigError(ValueError):
    """The publication figures config is invalid."""


@dataclass(frozen=True)
class MapConfig:
    """Inputs for the ``Map`` figure. ``None`` means the default under ``NESCAFE23_DATADIR``."""

    font: str | None = None
    bathymetry: str | None = None
    ssh: str | None = None
    ssh_dates: tuple[str, str] = ("2023-04-30", "2023-05-30")
    front_shift: bool = False
    front_shift_offset_km: float = -15.0

    @classmethod
    def from_mapping(cls, data: Any, path: str = "map") -> MapConfig:
        data = _require_mapping(data, path)
        _check_keys(
            data,
            (
                "font",
                "bathymetry",
                "ssh",
                "ssh_dates",
                "front_shift",
                "front_shift_offset_km",
            ),
            path,
        )
        front_shift = data.get("front_shift", cls.front_shift)
        if not isinstance(front_shift, bool):
            msg = f"{path}.front_shift: expected true or false, got {front_shift!r}"
            raise ConfigError(msg)
        offset = data.get("front_shift_offset_km", cls.front_shift_offset_km)
        if isinstance(offset, bool) or not isinstance(offset, int | float):
            msg = f"{path}.front_shift_offset_km: expected a number, got {offset!r}"
            raise ConfigError(msg)
        ssh_dates = data.get("ssh_dates", cls.ssh_dates)
        if (
            not isinstance(ssh_dates, Sequence)
            or isinstance(ssh_dates, str)
            or len(ssh_dates) != 2
        ):
            msg = f"{path}.ssh_dates: expected a list of two dates, got {ssh_dates!r}"
            raise ConfigError(msg)
        return cls(
            font=_get_optional_str(data, "font", path),
            bathymetry=_get_optional_str(data, "bathymetry", path),
            ssh=_get_optional_str(data, "ssh", path),
            ssh_dates=(str(ssh_dates[0]), str(ssh_dates[1])),
            front_shift=front_shift,
            front_shift_offset_km=float(offset),
        )


@dataclass(frozen=True)
class PublicationFiguresConfig:
    """Contents of a publication figures ``config.yaml``."""

    output_dir: Path = Path("figures")
    which: tuple[str, ...] = ()
    formats: tuple[str, ...] | None = None
    dpi: int | None = None
    usetex: bool = True
    seed: int = 0
    map: MapConfig = field(default_factory=MapConfig)

    def __post_init__(self) -> None:
        if not self.which:
            object.__setattr__(self, "which", tuple(FIGURES))

    @classmethod
    def from_mapping(
        cls, data: Any, base_dir: Path | None = None, path: str = "config"
    ) -> PublicationFiguresConfig:
        """Validate a parsed YAML mapping. A relative ``output_dir`` is resolved against ``base_dir``."""
        data = _require_mapping(data, path)
        _check_keys(
            data,
            ("output_dir", "which", "formats", "dpi", "usetex", "seed", "map"),
            path,
        )

        output_dir = Path(str(data.get("output_dir", "figures"))).expanduser()
        if base_dir is not None and not output_dir.is_absolute():
            output_dir = base_dir / output_dir

        which = _get_str_list(data, "which", path) or tuple(FIGURES)
        unknown = [name for name in which if name not in FIGURES]
        if unknown:
            msg = f"{path}.which: unknown figure(s) {unknown}; choose from {list(FIGURES)}"
            raise ConfigError(msg)

        formats = _get_str_list(data, "formats", path)
        if formats is not None:
            if not formats:
                msg = f"{path}.formats: must not be empty (use null for each figure's default)"
                raise ConfigError(msg)
            bad = [fmt for fmt in formats if fmt not in FIGURE_FORMATS]
            if bad:
                msg = f"{path}.formats: unknown format(s) {bad}; choose from {list(FIGURE_FORMATS)}"
                raise ConfigError(msg)

        dpi = data.get("dpi")
        if dpi is not None and (isinstance(dpi, bool) or not isinstance(dpi, int)):
            msg = f"{path}.dpi: expected an integer or null, got {dpi!r}"
            raise ConfigError(msg)

        usetex = data.get("usetex", True)
        if not isinstance(usetex, bool):
            msg = f"{path}.usetex: expected true or false, got {usetex!r}"
            raise ConfigError(msg)

        seed = data.get("seed", 0)
        if isinstance(seed, bool) or not isinstance(seed, int):
            msg = f"{path}.seed: expected an integer, got {seed!r}"
            raise ConfigError(msg)

        return cls(
            output_dir=output_dir,
            which=which,
            formats=formats,
            dpi=dpi,
            usetex=usetex,
            seed=seed,
            map=MapConfig.from_mapping(data.get("map") or {}, f"{path}.map"),
        )


def _require_mapping(data: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        msg = f"{path}: expected a mapping, got {type(data).__name__}"
        raise ConfigError(msg)
    return data


def _check_keys(data: Mapping[str, Any], allowed: Sequence[str], path: str) -> None:
    unknown = sorted(set(data) - set(allowed))
    if unknown:
        msg = f"{path}: unknown key(s) {unknown}; allowed keys are {list(allowed)}"
        raise ConfigError(msg)


def _get_optional_str(data: Mapping[str, Any], key: str, path: str) -> str | None:
    value = data.get(key)
    if value is not None and not isinstance(value, str):
        msg = f"{path}.{key}: expected a string or null, got {value!r}"
        raise ConfigError(msg)
    return value


def _get_str_list(
    data: Mapping[str, Any], key: str, path: str
) -> tuple[str, ...] | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, str) or not isinstance(value, Sequence):
        msg = f"{path}.{key}: expected a list, got {value!r}"
        raise ConfigError(msg)
    return tuple(str(v) for v in value)


def template_path() -> Path:
    """Path of the config template shipped with the package."""
    return Path(
        str(resources.files("n23_fluctuations_ragland") / "templates" / TEMPLATE_FILE)
    )


def write_config_template(dest: str | Path, *, overwrite: bool = False) -> Path:
    """Copy the config template to ``dest`` (a file, or a directory to put ``config.yaml`` in)."""
    dest = Path(dest).expanduser()
    if dest.is_dir():
        dest = dest / "config.yaml"
    if dest.exists() and not overwrite:
        msg = f"{dest} already exists"
        raise FileExistsError(msg)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(template_path().read_text())
    return dest


def load_config(path: str | Path) -> PublicationFiguresConfig:
    """Read and validate a publication figures config file."""
    path = Path(path).expanduser()
    if not path.exists():
        msg = (
            f"{path} does not exist. Create one with the CLI option "
            "'create publication figures config'."
        )
        raise FileNotFoundError(msg)
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        msg = f"{path}: could not parse YAML: {e}"
        raise ConfigError(msg) from e
    return PublicationFiguresConfig.from_mapping(
        data or {}, base_dir=path.parent, path=str(path)
    )


# ---------------------------------------------------------------------------
# Shared data loading
# ---------------------------------------------------------------------------


def _dataproduct(name: str, cli_step: str) -> Path:
    """Path of a cached data product, or an error naming the CLI step that makes it."""
    fn = get_data_dir() / "dataproducts" / name
    if not fn.exists():
        msg = (
            f"{fn} not found. Run '{cli_step}' in `n23_fluctuation_analysis` "
            "(or 'run all processing') first."
        )
        raise FileNotFoundError(msg)
    return fn


def _load_pickle(name: str, cli_step: str) -> Any:
    with _dataproduct(name, cli_step).open("rb") as f:
        return pickle.load(f)


@cache
def _receptions() -> list[xr.DataArray]:
    return tools.open_receptions()  # type: ignore[no-any-return]


@cache
def _ray_run() -> dict[str, Any]:
    return _load_pickle("ray_run_forward_back.pkl", "compute rays")  # type: ignore[no-any-return]


@cache
def _wavefront_values() -> dict[str, Any]:
    return _load_pickle("wavefront_values.pkl", "compute wavefront values")  # type: ignore[no-any-return]


def clear_cache() -> None:
    """Forget data loaded by previous figures (e.g. after rerunning a processing step)."""
    _receptions.cache_clear()
    _ray_run.cache_clear()
    _wavefront_values.cache_clear()


# ---------------------------------------------------------------------------
# Helpers copied from the notebooks
# ---------------------------------------------------------------------------


def lighten_color(hex_color, saturation_scale=0.5, lightness_scale=1.5):
    """
    Lighten a hex color similar to seaborn's tab20 palette

    Args:
        hex_color: Hex color string (e.g., '#1f77b4')
        saturation_scale: Multiply saturation by this (default 0.5)
        lightness_scale: Multiply lightness by this (default 1.5)
    """
    # Remove '#' if present
    hex_color = hex_color.lstrip("#")

    # Convert hex to RGB (0-1 range)
    r, g, b = tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))

    # Convert RGB to HLS
    h, lightness, s = colorsys.rgb_to_hls(r, g, b)

    # Adjust saturation and lightness
    s_new = min(1.0, s * saturation_scale)
    l_new = min(1.0, lightness * lightness_scale)

    # Convert back to RGB
    r_new, g_new, b_new = colorsys.hls_to_rgb(h, l_new, s_new)

    # Convert to hex
    return f"#{int(r_new * 255):02x}{int(g_new * 255):02x}{int(b_new * 255):02x}"


class HandlerTupleVertical(HandlerBase):
    """Stack the handles in a tuple vertically (offset in y) instead of
    HandlerTuple's default horizontal (x) offset."""

    def __init__(self, ndivide=1, pad=None, **kwargs):
        self._ndivide = ndivide
        self._pad = pad
        super().__init__(**kwargs)

    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        handler_map = legend.get_legend_handler_map()
        ndivide = len(orig_handle) if self._ndivide is None else self._ndivide
        pad = (
            (legend.borderpad * fontsize)
            if self._pad is None
            else (self._pad * fontsize)
        )

        if ndivide > 1:
            height = (height - pad * (ndivide - 1)) / ndivide

        yds_cycle = cycle(ydescent - (height + pad) * np.arange(ndivide))

        a_list = []
        for handle1 in orig_handle:
            handler = legend.get_legend_handler(handler_map, handle1)
            a_list.extend(
                handler.create_artists(
                    legend,
                    handle1,
                    xdescent,
                    next(yds_cycle),
                    width,
                    height,
                    fontsize,
                    trans,
                )
            )
        return a_list


def modulated_exponential(x, b):
    p_exp = np.exp(-x)
    return p_exp * (1 + b / 2 * (x**2 - 4 * x + 2))


def modulated_exp_gamma(x, alpha):
    return (
        2
        * alpha ** ((alpha + 1) / 2)
        / scipy.special.gamma(alpha)
        * x ** ((alpha - 1) / 2)
        * scipy.special.kv(alpha - 1, 2 * np.sqrt(alpha * x))
    )


def geodesic_level_crossings(
    field: xr.DataArray,
    start: tuple[float, float],
    end: tuple[float, float],
    level: float = 0.25,
    npts: int = 2001,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Where ``field`` crosses ``level`` along the WGS84 geodesic from ``start`` to ``end``.

    ``field`` has ``latitude``/``longitude`` coordinates and is interpolated bilinearly
    onto ``npts`` points of the geodesic. ``start``/``end`` are ``(lat, lon)``.

    Returns the along-geodesic distance of each crossing from ``start`` [m], the
    crossings' longitudes and latitudes, and the geodesic's length [m].
    """
    geod = pyproj.Geod(ellps="WGS84")
    _, _, length = geod.inv(start[1], start[0], end[1], end[0])
    pts = np.array(
        geod.npts(
            start[1], start[0], end[1], end[0], npts, initial_idx=0, terminus_idx=0
        )
    )
    s = np.linspace(0, length, len(pts))
    a = (
        field.interp(
            longitude=xr.DataArray(pts[:, 0], dims="s"),
            latitude=xr.DataArray(pts[:, 1], dims="s"),
        ).to_numpy()
        - level
    )

    idx = np.where(np.sign(a[:-1]) * np.sign(a[1:]) < 0)[0]
    frac = a[idx] / (a[idx] - a[idx + 1])
    dist = s[idx] + frac * (s[idx + 1] - s[idx])
    lons = pts[idx, 0] + frac * (pts[idx + 1, 0] - pts[idx, 0])
    lats = pts[idx, 1] + frac * (pts[idx + 1, 1] - pts[idx, 1])
    return dist, lons, lats, float(length)


def _annotate_front_shift(
    ax, fields, start, end, offset_km, *, color, bbox, text_color
) -> float | None:
    """Mark where each SSH field crosses 0.25 m on the ``start``→``end`` geodesic, and
    dimension the shift between the first two crossings.

    The dimension line is drawn parallel to the geodesic, ``offset_km`` to its right
    (looking from ``start`` to ``end``; negative puts it on the left). Returns the
    shift [m], or ``None`` if a field does not cross the geodesic exactly once.
    """
    crossings = []
    for ssh_field in fields:
        dist, lons, lats, _ = geodesic_level_crossings(ssh_field, start, end)
        if len(dist) != 1:
            warnings.warn(
                f"0.25 m SSH contour crosses the mooring geodesic {len(dist)} times "
                "(expected once); not annotating the front shift.",
                stacklevel=2,
            )
            return None
        crossings.append((float(dist[0]), float(lons[0]), float(lats[0])))

    geod = pyproj.Geod(ellps="WGS84")
    geo = ccrs.PlateCarree()._as_mpl_transform(ax)
    ends = []
    for _, lon, lat in crossings:
        # local direction of the geodesic, then step perpendicular to it
        az, _, _ = geod.inv(lon, lat, end[1], end[0])
        olon, olat, _ = geod.fwd(lon, lat, az + 90, offset_km * 1e3)
        ends.append((olon, olat))
        ax.plot(
            [lon, olon],
            [lat, olat],
            color=text_color,
            lw=1,
            linestyle=":",
            transform=ccrs.PlateCarree(),
            zorder=1050,
        )
        ax.scatter(
            lon,
            lat,
            transform=ccrs.PlateCarree(),
            c="w",
            edgecolor=color,
            lw=1.5,
            s=30,
            zorder=1100,
        )

    ax.annotate(
        "",
        xy=ends[1],
        xycoords=geo,
        xytext=ends[0],
        textcoords=geo,
        arrowprops={
            "arrowstyle": "<|-|>",
            "color": text_color,
            "lw": 1.5,
            "mutation_scale": 10,
            "shrinkA": 0,
            "shrinkB": 0,
        },
        zorder=1100,
    )

    shift = abs(crossings[1][0] - crossings[0][0])
    mid_lon, mid_lat = (ends[0][0] + ends[1][0]) / 2, (ends[0][1] + ends[1][1]) / 2
    az, _, _ = geod.inv(mid_lon, mid_lat, end[1], end[0])
    sign = 1 if offset_km >= 0 else -1
    label_lon, label_lat, _ = geod.fwd(mid_lon, mid_lat, az + 90 * sign, 5e3)

    # set the label parallel to the dimension line, reading left to right
    (x0, y0), (x1, y1) = geo.transform(ends)
    angle = np.degrees(np.arctan2(y1 - y0, x1 - x0))
    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180

    ax.annotate(
        f"{shift / 1e3:.0f} km",
        xy=(label_lon, label_lat),
        xycoords=geo,
        ha="center",
        va="center",
        rotation=angle,
        rotation_mode="anchor",
        fontsize=10,
        color=text_color,
        path_effects=[
            patheffects.withStroke(linewidth=3, foreground=bbox["facecolor"])
        ],
        zorder=2000,
    )
    return shift


def _format_date(date: str) -> str:
    ts = pd.Timestamp(date)
    return f"{ts:%B} {ts.day}, {ts.year}"


PANEL_LABEL_FONTSIZE = 7
PANEL_LABEL_FACE_ALPHA = 0.8

# corner -> (xy in axes fraction, inset direction, ha, va)
_PANEL_LABEL_CORNERS = {
    "upper right": ((1, 1), (-1, -1), "right", "top"),
    "upper left": ((0, 1), (1, -1), "left", "top"),
    "lower left": ((0, 0), (1, 1), "left", "bottom"),
    "lower right": ((1, 0), (-1, 1), "right", "bottom"),
}

PanelLabelLoc = str | tuple[float, float, str, str]


def label_panels(
    axes: Sequence[Any],
    locs: PanelLabelLoc | Sequence[PanelLabelLoc] = "upper right",
    *,
    inset: float = 3.0,
) -> None:
    """Letter each axis ``(a)``, ``(b)``, … in order; ``locs`` is one location or one per axis.

    A location is a corner name from ``_PANEL_LABEL_CORNERS`` (inset by ``inset`` points) or an
    ``(x, y, ha, va)`` tuple in axes fraction, used where the corner overlaps plotted elements.
    Pass the axes explicitly rather than ``fig.axes``, which also holds the colorbars.
    """
    if isinstance(locs, str) or (
        isinstance(locs, tuple) and len(locs) == 4 and isinstance(locs[2], str)
    ):
        locs = [locs] * len(axes)  # type: ignore[list-item]
    if len(locs) != len(axes):
        msg = f"got {len(locs)} label locations for {len(axes)} axes"
        raise ValueError(msg)

    for k, (ax, loc) in enumerate(zip(axes, locs, strict=True)):
        if isinstance(loc, str):
            if loc not in _PANEL_LABEL_CORNERS:
                msg = f"unknown panel label location {loc!r}; choose from {list(_PANEL_LABEL_CORNERS)}"
                raise ValueError(msg)
            xy, (dx, dy), ha, va = _PANEL_LABEL_CORNERS[loc]
            offset = (inset * dx, inset * dy)
        else:
            x, y, ha, va = loc
            xy, offset = (x, y), (0, 0)
        ax.annotate(
            f"({chr(ord('a') + k)})",
            xy=xy,
            xycoords="axes fraction",
            xytext=offset,
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=PANEL_LABEL_FONTSIZE,
            zorder=1000,
            bbox={
                "boxstyle": "square,pad=0.15",
                "facecolor": (1, 1, 1, PANEL_LABEL_FACE_ALPHA),
                "edgecolor": "k",
                "linewidth": 0.5,
            },
        )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_map(cfg: PublicationFiguresConfig) -> Figure:
    """Study-area map: bathymetry, SSH front, hydrographic survey, moorings, globe inset.

    Built from ``presentation_figures/Map_and_Environment.ipynb`` (background plate and
    ``map_inset`` cells), with the callouts of ``reference/Map.png`` added in code.
    """
    data_dir = get_data_dir()

    # font (poster presentation settings)
    font_path = (
        Path(cfg.map.font).expanduser()
        if cfg.map.font
        else data_dir / "fonts" / "RedHatDisplay-Medium.ttf"
    )
    if font_path.exists():
        fm.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = fm.FontProperties(fname=font_path).get_name()
    else:
        warnings.warn(
            f"Map font {font_path} not found; using the default font.", stacklevel=2
        )

    colors = [
        whoi_colors[0],
        lighten_color(whoi_colors[0], lightness_scale=2),
        whoi_colors[1],
        lighten_color(whoi_colors[1], lightness_scale=1.4, saturation_scale=1),
    ]

    # load environment data
    fn = (
        Path(cfg.map.bathymetry).expanduser()
        if cfg.map.bathymetry
        else data_dir / "environment_data" / "map_bathy.nc"
    )
    bathy = xr.open_dataset(fn)["elevation"]

    fn = data_dir / "environment_data" / "FCTD_SoundSpeed_Colosi_v2.mat"
    fctd = tools.get_fctd(fn=fn, r_smooth=3000)

    fn = (
        Path(cfg.map.ssh).expanduser()
        if cfg.map.ssh
        else data_dir
        / "environment_data"
        / "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D_1773078040918.nc"
    )
    ssh_full = xr.open_dataset(fn)

    ssh_n23 = ssh_full.sel({"time": pd.Timestamp(cfg.map.ssh_dates[0])})["adt"]
    ssh_fctd = ssh_full.sel({"time": pd.Timestamp(cfg.map.ssh_dates[1])})["adt"]

    plt.rcParams["text.usetex"] = False
    plt.rcParams["font.size"] = 9

    coords = {
        "SION": [39 + 37.194 / 60, -63 - 39.427 / 60],
        "SIOE": [38 + 45.906 / 60, -62 - 15.626 / 60],
    }

    fig = plt.figure(figsize=(5, 3.7))

    ax1 = fig.add_subplot(
        1,
        1,
        1,
        projection=ccrs.Orthographic(central_longitude=-62, central_latitude=35),
    )
    ax1.set_extent([-65, -61, 37.5, 40], ccrs.Geodetic())

    ## MAP
    plt.sca(ax1)
    # Add gridlines with labels only on bottom and left
    gl = ax1.gridlines(
        draw_labels=True,
        dms=False,
        x_inline=False,
        y_inline=False,
        color="#dddddd",
        alpha=0.2,
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = True
    gl.left_labels = True
    gl.xlabel_style = {"size": 12}
    gl.ylabel_style = {"size": 12}

    # Add axis labels
    ax1.text(
        0.5,
        -0.1,
        "Longitude",
        transform=ax1.transAxes,
        ha="center",
        va="top",
        fontsize=14,
    )
    ax1.text(
        -0.2,
        0.5,
        "Latitude",
        transform=ax1.transAxes,
        ha="right",
        va="center",
        rotation=90,
        fontsize=14,
    )

    # Bathymetry
    (-bathy).plot(
        cmap="mako",
        transform=ccrs.PlateCarree(),
        ax=ax1,
        vmax=6000,
        vmin=500,
        add_colorbar=False,
        rasterized=True,
    )

    bathy.plot.contour(
        x="lon",
        y="lat",
        transform=ccrs.PlateCarree(),
        ax=ax1,
        colors="k",
        linewidths=0.3,
        alpha=0.8,
        levels=np.linspace(-5000, -1000, 15),
        linestyles="-",
    )

    # FCTD (hydrographic survey track)
    plt.plot(
        fctd["longitude"],
        fctd["latitude"],
        c=whoi_colors[3],
        transform=ccrs.PlateCarree(),
        lw=3,
        solid_capstyle="round",
        zorder=100,
    )

    # Sea Surface Height (solid: during NESCAFE23, dashed: during the FCTD survey)
    ssh_n23.plot.contour(
        transform=ccrs.PlateCarree(),
        x="longitude",
        y="latitude",
        levels=[0.25],
        colors=colors[2],
        linestyles="solid",
        linewidths=2,
    )
    ssh_fctd.plot.contour(
        transform=ccrs.PlateCarree(),
        x="longitude",
        y="latitude",
        levels=[0.25],
        colors=colors[2],
        linestyles="dashed",
        linewidths=2,
    )
    plt.title("")

    # Moorings
    for mooring in coords:
        ax1.scatter(
            coords[mooring][1],
            coords[mooring][0],
            transform=ccrs.PlateCarree(),
            c="w",
            edgecolor="k",
            lw=1.5,
            s=60,
            zorder=1000,
        )

    # Callouts (positions match reference/Map.png)
    geo = ccrs.PlateCarree()._as_mpl_transform(ax1)
    bbox = {
        "boxstyle": "round,pad=0.45,rounding_size=0.3",
        "facecolor": "#f0f2f2",
        "edgecolor": whoi_colors[3],
        "linewidth": 1.5,
    }
    line = {
        "arrowstyle": "-",
        "color": whoi_colors[3],
        "lw": 2,
        "shrinkA": 0,
        "shrinkB": 7,
    }

    ax1.annotate(
        "SION",
        xy=(coords["SION"][1], coords["SION"][0]),
        xycoords=geo,
        xytext=(0.138, 0.904),
        textcoords="axes fraction",
        ha="center",
        va="center",
        fontsize=14,
        color=whoi_colors[3],
        bbox=bbox,
        arrowprops=line,
        zorder=2000,
    )
    ax1.annotate(
        "SIOE",
        xy=(coords["SIOE"][1], coords["SIOE"][0]),
        xycoords=geo,
        xytext=(0.882, 0.351),
        textcoords="axes fraction",
        ha="center",
        va="center",
        fontsize=14,
        color=whoi_colors[3],
        bbox=bbox,
        arrowprops=line,
        zorder=2000,
    )
    track_idx = int(np.argmin(np.abs(np.asarray(fctd["latitude"]) - 39.25)))
    ax1.annotate(
        "Hydrographic\nSurvey Track",
        xy=(
            float(np.asarray(fctd["longitude"])[track_idx]),
            float(np.asarray(fctd["latitude"])[track_idx]),
        ),
        xycoords=geo,
        xytext=(0.739, 0.834),
        textcoords="axes fraction",
        ha="center",
        va="center",
        fontsize=11,
        color=whoi_colors[3],
        bbox=bbox,
        arrowprops={**line, "shrinkB": 6},
        zorder=2000,
    )
    # Along-geodesic shift of the 0.25 m SSH contour between the two dates
    if cfg.map.front_shift:
        _annotate_front_shift(
            ax1,
            [ssh_n23, ssh_fctd],
            (coords["SION"][0], coords["SION"][1]),
            (coords["SIOE"][0], coords["SIOE"][1]),
            cfg.map.front_shift_offset_km,
            color=colors[2],
            bbox=bbox,
            text_color=whoi_colors[3],
        )

    shadow = [patheffects.withSimplePatchShadow(offset=(1.5, -1.5), alpha=0.3)]
    for text in ax1.texts:
        if text.get_bbox_patch() is not None:
            text.get_bbox_patch().set_path_effects(shadow)

    # SSH legend (in place of the "0.25 m SSH contour" callout)
    leg = ax1.legend(
        handles=[
            Line2D([0], [0], color=colors[2], lw=2, linestyle="solid"),
            Line2D([0], [0], color=colors[2], lw=2, linestyle="dashed"),
        ],
        labels=[_format_date(d) for d in cfg.map.ssh_dates],
        title="0.25 m SSH contour",
        title_fontsize=11,
        fontsize=10,
        labelcolor=whoi_colors[3],
        handlelength=2.6,
        loc="center",
        bbox_to_anchor=(0.6, 0.125),
        fancybox=True,
        framealpha=1,
        borderpad=0.6,
    )
    leg.get_title().set_color(whoi_colors[3])
    leg.set_zorder(2000)
    frame = leg.get_frame()
    frame.set_facecolor("#f0f2f2")
    frame.set_edgecolor(whoi_colors[3])
    frame.set_linewidth(1.5)
    frame.set_path_effects(shadow)

    # Globe inset (from the `map_inset` cell)
    ax_global = ax1.inset_axes(
        [-0.035, -0.03, 0.36, 0.36],
        projection=ccrs.Orthographic(central_longitude=-40, central_latitude=30),
        zorder=1500,
    )
    ax_global.set_global()

    lon_min, lon_max, lat_min, lat_max = -100, -20, 20, 70
    lons_bottom = np.linspace(lon_min, lon_max, 100)
    lons_top = np.linspace(lon_max, lon_min, 100)
    lats_left = np.linspace(lat_min, lat_max, 100)
    lats_right = np.linspace(lat_max, lat_min, 100)

    verts = (
        list(zip(lons_bottom, [lat_min] * 100, strict=True))
        + list(zip([lon_max] * 100, lats_left, strict=True))
        + list(zip(lons_top, [lat_max] * 100, strict=True))
        + list(zip([lon_min] * 100, lats_right, strict=True))
    )
    boundary = mpath.Path(verts)
    ax_global.set_boundary(boundary, transform=ccrs.PlateCarree())

    ax_global.coastlines(resolution="50m", linewidth=0.5)
    ax_global.add_feature(
        cfeature.LAND.with_scale("50m"), facecolor="lightgray", alpha=1
    )
    ax_global.add_feature(
        cfeature.OCEAN.with_scale("50m"), facecolor=whoi_colors[3], alpha=1
    )

    lon_min, lon_max = -65, -61
    lat_min, lat_max = 37.5, 40
    rect_lons = [lon_min, lon_max, lon_max, lon_min, lon_min]
    rect_lats = [lat_min, lat_min, lat_max, lat_max, lat_min]
    ax_global.plot(
        rect_lons,
        rect_lats,
        color=whoi_colors[1],
        linewidth=1,
        transform=ccrs.PlateCarree(),
        zorder=100,
    )
    ax_global.fill(
        rect_lons,
        rect_lats,
        color=whoi_colors[1],
        alpha=1,
        transform=ccrs.PlateCarree(),
        zorder=99,
    )
    ax_global.set_extent([-100, -20, 0, 70], ccrs.PlateCarree())

    fig.subplots_adjust(left=0.2, right=0.955, bottom=0.13, top=0.99)
    return fig


def plot_ctd_fctd(cfg: PublicationFiguresConfig) -> Figure:
    """``publication_figures/CTD_FCTD.ipynb``: mooring CTD and FCTD sections."""
    fctd = n23.compute_filtered_fctd_grid(r_smooth=None, z_smooth=None)
    sioe_ctd = n23.open_sioe_ctd()
    sion_ctd = n23.open_sion_ctd()

    plt.rcParams.update({"font.size": 8})
    plt.rcParams["text.usetex"] = cfg.usetex
    fig = plt.figure(figsize=(6.67, 3.5))
    gs = gridspec.GridSpec(2, 3)

    ax_ctd = fig.add_subplot(gs[:, 0])

    ax_fctd_temp = fig.add_subplot(gs[0, 1:])
    ax_fctd_sal = fig.add_subplot(gs[1, 1:])

    # FCTD temp
    plt.sca(ax_fctd_temp)
    fctd["t"].assign_coords({"range": fctd.range / 1000 + 35}).plot(
        x="range",
        cmap=cmocean.cm.thermal,
        cbar_kwargs={"label": "[deg C]"},
        rasterized=True,
    )
    plt.ylim([1500, 0])
    plt.xlim([0, 155])
    plt.xlabel("")
    plt.ylabel("depth [m]")
    plt.title("In-Situ Temperature", loc="left")
    ax_fctd_temp.set_xticklabels([])

    plt.sca(ax_fctd_sal)
    fctd["SA"].assign_coords({"range": fctd.range / 1000 + 35}).plot(
        x="range",
        cmap=cmocean.cm.haline,
        cbar_kwargs={"label": "[g/kg]"},
        rasterized=True,
    )
    plt.ylim([1500, 0])
    plt.xlim([0, 155])
    plt.xlabel("range [km]")
    plt.ylabel("depth [m]")
    plt.title("Absolute Salinity", loc="left")

    plt.sca(ax_ctd)
    plt.plot(
        sion_ctd["c"],
        sion_ctd.depth,
        lw=3,
        c=whoi_colors[0],
        zorder=100,
        alpha=0.8,
        label="SION",
    )
    plt.plot(
        sioe_ctd["c"],
        sioe_ctd.depth,
        lw=3,
        c=whoi_colors[1],
        zorder=100,
        alpha=0.8,
        label="SIOE",
    )
    plt.plot(fctd["c"][:, ::50], fctd.depth, c="gray", lw=1)
    plt.ylim([5000, 0])
    plt.xlim([1475, 1550])
    plt.ylabel("depth [m]")
    plt.title("Sound-Speed", loc="left")
    plt.legend(loc="lower left")

    plt.tight_layout()
    label_panels([ax_ctd, ax_fctd_temp, ax_fctd_sal], "upper left")
    return fig


def plot_arrival_time_shift(cfg: PublicationFiguresConfig) -> Figure:
    """Common arrival-time shift per transmission (from ``compute arrival time offset``)."""
    T_t = _load_pickle("T_t_estimated.pkl", "compute arrival time offset")
    if "dT_t_upper" not in T_t or "dT_t_lower" not in T_t:
        msg = (
            "T_t_estimated.pkl has no per-array offsets (dT_t_upper/dT_t_lower). It was "
            "probably written by the base `nescafe23` CLI or an older version of this "
            "project; rerun 'compute arrival time offset' in `n23_fluctuation_analysis`."
        )
        raise KeyError(msg)
    dT_t_upper = T_t["dT_t_upper"]
    dT_t_lower = T_t["dT_t_lower"]
    dT_avg = T_t["dT_t"]

    plt.rcParams.update({"font.size": 8})
    plt.rcParams.update({"text.usetex": cfg.usetex})

    fig = plt.figure(figsize=(3.37, 2))
    plt.plot(
        np.arange(1, 24), dT_t_upper, label="upper array", c="k", linestyle="--", lw=1
    )
    plt.plot(
        np.arange(1, 24), dT_t_lower, label="lower array", c="k", linestyle="-.", lw=1
    )
    plt.plot(np.arange(1, 24), dT_avg, c="k", lw=1, label="average")
    plt.ylabel("$\\Delta \\tau$ [s]")
    plt.legend()
    plt.xlabel("transmission number")
    plt.tight_layout()
    return fig


def plot_2d_slice(cfg: PublicationFiguresConfig) -> Figure:
    """``publication_figures/2D_slice.ipynb``: sound speed and N with the ducted eigenrays."""
    dss = _receptions()

    fctd_smooth = tools.compute_filtered_fctd_grid(z_smooth=200, r_smooth=15000)
    fctd_smooth = fctd_smooth.assign_coords(
        {"range": (fctd_smooth.range.to_numpy() + 35e3) / 1000}
    )
    fctd_smooth = fctd_smooth.interpolate_na("depth")

    ray_run = _ray_run()
    eigs = ray_run["eigs"]

    ray_ids_plot = ["+7", "-7", "-6", "+6", "-8"]

    sns.set_palette("Set2")

    plt.rcParams.update({"font.size": "7"})
    plt.rcParams.update({"text.usetex": cfg.usetex})

    fig, axes = plt.subplots(1, 2, figsize=(6.67, 2))
    axes = axes.flatten()

    plt.sca(axes[0])
    fctd_smooth["c"].plot(cmap="mako", rasterized=True)
    plt.title("Sound Speed", loc="left")

    for ray_id_idx, ray_id in enumerate(ray_ids_plot):
        for k in range(len(tools.EIG_INDEXS[ray_id])):
            ray_id_pr, (_sign, _number, _suffix) = tools.parse_ray_id(ray_id)
            eig_idx = np.where(eigs.ray_id[0] == ray_id_pr)[0][
                tools.EIG_INDEXS[ray_id][k][0]
            ]
            plt.plot(
                eigs.rs[0][eig_idx, :] / 1000,
                -eigs.zs[0][eig_idx, :],
                c=f"C{ray_id_idx}",
                lw=0.75,
                alpha=1,
            )

    plt.fill_between(fctd_smooth.range, 10000, 4908, lw=0, color="gray")

    # plot moorings
    plt.scatter(
        x=[153] * 60, y=dss[4].actual_depth.values, c="w", lw=0, s=1, zorder=100
    )
    plt.scatter(x=0, y=2300, s=25, lw=1, c="w", edgecolor="k", zorder=100)

    plt.xlim([-10, 160])
    plt.ylim([5100, 0])
    plt.xlabel("range [km]")
    plt.ylabel("depth [m]")

    plt.sca(axes[1])
    fctd_smooth["N"].plot(cmap="mako", vmin=0, vmax=5, rasterized=True)

    plt.title("Buoyancy Frequency", loc="left")

    for ray_id_idx, ray_id in enumerate(ray_ids_plot):
        for k in range(len(tools.EIG_INDEXS[ray_id])):
            ray_id_pr, (_sign, _number, _suffix) = tools.parse_ray_id(ray_id)
            eig_idx = np.where(eigs.ray_id[0] == ray_id_pr)[0][
                tools.EIG_INDEXS[ray_id][k][0]
            ]
            ray_id_label = ray_id if k == 0 else None
            plt.plot(
                eigs.rs[0][eig_idx, :] / 1000,
                -eigs.zs[0][eig_idx, :],
                c=f"C{ray_id_idx}",
                label=ray_id_label,
                lw=0.75,
                alpha=1,
            )

    leg = plt.legend(loc="lower right", ncols=3, framealpha=1)

    for line in leg.get_lines():
        line.set_linewidth(3)

    plt.fill_between(fctd_smooth.range, 10000, 4908, lw=0, color="gray")

    # plot moorings
    plt.scatter(
        x=[153] * 60, y=dss[4].actual_depth.values, c="w", lw=0, s=1, zorder=100
    )
    plt.scatter(x=0, y=2300, s=35, lw=1, c="w", edgecolor="k", zorder=100)

    plt.xlim([-10, 160])
    plt.ylim([5100, 0])
    plt.xlabel("range [km]")
    plt.ylabel("depth [m]")

    plt.tight_layout(pad=0)
    # the receiver-array markers at 153 km sit in the upper right
    label_panels(axes, "upper left")
    return fig


def plot_acoustic_receptions(cfg: PublicationFiguresConfig) -> Figure:
    """``publication_figures/Acoustic_Receptions.ipynb``: mean pulse and a beamformed reception."""
    plt.rcParams.update({"font.size": 8})
    plt.rcParams.update({"text.usetex": cfg.usetex})

    mean_pulse_lower = xr.open_dataarray(
        _dataproduct("mean_pulse_lower.nc", "compute mean pulse")
    )
    mean_pulse_lower_noise = xr.open_dataarray(
        _dataproduct("mean_pulse_lower_noise.nc", "compute mean pulse")
    )

    dss = _receptions()

    Tx_idx = 12

    thetas = np.linspace(-40, 40, 200)

    bf_lower = n23.ULA_conventional_beamform(
        dss[Tx_idx][32:]
        .swap_dims({"hydrophone": "nominal_depth"})
        .rename({"nominal_depth": "depth"}),
        thetas,
    )

    bf_lower_noise = 20 * np.log10(
        np.sqrt(np.abs(bf_lower.sel({"time": slice(90, 95)}) ** 2).mean(dim="time"))
    )

    fig = plt.figure(figsize=(6.67, 4))

    ax_pulse = plt.subplot(2, 1, 1)
    (20 * np.log10(mean_pulse_lower) - mean_pulse_lower_noise).sel(
        {"depth": slice(950, 1250)}
    ).plot(
        cmap=cmps.mako_r_white(),
        vmin=5,
        vmax=30,
        cbar_kwargs={"label": "SNR [dB]"},
        rasterized=True,
    )

    plt.xlim([100.25, 101.75])
    plt.ylim([1250, 950])
    plt.ylabel("depth [m]")
    plt.xlabel("")
    plt.gca().set_xticklabels("")
    plt.title("Acoustic Reception: Mean Pulse", loc="left", pad=-2)

    ax_bf = plt.subplot(2, 1, 2)

    (20 * np.log10(np.abs(bf_lower)) - bf_lower_noise).plot(
        x="time",
        cmap=cmps.mako_r_white(),
        vmin=7,
        vmax=40,
        cbar_kwargs={"label": "SNR [dB]"},
        rasterized=True,
    )
    plt.xlim([100.25, 101.75])
    plt.ylim([-25, 25])
    plt.xlabel("time [s]")
    plt.ylabel("angle [deg]")
    plt.title("Single Beamformed Reception", loc="left", pad=-2)

    # Arrival annotations (added by hand to the published figure; positions match
    # reference/Acoustic_Receptions.pdf). (label, ellipse center, width, height, label xy)
    arrivals = [
        ("-5b", (100.505, 11.2), 0.183, 16.7, (100.387, 17.3)),
        ("+6", (100.851, 7.6), 0.062, 9.9, (100.80, 11.3)),
        ("+7", (100.917, 5.4), 0.059, 9.6, (100.979, 3.9)),
        ("-6", (100.738, -11.6), 0.088, 11.0, (100.658, -7.2)),
        ("-7/+8", (101.065, -11.3), 0.115, 12.8, (101.19, -12.7)),
        ("+6b", (101.295, 14.5), 0.062, 8.4, (101.373, 9.4)),
        ("+7b", (101.728, -21.2), 0.059, 8.4, (101.648, -16.0)),
    ]
    for label, center, width, height, label_xy in arrivals:
        plt.gca().add_patch(
            mpatches.Ellipse(
                center,
                width,
                height,
                facecolor="none",
                edgecolor=whoi_colors[1],
                alpha=0.75,
                lw=1.2,
                zorder=100,
                clip_on=False,
            )
        )
        plt.text(
            *label_xy,
            f"\\textsf{{\\textbf{{{label}}}}}" if cfg.usetex else label,
            color=whoi_colors[1],
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=100,
        )

    plt.tight_layout(pad=0.25)
    # arrivals run along the right edge of both panes
    label_panels([ax_pulse, ax_bf], "upper left")
    return fig


def plot_phase_structure_integrand(cfg: PublicationFiguresConfig) -> Figure:
    """``publication_figures/ray_tube.ipynb``: ray paths, ray separation and D(Δz) integrand."""
    ray_vars = tools.compute_n23_ray_variables()

    ray_ids = ["-8", "+7", "-7", "-6", "+6"]
    integrand = {}
    delta_zr = np.array([200])

    ray_z = {}
    zeta1 = {}

    for ray_id in ray_ids:
        integrand[ray_id] = []
        ray_z[ray_id] = []
        zeta1[ray_id] = []

        for rv in ray_vars[ray_id]:
            integrand[ray_id].append(
                tools.compute_phase_structure_integrand(rv, delta_zr)
            )
            ray_z[ray_id].append(rv["zs"])
            zeta1[ray_id].append(rv["RT_zeta_1"])

    sns.set_palette("colorblind")

    plt.rcParams.update({"font.size": 8})
    plt.rcParams["text.usetex"] = cfg.usetex

    fig, axes = plt.subplots(3, 1, figsize=(6.67, 4))

    plt.sca(axes[2])

    for ray_id in ["-7", "+6", "-6"]:
        if ray_id == "-6":
            plt.plot(
                ray_vars[ray_id][0]["rs"] / 1000,
                integrand[ray_id][0][:, -1],
                label=f"${ray_id}_{{i}}$",
                alpha=0.85,
            )
            plt.plot(
                ray_vars[ray_id][1]["rs"] / 1000,
                integrand[ray_id][1][:, -1],
                label=f"${ray_id}_{{ii}}$",
                alpha=0.85,
            )
        else:
            plt.plot(
                ray_vars[ray_id][0]["rs"] / 1000,
                integrand[ray_id][0][:, -1],
                label=ray_id,
                alpha=0.85,
            )

    plt.title("")

    plt.xlabel("range [km]")
    plt.title("$D(\\Delta z)$ integrand", loc="left")
    plt.ylabel("")

    plt.sca(axes[0])

    for ray_id in ["-7", "+6", "-6"]:
        if ray_id == "-6":
            plt.plot(
                ray_vars[ray_id][0]["rs"] / 1000,
                -ray_z[ray_id][0],
                label=f"${ray_id}_{{i}}$",
                alpha=0.85,
            )
            plt.plot(
                ray_vars[ray_id][1]["rs"] / 1000,
                -ray_z[ray_id][1],
                label=f"${ray_id}_{{ii}}$",
                alpha=0.85,
            )

        else:
            plt.plot(
                ray_vars[ray_id][0]["rs"] / 1000,
                -ray_z[ray_id][0],
                label=ray_id,
                alpha=0.85,
            )

    plt.ylim([3000, 000])
    plt.ylabel("depth [m]")

    leg = plt.legend(ncols=4, framealpha=1)

    # change the line width for the legend
    for line in leg.get_lines():
        line.set_linewidth(4.0)

    plt.title("Ray paths", loc="left")
    axes[0].set_xticklabels([])
    plt.sca(axes[1])

    for ray_id in ["-7", "+6", "-6"]:
        if ray_id == "-6":
            plt.plot(
                ray_vars[ray_id][0]["rs"] / 1000,
                zeta1[ray_id][0],
                label=f"${ray_id}_{{i}}$",
                alpha=0.85,
            )
            plt.plot(
                ray_vars[ray_id][1]["rs"] / 1000,
                zeta1[ray_id][1],
                label=f"${ray_id}_{{ii}}$",
                alpha=0.85,
            )

        else:
            plt.plot(
                ray_vars[ray_id][0]["rs"] / 1000,
                zeta1[ray_id][0],
                label=ray_id,
                alpha=0.85,
            )

    plt.ylabel("ray separation [m]")
    plt.title("Ray separation $\\xi_1$", loc="left")
    axes[1].set_xticklabels([])

    plt.tight_layout(pad=0)
    # the xi_1 curves and the integrand spikes reach the upper right
    label_panels(axes, "upper left")
    return fig


def plot_F_phi_chi_ducted(cfg: PublicationFiguresConfig) -> Figure:
    """``publication_figures/Phi_Chi_Spectra.ipynb``: observed vs. GM-model phase/log-amplitude spectra."""
    whoi_colors = ["#00B7BD", "#EE5340"]

    plt.rcParams.update({"font.size": 8})
    plt.rcParams["text.usetex"] = cfg.usetex

    colors = [
        whoi_colors[0],
        lighten_color(whoi_colors[0], lightness_scale=2),
        whoi_colors[1],
        lighten_color(whoi_colors[1], lightness_scale=1.4, saturation_scale=1),
    ]

    # open observations
    m_data, _F_phi_data, _F_chi_data, F_phi_lm, F_phi_ls, F_chi_lm, F_chi_ls = (
        tools.estimate_F_phi_chi()
    )

    # open ls parameters for spectra + noise model
    results = _load_pickle("phi_chi_LS_fit.pkl", "ɸ χ GM least squares fit")

    m = 2 * np.pi * np.logspace(-5, 0, 500)[None, :]  # [rad / m]

    fig, axes = plt.subplots(2, 2, figsize=(6.67, 5))
    axes = axes.flatten()

    for k, ray_id in enumerate(["+7", "-7", "+6", "-6"]):
        F_phis_model, F_chis_model, F_phis_model_nonoise, F_chis_model_nonoise = (
            tools.phi_chi_spectrum_model(
                m,
                results[ray_id].x[0],
                results[ray_id].x[1],
                results[ray_id].x[2],
                ray_id,
                all_paths=True,
            )
        )
        F_phi_model, F_chi_model = F_phis_model[0], F_chis_model[0]

        # compute model spectra for -8 as well (if +7 arrival)
        if ray_id == "+7":
            _, _, F_phi_model_nonoise2, F_chi_model_nonoise2 = (
                tools.phi_chi_spectrum_model(
                    m,
                    results[ray_id].x[0],
                    results[ray_id].x[1],
                    results[ray_id].x[2],
                    "-8",
                )
            )

        # extend noise floor to the right
        nan_mask = np.isnan(F_phi_model)
        F_phi_model = np.hstack((F_phi_model[~nan_mask], F_phi_model[~nan_mask][-1]))
        F_chi_model = np.hstack((F_chi_model[~nan_mask], F_chi_model[~nan_mask][-1]))
        m_extend = np.concatenate((m[:, ~nan_mask], np.array([[100]])), axis=1)

        plt.sca(axes[k])

        # WFT GMIW F_phi
        for F_phi_model_nonoise in F_phis_model_nonoise:
            plt.plot(
                m[0] / (2 * np.pi), F_phi_model_nonoise, c=colors[0], lw=1.5, alpha=0.85
            )

        if ray_id == "+7":
            plt.plot(
                m[0] / (2 * np.pi),
                F_phi_model_nonoise2,
                c=colors[0],
                linestyle="dashdot",
                lw=1.5,
                alpha=0.85,
            )
        plt.plot(
            m_extend[0] / (2 * np.pi),
            F_phi_model,
            c=colors[0],
            lw=0.75,
            linestyle="dashed",
        )

        # F_phi observed
        plt.plot(m_data, F_phi_lm[ray_id], c=colors[1], marker="o", markersize=3)
        plt.fill_between(
            m_data,
            F_phi_lm[ray_id] * F_phi_ls[ray_id],
            F_phi_lm[ray_id] / F_phi_ls[ray_id],
            alpha=0.3,
            color=colors[1],
            lw=0,
        )

        # WFT GMIW F_chi

        for F_chi_model_nonoise in F_chis_model_nonoise:
            plt.plot(
                m[0] / (2 * np.pi), F_chi_model_nonoise, c=colors[2], lw=1.5, alpha=0.85
            )

        if ray_id == "+7":
            plt.plot(
                m[0] / (2 * np.pi),
                F_chi_model_nonoise2,
                c=colors[2],
                linestyle="dashdot",
                lw=1.5,
                alpha=0.85,
            )
        plt.plot(
            m_extend[0] / (2 * np.pi),
            F_chi_model,
            c=colors[2],
            lw=0.75,
            linestyle="dashed",
        )

        # F_chi observed
        plt.plot(m_data, F_chi_lm[ray_id], c=colors[3], marker="o", markersize=3)
        plt.fill_between(
            m_data,
            F_chi_lm[ray_id] * F_chi_ls[ray_id],
            F_chi_lm[ray_id] / F_chi_ls[ray_id],
            alpha=0.3,
            color=colors[3],
            lw=0,
        )

        plt.ylim([1e-5, 3e4])
        plt.xlim([2e-4, 4e-1])

        plt.xlabel("vertical wave number [1/m]")
        plt.ylabel("$F_{\\phi, \\chi}$ [$\\mathrm{rad} \\ m$]")

        plt.yscale("log")
        plt.xscale("log")

        plt.grid()

        if ray_id == "+7":
            plt.text(
                0.65,
                0.9,
                f"Ray ID: {ray_id}, -8",
                transform=axes[k].transAxes,
                fontsize=9,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "k",
                    "pad": 3,
                    "linewidth": 0.5,
                },
            )

        else:
            plt.text(
                0.73,
                0.9,
                f"Ray ID: {ray_id}",
                transform=axes[k].transAxes,
                fontsize=9,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "k",
                    "pad": 3,
                    "linewidth": 0.5,
                },
            )

        plt.text(
            0.04,
            0.07,
            f"{results[ray_id].x[0]:0.1f} $\\times$ GM Energy\n$F_{{\\phi}}$ noise level = {results[ray_id].x[1]:0.2f} [rad m]\n$F_{{\\chi}}$ noise level = {results[ray_id].x[2]:0.2f} [rad m]",
            transform=axes[k].transAxes,
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "k", "pad": 3, "linewidth": 0.5},
        )

        # add +7,-8 legend
        if ray_id == "+7":
            # Create custom legend handles: for each entry, overlay one Line2D per
            # whoi_color so the legend swatch shows both colors stacked, signaling
            # that the linestyle pattern applies to both.
            solid_lines = tuple(
                mlines.Line2D([], [], color=c, linestyle="-", linewidth=1.8)
                for c in whoi_colors
            )
            dashed_lines = tuple(
                mlines.Line2D([], [], color=c, linestyle="-.", linewidth=1.8)
                for c in whoi_colors
            )

            # Add the custom legend
            axes[k].legend(
                handles=[solid_lines, dashed_lines],
                labels=["+7", "-8"],
                handler_map={tuple: HandlerTupleVertical(ndivide=None, pad=0.1)},
                ncols=1,
                loc=(0.75, 0.05),
            )

    plt.tight_layout()
    # the Ray ID box takes the upper right
    label_panels(axes, "upper left")
    return fig


def plot_ducted_intensity_distribution(cfg: PublicationFiguresConfig) -> Figure:
    """``publication_figures/IntensityDistributions.ipynb`` (cell 7): ducted intensity / log-intensity PDFs."""
    rng = np.random.default_rng(cfg.seed)

    wavefront_values = _wavefront_values()

    ducted_peaks = ["+7", "-7", "+6", "-6"]

    I = {}  # noqa: E741
    lI = {}
    lI_dB = {}
    Ibar = {}
    lIbar = {}
    sigma_l = {}
    I_norm = {}
    lI_norm = {}
    SI = {}
    sigma_iota = {}

    for ray_id in wavefront_values:
        I[ray_id] = np.abs(wavefront_values[ray_id]) ** 2
        lI[ray_id] = np.log(I[ray_id])
        lI_dB[ray_id] = 10 * np.log10(I[ray_id])

        Ibar[ray_id] = np.mean(I[ray_id])
        lIbar[ray_id] = np.mean(lI[ray_id])
        sigma_l[ray_id] = np.std(lI[ray_id])

        I_norm[ray_id] = (I[ray_id] / Ibar[ray_id]).flatten()
        lI_norm[ray_id] = ((lI[ray_id] - lIbar[ray_id]) / sigma_l[ray_id]).flatten()

        SI[ray_id] = (np.mean(I[ray_id] ** 2) - np.mean(I[ray_id]) ** 2) / (
            np.mean(I[ray_id]) ** 2
        )

        sigma_iota[ray_id] = np.sqrt(
            np.mean(lI_dB[ray_id] ** 2) - np.mean(lI_dB[ray_id]) ** 2
        )

    plt.rcParams.update({"font.size": 8})
    plt.rcParams.update({"text.usetex": cfg.usetex})

    fig, axes = plt.subplots(2, 4, figsize=(6.68, 3.8))
    axes = axes.flatten()

    for k, ray_id in enumerate(ducted_peaks):
        # Intensity Distribution
        plt.sca(axes[k])

        bins = np.linspace(0, 10, 40)

        _ = plt.hist(
            I_norm[ray_id],
            bins=bins,
            alpha=0.3,
            density=True,
            color="gray",
        )

        plt.scatter(
            y=3e-3 * rng.uniform(0.7, 1.3, 644),
            x=I_norm[ray_id].flatten(),
            lw=0,
            alpha=0.3,
            s=1,
            c="k",
        )

        x_plot = np.linspace(0, 10, 1000)
        # exponential distribution
        plt.plot(x_plot, np.exp(-x_plot), c="k")

        # modulated exponential
        a = 1 / (SI[ray_id] - 1)
        plt.plot(
            x_plot,
            modulated_exp_gamma(x_plot, a),
            c="k",
            linestyle="--",
            label="Modulated Exponential",
        )
        plt.yscale("log")
        plt.ylim([1e-3, 1e0])

        if ray_id == "+7":
            plt.text(
                9.8,
                3e-1,
                f"Ray ID: +7/-8\nSI = {SI[ray_id]:0.2f}",
                horizontalalignment="right",
            )
        else:
            plt.text(
                9.8,
                3e-1,
                f"Ray ID: {ray_id}\nSI = {SI[ray_id]:0.2f}",
                horizontalalignment="right",
            )
        plt.xlabel("$ I / \\langle I \\rangle$")
        plt.xlim([0, 10])
        plt.xticks(np.arange(0, 10, 3))

        # Log-Intensity Distribution
        plt.sca(axes[k + 4])
        _ = plt.hist(
            lI_norm[ray_id],
            bins=np.linspace(-6, 3, 40),
            alpha=0.3,
            density=True,
            color="gray",
        )

        plt.scatter(
            y=3e-3 * rng.uniform(0.7, 1.3, 644),
            x=lI_norm[ray_id].flatten(),
            lw=0,
            alpha=0.3,
            s=1,
            c="k",
        )

        x_plot = np.linspace(-6, 4, 1000)
        plt.plot(x_plot, np.exp(x_plot - np.exp(x_plot)), linestyle="-", color="k")
        sigma = 1
        f_norm = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * (x_plot / sigma) ** 2)
        plt.plot(x_plot, f_norm, linestyle="-.", color="k")

        plt.yscale("log")
        plt.ylim([1e-3, 1e0])
        plt.xlim([-6, 4])
        plt.xticks(np.arange(-6, 4, 3))
        plt.xlabel(
            "$(\\mathrm{ln \\ I} - \\langle \\mathrm{ln \\ I} \\rangle) / \\sigma_{\\mathrm{ln \\ I}}$"
        )

        if ray_id == "+7":
            plt.text(
                -5.8,
                3e-1,
                f"Ray ID: +7/-8\n$\\sigma_{{\\iota}}$ = {sigma_iota[ray_id]:0.2f} dB",
            )
        else:
            plt.text(
                -5.8,
                3e-1,
                f"Ray ID: {ray_id}\n$\\sigma_{{\\iota}}$ = {sigma_iota[ray_id]:0.2f} dB",
            )

    # axes level formatting
    for k in range(8):
        if k % 4 == 0:
            axes[k].set_ylabel("Probability Density")
        else:
            axes[k].set_yticklabels("")

    plt.tight_layout(pad=0, h_pad=2)

    fig.subplots_adjust(bottom=0.18)

    custom_lines = [
        Line2D([0], [0], color="k", linestyle="-", linewidth=2),
        Line2D([0], [0], color="k", linestyle="--", linewidth=2),
        Line2D([0], [0], color="k", linestyle="-.", linewidth=2),
    ]
    labels = ["Exponential", "Modulated Exponential", "Log-Normal"]

    fig.legend(
        custom_lines,
        labels,
        loc="lower center",
        ncol=len(labels),
        frameon=True,
        handlelength=2.5,
    )
    return fig


def plot_ducted_vc(cfg: PublicationFiguresConfig) -> Figure:
    """``publication_figures/Vertical_Coherence.ipynb``: observed vs. modelled vertical coherence."""
    plt.rcParams["font.size"] = 8
    plt.rcParams["text.usetex"] = cfg.usetex

    wavefront_values = _wavefront_values()

    MC = {}
    AC = {}
    MC_var = {}

    for ray_id in wavefront_values:
        dz, MC[ray_id], MC_var[ray_id], _, _ = tools.estimate_mutual_coherence(
            wavefront_values[ray_id], 9
        )

        _, AC[ray_id], _, _, _ = tools.estimate_mutual_coherence(
            np.abs(wavefront_values[ray_id]), 9
        )

    latitude = 39.192499999
    Omega = 7.292115e-5  # Earth Rotation Speed [rad/s]
    f = 2 * Omega * np.sin(np.radians(latitude))  # [rad/s]

    delta_zr = np.linspace(0, 500, 200)

    zeta0 = 7.3 * np.sqrt(1)
    rvs = tools.compute_n23_ray_variables(f=f, zeta0=zeta0)

    Dzs = {}
    for ray_id in rvs:
        Dzs[ray_id] = []
        for rv in rvs[ray_id]:
            # compute_phase_structure_function now returns (D_dzr, integrand)
            Dzs[ray_id].append(tools.compute_phase_structure_function(rv, delta_zr)[0])

    zeta0 = 7.3 * np.sqrt(2)
    rvs = tools.compute_n23_ray_variables(f=f, zeta0=zeta0)

    Dzs2 = {}
    for ray_id in rvs:
        Dzs2[ray_id] = []
        for rv in rvs[ray_id]:
            # compute_phase_structure_function now returns (D_dzr, integrand)
            Dzs2[ray_id].append(tools.compute_phase_structure_function(rv, delta_zr)[0])

    pair_ids = {
        "+7": [0],
        "-7": [0],
        "+6": [0],
        "-6": [0, 1, 2],
    }

    fig, axes = plt.subplots(2, 2, figsize=(6.67, 4))
    axes = axes.flatten()

    ray_ids = ["+7", "-7", "+6", "-6"]

    for k, ray_id in enumerate(ray_ids):
        plt.sca(axes[k])

        plt.errorbar(
            dz,
            MC[ray_id],
            np.sqrt(MC_var[ray_id]),
            elinewidth=1,
            lw=0,
            capsize=3,
            marker="o",
            markersize=3,
            color="k",
            zorder=100,
        )

        plt.ylim([0, 1.1])

        plt.xlabel("$\\Delta z$ [m]")
        plt.ylabel("vertical coherence")

        if ray_id == "+7":
            plt.text(
                12,
                0.1,
                f"Ray ID: {ray_id}, -8",
                fontsize=9,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "k",
                    "pad": 3,
                    "linewidth": 0.5,
                },
            )
            # add legend# Create custom legend handles
            solid_line = mlines.Line2D(
                [], [], color="black", linestyle="-", linewidth=1.5, label="+7"
            )
            dashed_line = mlines.Line2D(
                [], [], color="black", linestyle="--", linewidth=1.5, label="-8"
            )

            # Add the custom legend
            leg = axes[k].legend(
                handles=[solid_line, dashed_line],
                ncols=2,
                loc=(0.03, 0.2),
                framealpha=1,
                fancybox=False,
                edgecolor="k",
            )
            leg.get_frame().set_linewidth(0.5)  # <-- thickness here
        else:
            plt.text(
                12,
                0.1,
                f"Ray ID: {ray_id}",
                fontsize=9,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "k",
                    "pad": 3,
                    "linewidth": 0.5,
                },
            )

        for pair_id in pair_ids[ray_id]:
            if ray_id == "+7":
                plt.plot(
                    delta_zr,
                    np.exp(-Dzs["-8"][pair_id] / 2),
                    c=whoi_colors[0],
                    lw=1.5,
                    linestyle="--",
                    label="-8",
                    alpha=0.85,
                )
                plt.plot(
                    delta_zr,
                    np.exp(-Dzs2["-8"][pair_id] / 2),
                    c=whoi_colors[1],
                    lw=1.5,
                    alpha=0.85,
                    linestyle="--",
                )
                plt.plot(
                    delta_zr,
                    np.exp(-Dzs[ray_id][pair_id] / 2),
                    c=whoi_colors[0],
                    lw=1.5,
                    alpha=0.85,
                    label="+7",
                )
                plt.plot(
                    delta_zr,
                    np.exp(-Dzs2[ray_id][pair_id] / 2),
                    c=whoi_colors[1],
                    lw=1.5,
                    alpha=0.85,
                )

            else:
                plt.plot(
                    delta_zr,
                    np.exp(-Dzs[ray_id][pair_id] / 2),
                    c=whoi_colors[0],
                    lw=1.5,
                    alpha=0.85,
                )
                plt.plot(
                    delta_zr,
                    np.exp(-Dzs2[ray_id][pair_id] / 2),
                    c=whoi_colors[1],
                    lw=1.5,
                    alpha=0.85,
                )

        plt.grid()
        plt.xlim([0, 300])

    plt.tight_layout(pad=0.5)

    fig.subplots_adjust(right=0.9)
    label_panels(axes)
    return fig


# ---------------------------------------------------------------------------
# Registry and driver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FigureSpec:
    """How to build and save one publication figure (defaults copy the notebook's ``savefig``)."""

    func: Callable[[PublicationFiguresConfig], Figure]
    default_format: str = "pdf"
    dpi: int | None = None
    bbox_inches: str | None = "tight"


# manuscript order; keys are the output file stems (matching ``reference/``)
FIGURES: dict[str, FigureSpec] = {
    "Map": FigureSpec(plot_map, "png", dpi=400, bbox_inches=None),
    "CTD_FCTD": FigureSpec(plot_ctd_fctd, dpi=400, bbox_inches=None),
    "common_arrival_time_shift_estimation": FigureSpec(plot_arrival_time_shift),
    "2D_slice": FigureSpec(plot_2d_slice, dpi=300, bbox_inches=None),
    "Acoustic_Receptions": FigureSpec(
        plot_acoustic_receptions, dpi=500, bbox_inches=None
    ),
    "phase_structure_integrand": FigureSpec(plot_phase_structure_integrand),
    "F_phi_chi_ducted": FigureSpec(plot_F_phi_chi_ducted),
    "Ducted_Intensity_Distribution": FigureSpec(plot_ducted_intensity_distribution),
    "Ducted_VC": FigureSpec(plot_ducted_vc),
}


@dataclass
class PublicationBuild:
    """Outcome of :func:`generate_publication_figures`."""

    written: list[Path] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)  # name -> traceback


def generate_publication_figures(
    config: str | Path | PublicationFiguresConfig,
    only: Sequence[str] | None = None,
    *,
    progress: Callable[[str], None] | None = None,
) -> PublicationBuild:
    """Build every figure named in the config (or ``only``) and save it to ``output_dir``.

    A figure that raises is recorded in ``failed`` and the others are still built.
    """
    cfg = (
        config if isinstance(config, PublicationFiguresConfig) else load_config(config)
    )
    which = tuple(only) if only is not None else cfg.which
    unknown = [name for name in which if name not in FIGURES]
    if unknown:
        msg = f"unknown figure(s) {unknown}; choose from {list(FIGURES)}"
        raise ConfigError(msg)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    build = PublicationBuild()

    for name in which:
        spec = FIGURES[name]
        if progress is not None:
            progress(name)
        fig = None
        try:
            # rcParams set inside a figure function (usetex, font, palette) stay local to it,
            # and are still in effect when the figure is saved
            with mpl.rc_context():
                fig = spec.func(cfg)
                for fmt in cfg.formats or (spec.default_format,):
                    out = cfg.output_dir / f"{name}.{fmt}"
                    fig.savefig(
                        out,
                        dpi=cfg.dpi or spec.dpi or "figure",
                        bbox_inches=spec.bbox_inches,
                    )
                    build.written.append(out)
        except Exception:
            build.failed[name] = traceback.format_exc()
        finally:
            if fig is not None:
                plt.close(fig)

    return build
