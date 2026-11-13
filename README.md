# NESCAFE23_fluctuations

[![Actions Status][actions-badge]][actions-link]
[![Documentation Status][rtd-badge]][rtd-link]

[![PyPI version][pypi-version]][pypi-link]
[![Conda-Forge][conda-badge]][conda-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

[![GitHub Discussion][github-discussions-badge]][github-discussions-link]

[![Coverage][coverage-badge]][coverage-link]

[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/format.json)](https://github.com/astral-sh/ruff)

<!-- SPHINX-START -->

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/John-Ragland/NESCAFE23_fluctuations/workflows/CI/badge.svg
[actions-link]:             https://github.com/John-Ragland/NESCAFE23_fluctuations/actions
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/NESCAFE23_fluctuations
[conda-link]:               https://github.com/conda-forge/NESCAFE23_fluctuations-feedstock
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/John-Ragland/NESCAFE23_fluctuations/discussions
[pypi-link]:                https://pypi.org/project/NESCAFE23_fluctuations/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/NESCAFE23_fluctuations
[pypi-version]:             https://img.shields.io/pypi/v/NESCAFE23_fluctuations
[rtd-badge]:                https://readthedocs.org/projects/NESCAFE23_fluctuations/badge/?version=latest
[rtd-link]:                 https://NESCAFE23_fluctuations.readthedocs.io/en/latest/?badge=latest
[coverage-badge]:           https://codecov.io/github/John-Ragland/NESCAFE23_fluctuations/branch/main/graph/badge.svg
[coverage-link]:            https://codecov.io/github/John-Ragland/NESCAFE23_fluctuations

<!-- prettier-ignore-end -->

Code supporting the data analysis of the NESCAFE 2023 Experiment. The NESCAFE
2023 experiment consisted of acoustic transmissions across the Gulf Stream
front. An analysis of the fluctuations of acoustic intensity, phase, and the
coherence of the acoustic receptions is described the paper _Observations of
Acoustic Fluctuations Across the Gulf Stream Front_, available in preprint
[here](wwww.placeholder.com).

This package contains the code used for the data analysis that is presented in
this paper, as well as generating the figures for publication.

## Installation / Setup

The package can be installed with pip via PyPI. The package is designed to run
on python 3.13.

```bash
pip install NESCAFE23_fluctuations
```

If you want to edit the code, or open the notebooks easily, you can clone the
repo from GitHub and install the package in editable mode

```bash
git clone https:/www.github.com/John-Ragland/NESCAFE23_fluctuations.git
cd NESCAFE23_fluctuations
pip install -e .[dev]
pre-commit install
```

Outputs of the notebooks are stripped via `nbstripout` to keep the repo small.
To enable this as a git filter, run the following command within the repo:

```bash
nbstripout --install
```

**Data** The raw data, intermediate data products, and final results are stored
or saved in a uniform directory structure. The location that this directory is
stored locally is specified in the .env file. To access the experiment data,
please reach out to me (John Ragland) directly. When you run the CLI, it will
ask you to specify the data directory path.

**Dependencies** most of the dependencies are automatically installed via the
pyproject.toml file. There are a few custom dependencies that are not on PyPI
and need to be installed manually

| Package | Source | url                                  | version |
| ------- | ------ | ------------------------------------ | ------- |
| cmps    | GitHub | https://github.com/John-Ragland/cmps | 0.1     |

## Reproducing the paper

Everything in the paper that comes from this package (Figs. 1, 2, 5–11 and
Table II) is made by one command:

```bash
n23_fluctuation_analysis --reproduce-paper --output paper_output
```

or interactively, with the first menu entry of `n23_fluctuation_analysis`
("reproduce paper"). It checks first that every input is present, then runs
the processing steps in order, writes Table II, builds every figure, and stops
at the first failure. Add `--reuse` to reuse data products that already exist
instead of recomputing them. Figs. 3, 4 and 12 (the moored internal-wave
observations) are not made by this package.

Outputs:

- `$NESCAFE23_DATADIR/dataproducts/`: the intermediate data products (arrival
  time offsets and their jackknife errors, mean pulse, peak picks, array tilt,
  hydrophone pattern, wavefront values, ray traces)
- `paper_output/table_II.tex`: Table II
- `paper_output/figures/`: the figures, named as in `reference/`
- `paper_output/reproduction_report.txt`: package version and commit, the
  analysis settings (`tools.PAPER_SETTINGS`), the time of each step, and the
  numbers the paper quotes from the data products

**Inputs.** `NESCAFE23_DATADIR` (in the `.env` of the workspace root) must
contain:

| Path                                       | Contents                                   |
| ------------------------------------------ | ------------------------------------------ |
| `dmod/dm*.mat`                             | the 23 demodulated receptions (on request) |
| `environment_data/FCTD_SoundSpeed_Colosi_v2.mat` | FCTD section                         |
| `environment_data/SION_CTD_new.mat`, `SIOE_CTD.mat` | mooring CTD casts                 |
| `environment_data/map_bathy.nc`            | GEBCO bathymetry for the map               |
| `environment_data/cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D_1773078040918.nc` | CMEMS sea surface height |
| `fonts/RedHatDisplay-Medium.ttf`           | map font (optional)                        |

The demodulated receptions are not public. They are available from the
corresponding author on reasonable request. Without LaTeX on the `PATH`, the
figures are built with `usetex: false`.

**Individual steps.** The other menu entries run one step at a time, in
pipeline order: arrival time offset → its jackknife error → mean pulse → peak
locations → array tilt → wavefront values and hydrophone pattern → rays →
Table II → publication figures. To build the figures with other settings, use
"create publication figures config" and "generate publication figures". The
config's defaults are the paper's.

**Exploratory analysis.** Analysis that is not in the paper (the GM + noise fit
of the spectra, the GM levels table, the transmission independence test, the
estimation experiments, the theory environments, the source geometry sweep) is
not in the CLI. Its code is in `tools`, `estimation` and `theory_environments`,
and the one-off studies are in `analysis/`.

**Source Localization** The depth and range of the acoustic source is estimated
with a linear Gauss-Markov approach, based on (Munk et. al. 1995). This method
is in the process of being published separately, so the code for this is omitted
from the initial public release of this package. If this code is of interest to
you, please feel free to reach out to me (John Ragland) directly.

**Source Geometry Sweep** The exact source location is unknown, because the
mooring was lost. For this paper, the source geometry is chosen based on the
best match of rays through the FCTD and the lineup of the ducted arrivals and
first boundary reflected arrivals. `tools.source_geometry_sweep` performs a grid
search over a range of source depths and ranges, whose result
(`ray_geo_sweep.pkl`) the source geometry sweep panel app shows. It is not
required for the analysis, but shows the context in which the source geometry
was chosen.

### Interactive Plots

Interactive dashboards and visualizations are available as Panel apps in the
`panel_apps/` directory. To serve Panel apps, use:

```bash
n23_panel
```

## Figures

The publication figures are made by `src/n23_fluctuations_ragland/figures.py`
(see above). The notebooks in `notebooks/publication_figures` are the original
figure source and may drift from the current code.

## Development

**TO DO**

- [x] add script for mixing together sound speeds, currently using
      'environment_data/dataproducts/full_field_new.nc'
- [ ] find bugs in source localization code. Currently using best guess of
      geometry below, which is the first transmission source localization

```python
source_depth = 2361.50872806
source_range = 0
launch_angles = np.linspace(-30, 30, 8000)
receiver_range = 150404.34909747
num_range_save = 1000
receiver_depths = [999.37979522]
```

- [ ] update calculation of rays with final environment / geometry, etc.
      Currently using 'NESCAFE23_analysis/dataproducts/ray_run_forward_back.pkl'
      /Users/john/Code/NESCAFE/nescafe_pilot_analysis/wave_statistics/ray_run_forward_back_new_stitch.pkl)
- [ ] update pygenray to reflect the latest version on pypi, so that everything
      works as expected.
  - right now I just have the local clone installating in the venv

- [ ] Fix $zeta_1$ and $zeta_2$ for +7b, currently the backward raytrace does
      not find the eigen ray

- [ ] currently using solutionsv1.mat for source localization. Evaluate if
      source localization code is needed.

## References

Munk, W., Worcester, P., and Wunsch, C. (1995). Ocean Acoustic Tomography,
Cambridge University Press, 457 pages.
