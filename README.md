# NESCAFE23_fluctuations

[![Actions Status][actions-badge]][actions-link]
[![Documentation Status][rtd-badge]][rtd-link]

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
[here](https://essopenarchive.org/doi/full/10.22541/essoar.15006016/v1).

This package contains the code used for the data analysis that is presented in
this paper, as well as generating the figures for publication.

## Installation / Setup

This package is a submodule to the parent package [NESCAFE23](https://github.com/John-Ragland/NESCAFE23)
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

## Data Analysis

The primary analysis is performed via the command line interface (CLI). Prior to
running the notebooks that generate the publication figures, the analysis must
be run. All steps can be run by selecting `run all`, or individual steps can be
selected within the CLI. To run the CLI type the following in the terminal

```bash
nescafe23-processing
```

### CLI Documentation

**Source Localization** The depth and range of the acoustic source is estimated
with a linear Gauss-Markov approach, based on (Munk et. al. 1995). This method
is in the process of being published separately, so the code for this is omitted
from the initial public release of this package. If this code is of interest to
you, please feel free to reach out to me (John Ragland) directly. The source
localization option in the CLI will cause an error if called without the
`src/NESCAFE23_fluctuations/source_localization.py` file.

**Source Geometry Sweep** The exact source location is unknown, because the
mooring was lost. For this paper, the source geometry is chosen based on the
best match of rays through the FCTD and the lineup of the ducted arrivals and
first boundary reflected arrivals. A more rigorous approach that makes uses a
Gauss-Markov linear inversion method is under development and will be published
separately. The source geometry sweep option in the CLI will perform a grid
search over a range of source depths and ranges. This must be computed prior to
running the source geometry sweep app. This option is not required for the full
analysis, but demostrates the context in which the source geometry was chosen
for the analysis of this paper.

## Figures

The figures for publication are generated with the cliJupyter Notebooks located
relative to the package structure at `notebooks/publication_figures`.


## References

Munk, W., Worcester, P., and Wunsch, C. (1995). Ocean Acoustic Tomography,
Cambridge University Press, 457 pages.
