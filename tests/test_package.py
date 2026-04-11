from __future__ import annotations

import importlib.metadata

import n23_fluctuations_ragland as m


def test_version():
    assert importlib.metadata.version("n23_fluctuations_ragland") == m.__version__
