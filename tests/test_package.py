from __future__ import annotations

import importlib.metadata

import nescafe23_fluctuations as m


def test_version():
    assert importlib.metadata.version("nescafe23_fluctuations") == m.__version__
