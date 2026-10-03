"""Shared pytest configuration: headless Qt + non-interactive matplotlib, optional-engine skips."""

from __future__ import annotations

import importlib.util
import os

# Must be set before any Qt / matplotlib import happens in the test process.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

import pytest


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


HAS_NEURALFOIL = _has("neuralfoil") and _has("aerosandbox")
HAS_QT = _has("PySide6")


def pytest_collection_modifyitems(config, items):
    skip_nf = pytest.mark.skip(reason="NeuralFoil/AeroSandbox not installed")
    skip_qt = pytest.mark.skip(reason="PySide6 not installed")
    for item in items:
        if "neuralfoil" in item.keywords and not HAS_NEURALFOIL:
            item.add_marker(skip_nf)
        if "gui" in item.keywords and not HAS_QT:
            item.add_marker(skip_qt)
