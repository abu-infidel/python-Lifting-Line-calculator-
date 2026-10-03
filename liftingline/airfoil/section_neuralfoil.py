"""Max-fidelity section engine: NeuralFoil (XFOIL-quality polars from a neural surrogate).

NeuralFoil is an optional dependency (``pip install neuralfoil``, listed in requirements.txt). It
works offline once installed. For each airfoil, polar tables over the angle-of-attack window
are built lazily at nodes of the global Reynolds grid (see :mod:`section_base`); the network
reports an ``analysis_confidence`` that the analysis pipeline surfaces as a warning when low.
"""

from __future__ import annotations

import importlib.util

import numpy as np

from .geometry import Airfoil
from .section_base import (
    AirfoilAero,
    EngineSettings,
    EngineUnavailable,
    PolarTable,
    TabulatedAirfoilAero,
    alpha_grid_deg,
)


def neuralfoil_available() -> bool:
    return importlib.util.find_spec("neuralfoil") is not None and importlib.util.find_spec("aerosandbox") is not None


class NeuralFoilAero(TabulatedAirfoilAero):
    def __init__(self, airfoil: Airfoil, settings: EngineSettings) -> None:
        super().__init__()
        try:
            import neuralfoil as nf  # imported lazily: it pulls in AeroSandbox
        except ImportError as exc:  # pragma: no cover - guarded by neuralfoil_available()
            raise EngineUnavailable("NeuralFoil is not installed (pip install neuralfoil)") from exc
        self._nf = nf
        self._settings = settings
        self._coords = airfoil.resample(80).coords
        self.label = f"NeuralFoil[{settings.model_size}] {airfoil.name}"
        self.approximate = airfoil.approximate

    def _build_table(self, re_node: float) -> PolarTable:
        s = self._settings
        alpha = alpha_grid_deg()
        r = self._nf.get_aero_from_coordinates(
            coordinates=self._coords,
            alpha=alpha,
            Re=re_node,
            n_crit=s.n_crit,
            xtr_upper=s.xtr_upper,
            xtr_lower=s.xtr_lower,
            model_size=s.model_size,
        )
        as_array = lambda key: np.asarray(r[key], dtype=float)  # noqa: E731
        return PolarTable(
            alpha_deg=alpha,
            cl=as_array("CL"),
            cd=as_array("CD"),
            cm=as_array("CM"),
            conf=as_array("analysis_confidence"),
        )


class NeuralFoilEngine:
    """Factory for :class:`NeuralFoilAero` objects (cached per airfoil geometry and settings)."""

    name = "NeuralFoil (max fidelity)"

    def __init__(self, settings: EngineSettings | None = None) -> None:
        if not neuralfoil_available():
            raise EngineUnavailable("NeuralFoil is not installed (pip install neuralfoil)")
        self.settings = settings or EngineSettings()
        self._cache: dict[str, NeuralFoilAero] = {}

    def aero(self, airfoil: Airfoil) -> AirfoilAero:
        aero = self._cache.get(airfoil.digest)
        if aero is None:
            aero = self._cache[airfoil.digest] = NeuralFoilAero(airfoil, self.settings)
        return aero
