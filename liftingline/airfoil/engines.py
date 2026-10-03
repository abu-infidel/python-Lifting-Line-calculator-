"""Section-engine registry: which engines exist, which are usable here, and a factory."""

from __future__ import annotations

from .section_base import EngineSettings, EngineUnavailable, SectionEngine
from .section_builtin import BuiltinEngine
from .section_neuralfoil import NeuralFoilEngine, neuralfoil_available

NEURALFOIL = "neuralfoil"
BUILTIN = "builtin"

LABELS = {
    NEURALFOIL: "Max fidelity - NeuralFoil",
    BUILTIN: "Fast - built-in",
}


def available_engines() -> dict[str, tuple[bool, str]]:
    """``{kind: (usable, note)}`` for every engine."""
    nf_ok = neuralfoil_available()
    return {
        NEURALFOIL: (nf_ok, "" if nf_ok else "not installed: pip install neuralfoil aerosandbox"),
        BUILTIN: (True, ""),
    }


def default_engine_kind() -> str:
    """NeuralFoil when installed (maximum fidelity), otherwise the built-in engine."""
    return NEURALFOIL if neuralfoil_available() else BUILTIN


def make_engine(kind: str, settings: EngineSettings | None = None) -> SectionEngine:
    if kind == NEURALFOIL:
        return NeuralFoilEngine(settings)
    if kind == BUILTIN:
        return BuiltinEngine(settings)
    raise ValueError(f"unknown section engine {kind!r}; choose one of {sorted(LABELS)}")


__all__ = ["BUILTIN", "NEURALFOIL", "LABELS", "EngineSettings", "EngineUnavailable", "available_engines", "default_engine_kind", "make_engine"]
