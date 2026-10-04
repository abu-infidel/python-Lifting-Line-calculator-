"""Shared services of the application: airfoil resolver and library, section engines, and the translation of a
:class:`~liftingline.io.project.ProjectState` into an :class:`~liftingline.core.analysis.AnalysisRequest`.

GUI-free, so the command-line tools and the tests use exactly what the window uses.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np

from ..airfoil.engines import BUILTIN, NEURALFOIL, available_engines, default_engine_kind, make_engine
from ..airfoil.geometry import Airfoil
from ..airfoil.library import AirfoilLibrary
from ..airfoil.resolver import AirfoilNotFound, AirfoilResolver, Resolution
from ..airfoil.section_base import EngineSettings, SectionEngine
from ..airfoil.section_model import OverriddenAero, SectionOverride, WingSectionModel, reference_cd0
from ..airfoil.section_polarfile import PolarParseError, TabulatedPolarAero, load_polar
from ..core.analysis import AnalysisRequest, MomentReference
from ..core.controls import ControlSet
from ..core.flow import FlowCondition
from ..core.highlift import HighLiftSection
from ..core.wing import Wing
from .project import REF_ABSOLUTE, ProjectError, ProjectState, SolverSpec

_NONLINEAR = {"auto": None, "on": True, "off": False}

#: names (and display order) of the section factors shown to the user
FACTORS = ("a0", "alpha_L0_deg", "cm_c4", "cl_max", "cd0")


@dataclass(frozen=True)
class KeyStatus:
    """What an airfoil key resolves to, for live feedback in the airfoil table."""

    level: str  # "ok" | "warning" | "error"
    text: str
    resolution: Resolution | None = None


@dataclass(frozen=True)
class SectionInfo:
    """One wing section as the analysis sees it: airfoil, data source and the factors at its Reynolds number."""

    index: int
    y: float
    chord: float
    re: float
    key: str
    airfoil_name: str
    source: str
    note: str
    approximate: bool
    t_over_c: float
    max_camber: float
    x_camber: float
    data_label: str  # the section engine / polar the data came from
    engine_factors: dict[str, float]  # keys of FACTORS, before overrides
    factors: dict[str, float]  # after overrides
    overridden: tuple[str, ...]
    airfoil: Airfoil


def _override_from_dict(d: dict[str, float] | None) -> SectionOverride | None:
    if not d:
        return None
    ov = SectionOverride(
        a0=d.get("a0"),
        alpha_L0=math.radians(d["alpha_L0_deg"]) if d.get("alpha_L0_deg") is not None else None,
        cm_c4=d.get("cm_c4"),
        cl_max=d.get("cl_max"),
        cd0=d.get("cd0"),
    )
    return ov if ov.active else None


class Backend:
    """Airfoil resolver + library + section engines, shared by every analysis of a session."""

    def __init__(self, library: AirfoilLibrary | None = None) -> None:
        self.resolver = AirfoilResolver(library)
        self._engines: dict[tuple[str, EngineSettings], SectionEngine] = {}
        self._polars: dict[tuple[str, float], TabulatedPolarAero] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------------- library
    @property
    def library(self) -> AirfoilLibrary | None:
        return self.resolver.library

    def set_library(self, library: AirfoilLibrary | None) -> None:
        self.resolver.set_library(library)

    def scan_library(self, folder: str | Path, progress: Callable[[int, int], None] | None = None) -> AirfoilLibrary:
        """Index a folder of ``.dat`` files and make it the library (raises ``FileNotFoundError``)."""
        lib = AirfoilLibrary.scan(folder, progress=progress)
        self.set_library(lib)
        return lib

    # -------------------------------------------------------------------------- engines
    @staticmethod
    def engine_kind(solver: SolverSpec) -> tuple[str, str]:
        """(engine kind actually used, note) for the user's choice; NeuralFoil falls back to the built-in engine."""
        want = default_engine_kind() if solver.engine == "auto" else solver.engine
        if want == NEURALFOIL and not available_engines()[NEURALFOIL][0]:
            return BUILTIN, "NeuralFoil is not installed: the built-in section engine was used instead."
        return want, ""

    def engine(self, solver: SolverSpec) -> tuple[SectionEngine, str]:
        kind, note = self.engine_kind(solver)
        settings = EngineSettings(
            n_crit=solver.n_crit, xtr_upper=solver.xtr_upper, xtr_lower=solver.xtr_lower, model_size=solver.model_size
        )
        if kind == BUILTIN:  # the built-in engine has no boundary-layer settings: one instance serves all
            settings = EngineSettings()
        with self._lock:
            key = (kind, settings)
            if key not in self._engines:
                self._engines[key] = make_engine(kind, settings)
            return self._engines[key], note

    def _polar(self, path: str) -> TabulatedPolarAero:
        p = Path(path)
        try:
            stamp = p.stat().st_mtime
        except OSError as exc:
            raise ProjectError(f"cannot read the polar file {path}: {exc}") from exc
        with self._lock:
            key = (str(p), stamp)
            if key not in self._polars:
                try:
                    self._polars[key] = load_polar(p)
                except PolarParseError as exc:
                    raise ProjectError(f"polar file {p.name}: {exc}") from exc
            return self._polars[key]

    # ---------------------------------------------------------------------- airfoil keys
    def resolve(self, key: str) -> Resolution:
        return self.resolver.resolve(key)

    def describe_key(self, key: str) -> KeyStatus:
        """Live feedback for an airfoil key: what it resolves to, or why it does not."""
        if not key.strip():
            return KeyStatus("error", "enter a NACA code, a library name or a .dat file")
        try:
            res = self.resolver.resolve(key)
        except AirfoilNotFound as exc:
            return KeyStatus("error", str(exc))
        except Exception as exc:  # a malformed file must never take the window down
            return KeyStatus("error", f"cannot use {key!r}: {exc}")
        af = res.airfoil
        try:
            geo = f"t/c {100 * af.t_over_c:.1f} %, camber {100 * abs(af.max_camber):.1f} %"
        except Exception:
            geo = ""
        text = f"{af.name} - {res.source}" + (f" - {geo}" if geo else "")
        if res.approximate or res.note:
            return KeyStatus("warning", res.note or text, res)
        return KeyStatus("ok", text, res)

    # --------------------------------------------------------------------- section model
    def _make_aeros(self, resolutions: list[Resolution], state: ProjectState):
        """Section data for every airfoil, with a warm-up call so that a broken engine fails *here* (and the
        built-in engine takes over with a note) rather than half-way through an analysis or a sweep."""
        engine, note = self.engine(state.solver)
        probe = np.array([1.0e6])
        try:
            aeros = [engine.aero(r.airfoil) for r in resolutions]
            for a in aeros:
                a.linear(probe)
        except Exception as exc:
            kind, _ = self.engine_kind(state.solver)
            if kind == BUILTIN:
                raise
            fallback, _ = self.engine(replace(state.solver, engine="builtin"))
            aeros = [fallback.aero(r.airfoil) for r in resolutions]
            note = f"The section engine could not be used ({type(exc).__name__}: {exc}); the built-in engine was used instead."
            engine = fallback
        return engine, aeros, note

    def section_model(self, wing: Wing, state: ProjectState) -> WingSectionModel:
        keys = [s.airfoil for s in wing.sections]
        resolutions: list[Resolution] = []
        for i, key in enumerate(keys):
            try:
                resolutions.append(self.resolver.resolve(key))
            except AirfoilNotFound as exc:
                raise ProjectError(f"Airfoil of section {i + 1}: {exc}") from exc
        engine, aeros, engine_note = self._make_aeros(resolutions, state)
        labels = [engine.name]
        for i, path in enumerate(state.airfoils.polar_files[: len(aeros)]):
            if path:
                aeros[i] = self._polar(path)
                labels.append("imported polar")
        overrides = [_override_from_dict(d) for d in state.airfoils.overrides[: len(aeros)]]
        overrides += [None] * (len(aeros) - len(overrides))
        notes = list(dict.fromkeys(r.note for r in resolutions if r.note))
        if engine_note:
            notes.append(engine_note)
        return WingSectionModel(
            wing, aeros, overrides=overrides, label=", ".join(dict.fromkeys(labels)), notes=notes, resolutions=resolutions
        )

    def build_request(self, state: ProjectState) -> AnalysisRequest:
        """The analysis request for a project; raises :class:`ProjectError` with a readable message."""
        problems = state.validate()
        if problems:
            raise ProjectError("\n".join(problems))
        wing, flow = state.build_wing(), state.build_flow()
        model = self.section_model(wing, state)
        high_lift = state.high_lift.system()
        section = HighLiftSection(model, high_lift, wing.half_span) if high_lift is not None else model
        f, s = state.flow, state.solver
        absolute = f.ref_mode == REF_ABSOLUTE
        controls = ControlSet.from_dicts(state.controls, wing.half_span) if state.controls else None
        return AnalysisRequest(
            wing=wing,
            flow=flow,
            section=section,
            solver=s.solver,
            n_terms=s.n_terms,
            nonlinear=_NONLINEAR[s.nonlinear],
            compressibility=f.compressibility,
            ref=MomentReference(
                x_frac_mac=f.ref_x_frac_mac,
                x_abs=f.ref_x_abs_m if absolute else None,
                z_abs=f.ref_z_abs_m if absolute else 0.0,
            ),
            extra_cd0=f.extra_cd0,
            convergence_check=s.convergence_check,
            ground_height=f.ground_height_m if f.ground_effect else None,
            n_panels=s.n_panels,
            controls=controls,
        )

    # ---------------------------------------------------------------------- description
    @staticmethod
    def describe_sections(model: WingSectionModel | HighLiftSection, wing: Wing, flow: FlowCondition) -> list[SectionInfo]:
        """Per-section factors at each section's own Reynolds number (engine values and effective values).

        These are the clean sections: high-lift devices are described by their own panel."""
        if isinstance(model, HighLiftSection):
            model = model.base
        out: list[SectionInfo] = []
        for i, (sec, aero, res) in enumerate(zip(wing.sections, model.aeros, model.resolutions)):
            chord = max(sec.chord, 0.1 * wing.root_chord)  # an elliptic tip has no chord: describe it at 10 % of the root
            re = np.array([float(flow.reynolds(chord))])
            base = aero.base if isinstance(aero, OverriddenAero) else aero

            def factors(a, cd0):
                lin = a.linear(re)
                return dict(
                    a0=float(lin.a0[0]),
                    alpha_L0_deg=math.degrees(float(lin.alpha_L0[0])),
                    cm_c4=float(lin.cm_c4[0]),
                    cl_max=float(lin.cl_max[0]),
                    cd0=float(cd0),
                )

            cd0_base = float(reference_cd0(base, re)[0])
            engine_factors = factors(base, cd0_base)
            eff_cd0 = aero.ov.cd0 if isinstance(aero, OverriddenAero) and aero.ov.cd0 is not None else cd0_base
            out.append(
                SectionInfo(
                    index=i,
                    y=sec.y,
                    chord=sec.chord,
                    re=float(re[0]),
                    key=sec.airfoil,
                    airfoil_name=res.airfoil.name,
                    source=res.source,
                    note=res.note,
                    approximate=res.approximate,
                    t_over_c=res.airfoil.t_over_c,
                    max_camber=res.airfoil.max_camber,
                    x_camber=res.airfoil.x_camber,
                    data_label=getattr(base, "label", ""),
                    engine_factors=engine_factors,
                    factors=factors(aero, eff_cd0),
                    overridden=_override_names(aero),
                    airfoil=res.airfoil,
                )
            )
        return out


def _override_names(aero) -> tuple[str, ...]:
    if not isinstance(aero, OverriddenAero):
        return ()
    ov = aero.ov
    names = []
    if ov.a0 is not None:
        names.append("a0")
    if ov.alpha_L0 is not None:
        names.append("alpha_L0_deg")
    if ov.cm_c4 is not None:
        names.append("cm_c4")
    if ov.cl_max is not None:
        names.append("cl_max")
    if ov.cd0 is not None:
        names.append("cd0")
    return tuple(names)


__all__ = ["Backend", "KeyStatus", "SectionInfo", "FACTORS"]
