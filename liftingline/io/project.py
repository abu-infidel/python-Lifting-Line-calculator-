"""Project state: everything the user can set, in SI units, JSON-serialisable and GUI-free.

The GUI edits a :class:`ProjectState`; this module turns it into the solver inputs
(:func:`wing_from_state`, :func:`flow_from_state`) and validates it with messages meant for people.
A project file is the ``to_dict()`` of the state plus a ``schema_version``.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from ..core.controls import validate_rows as validate_control_rows
from ..core.flow import FlowCondition
from ..core.wing import KIND_ELLIPTIC, Section, Wing

SCHEMA_VERSION = 1

# ----------------------------------------------------------------------------- planform modes
MODE_TRAPEZOID = "trapezoid"  # one straight-tapered panel (root and tip)
MODE_ELLIPTIC = "elliptic"  # exact elliptic chord distribution
MODE_SECTIONS = "sections"  # free table of sections

DEFINE_SPAN_CHORDS = "span_chords"  # span, root chord, tip chord
DEFINE_AREA_AR_TAPER = "area_ar_taper"  # area, aspect ratio, taper ratio
DEFINE_SPAN_AREA_TAPER = "span_area_taper"  # span, area, taper ratio
DEFINE_SPAN_AR_TAPER = "span_ar_taper"  # span, aspect ratio, taper ratio
DEFINE_MODES = (DEFINE_SPAN_CHORDS, DEFINE_AREA_AR_TAPER, DEFINE_SPAN_AREA_TAPER, DEFINE_SPAN_AR_TAPER)

ATMOSPHERE_ISA = "isa"
ATMOSPHERE_MANUAL = "manual"
SPEED_BY_VELOCITY = "speed"
SPEED_BY_MACH = "mach"
REF_MAC = "mac"
REF_ABSOLUTE = "absolute"

SOLVER_CHOICES = ("auto", "classical", "numerical")
ENGINE_CHOICES = ("auto", "neuralfoil", "builtin")
NONLINEAR_CHOICES = ("auto", "on", "off")
MODEL_SIZES = ("xxsmall", "xsmall", "small", "medium", "large", "xlarge", "xxlarge", "xxxlarge")

#: factors a section's airfoil can have overridden (keys of ``AirfoilSpec.overrides`` entries)
OVERRIDE_KEYS = ("a0", "alpha_L0_deg", "cm_c4", "cl_max", "cd0")


class ProjectError(ValueError):
    """The project contains something that cannot be analysed; the message is meant for the user."""


# ===================================================================================== wing
@dataclass
class WingSpec:
    """The planform, in SI units (m, degrees)."""

    mode: str = MODE_TRAPEZOID
    define_by: str = DEFINE_SPAN_CHORDS
    span: float = 11.0
    root_chord: float = 1.63
    tip_chord: float = 1.13
    area: float = 0.0  # inputs only when ``define_by`` selects them; always kept consistent by ``synced``
    aspect_ratio: float = 0.0
    taper_ratio: float = 0.0
    sweep_deg: float = 0.0
    sweep_at: float = 0.25  # chord fraction along which ``sweep_deg`` is measured (0 = LE, 0.25 = c/4, 1 = TE)
    dihedral_deg: float = 0.0
    twist_tip_deg: float = 0.0  # tip twist relative to the root, nose-up positive (washout is negative)
    incidence_deg: float = 0.0  # root incidence added to the flow angle of attack
    sections: list[dict[str, float]] = field(default_factory=list)  # table mode rows: y, chord, x_le, z, twist_deg
    name: str = "Wing"

    def __post_init__(self) -> None:
        if self.area <= 0.0 or self.aspect_ratio <= 0.0 or self.taper_ratio <= 0.0:
            self.sync()

    # --------------------------------------------------------------- derived dimensions
    def dimensions(self) -> tuple[float, float, float]:
        """(span, root chord, tip chord) in metres from whichever three inputs ``define_by`` selects."""
        d = self.define_by
        elliptic = self.mode == MODE_ELLIPTIC
        if d not in DEFINE_MODES:
            raise ProjectError(f"unknown way of defining the planform: {d!r}")
        if elliptic:
            if d == DEFINE_AREA_AR_TAPER:
                _need_positive(area=self.area, aspect_ratio=self.aspect_ratio)
                b = math.sqrt(self.aspect_ratio * self.area)
                return b, 4.0 * self.area / (math.pi * b), 0.0
            _need_positive(span=self.span, root_chord=self.root_chord)
            return self.span, self.root_chord, 0.0
        if d == DEFINE_SPAN_CHORDS:
            _need_positive(span=self.span, root_chord=self.root_chord)
            if self.tip_chord < 0.0:
                raise ProjectError("the tip chord cannot be negative")
            return self.span, self.root_chord, self.tip_chord
        if not (self.taper_ratio >= 0.0) or not math.isfinite(self.taper_ratio):
            raise ProjectError("the taper ratio cannot be negative")
        lam = self.taper_ratio
        if d == DEFINE_AREA_AR_TAPER:
            _need_positive(area=self.area, aspect_ratio=self.aspect_ratio)
            b = math.sqrt(self.aspect_ratio * self.area)
            s = self.area
        elif d == DEFINE_SPAN_AREA_TAPER:
            _need_positive(span=self.span, area=self.area)
            b, s = self.span, self.area
        else:  # DEFINE_SPAN_AR_TAPER
            _need_positive(span=self.span, aspect_ratio=self.aspect_ratio)
            b = self.span
            s = b * b / self.aspect_ratio
        root = 2.0 * s / (b * (1.0 + lam))
        return b, root, lam * root

    def sync(self) -> "WingSpec":
        """Recompute the six redundant planform numbers from the selected inputs (in place); returns self."""
        if self.mode == MODE_SECTIONS:
            return self
        try:
            b, c_root, c_tip = self.dimensions()
        except ProjectError:
            return self
        self.span, self.root_chord, self.tip_chord = b, c_root, c_tip
        if self.mode == MODE_ELLIPTIC:
            self.area = math.pi * b * c_root / 4.0
            self.taper_ratio = 0.0
        else:
            self.area = 0.5 * b * (c_root + c_tip)
            self.taper_ratio = c_tip / c_root if c_root > 0 else 0.0
        self.aspect_ratio = b * b / self.area if self.area > 0 else 0.0
        return self

    # ---------------------------------------------------------------------- sections
    def n_sections(self) -> int:
        return len(self.sections) if self.mode == MODE_SECTIONS else 2

    def to_table(self, airfoil_root: str = "", airfoil_tip: str | None = None) -> list[dict[str, float]]:
        """The planform as table rows (used when the user switches to the free section table)."""
        if self.mode == MODE_SECTIONS:
            return copy.deepcopy(self.sections)
        wing = wing_from_spec(self, [airfoil_root, airfoil_root if airfoil_tip is None else airfoil_tip])
        if wing.kind == KIND_ELLIPTIC:  # sample the ellipse with cosine-spaced stations
            n = 12
            ys = [wing.half_span * math.sin(0.5 * math.pi * k / n) for k in range(n)]
            ys.append(wing.half_span)
            rows = []
            for k, y in enumerate(ys):
                c = float(wing.chord(y))
                if k == n:
                    c = 0.0
                rows.append(dict(y=y, chord=c, x_le=float(wing.x_le(y)), z=0.0, twist_deg=wing.sections[-1].twist_deg * y / wing.half_span))
            return rows
        return [dict(y=s.y, chord=s.chord, x_le=s.x_le, z=s.z, twist_deg=s.twist_deg) for s in wing.sections]


def _need_positive(**values: float) -> None:
    labels = {
        "span": "span",
        "root_chord": "root chord",
        "tip_chord": "tip chord",
        "area": "wing area",
        "aspect_ratio": "aspect ratio",
        "taper_ratio": "taper ratio",
    }
    for key, value in values.items():
        if not (value > 0.0) or not math.isfinite(value):
            raise ProjectError(f"the {labels.get(key, key)} must be a positive number")


def wing_from_spec(spec: WingSpec, airfoils: list[str]) -> Wing:
    """Build the :class:`Wing` for a planform spec and one airfoil key per section."""
    n = spec.n_sections()
    keys = list(airfoils)
    if len(keys) < n:
        keys += [keys[-1] if keys else ""] * (n - len(keys))
    keys = keys[:n]
    try:
        if spec.mode == MODE_SECTIONS:
            if n < 2:
                raise ProjectError("a wing needs at least a root and a tip section")
            sections = tuple(
                Section(
                    y=float(r["y"]),
                    chord=float(r["chord"]),
                    x_le=float(r.get("x_le", 0.0)),
                    z=float(r.get("z", 0.0)),
                    twist_deg=float(r.get("twist_deg", 0.0)),
                    airfoil=k,
                )
                for r, k in zip(spec.sections, keys)
            )
            return Wing(sections=sections, incidence_deg=spec.incidence_deg, name=spec.name)
        b, c_root, c_tip = spec.dimensions()
        if spec.mode == MODE_ELLIPTIC:
            return Wing.elliptic(
                b, c_root, twist_tip_deg=spec.twist_tip_deg, incidence_deg=spec.incidence_deg,
                airfoil_root=keys[0], airfoil_tip=keys[1], name=spec.name,
            )
        if spec.mode != MODE_TRAPEZOID:
            raise ProjectError(f"unknown planform type {spec.mode!r}")
        if abs(spec.sweep_deg) >= 85.0 or abs(spec.dihedral_deg) >= 85.0:
            raise ProjectError("sweep and dihedral must be well below 90 degrees")
        return Wing.trapezoid(
            b, c_root, c_tip,
            sweep_deg=spec.sweep_deg, sweep_at=spec.sweep_at, dihedral_deg=spec.dihedral_deg,
            twist_tip_deg=spec.twist_tip_deg, incidence_deg=spec.incidence_deg,
            airfoil_root=keys[0], airfoil_tip=keys[1], name=spec.name,
        )
    except (KeyError, TypeError) as exc:
        raise ProjectError(f"incomplete section table (missing {exc})") from exc
    except ValueError as exc:
        if isinstance(exc, ProjectError):
            raise
        raise ProjectError(str(exc)) from exc


# ================================================================================= airfoils
@dataclass
class AirfoilSpec:
    """One airfoil key per wing section (a NACA code, a library name or a ``.dat`` path), plus overrides."""

    keys: list[str] = field(default_factory=lambda: ["NACA 2412", "NACA 2412"])
    overrides: list[dict[str, float]] = field(default_factory=lambda: [{}, {}])  # subset of OVERRIDE_KEYS per section
    polar_files: list[str] = field(default_factory=lambda: ["", ""])  # imported polar per section ("" = none)

    def fit(self, n: int) -> "AirfoilSpec":
        """Pad (repeating the last entry) or truncate every list to ``n`` entries, in place."""
        for name, filler in (("keys", None), ("overrides", {}), ("polar_files", "")):
            lst = getattr(self, name)
            if len(lst) < n:
                tail = lst[-1] if (lst and filler is None) else filler
                lst.extend([copy.deepcopy(tail) if tail is not None else ""] * (n - len(lst)))
            del lst[n:]
        return self


# ===================================================================================== flow
@dataclass
class FlowSpec:
    atmosphere: str = ATMOSPHERE_ISA
    altitude_m: float = 0.0
    delta_T: float = 0.0  # ISA temperature offset, K
    rho: float = 1.225  # manual atmosphere
    T: float = 288.15
    mu: float | None = None  # None: Sutherland's law at T
    speed_mode: str = SPEED_BY_VELOCITY
    V: float = 55.0  # m/s
    mach: float = 0.16
    alpha_deg: float = 3.0
    beta_deg: float = 0.0
    ground_effect: bool = False
    ground_height_m: float = 2.0  # c/4 height above the ground plane
    extra_cd0: float = 0.0  # parasite drag coefficient of the rest of the aircraft, on the wing area
    compressibility: bool = True
    ref_mode: str = REF_MAC
    ref_x_frac_mac: float = 0.25
    ref_x_abs_m: float = 0.0
    ref_z_abs_m: float = 0.0


def flow_from_spec(spec: FlowSpec) -> FlowCondition:
    try:
        speed = dict(V=spec.V) if spec.speed_mode == SPEED_BY_VELOCITY else dict(mach=spec.mach)
        if spec.atmosphere == ATMOSPHERE_ISA:
            return FlowCondition.from_isa(
                spec.altitude_m, alpha_deg=spec.alpha_deg, beta_deg=spec.beta_deg, delta_T=spec.delta_T, **speed
            )
        if spec.atmosphere != ATMOSPHERE_MANUAL:
            raise ProjectError(f"unknown atmosphere type {spec.atmosphere!r}")
        return FlowCondition.from_manual(
            rho=spec.rho, T=spec.T, mu=spec.mu, alpha_deg=spec.alpha_deg, beta_deg=spec.beta_deg, **speed
        )
    except ProjectError:
        raise
    except ValueError as exc:
        raise ProjectError(str(exc)) from exc


# =================================================================================== solver
@dataclass
class SolverSpec:
    solver: str = "auto"
    n_terms: int = 80
    n_panels: int = 80
    nonlinear: str = "auto"
    convergence_check: bool = True
    engine: str = "auto"
    model_size: str = "xlarge"
    n_crit: float = 9.0
    xtr_upper: float = 1.0
    xtr_lower: float = 1.0


# ===================================================================================== project
@dataclass
class ProjectState:
    wing: WingSpec = field(default_factory=WingSpec)
    airfoils: AirfoilSpec = field(default_factory=AirfoilSpec)
    flow: FlowSpec = field(default_factory=FlowSpec)
    solver: SolverSpec = field(default_factory=SolverSpec)
    name: str = "Untitled"
    notes: str = ""
    controls: list[dict[str, Any]] = field(default_factory=list)  # control surfaces (see core.controls.ControlSurface)

    def __post_init__(self) -> None:
        self.airfoils.fit(self.wing.n_sections())

    def copy(self) -> "ProjectState":
        return copy.deepcopy(self)

    # -------------------------------------------------------------------- serialisation
    def to_dict(self) -> dict[str, Any]:
        def dump(obj):
            return {f.name: copy.deepcopy(getattr(obj, f.name)) for f in fields(obj)}

        return dict(
            schema_version=SCHEMA_VERSION,
            name=self.name,
            notes=self.notes,
            wing=dump(self.wing),
            airfoils=dump(self.airfoils),
            flow=dump(self.flow),
            solver=dump(self.solver),
            controls=copy.deepcopy(self.controls),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectState":
        if not isinstance(data, dict):
            raise ProjectError("not a project file")
        version = data.get("schema_version", 1)
        if not isinstance(version, int) or version > SCHEMA_VERSION:
            raise ProjectError(
                f"this project was written by a newer version of the program (schema {version}, this one reads "
                f"up to {SCHEMA_VERSION})"
            )

        def load(klass, raw):
            known = {f.name for f in fields(klass)}
            return klass(**{k: v for k, v in (raw or {}).items() if k in known})

        wing = load(WingSpec, data.get("wing"))
        state = cls(
            wing=wing,
            airfoils=load(AirfoilSpec, data.get("airfoils")),
            flow=load(FlowSpec, data.get("flow")),
            solver=load(SolverSpec, data.get("solver")),
            name=str(data.get("name", "Untitled")),
            notes=str(data.get("notes", "")),
            controls=[dict(c) for c in (data.get("controls") or []) if isinstance(c, dict)],
        )
        state.wing.sync()
        state.airfoils.fit(state.wing.n_sections())
        return state

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ProjectState":
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProjectError(f"cannot read project file {path}: {exc}") from exc
        return cls.from_dict(data)

    # ------------------------------------------------------------------------- builders
    def build_wing(self) -> Wing:
        self.airfoils.fit(self.wing.n_sections())
        return wing_from_spec(self.wing, self.airfoils.keys)

    def build_flow(self) -> FlowCondition:
        return flow_from_spec(self.flow)

    def validate(self) -> list[str]:
        """Everything that would stop an analysis, as readable messages (empty = good to run)."""
        problems: list[str] = []
        try:
            self.build_wing()
        except ProjectError as exc:
            problems.append(f"Wing: {exc}")
        flow = None
        try:
            flow = self.build_flow()
        except ProjectError as exc:
            problems.append(f"Flow: {exc}")
        s, f = self.solver, self.flow
        if s.n_terms < 8 or s.n_terms > 800:
            problems.append("Solver: the number of Fourier terms must be between 8 and 800")
        if s.n_panels < 8 or s.n_panels > 800:
            problems.append("Solver: the number of elements per half-wing must be between 8 and 800")
        if not 0.0 < s.n_crit <= 20.0:
            problems.append("Solver: n_crit must be between 0 and 20")
        if f.ground_effect and not f.ground_height_m > 0.0:
            problems.append("Flow: the height above the ground must be positive")
        if flow is not None and flow.mach >= 1.0:
            problems.append(f"Flow: Mach {flow.mach:.2f} - lifting-line theory needs subsonic flow")
        if abs(f.alpha_deg) > 40.0 or abs(f.beta_deg) > 40.0:
            problems.append("Flow: angles above 40 degrees are far outside the range of lifting-line theory")
        if f.ref_mode == REF_ABSOLUTE and not math.isfinite(f.ref_x_abs_m):
            problems.append("Flow: the moment reference point must be a number")
        problems += [f"Controls: {m}" for m in validate_control_rows(self.controls)]
        return problems
