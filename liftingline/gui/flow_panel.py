"""The Flow tab: atmosphere, speed, angles, ground effect, extra drag and the moment reference point."""

from __future__ import annotations

import copy

from PySide6.QtCore import QSignalBlocker, Signal
from PySide6.QtWidgets import QCheckBox, QFormLayout, QGroupBox, QLabel, QVBoxLayout, QWidget

from ..core import units
from ..io.project import (
    ATMOSPHERE_ISA,
    ATMOSPHERE_MANUAL,
    REF_ABSOLUTE,
    REF_MAC,
    SPEED_BY_MACH,
    SPEED_BY_VELOCITY,
    FlowSpec,
    ProjectError,
    ProjectState,
    flow_from_spec,
)
from .widgets import GuardedCombo, SpeedField, UnitSpin, muted, set_role


class FlowPanel(QWidget):
    changed = Signal()

    def __init__(self, system: str = "SI", parent=None) -> None:
        super().__init__(parent)
        self._system = system
        self._spec = FlowSpec()
        self._busy = False
        self._build()
        self.load_spec(self._spec)

    # ======================================================================== layout
    def _spin(self, kind, **kw) -> UnitSpin:
        s = UnitSpin(kind, self._system, **kw)
        s.valueChanged.connect(self._edited)
        return s

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        # ---- atmosphere
        box = QGroupBox("Air")
        self.air_form = f = QFormLayout(box)
        self.atmosphere = GuardedCombo()
        self.atmosphere.addItem("Standard atmosphere (ISA)", ATMOSPHERE_ISA)
        self.atmosphere.addItem("Manual air properties", ATMOSPHERE_MANUAL)
        self.atmosphere.currentIndexChanged.connect(self._atmosphere_changed)
        f.addRow("Atmosphere", self.atmosphere)
        self.altitude = self._spin("length", decimals=1, minimum=-2000.0, maximum=47000.0, step=100.0, tooltip="Geopotential altitude, -2 km ... 47 km")
        self.delta_t = self._spin(None, unit="K", decimals=1, minimum=-60.0, maximum=60.0, step=1.0, tooltip="Temperature offset from the standard day (ISA + x)")
        self.rho = self._spin("density", decimals=5, minimum=1e-4, maximum=100.0, step=0.01, tooltip="Air density")
        self.temp = self._spin("temperature", decimals=1, minimum=150.0, maximum=500.0, step=1.0, tooltip="Static air temperature (sets the speed of sound and, unless given, the viscosity)")
        self.use_mu = QCheckBox("viscosity")
        self.use_mu.setToolTip("Tick to enter the dynamic viscosity; otherwise Sutherland's law at the temperature is used")
        self.use_mu.toggled.connect(self._edited)
        self.mu = self._spin("viscosity", decimals=8, minimum=1e-7, maximum=1e-2, step=1e-6, tooltip="Dynamic viscosity")
        f.addRow("Altitude", self.altitude)
        f.addRow("Temperature offset", self.delta_t)
        f.addRow("Density", self.rho)
        f.addRow("Temperature", self.temp)
        f.addRow(self.use_mu, self.mu)
        self.air_readout = muted()
        f.addRow(self.air_readout)
        root.addWidget(box)

        # ---- speed
        box = QGroupBox("Speed")
        f = QFormLayout(box)
        self.speed_mode = GuardedCombo()
        self.speed_mode.addItem("Airspeed", SPEED_BY_VELOCITY)
        self.speed_mode.addItem("Mach number", SPEED_BY_MACH)
        self.speed_mode.currentIndexChanged.connect(self._speed_mode_changed)
        f.addRow("Given as", self.speed_mode)
        self.speed = SpeedField(self._system)
        self.speed.valueChanged.connect(self._edited)
        self.speed_row = f.rowCount()
        f.addRow("True airspeed", self.speed)
        self.mach = self._spin(None, unit="", decimals=4, minimum=0.001, maximum=0.99, step=0.01, tooltip="Freestream Mach number (subsonic only)")
        f.addRow("Mach number", self.mach)
        self.speed_form = f
        self.speed_readout = muted()
        f.addRow(self.speed_readout)
        root.addWidget(box)

        # ---- attitude
        box = QGroupBox("Attitude")
        f = QFormLayout(box)
        self.alpha = self._spin(None, unit="deg", decimals=3, minimum=-40.0, maximum=40.0, step=0.5, tooltip="Angle of attack of the root chord line (root incidence is added)")
        self.beta = self._spin(None, unit="deg", decimals=3, minimum=-40.0, maximum=40.0, step=0.5, tooltip="Sideslip angle, positive when the wind comes from the right (numerical solver)")
        f.addRow("Angle of attack", self.alpha)
        f.addRow("Sideslip angle", self.beta)
        root.addWidget(box)

        # ---- environment and extras
        box = QGroupBox("Ground effect and extra drag")
        f = QFormLayout(box)
        self.ground = QCheckBox("wing near the ground")
        self.ground.setToolTip("Image vortices below a ground plane parallel to the flow (numerical solver)")
        self.ground.toggled.connect(self._ground_toggled)
        f.addRow(self.ground)
        self.ground_h = self._spin("length", decimals=3, minimum=0.01, maximum=1e4, step=0.1, tooltip="Height of the wing's quarter-chord line above the ground")
        f.addRow("Height above ground", self.ground_h)
        self.cd0_extra = self._spin(None, unit="", decimals=5, minimum=0.0, maximum=1.0, step=0.001, tooltip="Parasite drag of the rest of the aircraft (fuselage, tail, gear) as a coefficient on the wing area")
        f.addRow("Extra drag CD0", self.cd0_extra)
        self.compress = QCheckBox("Prandtl-Glauert compressibility")
        self.compress.setToolTip("Scale the section lift slope and moment by 1/sqrt(1 - M^2); recommended above Mach 0.3")
        self.compress.toggled.connect(self._edited)
        f.addRow(self.compress)
        root.addWidget(box)

        # ---- moment reference
        box = QGroupBox("Moment reference point")
        f = QFormLayout(box)
        self.ref_mode = GuardedCombo()
        self.ref_mode.addItem("Fraction of the mean chord", REF_MAC)
        self.ref_mode.addItem("Absolute position", REF_ABSOLUTE)
        self.ref_mode.currentIndexChanged.connect(self._ref_mode_changed)
        f.addRow("Given as", self.ref_mode)
        self.ref_frac = self._spin(None, unit="% MAC", decimals=2, minimum=-200.0, maximum=300.0, step=1.0, tooltip="Distance behind the leading edge of the mean aerodynamic chord (25 % = aerodynamic centre of a thin wing)")
        self.ref_x = self._spin("length", decimals=4, minimum=-1e3, maximum=1e3, step=0.05, tooltip="Distance behind the leading edge of the root chord")
        self.ref_z = self._spin("length", decimals=4, minimum=-1e3, maximum=1e3, step=0.05, tooltip="Height above the root quarter-chord point")
        f.addRow("Chordwise position", self.ref_frac)
        f.addRow("x behind root LE", self.ref_x)
        f.addRow("z above c/4", self.ref_z)
        self.ref_form = f
        root.addWidget(box)

        self.problem = QLabel()
        self.problem.setWordWrap(True)
        set_role(self.problem, "error")
        root.addWidget(self.problem)
        root.addStretch(1)

    # ========================================================================== state
    def load(self, state: ProjectState) -> None:
        self.load_spec(state.flow)

    def load_spec(self, spec: FlowSpec) -> None:
        self._spec = copy.deepcopy(spec)
        s = self._spec
        self._busy = True
        try:
            self.atmosphere.set_data_silently(s.atmosphere)
            self.altitude.set_si(s.altitude_m)
            self.delta_t.set_si(s.delta_T)
            self.rho.set_si(s.rho)
            self.temp.set_si(s.T)
            with QSignalBlocker(self.use_mu):
                self.use_mu.setChecked(s.mu is not None)
            self.mu.set_si(s.mu if s.mu is not None else 1.789e-5)
            self.speed_mode.set_data_silently(s.speed_mode)
            self.speed.set_si(s.V)
            self.mach.set_si(s.mach)
            self.alpha.set_si(s.alpha_deg)
            self.beta.set_si(s.beta_deg)
            with QSignalBlocker(self.ground):
                self.ground.setChecked(s.ground_effect)
            self.ground_h.set_si(s.ground_height_m)
            self.cd0_extra.set_si(s.extra_cd0)
            with QSignalBlocker(self.compress):
                self.compress.setChecked(s.compressibility)
            self.ref_mode.set_data_silently(s.ref_mode)
            self.ref_frac.set_si(100.0 * s.ref_x_frac_mac)
            self.ref_x.set_si(s.ref_x_abs_m)
            self.ref_z.set_si(s.ref_z_abs_m)
        finally:
            self._busy = False
        self._apply_visibility()
        self._refresh_readouts()

    def store(self, state: ProjectState) -> None:
        state.flow = self.spec()

    def spec(self) -> FlowSpec:
        s = copy.deepcopy(self._spec)
        s.atmosphere = self.atmosphere.currentData()
        s.altitude_m = self.altitude.si()
        s.delta_T = self.delta_t.si()
        s.rho = self.rho.si()
        s.T = self.temp.si()
        s.mu = self.mu.si() if self.use_mu.isChecked() else None
        s.speed_mode = self.speed_mode.currentData()
        s.V = self.speed.si()
        s.mach = self.mach.si()
        s.alpha_deg = self.alpha.si()
        s.beta_deg = self.beta.si()
        s.ground_effect = self.ground.isChecked()
        s.ground_height_m = self.ground_h.si()
        s.extra_cd0 = self.cd0_extra.si()
        s.compressibility = self.compress.isChecked()
        s.ref_mode = self.ref_mode.currentData()
        s.ref_x_frac_mac = self.ref_frac.si() / 100.0
        s.ref_x_abs_m = self.ref_x.si()
        s.ref_z_abs_m = self.ref_z.si()
        return s

    def set_system(self, system: str) -> None:
        self._system = system
        for w in (self.altitude, self.rho, self.temp, self.mu, self.ground_h, self.ref_x, self.ref_z):
            w.set_system(system)
        self.speed.set_system(system)
        self._refresh_readouts()

    def problems(self) -> list[str]:
        return []

    # ========================================================================= helpers
    def _apply_visibility(self) -> None:
        isa = self.atmosphere.currentData() == ATMOSPHERE_ISA
        f = self.air_form
        for w, show in ((self.altitude, isa), (self.delta_t, isa), (self.rho, not isa), (self.temp, not isa)):
            f.setRowVisible(w, show)
        f.setRowVisible(self.use_mu, not isa)
        by_speed = self.speed_mode.currentData() == SPEED_BY_VELOCITY
        self.speed_form.setRowVisible(self.speed, by_speed)
        self.speed_form.setRowVisible(self.mach, not by_speed)
        self.mu.setEnabled(self.use_mu.isChecked())
        self.ground_h.setEnabled(self.ground.isChecked())
        absolute = self.ref_mode.currentData() == REF_ABSOLUTE
        self.ref_form.setRowVisible(self.ref_frac, not absolute)
        self.ref_form.setRowVisible(self.ref_x, absolute)
        self.ref_form.setRowVisible(self.ref_z, absolute)

    def _refresh_readouts(self) -> None:
        spec = self.spec()
        try:
            flow = flow_from_spec(spec)
        except ProjectError as exc:
            self.problem.setText(f"Invalid flow condition: {exc}")
            self.air_readout.setText("")
            self.speed_readout.setText("")
            return
        self.problem.setText("")
        sys_ = self._system
        uD, uP = units.display_unit(sys_, "density"), units.display_unit(sys_, "pressure")
        uT, uV = units.display_unit(sys_, "temperature"), "m/s" if sys_ == "SI" else "kt"
        self.air_readout.setText(
            f"rho {units.from_si('density', flow.rho, uD):.5g} {uD},  T {units.from_si('temperature', flow.T, uT):.1f} {uT},  "
            f"p {units.from_si('pressure', flow.p, uP):.5g} {uP},  a {units.from_si('speed', flow.a, uV):.1f} {uV},  "
            f"mu {flow.mu:.4g} Pa s"
        )
        self.speed_readout.setText(
            f"V = {units.from_si('speed', flow.V, uV):.4g} {uV},  Mach {flow.mach:.3f},  q = {units.from_si('pressure', flow.q, uP):.5g} {uP}"
        )

    # ========================================================================== slots
    def _edited(self, *_a) -> None:
        if self._busy:
            return
        self._apply_visibility()
        self._refresh_readouts()
        self.changed.emit()

    def _atmosphere_changed(self, _i: int) -> None:
        self._edited()

    def _speed_mode_changed(self, _i: int) -> None:
        if self._busy:
            return
        # keep the two speed entries consistent when the user switches between them
        try:
            flow = flow_from_spec(self.spec())
            self._busy = True
            self.speed.set_si(flow.V)
            self.mach.set_si(min(flow.mach, 0.99))
        except ProjectError:
            pass
        finally:
            self._busy = False
        self._edited()

    def _ground_toggled(self, _on: bool) -> None:
        self._edited()

    def _ref_mode_changed(self, _i: int) -> None:
        self._edited()
