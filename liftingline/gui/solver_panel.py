"""The Solver tab: numerical settings, section-data (NeuralFoil) settings and the airfoil library folder."""

from __future__ import annotations

import copy

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..io.project import MODEL_SIZES, NONLINEAR_CHOICES, ProjectState, SolverSpec
from .widgets import GuardedCombo, GuardedSpinBox, UnitSpin, muted, set_role


class SolverPanel(QWidget):
    changed = Signal()
    library_browse_requested = Signal()
    library_rescan_requested = Signal()
    library_health_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._spec = SolverSpec()
        self._busy = False
        self._build()
        self.load_spec(self._spec)

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        # ---- lifting-line solution
        box = QGroupBox("Lifting-line solution")
        f = QFormLayout(box)
        self.n_terms = GuardedSpinBox()
        self.n_terms.setRange(8, 800)
        self.n_terms.setToolTip("Fourier terms of the classical solution; 80 is converged to better than 0.1 % for ordinary wings")
        self.n_terms.valueChanged.connect(self._edited)
        self.n_panels = GuardedSpinBox()
        self.n_panels.setRange(8, 800)
        self.n_panels.setToolTip("Horseshoe vortices per half-wing in the numerical solver (cosine spacing)")
        self.n_panels.valueChanged.connect(self._edited)
        self.nonlinear = GuardedCombo()
        for key, label in (("auto", "Automatic"), ("on", "Always (real section polars)"), ("off", "Never (linear lift)")):
            self.nonlinear.addItem(label, key)
        self.nonlinear.setToolTip(
            "Nonlinear: the section polars (stall, drag buckets) are used inside the lifting-line equations. "
            "Linear: lift = a0 (alpha - alpha_L0) with the polar's slope and zero-lift angle."
        )
        self.nonlinear.currentIndexChanged.connect(self._edited)
        self.convergence = QCheckBox("check convergence (re-solve at 2x resolution)")
        self.convergence.toggled.connect(self._edited)
        f.addRow("Fourier terms (classical)", self.n_terms)
        f.addRow("Elements per half-wing", self.n_panels)
        f.addRow("Section model", self.nonlinear)
        f.addRow(self.convergence)
        root.addWidget(box)

        # ---- section data
        box = QGroupBox("Section data (NeuralFoil)")
        f = QFormLayout(box)
        self.model_size = GuardedCombo()
        for m in MODEL_SIZES:
            self.model_size.addItem(m, m)
        self.model_size.setToolTip("Network size: larger is more accurate and slower; xlarge is the default")
        self.model_size.currentIndexChanged.connect(self._edited)
        self.n_crit = UnitSpin(None, unit="", decimals=1, minimum=1.0, maximum=20.0, step=0.5,
                               tooltip="Transition amplification factor e^N: 9 = average wind tunnel, 11-14 = quiet air / sailplane, 4-8 = rough or dirty surface")
        self.n_crit.valueChanged.connect(self._edited)
        self.xtr_u = UnitSpin(None, unit="", decimals=3, minimum=0.0, maximum=1.0, step=0.05, tooltip="Forced transition position on the upper surface (x/c); 1 = free transition")
        self.xtr_l = UnitSpin(None, unit="", decimals=3, minimum=0.0, maximum=1.0, step=0.05, tooltip="Forced transition position on the lower surface (x/c); 1 = free transition")
        for w in (self.xtr_u, self.xtr_l):
            w.valueChanged.connect(self._edited)
        f.addRow("Network size", self.model_size)
        f.addRow("n_crit", self.n_crit)
        f.addRow("Trip, upper (x/c)", self.xtr_u)
        f.addRow("Trip, lower (x/c)", self.xtr_l)
        f.addRow(muted("These settings apply to the NeuralFoil engine; the built-in engine ignores them."))
        root.addWidget(box)

        # ---- library
        box = QGroupBox("Airfoil library (UIUC .dat files)")
        v = QVBoxLayout(box)
        row = QHBoxLayout()
        self.library_path = QLineEdit()
        self.library_path.setReadOnly(True)
        self.library_path.setPlaceholderText("no folder chosen")
        row.addWidget(self.library_path, 1)
        b = QPushButton("Browse...")
        b.clicked.connect(self.library_browse_requested)
        row.addWidget(b)
        v.addLayout(row)
        row = QHBoxLayout()
        b = QPushButton("Rescan")
        b.clicked.connect(self.library_rescan_requested)
        row.addWidget(b)
        b = QPushButton("Health check...")
        b.setToolTip("Parse every file once and report formats, repairs and failures")
        b.clicked.connect(self.library_health_requested)
        row.addWidget(b)
        row.addStretch(1)
        v.addLayout(row)
        self.library_status = muted("")
        v.addWidget(self.library_status)
        root.addWidget(box)

        self.problem = QLabel()
        self.problem.setWordWrap(True)
        set_role(self.problem, "error")
        root.addWidget(self.problem)
        root.addStretch(1)

    # ======================================================================== state
    def load(self, state: ProjectState) -> None:
        self.load_spec(state.solver)

    def load_spec(self, spec: SolverSpec) -> None:
        self._spec = copy.deepcopy(spec)
        s = self._spec
        self._busy = True
        try:
            self.n_terms.set_silently(s.n_terms)
            self.n_panels.set_silently(s.n_panels)
            self.nonlinear.set_data_silently(s.nonlinear if s.nonlinear in NONLINEAR_CHOICES else "auto")
            self.convergence.blockSignals(True)
            self.convergence.setChecked(s.convergence_check)
            self.convergence.blockSignals(False)
            self.model_size.set_data_silently(s.model_size)
            self.n_crit.set_si(s.n_crit)
            self.xtr_u.set_si(s.xtr_upper)
            self.xtr_l.set_si(s.xtr_lower)
        finally:
            self._busy = False

    def store(self, state: ProjectState) -> None:
        state.solver = self.spec(state.solver)

    def spec(self, base: SolverSpec | None = None) -> SolverSpec:
        """The panel's settings merged into ``base`` (which carries the toolbar's solver and engine choice)."""
        s = copy.deepcopy(base if base is not None else self._spec)
        s.n_terms = self.n_terms.value()
        s.n_panels = self.n_panels.value()
        s.nonlinear = self.nonlinear.currentData()
        s.convergence_check = self.convergence.isChecked()
        s.model_size = self.model_size.currentData()
        s.n_crit = self.n_crit.si()
        s.xtr_upper = self.xtr_u.si()
        s.xtr_lower = self.xtr_l.si()
        return s

    def problems(self) -> list[str]:
        return []

    # ------------------------------------------------------------------- library UI
    def set_library_info(self, folder: str, text: str, level: str = "") -> None:
        self.library_path.setText(folder)
        self.library_path.setToolTip(folder)
        self.library_status.setText(text)
        set_role(self.library_status, level or "muted")

    def _edited(self, *_a) -> None:
        if not self._busy:
            self.changed.emit()
