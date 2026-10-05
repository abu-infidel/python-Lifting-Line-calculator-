"""Wing 3-D Model Exporter: a separate small app that turns a wing described in the main program into an STL model.

It runs in its own process, so meshing never slows down or disturbs the calculator. Three ways to use it:

* from the calculator: toolbar button **3-D model (STL)** (or File / Tools menu) - the current wing opens here;
* ``python stl_exporter.py [project.json]`` - the window, optionally with a saved project already loaded;
* ``python stl_exporter.py project.json -o wing.stl [--quality fine] [--unit mm] [--scale 1] [--half]`` - no window.

The project file is the one the calculator writes (File > Save). Airfoils given by name are looked up in the airfoil
library folder chosen in the calculator (shared settings) or given with ``--library``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from .airfoil.geometry import Airfoil
from .airfoil.resolver import AirfoilNotFound, AirfoilResolver
from .core.wing import Wing
from .io.project import ProjectError, ProjectState
from .io.stl import (
    DEFAULT_QUALITY,
    DEFAULT_UNIT,
    QUALITY,
    UNITS,
    MeshInfo,
    StlError,
    estimate_triangles,
    export_wing_stl,
    wing_mesh,
)

APP_TITLE = "Wing 3-D Model Exporter (STL)"


@dataclass
class LoadedWing:
    state: ProjectState
    wing: Wing
    airfoils: list[Airfoil]
    sources: list[str]  # one line per section: what its airfoil key resolved to
    notes: list[str]


def _settings_library_folder() -> str:
    try:
        from .gui.settings import Settings

        return Settings().get_str("library/folder", "")
    except Exception:  # no Qt or no settings: no library
        return ""


def load_wing(state: ProjectState, library_folder: str | None = None) -> LoadedWing:
    """Wing and section airfoils of a project state; the library is scanned only if a key needs it."""
    try:
        wing = state.build_wing()
    except ProjectError as exc:
        raise ProjectError(f"the wing cannot be built: {exc}") from exc
    resolver = AirfoilResolver()
    folder = library_folder if library_folder is not None else _settings_library_folder()
    airfoils, sources, notes = [], [], []
    for i, sec in enumerate(wing.sections):
        try:
            res = resolver.resolve(sec.airfoil)
        except AirfoilNotFound as exc:
            if resolver.library is None and folder:
                from .airfoil.library import AirfoilLibrary

                try:
                    resolver.set_library(AirfoilLibrary.scan(folder))
                except OSError as exc2:
                    raise ProjectError(f"airfoil library folder {folder!r}: {exc2}") from exc2
                try:
                    res = resolver.resolve(sec.airfoil)
                except AirfoilNotFound as exc3:
                    raise ProjectError(f"section {i + 1}: {exc3}") from exc3
            else:
                raise ProjectError(
                    f"section {i + 1}: {exc}" + ("" if folder else " (no airfoil library folder is set)")
                ) from exc
        airfoils.append(res.airfoil)
        sources.append(f"{res.airfoil.name} ({res.source})")
        if res.note:
            notes.append(res.note)
    return LoadedWing(state, wing, airfoils, sources, list(dict.fromkeys(notes)))


def load_project(path: str | Path, library_folder: str | None = None) -> LoadedWing:
    return load_wing(ProjectState.load(path), library_folder)


def export_project(project: str | Path, out: str | Path, *, library_folder: str | None = None, **options) -> MeshInfo:
    """Read a calculator project and write its wing as STL (options as for :func:`export_wing_stl`)."""
    lw = load_project(project, library_folder)
    return export_wing_stl(out, lw.wing, lw.airfoils, **options)


def describe(info: MeshInfo, unit: str, scale: float) -> str:
    f = scale / UNITS[unit]
    (x0, y0, z0), (x1, y1, z1) = info.bounds_m
    return (
        f"{info.n_triangles:,} triangles, {info.n_vertices:,} vertices, "
        f"{'watertight (closed, outward normals)' if info.watertight else 'NOT watertight'}; "
        f"size {f * (x1 - x0):.4g} x {f * (y1 - y0):.4g} x {f * (z1 - z0):.4g} {unit}; "
        f"volume {info.volume_m3:.4g} m^3 at full size; file {info.file_bytes / 1e6:.2f} MB"
    )


# ===================================================================================== command line
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="stl_exporter", description=APP_TITLE + ". Without -o the window opens.")
    p.add_argument("project", nargs="?", help="project file written by the calculator (File > Save)")
    p.add_argument("-o", "--output", help="write this STL file and exit (no window)")
    p.add_argument("--quality", choices=list(QUALITY), default=DEFAULT_QUALITY)
    p.add_argument("--unit", choices=list(UNITS), default=DEFAULT_UNIT)
    p.add_argument("--scale", type=float, default=1.0, help="model scale (0.05 = 1:20)")
    p.add_argument("--half", action="store_true", help="right half only, closed at the root")
    p.add_argument("--no-incidence", action="store_true", help="do not tilt the wing by its root incidence")
    p.add_argument("--ascii", action="store_true", help="ASCII STL instead of binary")
    p.add_argument("--library", help="airfoil library folder (default: the one chosen in the calculator)")
    a = p.parse_args(argv)
    if a.output:
        if not a.project:
            p.error("give the project file to export")
        try:
            info = export_project(a.project, a.output, library_folder=a.library, quality=a.quality, unit=a.unit,
                                  scale=a.scale, half=a.half, include_incidence=not a.no_incidence, ascii=a.ascii)
        except (ProjectError, StlError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"{a.output}: {describe(info, a.unit, a.scale)}")
        return 0 if info.watertight else 2
    return run_gui(a.project, a.library)


# ======================================================================================== window
def run_gui(project: str | None = None, library: str | None = None) -> int:
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationDisplayName(APP_TITLE)
    w = window_class()(library_folder=library)
    if project:
        w.open_project(project)
    w.show()
    return app.exec()


def _make_window_class():
    from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal
    from PySide6.QtWidgets import (
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QSplitter,
        QVBoxLayout,
        QWidget,
    )
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
    from matplotlib.figure import Figure

    class _Signals(QObject):
        done = Signal(object)
        failed = Signal(str)

    class _ExportTask(QRunnable):
        def __init__(self, fn) -> None:
            super().__init__()
            self.fn, self.signals = fn, _Signals()

        def run(self) -> None:  # noqa: A003 - Qt naming
            try:
                out = self.fn()
            except Exception as exc:  # report, never crash the window
                self.signals.failed.emit(f"{type(exc).__name__}: {exc}")
                return
            self.signals.done.emit(out)

    class ExporterWindow(QMainWindow):
        def __init__(self, library_folder: str | None = None) -> None:
            super().__init__()
            self.setWindowTitle(APP_TITLE)
            self.library_folder = library_folder if library_folder is not None else _settings_library_folder()
            self.loaded: LoadedWing | None = None
            self.project_path: Path | None = None
            self.last_info: MeshInfo | None = None
            self.busy = False
            self.pool = QThreadPool(self)
            self.pool.setMaxThreadCount(1)
            self._tasks: list = []
            self._build()
            self.resize(1150, 680)
            self._fit_to_screen()
            self.setAcceptDrops(True)

        # ------------------------------------------------------------------ layout
        def _build(self) -> None:
            split = QSplitter()
            left = QWidget()
            lv = QVBoxLayout(left)

            box = QGroupBox("Wing from the calculator")
            v = QVBoxLayout(box)
            row = QHBoxLayout()
            self.open_btn = QPushButton("Open project...")
            self.open_btn.setToolTip("A project saved by the calculator (File > Save), *.json")
            self.open_btn.clicked.connect(lambda: self.open_project())
            row.addWidget(self.open_btn)
            self.lib_btn = QPushButton("Airfoil library...")
            self.lib_btn.setToolTip("Folder of .dat files for airfoils given by name (the calculator's library)")
            self.lib_btn.clicked.connect(self.choose_library)
            row.addWidget(self.lib_btn)
            row.addStretch(1)
            v.addLayout(row)
            self.summary = QLabel("Open a project, or press 3-D model (STL) in the calculator to send its wing here.")
            self.summary.setWordWrap(True)
            self.summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
            v.addWidget(self.summary)
            lv.addWidget(box)

            box = QGroupBox("Mesh and file")
            f = QFormLayout(box)
            self.quality = QComboBox()
            for key, q in QUALITY.items():
                self.quality.addItem(q.label, key)
            self.quality.setCurrentIndex(self.quality.findData(DEFAULT_QUALITY))
            self.quality.setToolTip("Fine: 160 points per airfoil surface, spanwise steps <= 1/4 of the local chord")
            self.unit = QComboBox()
            for u in UNITS:
                self.unit.addItem(u, u)
            self.unit.setCurrentIndex(self.unit.findData(DEFAULT_UNIT))
            self.unit.setToolTip("STL files carry no units; CAD and slicer programs usually expect millimetres")
            self.scale = QDoubleSpinBox()
            self.scale.setDecimals(4)
            self.scale.setRange(1e-4, 1e4)
            self.scale.setValue(1.0)
            self.scale.setSingleStep(0.05)
            self.scale.setToolTip("1 = full size; 0.05 = a 1:20 model")
            self.model = QComboBox()
            self.model.addItem("Full wing (both halves)", False)
            self.model.addItem("Right half, closed at the root", True)
            self.incidence = QCheckBox("Tilt by the root incidence (body axes)")
            self.incidence.setChecked(True)
            self.ascii = QCheckBox("ASCII STL (larger file)")
            f.addRow("Mesh quality", self.quality)
            f.addRow("Units", self.unit)
            f.addRow("Model scale", self.scale)
            f.addRow("Model", self.model)
            f.addRow(self.incidence)
            f.addRow(self.ascii)
            self.estimate = QLabel()
            self.estimate.setWordWrap(True)
            f.addRow(self.estimate)
            lv.addWidget(box)

            row = QHBoxLayout()
            self.export_btn = QPushButton("Export STL...")
            self.export_btn.setEnabled(False)
            self.export_btn.clicked.connect(lambda: self.export())
            row.addWidget(self.export_btn)
            row.addStretch(1)
            lv.addLayout(row)
            self.status = QLabel("")
            self.status.setWordWrap(True)
            self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
            lv.addWidget(self.status)
            lv.addStretch(1)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(left)
            scroll.setMinimumWidth(340)
            split.addWidget(scroll)

            right = QWidget()
            rv = QVBoxLayout(right)
            self.figure = Figure(figsize=(6, 4), layout="constrained")
            self.canvas = FigureCanvasQTAgg(self.figure)
            rv.addWidget(NavigationToolbar2QT(self.canvas, right))
            rv.addWidget(self.canvas, 1)
            split.addWidget(right)
            split.setStretchFactor(1, 1)
            self.setCentralWidget(split)
            for w in (self.quality, self.unit, self.model):
                w.currentIndexChanged.connect(self._options_changed)
            self.scale.valueChanged.connect(self._options_changed)
            self.incidence.toggled.connect(self._options_changed)

        def _fit_to_screen(self) -> None:
            screen = self.screen()
            if screen is None:
                return
            a = screen.availableGeometry()
            w, h = min(self.width(), int(a.width() * 0.95)), min(self.height(), int(a.height() * 0.92))
            self.resize(w, h)
            self.move(a.left() + (a.width() - w) // 2, a.top() + (a.height() - h) // 2)

        # ------------------------------------------------------------------ state
        def options(self) -> dict:
            return dict(quality=self.quality.currentData(), unit=self.unit.currentData(), scale=self.scale.value(),
                        half=bool(self.model.currentData()), include_incidence=self.incidence.isChecked(),
                        ascii=self.ascii.isChecked())

        def open_project(self, path: str | None = None) -> bool:
            if not path:
                path, _ = QFileDialog.getOpenFileName(self, "Open wing project", "", "Wing projects (*.json);;All files (*)")
                if not path:
                    return False
            try:
                lw = load_project(path, self.library_folder)
            except (ProjectError, OSError) as exc:
                self.status.setText(f"<span style='color:#b91c1c'>Cannot use {Path(path).name}: {exc}</span>")
                return False
            self.loaded, self.project_path = lw, Path(path)
            w = lw.wing
            rows = "".join(f"<br>&nbsp;&nbsp;section {i + 1} at y = {s.y:.3f} m, chord {s.chord:.3f} m: {src}"
                           for i, (s, src) in enumerate(zip(w.sections, lw.sources)))
            sweep = w.max_abs_sweep_c4_deg
            self.summary.setText(
                f"<b>{lw.state.name or w.name}</b> ({Path(path).name})<br>span {w.span:.3f} m, area {w.area:.3f} m&sup2;, "
                f"AR {w.aspect_ratio:.2f}, taper {w.taper_ratio:.2f}, c/4 sweep {sweep:.1f}&deg;, "
                f"dihedral {w.max_abs_dihedral_deg:.1f}&deg;, incidence {w.incidence_deg:.2f}&deg;" + rows
                + "".join(f"<br><i>{n}</i>" for n in lw.notes)
            )
            self.export_btn.setEnabled(True)
            self.status.setText("")
            self._options_changed()
            return True

        def choose_library(self) -> None:
            folder = QFileDialog.getExistingDirectory(self, "Airfoil library folder", self.library_folder or "")
            if folder:
                self.library_folder = folder
                if self.project_path is not None:
                    self.open_project(str(self.project_path))

        def _options_changed(self, *_a) -> None:
            if self.loaded is None:
                return
            o = self.options()
            n = estimate_triangles(self.loaded.wing, o["quality"], o["half"])
            size = (84 + 50 * n) if not o["ascii"] else 260 * n
            self.estimate.setText(f"about {n:,} triangles, {size / 1e6:.1f} MB")
            self.draw_preview()

        def draw_preview(self) -> None:
            """A coarse version of the model (the export itself uses the chosen quality)."""
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection

            self.figure.clear()
            if self.loaded is None:
                self.canvas.draw_idle()
                return
            o = self.options()
            V, F = wing_mesh(self.loaded.wing, self.loaded.airfoils, quality="coarse", half=o["half"],
                             include_incidence=o["include_incidence"])
            ax = self.figure.add_subplot(111, projection="3d")
            ax.add_collection3d(Poly3DCollection(V[F], facecolor="#93b4dc", edgecolor="#1e293b33", linewidths=0.15))
            c, r = 0.5 * (V.max(0) + V.min(0)), 0.5 * float((V.max(0) - V.min(0)).max())
            ax.set_xlim(c[0] - r, c[0] + r)
            ax.set_ylim(c[1] - r, c[1] + r)
            ax.set_zlim(c[2] - r, c[2] + r)
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=28, azim=-125)
            ax.set_xlabel("x aft [m]")
            ax.set_ylabel("y right [m]")
            ax.set_zlabel("z up [m]")
            ax.set_title("Preview (coarse); drag to rotate", fontsize=9)
            self.canvas.draw_idle()

        # ------------------------------------------------------------------ export
        def export(self, path: str | None = None) -> bool:
            if self.loaded is None or self.busy:
                return False
            if not path:
                start = str((self.project_path.with_suffix(".stl")) if self.project_path else "wing.stl")
                path, _ = QFileDialog.getSaveFileName(self, "Export STL", start, "STL models (*.stl)")
                if not path:
                    return False
                if not path.lower().endswith(".stl"):
                    path += ".stl"
            o = self.options()
            lw = self.loaded
            self.busy = True
            self.export_btn.setEnabled(False)
            self.status.setText("Meshing and writing...")
            task = _ExportTask(lambda: export_wing_stl(path, lw.wing, lw.airfoils, **o))
            task.signals.done.connect(lambda info: self._done(path, info, o))
            task.signals.failed.connect(self._failed)
            self._tasks.append(task)
            self.pool.start(task)
            return True

        def _done(self, path: str, info: MeshInfo, o: dict) -> None:
            self.busy = False
            self.last_info = info
            self.export_btn.setEnabled(True)
            self.status.setText(f"<b>Written {Path(path).name}</b>: {describe(info, o['unit'], o['scale'])}")

        def _failed(self, msg: str) -> None:
            self.busy = False
            self.export_btn.setEnabled(True)
            self.status.setText(f"<span style='color:#b91c1c'>Export failed: {msg}</span>")
            QMessageBox.warning(self, APP_TITLE, f"Export failed:\n{msg}")

        def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt naming
            if event.mimeData().hasUrls():
                event.acceptProposedAction()

        def dropEvent(self, event) -> None:  # noqa: N802 - Qt naming
            for url in event.mimeData().urls():
                if url.toLocalFile().lower().endswith(".json"):
                    self.open_project(url.toLocalFile())
                    break

        def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
            self.pool.waitForDone(120000)
            super().closeEvent(event)

    return ExporterWindow


def window_class():
    """The exporter window class (built on first use, so the command line never imports Qt)."""
    return __getattr__("ExporterWindow")


def __getattr__(name: str):
    if name == "ExporterWindow":
        cls = _make_window_class()
        globals()["ExporterWindow"] = cls
        return cls
    raise AttributeError(name)


if __name__ == "__main__":
    raise SystemExit(main())
