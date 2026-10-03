"""The main window: input panels on the left, plots in the middle, results on the right.

The window owns no physics. It collects a :class:`ProjectState` from the panels, hands it to a background
:class:`RunTask` (which calls :func:`liftingline.core.analysis.run` through the shared
:class:`~liftingline.io.backend.Backend`) and shows what comes back.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThreadPool, QTimer, Qt
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QScrollArea,
    QStyle,
    QTabWidget,
    QWidget,
)

from .. import __version__
from ..airfoil.engines import NEURALFOIL, available_engines
from ..airfoil.library import AirfoilLibrary
from ..core.analysis import SOLVER_AUTO
from ..io.backend import Backend
from ..io.presets import PRESETS, light_aircraft
from ..io.project import ProjectError, ProjectState
from ..io.report import result_to_csv
from .airfoil_panel import AirfoilPanel
from .flow_panel import FlowPanel
from .library_dialog import LibraryDialog, ReportDialog
from .plots import make_canvas, plot_airfoils, plot_planform, plot_spanwise
from .results_view import ResultsView
from .settings import Settings
from .solver_panel import SolverPanel
from .theme import apply_theme, current_theme
from .widgets import set_role
from .wing_panel import WingPanel
from .workers import FuncTask, RunOutput, RunTask

APP_TITLE = "Lifting-Line Wing Calculator"

CONVENTIONS_HTML = """
<h3>Axes and sign conventions</h3>
<ul>
<li><b>Geometry</b>: x points aft (root leading edge at 0), y to the right wing, z up. Sweep is positive when the
wing is swept back, dihedral positive when the tips are up, twist positive nose-up (washout is negative).</li>
<li><b>Flow</b>: angle of attack &alpha; is the angle of the root chord line to the freestream (root incidence is
added); sideslip &beta; is positive when the wind comes from the right.</li>
<li><b>Forces</b>: lift is perpendicular to the freestream (up positive), drag along it (aft positive), side force
positive towards the right wing.</li>
<li><b>Moments</b>: pitch nose-up positive, roll right-wing-down positive, yaw nose-right positive, about the
reference point on the Flow tab (default 25&nbsp;% of the mean aerodynamic chord).</li>
<li><b>Coefficients</b>: C<sub>L</sub>, C<sub>D</sub> use the projected area S and the dynamic pressure; C<sub>m</sub>
uses S and the mean aerodynamic chord; C<sub>l</sub> and C<sub>n</sub> use S and the span.</li>
<li><b>Tip loss</b> is part of the theory: the circulation falls to zero at the tips, which produces the downwash and
the induced drag. No extra empirical tip factor is applied.</li>
</ul>
"""


class MainWindow(QMainWindow):
    def __init__(
        self,
        backend: Backend | None = None,
        settings: Settings | None = None,
        *,
        autorun: bool = True,
        restore_layout: bool = True,
        library_folder: str | None = None,
    ) -> None:
        super().__init__()
        self.backend = backend or Backend()
        self.settings = settings or Settings()
        self.system = self.settings.get_str("ui/units", "SI")
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self.last: RunOutput | None = None
        self.path: Path | None = None
        self.project_name = "Untitled"
        self.notes = ""
        self._dirty = False
        self._loading = False
        self._running = False
        self._pending = False
        self._run_id = 0
        self._stale = False
        self._tasks: list = []  # keep Python references to running tasks (their signal objects)

        self.setWindowTitle(APP_TITLE)
        self.resize(1500, 900)
        self._build_panels()
        self._build_central()
        self._build_results_dock()
        self._build_toolbar()
        self._build_menus()
        self._build_statusbar()

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(450)
        self._debounce.timeout.connect(self.request_run)

        self.resizeDocks([self.inputs_dock, self.results_dock], [420, 450], Qt.Horizontal)
        if restore_layout:
            geo, state = self.settings.get_bytes("window/geometry"), self.settings.get_bytes("window/state")
            if geo is not None:
                self.restoreGeometry(geo)
            if state is not None:
                self.restoreState(state)

        self.apply_state(light_aircraft())
        self._dirty = False
        self._update_title()
        folder = library_folder if library_folder is not None else self.settings.get_str("library/folder", "")
        if folder:
            self.scan_library(folder, quiet=True)
        if autorun:
            QTimer.singleShot(0, self.request_run)

    # ======================================================================== layout
    def _build_panels(self) -> None:
        self.wing_panel = WingPanel(self.system)
        self.airfoil_panel = AirfoilPanel(self.backend)
        self.flow_panel = FlowPanel(self.system)
        self.solver_panel = SolverPanel()
        self.input_tabs = QTabWidget()
        for panel, title in (
            (self.wing_panel, "Wing"),
            (self.airfoil_panel, "Airfoils"),
            (self.flow_panel, "Flow"),
            (self.solver_panel, "Solver"),
        ):
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QScrollArea.NoFrame)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            scroll.setWidget(panel)
            self.input_tabs.addTab(scroll, title)
        dock = QDockWidget("Inputs", self)
        dock.setObjectName("inputs-dock")
        dock.setWidget(self.input_tabs)
        dock.setMinimumWidth(380)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)
        self.inputs_dock = dock

        for panel in (self.wing_panel, self.airfoil_panel, self.flow_panel, self.solver_panel):
            panel.changed.connect(self._on_changed)
        self.wing_panel.sections_changed.connect(self._sync_stations)
        self.wing_panel.preset_requested.connect(self._apply_preset)
        self.airfoil_panel.selection_changed.connect(lambda _i: self._invalidate_views("airfoils"))
        self.airfoil_panel.library_pick_requested.connect(self.pick_from_library)
        self.solver_panel.library_browse_requested.connect(self.choose_library_folder)
        self.solver_panel.library_rescan_requested.connect(self.rescan_library)
        self.solver_panel.library_health_requested.connect(self.library_health_check)

    def _build_central(self) -> None:
        self.plot_tabs = QTabWidget()
        self.canvases = {
            "planform": make_canvas(),
            "spanwise": make_canvas(),
            "airfoils": make_canvas(),
        }
        self.plot_tabs.addTab(self.canvases["planform"], "Planform")
        self.plot_tabs.addTab(self.canvases["spanwise"], "Spanwise loads")
        self.plot_tabs.addTab(self.canvases["airfoils"], "Airfoils")
        self.plot_tabs.currentChanged.connect(lambda _i: self._draw_visible())
        self.setCentralWidget(self.plot_tabs)
        self._stale_views: set[str] = {"planform", "spanwise", "airfoils"}

    def _build_results_dock(self) -> None:
        self.results = ResultsView()
        dock = QDockWidget("Results", self)
        dock.setObjectName("results-dock")
        dock.setWidget(self.results)
        dock.setMinimumWidth(360)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        self.results_dock = dock

    def _build_toolbar(self) -> None:
        tb = self.addToolBar("Main")
        tb.setObjectName("main-toolbar")
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.act_run = QAction(self.style().standardIcon(QStyle.SP_MediaPlay), "Run", self)
        self.act_run.setShortcut(QKeySequence("Ctrl+R"))
        self.act_run.setToolTip("Run the analysis now (Ctrl+R)")
        self.act_run.triggered.connect(self.request_run)
        tb.addAction(self.act_run)
        self.act_auto = QAction("Auto-run", self)
        self.act_auto.setCheckable(True)
        self.act_auto.setChecked(self.settings.get_bool("ui/autorun", True))
        self.act_auto.setToolTip("Re-run automatically a moment after every change")
        self.act_auto.toggled.connect(self._autorun_toggled)
        tb.addAction(self.act_auto)
        tb.addSeparator()

        tb.addWidget(QLabel(" Solver "))
        self.solver_combo = QComboBox()
        for key, label in (
            ("auto", "Automatic"),
            ("classical", "Classical (straight wing)"),
            ("numerical", "Numerical (sweep, dihedral, sideslip, ground)"),
        ):
            self.solver_combo.addItem(label, key)
        self.solver_combo.setToolTip("Automatic uses the classical Prandtl solution whenever the wing and flow allow it")
        self.solver_combo.currentIndexChanged.connect(self._on_changed)
        tb.addWidget(self.solver_combo)

        tb.addWidget(QLabel("  Section data "))
        self.engine_combo = QComboBox()
        avail = available_engines()
        for key, label in (("auto", "Automatic"), ("neuralfoil", "Max fidelity (NeuralFoil)"), ("builtin", "Fast (built-in)")):
            self.engine_combo.addItem(label, key)
        if not avail[NEURALFOIL][0]:
            item = self.engine_combo.model().item(1)
            item.setEnabled(False)
            item.setToolTip(avail[NEURALFOIL][1])
        self.engine_combo.setToolTip("Where the airfoil lift, drag and moment data come from")
        self.engine_combo.currentIndexChanged.connect(self._on_changed)
        tb.addWidget(self.engine_combo)

        tb.addWidget(QLabel("  Units "))
        self.units_combo = QComboBox()
        self.units_combo.addItem("SI (m, N, m/s)", "SI")
        self.units_combo.addItem("Imperial (ft, lbf, kt)", "Imperial")
        self.units_combo.setCurrentIndex(0 if self.system == "SI" else 1)
        self.units_combo.currentIndexChanged.connect(self._units_changed)
        tb.addWidget(self.units_combo)

    def _build_menus(self) -> None:
        mb = self.menuBar()
        m = mb.addMenu("&File")
        self._add_action(m, "&New", self.new_project, "Ctrl+N")
        self._add_action(m, "&Open project...", self.open_project, "Ctrl+O")
        self.recent_menu = m.addMenu("Open &recent")
        self._add_action(m, "&Save", self.save_project, "Ctrl+S")
        self._add_action(m, "Save &as...", self.save_project_as, "Ctrl+Shift+S")
        m.addSeparator()
        self._add_action(m, "Export results as &CSV...", self.export_csv)
        m.addSeparator()
        self._add_action(m, "&Quit", self.close, "Ctrl+Q")
        self._refresh_recent_menu()

        m = mb.addMenu("&Examples")
        for name, make in PRESETS.items():
            act = QAction(name, self)
            act.triggered.connect(lambda _c=False, n=name: self._apply_preset(n))
            m.addAction(act)

        m = mb.addMenu("&Tools")
        self._add_action(m, "Airfoil library &folder...", self.choose_library_folder)
        self._add_action(m, "Library &health check...", self.library_health_check)

        m = mb.addMenu("&View")
        group = QActionGroup(self)
        for label, mode in (("System theme", "system"), ("Light theme", "light"), ("Dark theme", "dark")):
            act = QAction(label, self, checkable=True)
            act.setChecked(self.settings.get_str("ui/theme", "system") == mode)
            act.triggered.connect(lambda _c=False, md=mode: self.set_theme(md))
            group.addAction(act)
            m.addAction(act)
        m.addSeparator()
        m.addAction(self.inputs_dock.toggleViewAction())
        m.addAction(self.results_dock.toggleViewAction())

        m = mb.addMenu("&Help")
        self._add_action(m, "Axes and sign &conventions", self.show_conventions, "F1")
        self._add_action(m, "&About", self.show_about)

    def _add_action(self, menu, text: str, slot, shortcut: str | None = None) -> QAction:
        act = QAction(text, self)
        if shortcut:
            act.setShortcut(QKeySequence(shortcut))
        act.triggered.connect(lambda _checked=False: slot())
        menu.addAction(act)
        return act

    def _build_statusbar(self) -> None:
        sb = self.statusBar()
        self.status_label = QLabel("Ready")
        sb.addWidget(self.status_label, 1)
        self.busy_bar = QProgressBar()
        self.busy_bar.setRange(0, 0)
        self.busy_bar.setMaximumWidth(110)
        self.busy_bar.setTextVisible(False)
        self.busy_bar.hide()
        sb.addPermanentWidget(self.busy_bar)
        self.solver_label = QLabel("")
        sb.addPermanentWidget(self.solver_label)

    # ====================================================================== state
    def collect_state(self) -> ProjectState:
        st = ProjectState()
        self.wing_panel.store(st)
        self.airfoil_panel.store(st)
        self.flow_panel.store(st)
        self.solver_panel.store(st)
        st.solver.solver = self.solver_combo.currentData()
        st.solver.engine = self.engine_combo.currentData()
        st.name, st.notes = self.project_name, self.notes
        st.airfoils.fit(st.wing.n_sections())
        return st

    def apply_state(self, state: ProjectState) -> None:
        self._loading = True
        try:
            self.wing_panel.load(state)
            self.airfoil_panel.set_stations(self.wing_panel.station_labels())
            self.airfoil_panel.load(state)
            self.flow_panel.load(state)
            self.solver_panel.load(state)
            self.solver_combo.setCurrentIndex(max(self.solver_combo.findData(state.solver.solver), 0))
            self.engine_combo.setCurrentIndex(max(self.engine_combo.findData(state.solver.engine), 0))
            self.project_name, self.notes = state.name, state.notes
            self.last = None
            self.results.clear()
            self.airfoil_panel.show_infos(None)
        finally:
            self._loading = False
        self._invalidate_views("planform", "spanwise", "airfoils")

    def _apply_preset(self, name: str) -> None:
        if not self._confirm_discard():
            return
        self.apply_state(PRESETS[name]())
        self.path = None
        self._dirty = False
        self._update_title()
        self.results.log(f"loaded example: {name}")
        self.request_run()

    # ------------------------------------------------------------------ change handling
    def _on_changed(self, *_args) -> None:
        if self._loading:
            return
        self._dirty = True
        self._stale = True
        self._update_title()
        self._sync_stations()
        self._invalidate_views("planform", "airfoils")
        if self.act_auto.isChecked():
            self._debounce.start()

    def _sync_stations(self) -> None:
        if self._loading:
            return
        self.airfoil_panel.set_stations(self.wing_panel.station_labels())

    def _autorun_toggled(self, on: bool) -> None:
        self.settings.set("ui/autorun", on)
        if on and self._stale:
            self.request_run()

    def _units_changed(self, _i: int) -> None:
        self.system = self.units_combo.currentData()
        self.settings.set("ui/units", self.system)
        self._loading = True
        try:
            self.wing_panel.set_system(self.system)
            self.flow_panel.set_system(self.system)
            self.airfoil_panel.set_stations(self.wing_panel.station_labels())
        finally:
            self._loading = False
        self.results.set_system(self.system)
        self._invalidate_views("planform", "spanwise", "airfoils")

    def set_theme(self, mode: str) -> None:
        self.settings.set("ui/theme", mode)
        apply_theme(QApplication.instance(), mode)
        self.airfoil_panel.refresh_status()
        if self.last is not None:
            self.results.show_result(self.last.result)
        self._invalidate_views("planform", "spanwise", "airfoils")

    # ========================================================================== running
    def is_running(self) -> bool:
        return self._running

    def _collect_problems(self, state: ProjectState) -> list[str]:
        problems = self.wing_panel.problems() + self.flow_panel.problems() + self.solver_panel.problems()
        if problems:
            return problems
        problems = state.validate()
        if not problems:
            problems = self.airfoil_panel.problems()
        return problems

    def request_run(self) -> None:
        self._debounce.stop()
        state = self.collect_state()
        problems = self._collect_problems(state)
        if problems:
            self.results.show_problems(problems)
            self._set_status("Cannot run: " + problems[0], "error")
            return
        if self._running:
            self._pending = True
            return
        self._start_run(state)

    def _start_run(self, state: ProjectState) -> None:
        self._run_id += 1
        task = RunTask(self.backend, state, self._run_id)
        task.signals.finished.connect(self._on_run_finished)
        task.signals.failed.connect(self._on_run_failed)
        self._tasks.append(task)
        self._running = True
        self.busy_bar.show()
        self._set_status("Running...  (the first NeuralFoil analysis needs a few seconds to start up)", "")
        self.pool.start(task)

    def _finish_task(self, run_id: int) -> None:
        self._tasks = [t for t in self._tasks if getattr(t, "run_id", None) != run_id]
        self._running = False
        self.busy_bar.hide()

    def _start_func_task(self, task: FuncTask) -> None:
        """Run a library task in the global pool (it must not queue behind a long analysis)."""
        self._tasks.append(task)

        def forget(*_args) -> None:
            if task in self._tasks:
                self._tasks.remove(task)

        task.signals.finished.connect(forget)
        task.signals.failed.connect(forget)
        QThreadPool.globalInstance().start(task)

    def _on_run_finished(self, out: RunOutput) -> None:
        self._finish_task(out.run_id)
        if self._pending:
            self._pending = False
            self.request_run()
            return
        self.last = out
        self._stale = False
        res = out.result
        self.results.show_result(res)
        self.airfoil_panel.show_infos(out.infos)
        self._invalidate_views("planform", "spanwise", "airfoils")
        t = res.totals
        self.solver_label.setText(
            f"solver: {res.solver_used} | {res.info.get('n_terms', res.info.get('n_elements', ''))}"
            f" {'terms' if res.solver_used == 'classical' else 'elements'}"
            f" | residual {float(res.info.get('residual', 0.0)):.1e} | {out.elapsed * 1000:.0f} ms"
        )
        self._set_status(f"CL = {t.CL:.4f}   CD = {t.CD:.5f}   L/D = {t.L_over_D:.1f}   Cm = {t.Cm:.4f}", "ok")
        self.results.log(
            f"{res.solver_used} solver, {res.info.get('section_engine', '')}: CL {t.CL:.4f}, CD {t.CD:.5f} "
            f"({out.elapsed * 1000:.0f} ms)"
        )
        for w in res.warnings:
            self.results.log("note: " + w)

    def _on_run_failed(self, run_id: int, message: str, details: str, input_error: bool) -> None:
        self._finish_task(run_id)
        if self._pending:
            self._pending = False
            self.request_run()
            return
        if input_error:
            self.results.show_problems(message.split("\n"))
            self._set_status("Cannot run: " + message.split("\n")[0], "error")
        else:
            self.results.show_error(message)
            self._set_status("The analysis failed: " + message, "error")
        self.results.log(("input problem: " if input_error else "ERROR: ") + message)
        if details:
            self.results.log(details.strip().splitlines()[-1])

    def _set_status(self, text: str, level: str) -> None:
        self.status_label.setText(text)
        set_role(self.status_label, {"ok": "ok", "error": "error"}.get(level, ""))

    # ========================================================================== views
    def _invalidate_views(self, *names: str) -> None:
        self._stale_views.update(names)
        self._draw_visible()

    def _draw_visible(self) -> None:
        names = ("planform", "spanwise", "airfoils")
        current = names[self.plot_tabs.currentIndex()]
        if current in self._stale_views:
            self._stale_views.discard(current)
            getattr(self, f"_draw_{current}")()

    def _current_wing(self):
        try:
            return self.collect_state().build_wing()
        except ProjectError:
            return None

    def _draw_planform(self) -> None:
        wing = self._current_wing()
        canvas = self.canvases["planform"]
        if wing is None:
            canvas.draw_with(None, "The wing description is not valid yet.\nSee the message under the planform inputs.")
            return
        result = None
        if self.last is not None and self.last.result.request.wing == wing and not self._stale:
            result = self.last.result
        system = self.system
        canvas.draw_with(lambda fig, th: plot_planform(fig, result, wing, system, th))

    def _draw_spanwise(self) -> None:
        canvas = self.canvases["spanwise"]
        if self.last is None:
            canvas.draw_with(None, "Run an analysis to see the spanwise distributions.")
            return
        result, system = self.last.result, self.system
        canvas.draw_with(lambda fig, th: plot_spanwise(fig, result, system, th))

    def _draw_airfoils(self) -> None:
        entries = []
        labels = self.wing_panel.station_labels()
        for label, key in zip(labels, self.airfoil_panel.spec().keys):
            short = label.split(" (")[0]
            try:
                entries.append((short, self.backend.resolve(key).airfoil))
            except Exception:
                entries.append((short, None))
        sel = self.airfoil_panel.selected()
        self.canvases["airfoils"].draw_with(lambda fig, th: plot_airfoils(fig, entries, sel, th))

    # ===================================================================== library
    def choose_library_folder(self) -> None:
        start = self.settings.get_str("library/folder", "")
        folder = QFileDialog.getExistingDirectory(self, "Choose the folder with the airfoil .dat files (UIUC database)", start)
        if folder:
            self.scan_library(folder)

    def rescan_library(self) -> None:
        folder = self.settings.get_str("library/folder", "")
        if folder:
            self.scan_library(folder)
        else:
            self.choose_library_folder()

    def scan_library(self, folder: str, quiet: bool = False) -> None:
        self.solver_panel.set_library_info(folder, "Indexing...", "")
        task = FuncTask(lambda progress: AirfoilLibrary.scan(folder, progress=progress))
        task.signals.progress.connect(lambda done, total: self.solver_panel.set_library_info(folder, f"Indexing {done} / {total} files..."))
        task.signals.finished.connect(lambda lib, f=folder: self._library_ready(f, lib))
        task.signals.failed.connect(lambda msg, f=folder: self._library_failed(f, msg, quiet))
        self._start_func_task(task)

    def _library_ready(self, folder: str, lib: AirfoilLibrary) -> None:
        self.backend.set_library(lib)
        self.settings.set("library/folder", folder)
        self.solver_panel.set_library_info(folder, f"{len(lib)} airfoil files indexed", "ok")
        self.airfoil_panel.set_completions([e.name for e in lib])
        self.airfoil_panel.refresh_status()
        self.results.log(f"airfoil library: {len(lib)} files from {folder}")
        self._invalidate_views("airfoils")
        if not self._running:
            self.request_run()  # 6-series codes may now resolve to exact ordinates

    def _library_failed(self, folder: str, msg: str, quiet: bool) -> None:
        self.solver_panel.set_library_info(folder, f"Cannot use this folder: {msg}", "error")
        if not quiet:
            QMessageBox.warning(self, APP_TITLE, f"The airfoil library could not be indexed:\n{msg}")

    def pick_from_library(self, row: int) -> None:
        lib = self.backend.library
        if lib is None:
            ans = QMessageBox.question(
                self, APP_TITLE, "No airfoil library folder is set. Choose the folder with your UIUC .dat files now?"
            )
            if ans == QMessageBox.Yes:
                self.choose_library_folder()
            return
        key = self.airfoil_panel.spec().keys[row] if 0 <= row < len(self.airfoil_panel.spec().keys) else ""
        dlg = LibraryDialog(lib, self, query=key if key and not key.upper().startswith("NACA") else "")
        if dlg.exec() == LibraryDialog.Accepted:
            name = dlg.selected_name()
            if name:
                self.airfoil_panel.set_key(row, name)

    def library_health_check(self) -> None:
        lib = self.backend.library
        if lib is None:
            QMessageBox.information(self, APP_TITLE, "Choose the airfoil library folder first (Solver tab or Tools menu).")
            return
        dlg = ReportDialog("Airfoil library health check", self)
        task = FuncTask(lambda progress: lib.health_report(progress).to_text())
        task.signals.progress.connect(dlg.set_progress)
        task.signals.finished.connect(lambda text: dlg.finish(text, "Finished"))
        task.signals.failed.connect(lambda msg: dlg.finish(msg, "Failed"))
        self._start_func_task(task)
        dlg.exec()

    # ======================================================================== files
    def _update_title(self) -> None:
        name = self.path.name if self.path else self.project_name
        self.setWindowTitle(f"{name}{'*' if self._dirty else ''} - {APP_TITLE}")

    def _confirm_discard(self) -> bool:
        if not self._dirty:
            return True
        ans = QMessageBox.question(
            self, APP_TITLE, "The project has unsaved changes. Discard them?", QMessageBox.Discard | QMessageBox.Cancel
        )
        return ans == QMessageBox.Discard

    def new_project(self) -> None:
        if not self._confirm_discard():
            return
        self.apply_state(light_aircraft())
        self.path = None
        self._dirty = False
        self._update_title()
        self.request_run()

    def open_project(self, path: str | None = None) -> None:
        if not self._confirm_discard():
            return
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Open project", self.settings.get_str("files/last_dir", ""), "Wing projects (*.json);;All files (*)")
        if not path:
            return
        try:
            state = ProjectState.load(path)
        except ProjectError as exc:
            QMessageBox.warning(self, APP_TITLE, str(exc))
            return
        self.apply_state(state)
        self.path = Path(path)
        self._dirty = False
        self.settings.add_recent(path)
        self.settings.set("files/last_dir", str(Path(path).parent))
        self._refresh_recent_menu()
        self._update_title()
        self.results.log(f"opened {path}")
        self.request_run()

    def save_project(self) -> bool:
        if self.path is None:
            return self.save_project_as()
        return self._write(self.path)

    def save_project_as(self) -> bool:
        start = str(self.path) if self.path else str(Path(self.settings.get_str("files/last_dir", "")) / f"{self.project_name}.json")
        path, _ = QFileDialog.getSaveFileName(self, "Save project", start, "Wing projects (*.json)")
        if not path:
            return False
        if not path.lower().endswith(".json"):
            path += ".json"
        return self._write(Path(path))

    def _write(self, path: Path) -> bool:
        try:
            state = self.collect_state()
            if self.path is None or path != self.path:
                state.name = path.stem
                self.project_name = state.name
            state.save(path)
        except (OSError, ProjectError) as exc:
            QMessageBox.warning(self, APP_TITLE, f"Could not save the project:\n{exc}")
            return False
        self.path = path
        self._dirty = False
        self.settings.add_recent(str(path))
        self.settings.set("files/last_dir", str(path.parent))
        self._refresh_recent_menu()
        self._update_title()
        self.results.log(f"saved {path}")
        return True

    def _refresh_recent_menu(self) -> None:
        self.recent_menu.clear()
        recents = self.settings.recent_files()
        self.recent_menu.setEnabled(bool(recents))
        for p in recents:
            act = QAction(p, self)
            act.triggered.connect(lambda _c=False, q=p: self.open_project(q))
            self.recent_menu.addAction(act)

    def export_csv(self) -> None:
        if self.last is None:
            QMessageBox.information(self, APP_TITLE, "Run an analysis first.")
            return
        start = str(Path(self.settings.get_str("files/last_dir", "")) / f"{self.project_name}-results.csv")
        path, _ = QFileDialog.getSaveFileName(self, "Export results", start, "CSV files (*.csv)")
        if not path:
            return
        try:
            Path(path).write_text(result_to_csv(self.last.result, self.system, self.project_name), encoding="utf-8")
        except OSError as exc:
            QMessageBox.warning(self, APP_TITLE, f"Could not write the file:\n{exc}")
            return
        self.results.log(f"exported {path}")

    # ========================================================================== help
    def show_conventions(self) -> None:
        QMessageBox.information(self, "Conventions", CONVENTIONS_HTML)

    def show_about(self) -> None:
        QMessageBox.about(
            self,
            f"About {APP_TITLE}",
            f"<h3>{APP_TITLE} {__version__}</h3>"
            "<p>Lift, drag and moments of a wing from Prandtl's lifting-line theory, with the section data of "
            "NACA and UIUC airfoils. Works offline.</p>"
            "<p>Classical Glauert solution for straight wings, numerical horseshoe-vortex lifting line for sweep, "
            "dihedral, sideslip and ground effect.</p>",
        )

    # ======================================================================== close
    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if not self._confirm_discard():
            event.ignore()
            return
        self.settings.set("window/geometry", self.saveGeometry())
        self.settings.set("window/state", self.saveState())
        self.settings.sync()
        self._debounce.stop()
        self.pool.waitForDone(3000)
        super().closeEvent(event)


__all__ = ["MainWindow", "APP_TITLE", "current_theme", "QWidget", "SOLVER_AUTO"]
