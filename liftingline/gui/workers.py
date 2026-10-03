"""Background work for the window: analysis runs and other long tasks (library scan, health check)."""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass
from typing import Callable

from PySide6.QtCore import QObject, QRunnable, Signal

from ..core.analysis import AnalysisResult, run
from ..io.backend import Backend, SectionInfo
from ..io.project import ProjectError, ProjectState


@dataclass
class RunOutput:
    run_id: int
    state: ProjectState
    result: AnalysisResult
    infos: list[SectionInfo]
    elapsed: float  # seconds, including building the section model (engine start-up, polar tables)


class RunSignals(QObject):
    finished = Signal(object)  # RunOutput
    failed = Signal(int, str, str, bool)  # run id, message, details, input_error


class RunTask(QRunnable):
    """Build the request for a project state and run the analysis off the UI thread."""

    def __init__(self, backend: Backend, state: ProjectState, run_id: int) -> None:
        super().__init__()
        self.setAutoDelete(True)
        self.backend, self.state, self.run_id = backend, state, run_id
        self.signals = RunSignals()

    def run(self) -> None:  # noqa: A003 - Qt naming
        t0 = time.perf_counter()
        try:
            req = self.backend.build_request(self.state)
            result = run(req)
            infos = self.backend.describe_sections(req.section, req.wing, req.flow)
            out = RunOutput(self.run_id, self.state, result, infos, time.perf_counter() - t0)
        except ProjectError as exc:
            self.signals.failed.emit(self.run_id, str(exc), "", True)
            return
        except Exception as exc:  # never let a worker die silently
            self.signals.failed.emit(self.run_id, f"{type(exc).__name__}: {exc}", traceback.format_exc(), False)
            return
        self.signals.finished.emit(out)


class TaskSignals(QObject):
    progress = Signal(int, int)
    finished = Signal(object)
    failed = Signal(str)


class FuncTask(QRunnable):
    """Run ``fn(progress)`` in the pool; ``progress(done, total)`` reports back to the UI thread."""

    def __init__(self, fn: Callable[[Callable[[int, int], None]], object]) -> None:
        super().__init__()
        self.setAutoDelete(True)
        self.fn = fn
        self.signals = TaskSignals()

    def run(self) -> None:  # noqa: A003 - Qt naming
        try:
            result = self.fn(lambda done, total: self.signals.progress.emit(int(done), int(total)))
        except Exception as exc:
            self.signals.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.signals.finished.emit(result)
