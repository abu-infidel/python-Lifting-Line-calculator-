# Notes for Claude

* **Using the calculator as a tool** (MCP server, command line, Python API) and the meaning of every output:
  [README_FOR_CLAUDE.md](README_FOR_CLAUDE.md). The human manual is [README.md](README.md).
* Layout: `liftingline/core` (physics, GUI-free), `liftingline/airfoil` (airfoils, section engines), `liftingline/io`
  (projects, presets, reports, STL), `liftingline/gui` (PySide6), `liftingline/api.py` (one call per task, JSON-ready;
  used by `__main__.py` = the CLI and `mcp_server.py`). `api.py`, `mcp_server.py` and `core/` must never import Qt.
* Tests: `pytest -q` (GUI tests run headless with `QT_QPA_PLATFORM=offscreen`; NeuralFoil / MCP tests skip when those
  packages are missing). `tests/test_golden.py` pins the results of every example to 1e-12: a change that moves them is
  a change of the calculator and must be intentional (regenerate with `python tools/make_golden.py`).
* Example projects in `examples/` are generated from `liftingline/io/presets.py` by `python tools/make_examples.py`.
