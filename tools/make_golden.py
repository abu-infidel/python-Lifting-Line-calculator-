"""Record the calculator's results for every example wing (built-in section engine) as a golden reference.

    python tools/make_golden.py        # writes tests/data/golden_results.json

tests/test_golden.py checks that later versions reproduce these numbers exactly, so additions (scenario, CLI, MCP
server ...) provably leave the existing calculator unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

from liftingline.core.analysis import run
from liftingline.core.sweeps import ALPHA, linspace_values, run_sweep
from liftingline.io.backend import Backend
from liftingline.io.presets import PRESETS

FIELDS = ("CL", "CD", "CD_induced", "CD_profile", "Cm", "C_roll", "C_yaw", "L", "D", "e", "M_pitch")


def collect() -> dict:
    be = Backend()
    out: dict = {"engine": "builtin", "examples": {}, "sweeps": {}, "high_lift": {}}
    for name, make in PRESETS.items():
        st = make()
        st.solver.engine = "builtin"
        t = run(be.build_request(st)).totals
        out["examples"][name] = {f: float(getattr(t, f)) for f in FIELDS}
    st = PRESETS["Rectangular wing, AR 8"]()
    st.solver.engine = "builtin"
    sw = run_sweep(be.build_request(st), ALPHA, linspace_values(-4.0, 16.0, 6))
    out["sweeps"]["Rectangular wing, AR 8: alpha"] = [float(v) for v in sw.column("CL")]
    st = next(iter(PRESETS.values()))()
    st.solver.engine = "builtin"
    st.high_lift.flaps_on = st.high_lift.slats_on = True
    t = run(be.build_request(st)).totals
    out["high_lift"]["first example, flaps and slats"] = {f: float(getattr(t, f)) for f in FIELDS}
    return out


def main() -> None:
    path = Path(__file__).resolve().parents[1] / "tests" / "data" / "golden_results.json"
    path.write_text(json.dumps(collect(), indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
