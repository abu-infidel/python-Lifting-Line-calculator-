"""The programmatic API (liftingline.api) and the command line (python -m liftingline <command>)."""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from liftingline import api
from liftingline.__main__ import main
from liftingline.core.analysis import run
from liftingline.core.scenario import ScenarioSpec, run_scenario
from liftingline.io.backend import Backend
from liftingline.io.presets import CESSNA_172S, PRESETS
from liftingline.io.stl import read_stl

ROOT = Path(__file__).resolve().parents[1]
MPH = 0.44704


def builtin(name: str, **kw):
    return api.apply_overrides(api.load_project(example=name), engine="builtin", **kw)


def run_cli(capsys, *args) -> tuple[int, str, str]:
    code = main(list(args))
    out = capsys.readouterr()
    return code, out.out, out.err


# ================================================================================ projects
def test_examples_are_found_by_name_file_name_or_a_unique_part():
    assert api.find_example("Cessna 172S (published wing)") == CESSNA_172S
    assert api.find_example("cessna-172s-published-wing.json") == CESSNA_172S
    assert api.find_example("cessna 172s") == CESSNA_172S
    assert api.find_example("SAILPLANE") == "Sailplane wing, AR 22"
    with pytest.raises(api.ApiError, match="several examples"):
        api.find_example("cessna")
    with pytest.raises(api.ApiError, match="no example"):
        api.find_example("concorde")
    assert {r["name"] for r in api.examples().data["examples"]} == set(PRESETS)
    for r in api.examples().data["examples"]:
        assert (ROOT / r["file"]).is_file()  # the example project files are where the listing says


def test_projects_load_from_files_dicts_and_names(tmp_path):
    st = PRESETS["Tapered wing, AR 10"]()
    st.save(tmp_path / "w.json")
    assert api.load_project(str(tmp_path / "w.json")) == st
    assert api.load_project(st.to_dict()) == st
    assert api.load_project("tapered") == st
    with pytest.raises(api.ApiError, match="not found"):
        api.load_project(str(tmp_path / "missing.json"))
    with pytest.raises(api.ApiError, match="not both"):
        api.load_project(st, example="tapered")


def test_overrides_convert_units_and_leave_the_original_alone():
    st = api.load_project(example="rectangular")
    v0 = st.flow.V
    new = api.apply_overrides(st, speed=100.0, speed_unit="kt", alpha_deg=7.0, altitude_m=1000.0, ground_height_m=1.5,
                              engine="builtin", solver="numerical", flaps=True, phase="takeoff")
    assert new.flow.V == pytest.approx(100 * 1852 / 3600) and st.flow.V == v0
    assert (new.flow.alpha_deg, new.flow.altitude_m, new.flow.ground_effect, new.flow.ground_height_m) == (7.0, 1000.0, True, 1.5)
    assert (new.solver.engine, new.solver.solver, new.high_lift.flaps_on, new.high_lift.phase) == ("builtin", "numerical", True, "takeoff")
    assert not api.apply_overrides(new, ground_height_m=0.0).flow.ground_effect
    assert api.apply_overrides(st, mach=0.3).flow.speed_mode == "mach"
    for bad in (dict(engine="xfoil"), dict(solver="vlm"), dict(speed=1.0, speed_unit="furlong/s"), dict(phase="cruise"),
                dict(speed=1.0, mach=0.1)):
        with pytest.raises(api.ApiError):
            api.apply_overrides(st, **bad)


# ================================================================================ results
def test_analysis_json_equals_the_pipeline_and_is_plain_json():
    st = builtin(CESSNA_172S, altitude_m=0.0)
    res = api.analyze(st)
    direct = run(Backend().build_request(st))
    d = res.data
    assert d["schema"] == "liftingline.analysis/1" and d["solver_used"] == direct.solver_used
    t = d["totals"]
    assert t["L_N"] == direct.totals.L and t["CL"] == direct.totals.CL and t["M_pitch_Nm"] == direct.totals.M_pitch
    assert t["alpha_root_deg"] == pytest.approx(st.flow.alpha_deg + st.wing.incidence_deg)
    assert d["flow"]["q_Pa"] == pytest.approx(0.5 * d["flow"]["rho_kgm3"] * d["flow"]["V_mps"] ** 2)
    assert len(d["spanwise"]["y_m"]) == len(direct.load.y) and d["wing"]["area_m2"] == pytest.approx(direct.request.wing.area)
    text = json.dumps(d, allow_nan=False)  # no NaN / Infinity anywhere
    assert json.loads(text) == d
    assert "Lift L" in res.text and res.csv.startswith("# Lifting-line wing calculator")


def test_nan_becomes_null():
    assert api.clean({"a": math.nan, "b": [np.float64(1.5), np.inf], "c": np.int64(3), "d": np.bool_(True)}) == \
        {"a": None, "b": [1.5, None], "c": 3, "d": True}


def test_sweep_trim_and_max_lift():
    st = builtin("Rectangular wing, AR 8")
    sw = api.sweep(st, "alpha", -2.0, 14.0, 9).data
    cl = [p["CL"] for p in sw["points"]]
    assert sw["unit"] == "deg" and len(cl) == 9 and np.all(np.diff(cl) > 0)
    assert sw["stall"]["available"]
    with pytest.raises(api.ApiError, match="sweep variable"):
        api.sweep(st, "flap", 0, 1, 3)
    W = 600.0 * 9.80665
    tr = api.trim(st, mass_kg=600.0).data
    assert tr["solved_for"] == "alpha_deg" and tr["totals"]["L_N"] == pytest.approx(W, rel=1e-6)
    sp = api.trim(st, lift_N=W, solve_for="speed").data
    assert sp["solved_for"] == "V_mps" and sp["totals"]["L_N"] == pytest.approx(W, rel=1e-6)
    with pytest.raises(api.ApiError, match="exactly one"):
        api.trim(st, CL=0.5, mass_kg=1.0)
    ml = api.max_lift(st, weight_N=W).data
    c = ml["clean"]
    assert 0.8 < c["CL_max"] < 1.6 and c["alpha_first_stall_deg"] <= c["alpha_CL_max_deg"]
    vs = ml["speeds_mps"]["stall speed (clean)"]
    assert vs == pytest.approx(math.sqrt(2 * W / (1.225 * 8.0 * c["CL_max"])), rel=1e-3)


def test_scenario_spec_in_user_units_matches_the_core():
    st = builtin(CESSNA_172S, altitude_m=0.0)
    spec = api.scenario_spec(speed_unit="mph", v_start=0, v_end=90, v_rotate=60, ramp="per_speed",
                             ramp_rate_deg_per_speed_unit=0.3, mass_kg=1100, time_model=True, points=21)
    assert spec.v_end == pytest.approx(90 * MPH) and spec.ramp_rate_deg_per_mps == pytest.approx(0.3 / MPH)
    res = api.scenario(st, spec, max_frames=8)
    direct = run_scenario(Backend().build_request(st), spec)
    d = res.data
    assert d["liftoff"]["airspeed_mps"] == pytest.approx(direct.liftoff.V, rel=1e-12)
    assert d["liftoff"]["t_s"] == pytest.approx(direct.liftoff.t, rel=1e-12)
    assert len(d["frames"]) <= 8 and d["schema"] == "liftingline.scenario/1"
    json.dumps(d, allow_nan=False)
    with pytest.raises(api.ApiError, match="rotation speed"):
        api.scenario(st, api.scenario_spec(speed_unit="mph", v_end=50, v_rotate=60))
    with pytest.raises(api.ApiError, match="unknown scenario fields"):
        api.scenario(st, {"v_end": 30.0, "flaps": 1})
    assert api.scenario_spec(ground_height_m=2.0).ground_effect and not api.scenario_spec(ground_height_m=0.0).ground_effect


def test_the_api_and_the_mcp_module_never_import_qt():
    code = ("import sys, liftingline.api, liftingline.mcp_server, liftingline.__main__; "
            "print(any(m.startswith('PySide6') for m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"


# ================================================================================ command line
def test_cli_analyze_json_csv_and_text(capsys, tmp_path):
    code, out, _ = run_cli(capsys, "analyze", "-e", "rectangular", "--engine", "builtin", "--alpha", "4", "-f", "json")
    d = json.loads(out)
    st = builtin("Rectangular wing, AR 8", alpha_deg=4.0)
    assert code == 0 and d["totals"]["CL"] == api.analyze(st).data["totals"]["CL"]
    code, _, err = run_cli(capsys, "analyze", "-e", "rectangular", "--engine", "builtin", "-f", "csv", "-o",
                           str(tmp_path / "r.csv"))
    assert code == 0 and "wrote" in err and (tmp_path / "r.csv").read_text().startswith("# Lifting-line")
    code, out, _ = run_cli(capsys, "analyze", "-e", "rectangular", "--engine", "builtin", "--speed", "100",
                           "--speed-unit", "km/h")
    assert code == 0 and "V 27.778 m/s" in out


def test_cli_scenario_csv_matches_the_api(capsys, tmp_path):
    args = ["scenario", "-e", "cessna 172s", "--engine", "builtin", "--altitude", "0", "--speed-unit", "mph",
            "--v-end", "90", "--v-rotate", "60", "--target", "10", "--time-model", "--points", "21"]
    code, _, _ = run_cli(capsys, *args, "-f", "csv", "-o", str(tmp_path / "s.csv"))
    assert code == 0
    lines = (tmp_path / "s.csv").read_text().splitlines()
    rows = list(csv.DictReader(ln for ln in lines if not ln.startswith(("#", '"#'))))
    assert float(rows[-1]["V_mph"]) == pytest.approx(90.0) and "t_s" in rows[0]
    assert any("lift-off" in ln for ln in lines if ln.startswith(("#", '"#')))
    code, out, _ = run_cli(capsys, *args, "-f", "json", "--max-frames", "5")
    d = json.loads(out)
    assert 60 * MPH < d["liftoff"]["airspeed_mps"] < 90 * MPH and len(d["frames"]) <= 5


def test_cli_other_commands(capsys, tmp_path):
    code, out, _ = run_cli(capsys, "examples", "-f", "json")
    assert code == 0 and len(json.loads(out)["examples"]) == len(PRESETS)
    code, out, _ = run_cli(capsys, "show", "-e", "elliptic", "-f", "json")
    assert code == 0 and json.loads(out)["project"]["wing"]["mode"] == "elliptic"
    code, out, _ = run_cli(capsys, "trim", "-e", "rectangular", "--engine", "builtin", "--cl", "0.5")
    assert code == 0 and "needs angle of attack" in out
    code, out, _ = run_cli(capsys, "sweep", "-e", "rectangular", "--engine", "builtin", "--variable", "speed",
                           "--start", "36", "--stop", "72", "--speed-unit", "km/h", "-n", "3", "-f", "json")
    assert code == 0 and [p["value"] for p in json.loads(out)["points"]] == pytest.approx([10.0, 15.0, 20.0])
    stl = tmp_path / "w.stl"
    code, out, _ = run_cli(capsys, "stl", "-e", "rectangular", "--stl", str(stl), "--quality", "coarse", "-f", "json")
    assert code == 0 and json.loads(out)["watertight"] and len(read_stl(stl)) == json.loads(out)["n_triangles"]


def test_cli_errors_and_unchanged_entry_points(capsys, monkeypatch):
    code, out, err = run_cli(capsys, "analyze", "-e", "concorde")
    assert code == 1 and out == "" and err.startswith("error: no example")
    code, out, _ = run_cli(capsys, "--version")
    assert code == 0 and out.startswith("liftingline ")
    with pytest.raises(SystemExit):
        main(["trim", "-e", "rectangular"])  # a target is required
    capsys.readouterr()
    called = []
    import liftingline.gui.app as app

    monkeypatch.setattr(app, "run_gui", lambda: called.append(True) or 0)
    assert main([]) == 0 and called == [True]  # no command: the GUI, as before


def test_validate_as_json(capsys, monkeypatch):
    from liftingline.core import validate

    fake = [validate.Check("g", "ok", validate.PASS), validate.Check("g", "bad", validate.FAIL, "1", "2", "0")]
    monkeypatch.setattr(validate, "run_validation", lambda progress=None: fake)
    code, out, _ = run_cli(capsys, "validate", "-f", "json")
    d = json.loads(out)
    assert code == 1 and not d["passed"] and d["counts"] == {"pass": 1, "fail": 1, "skip": 0}


def test_scenario_defaults_come_from_the_project(tmp_path):
    st = builtin("Rectangular wing, AR 8")
    st.scenario = ScenarioSpec(v_end=40.0, v_rotate=20.0, mass_kg=400.0, points=11, speed_unit="m/s")
    d = api.scenario(st).data
    assert d["inputs"]["mass_kg"] == 400.0 and d["frames"][-1]["V_mps"] == pytest.approx(40.0)


def test_the_agent_guide_documents_every_output_key():
    """README_FOR_CLAUDE.md must name every key the JSON results contain (and the schemas)."""
    from liftingline.core.scenario import COLUMNS

    doc = (ROOT / "README_FOR_CLAUDE.md").read_text(encoding="utf-8")
    st = builtin("Rectangular wing, AR 8")
    an = api.analyze(st).data
    keys = set(an) | set(an["wing"]) | set(an["flow"]) | set(an["totals"]) | set(an["spanwise"])
    keys |= set(an["wing"]["sections"][0])
    sw = api.sweep(st, "alpha", 0.0, 10.0, 3).data
    keys |= set(sw) | set(sw["points"][0]) | set(sw["stall"])
    keys |= {k for k, _l, _u in COLUMNS} | set(ScenarioSpec().to_dict())
    keys |= set(api.examples().data["examples"][0])
    missing = sorted(k for k in keys if f"`{k}`" not in doc and not any(
        f"{sep}{k}{end}" in doc for sep in ("`", ", ", " ", "{", "[") for end in (",", "`", "}", " ", "]")))
    assert not missing, f"undocumented keys: {missing}"
    for schema in api.SCHEMA.values():
        assert schema in doc
