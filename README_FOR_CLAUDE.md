# Lifting-line wing calculator — guide for AI agents (Claude and others)

This file is for an AI agent that runs the calculator without its window: through the **MCP server**, the
**command line**, or the **Python API**. It explains what every output means. The human-facing manual is `README.md`.

The program computes the aerodynamic forces and moments of a wing with Prandtl's lifting-line theory: lift, induced and
profile drag, pitching/rolling/yawing moments, stall and maximum lift, flaps and slats, and forces along a takeoff run.
It works offline. All three interfaces call the same functions as the GUI (`liftingline/api.py`), so the numbers are
the same.

## 1. Connecting

| Interface | Start | Output |
|---|---|---|
| MCP server (stdio) | `python -m liftingline mcp [--library FOLDER]` | JSON objects (tool results) |
| Command line | `python -m liftingline <command> [options]` | `--format text` (default), `json` or `csv`; `-o FILE` |
| Python | `from liftingline import api` | `ApiResult(data=dict, text=str, csv=str or None)` |

Install: `pip install -r requirements.txt` (includes the MCP SDK, `mcp>=1.25,<2`). Run the commands from the
repository root, or set `PYTHONPATH` to it, or `pip install -e .` once (then `liftingline <command>` works anywhere).

**Claude Code:** `claude mcp add --transport stdio --env PYTHONPATH=/path/to/repo liftingline -- /path/to/python -m liftingline mcp`

**Claude Desktop / any MCP client** (`mcpServers` entry):

```json
{
  "mcpServers": {
    "liftingline": {
      "command": "/path/to/python",
      "args": ["-m", "liftingline", "mcp"],
      "env": {"PYTHONPATH": "/path/to/repo"}
    }
  }
}
```

`LIFTINGLINE_LIBRARY=/folder/of/dat/files` (or `--library`) points to a UIUC-style airfoil folder; it is only needed
for airfoil names that are not NACA 4-/5-digit codes (NACA 6-series, `e387`, ...).

## 2. Conventions (all outputs)

* **SI units, unit in the key name**: `L_N` (newton), `M_pitch_Nm` (N·m), `V_mps` (m/s), `q_Pa`, `S_m2`, `x_m`,
  `alpha_deg`, `t_s`, `a_mps2`, `rho_kgm3`, `mu_Pas`, `T_K`. Keys without a unit are dimensionless (`CL`, `CD`,
  `Cm`, `e`, `L_over_D`, `AR`, `mach`, `re_mac`).
* **`null` means "does not exist"** (e.g. no time model, no stall reached, span efficiency at zero lift). JSON never
  contains NaN or Infinity.
* **`schema`** names the format and its version (`liftingline.analysis/1` ...). Fields may be added in later versions;
  existing fields keep their meaning.
* **Geometry**: x aft from the root leading edge, y along the span to the right wing (the root at 0), z up. Sweep
  positive swept back, dihedral positive tips up, twist positive nose-up.
* **Forces and moments**: lift perpendicular to the free stream (up positive), drag along it (aft positive), side
  force to the right positive. Pitching moment **nose-up positive**, rolling moment **right-wing-down positive**,
  yawing moment **nose-right positive**. Sideslip is positive with the wind from the right.
* **Angle of attack**: inputs and `flow.alpha_deg` are the angle of the *fuselage reference line* (the GUI's Flow
  tab). The wing's root incidence is added inside the analysis; `totals.alpha_root_deg` = flow angle + incidence is
  the angle of the root chord. Twist (washout negative) is relative to the root.
* **Coefficients**: on the wing area `S` (projected), `q = ½ρV²`. `Cm` uses the mean aerodynamic chord `MAC` and is
  taken about the moment reference (default 25 % MAC, `x_ref_m` aft of the root leading edge). Roll and yaw
  coefficients use the span `b`.
* **Drag** `D = D_induced + D_profile + D_extra`: induced from the lifting line, profile from the section polars at
  the local Reynolds number and effective angle, extra = the project's parasite `CD0` for the rest of the aircraft
  (fuselage, tail; 0 by default). The wing alone: no fuselage, no tail, no trim download.
* **Atmosphere**: ISA (altitude in m, optional ΔT) or a manual density/temperature.
* **Section data** (`section_data`): `NeuralFoil` (max fidelity, default when installed) or `Built-in (fast)`. Choose
  with `engine` = `neuralfoil` / `builtin`. The built-in engine is faster and good for lift and moment; drag and stall
  are more accurate with NeuralFoil. Imported polar files show their own label.
* **Solver** (`solver_used`): `classical` (Glauert Fourier series: straight, planar wings) or `numerical`
  (horseshoe-vortex lifting line: sweep, dihedral, sideslip, ground effect). `auto` chooses.

Always read **`warnings`**: they say when a case is outside the theory's range (low aspect ratio, high sweep, near
stall, high Mach, low NeuralFoil confidence, a solve that did not converge) and how large the error may be.

## 3. Choosing the wing (inputs common to the tools)

Give **one** of:

* `example` — a built-in example: its name, its file name or a unique part (`"Cessna 172S"`, `"rectangular"`,
  `"sailplane"`). `list_examples` / `python -m liftingline examples` lists them; their project files are in
  `examples/`.
* `project_file` — a project JSON written by the GUI (File > Save) or by you.
* `project` — the project object itself (MCP: `describe_project(...)["project"]`, edit it, pass it back).

Nothing given: the GUI's start-up wing (Cessna 172-like light aircraft).

Overrides of the flight condition (any subset; the rest comes from the project): `alpha_deg`, `speed` with
`speed_unit` (`m/s`, `km/h`, `kt`, `mph`, `ft/s`) or `mach`, `altitude_m` (ISA), `beta_deg` (sideslip),
`ground_height_m` (quarter-chord height above the ground: ground effect; `0` = off), `extra_cd0`, `engine`, `solver`,
`flaps` / `slats` (`true` deploys the devices *defined in the project*; projects define a default device), and
`high_lift_setting` (`takeoff` / `landing`).

**Project format** (`schema` of `describe_project`: `liftingline.project/1`): `wing` (mode `trapezoid`, `elliptic` or
`sections`; span, root/tip chord, sweep, dihedral, twist, incidence, or a `sections` table of `y, chord, x_le, z,
twist_deg`), `airfoils.keys` (one airfoil name per section, e.g. `"NACA 2412"`), `flow` (`altitude_m`, `V`,
`alpha_deg`, `beta_deg`, `ground_effect`, `ground_height_m`, `extra_cd0`, ...), `solver` (`engine`, `solver`,
`n_terms`, `n_panels`, ...), `controls` (control surfaces), `high_lift` (flaps/slats definitions and switches),
`scenario` (the Scenario inputs, section 5.5). All in SI.

## 4. Tools / commands

| MCP tool | CLI command | Does | Result `schema` |
|---|---|---|---|
| `list_examples` | `examples` | built-in example wings | `liftingline.examples/1` |
| `describe_project` | `show` | every input + wing geometry + flow | `liftingline.project/1` |
| `analyze_wing` | `analyze` | forces and moments at one condition | `liftingline.analysis/1` |
| `sweep` | `sweep` | the analysis over alpha / speed / altitude / beta | `liftingline.sweep/1` |
| `max_lift` | `max-lift` | CL_max, first stall, stall speed; flaps/slats comparison | `liftingline.max_lift/1` |
| `trim` | `trim` | angle (or airspeed) for a CL, a lift or a mass | `liftingline.trim/1` |
| `run_scenario` | `scenario` | forces along a start→end airspeed run, lift-off | `liftingline.scenario/1` |
| `compare_real_aircraft` | `compare-aircraft` | program vs Cessna 172S / 152 handbooks | `liftingline.compare/1` |
| `export_stl` | `stl --stl FILE` | watertight STL model of the wing | `liftingline.stl/1` |
| `self_test` | `validate` | built-in validation suite | `liftingline.validate/1` |
| `get_documentation` | — | this file (also MCP resource `liftingline://docs/readme-for-claude`) | text |

The computing tools (`analyze_wing`, `sweep`, `max_lift`, `trim`, `run_scenario`, `compare_real_aircraft`) also take
`output`: `json` (default, the structured result), `text` (the readable report) or `csv` (the table: `analyze_wing`,
`sweep`, `run_scenario` only); `text`/`csv` come back as `{"schema", "format", "content"}`.
Errors come back as tool errors with a readable message (MCP) or `error: ...` on stderr and exit code 1 (CLI;
`validate` also exits 1 when a check fails; bad options exit 2).

Typical times with the built-in engine: an analysis 0.05–1 s, an 11-point sweep 0.5–5 s, `max_lift` 1–5 s, a
61-point scenario 2–10 s, `compare_real_aircraft` about 5 s. NeuralFoil takes several times longer (and loads its
model on the first call).

CLI examples:

```bash
python -m liftingline analyze -e "Cessna 172S" --speed 100 --speed-unit kt --alpha 2 --altitude-ft 5000
python -m liftingline analyze my_wing.json -f json -o result.json
python -m liftingline sweep -e rectangular --variable alpha --start -4 --stop 16 -n 11 -f csv -o polar.csv
python -m liftingline max-lift -e "Cessna 172S" --flaps --high-lift-setting landing --mass 1157
python -m liftingline trim -e "Cessna 172S" --mass 1157 --speed 110 --speed-unit kt
python -m liftingline scenario -e "Cessna 172S" --altitude 0 --speed-unit mph --v-end 90 --v-rotate 60 \
    --ramp target --target 10 --mass 1157 --time-model --thrust-static 2400 --thrust-end 1800 -f csv -o run.csv
python -m liftingline stl -e sailplane --stl sailplane.stl --unit mm --scale 0.1
```

## 5. Output formats

### 5.1 `liftingline.analysis/1` (`analyze_wing`, `analyze`)

| Key | Content |
|---|---|
| `program` | `{name, version}` |
| `project` | `{name, notes}` |
| `wing` | `span_m, area_m2, aspect_ratio, taper_ratio, mac_m, root_chord_m, tip_chord_m, incidence_deg, max_sweep_c4_deg, max_dihedral_deg, sections[{y_m, chord_m, x_le_m, z_m, twist_deg, airfoil}]` (right half-wing, y from the root) |
| `flow` | `V_mps, alpha_deg, beta_deg, altitude_m, rho_kgm3, mu_Pas, T_K, speed_of_sound_mps, mach, q_Pa, re_mac` |
| `solver_used`, `section_data`, `nonlinear` | which solver, which section data, whether full nonlinear polars were used |
| `high_lift` | deployed devices, e.g. `"landing setting: single-slotted flap"`, or `null` |
| `totals` | the results, below |
| `max_cl_over_clmax` | highest local cl/cl_max along the span (≥ 0.98: a section is stalling); `null` without cl_max |
| `warnings` | list of strings |
| `info` | solver details: `solver_reason`, `n_terms` (classical) or `n_elements` (numerical), `iterations`, `residual`, `convergence_dCL`, `convergence_dCDi` (change when the resolution is doubled), `solve_time_s`, ... |
| `spanwise` | only with `include_spanwise` (MCP) / always in the CLI JSON unless `--no-spanwise`: arrays `y_m, chord_m, gamma_m2s` (circulation), `cl, cd, cm, alpha_eff_deg, alpha_induced_deg, re, cl_max` over the whole span, left tip (−b/2) to right tip |

`totals`:

| Key | Unit | Meaning |
|---|---|---|
| `L_N`, `CL` | N, – | lift (perpendicular to the free stream) |
| `D_N`, `CD` | N, – | total drag |
| `D_induced_N`, `CD_induced` | N, – | induced drag |
| `D_profile_N`, `CD_profile` | N, – | profile (section) drag |
| `D_extra_N`, `CD_extra` | N, – | parasite drag of the rest of the aircraft (`extra_cd0`) |
| `Y_N`, `CY` | N, – | side force (sideslip) |
| `M_pitch_Nm`, `Cm` | N·m, – | pitching moment about the reference point, nose-up + |
| `M_roll_Nm`, `C_roll` | N·m, – | rolling moment, right wing down + |
| `M_yaw_Nm`, `C_yaw` | N·m, – | yawing moment, nose right + |
| `L_over_D` | – | lift / drag |
| `e` | – | span efficiency CL²/(π AR CDi) (1 = elliptic loading); `null` at zero induced drag |
| `root_bending_right_Nm`, `root_bending_left_Nm` | N·m | lift bending moment at the root of each half-wing |
| `root_shear_N` | N | lift of one half-wing |
| `S_m2, b_m, AR, MAC_m, q_Pa` | | reference values |
| `x_ref_m`, `z_ref_m` | m | moment reference point (aft of the root leading edge; up) |
| `alpha_root_deg`, `beta_deg` | deg | root-chord angle of attack (flow + incidence), sideslip |

Real example (`analyze -e rectangular --engine builtin -f json --no-spanwise`, rectangular wing 8 m × 1 m, NACA 0012,
40 m/s, 5°, sea level), shortened:

```json
{
  "schema": "liftingline.analysis/1",
  "flow": {"V_mps": 40.0, "alpha_deg": 5.0, "rho_kgm3": 1.225, "mach": 0.1175, "q_Pa": 980.0, "re_mac": 2738378.3},
  "solver_used": "classical",
  "section_data": "Built-in (fast)",
  "totals": {"L_N": 3247.33, "D_N": 110.41, "D_induced_N": 57.23, "D_profile_N": 53.18, "CL": 0.4142,
             "CD": 0.014083, "CD_induced": 0.007300, "Cm": 1.6e-16, "e": 0.9351, "L_over_D": 29.41,
             "root_bending_right_Nm": 2982.7, "root_shear_N": 1623.7, "alpha_root_deg": 5.0},
  "max_cl_over_clmax": 0.306,
  "warnings": []
}
```

(`Cm` ≈ 1e-16 is zero: a symmetric airfoil about 25 % MAC. Values like 1e-13 in moments of a symmetric case are
round-off zeros.)

**CSV** (`-f csv`): `#` comment lines (program, project, solver, warnings), an empty line, a table
`quantity,SI value,coefficient` (rows `L, D, D_induced, D_profile, M_pitch, M_roll, M_yaw, root_bending, L_over_D,
e, stall_margin, S, b, AR, MAC, q, Re_MAC, Mach, alpha_deg`), an empty line, then the spanwise table with the columns
`y [m], chord [m], circulation [m^2/s], cl, cd, cm, alpha_eff [deg], alpha_induced [deg], Re, cl_max`.

### 5.2 `liftingline.sweep/1`

`variable` (`alpha`, `speed`, `altitude`, `beta`, `deflection`), `unit` (deg, m/s, m, deg, deg — the values are
always SI even when the CLI was given `--speed-unit`), `solver_used`, `section_data`, `notes`, and `points`, one per
value: `value, ok, converged, error` (a failed point has `ok: false`, an `error` text and `null` results),
`max_cl_over_clmax`, `first_stall_y_m` (|y| of the most loaded station), and `CL, CD, CD_induced, CD_profile, CY, Cm,
C_roll, C_yaw, L_over_D, e, L_N, D_N`. An alpha sweep of 3+ points adds `stall`: `available` (the section data have a
cl_max), `reached`, `alpha_first_stall_deg`, `CL_first_stall`, `y_first_stall_m`, `CL_max` and `alpha_CL_max_deg`
(highest CL *of the sweep*), `CL_max_is_peak` (false: the sweep ended before the peak — use `max_lift`), `note`.
CSV: `value,CL,CD,CD_induced,CD_profile,CY,Cm,C_roll,C_yaw,L_over_D,e,L,D,max_cl_over_clmax,converged`.

### 5.3 `liftingline.max_lift/1`

`clean` and `deployed` (`null` without flaps/slats), each `{label, CL_max, alpha_CL_max_deg, definition,
CL_first_stall, alpha_first_stall_deg, y_first_stall_m, note}`. `definition` says how CL_max was found: `peak of the
lift curve` (NeuralFoil, imported polars), `knee of the lift curve` (built-in engine: its polars saturate, so CL_max is
read where the lift-curve slope has fallen to 20 % of its attached-flow value), or `first section stall`. With
devices: `dCL_max` (lifting line) and `handbook_dCL_max` (Raymer's estimate), `devices`. `weight_N` (if given) and
`speeds_mps`: `stall speed (clean)`, `stall speed (deployed)`, and the conventional reference speeds (takeoff
setting: `lift-off speed (1.1 Vs)`, `climb-out speed V2 (1.2 Vs)`; landing: `approach speed (1.3 Vs)`, `touchdown
speed (1.15 Vs)`), all in m/s at `altitude_m`; without devices only `stall speed (clean)`.
The wing alone at the given altitude: an airplane's handbook stall speed is typically a few per cent higher (tail
download, fuselage) — see `compare_real_aircraft`.

### 5.4 `liftingline.trim/1`

`solved_for` (`alpha_deg` or `V_mps`), `value` with `unit`, `target` (text), `evaluations` (analyses run), `totals`
at the solution (as in 5.1), `warnings`. Solving for the angle returns the lowest angle on the rising branch of the
lift curve; an unreachable target is an error naming the CL range that was reached.

### 5.5 `liftingline.scenario/1` (`run_scenario`, `scenario`, the GUI's Scenario workspace)

The run: the airspeed goes from `v_start` to `v_end`; the angle of attack is `alpha_ground_deg` until `v_rotate`,
then rises linearly — `ramp_mode` `target` (to `ramp_target_deg` at `v_end`), `per_speed`
(`ramp_rate_deg_per_mps` per m/s; the tools take the rate per `speed_unit`) or `per_second` (`ramp_rate_deg_s`, needs
the time model). At each point the full analysis gives the forces; weight = `mass_kg` × 9.80665. **Lift-off** is the
first point where lift = weight (found by bisection between grid points). Optional **time model** (`time_model`: a takeoff roll on a
level runway, no wind): `m dV/dt = T(V) − D − μ (W − L)`, thrust linear from `thrust_static_N` at 0 to `thrust_end_N`
at `v_end`, rolling friction μ = `rolling_friction` while on the ground; after lift-off friction is zero and the run
continues along the same schedule (the climb is not modelled). `ground_effect` uses image vortices at
`ground_height_m`. Below 10 % of `v_end` the coefficients are frozen and the forces scale with q (so a run can start
from 0).

| Key | Content |
|---|---|
| `inputs` | the scenario fields above, SI (speeds in m/s); `speed_unit` is only the display unit |
| `solver`, `section_data`, `has_time` (time model on), `weight_N` | |
| `events` | `[{name, t_s, airspeed_mps, distance_m, alpha_deg, text}]`, names `start`, `rotation`, `lift-off`, `stall` (first section reaches 98 % of cl_max), `thrust limit` (acceleration reaches zero: the run stops there), `end`; `t_s`/`distance_m` are `null` without the time model |
| `liftoff` | `{t_s, airspeed_mps, distance_m, alpha_deg}` or `null` (no lift-off: a warning says so) |
| `warnings` | |
| `columns` | `{key: {label, unit}}` for the frame keys |
| `frames` | one object per point (thinned to `max_frames`, default 61 in MCP): `t_s, V_mps, alpha_deg, q_Pa, L_N, W_N, L_minus_W_N, L_over_W, D_N, Di_N, Dp_N, Dx_N, CL, CD, Cm, M_pitch_Nm, T_N, F_roll_N, F_net_N, a_mps2, x_m, cl_ratio, on_ground (1/0), stalled (1/0)`; time-model columns are `null` without it |

Real example: Cessna 172S wing at sea level (built-in engine), 0 → 90 mph, rotation at 60 mph, 0° rising to 10°,
1157 kg, thrust 2400 → 1800 N, μ 0.02:

```json
"events": [
  {"name": "start",    "t_s": 0.0,   "airspeed_mps": 0.0,   "distance_m": 0.0,   "alpha_deg": 0.0},
  {"name": "rotation", "t_s": 15.86, "airspeed_mps": 26.82, "distance_m": 220.5, "alpha_deg": 0.0},
  {"name": "lift-off", "t_s": 22.97, "airspeed_mps": 36.54, "distance_m": 447.3, "alpha_deg": 7.24},
  {"name": "end",      "t_s": 26.96, "airspeed_mps": 40.23, "distance_m": 601.2, "alpha_deg": 10.0}
]
```

i.e. the wing's lift overcomes the weight at 81.7 mph, 23.0 s and 447 m into the run, at 7.2° angle of attack.

**CSV** (`-f csv`, the GUI's Export CSV): `# ` comment lines (inputs, solver, one line per event, notes) — read it
with `pandas.read_csv(path, comment="#")` — then a header and one row per point with the frame columns above
(time-model columns only when it is on) plus `V_<unit>` (e.g. `V_mph`) after `V_mps` when the speed unit is not m/s.
**Text** (`-f text`, Export text): the same summary and a fixed-width table.

### 5.6 Others

* `liftingline.examples/1`: `examples[{name, slug, file, notes, span_m, area_m2, aspect_ratio, airfoils, V_mps,
  alpha_deg, altitude_m, high_lift_devices, ground_effect}]`.
* `liftingline.project/1`: `project` (the whole project object, section 3), `wing` and `flow` as in 5.1.
* `liftingline.compare/1`: `section_data`, `rows[{aircraft, point, handbook, program, ratio_program_over_handbook,
  detail}]` — CL_max at the handbook stall points (ratio program / handbook; see section 6) and the level-flight
  angle at the handbook's cruise speeds.
* `liftingline.stl/1`: `path, unit, scale, quality, half, ascii, n_triangles, n_vertices, watertight, volume_m3,
  area_m2` (full size), `bounds_m{min, max}`, `file_bytes`, `airfoils` (what each section's name resolved to), `notes`.
* `liftingline.validate/1`: `passed`, `counts{pass, fail, skip}`, `elapsed_s`, `checks[{group, name, status, expected,
  got, tolerance, note}]`.

## 6. Reading the results correctly

* **This is the wing alone.** No fuselage, tail, propeller or trim download: an airplane needs a few per cent more
  lift than its wing alone makes (tail download); add the parasite drag of the rest with `extra_cd0`.
* **Lifting-line limits** (each one raises a warning): aspect ratio below ~4, sweep beyond ~30° (the numerical solver
  is calibrated to about 6 % in the sweep effect), Mach above 0.7, post-stall (indicative only). Ground effect misses
  the 2-D chordwise part (lift gain under-predicted by ≈ 7 % × (c/h)²).
* **Stall**: a station is stalled at cl/cl_max ≥ 0.98. `max_lift` is the right tool for CL_max; an alpha sweep's
  `CL_max` is only the highest CL of that sweep.
* **Accuracy**: the classical solver reproduces exact elliptic-wing results to 1e-9; profile drag from the built-in
  engine is about ±25 %, NeuralFoil is XFOIL-like. `compare_real_aircraft` (built-in engine): Cessna 172S clean CL_max
  0.998 × the handbook value, Cessna 152 (assumed planform) 1.13 ×; with flaps the program is 16–29 % above the
  handbook (an idealised flap on the wing alone), so treat flapped CL_max as an upper estimate.
* Numbers are full double precision; round for presentation (3–4 significant digits is honest).

## 7. Python API

```python
from liftingline import api

st = api.load_project(example="Cessna 172S")            # or a path, a dict, a ProjectState
st = api.apply_overrides(st, speed=100, speed_unit="kt", alpha_deg=2, engine="builtin")
res = api.analyze(st)                                   # ApiResult
res.data["totals"]["L_N"], res.text, res.csv
spec = api.scenario_spec(speed_unit="mph", v_end=90, v_rotate=60, ramp="target", ramp_target_deg=10,
                         mass_kg=1157, time_model=True)
run = api.scenario(st, spec)                            # run.data["liftoff"]
```

Functions: `examples, describe_project, analyze, sweep, max_lift, trim, scenario, scenario_spec, compare_aircraft,
export_stl, validate, load_project, apply_overrides, find_example, make_backend`; errors raise `api.ApiError`.
