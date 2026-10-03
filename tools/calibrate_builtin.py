#!/usr/bin/env python3
"""Fit the viscous corrections of the built-in section engine against NeuralFoil.

    python tools/calibrate_builtin.py            # regenerate liftingline/airfoil/builtin_calibration.py

The built-in engine takes the *inviscid* lift slope, zero-lift angle and moment from the panel method
(geometry-aware, works for any airfoil) and applies simple empirical corrections for the viscous effects:

    a0_viscous  = a0_inviscid * a_ratio(t, m, Re)
    alpha_L0    = alpha_L0_inviscid          (viscous shift is within NeuralFoil's own scatter)
    cm_c4       = cm_c4_inviscid
    cl_max      = f(t, m, Re)                        alpha_stall = alpha_L0 + cl_max/a0 + delta
    cd_min      = exp(g(t, m, Re))   cd = cd_min + k (cl - cl0)^2 + stall rise

``t`` is thickness/chord, ``m`` the maximum camber/chord and ``L = log10(Re / 1e6)``. The constants are
obtained here by least squares on a NeuralFoil dataset over NACA 4-digit sections (5 cambers x 6
thicknesses x 7 Reynolds numbers) and then **frozen** in ``builtin_calibration.py``. NeuralFoil itself is a
surrogate of XFOIL, so the built-in engine is a fast approximation of an approximation: see the residual
statistics written into the generated module for its honest accuracy.
"""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liftingline.airfoil import naca4  # noqa: E402
from liftingline.airfoil.designation import NacaCode  # noqa: E402
from liftingline.airfoil.panel2d import PanelSolver  # noqa: E402
from liftingline.airfoil.section_base import linear_parameters  # noqa: E402
from liftingline.airfoil.section_builtin_features import FEATURES  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "liftingline" / "airfoil" / "builtin_calibration.py"

TRAIN_M = (0.0, 0.01, 0.02, 0.04, 0.06)
TRAIN_T = (0.06, 0.09, 0.12, 0.15, 0.18, 0.21)
TRAIN_RE = (2e5, 5e5, 1e6, 2e6, 4e6, 8e6, 1.6e7)
HOLDOUT_M = (0.0, 0.03, 0.05)
HOLDOUT_T = (0.075, 0.105, 0.14, 0.17)
HOLDOUT_RE = (7e5, 3e6, 1.2e7)
LIFT_RE_FLOOR = 5e5  # lift-related fits use Re >= this (laminar-bubble effects below are not modelled)
ALPHA = np.arange(-12.0, 26.01, 0.25)


def sample(ms, ts, res):
    import neuralfoil as nf

    rows = []
    for m, t in itertools.product(ms, ts):
        code = NacaCode(family="4", thickness=t, m=m, p=0.4 if m > 0 else 0.0)
        af = naca4.naca4_airfoil(code)
        ps = PanelSolver(af, 200)
        co = af.resample(80).coords
        for Re in res:
            r = nf.get_aero_from_coordinates(co, ALPHA, Re, model_size="xlarge")
            cl, cd, cm = (np.asarray(r[k], float) for k in ("CL", "CD", "CM"))
            lin = linear_parameters(ALPHA, cl, cm)
            deg_L0, deg_s = math.degrees(lin["alpha_L0"]), math.degrees(lin["alpha_stall"])
            att = (ALPHA > deg_L0 - 4) & (ALPHA < deg_s - 4)
            j = np.where(att)[0][np.argmin(cd[att])]
            w = att & (np.abs(cl - cl[j]) < 0.7)
            k_cd = float(np.polyfit(cl[w] - cl[j], cd[w] - cd[j], 2)[0])
            rows.append(
                dict(m=m, t=t, Re=Re, a0_inv=ps.lift_slope, aL0_inv=ps.alpha_L0, cm0_inv=ps.cm_c4_zero_lift,
                     a0=lin["a0"], aL0=lin["alpha_L0"], cm0=lin["cm_c4"], clmax=lin["cl_max"], a_s=lin["alpha_stall"],
                     cdmin=float(cd[j]), cl_cdmin=float(cl[j]), kcd=k_cd)
            )
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def design(names, d):
    t, m = d["t"], np.abs(d["m"])
    L = np.log10(np.clip(d["Re"], LIFT_RE_FLOOR, 1.6e7) / 1e6)
    return np.column_stack([FEATURES[n](t, m, L) for n in names])


def fit(names, y, d, mask=None):
    X = design(names, d)
    if mask is not None:
        X, y = X[mask], y[mask]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    resid = y - pred
    return coef, float(np.sqrt(np.mean(resid**2))), float(1 - np.sum(resid**2) / np.sum((y - y.mean()) ** 2))


def main() -> int:
    print("sampling NeuralFoil (training grid) ...")
    tr = sample(TRAIN_M, TRAIN_T, TRAIN_RE)
    lift = tr["Re"] >= LIFT_RE_FLOOR

    spec: dict[str, dict] = {}
    # (name, features, target, mask)
    a_names = ("1", "t", "m", "L", "tL")
    spec["a_ratio"] = dict(features=a_names, **dict(zip(("coef", "rms", "r2"), fit(a_names, tr["a0"] / tr["a0_inv"], tr, lift))))
    cl_names = ("1", "t", "t2", "m", "L", "tL", "L2")
    spec["cl_max"] = dict(features=cl_names, **dict(zip(("coef", "rms", "r2"), fit(cl_names, tr["clmax"], tr, lift))))
    cd_names = ("1", "t", "t2", "m", "L", "tL")
    spec["log_cd_min"] = dict(features=cd_names, **dict(zip(("coef", "rms", "r2"), fit(cd_names, np.log(tr["cdmin"]), tr))))
    c0_names = ("m",)  # through the origin: a symmetric section has its drag minimum at cl = 0
    spec["cl_at_cd_min"] = dict(features=c0_names, **dict(zip(("coef", "rms", "r2"), fit(c0_names, tr["cl_cdmin"], tr, lift))))
    # stall angle above the "linear" estimate cl_max / a0
    a0_model = tr["a0_inv"] * (design(a_names, tr) @ np.array(spec["a_ratio"]["coef"]))
    delta = float(np.mean((np.degrees(tr["a_s"] - tr["aL0"]) - np.degrees(tr["clmax"] / a0_model))[lift]))
    k_cd = float(np.median(np.clip(tr["kcd"], 0.002, 0.03)))

    print("sampling NeuralFoil (hold-out grid) ...")
    ho = sample(HOLDOUT_M, HOLDOUT_T, HOLDOUT_RE)
    ho_lift = ho["Re"] >= LIFT_RE_FLOOR
    stats = {}
    pred_a = ho["a0_inv"] * (design(a_names, ho) @ np.array(spec["a_ratio"]["coef"]))
    stats["a0 relative error (hold-out) rms"] = float(np.sqrt(np.mean(((pred_a - ho["a0"]) / ho["a0"])[ho_lift] ** 2)))
    pred_clmax = design(cl_names, ho) @ np.array(spec["cl_max"]["coef"])
    stats["cl_max relative error (hold-out) rms"] = float(np.sqrt(np.mean(((pred_clmax - ho["clmax"]) / ho["clmax"])[ho_lift] ** 2)))
    pred_cd = np.exp(design(cd_names, ho) @ np.array(spec["log_cd_min"]["coef"]))
    stats["cd_min relative error (hold-out) rms"] = float(np.sqrt(np.mean(((pred_cd - ho["cdmin"]) / ho["cdmin"]) ** 2)))
    ho_c = np.abs(ho["m"]) > 0
    stats["alpha_L0 abs error deg (panel value used as is) rms"] = float(np.sqrt(np.mean(np.degrees((ho["aL0_inv"] - ho["aL0"])[ho_lift & ho_c]) ** 2)))
    stats["cm_c4 abs error (panel value used as is) rms"] = float(np.sqrt(np.mean((ho["cm0_inv"] - ho["cm0"])[ho_lift & ho_c] ** 2)))

    lines = [
        '"""Frozen calibration of the built-in section engine (GENERATED by tools/calibrate_builtin.py).',
        "",
        "Do not edit by hand. Fitted by least squares to NeuralFoil polars of NACA 4-digit sections",
        f"({len(TRAIN_M)} cambers x {len(TRAIN_T)} thicknesses x {len(TRAIN_RE)} Reynolds numbers); features are",
        "t = thickness/chord, m = |max camber|/chord, L = log10(Re/1e6) (Re clamped to the fitted range).",
        "",
        "Hold-out accuracy (sections and Reynolds numbers not used in the fit):",
    ]
    lines += [f"    {k}: {v:.3f}" for k, v in stats.items()]
    lines += ['"""', "", f"LIFT_RE_FLOOR = {LIFT_RE_FLOOR!r}", "RE_MAX = 1.6e7", "RE_MIN_DRAG = 2e5", "", "FITS = {"]
    for name, s in spec.items():
        lines.append(f'    "{name}": dict(features={tuple(s["features"])!r}, coef={[round(float(c), 6) for c in s["coef"]]!r}, rms={s["rms"]:.4f}, r2={s["r2"]:.3f}),')
    lines += ["}", f"STALL_DELTA_DEG = {delta:.3f}", f"K_CD = {k_cd:.5f}", ""]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:16]))
    print("fits: " + ", ".join(f"{k} (rms {v['rms']:.3f}, R2 {v['r2']:.2f})" for k, v in spec.items()))
    print(f"STALL_DELTA_DEG={delta:.3f}  K_CD={k_cd:.5f}")
    print(f"written {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
