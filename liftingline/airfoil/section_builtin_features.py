"""Regression features shared by the calibration tool and the built-in engine.

``t`` = thickness/chord, ``m`` = |max camber|/chord, ``L`` = log10(Re / 1e6).
"""

from __future__ import annotations

FEATURES = {
    "1": lambda t, m, L: 1.0 + 0.0 * t,
    "t": lambda t, m, L: t,
    "t2": lambda t, m, L: t * t,
    "m": lambda t, m, L: m,
    "L": lambda t, m, L: L,
    "tL": lambda t, m, L: t * L,
    "L2": lambda t, m, L: L * L,
    "tm": lambda t, m, L: t * m,
}
