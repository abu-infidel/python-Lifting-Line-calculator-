"""International Standard Atmosphere (ISA) with Sutherland viscosity.

Altitude is *geopotential* altitude in metres, valid from -2 km to 47 km. An ISA
temperature offset ``delta_T`` (ISA+x) is applied at constant pressure altitude,
the usual convention for off-standard days.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

G0 = 9.80665  # m/s^2
R_AIR = 287.05287  # J/(kg K), dry air
GAMMA = 1.4
T0 = 288.15  # K
P0 = 101325.0  # Pa

_SUTHERLAND_BETA = 1.458e-6  # kg/(m s sqrt(K))
_SUTHERLAND_S = 110.4  # K

H_MIN = -2000.0
H_MAX = 47000.0

# (base geopotential altitude [m], base temperature [K], lapse rate [K/m])
_LAYERS = (
    (0.0, 288.15, -0.0065),
    (11000.0, 216.65, 0.0),
    (20000.0, 216.65, 0.001),
    (32000.0, 228.65, 0.0028),
    (47000.0, 270.65, 0.0),
)


def _build_base_pressures() -> tuple[float, ...]:
    pressures = [P0]
    for (h0, t_base, lapse), (h1, _, _) in zip(_LAYERS[:-1], _LAYERS[1:]):
        if lapse != 0.0:
            t1 = t_base + lapse * (h1 - h0)
            pressures.append(pressures[-1] * (t1 / t_base) ** (-G0 / (R_AIR * lapse)))
        else:
            pressures.append(pressures[-1] * math.exp(-G0 * (h1 - h0) / (R_AIR * t_base)))
    return tuple(pressures)


_BASE_PRESSURES = _build_base_pressures()


@dataclass(frozen=True)
class Atmosphere:
    h: float  # geopotential altitude, m
    delta_T: float  # ISA temperature offset, K
    T: float  # static temperature, K
    p: float  # static pressure, Pa
    rho: float  # density, kg/m^3
    mu: float  # dynamic viscosity, Pa s
    a: float  # speed of sound, m/s


def sutherland_viscosity(T: float) -> float:
    """Dynamic viscosity of air from Sutherland's law, Pa s."""
    return _SUTHERLAND_BETA * T**1.5 / (T + _SUTHERLAND_S)


def speed_of_sound(T: float) -> float:
    return math.sqrt(GAMMA * R_AIR * T)


def temperature_from_speed_of_sound(a: float) -> float:
    return a * a / (GAMMA * R_AIR)


def isa(h: float, delta_T: float = 0.0) -> Atmosphere:
    """Standard-atmosphere state at geopotential altitude ``h`` (m), ISA + ``delta_T`` K."""
    if not (H_MIN <= h <= H_MAX):
        raise ValueError(f"altitude {h:.0f} m outside the supported ISA range [{H_MIN:.0f}, {H_MAX:.0f}] m")
    idx = 0
    for i in range(len(_LAYERS) - 1):
        if h >= _LAYERS[i][0]:
            idx = i
    h_base, t_base, lapse = _LAYERS[idx]
    p_base = _BASE_PRESSURES[idx]
    t_isa = t_base + lapse * (h - h_base)
    if lapse != 0.0:
        p = p_base * (t_isa / t_base) ** (-G0 / (R_AIR * lapse))
    else:
        p = p_base * math.exp(-G0 * (h - h_base) / (R_AIR * t_base))
    T = t_isa + delta_T
    if T <= 0.0:
        raise ValueError("temperature offset gives a non-positive absolute temperature")
    rho = p / (R_AIR * T)
    return Atmosphere(
        h=h,
        delta_T=delta_T,
        T=T,
        p=p,
        rho=rho,
        mu=sutherland_viscosity(T),
        a=speed_of_sound(T),
    )
