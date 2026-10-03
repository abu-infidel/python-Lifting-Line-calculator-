import math

import numpy as np
import pytest

from liftingline.core import atmosphere as atm
from liftingline.core import units
from liftingline.core.flow import FlowCondition


# ------------------------------------------------------------------ units
@pytest.mark.parametrize("kind", [k for k in units.kinds()])
def test_unit_round_trips(kind):
    for u in units.units_of(kind):
        for v in (0.0, 1.0, 12.345, -3.0):
            si = units.to_si(kind, v, u)
            assert units.from_si(kind, si, u) == pytest.approx(v, abs=1e-12)


def test_known_conversions():
    assert units.to_si("length", 1.0, "ft") == pytest.approx(0.3048)
    assert units.to_si("speed", 100.0, "kt") == pytest.approx(51.4444444, rel=1e-7)
    assert units.to_si("speed", 36.0, "km/h") == pytest.approx(10.0)
    assert units.to_si("force", 1.0, "lbf") == pytest.approx(4.4482216, rel=1e-7)
    assert units.to_si("density", 0.002377, "slug/ft³") == pytest.approx(1.2250, rel=2e-3)
    assert units.to_si("pressure", 14.6959, "psi") == pytest.approx(101325.0, rel=1e-5)
    assert units.to_si("temperature", 15.0, "°C") == pytest.approx(288.15)
    assert units.to_si("temperature", 59.0, "°F") == pytest.approx(288.15)
    assert units.to_si("angle", 180.0, "deg") == pytest.approx(math.pi)


def test_unit_conversion_arrays_and_errors():
    arr = np.array([1.0, 2.0])
    assert units.convert("length", arr, "m", "cm") == pytest.approx([100.0, 200.0])
    with pytest.raises(ValueError):
        units.to_si("length", 1.0, "parsec")


def test_display_systems_cover_all_kinds():
    for system in units.SYSTEMS.values():
        for kind in units.kinds():
            assert system[kind] in units.units_of(kind)


# ------------------------------------------------------------- atmosphere
# 1976 US Standard Atmosphere reference values
@pytest.mark.parametrize(
    "h, T, p, rho",
    [
        (0.0, 288.15, 101325.0, 1.2250),
        (5000.0, 255.65, 54019.9, 0.736116),
        (11000.0, 216.65, 22632.06, 0.363918),
        (20000.0, 216.65, 5474.89, 0.088035),
        (32000.0, 228.65, 868.019, 0.013225),
    ],
)
def test_isa_reference_values(h, T, p, rho):
    s = atm.isa(h)
    assert s.T == pytest.approx(T, abs=1e-6)
    assert s.p == pytest.approx(p, rel=2e-5)
    assert s.rho == pytest.approx(rho, rel=1e-4)


def test_isa_sea_level_transport_properties():
    s = atm.isa(0.0)
    assert s.a == pytest.approx(340.294, abs=2e-3)
    assert s.mu == pytest.approx(1.7894e-5, rel=1e-4)


def test_isa_offset_changes_density_not_pressure():
    std, hot = atm.isa(3000.0), atm.isa(3000.0, delta_T=15.0)
    assert hot.p == pytest.approx(std.p)
    assert hot.T == pytest.approx(std.T + 15.0)
    assert hot.rho == pytest.approx(std.p / (atm.R_AIR * (std.T + 15.0)))
    assert hot.rho < std.rho


def test_isa_is_continuous_across_layer_boundaries():
    for hb in (11000.0, 20000.0, 32000.0):
        lo, hi = atm.isa(hb - 1e-6), atm.isa(hb + 1e-6)
        assert lo.p == pytest.approx(hi.p, rel=1e-9)
        assert lo.T == pytest.approx(hi.T, abs=1e-6)


def test_isa_rejects_out_of_range():
    with pytest.raises(ValueError):
        atm.isa(60000.0)
    with pytest.raises(ValueError):
        atm.isa(0.0, delta_T=-400.0)


# ------------------------------------------------------------------- flow
def test_flow_from_isa_speed_and_mach():
    f = FlowCondition.from_isa(0.0, V=50.0, alpha_deg=3.0)
    assert f.q == pytest.approx(0.5 * 1.225 * 2500.0, rel=1e-4)
    assert f.mach == pytest.approx(50.0 / 340.294, rel=1e-4)
    assert f.reynolds(1.0) == pytest.approx(f.rho * 50.0 / f.mu)
    g = FlowCondition.from_isa(0.0, mach=0.1)
    assert g.V == pytest.approx(0.1 * g.a)


def test_flow_requires_exactly_one_speed_spec():
    with pytest.raises(ValueError):
        FlowCondition.from_isa(0.0)
    with pytest.raises(ValueError):
        FlowCondition.from_isa(0.0, V=10.0, mach=0.1)


def test_flow_manual_derives_missing_properties():
    f = FlowCondition.from_manual(rho=1.1, V=30.0, T=300.0)
    assert f.a == pytest.approx(atm.speed_of_sound(300.0))
    assert f.mu == pytest.approx(atm.sutherland_viscosity(300.0))
    g = FlowCondition.from_manual(rho=1.1, mach=0.2, a=340.0, mu=1.8e-5)
    assert g.V == pytest.approx(68.0) and g.mu == 1.8e-5


@pytest.mark.parametrize("kwargs", [dict(V=-1.0), dict(rho=0.0), dict(mu=-1e-5)])
def test_flow_rejects_non_physical_values(kwargs):
    base = dict(V=10.0, rho=1.2, mu=1.8e-5, a=340.0)
    base.update(kwargs)
    with pytest.raises(ValueError):
        FlowCondition(**base)
