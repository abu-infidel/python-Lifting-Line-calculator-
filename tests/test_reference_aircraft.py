"""The program against real aircraft: Cessna 172S / 152 handbook stall speeds and level-flight cruise."""

import pytest

from liftingline.io.backend import Backend
from liftingline.io.presets import CESSNA_172S, PRESETS
from liftingline.io.reference_aircraft import compare, format_comparison
from liftingline.core.analysis import run


@pytest.fixture(scope="module")
def comparison():
    return compare(Backend(), "builtin")


def _row(c, aircraft, point):
    return next(r for r in c.rows if r.aircraft.startswith(aircraft) and r.point.startswith(point))


def test_cessna_172s_clean_stall_matches_the_handbook(comparison):
    r = _row(comparison, "Cessna 172S", "stall, flaps up")
    assert 0.95 <= r.ratio <= 1.10  # smooth wing alone vs whole airplane: a few per cent above


def test_flaps_and_the_152_come_out_above_the_airplane_as_expected(comparison):
    for ac, pt, hi in (("Cessna 172S", "stall, flaps 30", 1.25), ("Cessna 152", "stall, flaps up", 1.25),
                       ("Cessna 152", "stall, flaps 30", 1.35)):
        assert 1.0 <= _row(comparison, ac, pt).ratio <= hi


def test_level_flight_needs_a_small_positive_angle_like_a_real_172(comparison):
    cruise = _row(comparison, "Cessna 172S", "cruise")
    top = _row(comparison, "Cessna 172S", "maximum speed")
    a_cruise = float(cruise.program.split("level at ")[1].split(" deg")[0])
    a_top = float(top.program.split("level at ")[1].split(" deg")[0])
    assert 0.3 <= a_cruise <= 2.5 and -1.0 <= a_top <= 1.0 and a_top < a_cruise
    assert "Cessna 172S POH" in format_comparison(comparison)


def test_the_published_172s_example_and_carries_its_weight_in_cruise():
    st = PRESETS[CESSNA_172S]()
    st.solver.engine = "builtin"
    wing = st.build_wing()
    assert wing.area == pytest.approx(174 * 0.09290304, rel=1e-6) and wing.incidence_deg == 1.5
    lift = run(Backend().build_request(st)).totals.L
    assert lift == pytest.approx(st.high_lift.weight_N, rel=0.05)  # 2550 lb at 124 KTAS, 8500 ft
