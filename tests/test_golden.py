"""The calculator reproduces the results recorded before the scenario / CLI / MCP additions (exactly)."""

import json
from pathlib import Path

import pytest

GOLDEN = json.loads((Path(__file__).parent / "data" / "golden_results.json").read_text())


def test_every_example_and_sweep_is_unchanged():
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from make_golden import collect

    now = collect()
    for group in ("examples", "high_lift"):
        assert set(now[group]) == set(GOLDEN[group])
        for name, ref in GOLDEN[group].items():
            for key, value in ref.items():
                assert now[group][name][key] == pytest.approx(value, rel=1e-12, abs=1e-12), (group, name, key)
    for name, ref in GOLDEN["sweeps"].items():
        assert now["sweeps"][name] == pytest.approx(ref, rel=1e-12, abs=1e-12)
