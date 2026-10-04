"""The built-in validation suite itself."""

from liftingline.__main__ import main as cli_main
from liftingline.core import atmosphere
from liftingline.core.validate import FAIL, PASS, SKIP, format_report, run_validation


def test_the_suite_passes_and_covers_every_group():
    checks = run_validation()
    failed = [c for c in checks if c.status == FAIL]
    assert not failed, "\n".join(f"{c.group}: {c.name}: expected {c.expected} got {c.got} {c.note}" for c in failed)
    assert len(checks) >= 65 and sum(1 for c in checks if c.status == PASS) >= 60
    assert len({c.group for c in checks}) == 7
    assert all(c.status in (PASS, SKIP) for c in checks)


def test_the_report_text_lists_groups_and_the_verdict():
    checks = run_validation()
    text = format_report(checks, 1.5)
    assert "ALL CHECKS PASSED" in text and "Classical lifting line" in text and "in 1.5 s" in text


def test_a_broken_atmosphere_is_caught(monkeypatch):
    real = atmosphere.isa

    def broken(h, delta_T=0.0):
        s = real(h, delta_T)
        return atmosphere.Atmosphere(s.h, s.delta_T, s.T, s.p, 1.1 * s.rho, s.mu, s.a)

    monkeypatch.setattr(atmosphere, "isa", broken)
    checks = run_validation()
    bad = [c for c in checks if c.status == FAIL]
    assert any("ISA sea level: density" in c.name for c in bad)
    assert "SOME CHECKS FAILED" in format_report(checks)


def test_command_line_entry_point(capsys):
    assert cli_main(["--validate"]) == 0
    out = capsys.readouterr().out
    assert "self-test" in out and "ALL CHECKS PASSED" in out
    assert cli_main(["--version"]) == 0 and capsys.readouterr().out.startswith("liftingline ")


def test_a_crashing_group_becomes_a_failure_not_an_abort(monkeypatch):
    from liftingline.core import validate

    def boom(c):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(validate, "GROUPS", (("x", boom),))
    checks = run_validation()
    assert len(checks) == 1 and checks[0].status == FAIL and "kaboom" in checks[0].note
