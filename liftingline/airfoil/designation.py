"""Parsing of NACA airfoil designations (4-digit, 5-digit, 6-series / 6A-series).

Accepted forms are deliberately tolerant of spaces, dashes, commas, parentheses, case and
Unicode subscripts, so the same grammar can read a user's typing, a library file name and a
``.dat`` header line::

    4-digit   2412        NACA 4412        0012
    5-digit   23012       NACA 23112 (reflex)
    6-series  63-412      63412      63(2)-415     632-415     63_2-415
              65(216)-415            66,2-015      64A010      65A410
              63-215 a=0.6           (mean-line parameter a, default 1.0, 0.8 for 6A)

For the 6-series the code is ``6 s [sub] - c tt``: second digit ``s`` = chordwise position of minimum
pressure (tenths), optional subscript = half-width of the low-drag range (tenths of cl), ``c`` =
design lift coefficient in tenths and ``tt`` = thickness in percent chord.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")

ACCEPTED_FORMS = "4-digit (2412), 5-digit (23012), 6-series (63-412, 63(2)-415, 65(216)-415, 64A010)"


class DesignationError(ValueError):
    """The text is not a recognised NACA designation."""


@dataclass(frozen=True)
class NacaCode:
    family: str  # "4", "5", "6" or "6A"
    thickness: float  # t/c as a fraction
    # 4-digit
    m: float = 0.0
    p: float = 0.0
    # 5-digit
    cl5: float = 0.0  # design lift coefficient (0.15 * first digit)
    p5: int = 0  # position digit (1..5)
    reflex: bool = False
    # 6-series
    series: int = 0  # 63 .. 67
    sub: int | None = None  # low-drag range subscript, when given
    a: float | None = None  # mean-line parameter, when given
    cli: float = 0.0  # design lift coefficient

    # ---------------------------------------------------------------- derived
    @property
    def is_6_series(self) -> bool:
        return self.family in ("6", "6A")

    @property
    def design_cl(self) -> float:
        return {"4": 0.0, "5": self.cl5}.get(self.family, self.cli)

    @property
    def effective_a(self) -> float:
        if self.a is not None:
            return self.a
        return 0.8 if self.family == "6A" else 1.0

    @property
    def key(self) -> tuple:
        """Canonical matching key (ignores the optional subscript and ``a``)."""
        if self.family == "4":
            return ("4", round(self.m * 100), round(self.p * 10), round(self.thickness * 100))
        if self.family == "5":
            return ("5", round(self.cl5 / 0.15), self.p5, int(self.reflex), round(self.thickness * 100))
        return ("6", self.series, self.family == "6A", round(self.cli * 10), round(self.thickness * 100))

    @property
    def name(self) -> str:
        t = round(self.thickness * 100)
        if self.family == "4":
            return f"NACA {round(self.m * 100)}{round(self.p * 10)}{t:02d}"
        if self.family == "5":
            return f"NACA {round(self.cl5 / 0.15)}{self.p5}{int(self.reflex)}{t:02d}"
        sub = f"({self.sub})" if self.sub is not None else ""
        a = f" a={self.a:g}" if self.a is not None else ""
        suffix = "A" if self.family == "6A" else ""
        return f"NACA {self.series}{suffix}{sub}-{round(self.cli * 10)}{t:02d}{a}"

    def describe(self) -> str:
        """One-line human description used for live feedback in the GUI."""
        t = f"{self.thickness * 100:.0f}% thick"
        if self.family == "4":
            if self.m == 0.0:
                return f"{self.name}: 4-digit symmetric, {t}"
            return f"{self.name}: 4-digit, camber {self.m * 100:.0f}% at {self.p * 100:.0f}% chord, {t}"
        if self.family == "5":
            kind = "reflex" if self.reflex else "standard"
            return f"{self.name}: 5-digit {kind}, design cl {self.cl5:.2f}, max camber at {self.p5 * 5}% chord, {t}"
        a = f", a={self.effective_a:g}"
        sub = f", low-drag range +/-{self.sub / 10:.1f}" if self.sub is not None else ""
        kind = "6A-series" if self.family == "6A" else "6-series"
        return f"{self.name}: {kind}, min pressure at {self.series % 10 * 10}% chord, design cl {self.cli:.1f}{sub}{a}, {t}"


# ----------------------------------------------------------------------------- grammar
_A_PARAM = re.compile(r"(?<![A-Z])A\s*=\s*(\d*\.\d+|\d)|(?<![0-9A-Z])A(\d?\.\d+)")
_SIX = [
    re.compile(r"^6([3-7])(A?)\((\d{1,3})\)-?(\d)(\d{2})$"),  # 63(2)-415, 65(216)-415
    re.compile(r"^6([3-7])(A?)(\d)-(\d)(\d{2})$"),  # 632-415
    re.compile(r"^6([3-7])(A?)-(\d)(\d{2})$"),  # 63-412, 63A-210
    re.compile(r"^6([3-7])(A?)(\d)(\d{2})$"),  # 63412, 64A010
    re.compile(r"^6([3-7])(\d)(\d)(\d{2})$"),  # 632415, 651212
]


def _clean(text: str) -> str:
    s = text.translate(_SUBSCRIPTS).upper()
    s = re.sub(r"NACA|AIRFOIL", " ", s)
    return s


def parse_naca(text: str) -> NacaCode:
    """Parse a designation; raises :class:`DesignationError` if it is not a NACA code."""
    s = _clean(text)
    a_value: float | None = None
    m_a = _A_PARAM.search(s)
    if m_a:
        a_value = float(m_a.group(1) or m_a.group(2))
        s = s[: m_a.start()] + s[m_a.end() :]
    s = re.sub(r"[\s_,]+", "", s)
    if not s:
        raise DesignationError("empty designation")

    # ------------------------------------------------------------- 6-series first
    if s.startswith("6") and len(s) >= 5:
        for i, rx in enumerate(_SIX):
            mt = rx.match(s)
            if not mt:
                continue
            g = mt.groups()
            series = 60 + int(g[0])
            if i == 0:
                is_a, sub, cl_d, tt = bool(g[1]), int(g[2][0]), int(g[3]), int(g[4])
            elif i == 1:
                is_a, sub, cl_d, tt = bool(g[1]), int(g[2]), int(g[3]), int(g[4])
            elif i == 2 or i == 3:
                is_a, sub, cl_d, tt = bool(g[1]), None, int(g[2]), int(g[3])
            else:
                is_a, sub, cl_d, tt = False, int(g[1]), int(g[2]), int(g[3])
            if not 1 <= tt <= 40:
                raise DesignationError(f"thickness {tt}% is outside 1-40%")
            return NacaCode(
                family="6A" if is_a else "6",
                thickness=tt / 100.0,
                series=series,
                sub=sub,
                a=a_value,
                cli=cl_d / 10.0,
            )
        raise DesignationError(f"{text!r} looks like a 6-series code but does not fit 63-412 / 63(2)-415 / 64A010")

    if a_value is not None:
        raise DesignationError("the a= parameter only applies to 6-series airfoils")

    # ----------------------------------------------------------------- 5-digit
    mt = re.match(r"^([1-9])(\d)([01])(\d{2})$", s)
    if mt:
        L, P, Q, tt = (int(v) for v in mt.groups())
        if not 1 <= P <= 5:
            raise DesignationError(f"5-digit position digit {P} must be 1-5")
        if Q == 1 and P < 2:
            raise DesignationError("reflex 5-digit camber lines exist for position digits 2-5 only")
        if not 1 <= tt <= 40:
            raise DesignationError(f"thickness {tt}% is outside 1-40%")
        return NacaCode(family="5", thickness=tt / 100.0, cl5=0.15 * L, p5=P, reflex=bool(Q))

    # ----------------------------------------------------------------- 4-digit
    mt = re.match(r"^(\d)(\d)(\d{2})$", s)
    if mt:
        m, p, tt = (int(v) for v in mt.groups())
        if m > 0 and p == 0:
            raise DesignationError("a cambered 4-digit airfoil needs a non-zero camber position")
        if not 1 <= tt <= 40:
            raise DesignationError(f"thickness {tt}% is outside 1-40%")
        return NacaCode(family="4", thickness=tt / 100.0, m=m / 100.0, p=p / 10.0)

    raise DesignationError(f"{text!r} is not a recognised NACA designation. Accepted: {ACCEPTED_FORMS}")


def try_parse_naca(text: str) -> NacaCode | None:
    try:
        return parse_naca(text)
    except DesignationError:
        return None


def extract_codes(text: str) -> list[NacaCode]:
    """NACA designations found in free text such as a file name or a ``.dat`` header line."""
    s = text.translate(_SUBSCRIPTS).upper()
    found: list[NacaCode] = []
    for m in re.finditer(r"NACA[\s_]*", s):
        tail = s[m.end() : m.end() + 30]
        for n in range(len(tail), 3, -1):
            code = try_parse_naca(tail[:n])
            if code is not None:
                found.append(code)
                break
    if not found:
        code = try_parse_naca(s)
        if code is not None:
            found.append(code)
    unique: list[NacaCode] = []
    for c in found:
        if c not in unique:
            unique.append(c)
    return unique
