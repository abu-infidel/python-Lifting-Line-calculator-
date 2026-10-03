#!/usr/bin/env python3
"""Check an airfoil library folder (for example the UIUC coordinate database).

    python tools/check_library.py /path/to/coord_seligFmt
    python tools/check_library.py /path/to/folder --resolve 63-412 "64A010" 65-212
    python tools/check_library.py /path/to/folder --list-naca6 --report report.txt

It indexes the folder, parses every file, and prints

* how many files parsed, which coordinate formats were found, how many needed repairs,
* every file that failed (and why),
* for NACA 6-series designations: which file each one resolves to (or "missing"/"ambiguous"),
  so naming differences in your collection are visible.

If something looks wrong, send the printed report back so the name matching can be adjusted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liftingline.airfoil.designation import DesignationError, parse_naca  # noqa: E402
from liftingline.airfoil.library import AirfoilLibrary  # noqa: E402

COMMON_SIX_SERIES = (
    "63-006 63-009 63-010 63-012 63-015 63-018 63-206 63-209 63-210 63-212 63-215 63-218 63-412 63-415 63-418 "
    "63-421 63-615 63-618 631-012 632-215 633-018 634-021 64-006 64-009 64-010 64-012 64-206 64-209 64-210 64-212 "
    "64-215 64-218 64-221 64-412 64-415 64-418 64-421 64-612 641-012 642-015 65-006 65-009 65-010 65-012 65-015 "
    "65-206 65-209 65-210 65-212 65-215 65-218 65-221 65-410 65-412 65-415 65-418 66-006 66-009 66-012 66-015 "
    "66-018 66-206 66-209 66-210 66-212 66-215 67-015 63A010 63A210 63A410 64A010 64A210 64A410 65A006 65A008 "
    "65A010 65A210 65A410"
).split()


def resolve_report(lib: AirfoilLibrary, designations) -> list[str]:
    lines = []
    for text in designations:
        try:
            code = parse_naca(text)
        except DesignationError as exc:
            lines.append(f"  {text:12s} -> not a valid designation ({exc})")
            continue
        entry, ties = lib.resolve_best(code)
        if entry is None:
            lines.append(f"  {text:12s} -> missing in this library")
        elif len(ties) > 1:
            lines.append(f"  {text:12s} -> AMBIGUOUS: " + ", ".join(e.path.name for e in ties))
        else:
            lines.append(f"  {text:12s} -> {entry.path.name}")
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="folder with .dat files (searched recursively)")
    ap.add_argument("--resolve", nargs="+", metavar="CODE", help="NACA designations to resolve")
    ap.add_argument("--list-naca6", action="store_true", help="list every file recognised as a 6-series airfoil")
    ap.add_argument("--report", metavar="FILE", help="also write the report to this file")
    args = ap.parse_args(argv)

    out: list[str] = []
    try:
        lib = AirfoilLibrary.scan(args.folder)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2
    out.append(f"Library: {args.folder}")
    out.append(lib.health_report().to_text())

    if args.list_naca6:
        out.append("\n6-series files recognised (file -> parsed designation):")
        for e in lib:
            for c in e.codes:
                if c.is_6_series:
                    out.append(f"  {e.path.name:28s} -> {c.name}  (key {c.key})")
    designations = args.resolve or COMMON_SIX_SERIES
    out.append("\nNACA designation -> library file" + ("" if args.resolve else " (common 6-series sections)"))
    out += resolve_report(lib, designations)

    text = "\n".join(out)
    print(text)
    if args.report:
        Path(args.report).write_text(text + "\n", encoding="utf-8")
        print(f"\nreport written to {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
