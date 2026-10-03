"""Command-line entry point:  ``python -m liftingline``  (or ``python main.py``)."""

from __future__ import annotations

import argparse
import sys

from . import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="liftingline",
        description="Lifting-line wing calculator (Prandtl LLT). Launches the GUI by default.",
    )
    parser.add_argument("--version", action="store_true", help="print the version and exit")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="run the built-in validation suite (analytic benchmarks) and exit",
    )
    args = parser.parse_args(argv)

    if args.version:
        print(f"liftingline {__version__}")
        return 0
    if args.validate:
        from .core.validate import main as validate_main

        return validate_main()

    from .gui.app import run_gui

    return run_gui()


if __name__ == "__main__":
    sys.exit(main())
