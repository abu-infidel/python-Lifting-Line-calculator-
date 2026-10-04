"""Write the example wings (liftingline.io.presets) as project files into ./examples.

    python tools/make_examples.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from liftingline.io.presets import PRESETS  # noqa: E402


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "examples"
    out.mkdir(exist_ok=True)
    for name, make in PRESETS.items():
        path = out / f"{slug(name)}.json"
        make().save(path)
        print("wrote", path.relative_to(out.parent))


if __name__ == "__main__":
    main()
