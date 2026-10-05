"""Launcher:  ``python stl_exporter.py [project.json]``  starts the Wing 3-D Model Exporter (a separate app).

``python stl_exporter.py project.json -o wing.stl`` exports without a window; ``--help`` lists the options.
"""

from liftingline.stl_app import main

if __name__ == "__main__":
    raise SystemExit(main())
