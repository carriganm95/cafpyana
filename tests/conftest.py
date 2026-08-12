"""Shared pytest setup.

The repo isn't installed as a package (imports rely on the repo root being on
``sys.path``, the same convention used throughout ``analysis_village/*``'s own
``sys.path.append`` hacks), so add it here once for every test module. Also
force a non-interactive matplotlib backend before any test imports pyplot,
since the plotting helpers under test call ``plt.savefig``/``plt.show`` and
CI has no display.
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import matplotlib
matplotlib.use("Agg")
