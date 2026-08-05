"""Centralized input-file paths for the nueNp0Pi analysis.

Single source of truth for absolute paths to external input files this
analysis depends on, instead of hardcoding personal paths as default
arguments scattered across multiple modules (this repo previously had the
same flux-file path hardcoded in both ``utils.py`` and a dead commented-out
block in ``constants.py``).

Every path here can be overridden with a matching environment variable, so
the analysis is portable across users/machines without editing this file.

As more absolute paths get pulled out of ``files_config.py`` and elsewhere,
they belong here too.
"""

import os

# SBND numu flux histogram (consumed by makedf.flux.get_integrated_flux / get_xsec_unit).
FLUX_FILE = os.environ.get(
    "NUENP0PI_FLUX_FILE",
    "/exp/sbnd/data/users/munjung/flux/sbnd_original_flux.root",
)
