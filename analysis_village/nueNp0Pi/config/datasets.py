"""Datasets config: file locations, output paths, and small standalone
constants for the nueNp0Pi analysis -- everything that isn't a plot variable
(:mod:`analysis_village.nueNp0Pi.config.plots`) or a selection cut / truth
category (:mod:`analysis_village.nueNp0Pi.config.settings`).

This holds ONLY settings (glob patterns, env-var-backed roots, small
constants) -- no path-computation LOGIC. The functions that build on these
settings (``iter_*_df_paths``, ``default_*_work_root``, ``sorted_glob``,
``summary_lines``/``print_summary``) live in
:mod:`analysis_village.nueNp0Pi.dataset_paths`, which imports its settings
from here. Run ``python -m analysis_village.nueNp0Pi.dataset_paths`` for the
human-readable summary that used to live at the bottom of this file.

NOTE: many of the internal names below (``NUMUCC_*`` env vars, docstring
references to "numu CC 1p0pi workflows") are historical -- this module was
originally copied from the numucc_1p0pi analysis and adapted for nueNp0Pi.
The env var names are left as-is here to avoid breaking anyone's existing
shell/grid-submission setup that already sets them; a full rename is a
separate, lower-risk-tolerance cleanup.

Edit paths **here only** so drivers stay thin:

- ``analysis_village.nueNp0Pi.event_selection.build_event_selection_pipeline`` — survey/bin-pack
  (≤1 GiB job groups of ``.df`` files)/map/aggregate, in-process via
  ``pyanalib.event_selection_pipeline.EventSelectionPipeline``.
- ``run_syst_multisim_chunked.sh`` — per systematic + CAF shard (default syst subset: Flux+G4;
  MCstat opt-in via ``MULTISIM_SYST_TYPES`` / ``--syst-types``); **Combined** chunks live under
  ``multisim_syst-chunked-*`` / ``chunks/Combined/``; **MCstat**, **Flux**, and **G4** map shards use
  parallel ``mcstat_syst-chunked-*``, ``flux_syst-chunked-*``, and ``g4_syst-chunked-*`` (see
  :func:`analysis_village.nueNp0Pi.dataset_paths.default_mcstat_syst_work_root`,
  :func:`analysis_village.nueNp0Pi.dataset_paths.default_flux_syst_work_root`,
  :func:`analysis_village.nueNp0Pi.dataset_paths.default_g4_syst_work_root`).
- ``syst_multisim_aggregate.py`` — merge neutrino multisim map outputs into ``MCstat/``, ``Flux/``, ``G4/``.
- ``syst_detvar_chunk.py`` / ``syst_detvar_aggregate.py`` — WireMod + calo variants;
  input globs are listed in ``DETVAR_WIREMOD_GLOBS`` /
  :func:`analysis_village.nueNp0Pi.dataset_paths.iter_detvar_chunk_jobs`.
- ``get_systematics_genie.py`` / ``run_syst_genie_chunked.sh`` — GENIE reweights: one glob per
  knob **group** (``GENIE_GROUP_GLOBS``) /
  :func:`analysis_village.nueNp0Pi.dataset_paths.iter_genie_chunk_map_tasks`;
  ``syst_genie_aggregate.py`` publishes ``GENIE/cov_mat_dict.pkl`` on the syst disk (phase 3 of
  the shell driver).
- ``run_syst_cosmics_chunked.sh`` — ``syst_cosmics_chunk.py`` / ``syst_cosmics_aggregate.py``;
  globs reuse ``EVENT_SELECTION_GLOBS`` ``offbeam`` / ``intime``.
- :func:`analysis_village.nueNp0Pi.dataset_paths.default_syst_disk_root` — unified
  ``syst_disk_layout`` root (``Cosmics/``, ``MCstat/``, ``Detector/``, …) used by the
  ``run_syst_*`` drivers unless ``SYST_DISK_ROOT`` is set or a script passes an explicit
  override.

**Naming:** *HDF splits*, *map shards* (one ``.df`` file), and *exposure batches* (time-ordered
data slices for staged access) are different concepts — see ``exposure_access``.

Relative globs are resolved from ``SPRING_GEN1_ROOT``. Override any constant by
setting environment variables of the same name before importing (advanced).

After producers fill the syst disk tree (``syst_disk_layout``), point ``utils.get_syst_unc`` /
``event_selection_aggregate.py`` at the **root** via ``SYST_DISK_ROOT`` or
``--syst-disk-root`` (subfolders ``MCstat``, ``Flux``, ``G4``, ``GENIE``, ``Cosmics``,
``Detector``). Loaders require every expected file; there are no alternate search paths.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from makedf.geniesyst import *

# -----------------------------------------------------------------------------
# Flux input (see makedf.flux.get_integrated_flux / get_xsec_unit).
# Folded in from the analysis's former paths_config.py -- this is now the one
# place absolute input-file paths for nueNp0Pi live.
# -----------------------------------------------------------------------------
FLUX_FILE = os.environ.get(
    "NUENP0PI_FLUX_FILE",
    "/exp/sbnd/data/users/munjung/flux/sbnd_original_flux.root",
)

# -----------------------------------------------------------------------------
# Base release (Spring Gen 1 — edit for other campaigns)
# -----------------------------------------------------------------------------
SPRING_GEN1_ROOT = Path(
    os.environ.get(
        "NUMUCC_SPRING_GEN1_ROOT",
        "/exp/sbnd/data/users/micarrig/nueNp0Pi/selection_v2/",
        # "/Users/micarrig/Desktop/SBND/data/",
    )
)

SPRING_GEN1_ROOT_EAF = Path(
    os.environ.get(
        "NUMUCC_SPRING_GEN1_ROOT",
        "/scratch/7DayLifetime/micarrig/xsec",
    )
)

PLOTS_BASE = Path(
    os.environ.get(
        "NUMUCC_PLOTS_BASE",
        "/nashome/m/micarrig/sbnd/nueCCNp/plots/",
        # "/Users/micarrig/Desktop/SBND/cafpyana/plots/",

    )
)

# -----------------------------------------------------------------------------
# DFs with all slices, for the event selection batched pipeline
# (analysis_village.nueNp0Pi.event_selection.build_event_selection_pipeline)
# Keys: mc, data, intime, offbeam, dirt
# -----------------------------------------------------------------------------
EVENT_SELECTION_GLOBS: Dict[str, str] = {
    # "mc": str(SPRING_GEN1_ROOT / "2026_05_11_041007__sel_all-mc-BNB_cosmics/*df"),
    #"mc": str(SPRING_GEN1_ROOT / "2026_05_11_183347__sel_all-mc-BNB_cosmics-EField_R00/*df"),
    # NOTE: this used to be joined as an ALREADY-ABSOLUTE path
    # (SPRING_GEN1_ROOT / "/pnfs/..."), which pathlib silently resolves to just the
    # absolute right-hand side -- so overriding NUMUCC_SPRING_GEN1_ROOT had no effect
    # on this glob. Made relative so it actually composes with SPRING_GEN1_ROOT.
    "mc": str(SPRING_GEN1_ROOT / "*.df"),
    # "data": str(SPRING_GEN1_ROOT / "2026_05_16_230859__sel_all-data-1e20/*.df"),
    # "intime": str(SPRING_GEN1_ROOT / "2026_05_11_040132__sel_all-mc-Intime/*.df"),
    # "offbeam": str(SPRING_GEN1_ROOT / "2026_05_11_035756__sel_all-data-OffBeamLight/*.df"),
    # "dirt": str(SPRING_GEN1_ROOT / "2026_05_11_040638__sel_all-mc-dirt/*.df"),
}

# -----------------------------------------------------------------------------
# Selected-events (final selection) bundles — used by selected_events*.py
# Keys: mc, data, intime, dirt
#
# These are produced by the "updated workflow" which writes timestamped directories
# like: 2026_05_01_102310__sel_mup-data-BNB_cosmics/sel_mup-data-BNB_cosmics_*.df
# -----------------------------------------------------------------------------
#
# By default we target the ``sel_mup`` campaign; override via:
#   export NUMUCC_SELECTED_EVENTS_TAG=sel_2prong
# (or any other selection tag that matches the directory naming convention).
SELECTED_EVENTS_TAG = os.environ.get("NUMUCC_SELECTED_EVENTS_TAG", "sel_mup")

SELECTED_EVENTS_GLOBS: Dict[str, str] = {
    "mc": str(SPRING_GEN1_ROOT / f"*__{SELECTED_EVENTS_TAG}-mc-BNB_cosmics/*.df"),
    "data": str(SPRING_GEN1_ROOT / f"*__{SELECTED_EVENTS_TAG}-data-Gen1/*.df"),
    "intime": str(SPRING_GEN1_ROOT / f"*__{SELECTED_EVENTS_TAG}-mc-Intime/*.df"),
    "offbeam": str(SPRING_GEN1_ROOT / f"*__{SELECTED_EVENTS_TAG}-data-OffBeamLight/*.df"),
    "dirt": str(SPRING_GEN1_ROOT / f"*__{SELECTED_EVENTS_TAG}-mc-dirt/*.df"),
}

# -----------------------------------------------------------------------------
# Multisim syst chunk inputs — **one glob per systematic** (often different dirs).
# Keys must match NEUTRINO_SYST_ORDER below: MCstat, Flux, G4.
# Defaults repeat the same patterns as the old single-glob workflow; edit per-syst paths here.
# ``final``: tight-selection-style bundles; ``sel_all``: loose + wgts.
# -----------------------------------------------------------------------------
# Own copy, not imported from analysis_village.numucc_1p0pi (that module doesn't
# exist in this repo -- the import was a copy-paste leftover from adapting this
# file from numucc_1p0pi; iter_multisim_syst_df_paths/iter_multisim_chunk_tasks
# in dataset_paths.py used to reach into it and would have raised ModuleNotFoundError
# the moment either was actually called).
NEUTRINO_SYST_ORDER: Tuple[str, ...] = ("MCstat", "Flux", "G4")
MULTISIM_SYST_GLOBS_FINAL: Dict[str, str] = {
    "MCstat": str(SPRING_GEN1_ROOT / "2026_05_18_145611__sel_mup-wgts_mcstat/merged_perTPC/*.df"),
    "Flux": str(SPRING_GEN1_ROOT / "2026_05_11_155745__sel_mup-wgts_flux/merged_perTPC/*.df"),
    # "Flux": str(SPRING_GEN1_ROOT_EAF / "2026_05_11_155745__sel_mup-wgts_flux/*.df"),
    "G4": str(SPRING_GEN1_ROOT / "2026_05_11_031351__sel_mup-wgts_g4/merged_perTPC/*.df"),
    # "G4": str(SPRING_GEN1_ROOT_EAF / "2026_05_11_031351__sel_mup-wgts_g4/*.df"),
    # "GENIE": str(SPRING_GEN1_ROOT / "/pnfs/sbnd/scratch/users/munjung/cafpyana_out/dfs/2026_05_23_103124__sel_mup-wgts_genie_slim/*df"),
    # "GENIE": str(SPRING_GEN1_ROOT / "/pnfs/sbnd/scratch/users/munjung/cafpyana_out/dfs/2026_05_11_024530__sel_mup-wgts_genie_CCQE/*df"),
    "GENIE": str(SPRING_GEN1_ROOT / f"2026_05_23_235202__sel_mup-wgts_genie_slim/perTPC/*.df"),
}
MULTISIM_SYST_GLOBS_SEL_ALL: Dict[str, str] = {
    # "MCstat": str(SPRING_GEN1_ROOT / "2026_05_11_084007__sel_mup-wgts_mcstat/*.df"),
    # "Flux": str(SPRING_GEN1_ROOT / "2026_05_11_031846__sel_mup-wgts_flux/*.df"),
    # "G4": str(SPRING_GEN1_ROOT / "2026_05_11_031351__sel_mup-wgts_g4/*.df"),
}

# Legacy single-glob names (union of per-syst globs); kept for scripts that need one pattern string.
# MULTISIM_MC_GLOB_FINAL = MULTISIM_SYST_GLOBS_FINAL["Flux"]
# MULTISIM_MC_GLOB_SEL_ALL = MULTISIM_SYST_GLOBS_SEL_ALL["Flux"]

# -----------------------------------------------------------------------------
# Detvar (WireMod + calo unisim) — typical chunk input dirs / globs
# -----------------------------------------------------------------------------
DETVAR_DF_GLOB_EXAMPLE_WIREMOD_YZ = str(
    SPRING_GEN1_ROOT / "2026_05_09_223419__sel_2prong-mc-BNB_cosmics-WireModYZ/*.df"
)
DETVAR_DF_GLOB_EXAMPLE_WIREMOD_XTXW = str(
    SPRING_GEN1_ROOT / "2026_05_11_103733__sel_2prong-mc-BNB_cosmics-WireModXTXW/*df"
)

# -----------------------------------------------------------------------------
# DetVar chunked driver — ``(WireMod tag, glob)`` pairs for ``syst_detvar_chunk.py``
# -----------------------------------------------------------------------------
DETVAR_WIREMOD_GLOBS: List[Tuple[str, str]] = [
    ("wiremod_yz", DETVAR_DF_GLOB_EXAMPLE_WIREMOD_YZ),
    ("wiremod_xtxw", DETVAR_DF_GLOB_EXAMPLE_WIREMOD_XTXW),
]

# -----------------------------------------------------------------------------
# GENIE reweight samples — one glob per **group** (same layout as monolithic drivers).
# CCQE lives under ``genie_wgts-CCQE`` with ``*_geniewgts_CCQE.df`` filenames; other
# groups use ``genie_wgts-<Tag>/*.df``.
# -----------------------------------------------------------------------------
GENIE_GROUP_ORDER: Tuple[str, ...] = ("CCQE", "MEC", "RES", "nonRES", "DIS", "Other", "Ar23p")

# Final-selection-style GENIE weight bundles (same convention as ``MULTISIM_SYST_GLOBS_FINAL``).
GENIE_GROUP_GLOBS: Dict[str, str] = {
    "CCQE": str(SPRING_GEN1_ROOT / "2026_05_11_024530__sel_mup-wgts_genie_CCQE/merged_perTPC/*.df"),
    "MEC": str(SPRING_GEN1_ROOT / "2026_05_11_030314__sel_mup-wgts_genie_MEC/merged_perTPC/*.df"),
    "RES": str(SPRING_GEN1_ROOT / "2026_05_11_030547__sel_mup-wgts_genie_RES/merged_perTPC/*.df"),
    "nonRES": str(SPRING_GEN1_ROOT / "2026_05_11_030906__sel_mup-wgts_genie_nonRES/merged_perTPC/*.df"),
    "DIS": str(SPRING_GEN1_ROOT / "2026_05_11_031206__sel_mup-wgts_genie_DIS/merged_perTPC/*.df"),
    "Other": str(SPRING_GEN1_ROOT / "2026_05_11_031520__sel_mup-wgts_genie_Other/merged_perTPC/*.df"),
    "Ar23p": str(SPRING_GEN1_ROOT / "2026_05_12_010953__sel_mup-wgts_genie_Ar23p/merged_perTPC/*.df"),
}

# Loose ``sel_all``-style MC + GENIE weights (evt / trk / hdr / mcnu). Fill when running
# ``get_systematics_genie.py chunk-map --input-stage sel_all``; empty groups are skipped.
GENIE_GROUP_GLOBS_SEL_ALL: Dict[str, str] = {}


GENIE_GROUP_KNOBS: Dict[str, List[str]] = dict(
    zip(
        GENIE_GROUP_ORDER,
        [
            list(qe_genie_systematics),
            list(mec_genie_systematics),
            list(res_genie_systematics),
            list(nonres_genie_systematics),
            list(dis_genie_systematics),
            list(other_genie_systematics),
            list(ar23p_genie_systematics),
        ],
    )
)


# ===========================================================================
# Small standalone constants (formerly analysis_village/nueNp0Pi/constants.py).
# ===========================================================================
DETECTOR = "SBND_nohighyz"
EPSILON = 1e-6  # for clipping distributions at bin ranges

# ===========================================================================
# Output/loading config (formerly analysis_village/nueNp0Pi/files_config.py).
# The rest of that file (``get_ana_dfs``) was a second, disagreeing dataset-
# path registry never called by the live batched workflow (which sources
# paths from EVENT_SELECTION_GLOBS above, via event_selection.py) --
# not carried over. Only the two names other live modules actually import:
# ===========================================================================
save_fig_base_dir = "/exp/sbnd/data/users/micarrig/plots/nueNp0Pi"
KEYS2LOAD = ["evt", "trk", "hdr"]
