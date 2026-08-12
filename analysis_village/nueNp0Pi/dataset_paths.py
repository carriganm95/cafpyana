"""nueNp0Pi's dataset-path wiring: binds the generic path-resolution helpers in
:mod:`pyanalib.dataset_paths` to this analysis's glob dicts / env-var names /
work-base path (``xsec/numucc_1p0pi`` -- historical, see
:mod:`analysis_village.nueNp0Pi.config.datasets` docstring; left as-is here too
so existing shell/grid env vars and output paths keep working).

The raw glob patterns / env-var-backed roots these functions consume live in
:mod:`analysis_village.nueNp0Pi.config.datasets` (settings only, no logic) --
this file only has to be read to see how nueNp0Pi's config plugs into the
generic ``pyanalib.dataset_paths`` machinery, not to find what's configurable.

Run this module directly (``python -m analysis_village.nueNp0Pi.dataset_paths``)
for a human-readable summary of every configured dataset/glob and how many
files each currently resolves to.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Sequence, Tuple

from pyanalib.syst_disk_layout import SYST_DISK_ENV
from pyanalib import dataset_paths as _pd

from analysis_village.nueNp0Pi.config.datasets import (
    SPRING_GEN1_ROOT,
    PLOTS_BASE,
    EVENT_SELECTION_GLOBS,
    SELECTED_EVENTS_GLOBS,
    MULTISIM_SYST_GLOBS_FINAL,
    MULTISIM_SYST_GLOBS_SEL_ALL,
    DETVAR_WIREMOD_GLOBS,
    GENIE_GROUP_ORDER,
    GENIE_GROUP_GLOBS,
    GENIE_GROUP_GLOBS_SEL_ALL,
    NEUTRINO_SYST_ORDER,
)

# Work-base path segment for every default_*_work_root() below. Historical
# name (see module docstring) -- kept as-is to avoid breaking existing
# shell/grid setups that already point at "xsec/numucc_1p0pi" output trees.
_WORK_BASE = "xsec/numucc_1p0pi"

# Re-exported so callers that used to import ``sorted_glob`` from this module
# keep working unchanged.
sorted_glob = _pd.sorted_glob


# -----------------------------------------------------------------------------
# Glob helpers
# -----------------------------------------------------------------------------
def iter_event_selection_df_paths(sample: str) -> Iterator[str]:
    yield from _pd.iter_keyed_df_paths(sample, EVENT_SELECTION_GLOBS)


def iter_detvar_chunk_jobs(
    wiremod_globs: Optional[Sequence[Tuple[str, str]]] = None,
) -> Iterator[Tuple[str, str]]:
    """Yield ``(wiremod_tag, df_path)`` for ``syst_detvar_chunk.py`` (paths sorted per glob)."""
    pairs = wiremod_globs if wiremod_globs is not None else DETVAR_WIREMOD_GLOBS
    yield from _pd.iter_tagged_chunk_jobs(pairs)


def _genie_glob_map(mc_df_stage: str) -> Dict[str, str]:
    if mc_df_stage == "final":
        return GENIE_GROUP_GLOBS
    if mc_df_stage == "sel_all":
        return GENIE_GROUP_GLOBS_SEL_ALL
    raise ValueError("mc_df_stage must be 'final' or 'sel_all', got %r" % mc_df_stage)


def iter_genie_group_df_paths(
    genie_group: str,
    group_globs: Optional[Dict[str, str]] = None,
    *,
    mc_df_stage: str = "final",
) -> Iterator[str]:
    """Yield sorted ``.df`` paths for one GENIE knob group.

    ``mc_df_stage`` selects :data:`GENIE_GROUP_GLOBS` vs :data:`GENIE_GROUP_GLOBS_SEL_ALL`
    when ``group_globs`` is omitted.
    """
    gmap = group_globs if group_globs is not None else _genie_glob_map(mc_df_stage)
    yield from _pd.iter_keyed_df_paths(genie_group, gmap)


def iter_genie_chunk_map_tasks(
    group_globs: Optional[Dict[str, str]] = None,
    mc_df_stage: str = "final",
) -> Iterator[Tuple[str, str]]:
    """Yield ``(genie_group_tag, df_path)`` for ``get_systematics_genie.py chunk-map``.

    ``mc_df_stage`` (``final`` | ``sel_all``) picks the default glob map; pass
    ``group_globs`` explicitly to override.

    Tags are every key present in ``gmap``, ordered by :data:`GENIE_GROUP_ORDER` first,
    then any remaining keys (sorted) so ad-hoc entries in ``GENIE_GROUP_GLOBS`` are included.
    """
    gmap = group_globs if group_globs is not None else _genie_glob_map(mc_df_stage)
    yield from _pd.iter_ordered_group_tasks(gmap, GENIE_GROUP_ORDER)


def iter_cosmics_chunk_df_paths(sample: str, input_stage: str = "sel_all") -> Iterator[str]:
    """Yield ``.df`` paths for cosmics chunk map.

    ``sample`` is ``offbeam`` or ``intime``. ``input_stage`` selects the input glob:

    * ``sel_all`` (default): :data:`EVENT_SELECTION_GLOBS` — raw evt/trk/hdr dfs.
    * ``final``: :data:`SELECTED_EVENTS_GLOBS` — already-final-selected dfs.
    """
    if sample not in ("offbeam", "intime"):
        raise ValueError("sample must be 'offbeam' or 'intime', got %r" % sample)
    if input_stage == "sel_all":
        yield from iter_event_selection_df_paths(sample)
        return
    if input_stage == "final":
        yield from _pd.iter_keyed_df_paths(sample, SELECTED_EVENTS_GLOBS)
        return
    raise ValueError(
        "input_stage must be 'sel_all' or 'final', got %r" % input_stage
    )


def _multisim_glob_map(mc_df_stage: str) -> Dict[str, str]:
    if mc_df_stage == "final":
        return MULTISIM_SYST_GLOBS_FINAL
    if mc_df_stage == "sel_all":
        return MULTISIM_SYST_GLOBS_SEL_ALL
    raise ValueError("mc_df_stage must be 'final' or 'sel_all', got %r" % mc_df_stage)


def iter_multisim_syst_df_paths(mc_df_stage: str, syst_name: str) -> Iterator[str]:
    """Yield ``.df`` paths for one neutrino systematic (its dedicated glob / directory)."""
    if syst_name not in NEUTRINO_SYST_ORDER:
        raise KeyError(
            "unknown syst_name %r; expected one of %s" % (syst_name, NEUTRINO_SYST_ORDER)
        )
    glob_map = _multisim_glob_map(mc_df_stage)
    yield from _pd.iter_keyed_df_paths(syst_name, glob_map)


def iter_multisim_chunk_tasks(mc_df_stage: str) -> Iterator[Tuple[str, str]]:
    """Yield ``(syst_name, df_path)`` for the multisim **map** phase (one CAF shard per path).

    Legacy name retained for drivers. Prefer :func:`iter_multisim_map_tasks`.

    When every systematic shares the **same** glob string for this stage, yields
    ``("COMBINED", path)`` once per file so the driver runs a single HDF pass with
    all weights (same as legacy). Different globs emit one row per (syst, path).
    """
    glob_map = _multisim_glob_map(mc_df_stage)
    patterns = tuple(glob_map[sn] for sn in NEUTRINO_SYST_ORDER)
    if len(set(patterns)) == 1:
        sn0 = NEUTRINO_SYST_ORDER[0]
        for p in iter_multisim_syst_df_paths(mc_df_stage, sn0):
            yield "COMBINED", p
        return
    for sn in NEUTRINO_SYST_ORDER:
        for p in iter_multisim_syst_df_paths(mc_df_stage, sn):
            yield sn, p


def iter_multisim_mc_df_paths(mc_df_stage: str) -> Iterator[str]:
    """Yield unique ``.df`` paths across all systematics (legacy / convenience)."""
    yield from _pd.iter_unique_df_paths(iter_multisim_map_tasks(mc_df_stage))


def iter_multisim_map_tasks(mc_df_stage: str) -> Iterator[Tuple[str, str]]:
    """Alias for :func:`iter_multisim_chunk_tasks` — clearer name (map shard, not exposure batch)."""
    yield from iter_multisim_chunk_tasks(mc_df_stage)


# -----------------------------------------------------------------------------
# Default work directories (override with env or shell ``WORK_BASE``)
# -----------------------------------------------------------------------------
def default_event_selection_work_root(tag: str | None = None) -> Path:
    return _pd.default_work_root("NUMUCC_EVENT_SELECTION_WORK_BASE", _WORK_BASE, "event_selection", tag)


def default_genie_syst_work_root(tag: str | None = None) -> Path:
    return _pd.default_work_root("NUMUCC_GENIE_SYST_WORK_BASE", _WORK_BASE, "genie_syst", tag)


def default_cosmics_syst_work_root(tag: str | None = None) -> Path:
    return _pd.default_work_root("NUMUCC_COSMICS_SYST_WORK_BASE", _WORK_BASE, "cosmics_syst", tag)


def default_multisim_syst_work_root(tag: str | None = None) -> Path:
    return _pd.default_work_root("NUMUCC_MULTISIM_SYST_WORK_BASE", _WORK_BASE, "multisim_syst", tag)


def default_g4_syst_work_root(tag: str | None = None) -> Path:
    """Default map-shard root for G4-only neutrino multisim chunks (parallel to multisim work dir)."""
    return _pd.default_work_root("NUMUCC_G4_SYST_WORK_BASE", _WORK_BASE, "g4_syst", tag)


def default_flux_syst_work_root(tag: str | None = None) -> Path:
    """Default map-shard root for Flux-only neutrino multisim chunks (parallel to multisim work dir)."""
    return _pd.default_work_root("NUMUCC_FLUX_SYST_WORK_BASE", _WORK_BASE, "flux_syst", tag)


def default_mcstat_syst_work_root(tag: str | None = None) -> Path:
    """Default map-shard root for MCstat-only neutrino multisim chunks (parallel to multisim work dir)."""
    return _pd.default_work_root("NUMUCC_MCSTAT_SYST_WORK_BASE", _WORK_BASE, "mcstat_syst", tag)


def default_detvar_syst_work_root(tag: str | None = None) -> Path:
    """Scratch/output root for chunked detvar map pickles (``chunks/`` under here)."""
    return _pd.default_work_root(
        "NUMUCC_DETVAR_SYST_WORK_BASE", _WORK_BASE, "detvar_systematics", tag, chunked=False
    )


def default_syst_disk_root() -> Path:
    """Default root for the unified ``syst_disk_layout`` tree (``Cosmics/``, ``MCstat/``, …).

    Same logical tree that ``utils.get_syst_unc`` reads when ``SYST_DISK_ROOT`` (see
    ``pyanalib.syst_disk_layout.SYST_DISK_ENV``) is set. If that environment variable is
    set, this function returns that path (expanded). If not, returns a stable per-user
    default so ``run_syst_*`` scripts can aggregate without extra args.
    """
    return _pd.default_syst_disk_root(SYST_DISK_ENV, _WORK_BASE)


def default_syst_disk_cc_root() -> Path:
    """Default root for **joint** (cross-variable) syst outputs (``syst_disk_CC`` tree).

    Set ``NUMUCC_SYST_DISK_CC_ROOT`` to override. Otherwise a sibling directory next to
    :func:`default_syst_disk_root`, named ``syst_disk_CC``.
    """
    env = os.environ.get("NUMUCC_SYST_DISK_CC_ROOT")
    if env:
        return Path(env).expanduser()
    return _pd.sibling_dir(default_syst_disk_root(), "syst_disk_CC")


def default_joint_genie_cc_work_root(tag: str | None = None) -> Path:
    """Default map-shard root for joint (cross-variable) GENIE CC chunks."""
    return _pd.default_work_root("NUMUCC_JOINT_GENIE_CC_WORK_BASE", _WORK_BASE, "joint_genie_cc", tag)


def default_joint_multisim_cc_work_root(tag: str | None = None) -> Path:
    """Default map-shard root for joint (cross-variable) multisim CC chunks."""
    return _pd.default_work_root("NUMUCC_JOINT_MULTISIM_CC_WORK_BASE", _WORK_BASE, "joint_multisim_cc", tag)


# -----------------------------------------------------------------------------
# Human-readable summary
# -----------------------------------------------------------------------------
def summary_lines() -> Iterable[str]:
    """Human-readable listing for logs."""
    yield "# dataset_locations (numucc_1p0pi)"
    yield "SPRING_GEN1_ROOT=%s" % SPRING_GEN1_ROOT
    yield "PLOTS_BASE=%s" % PLOTS_BASE
    yield "default_syst_disk_root=%s" % default_syst_disk_root()
    yield ""
    yield "## event_selection"
    for k, pat in EVENT_SELECTION_GLOBS.items():
        n = len(sorted_glob(pat))
        yield "  %s: %d file(s)  glob=%s" % (k, n, pat)
    yield ""
    yield "## multisim_mc (per-syst globs)"
    for stage, dmap in (("final", MULTISIM_SYST_GLOBS_FINAL), ("sel_all", MULTISIM_SYST_GLOBS_SEL_ALL)):
        yield "  [%s]" % stage
        for sn, pat in dmap.items():
            n = len(list(iter_multisim_syst_df_paths(stage, sn)))
            yield "    %s: %d file(s)  glob=%s" % (sn, n, pat)
    yield ""
    yield "## detvar (WireMod tags)"
    for tag, pat in DETVAR_WIREMOD_GLOBS:
        n = sum(1 for _ in sorted_glob(pat))
        yield "  %s: %d file(s)  glob=%s" % (tag, n, pat)
    yield ""
    yield "## genie (knob groups)"
    for tag, pat in GENIE_GROUP_GLOBS.items():
        n = len(list(iter_genie_group_df_paths(tag)))
        yield "  %s: %d file(s)  glob=%s" % (tag, n, pat)
    yield ""
    yield "## cosmics chunk inputs (offbeam / intime)"
    for sample in ("offbeam", "intime"):
        pat = EVENT_SELECTION_GLOBS[sample]
        n = len(sorted_glob(pat))
        yield "  %s: %d file(s)  glob=%s" % (sample, n, pat)


def print_summary() -> None:
    _pd.print_summary(summary_lines())


if __name__ == "__main__":
    print_summary()
