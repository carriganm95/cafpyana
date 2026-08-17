"""nueNp0Pi event-selection RUNNER: how a pipeline gets applied, not what it
contains. Holds the per-job map phase and the factory that wires everything
into the generic :class:`pyanalib.event_selection_pipeline.EventSelectionPipeline`.

The pipeline DEFINITION (what stages/cuts/plots exist) lives in
:mod:`analysis_village.nueNp0Pi.config.stages` -- edit that module to add or
remove a cut, add a plot, or follow a new variable through the efficiency
curve. This module just calls :func:`~analysis_village.nueNp0Pi.config.stages.build_pipeline`
and applies whatever ``Stage`` list comes back. Everything ANALYSIS-AGNOSTIC
(survey/bin-pack/map-pool/aggregate orchestration) lives in
:mod:`pyanalib.event_selection_pipeline`; this module only supplies the
nueNp0Pi-specific pieces that orchestration needs.

Formerly split across ``event_selection_pipeline_def.py`` (pipeline
definition -- now ``config/stages.py``), ``event_selection_batched.py``
(orchestration -- now in ``pyanalib.event_selection_pipeline``),
``event_selection_batch_core.py`` (map-phase core), and
``selection_framework.py`` (the ``ChunkRunner`` subclass). Rendering (the
reduce/aggregate phase) is a big enough, distinct enough concern that it
stays separate in ``event_selection_aggregate.py``.

Adding new things
-----------------
* New cut       : add a Stage(...) at the right point in
                  ``config.stages.build_pipeline``.
* New plot      : append a PlotSpec(...) to a stage's ``plots`` list, in
                  ``config.stages.build_pipeline``.
* New eff. var  : extend ``CORE_SELECTED_EVT_VARIABLE_CONFIGS`` or the extras passed
                  into ``with_final_selected_evt_variables`` for
                  ``config.stages.EFFICIENCY_VARS`` -- both live in
                  ``analysis_village/nueNp0Pi/config/plots.py``.

Typical notebook usage::

    from analysis_village.nueNp0Pi.event_selection import build_event_selection_pipeline
    from pyanalib.event_selection_pipeline import EventSelectionPipelineConfig

    pipeline = build_event_selection_pipeline(EventSelectionPipelineConfig())
    result = pipeline.run_full()
"""
from __future__ import annotations

import gc
import os
import sys
from datetime import datetime
from os import path
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

from analysis_village.nueNp0Pi.config.stages import build_pipeline, BREAKDOWN_REGISTRY
# Imported as a module (not `from ... import EFFICIENCY_VARS/DISABLE_EFFICIENCY_ACCUMULATION`)
# so build_runner()/run_batch_selection() below pick up whatever a notebook cell most recently
# set on the module -- a plain name import would freeze in the value from whenever this file
# was first imported, same reasoning as EVT_BREAKDOWN_TYPE/N_MINUS_1_* in
# event_selection_aggregate.py.
import analysis_village.nueNp0Pi.config.stages as stages_mod
from analysis_village.nueNp0Pi.selections import SIGNAL_MASK_FN
from analysis_village.nueNp0Pi.config.datasets import KEYS2LOAD
from analysis_village.nueNp0Pi.dataset_paths import iter_event_selection_df_paths, default_syst_disk_root
from analysis_village.nueNp0Pi.evt_derived_kinematics import (
    ensure_derived_trk_kinematics_cols,
    ensure_mc_level_phi_mcnu,
)

from pyanalib.chunked_selection import (
    ChunkRunner as _BaseChunkRunner,
    multicol_resolve_column_key,
)
from pyanalib.split_df_helpers_new import get_n_split, load_dfs
from pyanalib.pandas_helpers import pad_column_name
from pyanalib.event_selection_pipeline import (
    EventSelectionHooks,
    EventSelectionPipeline,
    EventSelectionPipelineConfig,
)
from pyanalib.logging_utils import get_logger

logger = get_logger(__name__)


SAMPLES = ("mc", "data", "intime", "offbeam", "dirt")


# ===========================================================================
# nueNp0Pi-flavored ChunkRunner (formerly selection_framework.py).
# ===========================================================================
class ChunkRunner(_BaseChunkRunner):
    """nueNp0Pi-flavored ``ChunkRunner``.

    Defaults ``breakdown_registry``/``signal_mask_fn`` to this analysis's
    definitions (:data:`analysis_village.nueNp0Pi.config.stages.BREAKDOWN_REGISTRY`
    / :data:`analysis_village.nueNp0Pi.selections.SIGNAL_MASK_FN`) so existing
    call sites keep working unchanged:

        ChunkRunner(sample, stages, efficiency_vars, mc_univ_syst_tags=...)

    Pass ``breakdown_registry=`` / ``signal_mask_fn=`` explicitly to override
    (or just construct ``pyanalib.chunked_selection.ChunkRunner`` directly).
    """

    def __init__(
        self,
        sample,
        stages,
        efficiency_vars,
        mc_univ_syst_tags=None,
        breakdown_registry=None,
        signal_mask_fn=None,
        bar_breakdown_types=("topology", "genie"),
        efficiency_denom_from_first_stage=True,
    ):
        super().__init__(
            sample=sample,
            stages=stages,
            efficiency_vars=efficiency_vars,
            breakdown_registry=breakdown_registry or BREAKDOWN_REGISTRY,
            signal_mask_fn=signal_mask_fn or SIGNAL_MASK_FN,
            mc_univ_syst_tags=mc_univ_syst_tags,
            bar_breakdown_types=bar_breakdown_types,
            efficiency_denom_from_first_stage=efficiency_denom_from_first_stage,
        )


def build_runner(
    sample: str,
    mc_univ_syst_tags: tuple[str, ...] | None = None,
) -> ChunkRunner:
    """Build a ChunkRunner instance for a given sample.

    Always uses the same pipeline definition, but knows which sample-slot to
    fill in the histogram accumulators.

    ``mc_univ_syst_tags``: optional tuple of MC multi-universe syst names (e.g.
    ``("Flux", "G4", "GENIE")``) whose columns ``mc[s]['univ_i']`` are summed into
    chunked histograms for later fractional covariance (see aggregate flag).

    Reads ``config.stages.DISABLE_EFFICIENCY_ACCUMULATION`` live: when True, this
    ChunkRunner gets ``efficiency_vars=[]`` instead of ``EFFICIENCY_VARS``, so
    ``_fill_efficiency`` never fills anything (a no-op per stage) -- useful when a batch
    is only being (re-)mapped to pick up new N-1 / EVT_BREAKDOWN plots and the
    efficiency-curve accumulation isn't needed for that run. See
    ``run_batch_selection`` for the matching ``mcnu``-load skip.
    """
    efficiency_vars = (
        [] if stages_mod.DISABLE_EFFICIENCY_ACCUMULATION else stages_mod.EFFICIENCY_VARS
    )
    return ChunkRunner(
        sample=sample,
        stages=build_pipeline(),
        efficiency_vars=efficiency_vars,
        mc_univ_syst_tags=mc_univ_syst_tags,
    )


# ===========================================================================
# Map-phase core (formerly event_selection_batch_core.py): load a batch of
# ``.df`` files, run the pipeline, write one pickle.
# ===========================================================================
def hdf_has_mcnu(df_file: str) -> bool:
    try:
        with pd.HDFStore(df_file, mode="r") as store:
            keys = store.keys()
        return any(str(k).startswith("/mcnu_") for k in keys)
    except Exception:
        return False


def hdr_chunk_pot(hdr_df: pd.DataFrame | None) -> float:
    if hdr_df is None or "pot" not in hdr_df.columns:
        return 0.0
    return float(hdr_df["pot"].sum())


def hdr_data_gates_bnb(hdr_df: pd.DataFrame | None) -> float:
    if hdr_df is None or "nbnbinfo" not in hdr_df.columns:
        return 0.0
    return float(hdr_df["nbnbinfo"].sum())


def hdr_cosmic_gates_intime(hdr_df: pd.DataFrame | None) -> float:
    if hdr_df is None or "ngenevt" not in hdr_df.columns:
        return 0.0
    return float(hdr_df.loc[hdr_df["first_in_subrun"] == 1, "ngenevt"].sum())


def hdr_cosmic_gates_offbeam(hdr_df: pd.DataFrame | None) -> float:
    if hdr_df is None or "noffbeambnb" not in hdr_df.columns:
        return 0.0
    return float(hdr_df.loc[hdr_df["first_in_subrun"] == 1, "noffbeambnb"].sum())


def file_hdr_meta(sample: str, hdr_df: pd.DataFrame | None, df_file: str) -> Dict[str, Any]:
    meta = {
        "path": df_file,
        "size_bytes": os.path.getsize(df_file) if os.path.isfile(df_file) else 0,
        "pot": hdr_chunk_pot(hdr_df),
    }
    if sample == "data":
        meta["gates_bnb"] = hdr_data_gates_bnb(hdr_df)
    elif sample == "intime":
        meta["cosmic_gates_intime"] = hdr_cosmic_gates_intime(hdr_df)
    elif sample == "offbeam":
        meta["cosmic_gates_offbeam"] = hdr_cosmic_gates_offbeam(hdr_df)
    return meta


def accumulate_hdr_meta(
    sample: str,
    hdr_df: pd.DataFrame | None,
    chunk_pot: List[float],
    chunk_gates_bnb: List[float],
    chunk_cosmic_gates_intime: List[float],
    chunk_cosmic_gates_offbeam: List[float],
) -> None:
    chunk_pot[0] += hdr_chunk_pot(hdr_df)
    if sample == "data":
        chunk_gates_bnb[0] += hdr_data_gates_bnb(hdr_df)
    elif sample == "intime":
        chunk_cosmic_gates_intime[0] += hdr_cosmic_gates_intime(hdr_df)
    elif sample == "offbeam":
        chunk_cosmic_gates_offbeam[0] += hdr_cosmic_gates_offbeam(hdr_df)


def intrinsic_weight_series(
    df: pd.DataFrame | None,
    sample: str,
    use_mc_genweight: bool = False,
) -> np.ndarray:
    if df is None or len(df) == 0:
        return np.ones(0, dtype=float)
    if sample == "data":
        return np.ones(len(df), dtype=float)
    if sample in ("intime", "offbeam"):
        return np.ones(len(df), dtype=float)
    if sample in ("mc", "dirt"):
        if not use_mc_genweight:
            return np.ones(len(df), dtype=float)
        try:
            gw = df["mc"]["genweight"]
            w = np.asarray(gw, dtype=float).reshape(-1)
            return np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            return np.ones(len(df), dtype=float)
    raise ValueError(sample)


def attach_intrinsic_weights(
    evt_df: pd.DataFrame | None,
    trk_df: pd.DataFrame | None,
    sample: str,
    use_mc_genweight: bool = False,
) -> None:
    if evt_df is not None and len(evt_df) > 0:
        evt_df["pot_weight"] = intrinsic_weight_series(evt_df, sample, use_mc_genweight)
    if trk_df is not None and len(trk_df) > 0:
        trk_df["pot_weight"] = intrinsic_weight_series(trk_df, sample, use_mc_genweight)


def prefix_mcnu_columns(mc_nu_df: pd.DataFrame) -> None:
    if isinstance(mc_nu_df.columns, pd.MultiIndex):
        try:
            first_level = mc_nu_df.columns.get_level_values(0)
            need_prefix = not np.all(first_level == "mc")
        except Exception:
            need_prefix = True
        if need_prefix:
            mc_nu_df.columns = pd.MultiIndex.from_tuples(
                [tuple(["mc"] + list(c)) for c in mc_nu_df.columns]
            )


def ensure_trk_phi_col(trk_df: pd.DataFrame | None) -> None:
    if trk_df is None or len(trk_df) == 0:
        return
    if not isinstance(trk_df.columns, pd.MultiIndex):
        return
    if multicol_resolve_column_key(trk_df, ("pfp", "trk", "phi", "", "", "")) is not None:
        return
    kx = multicol_resolve_column_key(trk_df, ("pfp", "trk", "dir", "x", ""))
    ky = multicol_resolve_column_key(trk_df, ("pfp", "trk", "dir", "y", ""))
    if kx is None or ky is None:
        return
    phi_col = pad_column_name(("pfp", "trk", "phi", "", "", ""), trk_df)
    trk_df.loc[:, phi_col] = np.degrees(
        np.arctan2(
            np.asarray(trk_df.loc[:, kx], dtype=float),
            np.asarray(trk_df.loc[:, ky], dtype=float),
        )
    )


def ensure_phi_and_kinematics_cols(
    evt_df: pd.DataFrame,
    trk_df: pd.DataFrame | None,
    mcnu_df: pd.DataFrame | None,
) -> Tuple[pd.DataFrame, pd.DataFrame | None]:
    evt_df = ensure_derived_trk_kinematics_cols(evt_df)
    ensure_trk_phi_col(trk_df)
    if mcnu_df is not None and len(mcnu_df) > 0:
        prefix_mcnu_columns(mcnu_df)
        mcnu_df = ensure_mc_level_phi_mcnu(mcnu_df)
    return evt_df, mcnu_df


def run_batch_selection(
    sample: str,
    df_files: Sequence[str],
    out_path: str,
    *,
    job_id: str = "",
    use_mc_genweight: bool = False,
    mc_univ_syst_tags: Sequence[str] = (),
    max_splits_per_file: int | None = None,
    pipeline_trace=None,
) -> Dict[str, Any]:
    """Load a file batch, run the notebook pipeline, write one pickle."""
    keys = list(KEYS2LOAD)
    # ``mcnu`` is only ever consumed by EfficiencyAccumulator's fill_denominator_from_mcnu
    # path, which nueNp0Pi's ChunkRunner never uses anyway (efficiency_denom_from_first_stage
    # defaults to True there -- fill_from_mcnu is unconditionally False). Skipping the load
    # entirely when DISABLE_EFFICIENCY_ACCUMULATION is set avoids that (already-unused) I/O too.
    load_mcnu = (
        sample == "mc"
        and not stages_mod.DISABLE_EFFICIENCY_ACCUMULATION
        and any(hdf_has_mcnu(f) for f in df_files)
    )
    keys_load = keys + (["mcnu"] if load_mcnu else [])

    mc_univ_tags = tuple(mc_univ_syst_tags) if sample == "mc" else ()
    runner = build_runner(sample, mc_univ_syst_tags=mc_univ_tags or None)

    chunk_pot = [0.0]
    chunk_gates_bnb = [0.0]
    chunk_cosmic_gates_intime = [0.0]
    chunk_cosmic_gates_offbeam = [0.0]
    per_file_meta: List[Dict[str, Any]] = []
    n_evt_total = 0

    for df_file in df_files:
        n_splits = get_n_split(df_file)
        cap = n_splits if max_splits_per_file is None else min(n_splits, max_splits_per_file)
        file_keys = keys_load
        file_dfs = load_dfs(df_file, file_keys, n_max_concat=cap)

        hdr_df = file_dfs.get("hdr")
        per_file_meta.append(file_hdr_meta(sample, hdr_df, df_file))
        accumulate_hdr_meta(
            sample,
            hdr_df,
            chunk_pot,
            chunk_gates_bnb,
            chunk_cosmic_gates_intime,
            chunk_cosmic_gates_offbeam,
        )

        logger.debug(f"Keys available in {df_file}: {list(file_dfs.keys())}")

        evt_df = file_dfs["evt"]
        trk_df = file_dfs["trk"]
        mcnu_df = file_dfs.get("mcnu") if load_mcnu else None

        attach_intrinsic_weights(evt_df, trk_df, sample, use_mc_genweight)
        evt_df, mcnu_df = ensure_phi_and_kinematics_cols(evt_df, trk_df, mcnu_df)
        n_evt_total += int(len(evt_df))

        runner.run(
            {"evt": evt_df, "trk": trk_df, "hdr": hdr_df, "mcnu": mcnu_df},
            pipeline_trace=pipeline_trace,
        )

        del file_dfs, evt_df, trk_df, hdr_df, mcnu_df
        gc.collect()

    meta = {
        "weight_scheme": "intrinsic",
        "use_mc_genweight": bool(use_mc_genweight),
        "mc_univ_syst_tags": list(mc_univ_tags),
        "df_files": list(df_files),
        "per_file": per_file_meta,
        "sample": sample,
        "job_id": job_id,
        "chunk_pot": chunk_pot[0],
        "chunk_gates_bnb": chunk_gates_bnb[0],
        "chunk_cosmic_gates_intime": chunk_cosmic_gates_intime[0],
        "chunk_cosmic_gates_offbeam": chunk_cosmic_gates_offbeam[0],
        "n_evt": n_evt_total,
        "n_files": len(df_files),
        "workflow": "batched_notebook",
        "mc_efficiency_enabled": load_mcnu,
    }
    runner.save(out_path, extra_meta=meta)
    return meta


# ===========================================================================
# Wiring: build a generic EventSelectionPipeline with nueNp0Pi's hooks plugged in.
# (formerly event_selection_batched.py's module-level functions)
# ===========================================================================
def default_event_selection_batched_work_root(tag: str | None = None) -> Path:
    t = tag or datetime.now().strftime("%Y%m%d")
    base = os.environ.get("NUMUCC_EVENT_SELECTION_WORK_BASE")
    if base:
        return Path(base)
    user = os.environ.get("USER", "user")
    return Path(f"/exp/sbnd/data/users/{user}/xsec/numucc_1p0pi/event_selection-batched-{t}")


def build_event_selection_pipeline(
    cfg: EventSelectionPipelineConfig | None = None,
) -> EventSelectionPipeline:
    """Build an :class:`~pyanalib.event_selection_pipeline.EventSelectionPipeline`
    wired up with nueNp0Pi's map phase (:func:`run_batch_selection`), reduce/render
    phase (:func:`analysis_village.nueNp0Pi.event_selection_aggregate.aggregate_and_render`),
    and dataset glob lookup (:func:`analysis_village.nueNp0Pi.dataset_paths.iter_event_selection_df_paths`).
    """
    from analysis_village.nueNp0Pi.config.datasets import EVENT_SELECTION_GLOBS
    # Deferred: event_selection_aggregate imports build_pipeline/EFFICIENCY_VARS
    # from this module, so importing it back at module load time would cycle.
    from analysis_village.nueNp0Pi.event_selection_aggregate import aggregate_and_render

    cfg = cfg or EventSelectionPipelineConfig()
    hooks = EventSelectionHooks(
        iter_df_paths=iter_event_selection_df_paths,
        run_batch_selection=run_batch_selection,
        aggregate_and_render=aggregate_and_render,
        default_work_root=default_event_selection_batched_work_root,
        default_syst_disk_root=default_syst_disk_root,
        describe_glob=lambda sample: EVENT_SELECTION_GLOBS.get(sample, "(no glob configured)"),
    )
    return EventSelectionPipeline(cfg, hooks)
