"""Stages config: the pipeline definition -- what cuts run, in what order,
and what plots each stage produces.

This is the SINGLE place to edit when you want to add/remove a cut, add a
plot, or change the efficiency-curve variable list. :mod:`analysis_village.
nueNp0Pi.event_selection` (the runner) just calls :func:`build_pipeline` and
applies whatever ``Stage`` list comes back -- it doesn't know what stages
exist, only how to run them.

Cut/predicate functions themselves (``fiducial_volume``, ``no_muons``, etc.)
live in :mod:`analysis_village.nueNp0Pi.selections`; this module only
sequences them into ``Stage`` objects. Variable definitions used by
``PlotSpec``/``EFFICIENCY_VARS`` live in
:mod:`analysis_village.nueNp0Pi.config.plots`.

Also holds ``BREAKDOWN_REGISTRY`` -- "what breakdown categories exist, and
which function computes each one" -- assembled from the category functions in
``../selections.py`` (see that module's docstring for why it isn't in
``config/settings.py``).

By default, ``build_pipeline()`` also attaches one stacked event-distribution
breakdown plot per ``EFFICIENCY_VARS`` variable, by topology, at the final
stage -- configurable from a notebook cell via the ``EVT_BREAKDOWN_*`` module
attributes (see the comment above them) without editing this file.

It can also attach "N-1" plots -- for a held-out cut, apply every OTHER cut and plot a
chosen variable, the classic diagnostic for validating a cut threshold. Off by default;
configurable via the ``N_MINUS_1_*`` module attributes (see the comment above them).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from analysis_village.nueNp0Pi.config.plots import (
    VariableConfig,
    CORE_SELECTED_EVT_VARIABLE_CONFIGS,
    with_final_selected_evt_variables,
)
from pyanalib.variable_config import is_integrated_var_config
from analysis_village.nueNp0Pi.config.settings import (
    topology_labels, genie_mode_labels, genie_sb_mode_labels, pdg_labels,
    ELE_SOFTMAX_TH, ELE_PRIMARY_TH, P_SOFTMAX_TH, ELE_VTXDIST_TH, ELE_DEDX_TH,
)
from analysis_village.nueNp0Pi.selections import (
    fiducial_volume, contained, flash_matched,
    no_muons, no_pions, no_photons, good_electron, good_proton, electron_softmax, proton_softmax,
    electron_primary, electron_vertex_distance, electron_dedx,
    get_topo_category_nueNp0Pi, get_genie_category_spine, get_pdg_category_spine,
)

from pyanalib.chunked_selection import (
    PlotSpec, Stage, multicol_resolve_column_key,
    mask_from_filter, n_minus_1_selector,
)
from pyanalib.logging_utils import get_logger

logger = get_logger(__name__)


# ===========================================================================
# Breakdown-category registry: what breakdown categories exist, and which
# category function computes each one. Consumed by
# :class:`analysis_village.nueNp0Pi.event_selection.ChunkRunner` (default
# ``breakdown_registry``) and any :class:`~pyanalib.chunked_selection.PlotSpec`
# whose ``breakdown_type`` names a key here.
#
#   - n_categories must match the LENGTH of the list returned by
#     ``get_*_category(df, ret_cuts=True)``.
#   - The histograms in OverlayHistData.mc_hist are stored in the SAME order as
#     the cuts returned by ``get_*_category(...ret_cuts=True)`` -- which is the
#     order overlay_hists / overlay_hists_from_histdata expect for stacking.
#
# Lives here (not in config/settings.py) to avoid a circular import: this
# needs the category FUNCTIONS from ../selections.py, and settings.py has no
# imports from that file (selections.py imports settings.py's thresholds, not
# the other way around). config/stages.py already imports from ../selections.py
# with no cycle, and conceptually this already belongs with "what breakdown
# categories a stage's plots can use."
# ===========================================================================
BREAKDOWN_REGISTRY: Dict[str, Tuple[int, Any]] = {
    "topology": (len(topology_labels), get_topo_category_nueNp0Pi),
    "genie":    (len(genie_mode_labels), get_genie_category_spine),
    # "genie_sb": (len(genie_sb_mode_labels), get_genie_sb_category),
    "pdg":      (len(pdg_labels), get_pdg_category_spine),
}


# ===========================================================================
# Sample-aware state ops
# ---------------------------------------------------------------------------
# A "state" is a flat dict carrying both per-event and per-track dataframes
# for the sample currently being processed:
#   { "evt": df, "trk": df, "hdr": df }
# We keep the SAME state structure regardless of sample so cuts are uniform.
# Cuts are written as small functions ``(state, sample) -> state`` that mutate
# (and return) the state dict.
# ===========================================================================
def _apply_to_evt(fn: Callable[[pd.DataFrame], pd.DataFrame]) -> Callable:
    """Return a stage-cut that applies ``fn`` to state['evt'] only."""
    def _cut(state, sample):
        if state.get("evt") is not None:
            state["evt"] = fn(state["evt"])
        return state
    return _cut


def _evt_cut(fn: Callable[[pd.DataFrame], pd.DataFrame]) -> Dict[str, Callable]:
    """Return ``{"cut": ..., "mask_fn": ...}`` for a simple ``df -> filtered df``
    evt-level predicate, for ``Stage(**_evt_cut(fn), ...)``.

    Every reco cut in ``../selections.py`` (``fiducial_volume``, ``no_muons``, ...) is
    exactly this shape: a boolean condition on columns already present on ``evt`` before
    ANY of this pipeline's cuts run, independent of what other cuts have or haven't been
    applied yet. That's what makes ``mask_from_filter`` (see
    ``pyanalib.chunked_selection``) valid here -- it lets ``N_MINUS_1_STAGE_KEYS`` below
    ask "what if every OTHER cut were applied, but not this one" without re-running the
    pipeline. If a future cut's predicate ever depends on a column added by an EARLIER
    stage (order-dependent), give it plain ``cut=_apply_to_evt(fn)`` instead of
    ``**_evt_cut(fn)`` -- leaving ``mask_fn`` unset just excludes it from N-1 automatically.
    """
    return {"cut": _apply_to_evt(fn), "mask_fn": mask_from_filter(fn)}


# ===========================================================================
# Selectors used by PlotSpec to pluck the right rows for a given plot.
# Each takes (state, sample) and returns a DataFrame (or None to skip).
# ===========================================================================
def sel_primary_trks(state, sample):
    """Primary reconstructed particles (all species) in slices that have
    survived selection so far -- for track/particle-level breakdown plots.

    ``breakdown_type="pdg"`` categorizes individual reconstructed particles
    by their TRUE pdg code (``rec.dlp_true.particles.pdg_code``), which only
    exists on the per-particle table (``state["trk"]``), not the per-event
    one (``state["evt"]`` has no ``particles`` sub-level at all -- confirmed
    against real data). ``state["trk"]`` itself is never cut alongside
    ``state["evt"]`` in this pipeline, so this selector also restricts to
    particles belonging to slices still present in ``evt`` at this stage.
    """
    trk = state.get("trk")
    evt = state.get("evt")
    if trk is None or evt is None or len(evt) == 0 or len(trk) == 0:
        return None
    primary_col = multicol_resolve_column_key(trk, ("rec", "dlp", "particles", "is_primary"))
    if primary_col is None:
        return None
    trk = trk[trk.loc[:, primary_col] == 1]
    if len(trk) == 0:
        return None
    # trk's index has one extra (particle-level) level beyond evt's
    # event/slice-level index -- drop it to match against evt's surviving rows.
    evt_level_idx = trk.index.droplevel(-1)
    trk = trk[evt_level_idx.isin(evt.index)]
    return trk if len(trk) > 0 else None


def sel_evt(state, sample):
    """Event-level (per-slice) rows still surviving selection at this stage --
    for stacked event-DISTRIBUTION breakdown plots, as opposed to
    :func:`sel_primary_trks` (per-particle). Returns ``state["evt"]`` as-is
    (no extra inline cuts), or ``None`` if there's nothing to plot.
    """
    evt = state.get("evt")
    if evt is None or len(evt) == 0:
        return None
    return evt


def evt_breakdown_plot(var_config, breakdown_type, *, name_suffix="", ratio=False):
    """Build a :class:`~pyanalib.chunked_selection.PlotSpec` for an event-level
    stacked breakdown plot, appendable to ANY stage's ``plots`` list in
    :func:`build_pipeline`.

    ``build_pipeline()`` calls this automatically for every stage key in
    ``EVT_BREAKDOWN_STAGE_KEYS`` -- see the module attributes below for the
    notebook-configurable default wiring. Call it directly yourself if you
    want a one-off plot outside that mechanism (e.g. a different
    ``name_suffix``/``ratio`` per call).
    """
    return PlotSpec(
        var_config=var_config,
        breakdown_type=breakdown_type,
        selector=sel_evt,
        name_suffix=name_suffix,
        save_kwargs={"ratio": ratio, "show_genie_label": False},
    )


def verify_evt_breakdown_vars(evt_df, var_configs=None) -> Dict[str, bool]:
    """Check whether each ``var_config.var_evt_reco_col`` resolves against a real
    event dataframe, WITHOUT running the full pipeline.

    Cheap sanity check for a notebook cell before a long batched run -- catches
    the "stale reco column" failure mode a couple of ``EFFICIENCY_VARS`` entries
    hit (see the comment above ``EVT_BREAKDOWN_VARS``) before it aborts a
    multi-file batch partway through. ``var_configs`` defaults to
    ``EVT_BREAKDOWN_VARS``. Returns ``{var_save_name: resolves_bool}``; doesn't
    raise on its own.
    """
    if var_configs is None:
        var_configs = EVT_BREAKDOWN_VARS
    out: Dict[str, bool] = {}
    for vc in var_configs:
        col = getattr(vc, "var_evt_reco_col", None)
        resolved = multicol_resolve_column_key(evt_df, col) if col is not None else None
        out[vc.var_save_name] = resolved is not None
    return out


# ===========================================================================
# Variables for the efficiency curve (cell 82-86 in the notebook)
# ===========================================================================
EFFICIENCY_VARS = with_final_selected_evt_variables(
    list(CORE_SELECTED_EVT_VARIABLE_CONFIGS) + [VariableConfig.neutrino_energy()]
)


# ===========================================================================
# Event-distribution breakdown plots: notebook-configurable knobs.
# ---------------------------------------------------------------------------
# build_pipeline() reads these three module attributes each time it's called
# and appends one evt_breakdown_plot(...) per (stage, variable) pair. Set them
# from a notebook cell BEFORE calling build_event_selection_pipeline() /
# build_runner() -- a plain attribute assignment on the imported module is
# enough, no function argument needed:
#
#     import analysis_village.nueNp0Pi.config.stages as stages_mod
#     stages_mod.EVT_BREAKDOWN_TYPE = "genie"                             # default: "topology"
#     stages_mod.EVT_BREAKDOWN_STAGE_KEYS = ["no_photons", "electron_dedx"]  # default: final stage only
#     stages_mod.EVT_BREAKDOWN_VARS = [VariableConfig.neutrino_energy()]  # default: EFFICIENCY_VARS
#
# ``EVT_BREAKDOWN_TYPE`` must be a key in BREAKDOWN_REGISTRY (currently
# "topology", "genie", or "pdg" -- "genie_sb" is commented out there, not
# usable here either, until it's filled in). Set EVT_BREAKDOWN_VARS = [] to
# turn these plots off entirely.
#
# Defaults to EFFICIENCY_VARS minus four entries that don't work through this
# particular fill path, confirmed against real data (verify_evt_breakdown_vars()
# below re-checks this any time it's run):
#   - "integrated": the OverlayHistData fill path these plots go through
#     (pyanalib.chunked_selection's get_clipped_evts, at chunk-accumulation
#     time) does not special-case INTEGRATED_VAR_SAVE_NAME the way
#     pyanalib.overlay_plotting.get_clipped_evts (the render-time one, used by
#     the efficiency curve / regular overlay plots) does -- it tries to
#     resolve "integrated"'s placeholder reco column (rec.iscc) literally and
#     raises KeyError. A stacked single-bin bar chart isn't very informative
#     anyway.
#   - "vertex_x", "vertex_y", "vertex_z": their var_evt_reco_col (config/plots.py)
#     is stale -- e.g. vertex_x wants ('slc', 'vertex', 'x') but real .df files
#     have the vertex under ('rec', 'dlp', 'vertex', 'x') / ('rec', 'dlp_true',
#     'vertex', 'x') / ('rec', 'dlp_true', 'reco_vertex', 'x') (three plausible
#     candidates -- which one is semantically correct needs someone who knows
#     the SPINE reco schema, not guessed at here). Pre-existing bugs, not
#     introduced by this change -- just never exercised against a real reco
#     column before, since the efficiency curve itself doesn't read
#     var_evt_reco_col. Fix the VariableConfig in config/plots.py and re-add
#     the variable here once its reco column is correct.
#   ("electron-e" WAS in this list too -- its stale "..._GeV"-suffixed reco/truth
#   columns have been fixed in config/plots.py, confirmed against real data, so
#   it's back in the default set below.)
# ===========================================================================
_EVT_BREAKDOWN_STALE_RECO_COL_VARS = frozenset({"vertex_x", "vertex_y", "vertex_z"})
EVT_BREAKDOWN_VARS: List[Any] = [
    vc for vc in EFFICIENCY_VARS
    if not is_integrated_var_config(vc) and vc.var_save_name not in _EVT_BREAKDOWN_STALE_RECO_COL_VARS
]
EVT_BREAKDOWN_TYPE: str = "topology"
EVT_BREAKDOWN_STAGE_KEYS: Optional[Sequence[str]] = None  # None -> final stage only
# Subdirectory (under the plots dir) these plots render to -- None -> "selection", matching
# the pre-existing default. Set this alongside EVT_BREAKDOWN_TYPE (e.g. to "selection_genie")
# if you're rendering more than one breakdown_type from the same plots dir across separate
# runs and don't want the later run's selection_<var>.png files to overwrite the earlier
# one's -- read by event_selection_aggregate.py's render_overlay_plots via
# ``stages_mod.EVT_BREAKDOWN_DIR_NAME`` (module attribute, not a plain import, so it reflects
# whatever a notebook cell set most recently -- same pattern as the other EVT_BREAKDOWN_* knobs).
EVT_BREAKDOWN_DIR_NAME: Optional[str] = None

# Global kill switch for the per-VariableConfig ratio subplot (VariableConfig.ratio_mode --
# "data_mc" / "reco_true" / "signal_bkgd" -- see pyanalib.variable_config). Set True from a
# notebook cell (``stages_mod.DISABLE_RATIO_PLOTS = True``) to force every plot's ratio panel
# off for a render pass, without editing ratio_mode on every VariableConfig. Same
# live-read-at-render-time pattern as EVT_BREAKDOWN_DIR_NAME above -- read by
# event_selection_aggregate.py's render_overlay_plots via ``stages_mod.DISABLE_RATIO_PLOTS``.
DISABLE_RATIO_PLOTS: bool = False

# Global kill switch for the per-VariableConfig true-vs-reco response-matrix plot
# (VariableConfig.response_matrix -- see pyanalib.variable_config /
# pyanalib.response_matrix_plotting). Set True from a notebook cell
# (``stages_mod.DISABLE_RESPONSE_MATRIX_PLOTS = True``) to skip rendering every such plot for a
# render pass, without editing response_matrix on every VariableConfig. Same
# live-read-at-render-time pattern as DISABLE_RATIO_PLOTS above -- read by
# event_selection_aggregate.py's render_response_matrix_plots via
# ``stages_mod.DISABLE_RESPONSE_MATRIX_PLOTS``.
DISABLE_RESPONSE_MATRIX_PLOTS: bool = False

# Skip the efficiency-curve accumulation entirely (every stage's EfficiencyAccumulator fill
# becomes a no-op) -- and, since it's the only consumer, skip loading the "mcnu" truth table
# too (see event_selection.py's run_batch_selection/build_runner, which read this live, same
# pattern as the other knobs here). Useful when (re-)mapping a batch purely to pick up new
# N-1 / EVT_BREAKDOWN plots, where the efficiency curves themselves aren't needed for that
# run -- set True from a notebook cell (``stages_mod.DISABLE_EFFICIENCY_ACCUMULATION = True``)
# before calling build_event_selection_pipeline()/build_runner(). Does NOT affect
# save_for_breakdown (bar/cutflow counts) -- those come from a separate accumulator.
DISABLE_EFFICIENCY_ACCUMULATION: bool = False


# ===========================================================================
# N-1 plots: notebook-configurable knobs.
# ---------------------------------------------------------------------------
# Every reco-cut Stage above is built via ``_evt_cut(fn)``, so every one of them has a
# ``mask_fn`` (independent of the others -- see ``pyanalib.chunked_selection.
# mask_from_filter``'s docstring for why this is valid here) and is therefore ELIGIBLE
# to be held out. Nothing is wired up by default though (``N_MINUS_1_STAGE_KEYS`` starts
# empty) -- there's no single sensible default for "which variable(s) go with which
# held-out cut", so build_pipeline() only adds N-1 plots once you populate these from a
# notebook cell, same pattern as EVT_BREAKDOWN_*:
#
#     import analysis_village.nueNp0Pi.config.stages as stages_mod
#     stages_mod.N_MINUS_1_STAGE_KEYS = ["electron_softmax", "electron_dedx"]
#     stages_mod.N_MINUS_1_VARS = {
#         "electron_softmax": [VariableConfig.electron_softmax_score()],
#         "electron_dedx": [VariableConfig.electron_dedx()],
#     }
#
# For each key in ``N_MINUS_1_STAGE_KEYS``, build_pipeline() appends one PlotSpec per
# VariableConfig in ``N_MINUS_1_VARS[key]`` (falling back to ``N_MINUS_1_DEFAULT_VARS``
# if that key has no entry) to the FINAL stage's plots list. Each PlotSpec's selector
# (``pyanalib.chunked_selection.n_minus_1_selector``) applies every OTHER
# ``mask_fn``-eligible cut and skips just the held-out one -- i.e. "final selection minus
# this one cut" -- so you see where the held-out threshold would fall on the resulting
# distribution. A cut you never add to this list (e.g. the baseline is_fiducial /
# is_contained / is_flash_matched quality cuts) is simply never held out, but -- because
# it still has its own ``mask_fn`` -- stays enforced in every other cut's N-1 plot.
#
# ``N_MINUS_1_VLINES`` optionally draws the held-out cut's own threshold as a vertical
# marker line on its N-1 plot (``pyanalib.overlay_plotting.overlay_hists_from_histdata``'s
# existing ``vline=[(x, direction)]`` kwarg -- ``direction=0`` arrows left/"keep smaller
# values", ``direction=1`` arrows right/"keep larger values"). Pre-filled below for the
# five PID/kinematic cuts that have a single scalar threshold on a continuous variable
# (reusing the SAME named constants selections.py's cut functions default to -- see
# config/settings.py -- so the line always matches the actual cut). The other reco cuts
# (fiducial_volume, contained, flash_matched, no_muons, no_pions, no_photons,
# good_electron, good_proton) gate on precomputed boolean reco flags, not a continuous
# score with a meaningful threshold to draw, so they have no entry here -- add one only if
# you plot a variable for one of those cuts where a line would be meaningful.
N_MINUS_1_STAGE_KEYS: Sequence[str] = ()
N_MINUS_1_VARS: Dict[str, Sequence[Any]] = {}
N_MINUS_1_DEFAULT_VARS: Sequence[Any] = ()
N_MINUS_1_BREAKDOWN_TYPE: str = "topology"
N_MINUS_1_DIR_NAME: Optional[str] = None  # None -> "n_minus_1"
N_MINUS_1_VLINES: Dict[str, Dict[str, Sequence[Tuple[float, int]]]] = {
    # Inner key is var_save_name; vline only applied to the matching variable.
    "electron_softmax":         {"electron-softmax-score":   [(ELE_SOFTMAX_TH, 1)]},
    "electron_primary":         {"electron-primary-score":   [(ELE_PRIMARY_TH, 1)]},
    "proton_softmax":           {"proton-softmax-score":     [(P_SOFTMAX_TH, 1)]},
    "electron_vertex_distance": {"electron-vertex-distance": [(ELE_VTXDIST_TH, 0)]},
    "electron_dedx":            {"electron-dedx":            [(ELE_DEDX_TH, 0)]},
    "good_electron":            {"electron-e":               [(0.5, 1)]},
    "good_proton":              {"leading_proton_ke":        [(0.04, 1)]},
    "no_muons":                 {"leading_muon_ke":          [(0.025, 0)]},
    "no_pions":                 {"leading_pion_ke":          [(0.025, 0)]},
    # no_photons: leading_photon_ke not yet defined; add entry here when it is.
}


# ===========================================================================
# Pipeline definition
# ===========================================================================
def build_pipeline() -> List[Stage]:
    """Return the list of stages run by the chunk processor.

    Edit this function to add/remove cuts and plots. Both the per-chunk
    processor and the aggregator import this same list, so changes here
    propagate end-to-end.
    """
    stages: List[Stage] = []

    # ------------------------------------------------------------------
    # Stage 0: all reconstructed slices (no cut)
    # ------------------------------------------------------------------
    stages.append(Stage(
        key="allreco",
        label="All reconstructed interactions",
        cut=None,
        plots=[],
        save_for_efficiency=True,
        save_for_breakdown=True,  # stage counts (summary bar plot still skips this stage)
    ))

    # ------------------------------------------------------------------
    # Stage 1: In FV,
    # ------------------------------------------------------------------
    stages.append(Stage(
        key="is_fiducial",
        label="In FV",
        **_evt_cut(fiducial_volume),
        plots=[],
        save_for_efficiency=True,
        save_for_breakdown=True,
    ))

    stages.append(Stage(
        key="is_flash_matched",
        label="Flash Matched",
        **_evt_cut(flash_matched),
        plots=[],
        save_for_efficiency=True,
        save_for_breakdown=True,
    ))

    stages.append(Stage(
        key="is_contained",
        label="Contained",
        **_evt_cut(contained),
        plots=[],
        save_for_efficiency=True,
        save_for_breakdown=True,
    ))

    stages.append(Stage(
        key="no_muons",
        label="No Muons > 25 MeV",
        **_evt_cut(no_muons),
        plots=[],
        save_for_efficiency=True,
        save_for_breakdown=True,
    ))

    stages.append(Stage(
        key="no_pions",
        label="No Pions > 25 MeV",
        **_evt_cut(no_pions),
        plots=[],
        save_for_efficiency=True,
        save_for_breakdown=True,
    ))

    stages.append(Stage(
        key="no_photons",
        label="No Photons > 100 MeV",
        **_evt_cut(no_photons),
        plots=[
            PlotSpec(
                var_config=VariableConfig.particle_ke(),
                breakdown_type="pdg",
                selector=sel_primary_trks,
                plot_label_template=("Particle KE", "Primary Particles / Bin (POT={pot})", ""),
                save_kwargs={"ratio": False},
            ),
        ],
        save_for_efficiency=True,
        save_for_breakdown=True,
    ))

    stages.append(Stage(
        key="good_electron",
        label="Electron > 500 MeV",
        **_evt_cut(good_electron),
        plots=[],
        save_for_efficiency=True,
        save_for_breakdown=True,
    ))

    stages.append(Stage(
        key="good_proton",
        label="Proton > 40 MeV",
        **_evt_cut(good_proton),
        plots=[],
        save_for_efficiency=True,
        save_for_breakdown=True,
    ))

    stages.append(Stage(
        key="electron_softmax",
        label="Electron Softmax > 0.9",
        **_evt_cut(electron_softmax),
        plots=[],
        save_for_efficiency=True,
        save_for_breakdown=True,
    ))

    stages.append(Stage(
        key='electron_primary',
        label='Electron Primary Score > 0.99',
        **_evt_cut(electron_primary),
        plots=[],
        save_for_efficiency=True,
        save_for_breakdown=True,
    ))

    stages.append(Stage(
        key="proton_softmax",
        label="Proton Softmax > 0.75",
        **_evt_cut(proton_softmax),
        plots=[],
        save_for_efficiency=True,
        save_for_breakdown=True,
    ))

    stages.append(Stage(
        key='electron_vertex_distance',
        label='Electron Vertex Distance < 3.5cm',
        **_evt_cut(electron_vertex_distance),
        plots=[],
        save_for_efficiency=True,
        save_for_breakdown=True,
    ))

    stages.append(Stage(
        key='electron_dedx',
        label='Electron dEdx < 4 MeV/cm',
        **_evt_cut(electron_dedx),
        plots=[],
        save_for_efficiency=True,
        save_for_breakdown=True,
    ))

    # ------------------------------------------------------------------
    # Event-distribution breakdown plots (notebook-configurable -- see the
    # EVT_BREAKDOWN_* module attributes above).
    # ------------------------------------------------------------------
    if EVT_BREAKDOWN_VARS:
        if EVT_BREAKDOWN_TYPE not in BREAKDOWN_REGISTRY:
            raise ValueError(
                "EVT_BREAKDOWN_TYPE=%r is not a key in BREAKDOWN_REGISTRY (have: %s)"
                % (EVT_BREAKDOWN_TYPE, sorted(BREAKDOWN_REGISTRY))
            )
        stage_by_key = {s.key: s for s in stages}
        stage_keys = (
            EVT_BREAKDOWN_STAGE_KEYS if EVT_BREAKDOWN_STAGE_KEYS is not None else (stages[-1].key,)
        )
        for key in stage_keys:
            if key not in stage_by_key:
                raise ValueError(
                    "EVT_BREAKDOWN_STAGE_KEYS: unknown stage key %r (have: %s)"
                    % (key, list(stage_by_key))
                )
            for var_config in EVT_BREAKDOWN_VARS:
                stage_by_key[key].plots.append(evt_breakdown_plot(var_config, EVT_BREAKDOWN_TYPE))

    # ------------------------------------------------------------------
    # N-1 plots (notebook-configurable -- see the N_MINUS_1_* module attributes above).
    # Off by default (N_MINUS_1_STAGE_KEYS starts empty).
    # ------------------------------------------------------------------
    if N_MINUS_1_STAGE_KEYS:
        if N_MINUS_1_BREAKDOWN_TYPE not in BREAKDOWN_REGISTRY:
            raise ValueError(
                "N_MINUS_1_BREAKDOWN_TYPE=%r is not a key in BREAKDOWN_REGISTRY (have: %s)"
                % (N_MINUS_1_BREAKDOWN_TYPE, sorted(BREAKDOWN_REGISTRY))
            )
        stage_by_key = {s.key: s for s in stages}
        final_stage = stages[-1]
        eligible_keys = sorted(s.key for s in stages if s.mask_fn is not None)
        for held_out_key in N_MINUS_1_STAGE_KEYS:
            if held_out_key not in stage_by_key:
                raise ValueError(
                    "N_MINUS_1_STAGE_KEYS: unknown stage key %r (have: %s)"
                    % (held_out_key, list(stage_by_key))
                )
            if held_out_key not in eligible_keys:
                raise ValueError(
                    "N_MINUS_1_STAGE_KEYS: stage %r has no mask_fn (not built via "
                    "_evt_cut(...), so it can't be held out for an N-1 plot) -- have "
                    "mask_fn-eligible stages: %s" % (held_out_key, sorted(eligible_keys))
                )
            var_configs = N_MINUS_1_VARS.get(held_out_key, N_MINUS_1_DEFAULT_VARS)
            vline_map = N_MINUS_1_VLINES.get(held_out_key)
            for var_config in var_configs:
                save_kwargs = {"show_genie_label": False}
                if vline_map is not None:
                    vline = vline_map.get(var_config.var_save_name)
                    if vline is not None:
                        save_kwargs["vline"] = vline
                final_stage.plots.append(PlotSpec(
                    var_config=var_config,
                    breakdown_type=N_MINUS_1_BREAKDOWN_TYPE,
                    selector=n_minus_1_selector(held_out_key, eligible_keys),
                    name_suffix=f"n_minus_1_{held_out_key}",
                    save_kwargs=save_kwargs,
                ))

    logger.debug("[pipeline] built %d stages:", len(stages))
    for i, s in enumerate(stages):
        flags = []
        if s.save_for_efficiency:
            flags.append("eff")
        if s.save_for_breakdown:
            flags.append("breakdown")
        plot_summary = (
            ", ".join(
                f"{ps.var_config.var_save_name}[{ps.breakdown_type}]"
                + (f"+{ps.name_suffix}" if ps.name_suffix else "")
                for ps in s.plots
            )
            if s.plots else "—"
        )
        logger.debug(
            "  [%02d] %-25s  cut=%-5s  plots=%d (%s)  flags=%s",
            i,
            s.key,
            "yes" if s.cut is not None else "no",
            len(s.plots),
            plot_summary,
            ",".join(flags) if flags else "none",
        )
    return stages
