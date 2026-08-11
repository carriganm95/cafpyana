"""numuCC 1p0pi event-selection pipeline config, built on the generic chunked
event-selection framework in ``pyanalib.chunked_selection``.

This module now only holds the numuCC-1p0pi-SPECIFIC pieces:

* ``BREAKDOWN_REGISTRY`` -- maps breakdown_type -> (n_categories, get_cuts_fn)
  using this analysis's category functions from ``categories.py``.
* ``SIGNAL_MASK_FN`` -- the numuCC CC 1p0pi-in-FV truth-signal definition used
  by efficiency accumulators.
* ``ChunkRunner`` -- a thin subclass of ``pyanalib.chunked_selection.ChunkRunner``
  that defaults to the two above, so existing call sites
  (``ChunkRunner(sample, stages, efficiency_vars, mc_univ_syst_tags=...)``)
  keep working unchanged.

Everything generic -- ``OverlayHistData``, ``BarBreakdown``,
``EfficiencyAccumulator``, ``PlotSpec``, ``Stage``, chunk aggregation/merge/
exposure-scaling helpers, and the MultiIndex column-resolution utilities --
now lives in ``pyanalib.chunked_selection`` and can be reused by any
``analysis_village`` folder that wants a chunked/streaming event-selection
pipeline: build your own ``BREAKDOWN_REGISTRY`` + ``signal_mask_fn`` and
construct ``pyanalib.chunked_selection.ChunkRunner`` directly (this file's
``ChunkRunner`` subclass is just a numuCC-1p0pi-flavored convenience
wrapper, not a required base class).

A second pass (see ``scripts/event_selection_aggregate.py`` if present) reads
back the per-chunk pickles, sums intrinsic-weight histograms across chunks,
applies **global** POT / gates factors from summed chunk metadata, and then
renders the final plots through ``overlay_hists_from_histdata`` -- which
produces a plot that is bit-for-bit identical to ``overlay_hists`` called
with the raw dataframes.

Memory note
-----------
After this refactor the scripts only ever hold one file's events in RAM at a
time, plus the (negligible) accumulated histograms. This is what makes it run
on the full sample.

Systematics on overlay plots
----------------------------
Consumer paths load **pre-saved** fractional covariances from the syst-disk tree
via ``utils.get_syst_unc`` (``SYST_DISK_ROOT`` / ``--syst-disk-root``), or
the pre-summed ``CategorySummary/category_syst_summary.npz`` when overlay helpers
default to that. On-the-fly covariances from MC multi-universe weights at plot
time are retired (see ``cafpyana_trash/…/legacy_get_frac_unc.py`` and
``legacy_frac_cov_from_mc_univ_histdata.py``). Producer pipelines that *write*
those pre-saved files still use ``utils.get_univ_rates`` / DETVAR scripts.

Adding new things
-----------------
* A new selection cut    -> add a new ``Stage`` to ``build_pipeline()``.
* A new variable to plot -> add a new ``PlotSpec`` to a stage's ``plots`` list.
* A new variable to follow through every stage -> add a ``VariableConfig`` to
  the ``efficiency_vars`` list passed to the pipeline.

Compat
------
The module supports either MultiIndex per-event dataframes (``mc_df``) or
per-track dataframes built via ``pd.concat([df.trk1, df.trk2])`` -- a
``PlotSpec`` simply specifies which one to use through its ``selector`` field.
"""

from __future__ import annotations

import sys
from os import path
from typing import Any, Dict, Tuple

# Local imports (kept minimal here; the pipeline definition imports the rest)
sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

from pyanalib.chunked_selection import (
    ChunkRunner as _BaseChunkRunner,
    OverlayHistData,
    BarBreakdown,
    EfficiencyAccumulator,
    PlotSpec,
    Stage,
    ExposureTotals,
    aggregate_chunk_files,
    apply_global_exposure_scales,
    get_clipped_evts,
    mc_univ_weight_matrix,
    merge_samples,
    multicol_get_series,
    multicol_resolve_column_key,
    sanitize_merged_histdata_finite,
)

from analysis_village.nueNp0Pi.categories import (
    get_topo_category, get_genie_category, get_genie_sb_category, get_pdg_category,
    topology_labels, genie_mode_labels, genie_sb_mode_labels,
    pdg_labels,
    IsNuInFV_NumuCC_1p0pi, DETECTOR, get_topo_category_nueNp0Pi, get_genie_category_spine, get_pdg_category_spine,
    IsNuInFV_NueCC_1p0pi, IsNuInFV
)

# ---------------------------------------------------------------------------
# Map breakdown_type -> (n_categories, get_cuts_fn), FOR THIS ANALYSIS.
#   - n_categories must match the LENGTH of the list returned by
#     ``get_*_category(df, ret_cuts=True)``.
#   - The histograms in OverlayHistData.mc_hist are stored in the SAME order as
#     the cuts returned by ``get_*_category(...ret_cuts=True)`` -- which is the
#     order overlay_hists / overlay_hists_from_histdata expect for stacking.
# ---------------------------------------------------------------------------
BREAKDOWN_REGISTRY: Dict[str, Tuple[int, Any]] = {
    "topology": (len(topology_labels), get_topo_category_nueNp0Pi),
    "genie":    (len(genie_mode_labels), get_genie_category_spine),
    # "genie_sb": (len(genie_sb_mode_labels), get_genie_sb_category),
    "pdg":      (len(pdg_labels), get_pdg_category_spine),
}


# DLP truth signal keys — present in evt_df, absent in mcnu_df.
_DLP_SIGNAL1P_KEY = ('rec', 'dlp_true', 'true_signal1p', '', '')
_DLP_SIGNAL_NP_KEY = ('rec', 'dlp_true', 'true_signalNp', '', '')


def _mcnu_signal_mask_dlp_phase_space(df):
    """GENIE-level approximation of DLP true_signal1p|true_signalNp phase space.

    NueCC in FV with ≥1 primary electron (no KE threshold — DLP accepts electrons
    below 500 MeV), ≥1 proton KE>40 MeV, and no pions/muons/photons above threshold.
    Gives ~283 events per file vs ~160 DLP truth events (~57% DLP reco efficiency).

    To use all nue CC in FV as denominator instead (~484/file), replace with:
        IsNuInFV(df) & (df.mc.iscc == 1) & (df.mc.pdg.abs() == 12)
    """
    ne_key = multicol_resolve_column_key(df, ("mc", "ne", ""))
    np_key = multicol_resolve_column_key(df, ("mc", "np_40MeV", ""))
    npi_key = multicol_resolve_column_key(df, ("mc", "npi_25MeV", ""))
    nmu_key = multicol_resolve_column_key(df, ("mc", "nmu_25MeV", ""))
    ng_key = multicol_resolve_column_key(df, ("mc", "ng_100MeV", ""))
    if any(k is None for k in (ne_key, np_key, npi_key, nmu_key, ng_key)):
        return IsNuInFV(df) & (df.mc.iscc == 1) & (df.mc.pdg.abs() == 12)
    return (
        IsNuInFV(df)
        & (df.mc.iscc == 1)
        & (df.mc.pdg.abs() == 12)
        & (df.loc[:, ne_key] >= 1)
        & (df.loc[:, np_key] >= 1)
        & (df.loc[:, npi_key] == 0)
        & (df.loc[:, nmu_key] == 0)
        & (df.loc[:, ng_key] == 0)
    )


def SIGNAL_MASK_FN(df):
    """Signal mask dispatching on truth source.

    evt_df:  true_signal1p | true_signalNp (DLP particle-level truth matching).
    mcnu_df: GENIE nue CC in FV with DLP-compatible topology cuts (no electron
             KE threshold; proton KE>40 MeV; no pi/mu/gamma above threshold).
    """
    if _DLP_SIGNAL1P_KEY in df.columns:
        mask = df[_DLP_SIGNAL1P_KEY] == 1
        if _DLP_SIGNAL_NP_KEY in df.columns:
            mask = mask | (df[_DLP_SIGNAL_NP_KEY] == 1)
        return mask
    return _mcnu_signal_mask_dlp_phase_space(df)


class ChunkRunner(_BaseChunkRunner):
    """numuCC-1p0pi-flavored ``ChunkRunner``.

    Defaults ``breakdown_registry``/``signal_mask_fn`` to this analysis's
    definitions so existing call sites keep working unchanged:

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
