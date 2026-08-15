"""Generic container for a single "variable configuration".

A ``VariableConfig`` bundles everything needed to histogram one physics
variable across an analysis's selection/efficiency/unfolding pipeline: the
reco/truth/nu dataframe column paths, bin edges, and plot labels. It has no
built-in physics -- it's just a record shape plus the ``bin_centers``
convenience derived from ``bins``.

Analyses should subclass ``VariableConfig`` and add their own
``@classmethod`` factories, one per variable they care about (e.g.
``muon_momentum``, TKI variables, ...). See
``analysis_village/nueNp0Pi/config/plots.py`` for an example with ~70
variables built this way.

``INTEGRATED_VAR_SAVE_NAME`` / ``is_integrated_var_config`` are a small shared
convention: several analyses use a sentinel ``var_save_name == "integrated"``
VariableConfig to represent a single-bin "all events" measurement (for total/
integrated-flux cross sections). The specific dummy column/value used to fill
that single bin is analysis-specific (depends on each analysis's own on-disk
schema and production conventions), so it is NOT defined here -- only the
shared sentinel name and the predicate that checks for it.

``ratio_mode`` and friends opt a variable into an automatic ratio subplot on its
overlay plots (see ``pyanalib.overlay_plotting.overlay_hists_from_histdata``'s
``ratio``/``ratio_vars``/... kwargs, and ``analysis_village.nueNp0Pi.event_selection_aggregate
.render_overlay_plots``, which reads these fields and translates them into that
function's kwargs at render time). ``ratio_mode`` is one of:

- ``None`` (default) -- no change: whatever ``ratio``/``ratio_vars`` a ``PlotSpec``'s own
  ``save_kwargs`` sets (if anything) is used as-is.
- ``"data_mc"`` -- explicitly request the engine's built-in Data/MC ratio panel.
- ``"reco_true"`` -- Reco/True ratio, built from this same variable's ``var_evt_reco_col``
  vs. ``var_evt_truth_col`` (no separate columns needed). Requires the truth-side
  histogram to have actually been filled at map time (only happens when this variable's
  ``ratio_mode`` was already ``"reco_true"`` *before* that batch was mapped -- see
  ``pyanalib.chunked_selection.OverlayHistData.fill_truth_from_df``).
- ``"signal_bkgd"`` -- ratio of two disjoint sets of breakdown-category indices (e.g.
  topology categories) summed from the SAME plot's already-filled ``mc_hist`` -- no extra
  data collection needed. Requires ``ratio_signal_indices``/``ratio_bkgd_indices`` (cuts-order
  indices into ``OverlayHistData.mc_hist``'s category axis) and ``ratio_breakdown_type``
  (which ``breakdown_type`` those indices are valid for -- the ratio is skipped, not
  misindexed, if a plot's actual ``breakdown_type`` doesn't match).

``ratio_label`` optionally overrides the ratio panel's y-axis label (falls back to a
sensible default per mode if omitted); a ``PlotSpec.save_kwargs["ratio_label"]`` still wins
over this if both are set.

``response_matrix`` opts a variable into an automatic true(x)-vs-reco(y) "response matrix"
(a.k.a. migration matrix) heatmap -- a separate plot from the overlay+ratio panel above, built
from this variable's own ``var_evt_reco_col``/``var_evt_truth_col`` (same columns
``ratio_mode="reco_true"`` uses, but paired per-event into a 2D histogram instead of two 1D
ones). See ``pyanalib.chunked_selection.OverlayHistData.fill_response_from_df`` for the map-time
fill (MC only -- gated on this flag, same "must be set BEFORE the batch is mapped" caveat as
``ratio_mode="reco_true"``) and ``pyanalib.response_matrix_plotting.response_matrix_from_histdata``
for the render-time plot (color = raw POT-weighted counts, annotated text = percent of that
column/true-bin's total, i.e. each column sums to ~100%). ``response_matrix_bins`` optionally
overrides the bin edges used for the 2D histogram (defaults to this variable's own ``bins`` --
useful to coarsen a variable whose normal 1D binning would make an illegibly large NxN grid).
``response_matrix_label`` optionally sets a real plot title (``ax.set_title``, blank by default)
-- independent of the "SBND Work in Progress" approval tag, which always renders regardless of
this field, via the same ``analysis_village.plot_style.sbnd_style.add_approval_text`` call (same
position/fontsize) every other plot in the pipeline uses for its own "SBND Internal"/"SBND
Preliminary" tag.
"""

INTEGRATED_VAR_SAVE_NAME = "integrated"


def is_integrated_var_config(var_config) -> bool:
    """True if ``var_config`` is the sentinel single-bin "integrated" measurement."""
    return getattr(var_config, "var_save_name", None) == INTEGRATED_VAR_SAVE_NAME


class VariableConfig:
    """
    A configurable record for setting up a histogrammed variable.
    Analyses subclass this and add ``@classmethod`` factories, or instantiate
    directly with custom parameters.
    """
    def __init__(
        self,
        var_save_name,
        var_plot_name,
        var_labels,
        bins,
        var_evt_reco_col,
        var_evt_truth_col,
        var_nu_col,
        xsec_label,
        category_syst_var_save_name=None,
        ratio_mode=None,
        ratio_label=None,
        ratio_signal_indices=None,
        ratio_bkgd_indices=None,
        ratio_breakdown_type=None,
        response_matrix=False,
        response_matrix_bins=None,
        response_matrix_label=None,
    ):
        self.var_save_name = var_save_name
        self.var_plot_name = var_plot_name
        self.var_labels = var_labels
        self.bins = bins
        self.bin_centers = (bins[:-1] + bins[1:]) / 2.
        self.var_evt_reco_col = var_evt_reco_col
        self.var_evt_truth_col = var_evt_truth_col
        self.var_nu_col = var_nu_col
        self.xsec_label = xsec_label
        # Optional: load category-summary syst for a different variable (same bin count).
        self.category_syst_var_save_name = category_syst_var_save_name
        # Optional: opt this variable into an automatic ratio subplot -- see the class
        # docstring above for the full description of each field.
        self.ratio_mode = ratio_mode
        self.ratio_label = ratio_label
        self.ratio_signal_indices = ratio_signal_indices
        self.ratio_bkgd_indices = ratio_bkgd_indices
        self.ratio_breakdown_type = ratio_breakdown_type
        # Optional: opt this variable into an automatic true-vs-reco response-matrix plot --
        # see the class docstring above for the full description of each field.
        self.response_matrix = response_matrix
        self.response_matrix_bins = response_matrix_bins
        self.response_matrix_label = response_matrix_label
