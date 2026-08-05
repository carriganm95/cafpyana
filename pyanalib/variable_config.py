"""Generic container for a single "variable configuration".

A ``VariableConfig`` bundles everything needed to histogram one physics
variable across an analysis's selection/efficiency/unfolding pipeline: the
reco/truth/nu dataframe column paths, bin edges, and plot labels. It has no
built-in physics -- it's just a record shape plus the ``bin_centers``
convenience derived from ``bins``.

Analyses should subclass ``VariableConfig`` and add their own
``@classmethod`` factories, one per variable they care about (e.g.
``muon_momentum``, TKI variables, ...). See
``analysis_village/nueNp0Pi/variable_configs.py`` for an example with ~70
variables built this way.

``INTEGRATED_VAR_SAVE_NAME`` / ``is_integrated_var_config`` are a small shared
convention: several analyses use a sentinel ``var_save_name == "integrated"``
VariableConfig to represent a single-bin "all events" measurement (for total/
integrated-flux cross sections). The specific dummy column/value used to fill
that single bin is analysis-specific (depends on each analysis's own on-disk
schema and production conventions), so it is NOT defined here -- only the
shared sentinel name and the predicate that checks for it.
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
