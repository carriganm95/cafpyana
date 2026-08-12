"""nueNp0Pi's systematics-loading / overlay-plotting wiring: binds the generic
engine in :mod:`pyanalib.syst_loading` / :mod:`pyanalib.overlay_plotting` to
this analysis's category labels/colors (``config/settings.py``), dataset
constants (``config/datasets.py``, ``config/plots.py``), and plot style
(``notebooks/presentation.mplstyle``).

The generic systematics-covariance loading and overlay-histogram rendering
logic lives in :mod:`pyanalib.syst_loading` / :mod:`pyanalib.overlay_plotting`
-- this file only has to be read to see how nueNp0Pi's config plugs into that
generic machinery, not to find the actual plotting/loading logic. Exposes the
same public names as before this split so existing imports (including the
notebook's ``from analysis_village.nueNp0Pi.utils import *``) keep working.

Three pre-existing bugs (found while doing this split, not introduced by it,
preserved as-is rather than guessed at -- see each function's docstring):
1. ``get_syst_unc``'s ``cosmics`` component imports
   ``analysis_village.numucc_1p0pi.syst_cosmics_common.flat_uncorrelated_cov_frac``,
   which does not exist anywhere in this repo.
2. ``load_overlay_syst_cov_frac`` imports
   ``analysis_village.nueNp0Pi.syst_category_summary`` (``load_category_syst_summary``,
   ``total_cov_frac``), which also does not exist anywhere in this repo.
3. ``overlay_hists_from_histdata`` calls ``get_chi2``/``get_chi2_shape`` (chi2-text
   path) and ``Matrix_Decomp`` (``syst_decomp=True`` path) -- none of the three are
   defined or imported anywhere in this repo. Dormant unless you run with real data
   present and either ``textchi2=True`` or ``syst_decomp=True``.
"""
import os

import numpy as np
import pandas as pd
import string
import pickle

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from pyanalib.split_df_helpers import *
from pyanalib.stat_helpers import *
from pyanalib.covariance import *

from makedf.constants import *
from analysis_village.nueNp0Pi.config.settings import *
# Only DETECTOR/EPSILON explicitly (not a wildcard import of all of
# config.datasets) -- this deliberately re-binds DETECTOR to the
# "SBND_nohighyz" value after config.settings' wildcard import above set it
# to "SBND_Gen1" (categories' own default). See
# tests/test_nueNp0Pi_utils.py::test_detector_constant_not_silently_shadowed_incorrectly.
from analysis_village.nueNp0Pi.config.datasets import DETECTOR, EPSILON, save_fig_base_dir
from pyanalib.chunked_selection import multicol_get_series
from analysis_village.nueNp0Pi.config.plots import INTEGRATED_HIST_DUMMY, INTEGRATED_VAR_SAVE_NAME
from pyanalib.syst_disk_layout import (
    FILE_GENIE,
    SUB_GENIE,
    SYST_DISK_ENV,
    category_out_dir,
    category_summary_npz_path,
    syst_disk_paths,
)
from analysis_village.plot_style.sbnd_style import (
    get_textloc_x,
    add_approval_text,
    add_pot_text,
    add_chi2_text,
    add_genie_version_text,
    format_singlebin_plot,
    get_text_color,
    bin_range_labels,
    plot_heatmap,
    plot_frac_unc,
)
from makedf.flux import (
    get_active_volume,
    get_integrated_flux,
    get_xsec_unit,
    print_sbnd_octant_vertex_ranges,
)
from analysis_village.nueNp0Pi.config.datasets import FLUX_FILE
from pyanalib.logging_utils import get_logger

from pyanalib import syst_loading as _sl
from pyanalib import overlay_plotting as _op
from pyanalib.overlay_plotting import get_pot_str, generate_tags

logger = get_logger(__name__)

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.legend import Legend
plt.style.use(os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebooks", "presentation.mplstyle"))
cmap = mpl.cm.viridis
norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)

dpi = 300
fig_ext = ".png"

# ===========================================================================
# Breakdown display registry: {breakdown_type: (labels, colors, hatches_or_None)}
# -- the analysis-specific piece pyanalib.overlay_plotting.overlay_hists_from_histdata
# needs to render any breakdown_type this analysis uses.
# ===========================================================================
_genie_sb_hatches = [None] * len(genie_sb_mode_labels)
for _i in range(5, len(genie_sb_mode_labels), 2):
    _genie_sb_hatches[_i] = '////'

BREAKDOWN_DISPLAY = {
    "pdg": (pdg_labels, pdg_colors, None),
    "topology": (topology_labels, topology_colors, None),
    "genie": (genie_mode_labels, genie_mode_colors, None),
    "genie_sb": (genie_sb_mode_labels, genie_sb_mode_colors, _genie_sb_hatches),
}


# ===========================================================================
# Systematics disk: root resolution + covariance loading
# (pyanalib.syst_loading, generic engine).
# ===========================================================================
def resolve_syst_disk_root(explicit):
    """Return normalized syst disk root from argument or ``SYST_DISK_ROOT`` (see ``SYST_DISK_ENV``)."""
    return _sl.resolve_syst_disk_root(explicit, syst_disk_env=SYST_DISK_ENV)


# Keys accepted by ``get_syst_unc(..., syst_components=...)`` (case-insensitive strings).
SYST_UNC_DISK_KEYS = _sl.SYST_UNC_DISK_KEYS
SYST_UNC_FLAT_KEYS = _sl.SYST_UNC_FLAT_KEYS
SYST_UNC_ALL_KEYS = _sl.SYST_UNC_ALL_KEYS


def _cosmics_flat_uncorrelated_cov_frac(cov_frac):
    """Pre-existing bug, preserved as-is (not fixed): this import target does not exist
    anywhere in this repo. Dormant unless ``get_syst_unc`` is called with ``cosmics``
    included in ``syst_components`` (default: all components, so this DOES run by
    default whenever the syst disk actually has a ``Cosmics/`` tree).
    """
    from analysis_village.numucc_1p0pi.syst_cosmics_common import (
        flat_uncorrelated_cov_frac,
    )
    return flat_uncorrelated_cov_frac(cov_frac)


def get_syst_unc(
    var_config,
    plot=False,
    save_fig=False,
    save_name=None,
    syst_disk_root=None,
    syst_components=None,
    genie_cov_frac_key="genie",
    skip_missing_vars=False,
):
    """Load fractional covariance blocks from the syst-disk tree and combine into total covariance.

    Thin nueNp0Pi wrapper around :func:`pyanalib.syst_loading.get_syst_unc`; see that
    function's docstring for the full parameter reference.
    """
    return _sl.get_syst_unc(
        var_config,
        plot=plot,
        save_fig=save_fig,
        save_name=save_name,
        syst_disk_root=syst_disk_root,
        syst_components=syst_components,
        genie_cov_frac_key=genie_cov_frac_key,
        skip_missing_vars=skip_missing_vars,
        syst_disk_env=SYST_DISK_ENV,
        cosmics_flat_uncorrelated_cov_frac=_cosmics_flat_uncorrelated_cov_frac,
        dpi=dpi,
        fig_ext=fig_ext,
    )


_DEFAULT_SYST_DISK_ROOT = "/exp/sbnd/data/users/munjung/plots/numucc1p0pi/systematics-final"


def _load_syst_category_summary_module():
    """Pre-existing bug, preserved as-is (not fixed): this module does not exist
    anywhere in this repo. Dormant unless ``load_overlay_syst_cov_frac`` /
    ``get_category_summary_syst_unc`` is called directly (both require this loader
    explicitly, so a caller of those two functions is presumed to want the real
    error if the module is genuinely missing).
    """
    from analysis_village.nueNp0Pi.syst_category_summary import (
        load_category_syst_summary,
        total_cov_frac,
    )
    return load_category_syst_summary, total_cov_frac


# Whether analysis_village.nueNp0Pi.syst_category_summary actually exists, checked once
# at import time (cheap -- find_spec doesn't import the module). Used below to decide
# whether overlay_hists_from_histdata's *opportunistic* syst-band lookup should even
# attempt _load_syst_category_summary_module: since that module is a known, permanent
# gap (see the module docstring's bug #2), wiring it in unconditionally meant every
# batch-rendered plot logged "could not load category_syst_summary (No module named
# ...)" -- correctly caught and non-fatal (overlay_plotting.py's syst-resolution path
# already treats a missing loader/None return as "no band available"), but noisy on
# every run. If this module is ever added, this starts wiring it in automatically with
# no further edit needed here.
import importlib.util as _importlib_util
_HAS_SYST_CATEGORY_SUMMARY_MODULE = (
    _importlib_util.find_spec("analysis_village.nueNp0Pi.syst_category_summary") is not None
)


def resolve_category_syst_summary_path(category_syst_summary_path=None, syst_disk_root=None):
    """Path to ``CategorySummary/category_syst_summary.npz`` from export cell."""
    return _sl.resolve_category_syst_summary_path(
        category_syst_summary_path, syst_disk_root,
        default_root=_DEFAULT_SYST_DISK_ROOT, syst_disk_env=SYST_DISK_ENV,
    )


def load_overlay_syst_cov_frac(
    var_config, *, syst_kind="rate", syst_disk_root=None, category_syst_summary_path=None,
):
    """Fractional covariance for overlay bands (default: summed category summary)."""
    return _sl.load_overlay_syst_cov_frac(
        var_config,
        syst_kind=syst_kind,
        syst_disk_root=syst_disk_root,
        category_syst_summary_path=category_syst_summary_path,
        default_root=_DEFAULT_SYST_DISK_ROOT,
        syst_category_summary_loader=_load_syst_category_summary_module,
    )


def get_category_summary_syst_unc(
    var_config, *, syst_kind="rate", syst_disk_root=None, category_syst_summary_path=None,
):
    """Fractional diagonal uncertainty and covariance from ``category_syst_summary.npz``."""
    return _sl.get_category_summary_syst_unc(
        var_config,
        syst_kind=syst_kind,
        syst_disk_root=syst_disk_root,
        category_syst_summary_path=category_syst_summary_path,
        default_root=_DEFAULT_SYST_DISK_ROOT,
        syst_category_summary_loader=_load_syst_category_summary_module,
    )


GENIE_SB_BKGD_RATE_KEY = _sl.GENIE_SB_BKGD_RATE_KEY


def resolve_genie_sb_cov_mat_pkl(genie_sb_cov_mat_pkl=None):
    """Path to ``systematics-genie-SB`` ``GENIE/cov_mat_dict.pkl`` (``genie_bkgd_rate``)."""
    return _sl.resolve_genie_sb_cov_mat_pkl(
        genie_sb_cov_mat_pkl, default_root_base=save_fig_base_dir,
    )


def load_genie_sb_bkgd_rate_cov_frac(var_config, genie_sb_cov_mat_pkl=None):
    """Fractional covariance on background topology rate from ``systematics-genie-SB.ipynb``."""
    return _sl.load_genie_sb_bkgd_rate_cov_frac(
        var_config, genie_sb_cov_mat_pkl, default_root_base=save_fig_base_dir,
    )


# ===========================================================================
# Event clipping + overlay rendering (pyanalib.overlay_plotting, generic engine).
# ===========================================================================
def get_clipped_evts(df, var_col, bins, verbose=False, var_save_name=None):
    """Clip variable to bin range and return weights.

    Thin nueNp0Pi wrapper around :func:`pyanalib.overlay_plotting.get_clipped_evts`,
    binding this analysis's ``INTEGRATED_HIST_DUMMY``/``EPSILON`` convention.
    """
    return _op.get_clipped_evts(
        df, var_col, bins, verbose=verbose, var_save_name=var_save_name,
        integrated_hist_dummy=INTEGRATED_HIST_DUMMY, epsilon=EPSILON,
        integrated_var_save_name=INTEGRATED_VAR_SAVE_NAME,
    )


# ===========================================================================
# Back-compat re-exports: internal helpers with no analysis-specific behavior,
# kept importable under their original names for ``from ...utils import *``
# (no external .py call sites as of this split -- verified by repo-wide grep --
# but the notebook's wildcard import means any of these could be used ad hoc
# in a cell).
# ===========================================================================
_fail_syst_disk = _sl._fail_syst_disk
_as_1d_float_array = _op._as_1d_float_array
_var_weights_for_cut = _op._var_weights_for_cut
_overlay_bkgd_syst_sigma = _op._overlay_bkgd_syst_sigma
_overlay_draw_bkgd_syst_band = _op._overlay_draw_bkgd_syst_band
_overlay_add_poisson_mc_stat_to_band = _op._overlay_add_poisson_mc_stat_to_band
_overlay_syst_sigma = _op._overlay_syst_sigma
_overlay_chi2_valid_bins = _op._overlay_chi2_valid_bins
_overlay_compute_chi2 = _op._overlay_compute_chi2
_overlay_histdata_legend_mc_index_order = _op._overlay_histdata_legend_mc_index_order
_CATEGORY_SYST_SUMMARY_CACHE = _sl._CATEGORY_SYST_SUMMARY_CACHE
_GENIE_SB_COV_MAT_CACHE = _sl._GENIE_SB_COV_MAT_CACHE


def overlay_hists_from_histdata(histdata, **kwargs):
    """Render an overlay histogram plot from precomputed histograms.

    Thin nueNp0Pi wrapper around
    :func:`pyanalib.overlay_plotting.overlay_hists_from_histdata`, binding this
    analysis's breakdown label/color registry and systematics-loading hooks. See
    that function's docstring for the full parameter reference (``var_config``,
    ``plot_labels``, ``syst``, ``show_bkgd_syst_band``, ``textchi2``, ``ratio``, …
    all pass through unchanged via ``**kwargs``).
    """
    kwargs.setdefault("syst_default_root", _DEFAULT_SYST_DISK_ROOT)
    if _HAS_SYST_CATEGORY_SUMMARY_MODULE:
        kwargs.setdefault("syst_category_summary_loader", _load_syst_category_summary_module)
    kwargs.setdefault("bkgd_syst_loader", load_genie_sb_bkgd_rate_cov_frac)
    kwargs.setdefault("genie_sb_patch_colors", genie_mode_colors)
    kwargs.setdefault("dpi", dpi)
    kwargs.setdefault("fig_ext", fig_ext)
    return _op.overlay_hists_from_histdata(histdata, BREAKDOWN_DISPLAY, **kwargs)
