"""Generic overlay-histogram rendering: turn a precomputed
:class:`pyanalib.chunked_selection.OverlayHistData` into the stacked MC/data
comparison plot (the entry point the chunked event-selection framework uses
after aggregating histograms across input files).

No hardcoded analysis-specific breakdown labels/colors, output paths, or
plot-style calls live here beyond the shared SBND style helpers in
``analysis_village.plot_style.sbnd_style`` (a repo-wide style library used by
every analysis, not owned by one -- the one accepted exception to "pyanalib
never imports analysis_village", same tier as ``makedf``). Analysis packages
(e.g. ``analysis_village.nueNp0Pi.utils``) supply their own breakdown
label/color registry and, for the optional background-systematics band, a
loader hook. Mirrors the ``pyanalib.dataset_paths`` / ``pyanalib.syst_loading``
split: this module is the reusable engine, the analysis package is the wiring.
"""

from __future__ import annotations

import string
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.legend import Legend

from pyanalib.logging_utils import get_logger
from pyanalib.stat_helpers import return_data_stat_err
from pyanalib.variable_config import INTEGRATED_VAR_SAVE_NAME
from pyanalib.chunked_selection import get_clipped_evts as _get_clipped_evts_shared
from pyanalib.syst_loading import load_overlay_syst_cov_frac

from analysis_village.plot_style.sbnd_style import (
    get_textloc_x,
    add_approval_text,
    add_pot_text,
    add_chi2_text,
    add_genie_version_text,
    format_singlebin_plot,
)

logger = get_logger(__name__)

# Type alias: breakdown_type -> (labels, colors, hatches-or-None).
BreakdownDisplay = Dict[str, Tuple[Sequence[str], Sequence[str], Optional[Sequence[Optional[str]]]]]


# ===========================================================================
# General helpers: formatting, arrays, event clipping.
# ===========================================================================
def get_pot_str(tot_pot):
    pot_str = "{:.2e}".format(tot_pot).replace("e+0", "e").replace("e+", "e")\
        .replace("e", " $\\times 10^{") + "}$"
    pot_str = pot_str.replace(".00", "")
    pot_str = pot_str.replace("$\\times 10^{", "$\\times 10^{")
    pot_str = pot_str.replace("}$", "}")
    pot_str = pot_str if pot_str.endswith("$") else pot_str + "$"
    pot_str = pot_str.replace(" $", "$")
    return pot_str


def generate_tags(end_tag=""):
    tags = []
    for first in string.ascii_lowercase:
        for second in string.ascii_lowercase:
            tag = first + second
            if tag == end_tag:
                break
            tags.append(tag)
        if tag == end_tag:
            break
    return tags


def _as_1d_float_array(values, name="values"):
    """Coerce event-level values to a 1D float ndarray (handles duplicate-column DataFrames)."""
    if isinstance(values, pd.DataFrame):
        if values.shape[1] == 1:
            values = values.iloc[:, 0]
        else:
            raise ValueError(
                "%s matched %d columns; expected a single event-level series"
                % (name, values.shape[1])
            )
    arr = np.ravel(np.asarray(values, dtype=float))
    return arr


def _var_weights_for_cut(var, weights, cut):
    """Return aligned 1D reco-variable and POT-weight arrays for one boolean category mask."""
    mask = np.asarray(cut, dtype=bool)
    v = _as_1d_float_array(var, name="variable")
    w = _as_1d_float_array(weights, name="pot_weight")
    n = len(mask)
    if len(v) != n or len(w) != n:
        raise ValueError(
            "Event array length mismatch: variable=%d pot_weight=%d mask=%d"
            % (len(v), len(w), n)
        )
    return v[mask], w[mask]


def _overlay_hist_weighted(values, weights, bins):
    """Weighted 1D histogram + per-bin stat. error (``sqrt(sum(weights**2))``)."""
    values = _as_1d_float_array(values, name="ratio_values")
    if weights is None:
        weights = np.ones_like(values)
    else:
        weights = _as_1d_float_array(weights, name="ratio_weights")
    if len(weights) != len(values):
        raise ValueError(
            "ratio variable/weight length mismatch: values=%d weights=%d"
            % (len(values), len(weights))
        )
    hist, _ = np.histogram(values, bins=bins, weights=weights)
    err2, _ = np.histogram(values, bins=bins, weights=weights ** 2)
    return hist.astype(float), np.sqrt(err2)


def _overlay_custom_ratio(num_values, denom_values, num_weights, denom_weights, bins):
    """Histogram two raw per-event variables (e.g. reco/true) and return their ratio.

    Returns ``(num_hist, denom_hist, ratio, ratio_err)``, all length ``len(bins) - 1``.
    ``ratio_err`` propagates the two histograms' independent Poisson-style stat. errors
    in quadrature (relative errors added in quadrature, scaled by the ratio) -- an
    approximation that ignores any event-by-event correlation between numerator and
    denominator (e.g. the same event's reco and true value), so it's a useful quick-look
    uncertainty rather than a rigorous one.
    """
    num_hist, num_err = _overlay_hist_weighted(num_values, num_weights, bins)
    denom_hist, denom_err = _overlay_hist_weighted(denom_values, denom_weights, bins)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(denom_hist != 0, num_hist / denom_hist, 0.0)
        num_rel_err2 = np.where(num_hist != 0, (num_err / np.where(num_hist != 0, num_hist, 1.0)) ** 2, 0.0)
        denom_rel_err2 = np.where(
            denom_hist != 0, (denom_err / np.where(denom_hist != 0, denom_hist, 1.0)) ** 2, 0.0
        )
        ratio_err = np.abs(ratio) * np.sqrt(num_rel_err2 + denom_rel_err2)
    ratio = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)
    ratio_err = np.nan_to_num(ratio_err, nan=0.0, posinf=0.0, neginf=0.0)
    return num_hist, denom_hist, ratio, ratio_err


def get_clipped_evts(
    df, var_col, bins, verbose=False, var_save_name=None,
    *,
    integrated_hist_dummy: float,
    epsilon: float,
    integrated_var_save_name: str = INTEGRATED_VAR_SAVE_NAME,
):
    """Clip variable to bin range and return weights.

    Delegates to :func:`pyanalib.chunked_selection.get_clipped_evts` for the normal path;
    the only local behavior is the single-bin "integrated" sentinel (all events placed in
    one dummy bin for total/integrated-flux cross-section plots), via
    ``var_save_name == integrated_var_save_name``. ``integrated_hist_dummy`` is the
    analysis's production-convention dummy value for that bin (e.g. the numeric value
    every event's dummy variable is filled with).
    """
    if var_save_name == integrated_var_save_name:
        var = np.full(len(df), integrated_hist_dummy, dtype=float)
        var = _as_1d_float_array(var, name=str(var_col))
        var = np.clip(var, bins[0], bins[-1] - epsilon)
        if 'pot_weight' in df.columns:
            weights = df.loc[:, 'pot_weight']
        else:
            if verbose:
                print("No pot_weight column found, return 1 as pot scale (expected for data)")
            weights = np.ones_like(var)
        weights = _as_1d_float_array(weights, name="pot_weight")
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        return var, weights
    return _get_clipped_evts_shared(df, var_col, bins, verbose=verbose)


# ===========================================================================
# Overlay-plot helpers: legend order, syst bands, chi2.
# ===========================================================================
def _overlay_histdata_legend_mc_index_order(n_layers, has_dirt):
    """Indices into mpl stacked layers for legend: physics high→…→low, Dirt, Cosmic.

    Stacked ``hist`` uses dataset order bottom→top: with dirt,
    layer 0 = Dirt, 1 = Cosmic, layers 2… = GENIE/topology blocks with signal
    at ``n_layers - 1``. Desired legend after Data:
    signal (top) … physics … Dirt … Cosmic.
    """
    if n_layers <= 0:
        return []
    if not has_dirt:
        return list(range(n_layers - 1, -1, -1))
    return list(range(n_layers - 1, 1, -1)) + [0, 1]


def _overlay_bkgd_syst_sigma(total_mc_bkgd, bkgd_frac_cov):
    """Per-bin 1σ GENIE background-rate uncertainty (absolute event units)."""
    total_mc_bkgd = np.asarray(total_mc_bkgd, dtype=float)
    frac_diag = np.maximum(np.diag(np.asarray(bkgd_frac_cov, dtype=float)), 0.0)
    with np.errstate(invalid="ignore"):
        return np.sqrt(frac_diag) * total_mc_bkgd


def _overlay_draw_bkgd_syst_band(
    ax,
    bin_centers,
    bins,
    total_mc,
    bkgd_syst_err,
    *,
    edgecolor="darkorange",
    hatch="+++",
    label="Bkgd. GENIE unc.",
    zorder=9,
):
    bkgd_syst_err = np.asarray(bkgd_syst_err, dtype=float)
    ax.bar(
        bin_centers,
        2 * bkgd_syst_err,
        width=np.diff(bins),
        bottom=np.asarray(total_mc, dtype=float) - bkgd_syst_err,
        facecolor="none",
        hatch=hatch,
        linewidth=0.0,
        edgecolor=edgecolor,
        label=label,
        zorder=zorder,
    )


def _overlay_add_poisson_mc_stat_to_band(syst_explicit, load_syst_from_summary):
    """``category_syst_summary`` totals already include MC stat.; skip Poisson MC stat on the band."""
    return not (load_syst_from_summary and not syst_explicit)


def _overlay_syst_sigma(total_mc, mc_stat_err, syst_frac_cov, *, add_poisson_mc_stat):
    """Per-bin 1σ systematic uncertainty for hatched MC bands (absolute event units)."""
    total_mc = np.asarray(total_mc, dtype=float)
    syst_err_frac = np.sqrt(np.maximum(np.diag(np.asarray(syst_frac_cov, dtype=float)), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        syst_sigma = syst_err_frac * total_mc
        if add_poisson_mc_stat:
            mc_stat_err_frac = np.where(total_mc != 0, np.asarray(mc_stat_err, dtype=float) / total_mc, 0.0)
            syst_sigma = np.sqrt(syst_sigma ** 2 + (mc_stat_err_frac * total_mc) ** 2)
    return syst_sigma


def _overlay_chi2_valid_bins(total_data, total_mc):
    """Bins with MC or data content (skip empty bins in χ²)."""
    total_data = np.asarray(total_data, dtype=float)
    total_mc = np.asarray(total_mc, dtype=float)
    return (total_mc > 0) | (total_data > 0)


def _overlay_compute_chi2(total_data, total_mc, syst_frac_cov, data_eylow, data_eyhigh):
    """χ² using diagonal errors consistent with hatched syst. band + data error bars.

    The hatched MC band is drawn from ``√diag(cov)`` only.  Using the full
    covariance inverse here would charge shape residuals against a nearly
    rank-1 (rate-like) WireMod/unisim matrix and can make χ² explode even when
    every point sits inside the band.  The displayed χ² therefore uses the
    same diagonal variances as the band (syst) plus data stat. errors.

    Also computes a shape-only χ² (MC normalized to the data integral) with the
    same diagonal treatment.

    NOTE (pre-existing bug, preserved as-is -- not fixed as part of the
    pyanalib move, per explicit instruction): ``get_chi2``/``get_chi2_shape``
    below are not defined or imported anywhere in this repo. This function
    raises ``NameError`` if it's ever actually reached (i.e. ``histdata.has_data``
    is True and ``syst is not None``) until those are supplied.
    """
    total_data = np.asarray(total_data, dtype=float)
    total_mc = np.asarray(total_mc, dtype=float)
    valid = _overlay_chi2_valid_bins(total_data, total_mc)
    if not np.any(valid):
        return None, None, None, None, None, None, None, None

    data_stat_var = (
        0.5
        * (
            np.asarray(data_eylow, dtype=float)
            + np.asarray(data_eyhigh, dtype=float)
        )
    ) ** 2
    # Absolute syst variance matching the hatched band: (√diag(frac) * MC)^2
    syst_frac_cov = np.nan_to_num(
        np.asarray(syst_frac_cov, dtype=float), nan=0.0, posinf=0.0, neginf=0.0
    )
    syst_var = np.maximum(np.diag(syst_frac_cov), 0.0) * (total_mc ** 2)
    combined_var = syst_var + data_stat_var

    d = total_data[valid]
    m = total_mc[valid]
    var = combined_var[valid]
    good = var > 0
    d, m, var = d[good], m[good], var[good]
    ndof = int(len(d))
    if ndof == 0:
        return None, None, None, None, None, None, None, None

    c = np.diag(var)
    chi2_total, p_val = get_chi2(d, m, c)  # noqa: F821 -- see docstring note above
    chi2_reduced = chi2_total / ndof if ndof > 0 else None

    # Shape-only χ²: MC normalized to data integral; diagonal cov scaled with MC².
    chi2_shape = None
    p_val_shape = None
    ndof_shape = None
    if m.sum() > 0 and ndof > 1:
        scale = d.sum() / m.sum()
        var_shape = syst_var[valid][good] * scale**2 + data_stat_var[valid][good]
        good_s = var_shape > 0
        n_shape = int(np.sum(good_s))
        if n_shape > 1:
            chi2_shape, p_val_shape = get_chi2_shape(  # noqa: F821 -- see docstring note above
                d[good_s], m[good_s], np.diag(var_shape[good_s])
            )
            ndof_shape = n_shape - 1  # overall rate floated

    chi2_pull = np.full_like(total_data, np.nan, dtype=float)
    pull_idx = np.flatnonzero(valid)[good]
    chi2_pull[pull_idx] = (d - m) / np.sqrt(np.maximum(var, 1e-10))
    return (
        chi2_total,
        chi2_reduced,
        p_val,
        ndof,
        chi2_pull,
        chi2_shape,
        p_val_shape,
        ndof_shape,
    )


def _resolve_overlay_syst_cov_frac(
    var_config,
    syst,
    *,
    syst_kind="rate",
    syst_disk_root=None,
    category_syst_summary_path=None,
    load_syst_from_summary=True,
    syst_default_root: Optional[str] = None,
    syst_category_summary_loader: Optional[Callable[[], Tuple[Callable, Callable]]] = None,
):
    """Use explicit *syst* or load from ``category_syst_summary.npz`` when enabled."""
    if syst is not None:
        return np.asarray(syst, dtype=np.float64)
    if not load_syst_from_summary or var_config is None:
        return None
    if syst_category_summary_loader is None:
        return None
    try:
        return load_overlay_syst_cov_frac(
            var_config,
            syst_kind=syst_kind,
            syst_disk_root=syst_disk_root,
            category_syst_summary_path=category_syst_summary_path,
            default_root=syst_default_root,
            syst_category_summary_loader=syst_category_summary_loader,
        )
    except Exception as ex:
        logger.warning("could not load category_syst_summary (%s)", ex)
        return None


# ===========================================================================
# Main entry point: render an overlay plot from a precomputed OverlayHistData.
# ===========================================================================
def overlay_hists_from_histdata(
    histdata,
    breakdown_display: BreakdownDisplay,
    var_config=None,
    plot_labels=["", "", ""],
    ax_ylim_ratio=1.5,
    ratio=False,
    ratio_vars: Optional[Tuple[Any, Any]] = None,
    ratio_weights: Optional[Tuple[Any, Any]] = None,
    ratio_bins: Optional[Any] = None,
    ratio_label: Optional[str] = None,
    ratio_ylim: Optional[Tuple[float, float]] = None,
    density=False,
    syst=None,
    syst_kind="xsec",
    syst_disk_root=None,
    category_syst_summary_path=None,
    load_syst_from_summary=True,
    show_bkgd_syst_band=False,
    bkgd_syst_frac_cov=None,
    genie_sb_cov_mat_pkl=None,
    bkgd_syst_band_color="darkorange",
    syst_decomp=False,  # False -> hatched band (``selected_events.ipynb`` style)
    textchi2=False,
    vline=None,
    textloc=[0.05, 0.55],
    approval="internal",
    plot=True,
    save_fig=False,
    save_name=None,
    verbose_hist=False,
    *,
    syst_default_root: Optional[str] = None,
    syst_category_summary_loader: Optional[Callable[[], Tuple[Callable, Callable]]] = None,
    bkgd_syst_loader: Optional[Callable[[Any, Any], np.ndarray]] = None,
    genie_sb_patch_colors: Optional[Sequence[str]] = None,
    pot_annotation_text: str = "8.8 $\\times 10^{19}$ POT",
    show_genie_label: bool = True,
    dpi: int = 300,
    fig_ext: str = ".png",
):
    """Render an overlay histogram plot from precomputed histograms.

    The plot output is bit-for-bit identical to the raw-dataframe ``overlay_hists(...)``
    path (since removed) when given equivalent inputs.

    Parameters
    ----------
    histdata : OverlayHistData
        Container with breakdown_type, bins, per-category MC histograms,
        intime/dirt/data histograms, and corresponding sum-of-weights^2
        arrays (for stat errors). Cosmic contribution uses intime when present,
        else offbeam.
    breakdown_display : BreakdownDisplay
        ``{breakdown_type: (labels, colors, hatches_or_None)}`` -- the analysis's own
        category labels/colors/hatches for every ``breakdown_type`` it uses (e.g.
        ``"pdg"``, ``"topology"``, ``"genie"``, ``"genie_sb"``). ``KeyError`` if
        ``histdata.breakdown_type`` isn't a key.
    var_config : VariableConfig
        Carries bins, labels, var_save_name (for "integrated" formatting).
    ratio, ratio_vars, ratio_weights, ratio_bins, ratio_label, ratio_ylim
        Control the bottom ratio subplot (only drawn when ``ratio=True``). By default
        (``ratio_vars=None``) the panel shows Data/MC, computed from ``histdata`` exactly
        as before -- fully backward compatible. Pass ``ratio_vars=(num_values, denom_values)``
        (two raw per-event 1D arrays, e.g. a reco column and the matching truth column) to
        instead show the ratio of those two variables' histograms, e.g. Reco/True. Optional
        ``ratio_weights=(num_weights, denom_weights)`` supplies per-event weights for each
        (defaults to unweighted, i.e. all ones); ``ratio_bins`` overrides the bin edges used
        to histogram ``ratio_vars`` (defaults to this plot's ``bins``, which is normally
        correct since reco/truth share ``var_config.bins``). ``ratio_label`` sets the panel's
        y-axis label (defaults to ``"Data/MC"`` in the built-in mode, ``"Ratio"`` in custom
        mode). ``ratio_ylim`` overrides the panel's y-limits (default stays ``(0, 2)``, same
        as before). In custom mode the Data/MC syst band and data points are not drawn on
        the ratio panel -- only the custom ratio -- since they describe a different quantity.
    syst_default_root, syst_category_summary_loader
        Passed through to :func:`pyanalib.syst_loading.load_overlay_syst_cov_frac` when
        ``syst`` is omitted and ``load_syst_from_summary`` is True. Both required for that
        fallback path to actually load anything; if either is omitted, no category-summary
        syst band is drawn (silently, matching ``skip_missing_vars``-style tolerance).
    bkgd_syst_loader
        Optional ``(var_config, genie_sb_cov_mat_pkl) -> frac_cov`` callable used when
        ``show_bkgd_syst_band=True`` and ``bkgd_syst_frac_cov`` is not given explicitly.
    genie_sb_patch_colors
        Only used when ``histdata.breakdown_type == "genie_sb"``: the legend swatch color
        per collapsed (signal+background) mode pair -- typically the plain GENIE-mode
        palette (one color per mode, not per signal/background pair). Falls back to the
        ``genie_sb`` entry's own (paired) colors from ``breakdown_display`` if omitted.
    pot_annotation_text
        Text for the top-right POT annotation. Defaults to a fixed placeholder string
        (a pre-existing behavior, not derived from ``histdata`` or ``plot_labels`` --
        pass the real POT string explicitly if you want it to reflect this plot's exposure).
    show_genie_label
        Set False to suppress the "GENIE vX.Y.Z ..." watermark (bottom-right). Already
        skipped automatically for ``breakdown_type == "pdg"``.
    Other arguments behave identically to the removed ``overlay_hists()``.
    """

    breakdown_type = histdata.breakdown_type
    bins = np.asarray(histdata.bins)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    n_bins = len(bins) - 1

    # ---- choose breakdown labels & colors (cosmic-first order to match cuts)
    if breakdown_type not in breakdown_display:
        raise ValueError(
            "Invalid breakdown_type: %r (no entry in breakdown_display; have: %s)"
            % (breakdown_type, sorted(breakdown_display))
        )
    labels, colors, hatches = breakdown_display[breakdown_type]
    labels, colors = list(labels), list(colors)
    hatches = list(hatches) if hatches is not None else None

    # Draw MC/cosmic/dirt stack whenever there is anything to show — do **not** rely on
    # ``has_mc`` alone (merged pickles / older chunks can have nonzero ``mc_hist`` with a
    # stale ``has_mc`` flag, or the inverse).
    mc_hist_sum = float(np.sum(histdata.mc_hist)) if histdata.mc_hist is not None else 0.0
    plot_mc_stack = histdata.mc_hist is not None and (
        histdata.has_mc
        or mc_hist_sum != 0.0
        or histdata.has_intime
        or getattr(histdata, "has_offbeam", False)
        or histdata.has_dirt
    )

    # ---- MC: per-category histograms (already in cuts/cosmic-first order)
    if plot_mc_stack:
        each_mc_hist_data = [histdata.mc_hist[i] for i in range(histdata.mc_hist.shape[0])]
        each_mc_hist_err2 = [histdata.mc_err2[i] for i in range(histdata.mc_err2.shape[0])]
        total_mc = np.sum(histdata.mc_hist, axis=0).astype(float)
        total_mc_err2 = np.sum(histdata.mc_err2, axis=0).astype(float)
        mc_stat_err = np.sqrt(total_mc_err2)

        # For stacked plotting we fake events at bin_centers with weights = histogram values
        var_categ = [bin_centers] * len(each_mc_hist_data)
        weights_categ = [h.copy() for h in each_mc_hist_data]

        total_mc_bkgd = None
    else:
        each_mc_hist_data = None
        each_mc_hist_err2 = None
        total_mc = None
        mc_stat_err = None
        var_categ = None
        weights_categ = None
        total_mc_bkgd = None

    # Cosmic estimate: prefer intime, fall back to offbeam.
    cosmic_hist_bins = None
    if histdata.has_intime:
        cosmic_hist_bins = histdata.intime_hist.astype(float)
    elif getattr(histdata, "has_offbeam", False):
        cosmic_hist_bins = histdata.offbeam_hist.astype(float)

    # Merge scaled cosmic estimate into MC category 0 (same as the removed
    # ``overlay_hists``: concat events). For pre-binned histograms, **add** bin
    # contents — do not duplicate ``bin_centers`` (that doubles fake samples per
    # bin and breaks stacking / legend fraction accounting).
    if cosmic_hist_bins is not None and var_categ is not None:
        weights_categ[0] = np.asarray(weights_categ[0], dtype=float) + np.asarray(
            cosmic_hist_bins, dtype=float
        )
        total_mc = np.asarray(total_mc, dtype=float) + np.asarray(cosmic_hist_bins, dtype=float)

    # ---- Dirt: prepend a new category at front (cuts order: dirt-first)
    if histdata.has_dirt:
        total_dirt = histdata.dirt_hist.astype(float)
        if var_categ is not None:
            var_categ = [bin_centers] + var_categ
            weights_categ = [total_dirt.copy()] + weights_categ
            colors = colors + ["black"]
            labels = labels + ["Low E\nDirt"]
            total_mc = total_mc + total_dirt
    else:
        total_dirt = None

    # ---- Data
    if histdata.has_data:
        total_data = histdata.data_hist.astype(float)
        sum_data = np.sum(total_data)
        # compute asymmetric stat errors from data counts
        # (note: when bin contents come from POT-weighted off-beam they aren't
        # raw counts -- callers should use raw-count data for proper stat errs)
        data_eylow, data_eyhigh = return_data_stat_err(total_data)

        if total_mc is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                data_ratio = np.where(total_mc != 0, total_data / total_mc, 0.0)
                data_ratio_eylow = np.where(total_mc != 0, data_eylow / total_mc, 0.0)
                data_ratio_eyhigh = np.where(total_mc != 0, data_eyhigh / total_mc, 0.0)
            data_ratio = np.nan_to_num(data_ratio, nan=-999.)
            data_ratio_eylow = np.nan_to_num(data_ratio_eylow, nan=0.)
            data_ratio_eyhigh = np.nan_to_num(data_ratio_eyhigh, nan=0.)
        else:
            data_ratio = data_ratio_eylow = data_ratio_eyhigh = None
    else:
        total_data = None
        sum_data = None
        data_eylow = data_eyhigh = None
        data_ratio = data_ratio_eylow = data_ratio_eyhigh = None

    # ---- Density: area-normalize MC to data
    density_factor = 1.0
    if plot_mc_stack and histdata.has_data and density:
        mc_area = np.sum(total_mc)
        data_area = np.sum(total_data)
        density_factor = (data_area / mc_area) if mc_area > 0 else 1.0
        weights_categ = [np.asarray(w) * density_factor for w in weights_categ]
        total_mc = total_mc * density_factor

    if plot_mc_stack and histdata.has_mc and total_mc is not None:
        # each_mc_hist_data is in histdata.mc_hist's native order, i.e. get_cuts_fn's
        # "cuts order" -- confirmed (empirically, against real data, and by reading
        # every current get_*_category function in analysis_village.nueNp0Pi.selections)
        # to be SIGNAL FIRST, matching breakdown_display's label/color order. Signal is
        # each_mc_hist_data[0], not [-1].
        hist_signal = np.asarray(each_mc_hist_data[0], dtype=float)
        if density:
            hist_signal = hist_signal * density_factor
        total_mc_bkgd = np.asarray(total_mc, dtype=float) - hist_signal

    # Reverse colors/labels so the LAST entry is signal (matches
    # _overlay_histdata_legend_mc_index_order's documented stacking convention:
    # dirt, cosmic, then physics categories with signal drawn last/on top).
    #
    # weights_categ/var_categ (built straight from histdata.mc_hist, in get_cuts_fn's
    # signal-first "cuts order") must be reversed the SAME way, or colors[i]/labels[i]
    # end up paired with the WRONG category's actual counts at every index but the
    # dirt slot (which was deliberately pre-positioned for this reversal -- see the
    # dirt-prepend block above). This was a real, pre-existing bug (present before
    # this file existed, in the original analysis_village/nueNp0Pi/utils.py this was
    # split from) confirmed against real data: the "topology" breakdown's legend was
    # showing background labels (NC, Other, ...) attached to signal event counts.
    colors, labels = colors[::-1], labels[::-1]
    if var_categ is not None:
        if histdata.has_dirt:
            # Dirt is already at index 0 in both arrays (prepended to
            # var_categ/weights_categ, appended to colors/labels before the reversal
            # above) -- reverse only the physics-category tail, leave dirt in place.
            var_categ = [var_categ[0]] + var_categ[1:][::-1]
            weights_categ = [weights_categ[0]] + weights_categ[1:][::-1]
        else:
            var_categ = var_categ[::-1]
            weights_categ = weights_categ[::-1]

    if verbose_hist and var_categ is not None:
        layer_totals = [float(np.sum(np.asarray(w, dtype=float))) for w in weights_categ]
        dtot = float(np.sum(histdata.data_hist)) if histdata.has_data else 0.0
        print(
            f"[overlay_histdata] var={histdata.var_save_name!r} breakdown={breakdown_type!r} "
            f"plot_mc_stack={plot_mc_stack} has_mc={histdata.has_mc} "
            f"sum(mc_hist)={mc_hist_sum:.6g} n_layers={len(weights_categ)} "
            f"layer_totals={layer_totals} sum_layers={sum(layer_totals):.6g} "
            f"has_data={histdata.has_data} sum(data_hist)={dtot:.6g}",
            flush=True,
        )

    # ============ plot template ============
    custom_ratio = ratio_vars is not None
    resolved_ratio_label = ratio_label if ratio_label is not None else ("Ratio" if custom_ratio else "Data/MC")
    resolved_ratio_ylim = ratio_ylim if ratio_ylim is not None else (0., 2.)
    if ratio:
        fig, axs = plt.subplots(2, 1, figsize=(8.5, 8.5),
                               sharex=True, gridspec_kw={'height_ratios': [4, 1]})
        ax, ax_r = axs[0], axs[1]
        fig.subplots_adjust(hspace=0.1)
        ax_r.axhline(1.0, color='red', linestyle='--', linewidth=1)
        ax_r.set_ylim(*resolved_ratio_ylim)
        ax_r.set_xlabel(plot_labels[0], fontsize=20)
        ax_r.set_ylabel(resolved_ratio_label, fontsize=16)
        # Solid gridlines on both axes (horizontal + vertical), major ticks only --
        # minor gridlines were tried and reverted per explicit request (too dense;
        # the sparser major-only spacing from the upper panel was preferred).
        ax_r.grid(True, linestyle='-')
        ax_r.tick_params(axis='both', which='major', labelsize=15)
        ax_r.tick_params(axis='both', which='minor', labelsize=13)
    else:
        fig, ax = plt.subplots(figsize=(8.5, 7))
        ax.set_xlabel(plot_labels[0], fontsize=20)

    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylabel(plot_labels[1], fontsize=20)
    ax.set_title(plot_labels[2], fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=13)
    # Solid gridlines on both axes (horizontal + vertical) on the main/upper panel too
    # (matches the ratio panel below it, when present) -- applies to both the ratio and
    # non-ratio (single-panel) layouts since this runs after the if/else above.
    # Major ticks only (no minorticks_on()) -- keeps the sparser gridline spacing.
    ax.grid(True, linestyle='-')

    # ============ plot histograms ============
    mc_stack = None
    breakdown_fractions = None
    if var_categ is not None:
        if breakdown_type == "genie_sb":
            # GENIE S/B uses mpl stacked hist + hatch overlay (unchanged).
            mc_stack, _, _ = ax.hist(var_categ,
                                     weights=weights_categ,
                                     bins=bins,
                                     stacked=True,
                                     color=colors,
                                     linewidth=0,
                                     edgecolor='none',
                                     histtype='stepfilled')

            breakdown_accum = [np.sum(this_mode) for this_mode in mc_stack]
            # Incremental (non-cumulative) count per layer, in the same (cuts) order as
            # ``labels``/``colors`` at this point -- ``breakdown_accum[i]`` is the
            # cumulative stacked sum through layer i (mpl ``hist(..., stacked=True)``).
            breakdown_counts = [breakdown_accum[0]] + \
                [(breakdown_accum[i+1] - breakdown_accum[i]) for i in range(len(breakdown_accum) - 1)]
            if breakdown_accum[-1] > 0:
                breakdown_fractions = [c / breakdown_accum[-1] for c in breakdown_counts]
            else:
                breakdown_fractions = [0.0 for _ in breakdown_counts]

            bottom = np.zeros(n_bins)
            cuts_order_hatches = (hatches[::-1] if hatches is not None else [None] * len(var_categ))
            for i, (v, w, h) in enumerate(zip(var_categ, weights_categ, cuts_order_hatches)):
                hist_vals, _ = np.histogram(v, weights=w, bins=bins)
                ax.bar(bin_centers, hist_vals, width=np.diff(bins),
                       bottom=bottom, color='none', hatch=h,
                       edgecolor='white', linewidth=0.0, align='center')
                bottom += hist_vals
        else:
            # Explicit stacked bars — ``ax.hist(..., stacked=True)`` with synthetic bin-centre
            # samples is fragile across mpl versions; bars match the notebook intent.
            bottom = np.zeros(n_bins, dtype=float)
            for w, col in zip(weights_categ, colors):
                vals = np.asarray(w, dtype=float)
                ax.bar(
                    bin_centers,
                    vals,
                    width=np.diff(bins),
                    bottom=bottom,
                    align="center",
                    color=col,
                    edgecolor="none",
                    linewidth=0,
                    zorder=2,
                )
                bottom = bottom + vals
            layer_integrals = [float(np.sum(np.asarray(w, dtype=float))) for w in weights_categ]
            tot_int = float(sum(layer_integrals))
            if tot_int > 0:
                breakdown_fractions = [li / tot_int for li in layer_integrals]
            else:
                breakdown_fractions = [0.0] * len(layer_integrals)

    chi2_val = None
    chi2_reduced = None
    p_val = None
    ndof = None
    chi2_pull = None
    chi2_shape_val = None
    p_val_shape = None
    ndof_shape = None
    syst_err = syst_err_norm = syst_err_mixed = syst_err_shape = None
    bkgd_syst_err = None

    syst_explicit = syst is not None
    syst = _resolve_overlay_syst_cov_frac(
        var_config,
        syst,
        syst_kind=syst_kind,
        syst_disk_root=syst_disk_root,
        category_syst_summary_path=category_syst_summary_path,
        load_syst_from_summary=load_syst_from_summary,
        syst_default_root=syst_default_root,
        syst_category_summary_loader=syst_category_summary_loader,
    )

    if syst is not None and total_mc is not None:
        add_poisson_mc_stat = _overlay_add_poisson_mc_stat_to_band(
            syst_explicit, load_syst_from_summary
        )
        cov_norm, cov_mixed, cov_shape = Matrix_Decomp(total_mc, syst * (total_mc**2))  # noqa: F821 -- pre-existing bug, see module docstring
        syst_err_norm = np.sqrt(np.abs(np.diag(cov_norm)))
        syst_err_mixed = np.sqrt(np.abs(np.diag(cov_mixed)))
        syst_err_shape = np.sqrt(np.abs(np.diag(cov_shape)))

        syst_err = _overlay_syst_sigma(
            total_mc, mc_stat_err, syst, add_poisson_mc_stat=add_poisson_mc_stat
        )

        if syst_decomp == False:
            ax.bar(bin_centers, 2 * syst_err, width=np.diff(bins),
                   bottom=total_mc - syst_err,
                   facecolor='none', hatch='xxx', linewidth=0.0,
                   edgecolor='dimgray', label='Syst. Unc.', zorder=8)
            if show_bkgd_syst_band and total_mc_bkgd is not None:
                bkgd_frac = bkgd_syst_frac_cov
                if bkgd_frac is None and bkgd_syst_loader is not None:
                    bkgd_frac = bkgd_syst_loader(var_config, genie_sb_cov_mat_pkl)
                if bkgd_frac is not None:
                    bkgd_syst_err = _overlay_bkgd_syst_sigma(total_mc_bkgd, bkgd_frac)
                    _overlay_draw_bkgd_syst_band(
                        ax,
                        bin_centers,
                        bins,
                        total_mc,
                        bkgd_syst_err,
                        edgecolor=bkgd_syst_band_color,
                        label="Bkgd. GENIE unc.",
                    )
        else:
            ax.bar(bin_centers, 2*syst_err_shape, width=np.diff(bins),
                   bottom=total_mc - syst_err_shape,
                   facecolor='red', edgecolor='red', alpha=0.3,
                   linewidth=0.0, label='Syst. Unc. (Shape)')
            ax.bar(bin_centers, 2*syst_err_mixed, width=np.diff(bins),
                   bottom=total_mc - syst_err_mixed,
                   facecolor='none', edgecolor='green', hatch='////',
                   linewidth=0.0, label='Syst. Unc. (Mixed)')
            ax.bar(bin_centers, 2*syst_err_norm, width=np.diff(bins),
                   bottom=total_mc - syst_err_norm,
                   facecolor='dimgray', edgecolor='dimgray', alpha=0.3,
                   linewidth=0.0, label='Syst. Unc. (Norm)')

        if histdata.has_data:
            chi2_val, chi2_reduced, p_val, ndof, chi2_pull, chi2_shape_val, p_val_shape, ndof_shape = _overlay_compute_chi2(
                total_data, total_mc, syst, data_eylow, data_eyhigh
            )

    # Data points (draw on top of MC stack)
    if histdata.has_data:
        ax.errorbar(bin_centers, total_data,
                    yerr=np.vstack((data_eylow, data_eyhigh)),
                    color='black', fmt='o', markersize=5, capsize=3, linewidth=1.5,
                    label='Data', zorder=10)

    # Ratio panel
    ratio_num_hist = ratio_denom_hist = ratio_vals = ratio_err = None
    if ratio and custom_ratio:
        # Custom mode: ratio of two caller-supplied per-event variables (e.g. Reco/True)
        # instead of the built-in Data/MC ratio. Deliberately skips the Data/MC syst band
        # and data errorbar above -- those describe MC uncertainty / data stats, not the
        # relationship between the two arbitrary variables being compared here.
        num_values, denom_values = ratio_vars
        if ratio_weights is not None:
            num_weights, denom_weights = ratio_weights
        else:
            num_weights = denom_weights = None
        _ratio_bins = np.asarray(ratio_bins) if ratio_bins is not None else bins
        ratio_num_hist, ratio_denom_hist, ratio_vals, ratio_err = _overlay_custom_ratio(
            num_values, denom_values, num_weights, denom_weights, _ratio_bins
        )
        _ratio_bin_centers = 0.5 * (_ratio_bins[:-1] + _ratio_bins[1:])
        # Only draw bins where the denominator is non-zero
        valid = ratio_denom_hist != 0
        if np.any(valid):
            ax_r.errorbar(_ratio_bin_centers[valid], ratio_vals[valid], yerr=ratio_err[valid],
                          fmt='o', color='black',
                          markersize=5, capsize=3, linewidth=1.5, zorder=10)
            # Auto-scale y-axis to cover error bars when the caller hasn't fixed the range
            if ratio_ylim is None:
                y_lo = np.nanmin(ratio_vals[valid] - ratio_err[valid])
                y_hi = np.nanmax(ratio_vals[valid] + ratio_err[valid])
                margin = max(0.05 * (y_hi - y_lo), 0.05)
                ax_r.set_ylim(y_lo - margin, y_hi + margin)
        ax_r.set_xlim(_ratio_bins[0], _ratio_bins[-1])
    elif ratio:
        if syst is not None and total_mc is not None:
            if syst_decomp == False:
                mc_content_ratio = np.ones_like(total_mc)
                with np.errstate(divide='ignore', invalid='ignore'):
                    mc_stat_err_ratio = np.where(total_mc != 0, syst_err / total_mc, 0.0)
                mc_stat_err_ratio = np.nan_to_num(mc_stat_err_ratio, nan=0.)
                ax_r.bar(bin_centers, 2*mc_stat_err_ratio, width=np.diff(bins),
                         bottom=mc_content_ratio - mc_stat_err_ratio,
                         facecolor='none', edgecolor='dimgray', hatch='xxx',
                         linewidth=0.0, label='Syst. Unc.', zorder=8)
                if bkgd_syst_err is not None:
                    bkgd_err_ratio = np.where(
                        total_mc != 0, bkgd_syst_err / total_mc, 0.0
                    )
                    bkgd_err_ratio = np.nan_to_num(bkgd_err_ratio, nan=0.0)
                    ax_r.bar(
                        bin_centers,
                        2 * bkgd_err_ratio,
                        width=np.diff(bins),
                        bottom=mc_content_ratio - bkgd_err_ratio,
                        facecolor="none",
                        edgecolor=bkgd_syst_band_color,
                        hatch="+++",
                        linewidth=0.0,
                        label="Bkgd. GENIE unc.",
                    )
            else:
                mc_content_ratio = np.ones_like(total_mc)
                with np.errstate(divide='ignore', invalid='ignore'):
                    r_norm = np.where(total_mc != 0, syst_err_norm / total_mc, 0.0)
                    r_mixed = np.where(total_mc != 0, syst_err_mixed / total_mc, 0.0)
                    r_shape = np.where(total_mc != 0, syst_err_shape / total_mc, 0.0)
                r_norm = np.nan_to_num(r_norm, nan=0.)
                r_mixed = np.nan_to_num(r_mixed, nan=0.)
                r_shape = np.nan_to_num(r_shape, nan=0.)

                ax_r.bar(bin_centers, 2*r_shape, width=np.diff(bins),
                         bottom=mc_content_ratio - r_shape,
                         facecolor='red', edgecolor='red', alpha=0.3,
                         linewidth=0.0, label='Syst. Unc. (Shape)')
                ax_r.bar(bin_centers, 2*r_mixed, width=np.diff(bins),
                         bottom=mc_content_ratio - r_mixed,
                         facecolor='none', edgecolor='green', hatch='////',
                         linewidth=0.0, label='Syst. Unc. (Mixed)')
                ax_r.bar(bin_centers, 2*r_norm, width=np.diff(bins),
                         bottom=mc_content_ratio - r_norm, alpha=0.3,
                         facecolor='dimgray', edgecolor='dimgray',
                         linewidth=0.0, label='Syst. Unc. (Norm)')

        if histdata.has_data and total_mc is not None:
            ax_r.errorbar(bin_centers, data_ratio,
                          yerr=np.vstack((data_ratio_eylow, data_ratio_eyhigh)),
                          fmt='o', color='black',
                          markersize=5, capsize=3, linewidth=1.5, zorder=10)
        ax_r.set_xlim(bins[0], bins[-1])

    # Legend: Data first; MC rows = signal-first physics categories → … → Low-E Dirt →
    # Cosmic (topology / pdg / genie). MC patches match stacked-layer colors (no
    # cosmic-uncertainty band). Each MC category shows its POT-scaled predicted event
    # count and its percentage of the total, e.g. "label (142, 34.2%)".
    handles, labels_orig = ax.get_legend_handles_labels()
    ordered_handles = []
    ordered_labels = []

    if histdata.has_data:
        try:
            data_handle_index = labels_orig.index('Data')
            ordered_handles.append(handles[data_handle_index])
            ordered_labels.append('Data ({:.0f})'.format(sum_data))
        except ValueError:
            pass

    if breakdown_fractions is not None:
        if breakdown_type == "genie_sb":
            patch_colors = genie_sb_patch_colors if genie_sb_patch_colors is not None else colors
            for color in patch_colors:
                ordered_handles.append(Patch(facecolor=color, edgecolor='none'))

            # breakdown_counts/breakdown_fractions are in the same (cuts) order as
            # ``labels`` here; consecutive equal labels are a signal/background pair
            # (genie_sb_mode_labels repeats each mode name twice) collapsed into one
            # legend row "label (bkgd_n, bkgd_%/sig_n, sig_%)".
            legend_labels = []
            i, n = 0, len(labels)
            while i < n:
                label_base = labels[i]
                if (i + 1 < n) and (labels[i + 1] == label_base):
                    count1, frac1 = breakdown_counts[i], breakdown_fractions[i]
                    count2, frac2 = breakdown_counts[i + 1], breakdown_fractions[i + 1]
                    legend_labels.append(
                        f"{label_base} ({count2:.0f}, {frac2*100:.1f}%/{count1:.0f}, {frac1*100:.1f}%)"
                    )
                    i += 2
                else:
                    count1, frac1 = breakdown_counts[i], breakdown_fractions[i]
                    legend_labels.append(f"{label_base} ({count1:.0f}, {frac1*100:.1f}%)")
                    i += 1
            ordered_labels.extend(legend_labels[::-1])
        else:
            idx_order = _overlay_histdata_legend_mc_index_order(
                len(labels), histdata.has_dirt
            )
            for i in idx_order:
                ordered_handles.append(Patch(facecolor=colors[i], edgecolor='none'))
                ordered_labels.append(
                    f"{labels[i]} ({layer_integrals[i]:.0f}, {breakdown_fractions[i]*100:.1f}%)"
                )

    # Synthetic patches last so "Syst. Unc." never precedes Data / MC stack rows.
    if syst is not None and total_mc is not None:
        if syst_decomp == False:
            ordered_handles.append(
                Patch(
                    facecolor='none', edgecolor='dimgray', hatch='xxx',
                    linewidth=0.5, label='Syst. Unc.',
                )
            )
            ordered_labels.append('Syst. Unc.')
        else:
            ordered_handles.extend([
                Patch(facecolor='red', edgecolor='red', alpha=0.3,
                      linewidth=0.0, label='Syst. Unc. (Shape)'),
                Patch(facecolor='none', edgecolor='green', hatch='////',
                      linewidth=0.0, label='Syst. Unc. (Mixed)'),
                Patch(facecolor='dimgray', edgecolor='dimgray', alpha=0.3,
                      linewidth=0.0, label='Syst. Unc. (Norm)'),
            ])
            ordered_labels.extend([
                'Syst. Unc. (Shape)',
                'Syst. Unc. (Mixed)',
                'Syst. Unc. (Norm)',
            ])

    fontsize = 12
    ncol = 3
    if breakdown_type == "genie_sb":
        textloc_x_tmp, textloc_ha_tmp = get_textloc_x(total_mc, bins, textloc)
        ncol = 2
        example_signal = Patch(facecolor="black", edgecolor='white', label='Signal')
        example_background = Patch(facecolor="black", edgecolor='white', hatch='////', linewidth=0, label='Background')
        box_ax = ax.inset_axes([0.625, 0.66, 0.13, 0.13], transform=ax.transAxes)
        box_ax.axis('off')
        mini_legend = Legend(
            box_ax, handles=[example_signal, example_background],
            labels=['Signal', 'Background'], loc='center', fontsize=fontsize,
            frameon=False, borderpad=0.7, handlelength=2.1, handleheight=0.9,
            ncol=1, fancybox=True, framealpha=1.0)
        box_ax.add_artist(mini_legend)

        ax.legend(ordered_handles, ordered_labels, loc='upper left',
                  fontsize=fontsize, frameon=False, ncol=ncol,
                  bbox_to_anchor=(0.05, 0.9, 0.8, 0.1), mode='expand')
    else:
        # breakdown_type in {"topology", "genie", "pdg"} can have up to ~12 rows
        # with long category names (e.g. "nu_e CC 1p0pi (43, 48.3%)"). A fixed
        # 3-column layout at fontsize=12 fits a handful of short labels but runs
        # past the right edge of the axes once there are this many rows --
        # confirmed visually against a real topology-breakdown plot, where the
        # third column spilled outside the plot box. Shrink columns/fontsize as
        # entry count grows so the legend stays inside the axes.
        n_entries = len(ordered_labels)
        if n_entries <= 4:
            legend_ncol, legend_fontsize = 1, fontsize
        elif n_entries <= 8:
            legend_ncol, legend_fontsize = 2, fontsize
        else:
            legend_ncol, legend_fontsize = 2, fontsize - 2
        ax.legend(ordered_handles, ordered_labels, loc='upper left',
                  fontsize=legend_fontsize, frameon=False, ncol=legend_ncol)

    # y-axis limit
    if total_mc is not None and np.max(total_mc) > 0:
        ax.set_ylim(0., ax_ylim_ratio * np.max(total_mc))
    elif total_data is not None and np.max(total_data) > 0:
        ax.set_ylim(0., ax_ylim_ratio * np.max(total_data))

    # vertical lines
    if vline is not None:
        for v in vline:
            ymax = ax.get_ylim()[1]
            ax.vlines(x=v[0], ymin=0, ymax=ymax*0.75, color='red', linestyle='--', zorder=50)
            if len(v) > 1:
                direction = v[1]
                arrow_params = {
                    'y': ymax * 0.4,
                    'dx': 0.18 * (ax.get_xlim()[1] - ax.get_xlim()[0]),
                    'width': 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                    'color': 'red',
                    'head_width': 0.04 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                    'head_length': 0.03 * (ax.get_xlim()[1] - ax.get_xlim()[0]),
                    'length_includes_head': True
                }
                # clip_on=False: the arrow's fixed length (18% of the axis width, set
                # above) can push its tip past the axis edge when the cut threshold
                # sits near xlim -- e.g. a "keep larger" (direction=1) cut at x=0.9 on
                # a 0-1 axis wants to end at 1.08. With clipping on, matplotlib just
                # truncates the arrow at the boundary, which can chop off the
                # arrowhead entirely and leave what looks like a stray red bar with no
                # direction indicator. Since this arrow's only job is to show which
                # side of the dashed threshold line survives the cut, letting it
                # overflow past the plot box (rather than disappear) keeps that
                # meaning visible even when there isn't enough room inside the axes.
                if direction == 0:
                    ax.arrow(v[0], arrow_params['y'], -arrow_params['dx'], 0,
                             width=arrow_params['width'],
                             color=arrow_params['color'],
                             head_width=arrow_params['head_width'],
                             head_length=arrow_params['head_length'],
                             length_includes_head=arrow_params['length_includes_head'],
                             clip_on=False,
                             zorder=60)
                elif direction == 1:
                    ax.arrow(v[0], arrow_params['y'], arrow_params['dx'], 0,
                             width=arrow_params['width'],
                             color=arrow_params['color'],
                             head_width=arrow_params['head_width'],
                             head_length=arrow_params['head_length'],
                             length_includes_head=arrow_params['length_includes_head'],
                             clip_on=False,
                             zorder=60)

    # textboxes
    if total_mc is not None:
        textloc_x, textloc_ha = get_textloc_x(total_mc, bins, textloc)
    elif total_data is not None:
        textloc_x, textloc_ha = get_textloc_x(total_data, bins, textloc)
    else:
        textloc_x, textloc_ha = textloc[0], 'left'
    textloc_y = textloc[1]

    if textchi2 and chi2_val is not None:
        add_chi2_text(
            chi2_val,
            p_val,
            ndof,
            textloc_x,
            textloc_y + 0.08,
            textloc_ha,
            chi2_shape=chi2_shape_val,
            p_val_shape=p_val_shape,
            ndof_shape=ndof_shape,
        )

    fig.subplots_adjust(top=0.9)
    add_approval_text(approval, 0.03, 1.07, "left")
    add_pot_text(pot_annotation_text, 0.99, 1.06, "right", fontsize=16)
    if breakdown_type != "pdg" and show_genie_label:
        # Bottom-right, not upper-left where it used to sit: breakdown legends
        # (see above) are anchored 'upper left' and can run several rows deep
        # (up to ~12 for topology), so a fixed y=0.83 slot ended up rendered
        # underneath the legend text. Bottom-right stays clear of the legend
        # regardless of its row count, and is typically clear of data too for
        # these left-peaked, decreasing distributions.
        add_genie_version_text(0.97, 0.06, "right")

    if var_config is not None and getattr(var_config, "var_save_name", None) == INTEGRATED_VAR_SAVE_NAME:
        format_singlebin_plot()

    if save_fig:
        plt.savefig(save_name+fig_ext, bbox_inches="tight", dpi=dpi)

    if plot:
        plt.show()
    else:
        plt.close()

    return {"breakdown_type": breakdown_type,
            "var_name": var_config.var_save_name if var_config is not None else None,
            "bins": bins,
            "mc_stack": mc_stack,
            "total_mc": total_mc,
            "total_mc_bkgd": total_mc_bkgd,
            "total_data": total_data,
            "chi2_val": chi2_val,
            "p_val": p_val,
            "ndof": ndof,
            "chi2_pull": chi2_pull,
            "chi2_shape_val": chi2_shape_val,
            "p_val_shape": p_val_shape,
            "ndof_shape": ndof_shape,
            # Custom ratio-panel outputs (only populated when ratio=True and ratio_vars
            # was given -- see the ``ratio_vars``/``ratio_weights`` docstring entry above).
            "ratio_num_hist": ratio_num_hist,
            "ratio_denom_hist": ratio_denom_hist,
            "ratio_vals": ratio_vals,
            "ratio_err": ratio_err,
            # The Figure object itself -- note it's already been through plt.close()
            # above when plot=False (the batch-render default). That's harmless for
            # later use: close() just detaches it from pyplot's global figure
            # manager, it doesn't clear the Axes/Artists, so the object is still
            # fully intact and picklable (a caller wanting to pickle.dump it for
            # later pickle.load + inspection, e.g. event_selection_aggregate.py's
            # image/pkl output split, just needs this reference -- plt.gcf() alone
            # wouldn't find it anymore post-close).
            "fig": fig}
