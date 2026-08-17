#!/usr/bin/env python
"""Aggregate per-map-shard pickles and produce all event-selection plots.

This is the **reduce** pass of the chunked histogram workflow (one pickle per
*(sample, CAF .df file)* — not an exposure batch in time; see ``exposure_access``).

  1. globs the per-(sample, shard) pickles under ``--in_dir``,
  2. sums histograms across chunks within each sample,
  3. merges sample-level results into one combined histogram dict,
  4. renders each plot through ``overlay_hists_from_histdata`` -- yielding the
     SAME visual output as the original notebook, but starting from the
     pre-binned content,
  5. produces the summary breakdown bar plot and the efficiency curve plots.

**Systematic uncertainty bands** load **pre-saved** fractional covariances from the
syst-disk tree via :func:`utils.get_syst_unc` (``--syst-disk-root`` /
``NUMUCC_SYST_DISK_ROOT``). On-the-fly covariances from MC multi-universe weights
at plot time are no longer supported; see the ``cafpyana_trash`` legacy copies of
``get_frac_unc`` / ``frac_cov_from_mc_univ_histdata``.

The plotting step is a thin layer on top of the existing ``overlay_hists`` and
``plot_efficiency`` routines in ``utils.py``; the new precomputed-histogram
entry point is ``overlay_hists_from_histdata``.

Usage
-----
    python event_selection_aggregate.py --in_dir AGG_INPUT_DIR \
                                         --out_dir PLOTS_OUTPUT_DIR

The expected pickle layout in ``--in_dir`` is::

    mc__<chunkname>.pkl
    data__<chunkname>.pkl
    intime__<chunkname>.pkl
    offbeam__<chunkname>.pkl
    dirt__<chunkname>.pkl

i.e. the names produced by ``event_selection_chunk.py``.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from os import path, makedirs
import pickle
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# turn off pandas chatter
import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

sys.path.append(path.dirname(path.dirname(path.dirname(path.dirname(path.abspath(__file__))))))

from analysis_village.nueNp0Pi.config.stages import build_pipeline, EFFICIENCY_VARS, sel_evt
# Imported as a module (not `from ... import EVT_BREAKDOWN_DIR_NAME`) so that reading
# ``stages_mod.EVT_BREAKDOWN_DIR_NAME`` below picks up whatever a notebook cell most recently
# set on the module -- a plain name import would freeze in the value from whenever this file
# was first imported, same reasoning as EVT_BREAKDOWN_TYPE/EVT_BREAKDOWN_STAGE_KEYS elsewhere.
import analysis_village.nueNp0Pi.config.stages as stages_mod
from analysis_village.nueNp0Pi.event_selection import ChunkRunner
from pyanalib.chunked_selection import (
    BarBreakdown,
    ExposureTotals, aggregate_chunk_files, merge_samples,
    sanitize_merged_histdata_finite,
    apply_global_exposure_scales,
)
from pyanalib.response_matrix_plotting import response_matrix_from_histdata
from analysis_village.nueNp0Pi.utils import (
    overlay_hists_from_histdata,
    get_pot_str,
    fig_ext,
    dpi,
    add_approval_text,
    format_singlebin_plot,
    get_syst_unc,
)
from analysis_village.nueNp0Pi.config.settings import (
    topology_labels, topology_colors,
    genie_mode_labels, genie_mode_colors,
    pdg_labels,
)
from pyanalib.stat_helpers import return_data_stat_err

# style sheet that the notebook uses
try:
    plt.style.use(path.join(path.dirname(__file__), "presentation.mplstyle"))
except Exception:
    try:
        plt.style.use("presentation.mplstyle")
    except Exception:
        pass


SAMPLES = ("mc", "data", "intime", "offbeam", "dirt")


# ===========================================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", required=True, help="Dir holding per-chunk pickles")
    p.add_argument("--out_dir", required=True, help="Where to write the plots")
    p.add_argument("--data_pot", type=float, default=None,
                   help="Override data POT for legend (default: sum of chunk_pot from data pickles)")
    p.add_argument("--cosmic_estimate", choices=("intime", "offbeam"), default="intime",
                   help="Which cosmic sample drives the summary bar-plot breakdown (intime default)")
    p.add_argument("--f_offbeam_frac", type=float, default=0.08,
                   help="Offbeam-coincident-with-BNB fraction used in cosmic gates scaling")
    p.add_argument("--skip_global_exposure", action="store_true",
                   help="Skip POT/gates rescaling (only for legacy chunks already scaled)")
    p.add_argument("--syst_tag", default="", help="Prefix tag for output names")
    p.add_argument("--save_fig", action="store_true", default=True,
                   help="Save figures to disk (default True)")
    p.add_argument("--show_fig", action="store_true", default=False,
                   help="Show figures interactively (off by default)")
    p.add_argument(
        "--syst-disk-root",
        default=None,
        help="Root directory with MCstat/, Flux/, G4/, GENIE/, Cosmics/, Detector/ trees "
             "(see analysis_village.nueNp0Pi.syst_disk_layout). Env: NUMUCC_SYST_DISK_ROOT.",
    )
    p.add_argument(
        "--no_breakdown_independent", dest="render_breakdown_independent",
        action="store_false", default=True,
        help="Skip summary/efficiency/cutflow-table/breakdown-table (unchanged across "
             "EVT_BREAKDOWN_TYPE reruns of the same batch pickles) -- see "
             "aggregate_and_render's render_breakdown_independent.",
    )
    return p.parse_args()


# ===========================================================================
def collect_chunks(in_dir: str) -> Dict[str, List[str]]:
    """Group pickles by sample name (the prefix before ``__``)."""
    out = {s: sorted(glob.glob(path.join(in_dir, f"{s}__*.pkl"))) for s in SAMPLES}
    for s, files in out.items():
        print(f"[aggregate] sample={s} -> {len(files)} chunks")
    return out


# ===========================================================================
def accumulate_exposure_totals_from_dir(in_dir: str) -> ExposureTotals:
    """Sum POT / gates denominators from every per-chunk pickle (map phase metadata)."""
    totals = ExposureTotals()
    pattern = path.join(in_dir, "*__*.pkl")
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"[aggregate] WARN: no chunk pickles matched {pattern}")
        return totals
    for cf in paths:
        with open(cf, "rb") as f:
            d = pickle.load(f)
        sample = d.get("sample")
        m = d.get("meta", {}) or {}
        if sample == "data":
            totals.data_pot += float(m.get("chunk_pot", 0.0))
            totals.data_gates_bnb += float(m.get("chunk_gates_bnb", 0.0))
        elif sample == "mc":
            totals.mc_pot += float(m.get("chunk_pot", 0.0))
        elif sample == "dirt":
            totals.dirt_pot += float(m.get("chunk_pot", 0.0))
        elif sample == "intime":
            totals.intime_gates += float(m.get("chunk_cosmic_gates_intime", 0.0))
        elif sample == "offbeam":
            totals.offbeam_gates += float(m.get("chunk_cosmic_gates_offbeam", 0.0))
    print(
        f"[aggregate] exposure totals: data_pot={totals.data_pot:.3e} "
        f"bnb_gates={totals.data_gates_bnb:.3e} mc_pot={totals.mc_pot:.3e} "
        f"dirt_pot={totals.dirt_pot:.3e} intime_gates={totals.intime_gates:.3e} "
        f"offbeam_gates={totals.offbeam_gates:.3e}"
    )
    return totals


def _save_fig_and_pkl(fig, save_fig_dir: str, rel_name: str, save_fig: bool, show_fig: bool):
    """Save ``fig`` as both a PNG and a pickled Figure object, in a mirrored
    ``image/``/``pkl/`` layout under ``save_fig_dir``.

    ``rel_name`` is the path *relative to* ``save_fig_dir``, with no extension (e.g.
    ``"cutflow_table"`` or ``"selection/selection_proton-p"``) -- it becomes
    ``<save_fig_dir>/image/<rel_name>{fig_ext}`` and ``<save_fig_dir>/pkl/<rel_name>.pkl``.
    The pickled figure is the actual ``matplotlib.figure.Figure`` object (via
    ``pickle.dump(fig, ...)``), not a data summary -- ``pickle.load`` it back and you get
    the same interactive figure, re-showable/resizable, without rererunning any of the
    aggregation or plotting code. Note this ties the pickle to the matplotlib version it
    was written with; that trade-off (vs. pickling plain arrays) was a deliberate choice.
    """
    if save_fig:
        image_path = path.join(save_fig_dir, "image", rel_name + fig_ext)
        pkl_path = path.join(save_fig_dir, "pkl", rel_name + ".pkl")
        makedirs(path.dirname(image_path), exist_ok=True)
        makedirs(path.dirname(pkl_path), exist_ok=True)
        fig.savefig(image_path, dpi=dpi, bbox_inches="tight")
        with open(pkl_path, "wb") as f:
            pickle.dump(fig, f)
    if show_fig:
        plt.show()
    else:
        plt.close(fig)


# ===========================================================================
def _apply_ratio_kwargs(hd, var_config, kwargs: dict, disable_ratio_plots: bool) -> dict:
    """Translate ``var_config.ratio_mode`` (see ``pyanalib.variable_config.VariableConfig``)
    into the ``ratio``/``ratio_vars``/``ratio_weights``/``ratio_bins``/``ratio_label`` kwargs
    ``overlay_hists_from_histdata`` (``pyanalib/overlay_plotting.py``) understands. Mutates
    and returns ``kwargs`` in place; also returned for convenience.

    ``disable_ratio_plots`` (``stages_mod.DISABLE_RATIO_PLOTS``) is a global override: when
    True, every plot's ratio panel is forced off regardless of ``ratio_mode`` or whatever a
    ``PlotSpec``'s own ``save_kwargs`` set.

    ``ratio_mode is None`` leaves ``kwargs["ratio"]`` exactly as already set by
    ``PlotSpec.save_kwargs`` / the caller's own ``kwargs.setdefault("ratio", False)`` --
    fully backward compatible with plots that predate this mechanism.

    A precedence note on labels: ``kwargs.setdefault("ratio_label", ...)`` below means a
    ``PlotSpec.save_kwargs["ratio_label"]`` (if a pipeline author ever sets one) always wins
    over ``var_config.ratio_label``, which in turn wins over the mode's own default string.

    Both data-driven modes reuse ``overlay_hists_from_histdata``'s bin-centers-as-fake-events
    trick (``ratio_vars=(bin_centers, bin_centers)``, ``ratio_weights=(num_hist, denom_hist)``)
    to feed already-binned arrays through its per-event ``ratio_vars`` mechanism -- so no
    further changes to ``pyanalib/overlay_plotting.py`` are needed for either mode.
    """
    if disable_ratio_plots:
        kwargs["ratio"] = False
        return kwargs

    ratio_mode = getattr(var_config, "ratio_mode", None)
    if ratio_mode is None:
        return kwargs

    if ratio_mode == "data_mc":
        kwargs["ratio"] = True
        if getattr(var_config, "ratio_label", None):
            kwargs.setdefault("ratio_label", var_config.ratio_label)
        return kwargs

    if ratio_mode == "reco_true":
        if not getattr(hd, "has_truth", False):
            print(
                f"[aggregate] WARN: ratio_mode='reco_true' set on "
                f"{var_config.var_save_name!r} but no truth histogram was filled "
                f"(has_truth=False) -- skipping ratio panel. ratio_mode must be set on the "
                f"VariableConfig BEFORE the batch producing this histdata was mapped (see "
                f"pyanalib.chunked_selection.OverlayHistData.fill_truth_from_df)."
            )
            return kwargs
        bin_centers = 0.5 * (hd.bins[:-1] + hd.bins[1:])
        total_reco = np.sum(hd.mc_hist, axis=0)
        kwargs["ratio"] = True
        kwargs["ratio_vars"] = (bin_centers, bin_centers)
        kwargs["ratio_weights"] = (total_reco, hd.truth_hist)
        kwargs["ratio_bins"] = hd.bins
        kwargs.setdefault("ratio_label", var_config.ratio_label or "Reco/True")
        return kwargs

    if ratio_mode == "signal_bkgd":
        ratio_breakdown_type = getattr(var_config, "ratio_breakdown_type", None)
        if hd.breakdown_type != ratio_breakdown_type:
            print(
                f"[aggregate] WARN: ratio_mode='signal_bkgd' on "
                f"{var_config.var_save_name!r} expects breakdown_type="
                f"{ratio_breakdown_type!r} but this plot's breakdown_type is "
                f"{hd.breakdown_type!r} -- skipping ratio panel (the category indices "
                f"aren't valid for a different breakdown_type's category order/count)."
            )
            return kwargs
        sig_idx = list(getattr(var_config, "ratio_signal_indices", None) or [])
        bkg_idx = list(getattr(var_config, "ratio_bkgd_indices", None) or [])
        n_cat = hd.mc_hist.shape[0]
        if not sig_idx or not bkg_idx:
            print(
                f"[aggregate] WARN: ratio_mode='signal_bkgd' on "
                f"{var_config.var_save_name!r} is missing ratio_signal_indices/"
                f"ratio_bkgd_indices -- skipping ratio panel."
            )
            return kwargs
        out_of_range = [i for i in sig_idx + bkg_idx if i < 0 or i >= n_cat]
        if out_of_range:
            print(
                f"[aggregate] WARN: ratio_mode='signal_bkgd' on "
                f"{var_config.var_save_name!r} has ratio_signal_indices/ratio_bkgd_indices "
                f"{out_of_range} out of range for this plot's {n_cat} categories -- "
                f"skipping ratio panel."
            )
            return kwargs
        bin_centers = 0.5 * (hd.bins[:-1] + hd.bins[1:])
        sig = np.sum(hd.mc_hist[sig_idx], axis=0)
        bkg = np.sum(hd.mc_hist[bkg_idx], axis=0)
        kwargs["ratio"] = True
        kwargs["ratio_vars"] = (bin_centers, bin_centers)
        kwargs["ratio_weights"] = (sig, bkg)
        kwargs["ratio_bins"] = hd.bins
        kwargs.setdefault("ratio_label", var_config.ratio_label or "S/B")
        return kwargs

    print(
        f"[aggregate] WARN: unknown ratio_mode={ratio_mode!r} on "
        f"{var_config.var_save_name!r} (expected None/'data_mc'/'reco_true'/'signal_bkgd') "
        f"-- ignoring, ratio panel left as previously set."
    )
    return kwargs


# ===========================================================================
def render_overlay_plots(
    merged: dict,
    plot_label_map: dict,
    save_fig_dir: str,
    pot_str: str,
    save_fig: bool,
    show_fig: bool,
    syst_disk_root: str | None = None,
):
    """Render every plot stored in ``merged['histdata']``."""
    # We need the pipeline definition to recover the per-plot kwargs and labels.
    pipeline = build_pipeline()
    spec_lookup = {}  # (stage_key, plot_key) -> PlotSpec (recomputed from build_pipeline)
    for stage in pipeline:
        for ps in stage.plots:
            key = (stage.key, ChunkRunner.plot_key(stage.key, ps))
            spec_lookup[key] = ps

    vars_missing_syst = []

    for key, hd in merged["histdata"].items():
        stage_key, plot_key = key
        ps = spec_lookup.get(key)
        if ps is None:
            print(f"[aggregate] WARN: no PlotSpec for {key}; skipping")
            continue

        # build plot labels
        if ps.plot_label_template is not None:
            plot_labels = [
                ps.plot_label_template[0],
                ps.plot_label_template[1].replace("{pot}", pot_str),
                ps.plot_label_template[2].replace("{pot}", pot_str) if len(ps.plot_label_template) >= 3 else "",
            ]
        else:
            plot_labels = [ps.var_config.var_labels[0], f"Events / Bin (POT={pot_str})", ""]

        if ps.selector is sel_evt:
            # Stacked event-distribution breakdown plots (config/stages.py's
            # evt_breakdown_plot / EVT_BREAKDOWN_*) get their own subdirectory (default
            # "selection", overridable via EVT_BREAKDOWN_DIR_NAME) and a simpler name --
            # there's normally only one of these per variable (the final stage), so the
            # stage/breakdown_type prefix that disambiguates OTHER PlotSpec-driven plots
            # (which can share a variable across several stages) isn't needed here. If
            # EVT_BREAKDOWN_STAGE_KEYS is configured to attach these to more than one
            # stage, use evt_breakdown_plot's name_suffix to keep filenames from
            # colliding. Re-running with a different EVT_BREAKDOWN_TYPE from the same
            # plots dir overwrites these files unless EVT_BREAKDOWN_DIR_NAME is also set
            # to something distinct per run (e.g. "selection_genie").
            selection_subdir = stages_mod.EVT_BREAKDOWN_DIR_NAME or "selection"
            rel_name = path.join(
                selection_subdir,
                f"selection_{ps.var_config.var_save_name}"
                + (("_" + ps.name_suffix) if ps.name_suffix else ""),
            )
        elif (ps.name_suffix or "").startswith("n_minus_1_"):
            # N-1 diagnostic plots (config/stages.py's N_MINUS_1_STAGE_KEYS/N_MINUS_1_VARS)
            # get their own subdirectory (default "n_minus_1", overridable via
            # N_MINUS_1_DIR_NAME) -- name_suffix is already "n_minus_1_<held_out_key>" (see
            # build_pipeline()), so the file name alone disambiguates held-out cut + variable
            # without needing the stage_key/breakdown_type prefix other plots use (these are
            # all attached to the same final stage).
            n1_subdir = stages_mod.N_MINUS_1_DIR_NAME or "n_minus_1"
            rel_name = path.join(
                n1_subdir, f"{ps.name_suffix}__{ps.var_config.var_save_name}"
            )
        else:
            rel_name = (
                f"{stage_key}__{ps.breakdown_type}__{ps.var_config.var_save_name}"
                + (("_" + ps.name_suffix) if ps.name_suffix else "")
            )
        # image/pkl mirrored layout (see _save_fig_and_pkl) -- overlay_hists_from_histdata
        # appends fig_ext itself, so save_name here stays extension-less.
        save_name = path.join(save_fig_dir, "image", rel_name)
        makedirs(path.dirname(save_name), exist_ok=True)
        kwargs = dict(ps.save_kwargs)
        kwargs.setdefault("ratio", False)
        kwargs.setdefault("save_fig", save_fig)
        kwargs.setdefault("save_name", save_name)
        kwargs.setdefault("plot", show_fig)
        kwargs["plot_labels"] = plot_labels
        # Without this, overlay_hists_from_histdata falls back to its hardcoded
        # placeholder default ("8.8 x10^19 POT") for the top-right annotation --
        # a fixed string that has nothing to do with this run's actual exposure,
        # while the y-axis label ("Events / Bin (POT=...)") right next to it
        # already uses the real computed pot_str. Confirmed by a user screenshot
        # showing the two disagree (8.8e19 top-right vs. 1.42e19 on the y-axis).
        kwargs.setdefault("pot_annotation_text", f"{pot_str} POT")
        kwargs.setdefault("verbose_hist", (ps.name_suffix or "") == "final")
        # Match ``selected_events.ipynb``: combined syst as hatched band (not norm/shape/mixed fill).
        kwargs.setdefault("syst_decomp", False)
        # Drop retired overlay kwargs if any pipeline PlotSpec still sets them.
        kwargs.pop("cosmic_estimate", None)
        kwargs.pop("show_cosmic_model_unc", None)
        kwargs.pop("legend_percentages", None)

        # VariableConfig.ratio_mode ("data_mc" / "reco_true" / "signal_bkgd") -> the
        # ratio/ratio_vars/ratio_weights/ratio_bins/ratio_label kwargs overlay_hists_from_histdata
        # understands, or a global no-op if stages_mod.DISABLE_RATIO_PLOTS is set. See
        # _apply_ratio_kwargs's own docstring for the full precedence rules.
        kwargs = _apply_ratio_kwargs(hd, ps.var_config, kwargs, stages_mod.DISABLE_RATIO_PLOTS)

        # Pre-saved fractional covariance on the syst disk (GENIE / flux / …).
        if kwargs.get("syst") is None and syst_disk_root is not None:
            _, cov_disk = get_syst_unc(
                ps.var_config,
                syst_disk_root=syst_disk_root,
                skip_missing_vars=True,
            )
            if np.any(cov_disk):
                kwargs["syst"] = cov_disk

        if kwargs.get("syst") is None:
            vars_missing_syst.append(ps.var_config.var_save_name)

        try:
            result = overlay_hists_from_histdata(hd, var_config=ps.var_config, **kwargs)
            if save_fig and result.get("fig") is not None:
                pkl_path = path.join(save_fig_dir, "pkl", rel_name + ".pkl")
                makedirs(path.dirname(pkl_path), exist_ok=True)
                with open(pkl_path, "wb") as f:
                    pickle.dump(result["fig"], f)
        except Exception as e:
            print(f"[aggregate] WARN: plot {key} failed: {e}")
            plt.close('all')

    if vars_missing_syst:
        uniq = sorted(set(vars_missing_syst))
        print(
            f"[aggregate] overlay plots without syst covariance ({len(vars_missing_syst)} plots, "
            f"{len(uniq)} distinct var_save_name): {', '.join(uniq)}",
            flush=True,
        )


# ===========================================================================
def render_response_matrix_plots(merged: dict, save_fig_dir: str, save_fig: bool, show_fig: bool):
    """Render the optional true-vs-reco response-matrix heatmap (see
    ``pyanalib.response_matrix_plotting.response_matrix_from_histdata``) for every plot whose
    ``VariableConfig.response_matrix`` is True (``pyanalib.variable_config``).

    Skips (with a WARN, not a crash) any such variable whose histdata never had its response
    matrix filled at map time -- ``response_matrix`` must be set on the VariableConfig BEFORE
    the batch producing this histdata was mapped, same caveat as ``ratio_mode="reco_true"``
    (see ``pyanalib.chunked_selection.OverlayHistData.fill_response_from_df``).

    ``stages_mod.DISABLE_RESPONSE_MATRIX_PLOTS`` is a global override: when True, this whole
    function is a no-op (see ``config/stages.py``).
    """
    if stages_mod.DISABLE_RESPONSE_MATRIX_PLOTS:
        return

    pipeline = build_pipeline()
    spec_lookup = {}
    for stage in pipeline:
        for ps in stage.plots:
            key = (stage.key, ChunkRunner.plot_key(stage.key, ps))
            spec_lookup[key] = ps

    for key, hd in merged["histdata"].items():
        ps = spec_lookup.get(key)
        if ps is None or not getattr(ps.var_config, "response_matrix", False):
            continue
        if not getattr(hd, "has_response", False):
            print(
                f"[aggregate] WARN: response_matrix=True on "
                f"{ps.var_config.var_save_name!r} but no response histogram was filled "
                f"(has_response=False) -- skipping. response_matrix must be set on the "
                f"VariableConfig BEFORE the batch producing this histdata was mapped (see "
                f"pyanalib.chunked_selection.OverlayHistData.fill_response_from_df)."
            )
            continue

        rel_name = (
            f"response_matrix/response_matrix_{ps.var_config.var_save_name}"
            + (("_" + ps.name_suffix) if ps.name_suffix else "")
        )
        save_name = path.join(save_fig_dir, "image", rel_name)
        makedirs(path.dirname(save_name), exist_ok=True)
        try:
            result = response_matrix_from_histdata(
                hd, var_config=ps.var_config,
                save_fig=save_fig, save_name=save_name, plot=show_fig,
            )
            if not result.get("ok"):
                continue
            if save_fig and result.get("fig") is not None:
                pkl_path = path.join(save_fig_dir, "pkl", rel_name + ".pkl")
                makedirs(path.dirname(pkl_path), exist_ok=True)
                with open(pkl_path, "wb") as f:
                    pickle.dump(result["fig"], f)
        except Exception as e:
            print(f"[aggregate] WARN: response matrix plot {key} failed: {e}")
            plt.close('all')


# ===========================================================================
def render_summary_breakdown_plot(merged: dict, save_fig_dir: str,
                                  save_fig: bool, show_fig: bool,
                                  cosmic_estimate: str):
    """Reproduce the cell-79 summary bar plot from the notebook."""
    bar = merged["bar"]
    stage_keys = merged["stage_keys"]
    stage_labels = merged["stage_labels"]

    # only stages that we actually filled the breakdown for
    avail_stages = [k for k in stage_keys if k in bar]
    if not avail_stages:
        print("[aggregate] no breakdown stages -> skipping summary plot")
        return
    stage_label_lookup = dict(zip(stage_keys, stage_labels))

    # Build the percentage matrices (stages x categories)
    def fractions(bb: BarBreakdown) -> np.ndarray:
        # in cuts order; reverse to match the labels/colors used by bar plot
        v = bb.mc_counts.astype(float)
        cosmic_part = bb.offbeam_count if cosmic_estimate == "offbeam" else bb.intime_count
        tot = v.sum() + cosmic_part + bb.dirt_count
        if tot <= 0:
            return np.zeros_like(v)
        return 100.0 * v / tot

    topo_data = np.array([fractions(bar[k]["topology"]) for k in avail_stages[::-1]])
    genie_data = np.array([fractions(bar[k]["genie"]) for k in avail_stages[::-1]])

    y = np.arange(len(avail_stages))
    bar_width = 0.3

    def stack_bars(ax, data, yoffset, colors, label):
        left = np.zeros(len(avail_stages))
        for i, color in enumerate(colors[:data.shape[1]]):
            ax.barh(y + yoffset, data[:, i], bar_width, left=left, color=color,
                    label=label if i == 0 else None)
            left += data[:, i]

    fig, ax = plt.subplots(figsize=(10, 10))
    stack_bars(ax, topo_data, -bar_width / 2, topology_colors, "Topology")
    stack_bars(ax, genie_data,  bar_width / 2, genie_mode_colors, "GENIE")

    if not np.any(topo_data > 0) and not np.any(genie_data > 0):
        print(
            "[aggregate] WARN: summary breakdown fractions are all zero "
            "(need MC chunk pickles with ``save_for_breakdown`` filled — "
            "mc_counts sum + cosmic gates term).",
            flush=True,
        )

    ax.set_xlabel("Percentage (%)")
    ax.set_yticks(y)
    ax.set_yticklabels([stage_label_lookup[k] for k in avail_stages[::-1]], fontsize=12)

    topo_patches = [Patch(facecolor=c, label=l)
                    for c, l in zip(topology_colors, topology_labels)]
    genie_patches = [Patch(facecolor=c, label=l)
                     for c, l in zip(genie_mode_colors, genie_mode_labels)]

    ax.legend(handles=topo_patches, loc='upper left', bbox_to_anchor=(0.01, 1.22),
              ncol=4, fontsize=11, frameon=False, title="Topology", title_fontsize=11)
    ax_genie = ax.twinx()
    ax_genie.legend(handles=genie_patches, loc='upper left', bbox_to_anchor=(0.01, 1.10),
                    ncol=4, fontsize=11, frameon=False, title="GENIE Mode", title_fontsize=11)
    ax_genie.set_yticks([])

    fig.tight_layout()
    _save_fig_and_pkl(fig, save_fig_dir, "event_selection_summary", save_fig, show_fig)


# ===========================================================================
def _compute_efficiency_stage_data(eff, stage_keys, stage_label_lookup, var_save_name):
    """Per-stage (label, color, n_pot histogram, efficiency, efficiency error) for
    one variable -- the numeric content shared by all three efficiency-plot
    variants below (combined / event-counts-only / efficiency-only), so the
    actual accumulator math is computed exactly once per variable.

    Returns ``None`` if this variable has no usable denominator.
    """
    from statsmodels.stats.proportion import proportion_confint

    denom_stage = None
    for sk in stage_keys:
        if sk in eff and var_save_name in eff[sk]:
            denom_stage = sk
            break
    if denom_stage is None:
        return None
    denom = eff[denom_stage][var_save_name]
    n_tot_pot = np.asarray(
        getattr(denom, "n_truth_nu_pot", denom.n_signal_pot), dtype=float
    )
    n_tot_raw = np.asarray(
        getattr(denom, "n_truth_nu_raw", denom.n_signal_raw), dtype=float
    )
    # Legacy chunk pickles (pre mcnu denominator): fall back to evt-only first stage.
    if float(np.sum(n_tot_raw)) <= 0.0 and float(np.sum(n_tot_pot)) <= 0.0:
        n_tot_pot = np.asarray(denom.n_signal_pot, dtype=float)
        n_tot_raw = np.asarray(denom.n_signal_raw, dtype=float)
    # When the denominator was filled from evt_df (first-stage mode), use the
    # pre-tracked total integral so NaN x-axis values don't cause denom < numerator.
    n_total_truth_nu_int = float(getattr(denom, "n_total_truth_nu_int", 0.0))
    denom_int_pot = n_total_truth_nu_int if n_total_truth_nu_int > 0.0 else float(np.sum(n_tot_pot))

    stages = []
    plot_idx = 0
    for stage_key in stage_keys:
        if stage_key not in eff or var_save_name not in eff[stage_key]:
            continue
        ea = eff[stage_key][var_save_name]
        n_pot = np.asarray(ea.n_signal_pot, dtype=float)
        n_int = ea.n_total_signal_int
        with np.errstate(divide='ignore', invalid='ignore'):
            this_eff = np.where(n_tot_pot > 0, n_pot / n_tot_pot, 0.0)
        # raw counts for Wilson interval (avoid div-zero)
        n_succ = ea.n_signal_raw
        err_low, err_high = [], []
        for i in range(len(this_eff)):
            if n_tot_raw[i] > 0:
                interval = proportion_confint(int(n_succ[i]), int(n_tot_raw[i]), method='wilson')
                err_low.append(abs(this_eff[i] - interval[0]))
                err_high.append(abs(this_eff[i] - interval[1]))
            else:
                err_low.append(0.0)
                err_high.append(0.0)
        eff_int_pct = (n_int / denom_int_pot * 100) if denom_int_pot > 0 else 0.0
        stages.append({
            "stage_key": stage_key,
            "label": stage_label_lookup.get(stage_key, stage_key),
            "eff_pct": eff_int_pct,
            "color": plt.cm.tab10(plot_idx % 10),
            "n_pot": n_pot,
            "eff": this_eff,
            "eff_err": [err_low, err_high],
        })
        plot_idx += 1
    return stages


def _render_efficiency_combined(stages, var_config, bins, bin_centers, save_fig_dir,
                                var_save_name, save_fig, show_fig):
    """Both event counts and efficiency on one dual-axis figure (original view)."""
    fig, ax = plt.subplots()
    ax_eff = ax.twinx()

    ymax_hist = 0.0
    for s in stages:
        ymax_hist = max(ymax_hist, float(np.max(s["n_pot"])) if s["n_pot"].size else 0.0)
        label = f'{s["label"]} ({s["eff_pct"]:.2f}%)'
        ax.stairs(s["n_pot"], bins, fill=False, alpha=0.5, color=s["color"], linewidth=1.5, zorder=1)
        ax_eff.errorbar(bin_centers, s["eff"], yerr=s["eff_err"],
                        fmt="+", color=s["color"], label=label, markersize=8,
                        markeredgewidth=1.5, zorder=4, capsize=3)

    ax.set_xlabel(var_config.var_labels[0])
    ax.set_ylabel("Events")
    ax_eff.set_ylabel("Efficiency")
    ax_eff.set_ylim(0, 1.05)
    ax.set_xlim(bins[0], bins[-1])
    if ymax_hist > 0:
        ax.set_ylim(0.0, ymax_hist * 1.15)
    ax_eff.set_zorder(3)
    ax_eff.patch.set_visible(False)

    fig.subplots_adjust(top=0.88)
    ax_eff.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=8, frameon=False)

    add_approval_text("internal", 0.98, 0.97, "right", fontsize=14)
    if var_config.var_save_name == "integrated":
        format_singlebin_plot()
    _save_or_show(fig, save_fig_dir, "combined", var_save_name, save_fig, show_fig)


def _render_efficiency_event_counts(stages, var_config, bins, bin_centers, save_fig_dir,
                                    var_save_name, save_fig, show_fig):
    """Event counts only, one axis, legend by stage name (no % clutter)."""
    fig, ax = plt.subplots()

    ymax_hist = 0.0
    for s in stages:
        ymax_hist = max(ymax_hist, float(np.max(s["n_pot"])) if s["n_pot"].size else 0.0)
        ax.stairs(s["n_pot"], bins, fill=False, alpha=0.8, color=s["color"], linewidth=1.5,
                  label=s["label"])

    ax.set_xlabel(var_config.var_labels[0])
    ax.set_ylabel("Events")
    ax.set_xlim(bins[0], bins[-1])
    if ymax_hist > 0:
        ax.set_ylim(0.0, ymax_hist * 1.15)

    fig.subplots_adjust(top=0.88)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=8, frameon=False)

    add_approval_text("internal", 0.98, 0.97, "right", fontsize=14)
    if var_config.var_save_name == "integrated":
        format_singlebin_plot()
    _save_or_show(fig, save_fig_dir, "event_counts", var_save_name, save_fig, show_fig)


def _render_efficiency_only(stages, var_config, bins, bin_centers, save_fig_dir,
                            var_save_name, save_fig, show_fig):
    """Efficiency only, one axis 0-1.05, legend keeps the % (directly on-topic here)."""
    fig, ax = plt.subplots()

    for s in stages:
        label = f'{s["label"]} ({s["eff_pct"]:.2f}%)'
        ax.errorbar(bin_centers, s["eff"], yerr=s["eff_err"],
                    fmt="+", color=s["color"], label=label, markersize=8,
                    markeredgewidth=1.5, capsize=3)

    ax.set_xlabel(var_config.var_labels[0])
    ax.set_ylabel("Efficiency")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(bins[0], bins[-1])

    fig.subplots_adjust(top=0.88)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=8, frameon=False)

    add_approval_text("internal", 0.98, 0.97, "right", fontsize=14)
    if var_config.var_save_name == "integrated":
        format_singlebin_plot()
    _save_or_show(fig, save_fig_dir, "efficiency", var_save_name, save_fig, show_fig)


def _save_or_show(fig, save_fig_dir, subdir, var_save_name, save_fig, show_fig):
    rel_name = path.join(subdir, f"efficiency-{var_save_name}")
    _save_fig_and_pkl(fig, save_fig_dir, rel_name, save_fig, show_fig)


def render_efficiency_plots(merged: dict, save_fig_dir: str, pot_str: str,
                            save_fig: bool, show_fig: bool):
    """Reproduce ``plot_efficiency`` from the notebook using the eff accumulators.

    Writes THREE plots per variable, one file per variable under each of
    ``<save_fig_dir>/combined/``, ``<save_fig_dir>/event_counts/``, and
    ``<save_fig_dir>/efficiency/``: ``combined/`` (event counts + efficiency
    on one dual-axis figure, same as before), ``event_counts/`` (just the
    event-count outlines), and ``efficiency/`` (just the efficiency curves)
    -- splitting these out because the combined view crams up to ~13
    overlapping count histograms AND ~13 efficiency error-bar series onto
    one dual-axis figure.
    """
    eff = merged["eff"]
    if not eff:
        print("[aggregate] no efficiency data -> skipping")
        return
    stage_keys = merged["stage_keys"]
    stage_labels = merged["stage_labels"]
    stage_label_lookup = dict(zip(stage_keys, stage_labels))

    # group var-keyed accumulators by var
    vars_seen = set()
    for stage_key, by_v in eff.items():
        vars_seen.update(by_v.keys())

    var_lookup = {vc.var_save_name: vc for vc in EFFICIENCY_VARS}

    eff_dict = {}

    for var_save_name in sorted(vars_seen):
        var_config = var_lookup.get(var_save_name)

        if var_config is None:
            continue

        if var_config.var_save_name == "muon-dir_phi":
            continue

        stages = _compute_efficiency_stage_data(eff, stage_keys, stage_label_lookup, var_save_name)
        if not stages:
            continue

        bins = var_config.bins
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        _render_efficiency_combined(stages, var_config, bins, bin_centers, save_fig_dir,
                                    var_save_name, save_fig, show_fig)
        _render_efficiency_event_counts(stages, var_config, bins, bin_centers, save_fig_dir,
                                        var_save_name, save_fig, show_fig)
        _render_efficiency_only(stages, var_config, bins, bin_centers, save_fig_dir,
                                var_save_name, save_fig, show_fig)

        eff_dict[var_save_name] = {
            "eff_list": [s["eff"] for s in stages],
            "eff_err_list": [s["eff_err"] for s in stages],
        }

    # Integrated purity is identical for every efficiency variable (same event counts).
    purity_printed = False
    for vs in sorted(vars_seen):
        if vs not in var_lookup:
            continue
        last_stage = next((s for s in reversed(stage_keys) if s in eff and vs in eff[s]), None)
        if last_stage is None:
            continue
        ea_last = eff[last_stage][vs]
        if ea_last.n_at_stage_int <= 0:
            continue
        purity_w = ea_last.n_total_signal_int / ea_last.n_at_stage_int * 100.0
        n_raw_denom = getattr(ea_last, "n_at_stage_int_raw", 0.0)
        purity_raw = (
            ea_last.n_total_signal_int_raw / n_raw_denom * 100.0
            if n_raw_denom > 0
            else 0.0
        )
        print(
            f"[aggregate] final selection purity: {purity_w:.2f}% (weighted)  "
            f"{purity_raw:.2f}% (raw counts, notebook-style)",
            flush=True,
        )
        purity_printed = True
        break

    if not purity_printed:
        final_stage = "2prong-mup"
        bar_final = merged.get("bar", {}).get(final_stage, {}).get("topology")
        if bar_final is not None:
            total = float(np.sum(bar_final.mc_counts))
            if total > 0:
                signal = float(bar_final.mc_counts[0])  # νμ CC 1p0π (first topology bin)
                print(
                    f"[aggregate] final selection purity (from bar topology, no mcnu eff): "
                    f"{100.0 * signal / total:.2f}%  (signal={signal:.0f} / total={total:.0f})",
                    flush=True,
                )
            else:
                print("[aggregate] WARN: no MC events at final stage for purity", flush=True)
        else:
            print(
                "[aggregate] WARN: could not compute purity (no mcnu efficiency accumulators "
                "and no final-stage bar breakdown)",
                flush=True,
            )

    # Save the eff dict for downstream tools -- under pkl/ alongside every other
    # pickled output (see _save_fig_and_pkl), even though this one has no PNG
    # counterpart (it's the raw accumulator dict, not a single rendered plot).
    # Written unconditionally (matches this function's pre-existing behavior --
    # not gated on save_fig, unlike the per-plot PNG/pkl pairs above).
    pkl_dir = path.join(save_fig_dir, "pkl")
    makedirs(pkl_dir, exist_ok=True)
    out_pkl = path.join(pkl_dir, "eff_dict.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(eff_dict, f)
    print(f"[aggregate] wrote {out_pkl}")


# ===========================================================================
def render_cutflow_table(merged: dict, save_fig_dir: str, pot_str: str,
                         save_fig: bool, show_fig: bool):
    """Styled cutflow table — one row per selection stage.

    Columns: Selection | Step Eff. | Cum. Eff. | Purity
      - Step Eff.  = signal at this stage / signal at PREVIOUS stage
                     (survival rate of each sequential cut; 100% for the first stage)
      - Cum. Eff.  = signal at this stage / signal at the first (pre-cut) stage
      - Purity     = signal / all-topology total at this cumulative stage

    Signal = topology categories 0 + 1 (Signal 1p + Signal Np).
    All counts are POT-weighted (already scaled by apply_global_exposure_scales).
    """
    bar = merged["bar"]
    stage_keys   = merged.get("stage_keys", [])
    stage_labels = merged.get("stage_labels", [])
    label_lookup = dict(zip(stage_keys, stage_labels))

    avail_stages = [k for k in stage_keys if k in bar and "topology" in bar[k]]
    if not avail_stages:
        print("[aggregate] no breakdown stages -> skipping cutflow table")
        return

    missing = [k for k in stage_keys if k not in bar]
    if missing:
        print(
            f"[aggregate] WARN: cutflow table has {len(avail_stages)}/{len(stage_keys)} stages. "
            f"Missing from bar (stale pickle?): {missing}. "
            "Delete the batch pickle(s) and rerun with skip_existing_batches=False to regenerate.",
            flush=True,
        )

    # Pre-cut signal total (denominator for cumulative efficiency)
    first_counts = np.asarray(bar[avail_stages[0]]["topology"].mc_counts, dtype=float)
    sig_total = float(first_counts[0] + first_counts[1])

    rows = []
    prev_sig = sig_total
    for stage_key in avail_stages:
        counts   = np.asarray(bar[stage_key]["topology"].mc_counts, dtype=float)
        cum_sig  = float(counts[0] + counts[1])
        cum_tot  = float(counts.sum())

        step_eff = cum_sig / prev_sig  if prev_sig  > 0 else 0.0
        cum_eff  = cum_sig / sig_total if sig_total  > 0 else 0.0
        purity   = cum_sig / cum_tot   if cum_tot    > 0 else 0.0

        rows.append([
            label_lookup.get(stage_key, stage_key),
            f"{step_eff:.1%} ({cum_sig:.0f}/{prev_sig:.0f})",
            f"{cum_eff:.1%}  ({cum_sig:.0f}/{sig_total:.0f})",
            f"{purity:.1%}  ({cum_sig:.0f}/{cum_tot:.0f})",
        ])
        prev_sig = cum_sig

    col_labels = ["Selection", "Step Eff.", "Cum. Eff.", "Purity"]

    fig, ax = plt.subplots(figsize=(24, len(rows) * 0.42 + 1.5))
    ax.axis("off")

    table = ax.table(cellText=rows, colLabels=col_labels, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 4.5)

    for cell in table.get_celld().values():
        cell.set_linewidth(0)

    for i in range(1, len(rows) + 1):
        bg = "#f0f4f8" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            table[i, j].set_facecolor(bg)
        table[i, 0].set_text_props(ha="left")
        table[i, 0].PAD = 0.05

    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#2c3e50")
        table[0, j].set_text_props(color="white", fontweight="bold")
    table[0, 0].set_text_props(ha="left", color="white", fontweight="bold")
    table[0, 0].PAD = 0.05

    fig.canvas.draw()
    renderer  = fig.canvas.get_renderer()
    ax_bbox   = ax.get_window_extent(renderer)

    def hline(row_idx, use_top, color, lw):
        c0 = table[row_idx, 0].get_window_extent(renderer)
        cn = table[row_idx, len(col_labels) - 1].get_window_extent(renderer)
        y  = ((c0.y1 if use_top else c0.y0) - ax_bbox.y0) / ax_bbox.height
        x0 = (c0.x0 - ax_bbox.x0) / ax_bbox.width
        x1 = (cn.x1 - ax_bbox.x0) / ax_bbox.width
        ax.plot([x0, x1], [y, y], color=color, lw=lw, transform=ax.transAxes, clip_on=False)

    hline(0, use_top=True,  color="#2c3e50", lw=1.5)
    hline(0, use_top=False, color="#2c3e50", lw=1.5)
    for i in range(1, len(rows) + 1):
        hline(i, use_top=False, color="#cccccc", lw=0.7)

    x_left   = (table[0, 0].get_window_extent(renderer).x0 - ax_bbox.x0) / ax_bbox.width
    x_right  = (table[0, len(col_labels)-1].get_window_extent(renderer).x1 - ax_bbox.x0) / ax_bbox.width
    x_center = (x_left + x_right) / 2
    y_top    = (table[0, 0].get_window_extent(renderer).y1 - ax_bbox.y0) / ax_bbox.height

    ax.text(x_left,   y_top + 0.02, "SBND Work-In-Progress",
            transform=ax.transAxes, ha="left",   va="bottom", fontweight="bold", fontsize=14)
    ax.text(x_right,  y_top + 0.02, pot_str,
            transform=ax.transAxes, ha="right",  va="bottom", fontsize=14)
    ax.text(x_center, y_top + 0.12,
            r"$\boldsymbol{\nu_e}$ CC $\boldsymbol{1eNp0\pi}$ Selection Cutflow",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=28, fontweight="bold")

    _save_fig_and_pkl(fig, save_fig_dir, "cutflow_table", save_fig, show_fig)


# ===========================================================================
def render_breakdown_table(merged: dict, save_fig_dir: str, pot_str: str,
                           save_fig: bool, show_fig: bool):
    """Styled table of topology category counts at the final selection stage.

    Rows: one per topology category (topology_labels order).
    Columns: Category | Selected / Total | Efficiency | % of Selected

    'Total' is the first available breakdown stage (pre-cut denominator).
    'Selected' is the last available breakdown stage (final selection).
    Counts are POT-weighted (already scaled by apply_global_exposure_scales).
    """
    bar = merged["bar"]
    stage_keys = merged["stage_keys"]

    avail_stages = [k for k in stage_keys if k in bar]
    if len(avail_stages) < 1:
        print("[aggregate] no breakdown stages -> skipping breakdown table")
        return

    bb_total    = bar[avail_stages[0]].get("topology")
    bb_selected = bar[avail_stages[-1]].get("topology")
    if bb_total is None or bb_selected is None:
        print("[aggregate] no topology breakdown data -> skipping breakdown table")
        return

    totals_arr   = np.asarray(bb_total.mc_counts,    dtype=float)
    selected_arr = np.asarray(bb_selected.mc_counts, dtype=float)
    total_selected = float(selected_arr.sum())

    rows = []
    for i, label in enumerate(topology_labels):
        tot = float(totals_arr[i])
        sel = float(selected_arr[i])
        eff  = sel / tot  if tot  > 0 else 0.0
        frac = sel / total_selected if total_selected > 0 else 0.0
        rows.append([label, f"{sel:.1f} / {tot:.1f}", f"{eff:.1%}", f"{frac:.1%}"])

    col_labels = ["Category", "Selected / Total", "Efficiency", "% of Selected"]

    fig, ax = plt.subplots(figsize=(14, len(rows) * 0.55 + 1.5))
    ax.axis("off")

    table = ax.table(cellText=rows, colLabels=col_labels, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 3.5)

    for cell in table.get_celld().values():
        cell.set_linewidth(0)

    for i in range(1, len(rows) + 1):
        bg = "#f0f4f8" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            table[i, j].set_facecolor(bg)
        table[i, 0].set_text_props(ha="left")
        table[i, 0].PAD = 0.05

    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#2c3e50")
        table[0, j].set_text_props(color="white", fontweight="bold")
    table[0, 0].set_text_props(ha="left", color="white", fontweight="bold")
    table[0, 0].PAD = 0.05

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_bbox = ax.get_window_extent(renderer)

    def hline(row_idx, use_top, color, lw):
        c0 = table[row_idx, 0].get_window_extent(renderer)
        cn = table[row_idx, len(col_labels) - 1].get_window_extent(renderer)
        y  = ((c0.y1 if use_top else c0.y0) - ax_bbox.y0) / ax_bbox.height
        x0 = (c0.x0 - ax_bbox.x0) / ax_bbox.width
        x1 = (cn.x1 - ax_bbox.x0) / ax_bbox.width
        ax.plot([x0, x1], [y, y], color=color, lw=lw, transform=ax.transAxes, clip_on=False)

    hline(0, use_top=True,  color="#2c3e50", lw=1.5)
    hline(0, use_top=False, color="#2c3e50", lw=1.5)
    for i in range(1, len(rows) + 1):
        hline(i, use_top=False, color="#cccccc", lw=0.7)

    x_left  = (table[0, 0].get_window_extent(renderer).x0 - ax_bbox.x0) / ax_bbox.width
    x_right = (table[0, len(col_labels)-1].get_window_extent(renderer).x1 - ax_bbox.x0) / ax_bbox.width
    x_center = (x_left + x_right) / 2
    y_top = (table[0, 0].get_window_extent(renderer).y1 - ax_bbox.y0) / ax_bbox.height

    last_label = merged.get("stage_labels", [""])[
        merged.get("stage_keys", []).index(avail_stages[-1])
        if avail_stages[-1] in merged.get("stage_keys", []) else -1
    ]
    ax.text(x_left,   y_top + 0.02, "SBND Work-In-Progress",
            transform=ax.transAxes, ha="left",   va="bottom", fontweight="bold", fontsize=12)
    ax.text(x_right,  y_top + 0.02, pot_str,
            transform=ax.transAxes, ha="right",  va="bottom", fontsize=12)
    ax.text(x_center, y_top + 0.10,
            r"Background Breakdown — " + (last_label or avail_stages[-1]),
            transform=ax.transAxes, ha="center", va="bottom", fontsize=20, fontweight="bold")

    _save_fig_and_pkl(fig, save_fig_dir, "bkgd_breakdown_table", save_fig, show_fig)


# ===========================================================================
def aggregate_and_render(
    in_dir: str,
    save_fig_dir: str,
    *,
    data_pot: float | None = None,
    cosmic_estimate: str = "intime",
    f_offbeam_frac: float = 0.08,
    skip_global_exposure: bool = False,
    syst_disk_root: str | None = None,
    save_fig: bool = True,
    show_fig: bool = False,
    render_breakdown_independent: bool = True,
) -> dict:
    """Aggregate per-chunk pickles under ``in_dir`` and render every plot/table
    into ``save_fig_dir``.

    ``render_breakdown_independent``: set False to skip
    ``render_summary_breakdown_plot``/``render_efficiency_plots``/``render_cutflow_table``/
    ``render_breakdown_table``. Their content doesn't depend on
    ``config.stages.EVT_BREAKDOWN_TYPE`` (they're built from the ``eff``/``bar``
    accumulators, which are filled the same way regardless of that setting) -- so if
    you're re-running just to pick up a different ``EVT_BREAKDOWN_TYPE``'s
    ``selection_<var>.png`` plots (see ``EVT_BREAKDOWN_DIR_NAME``), those four would
    otherwise be recomputed and rewritten with numerically-identical content every time,
    for no reason. ``render_overlay_plots`` (which renders both the EVT_BREAKDOWN
    selection plots and the always-present pdg particle-KE plot) always runs, as does
    ``render_response_matrix_plots`` (renders a plot per ``VariableConfig.response_matrix=True``
    variable; a no-op if none are set, or if ``stages_mod.DISABLE_RESPONSE_MATRIX_PLOTS``).

    Single implementation for the whole reduce pass -- both the CLI (``main``,
    below) and ``pyanalib.event_selection_pipeline.EventSelectionPipeline.run_aggregate``
    (via this analysis's ``hooks.aggregate_and_render``) call this directly
    now, instead of each maintaining its own ~90-line copy of the same five
    aggregate-then-render steps (they had quietly drifted: one used a
    ``mc_pot`` POT fallback, the other a hardcoded ``1.0`` -- see the
    ``mc_pot``-preferring fallback below, which matches what the live batched
    workflow already produces).

    Returns the ``merged_histdata.pkl`` payload dict (also written to disk
    at ``<save_fig_dir>/merged_histdata.pkl``).
    """
    makedirs(save_fig_dir, exist_ok=True)

    # ---- discover chunk pickles by sample
    chunk_groups = collect_chunks(in_dir)

    # ---- aggregate per sample
    samples = {}
    for s, files in chunk_groups.items():
        if not files:
            continue
        print(f"[aggregate] aggregating {len(files)} chunks for sample={s}")
        samples[s] = aggregate_chunk_files(files)

    if not samples:
        raise RuntimeError(f"No chunk pickles found under {in_dir}")

    # ---- merge across samples
    merged = merge_samples(samples)
    n_hd_fixed, n_bar_fixed = sanitize_merged_histdata_finite(merged)
    if n_hd_fixed or n_bar_fixed:
        print(
            f"[aggregate] sanitized NaN/inf histogram bins (legacy weights): "
            f"{n_hd_fixed} OverlayHistData keys, {n_bar_fixed} bar breakdown rows",
            flush=True,
        )

    # ---- exposure denominators from chunk metadata (every pickle), then global scales
    totals = accumulate_exposure_totals_from_dir(in_dir)
    exposure_scales = None
    if not skip_global_exposure:
        exposure_scales = apply_global_exposure_scales(
            merged, totals, f_offbeam_coincident=f_offbeam_frac
        )
        print(f"[aggregate] applied global scales: {exposure_scales}")
    else:
        print("[aggregate] skip_global_exposure: histograms left as in pickles")

    # ---- POT string for axis labels / legend
    if data_pot is None:
        if totals.data_pot > 0:
            data_pot = totals.data_pot
        elif totals.mc_pot > 0:
            data_pot = totals.mc_pot
        else:
            data_pot = 1.0
    pot_str = get_pot_str(data_pot)
    print(f"[aggregate] data_pot (legend)={data_pot:.3e} -> POT label={pot_str}")

    # ---- render (syst bands only from pre-saved disk covariances)
    from analysis_village.nueNp0Pi.utils import _DEFAULT_SYST_DISK_ROOT
    _env_syst = os.environ.get("NUMUCC_SYST_DISK_ROOT")
    if _env_syst is not None:
        # env var explicitly set: empty string means "no systematics"
        resolved_syst_disk_root = _env_syst or None
    else:
        resolved_syst_disk_root = syst_disk_root or _DEFAULT_SYST_DISK_ROOT
    if resolved_syst_disk_root and not os.path.isdir(resolved_syst_disk_root):
        print(f"[aggregate] systematics disk root not found, skipping syst bands: {resolved_syst_disk_root}", flush=True)
        resolved_syst_disk_root = None
    else:
        print(f"[aggregate] systematics disk root: {resolved_syst_disk_root}", flush=True)
    render_overlay_plots(
        merged, plot_label_map={}, save_fig_dir=save_fig_dir,
        pot_str=pot_str, save_fig=save_fig, show_fig=show_fig,
        syst_disk_root=resolved_syst_disk_root,
    )
    render_response_matrix_plots(
        merged, save_fig_dir=save_fig_dir, save_fig=save_fig, show_fig=show_fig,
    )
    if not render_breakdown_independent:
        print(
            "[aggregate] render_breakdown_independent=False: skipping summary/efficiency/"
            "cutflow-table/breakdown-table (unchanged from a prior run with the same "
            "underlying batch pickles)",
            flush=True,
        )
    else:
        render_summary_breakdown_plot(
            merged, save_fig_dir,
            save_fig=save_fig, show_fig=show_fig,
            cosmic_estimate=cosmic_estimate,
        )
        render_efficiency_plots(merged, save_fig_dir, pot_str,
                                save_fig=save_fig, show_fig=show_fig)
        render_cutflow_table(merged, save_fig_dir, pot_str,
                             save_fig=save_fig, show_fig=show_fig)
        render_breakdown_table(merged, save_fig_dir, pot_str,
                               save_fig=save_fig, show_fig=show_fig)

    # save the merged dict so downstream stuff can pull histdata directly
    merged_payload = {
        "merged": merged,
        "data_pot": data_pot,
        "pot_str": pot_str,
        "exposure_totals": totals,
        "exposure_scales": exposure_scales,
        "cosmic_estimate": cosmic_estimate,
        "f_offbeam_frac": f_offbeam_frac,
    }
    pkl_dir = path.join(save_fig_dir, "pkl")
    makedirs(pkl_dir, exist_ok=True)
    out_pkl = path.join(pkl_dir, "merged_histdata.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(merged_payload, f)
    print(f"[aggregate] wrote {out_pkl}")

    return merged_payload


def main():
    args = parse_args()
    try:
        aggregate_and_render(
            args.in_dir,
            args.out_dir,
            data_pot=args.data_pot,
            cosmic_estimate=args.cosmic_estimate,
            f_offbeam_frac=args.f_offbeam_frac,
            skip_global_exposure=args.skip_global_exposure,
            syst_disk_root=args.syst_disk_root,
            save_fig=args.save_fig,
            show_fig=args.show_fig,
            render_breakdown_independent=args.render_breakdown_independent,
        )
    except RuntimeError as exc:
        print(f"[aggregate] {exc}; nothing to do")


if __name__ == "__main__":
    main()