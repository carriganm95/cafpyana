"""Generic true(x)-vs-reco(y) "response matrix" (a.k.a. migration matrix) plotting.

Renders a 2D heatmap of a variable's generator-truth value (x-axis) against its
reconstructed value (y-axis), built from a filled
:class:`pyanalib.chunked_selection.OverlayHistData`'s ``response_hist`` (see that class's
``fill_response_from_df`` and :class:`pyanalib.variable_config.VariableConfig`'s
``response_matrix``/``response_matrix_bins``/``response_matrix_label`` fields, which opt a
variable into this plot being filled/rendered automatically by the nueNp0Pi pipeline).

Cell color encodes the raw (POT-weighted) bin count; the annotated text in each cell is the
column-normalized percentage -- i.e. of all events whose TRUE value landed in that column's
bin, what percent reconstructed into each row's RECO bin -- so each column's percentages sum to
(approximately) 100%. This is the standard way physics analyses visualize bin migration/
resolution/bias between a generator-truth and a reconstructed variable.

No hardcoded analysis-specific paths or breakdown registries live here -- same "reusable
engine, analysis package does the wiring" split as ``pyanalib.overlay_plotting`` (see that
module's docstring). Style calls route through the same shared
``analysis_village.plot_style.sbnd_style`` library.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from pyanalib.logging_utils import get_logger
from analysis_village.plot_style.sbnd_style import add_approval_text

logger = get_logger(__name__)


def response_matrix_from_histdata(
    histdata,
    var_config,
    title: Optional[str] = None,
    approval: str = "wip",
    cmap: str = "Blues",
    annotate: bool = True,
    annotate_fmt: str = "{:.1f}%",
    annotate_fontsize: Optional[float] = None,
    colorbar_label: str = "Counts",
    figsize: Tuple[float, float] = (8.5, 7.5),
    plot: bool = True,
    save_fig: bool = False,
    save_name: Optional[str] = None,
    dpi: int = 300,
    fig_ext: str = ".png",
) -> Dict[str, Any]:
    """Render the true(x)-vs-reco(y) response-matrix heatmap for one variable.

    Parameters
    ----------
    histdata : OverlayHistData
        Must have been filled via ``fill_response_from_df`` (``has_response=True`` and
        ``response_hist`` not None) -- typically by opting a variable into
        ``VariableConfig.response_matrix=True`` before the batch producing this histdata was
        mapped. If not filled, no plot is made and ``{"ok": False, ...}`` is returned.
    var_config : VariableConfig
        Used for axis labels (prefers ``var_labels[1]``/``var_labels[2]`` -- the
        reco-specific/true-specific labels several ``config/plots.py`` factories already
        provide -- falling back to ``"Reco "``/``"True "`` + ``var_labels[0]`` when those
        aren't set) and ``response_matrix_label`` (optional real ``ax.set_title`` text --
        independent of the approval watermark below, same as the rest of the pipeline where a
        plot's title and its "SBND Internal"/etc. tag are two separate pieces of text).
    approval : str
        Passed straight to ``analysis_village.plot_style.sbnd_style.add_approval_text`` -- same
        function, same fixed position/fontsize (``0.03, 1.07, "left"``, default ``fontsize=20``)
        that ``pyanalib.overlay_plotting.overlay_hists_from_histdata`` calls it with for every
        other plot in the pipeline, so this tag renders identically to theirs. One of
        ``"wip"`` (default here -- "SBND Work in Progress"), ``"internal"`` ("SBND Internal",
        the overlay engine's own default), ``"preliminary"``, or anything else to render no tag.
    annotate_fontsize : float, optional
        Defaults to a size that scales down automatically as the grid gets denser (more bins
        -> smaller text), so labels stay legible without per-variable tuning.

    Returns
    -------
    dict with ``ok`` (bool). When ``ok`` is True: ``fig``, ``ax``, ``counts`` (raw POT-weighted
    2D histogram, indexed ``[reco_bin, true_bin]``), ``pct`` (column-normalized percentages,
    same indexing), and ``bins`` (the edges actually used).
    """
    if not getattr(histdata, "has_response", False) or getattr(histdata, "response_hist", None) is None:
        logger.warning(
            "response_matrix_from_histdata(%r): has_response=False / response_hist is None -- "
            "no plot made. response_matrix must be set on the VariableConfig BEFORE the batch "
            "producing this histdata was mapped.",
            getattr(var_config, "var_save_name", None),
        )
        return {"ok": False, "reason": "no response_hist filled (has_response=False)"}

    counts = np.asarray(histdata.response_hist, dtype=float)  # [reco_bin, true_bin]
    bins = np.asarray(
        histdata.response_bins if getattr(histdata, "response_bins", None) is not None
        else histdata.bins
    )
    n_bin = counts.shape[0]

    col_sums = counts.sum(axis=0, keepdims=True)  # (1, n_true_bin)
    with np.errstate(invalid="ignore", divide="ignore"):
        pct = np.where(col_sums > 0, 100.0 * counts / col_sums, 0.0)

    var_labels = getattr(var_config, "var_labels", None) or [
        getattr(var_config, "var_save_name", "")
    ]
    if len(var_labels) >= 3:
        x_label, y_label = var_labels[2], var_labels[1]
    else:
        x_label, y_label = f"True {var_labels[0]}", f"Reco {var_labels[0]}"

    resolved_title = title
    if resolved_title is None:
        resolved_title = getattr(var_config, "response_matrix_label", None)

    fig, ax = plt.subplots(figsize=figsize)
    mesh = ax.pcolormesh(bins, bins, counts, cmap=cmap, shading="flat")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(colorbar_label, fontsize=14)

    if annotate:
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        vmax = counts.max() if counts.size else 0.0
        resolved_fontsize = (
            annotate_fontsize if annotate_fontsize is not None
            else max(4.0, min(9.0, 110.0 / max(n_bin, 1)))
        )
        for i in range(n_bin):        # reco bin (row / y)
            for j in range(n_bin):    # true bin (col / x)
                color = "white" if (vmax > 0 and counts[i, j] / vmax > 0.5) else "black"
                ax.text(
                    bin_centers[j], bin_centers[i], annotate_fmt.format(pct[i, j]),
                    ha="center", va="center", fontsize=resolved_fontsize, color=color,
                )

    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(bins[0], bins[-1])
    ax.set_xlabel(x_label, fontsize=18)
    ax.set_ylabel(y_label, fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=13)
    if resolved_title:
        ax.set_title(resolved_title, fontsize=16)
    # Same call, same position/fontsize, same function every other plot in the pipeline uses
    # for its "SBND Internal"/"SBND Preliminary" tag (see
    # pyanalib.overlay_plotting.overlay_hists_from_histdata's identical
    # add_approval_text(approval, 0.03, 1.07, "left") call) -- independent of whatever
    # resolved_title is set to, same as elsewhere in the pipeline where the approval tag and
    # the plot's own title are two separate pieces of text.
    add_approval_text(approval, 0.03, 1.07, "left")

    if save_fig and save_name is not None:
        fig.savefig(save_name + fig_ext, bbox_inches="tight", dpi=dpi)

    if plot:
        plt.show()
    else:
        plt.close(fig)

    return {"ok": True, "fig": fig, "ax": ax, "counts": counts, "pct": pct, "bins": bins}
