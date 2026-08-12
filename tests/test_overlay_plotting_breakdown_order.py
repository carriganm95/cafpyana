"""Regression test for a real, pre-existing bug found and fixed in
pyanalib/overlay_plotting.py: overlay_hists_from_histdata reversed the
breakdown_display labels/colors to match its stacking convention, but never
reversed the matching per-category data (weights_categ/var_categ, built
straight from histdata.mc_hist in "cuts order"). Every legend label except the
optional Dirt entry ended up describing the WRONG category's actual counts --
confirmed against real SBND data, where a topology-breakdown plot at the final
selection stage showed mostly "NC"/"Other" despite the sample actually being
>90% signal (nu_e CC 1p0pi / Np0pi).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyanalib.chunked_selection import OverlayHistData
from pyanalib.overlay_plotting import overlay_hists_from_histdata


def _legend_texts(fig):
    ax = fig.axes[0]
    leg = ax.get_legend()
    assert leg is not None, "expected a legend on the rendered plot"
    return [t.get_text() for t in leg.get_texts()]


def test_signal_first_category_gets_signal_label_not_background():
    """All events in the FIRST ("signal") cuts-order category must be
    reported under the FIRST breakdown_display label, not the last one.
    """
    bins = np.array([0.0, 1.0, 2.0])
    labels = ["Signal", "Middle", "Background"]
    colors = ["navy", "gray", "black"]
    breakdown_display = {"topology": (labels, colors, None)}

    n_cat = 3
    hd = OverlayHistData(var_save_name="x", breakdown_type="topology", bins=bins, n_cat=n_cat)
    hd.mc_hist = np.zeros((n_cat, 2))
    hd.mc_hist[0] = [10.0, 0.0]  # ALL weight in cuts-order index 0 ("Signal")
    hd.mc_err2 = hd.mc_hist.copy()
    hd.has_mc = True

    plt.close("all")
    fig_before = set(plt.get_fignums())
    overlay_hists_from_histdata(hd, breakdown_display, plot=True, save_fig=False, load_syst_from_summary=False)
    fig_after = [n for n in plt.get_fignums() if n not in fig_before]
    fig = plt.figure(fig_after[0])
    texts = _legend_texts(fig)
    plt.close(fig)

    signal_line = next(t for t in texts if t.startswith("Signal"))
    background_line = next(t for t in texts if t.startswith("Background"))
    assert "(10, 100.0%)" in signal_line, texts
    assert "(0, 0.0%)" in background_line, texts


def test_last_category_gets_last_label():
    """Symmetric check: all weight in the LAST cuts-order category must be
    reported under the LAST label.
    """
    bins = np.array([0.0, 1.0, 2.0])
    labels = ["Signal", "Middle", "Background"]
    colors = ["navy", "gray", "black"]
    breakdown_display = {"topology": (labels, colors, None)}

    n_cat = 3
    hd = OverlayHistData(var_save_name="x", breakdown_type="topology", bins=bins, n_cat=n_cat)
    hd.mc_hist = np.zeros((n_cat, 2))
    hd.mc_hist[-1] = [7.0, 0.0]  # ALL weight in cuts-order index -1 ("Background")
    hd.mc_err2 = hd.mc_hist.copy()
    hd.has_mc = True

    plt.close("all")
    fig_before = set(plt.get_fignums())
    overlay_hists_from_histdata(hd, breakdown_display, plot=True, save_fig=False, load_syst_from_summary=False)
    fig_after = [n for n in plt.get_fignums() if n not in fig_before]
    fig = plt.figure(fig_after[0])
    texts = _legend_texts(fig)
    plt.close(fig)

    signal_line = next(t for t in texts if t.startswith("Signal"))
    background_line = next(t for t in texts if t.startswith("Background"))
    assert "(0, 0.0%)" in signal_line, texts
    assert "(7, 100.0%)" in background_line, texts
