"""Regression test for a real bug in pyanalib/overlay_plotting.py's
overlay_hists_from_histdata: the vline-marker arrow (drawn to indicate which
side of a cut threshold survives) has a fixed length -- 18% of the full axis
width -- computed without regard to how close the threshold sits to the axis
edge. For a threshold near xlim (e.g. an N-1 plot's electron-softmax cut at
x=0.9 on a 0-1 axis, "keep larger" direction), the arrow's tip lands past the
axis boundary; with clip_on=True (the old default) matplotlib truncated the
arrow right at the edge, which could chop off the arrowhead entirely and
leave what looked like a stray red bar with no direction indicator (caught by
the user via a real N-1 plot screenshot).

Fixed by drawing the arrow with clip_on=False so it's fully visible -- head
included -- even when it pokes out past the plot box, rather than silently
losing its meaning when clipped.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from pyanalib.chunked_selection import OverlayHistData
from pyanalib.overlay_plotting import overlay_hists_from_histdata


def _render_with_vline(vline):
    bins = np.array([0.0, 0.85, 0.9, 0.95, 1.0])
    labels = ["Signal", "Bkg"]
    colors = ["navy", "gray"]
    breakdown_display = {"topology": (labels, colors, None)}

    hd = OverlayHistData(var_save_name="x", breakdown_type="topology", bins=bins, n_cat=2)
    hd.mc_hist = np.zeros((2, 4))
    hd.mc_hist[0] = [0.0, 1.0, 2.0, 13.0]
    hd.mc_err2 = hd.mc_hist.copy()
    hd.has_mc = True

    plt.close("all")
    fig_before = set(plt.get_fignums())
    overlay_hists_from_histdata(
        hd, breakdown_display, plot=True, save_fig=False,
        load_syst_from_summary=False, vline=vline,
    )
    fig_after = [n for n in plt.get_fignums() if n not in fig_before]
    fig = plt.figure(fig_after[0])
    ax = fig.axes[0]
    arrows = [p for p in ax.patches if p.__class__.__name__ == "FancyArrow"]
    plt.close(fig)
    return arrows


@pytest.mark.parametrize("direction", [0, 1])
def test_vline_arrow_is_not_clipped_near_axis_edge(direction):
    """Regardless of which way the arrow points, it must not be clip_on=True
    -- that's what silently truncated/decapitated the arrow near an axis
    edge in the reported bug."""
    arrows = _render_with_vline([(0.9, direction)])
    assert len(arrows) == 1
    assert arrows[0].get_clip_on() is False


def test_vline_without_direction_draws_no_arrow():
    """A vline entry with no direction (just a threshold) should still only
    draw the dashed marker line, no arrow -- untouched by this fix."""
    arrows = _render_with_vline([(0.9,)])
    assert len(arrows) == 0
