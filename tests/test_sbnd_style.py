"""analysis_village/plot_style/sbnd_style.py: generic plot-annotation and
heatmap helpers moved out of analysis_village/nueNp0Pi/utils.py.

These mostly draw onto the current matplotlib figure, so tests mainly check
they run without error and produce the expected returned/derived values
rather than doing pixel comparisons.
"""
import numpy as np
import matplotlib.pyplot as plt
import pytest

from analysis_village.plot_style.sbnd_style import (
    add_approval_text,
    add_chi2_text,
    add_genie_version_text,
    add_pot_text,
    bin_range_labels,
    format_singlebin_plot,
    get_text_color,
    get_textloc_x,
    plot_heatmap,
)


def test_bin_range_labels():
    assert bin_range_labels([0.0, 1.0, 2.5]) == ["0.00–1.00", "1.00–2.50"]


def test_get_text_color_picks_contrasting_color():
    # low value under default viridis+[0,1] norm -> dark color -> white text
    assert get_text_color(0.0) == "white"
    # high value -> bright color -> black text
    assert get_text_color(1.0) == "black"


def test_get_textloc_x_prefers_emptier_half():
    bins = np.array([0, 1, 2, 3, 4])
    # more weight in the first half -> put text on the right, at 1-x
    x, ha = get_textloc_x(np.array([10, 10, 1, 1]), bins, textloc=[0.05, 0.5])
    assert ha == "right"
    assert x == pytest.approx(0.95)
    # more weight in the second half -> put text on the left, at x
    x, ha = get_textloc_x(np.array([1, 1, 10, 10]), bins, textloc=[0.05, 0.5])
    assert ha == "left"
    assert x == pytest.approx(0.05)


@pytest.fixture(autouse=True)
def _close_figures_after_each_test():
    yield
    plt.close("all")


def test_add_approval_text_internal_and_preliminary_do_not_raise():
    fig, ax = plt.subplots()
    add_approval_text("internal", 0.5, 0.5, "right")
    add_approval_text("preliminary", 0.5, 0.6, "right")


def test_add_approval_text_unknown_mode_is_a_noop():
    fig, ax = plt.subplots()
    n_texts_before = len(ax.texts)
    add_approval_text("not-a-real-mode", 0.5, 0.5, "right")
    assert len(ax.texts) == n_texts_before


def test_add_pot_and_chi2_and_genie_text_do_not_raise():
    fig, ax = plt.subplots()
    add_pot_text("1e20 POT", 0.1, 0.1, "left")
    add_chi2_text(5.0, 0.3, 4, 0.2, 0.9, "left")
    add_genie_version_text(0.2, 0.8, "left")
    # explicit version override should also work
    add_genie_version_text(0.2, 0.7, "left", version="GENIE vTEST")


def test_format_singlebin_plot_clears_xticks():
    fig, ax = plt.subplots()
    ax.set_xticks([0, 1, 2])
    format_singlebin_plot()
    assert list(plt.gcf().axes[0].get_xticks()) == []


def test_plot_heatmap_runs_without_error():
    matrix = np.array([[1.0, 0.2], [0.2, 1.0]])
    plot_heatmap(matrix, bins=[0, 1, 2], plot=False, save_fig=False)


def test_plot_heatmap_asserts_matrix_matches_bins():
    matrix = np.array([[1.0, 0.2], [0.2, 1.0]])
    with pytest.raises(AssertionError):
        plot_heatmap(matrix, bins=[0, 1, 2, 3], plot=False, save_fig=False)
