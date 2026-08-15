"""Tests for pyanalib.response_matrix_plotting.response_matrix_from_histdata: the true(x)-vs-
reco(y) heatmap rendered from OverlayHistData.response_hist (see
test_chunked_selection_response_matrix.py for the map-time fill it consumes).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyanalib.chunked_selection import OverlayHistData
from pyanalib.response_matrix_plotting import response_matrix_from_histdata


class _VarConfig:
    var_save_name = "dpT"
    var_labels = [r"$\delta p_T$ [MeV/c]", r"$\delta p_T^{reco.}$ [MeV/c]", r"$\delta p_T^{true}$ [MeV/c]"]
    response_matrix = True
    response_matrix_label = None


def _make_filled_hd():
    bins = np.array([0.0, 1.0, 2.0])
    hd = OverlayHistData(var_save_name="dpT", breakdown_type="topology", bins=bins, n_cat=1)
    # true bin0: 3 reco-bin0 events + 1 reco-bin1 event -> col sums to 4, pct = [75%, 25%]
    # true bin1: 0 events -> pct = [0%, 0%] (guarded div-by-zero)
    hd.response_hist = np.array([[3.0, 0.0],
                                  [1.0, 0.0]])
    hd.response_bins = bins
    hd.has_response = True
    return hd


def test_returns_not_ok_when_never_filled():
    bins = np.array([0.0, 1.0, 2.0])
    hd = OverlayHistData(var_save_name="dpT", breakdown_type="topology", bins=bins, n_cat=1)

    result = response_matrix_from_histdata(hd, _VarConfig(), plot=False)

    assert result["ok"] is False
    assert "fig" not in result


def test_column_normalized_percentages():
    hd = _make_filled_hd()
    plt.close("all")

    result = response_matrix_from_histdata(hd, _VarConfig(), plot=False, save_fig=False)

    assert result["ok"] is True
    pct = result["pct"]
    assert pct.shape == (2, 2)
    np.testing.assert_allclose(pct[:, 0], [75.0, 25.0])   # true bin0 column sums to 100%
    np.testing.assert_allclose(pct[:, 1], [0.0, 0.0])     # true bin1 column: empty, no div-by-zero
    plt.close(result["fig"])


def test_axis_labels_prefer_reco_true_var_labels():
    hd = _make_filled_hd()
    plt.close("all")

    result = response_matrix_from_histdata(hd, _VarConfig(), plot=False, save_fig=False)

    ax = result["ax"]
    assert ax.get_xlabel() == _VarConfig.var_labels[2]  # true -> x-axis
    assert ax.get_ylabel() == _VarConfig.var_labels[1]  # reco -> y-axis
    plt.close(result["fig"])


class _VarConfigSingleLabel:
    var_save_name = "dpT"
    var_labels = [r"$\delta p_T$ [MeV/c]"]
    response_matrix = True
    response_matrix_label = None


def test_axis_labels_fall_back_when_only_one_label_given():
    hd = _make_filled_hd()
    plt.close("all")

    result = response_matrix_from_histdata(hd, _VarConfigSingleLabel(), plot=False, save_fig=False)

    ax = result["ax"]
    assert ax.get_xlabel().startswith("True ")
    assert ax.get_ylabel().startswith("Reco ")
    plt.close(result["fig"])


def test_title_override_via_response_matrix_label():
    hd = _make_filled_hd()
    plt.close("all")

    class _VarConfigTitled(_VarConfig):
        response_matrix_label = "My Custom Title"

    result = response_matrix_from_histdata(hd, _VarConfigTitled(), plot=False, save_fig=False)

    assert result["ax"].get_title() == "My Custom Title"
    plt.close(result["fig"])


def _sbnd_tag_texts(ax):
    return [t for t in ax.texts if "SBND" in t.get_text()]


def test_approval_watermark_matches_add_approval_text_style():
    """The "SBND Work in Progress" tag must render through the same
    analysis_village.plot_style.sbnd_style.add_approval_text call (same position/fontsize/
    mathtext-bold-SBND styling) that every other plot in the pipeline uses for its own
    "SBND Internal"/"SBND Preliminary" tag -- not a separately-styled one-off.
    """
    hd = _make_filled_hd()
    plt.close("all")

    result = response_matrix_from_histdata(hd, _VarConfig(), plot=False, save_fig=False)

    ax = result["ax"]
    tags = _sbnd_tag_texts(ax)
    assert len(tags) == 1
    tag = tags[0]
    assert tag.get_text() == r"$\mathbf{SBND}$ Work in Progress"
    assert tag.get_position() == (0.03, 1.07)
    assert tag.get_color() == "black"
    plt.close(result["fig"])


def test_approval_watermark_present_alongside_a_custom_title():
    """response_matrix_label sets an independent real title -- it must not suppress the
    approval tag, matching how the rest of the pipeline treats title vs. approval tag as two
    separate pieces of text.
    """
    hd = _make_filled_hd()
    plt.close("all")

    class _VarConfigTitled(_VarConfig):
        response_matrix_label = "My Custom Title"

    result = response_matrix_from_histdata(hd, _VarConfigTitled(), plot=False, save_fig=False)

    ax = result["ax"]
    assert ax.get_title() == "My Custom Title"
    assert len(_sbnd_tag_texts(ax)) == 1
    plt.close(result["fig"])


def test_approval_param_switches_modes():
    hd = _make_filled_hd()
    plt.close("all")

    result = response_matrix_from_histdata(
        hd, _VarConfig(), plot=False, save_fig=False, approval="internal"
    )

    tags = _sbnd_tag_texts(result["ax"])
    assert tags[0].get_text() == r"$\mathbf{SBND}$ Internal"
    plt.close(result["fig"])
