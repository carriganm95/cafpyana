"""Tests for the custom (non-Data/MC) ratio-panel option added to
overlay_hists_from_histdata: ratio_vars / ratio_weights / ratio_bins / ratio_label.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyanalib.chunked_selection import OverlayHistData
from pyanalib.overlay_plotting import overlay_hists_from_histdata


def _make_hd():
    bins = np.array([0.0, 1.0, 2.0, 3.0])
    n_cat = 1
    hd = OverlayHistData(var_save_name="x", breakdown_type="topology", bins=bins, n_cat=n_cat)
    hd.mc_hist = np.array([[5.0, 8.0, 3.0]])
    hd.mc_err2 = hd.mc_hist.copy()
    hd.has_mc = True
    return hd, bins


def test_backward_compat_default_ratio_is_still_data_mc():
    """ratio=True with no ratio_vars must behave exactly as before: y-label 'Data/MC'."""
    hd, bins = _make_hd()
    labels = ["Signal"]
    colors = ["navy"]
    breakdown_display = {"topology": (labels, colors, None)}

    plt.close("all")
    result = overlay_hists_from_histdata(
        hd, breakdown_display, ratio=True, plot=False, save_fig=False,
        load_syst_from_summary=False,
    )
    fig = result["fig"]
    ax_r = fig.axes[1]
    assert ax_r.get_ylabel() == "Data/MC"
    assert result["ratio_vals"] is None
    assert result["ratio_num_hist"] is None
    plt.close(fig)


def test_custom_ratio_reco_vs_true():
    """ratio_vars=(reco, true) should histogram both and plot num/denom, not Data/MC."""
    hd, bins = _make_hd()
    labels = ["Signal"]
    colors = ["navy"]
    breakdown_display = {"topology": (labels, colors, None)}

    rng = np.random.default_rng(1)
    true_vals = rng.uniform(0, 3, 5000)
    reco_vals = true_vals  # identical -> ratio should be ~1 in every populated bin

    plt.close("all")
    result = overlay_hists_from_histdata(
        hd, breakdown_display, ratio=True,
        ratio_vars=(reco_vals, true_vals),
        ratio_label="Reco/True",
        plot=False, save_fig=False, load_syst_from_summary=False,
    )
    fig = result["fig"]
    ax_r = fig.axes[1]
    assert ax_r.get_ylabel() == "Reco/True"

    ratio_vals = result["ratio_vals"]
    ratio_err = result["ratio_err"]
    num_hist = result["ratio_num_hist"]
    denom_hist = result["ratio_denom_hist"]
    assert ratio_vals is not None and len(ratio_vals) == len(bins) - 1
    assert np.allclose(num_hist, denom_hist)  # identical inputs -> identical histograms
    assert np.allclose(ratio_vals, 1.0, atol=1e-9)
    assert np.all(ratio_err >= 0)

    # Data/MC-specific artifacts (syst hatch legend entries etc.) shouldn't appear
    # in the ratio-panel content when in custom mode -- no bar patches on ax_r besides
    # the errorbar's own artifacts and axhline.
    bar_patches = [p for p in ax_r.patches]
    assert bar_patches == [], "custom ratio mode should not draw the Data/MC syst band"
    plt.close(fig)


def test_custom_ratio_with_weights_and_custom_bins():
    """ratio_weights and ratio_bins should be honored independently of the main plot bins."""
    hd, _ = _make_hd()
    labels = ["Signal"]
    colors = ["navy"]
    breakdown_display = {"topology": (labels, colors, None)}

    num_vals = np.array([1.0, 1.0, 1.0, 3.0])
    denom_vals = np.array([1.0, 1.0, 3.0, 3.0])
    num_w = np.array([1.0, 1.0, 1.0, 1.0])
    denom_w = np.array([2.0, 2.0, 1.0, 1.0])
    custom_bins = np.array([0.0, 2.0, 4.0])

    plt.close("all")
    result = overlay_hists_from_histdata(
        hd, breakdown_display, ratio=True,
        ratio_vars=(num_vals, denom_vals),
        ratio_weights=(num_w, denom_w),
        ratio_bins=custom_bins,
        plot=False, save_fig=False, load_syst_from_summary=False,
    )
    # bin [0,2): num has three 1.0's, weight 1 each -> 3.0; denom has two 1.0's, weight 2 each -> 4.0
    # bin [2,4): num has one 3.0, weight 1 -> 1.0; denom has two 3.0's, weight 1 each -> 2.0
    assert np.allclose(result["ratio_num_hist"], [3.0, 1.0])
    assert np.allclose(result["ratio_denom_hist"], [4.0, 2.0])
    assert np.allclose(result["ratio_vals"], [0.75, 0.5])
    plt.close(result["fig"])


def test_custom_ratio_empty_denominator_bin_is_safe():
    """A bin with zero denominator weight must yield ratio=0, err=0, no NaN/inf."""
    hd, bins = _make_hd()
    labels = ["Signal"]
    colors = ["navy"]
    breakdown_display = {"topology": (labels, colors, None)}

    num_vals = np.array([2.5, 2.5])
    denom_vals = np.array([0.5, 0.5])  # falls in a different bin than num_vals

    plt.close("all")
    result = overlay_hists_from_histdata(
        hd, breakdown_display, ratio=True,
        ratio_vars=(num_vals, denom_vals),
        plot=False, save_fig=False, load_syst_from_summary=False,
    )
    ratio_vals = result["ratio_vals"]
    ratio_err = result["ratio_err"]
    assert not np.any(np.isnan(ratio_vals))
    assert not np.any(np.isinf(ratio_vals))
    assert not np.any(np.isnan(ratio_err))
    assert not np.any(np.isinf(ratio_err))
    # bin containing num_vals (2.5) has zero denom -> ratio forced to 0 there
    bin_idx = np.searchsorted(bins, 2.5, side="right") - 1
    assert ratio_vals[bin_idx] == 0.0
    plt.close(result["fig"])
