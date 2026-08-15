"""Tests for _apply_ratio_kwargs, the VariableConfig.ratio_mode -> overlay_hists_from_histdata
kwargs translator added to event_selection_aggregate.py's render_overlay_plots.

Uses minimal duck-typed stand-ins for OverlayHistData/VariableConfig (only the attributes
_apply_ratio_kwargs actually reads) rather than the real classes/pipeline, so these tests
don't need any real event data or the full pipeline dependency chain -- just this one
function's branching and arithmetic.
"""
import numpy as np
import pytest

from analysis_village.nueNp0Pi.event_selection_aggregate import _apply_ratio_kwargs


class FakeHD:
    def __init__(self, bins, mc_hist, breakdown_type="topology", has_truth=False, truth_hist=None):
        self.bins = np.asarray(bins)
        self.mc_hist = np.asarray(mc_hist)
        self.breakdown_type = breakdown_type
        self.has_truth = has_truth
        self.truth_hist = np.asarray(truth_hist) if truth_hist is not None else None


class FakeVC:
    """ratio_mode=None and all ratio_* fields default to the same values a real
    VariableConfig has for a variable that never opted in, per pyanalib.variable_config."""
    def __init__(self, **kw):
        self.var_save_name = "x"
        self.ratio_mode = None
        self.ratio_label = None
        self.ratio_signal_indices = None
        self.ratio_bkgd_indices = None
        self.ratio_breakdown_type = None
        for k, v in kw.items():
            setattr(self, k, v)


BINS = np.array([0.0, 1.0, 2.0, 3.0])


def test_ratio_mode_none_leaves_kwargs_untouched():
    vc = FakeVC(ratio_mode=None)
    hd = FakeHD(BINS, [[1, 2, 3], [4, 5, 6]])
    kwargs = {"ratio": False, "foo": "bar"}
    out = _apply_ratio_kwargs(hd, vc, dict(kwargs), disable_ratio_plots=False)
    assert out == kwargs


def test_disable_ratio_plots_overrides_explicit_ratio_true():
    vc = FakeVC(ratio_mode="reco_true")
    hd = FakeHD(BINS, [[1, 2, 3], [4, 5, 6]], has_truth=True, truth_hist=[1, 1, 1])
    out = _apply_ratio_kwargs(hd, vc, {"ratio": True}, disable_ratio_plots=True)
    assert out["ratio"] is False
    assert "ratio_vars" not in out


def test_data_mc_mode_sets_ratio_true_and_label():
    vc = FakeVC(ratio_mode="data_mc", ratio_label="Custom label")
    hd = FakeHD(BINS, [[1, 2, 3], [4, 5, 6]])
    out = _apply_ratio_kwargs(hd, vc, {}, disable_ratio_plots=False)
    assert out["ratio"] is True
    assert out["ratio_label"] == "Custom label"
    assert "ratio_vars" not in out  # built-in engine Data/MC path, no custom ratio_vars needed


def test_reco_true_skips_gracefully_without_truth_hist():
    vc = FakeVC(ratio_mode="reco_true")
    hd = FakeHD(BINS, [[1, 2, 3], [4, 5, 6]], has_truth=False)
    kwargs = {"ratio": False}
    out = _apply_ratio_kwargs(hd, vc, dict(kwargs), disable_ratio_plots=False)
    assert out == kwargs  # untouched, not a crash


def test_reco_true_builds_correct_num_denom_histograms():
    vc = FakeVC(ratio_mode="reco_true")
    mc_hist = [[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]]  # total reco per bin = [2, 3, 4]
    truth_hist = [2.0, 3.0, 8.0]
    hd = FakeHD(BINS, mc_hist, has_truth=True, truth_hist=truth_hist)
    out = _apply_ratio_kwargs(hd, vc, {}, disable_ratio_plots=False)
    assert out["ratio"] is True
    num_w, denom_w = out["ratio_weights"]
    assert np.allclose(num_w, [2.0, 3.0, 4.0])
    assert np.allclose(denom_w, [2.0, 3.0, 8.0])
    assert np.allclose(out["ratio_bins"], BINS)
    assert out["ratio_label"] == "Reco/True"


def test_signal_bkgd_skips_on_breakdown_type_mismatch():
    vc = FakeVC(
        ratio_mode="signal_bkgd", ratio_breakdown_type="topology",
        ratio_signal_indices=[0], ratio_bkgd_indices=[1],
    )
    hd = FakeHD(BINS, [[1, 2, 3], [4, 5, 6]], breakdown_type="genie")
    kwargs = {"ratio": False}
    out = _apply_ratio_kwargs(hd, vc, dict(kwargs), disable_ratio_plots=False)
    assert out == kwargs


@pytest.mark.parametrize("sig, bkg", [(None, [1]), ([0], None), ([], [1]), ([0], [])])
def test_signal_bkgd_skips_on_missing_indices(sig, bkg):
    vc = FakeVC(
        ratio_mode="signal_bkgd", ratio_breakdown_type="topology",
        ratio_signal_indices=sig, ratio_bkgd_indices=bkg,
    )
    hd = FakeHD(BINS, [[1, 2, 3], [4, 5, 6]], breakdown_type="topology")
    kwargs = {"ratio": False}
    out = _apply_ratio_kwargs(hd, vc, dict(kwargs), disable_ratio_plots=False)
    assert out == kwargs


def test_signal_bkgd_skips_on_out_of_range_index():
    vc = FakeVC(
        ratio_mode="signal_bkgd", ratio_breakdown_type="topology",
        ratio_signal_indices=[0], ratio_bkgd_indices=[5],  # only 2 categories exist
    )
    hd = FakeHD(BINS, [[1, 2, 3], [4, 5, 6]], breakdown_type="topology")
    kwargs = {"ratio": False}
    out = _apply_ratio_kwargs(hd, vc, dict(kwargs), disable_ratio_plots=False)
    assert out == kwargs


def test_signal_bkgd_sums_multiple_indices_correctly():
    mc_hist = [
        [1.0, 1.0],  # category 0 -- signal
        [2.0, 2.0],  # category 1 -- signal
        [3.0, 3.0],  # category 2 -- background
        [4.0, 4.0],  # category 3 -- background
    ]
    vc = FakeVC(
        ratio_mode="signal_bkgd", ratio_breakdown_type="topology",
        ratio_signal_indices=[0, 1], ratio_bkgd_indices=[2, 3],
    )
    hd = FakeHD(np.array([0.0, 1.0, 2.0]), mc_hist, breakdown_type="topology")
    out = _apply_ratio_kwargs(hd, vc, {}, disable_ratio_plots=False)
    assert out["ratio"] is True
    sig_w, bkg_w = out["ratio_weights"]
    assert np.allclose(sig_w, [3.0, 3.0])  # 1 + 2
    assert np.allclose(bkg_w, [7.0, 7.0])  # 3 + 4
    assert out["ratio_label"] == "Signal/Background"


def test_plotspec_level_ratio_label_wins_over_var_config_level():
    """kwargs.setdefault means a PlotSpec.save_kwargs["ratio_label"] (already in kwargs
    before this function runs) always wins over VariableConfig.ratio_label."""
    vc = FakeVC(ratio_mode="reco_true", ratio_label="VC label")
    hd = FakeHD(BINS, [[1, 2, 3], [4, 5, 6]], has_truth=True, truth_hist=[1, 1, 1])
    out = _apply_ratio_kwargs(
        hd, vc, {"ratio_label": "PlotSpec label", "ratio": False}, disable_ratio_plots=False
    )
    assert out["ratio_label"] == "PlotSpec label"


def test_unknown_ratio_mode_leaves_kwargs_untouched():
    vc = FakeVC(ratio_mode="not_a_real_mode")
    hd = FakeHD(BINS, [[1, 2, 3], [4, 5, 6]])
    kwargs = {"ratio": False}
    out = _apply_ratio_kwargs(hd, vc, dict(kwargs), disable_ratio_plots=False)
    assert out == kwargs
