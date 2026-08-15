"""Tests for the optional truth-side histogram (OverlayHistData.truth_hist) added to
support VariableConfig.ratio_mode == "reco_true": pyanalib.chunked_selection.OverlayHistData
.fill_truth_from_df, its participation in __iadd__/merge, and ChunkRunner wiring it up only
for variables that opt in.
"""
import numpy as np
import pandas as pd
import pytest

from pyanalib.chunked_selection import (
    ChunkRunner,
    OverlayHistData,
    PlotSpec,
    Stage,
)


def two_category_cuts(df, ret_cuts=True):
    return [df["x"] < 1.5, df["x"] >= 1.5]


FAKE_REGISTRY = {"fake": (2, two_category_cuts)}


# --------------------------------------------------------------------------
# OverlayHistData.fill_truth_from_df / __iadd__
# --------------------------------------------------------------------------
def test_fill_truth_from_df_fills_weighted_histogram():
    bins = np.array([0.0, 1.0, 2.0])
    hd = OverlayHistData(var_save_name="v", breakdown_type="fake", bins=bins, n_cat=2)
    df = pd.DataFrame({"y": [0.5, 0.5, 1.5], "pot_weight": [1.0, 2.0, 3.0]})

    hd.fill_truth_from_df(df, "y")

    assert hd.has_truth is True
    assert hd.truth_hist.tolist() == [3.0, 3.0]  # bin0: 1+2=3, bin1: 3
    assert hd.truth_err2.tolist() == [1.0**2 + 2.0**2, 3.0**2]
    # mc_hist/data_hist untouched by a truth fill
    assert hd.mc_hist.sum() == 0.0
    assert hd.data_hist.sum() == 0.0


def test_fill_truth_from_df_empty_df_still_marks_has_truth():
    bins = np.array([0.0, 1.0, 2.0])
    hd = OverlayHistData(var_save_name="v", breakdown_type="fake", bins=bins, n_cat=2)
    empty = pd.DataFrame({"y": [], "pot_weight": []})

    hd.fill_truth_from_df(empty, "y")

    assert hd.has_truth is True
    assert hd.truth_hist.sum() == 0.0


def test_overlay_hist_data_iadd_merges_truth_hist():
    bins = np.array([0.0, 1.0, 2.0])
    a = OverlayHistData(var_save_name="v", breakdown_type="fake", bins=bins, n_cat=2)
    b = OverlayHistData(var_save_name="v", breakdown_type="fake", bins=bins, n_cat=2)
    a.fill_truth_from_df(pd.DataFrame({"y": [0.5], "pot_weight": [1.0]}), "y")
    b.fill_truth_from_df(pd.DataFrame({"y": [0.5], "pot_weight": [4.0]}), "y")

    a += b

    assert a.truth_hist.tolist() == [5.0, 0.0]
    assert a.truth_err2.tolist() == [1.0 + 16.0, 0.0]
    assert a.has_truth is True


def test_overlay_hist_data_iadd_tolerates_legacy_object_without_truth_hist():
    """Merging in an object that predates truth_hist (e.g. an old unpickled instance)
    must not crash -- __iadd__ falls back to 0.0 via getattr.
    """
    shared_bins = np.array([0.0, 1.0, 2.0])
    a = OverlayHistData(var_save_name="v", breakdown_type="fake", bins=shared_bins, n_cat=2)

    class _LegacyStandIn:
        var_save_name = "v"
        breakdown_type = "fake"
        bins = shared_bins
        mc_hist = np.zeros((2, 2))
        mc_err2 = np.zeros((2, 2))
        intime_hist = np.zeros(2)
        intime_err2 = np.zeros(2)
        offbeam_hist = np.zeros(2)
        offbeam_err2 = np.zeros(2)
        dirt_hist = np.zeros(2)
        dirt_err2 = np.zeros(2)
        data_hist = np.zeros(2)
        data_err2 = np.zeros(2)
        mc_univ_hist = None
        has_mc = has_intime = has_offbeam = has_dirt = has_data = False
        # no truth_hist / truth_err2 / has_truth attributes at all

    a += _LegacyStandIn()
    assert a.truth_hist.tolist() == [0.0, 0.0]
    assert a.has_truth is False


# --------------------------------------------------------------------------
# ChunkRunner wiring: only fills truth_hist for ratio_mode == "reco_true", mc sample only
# --------------------------------------------------------------------------
class _RecoTrueVarConfig:
    var_save_name = "x"
    var_nu_col = "x"
    var_evt_reco_col = "x"
    var_evt_truth_col = "y"
    bins = np.array([0.0, 1.0, 2.0])
    ratio_mode = "reco_true"


class _DefaultVarConfig:
    var_save_name = "x"
    var_nu_col = "x"
    var_evt_reco_col = "x"
    var_evt_truth_col = "y"
    bins = np.array([0.0, 1.0, 2.0])
    ratio_mode = None


def _make_runner(var_config, sample="mc"):
    plot = PlotSpec(
        var_config=var_config,
        breakdown_type="fake",
        selector=lambda state, sample: state.get("evt"),
    )
    stage = Stage(key="s0", label="Stage 0", plots=[plot])
    return ChunkRunner(
        sample=sample,
        stages=[stage],
        efficiency_vars=[],
        breakdown_registry=FAKE_REGISTRY,
        signal_mask_fn=lambda df: df["x"] > 1.0,
    )


def test_chunk_runner_fills_truth_hist_when_ratio_mode_reco_true():
    runner = _make_runner(_RecoTrueVarConfig(), sample="mc")
    evt = pd.DataFrame({"x": [0.5, 1.5], "y": [1.5, 0.5], "pot_weight": [1.0, 1.0]})
    runner.run({"evt": evt})

    hd = next(iter(runner.histdata.values()))
    # reco (x) fills mc_hist as usual: bin0 gets x=0.5, bin1 gets x=1.5
    assert hd.mc_hist.sum(axis=0).tolist() == [1.0, 1.0]
    # truth (y) fills truth_hist from the SAME events, via the truth column instead --
    # y=1.5 (from the x=0.5 event) lands in bin1, y=0.5 (from the x=1.5 event) in bin0.
    assert hd.has_truth is True
    assert hd.truth_hist.tolist() == [1.0, 1.0]


def test_chunk_runner_does_not_fill_truth_hist_when_ratio_mode_unset():
    """Default (ratio_mode=None) variables must not pay for a truth-side fill."""
    runner = _make_runner(_DefaultVarConfig(), sample="mc")
    evt = pd.DataFrame({"x": [0.5, 1.5], "y": [1.5, 0.5], "pot_weight": [1.0, 1.0]})
    runner.run({"evt": evt})

    hd = next(iter(runner.histdata.values()))
    assert hd.mc_hist.sum() == 2.0  # reco side still filled normally
    assert hd.has_truth is False
    assert hd.truth_hist.sum() == 0.0


def test_chunk_runner_does_not_fill_truth_hist_for_non_mc_samples():
    """ratio_mode='reco_true' only makes sense for MC (no truth for data/intime/etc.)."""
    runner = _make_runner(_RecoTrueVarConfig(), sample="data")
    evt = pd.DataFrame({"x": [0.5, 1.5], "y": [1.5, 0.5]})
    runner.run({"evt": evt})

    hd = next(iter(runner.histdata.values()))
    assert hd.data_hist.sum() == 2.0
    assert hd.has_truth is False
    assert hd.truth_hist.sum() == 0.0
