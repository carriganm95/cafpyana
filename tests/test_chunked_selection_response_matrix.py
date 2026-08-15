"""Tests for the optional true-vs-reco 2D histogram (OverlayHistData.response_hist) added to
support VariableConfig.response_matrix: pyanalib.chunked_selection.OverlayHistData
.fill_response_from_df, its participation in __iadd__/merge, and ChunkRunner wiring it up only
for variables that opt in (mirrors test_chunked_selection_ratio_truth.py's structure for the
sibling ratio_mode=="reco_true" mechanism).
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
# OverlayHistData.fill_response_from_df / __iadd__
# --------------------------------------------------------------------------
def test_fill_response_from_df_fills_2d_weighted_histogram():
    bins = np.array([0.0, 1.0, 2.0])
    hd = OverlayHistData(var_save_name="v", breakdown_type="fake", bins=bins, n_cat=2)
    # reco=x, true=y. Event0: reco bin0, true bin1. Event1: reco bin1, true bin0.
    df = pd.DataFrame({"x": [0.5, 1.5], "y": [1.5, 0.5], "pot_weight": [2.0, 3.0]})

    hd.fill_response_from_df(df, truth_col="y", reco_col="x")

    assert hd.has_response is True
    assert hd.response_hist.shape == (2, 2)
    # response_hist[reco_bin, true_bin]
    expected = np.zeros((2, 2))
    expected[0, 1] = 2.0  # reco bin0, true bin1, weight 2.0
    expected[1, 0] = 3.0  # reco bin1, true bin0, weight 3.0
    assert hd.response_hist.tolist() == expected.tolist()
    assert hd.response_bins.tolist() == bins.tolist()
    # mc_hist/truth_hist untouched by a response fill
    assert hd.mc_hist.sum() == 0.0
    assert hd.truth_hist.sum() == 0.0


def test_fill_response_from_df_empty_df_still_marks_has_response():
    bins = np.array([0.0, 1.0, 2.0])
    hd = OverlayHistData(var_save_name="v", breakdown_type="fake", bins=bins, n_cat=2)
    empty = pd.DataFrame({"x": [], "y": [], "pot_weight": []})

    hd.fill_response_from_df(empty, truth_col="y", reco_col="x")

    assert hd.has_response is True
    # response_hist stays None for a fully-empty first fill (never allocated)
    assert hd.response_hist is None


def test_fill_response_from_df_respects_bins_override():
    bins = np.array([0.0, 1.0, 2.0])
    override_bins = np.array([0.0, 2.0])  # coarser: single bin covering the whole range
    hd = OverlayHistData(var_save_name="v", breakdown_type="fake", bins=bins, n_cat=2)
    df = pd.DataFrame({"x": [0.5, 1.5], "y": [0.2, 1.8], "pot_weight": [1.0, 1.0]})

    hd.fill_response_from_df(df, truth_col="y", reco_col="x", bins=override_bins)

    assert hd.response_hist.shape == (1, 1)
    assert hd.response_hist.tolist() == [[2.0]]
    assert hd.response_bins.tolist() == override_bins.tolist()


def test_overlay_hist_data_iadd_merges_response_hist():
    bins = np.array([0.0, 1.0, 2.0])
    a = OverlayHistData(var_save_name="v", breakdown_type="fake", bins=bins, n_cat=2)
    b = OverlayHistData(var_save_name="v", breakdown_type="fake", bins=bins, n_cat=2)
    a.fill_response_from_df(
        pd.DataFrame({"x": [0.5], "y": [0.5], "pot_weight": [1.0]}), "y", "x"
    )
    b.fill_response_from_df(
        pd.DataFrame({"x": [0.5], "y": [0.5], "pot_weight": [4.0]}), "y", "x"
    )

    a += b

    expected = np.zeros((2, 2))
    expected[0, 0] = 5.0
    assert a.response_hist.tolist() == expected.tolist()
    assert a.has_response is True


def test_overlay_hist_data_iadd_merges_response_hist_when_only_one_side_filled():
    """One chunk had events for this variable's response matrix, the other didn't (e.g. an
    empty shard) -- merging a None response_hist into an already-filled one must not crash or
    wipe the existing data.
    """
    bins = np.array([0.0, 1.0, 2.0])
    a = OverlayHistData(var_save_name="v", breakdown_type="fake", bins=bins, n_cat=2)
    a.fill_response_from_df(
        pd.DataFrame({"x": [0.5], "y": [0.5], "pot_weight": [1.0]}), "y", "x"
    )
    b = OverlayHistData(var_save_name="v", breakdown_type="fake", bins=bins, n_cat=2)
    # b never had fill_response_from_df called -- response_hist stays None, has_response False

    a += b

    assert a.response_hist[0, 0] == 1.0
    assert a.has_response is True


def test_overlay_hist_data_iadd_tolerates_legacy_object_without_response_hist():
    """Merging in an object that predates response_hist (e.g. an old unpickled instance)
    must not crash -- __iadd__ falls back to None/False via getattr.
    """
    shared_bins = np.array([0.0, 1.0, 2.0])
    a = OverlayHistData(var_save_name="v", breakdown_type="fake", bins=shared_bins, n_cat=2)
    a.fill_response_from_df(
        pd.DataFrame({"x": [0.5], "y": [0.5], "pot_weight": [1.0]}), "y", "x"
    )

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
        truth_hist = np.zeros(2)
        truth_err2 = np.zeros(2)
        mc_univ_hist = None
        has_mc = has_intime = has_offbeam = has_dirt = has_data = has_truth = False
        # no response_hist / response_bins / has_response attributes at all

    a += _LegacyStandIn()
    assert a.response_hist[0, 0] == 1.0  # unchanged, not wiped
    assert a.has_response is True


# --------------------------------------------------------------------------
# ChunkRunner wiring: only fills response_hist for response_matrix=True, mc sample only
# --------------------------------------------------------------------------
class _ResponseMatrixVarConfig:
    var_save_name = "x"
    var_nu_col = "x"
    var_evt_reco_col = "x"
    var_evt_truth_col = "y"
    bins = np.array([0.0, 1.0, 2.0])
    response_matrix = True
    response_matrix_bins = None


class _DefaultVarConfig:
    var_save_name = "x"
    var_nu_col = "x"
    var_evt_reco_col = "x"
    var_evt_truth_col = "y"
    bins = np.array([0.0, 1.0, 2.0])
    response_matrix = False


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


def test_chunk_runner_fills_response_hist_when_response_matrix_true():
    runner = _make_runner(_ResponseMatrixVarConfig(), sample="mc")
    evt = pd.DataFrame({"x": [0.5, 1.5], "y": [1.5, 0.5], "pot_weight": [1.0, 1.0]})
    runner.run({"evt": evt})

    hd = next(iter(runner.histdata.values()))
    assert hd.has_response is True
    # reco=x bin0/true=y bin1 for event0; reco=x bin1/true=y bin0 for event1.
    expected = np.zeros((2, 2))
    expected[0, 1] = 1.0
    expected[1, 0] = 1.0
    assert hd.response_hist.tolist() == expected.tolist()


def test_chunk_runner_does_not_fill_response_hist_when_unset():
    """Default (response_matrix=False) variables must not pay for a 2D fill."""
    runner = _make_runner(_DefaultVarConfig(), sample="mc")
    evt = pd.DataFrame({"x": [0.5, 1.5], "y": [1.5, 0.5], "pot_weight": [1.0, 1.0]})
    runner.run({"evt": evt})

    hd = next(iter(runner.histdata.values()))
    assert hd.mc_hist.sum() == 2.0  # reco side still filled normally
    assert hd.has_response is False
    assert hd.response_hist is None


def test_chunk_runner_does_not_fill_response_hist_for_non_mc_samples():
    """response_matrix only makes sense for MC (no truth for data/intime/etc.)."""
    runner = _make_runner(_ResponseMatrixVarConfig(), sample="data")
    evt = pd.DataFrame({"x": [0.5, 1.5], "y": [1.5, 0.5]})
    runner.run({"evt": evt})

    hd = next(iter(runner.histdata.values()))
    assert hd.data_hist.sum() == 2.0
    assert hd.has_response is False
    assert hd.response_hist is None
