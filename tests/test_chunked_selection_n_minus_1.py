"""Tests for the "N-1" selection mechanism added to pyanalib.chunked_selection:
Stage.mask_fn, mask_from_filter, and n_minus_1_selector.

Uses fake, tiny cut/breakdown functions instead of any real physics -- this
mechanism is analysis-agnostic, so tests should prove that (same convention as
test_chunked_selection.py / test_chunked_selection_ratio_truth.py).
"""
import numpy as np
import pandas as pd
import pytest

from pyanalib.chunked_selection import (
    ChunkRunner,
    PlotSpec,
    Stage,
    mask_from_filter,
    n_minus_1_selector,
)


def two_category_cuts(df, ret_cuts=True):
    return [df["x"] < 1.5, df["x"] >= 1.5]


FAKE_REGISTRY = {"fake": (2, two_category_cuts)}


def _apply_to_evt(fn):
    """Local stand-in for config/stages.py's ``_apply_to_evt`` helper."""
    def _cut(state, sample):
        if state.get("evt") is not None:
            state["evt"] = fn(state["evt"])
        return state
    return _cut


def cut_a(df):
    """Keep rows where a == 1."""
    return df[df["a"] == 1]


def cut_b(df):
    """Keep rows where b == 1."""
    return df[df["b"] == 1]


class _FakeVarConfig:
    var_save_name = "x"
    var_evt_reco_col = "x"
    bins = np.array([0.0, 1.0, 2.0, 3.0])


# --------------------------------------------------------------------------
# mask_from_filter
# --------------------------------------------------------------------------
def test_mask_from_filter_matches_the_filter_it_wraps():
    df = pd.DataFrame({"a": [1, 0, 1, 1]}, index=[10, 11, 12, 13])
    mask = mask_from_filter(cut_a)(df)
    assert list(mask) == [True, False, True, True]


def test_mask_from_filter_empty_df_returns_empty_mask():
    df = pd.DataFrame({"a": []})
    mask = mask_from_filter(cut_a)(df)
    assert len(mask) == 0


def test_mask_from_filter_none_df_returns_empty_mask():
    mask = mask_from_filter(cut_a)(None)
    assert len(mask) == 0


# --------------------------------------------------------------------------
# ChunkRunner: precomputing _n1_base_evt / _n1_masks
# --------------------------------------------------------------------------
def _build_two_cut_runner(plots_on_stage_b):
    stage0 = Stage(key="allreco", label="All", cut=None, plots=[])
    stage_a = Stage(
        key="stage_a", label="A", cut=_apply_to_evt(cut_a), mask_fn=mask_from_filter(cut_a), plots=[]
    )
    stage_b = Stage(
        key="stage_b", label="B", cut=_apply_to_evt(cut_b), mask_fn=mask_from_filter(cut_b),
        plots=plots_on_stage_b,
    )
    return ChunkRunner(
        sample="mc",
        stages=[stage0, stage_a, stage_b],
        efficiency_vars=[],
        breakdown_registry=FAKE_REGISTRY,
        signal_mask_fn=lambda df: df["x"] > 1.0,
    )


def test_no_mask_fn_anywhere_means_no_n1_state_populated():
    """A pipeline with zero Stage.mask_fn set should pay no cost / add no state keys."""
    stage0 = Stage(key="s0", label="S0", cut=None, plots=[])
    runner = ChunkRunner(
        sample="mc", stages=[stage0], efficiency_vars=[],
        breakdown_registry=FAKE_REGISTRY, signal_mask_fn=lambda df: df["x"] > 1.0,
    )
    evt = pd.DataFrame({"x": [0.5], "pot_weight": [1.0]})
    # Run doesn't crash and doesn't require any plots to exercise state population.
    runner.run({"evt": evt})
    assert runner.histdata == {}


def test_n_minus_1_selector_holds_out_exactly_one_cut():
    """4 events: a=[1,1,0,1], b=[1,0,1,1], x=[0.5,1.5,2.5,0.2].

    Held out stage_a -> only cut_b enforced -> rows where b==1: x = 0.5, 2.5, 0.2 (3 events).
    Held out stage_b -> only cut_a enforced -> rows where a==1: x = 0.5, 1.5, 0.2 (3 events).
    """
    plot_a = PlotSpec(
        var_config=_FakeVarConfig(), breakdown_type="fake",
        selector=n_minus_1_selector("stage_a"), name_suffix="n_minus_1_stage_a",
    )
    plot_b = PlotSpec(
        var_config=_FakeVarConfig(), breakdown_type="fake",
        selector=n_minus_1_selector("stage_b"), name_suffix="n_minus_1_stage_b",
    )
    runner = _build_two_cut_runner(plots_on_stage_b=[plot_a, plot_b])
    evt = pd.DataFrame({
        "x": [0.5, 1.5, 2.5, 0.2],
        "a": [1, 1, 0, 1],
        "b": [1, 0, 1, 1],
        "pot_weight": [1.0, 1.0, 1.0, 1.0],
    })
    runner.run({"evt": evt})

    hd_a = runner.histdata[("stage_b", ChunkRunner.plot_key("stage_b", plot_a))]
    hd_b = runner.histdata[("stage_b", ChunkRunner.plot_key("stage_b", plot_b))]
    assert hd_a.mc_hist.sum() == 3.0
    assert hd_b.mc_hist.sum() == 3.0
    # sanity: the plain sequential cut (a AND b) would leave only 2 events (x=0.5, 0.2) --
    # confirm N-1 genuinely relaxes one cut rather than silently applying both.
    both_cuts_survivors = evt[(evt["a"] == 1) & (evt["b"] == 1)]
    assert len(both_cuts_survivors) == 2


def test_n_minus_1_selector_excludes_stage_not_in_cut_stage_keys():
    """A stage with mask_fn set but omitted from cut_stage_keys is never enforced,
    even when it isn't the held-out key -- e.g. a stage the analysis doesn't want
    to gate N-1 plots on at all."""
    plot = PlotSpec(
        var_config=_FakeVarConfig(), breakdown_type="fake",
        selector=n_minus_1_selector("stage_a", cut_stage_keys=["stage_a"]),
        name_suffix="n_minus_1_stage_a",
    )
    runner = _build_two_cut_runner(plots_on_stage_b=[plot])
    evt = pd.DataFrame({
        "x": [0.5, 1.5, 2.5, 0.2],
        "a": [1, 1, 0, 1],
        "b": [1, 0, 1, 1],
        "pot_weight": [1.0, 1.0, 1.0, 1.0],
    })
    runner.run({"evt": evt})
    hd = next(iter(runner.histdata.values()))
    # cut_stage_keys=["stage_a"] and stage_a is held out -> nothing enforced -> all 4 events.
    assert hd.mc_hist.sum() == 4.0


def test_n_minus_1_selector_returns_none_without_precomputed_masks():
    """If no Stage in the pipeline sets mask_fn, ChunkRunner never populates
    state['_n1_masks'], so the selector must return None (skip), not crash."""
    plot = PlotSpec(
        var_config=_FakeVarConfig(), breakdown_type="fake",
        selector=n_minus_1_selector("stage_a"), name_suffix="n_minus_1_stage_a",
    )
    stage0 = Stage(key="s0", label="S0", cut=None, plots=[plot])
    runner = ChunkRunner(
        sample="mc", stages=[stage0], efficiency_vars=[],
        breakdown_registry=FAKE_REGISTRY, signal_mask_fn=lambda df: df["x"] > 1.0,
    )
    evt = pd.DataFrame({"x": [0.5], "pot_weight": [1.0]})
    runner.run({"evt": evt})
    assert runner.histdata == {}  # selector returned None -> _fill_plot skipped


def test_n_minus_1_selector_returns_none_when_nothing_survives():
    plot = PlotSpec(
        var_config=_FakeVarConfig(), breakdown_type="fake",
        selector=n_minus_1_selector("stage_a"), name_suffix="n_minus_1_stage_a",
    )
    runner = _build_two_cut_runner(plots_on_stage_b=[plot])
    # No row has b == 1 -> holding out stage_a still requires cut_b -> zero survivors.
    evt = pd.DataFrame({
        "x": [0.5, 1.5], "a": [1, 1], "b": [0, 0], "pot_weight": [1.0, 1.0],
    })
    runner.run({"evt": evt})
    assert runner.histdata == {}


def test_baseline_stage_never_held_out_stays_enforced_in_every_n1_plot():
    """A stage that has mask_fn but is never passed as held_out_key (e.g. a baseline
    quality cut) should still gate every other cut's N-1 selection."""
    plot_a = PlotSpec(
        var_config=_FakeVarConfig(), breakdown_type="fake",
        selector=n_minus_1_selector("stage_a"), name_suffix="n_minus_1_stage_a",
    )
    runner = _build_two_cut_runner(plots_on_stage_b=[plot_a])
    evt = pd.DataFrame({
        "x": [0.5, 1.5, 2.5, 0.2],
        "a": [1, 1, 0, 1],
        "b": [1, 0, 1, 1],
        "pot_weight": [1.0, 1.0, 1.0, 1.0],
    })
    runner.run({"evt": evt})
    hd = next(iter(runner.histdata.values()))
    # stage_a held out -> stage_b (never held out here) still enforced -> b==1 -> 3 events.
    assert hd.mc_hist.sum() == 3.0
