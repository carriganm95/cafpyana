"""pyanalib.chunked_selection: the generic chunked event-selection /
histogram-accumulation framework hoisted out of
analysis_village/nueNp0Pi/selection_framework.py.

These tests use a fake, tiny "breakdown registry" and "signal mask" instead
of any real physics category functions -- the whole point of this module is
that it doesn't know or care what those are, so tests should prove that.
"""
import pickle

import numpy as np
import pandas as pd
import pytest

from pyanalib.chunked_selection import (
    BarBreakdown,
    ChunkRunner,
    EfficiencyAccumulator,
    ExposureTotals,
    OverlayHistData,
    PlotSpec,
    Stage,
    aggregate_chunk_files,
    apply_global_exposure_scales,
    get_clipped_evts,
    mc_univ_weight_matrix,
    merge_samples,
    multicol_get_series,
    multicol_resolve_column_key,
    sanitize_merged_histdata_finite,
)


def two_category_cuts(df, ret_cuts=True):
    """Fake breakdown: category 0 is x<1.5, category 1 is x>=1.5."""
    return [df["x"] < 1.5, df["x"] >= 1.5]


FAKE_REGISTRY = {"fake": (2, two_category_cuts)}


def fake_signal_mask(df):
    return df["x"] > 1.0


# --------------------------------------------------------------------------
# get_clipped_evts / multicol helpers
# --------------------------------------------------------------------------
def test_get_clipped_evts_clips_and_extracts_weight():
    df = pd.DataFrame({"x": [-5.0, 0.5, 1.5, 99.0], "pot_weight": [1.0, 2.0, 3.0, 4.0]})
    var, w = get_clipped_evts(df, "x", np.array([0.0, 1.0, 2.0]))
    assert var[0] == pytest.approx(0.0)       # clipped to lower edge
    assert var[-1] == pytest.approx(2.0 - 1e-6)  # clipped just inside upper edge
    assert list(w) == [1.0, 2.0, 3.0, 4.0]


def test_get_clipped_evts_defaults_weight_to_ones_when_missing():
    df = pd.DataFrame({"x": [0.1, 0.2]})
    var, w = get_clipped_evts(df, "x", np.array([0.0, 1.0]))
    assert list(w) == [1.0, 1.0]


def test_get_clipped_evts_sanitizes_nan_weights():
    df = pd.DataFrame({"x": [0.1, 0.2], "pot_weight": [1.0, np.nan]})
    _, w = get_clipped_evts(df, "x", np.array([0.0, 1.0]))
    assert list(w) == [1.0, 0.0]


def test_multicol_resolve_column_key_flat_index():
    df = pd.DataFrame({"x": [1, 2, 3]})
    assert multicol_resolve_column_key(df, "x") == "x"
    assert multicol_resolve_column_key(df, "missing") is None


def test_multicol_get_series_multiindex_depth_mismatch():
    cols = pd.MultiIndex.from_tuples([("mu", "pfp", "trk", "P", "p_muon", "", "")])
    df = pd.DataFrame([[1.0], [2.0]], columns=cols)
    # shorter tuple than the on-disk depth should still resolve via padding
    series = multicol_get_series(df, ("mu", "pfp", "trk", "P", "p_muon"))
    assert list(series) == [1.0, 2.0]


def test_multicol_get_series_raises_keyerror_when_unresolvable():
    df = pd.DataFrame({"x": [1, 2]})
    with pytest.raises(KeyError):
        multicol_get_series(df, "nope")


def test_mc_univ_weight_matrix_stacks_universes_and_sanitizes():
    cols = pd.MultiIndex.from_tuples([
        ("mc", "flux", "univ_0"), ("mc", "flux", "univ_1"),
    ])
    df = pd.DataFrame([[1.0, np.nan], [2.0, np.inf]], columns=cols)
    mat = mc_univ_weight_matrix(df, "flux")
    assert mat.shape == (2, 2)
    assert mat[1, 0] == pytest.approx(1.0)  # NaN -> 1.0
    assert mat[1, 1] == pytest.approx(10.0)  # +inf clipped to 10.0


def test_mc_univ_weight_matrix_returns_none_for_empty_df():
    df = pd.DataFrame({"x": []})
    assert mc_univ_weight_matrix(df, "flux") is None


# --------------------------------------------------------------------------
# OverlayHistData
# --------------------------------------------------------------------------
def test_overlay_hist_data_requires_get_cuts_fn_for_mc():
    bins = np.array([0.0, 1.0, 2.0])
    hd = OverlayHistData(var_save_name="v", breakdown_type="fake", bins=bins, n_cat=2)
    df = pd.DataFrame({"x": [0.5], "pot_weight": [1.0]})
    with pytest.raises(ValueError):
        hd.fill_from_df(df, "x", "mc")  # no get_cuts_fn


def test_overlay_hist_data_fills_mc_by_category():
    bins = np.array([0.0, 1.0, 2.0, 3.0])
    hd = OverlayHistData(var_save_name="v", breakdown_type="fake", bins=bins, n_cat=2)
    df = pd.DataFrame({"x": [0.5, 1.5, 2.5, 0.2], "pot_weight": [1.0, 1.0, 1.0, 1.0]})
    hd.fill_from_df(df, "x", "mc", get_cuts_fn=two_category_cuts)
    assert hd.has_mc is True
    assert hd.mc_hist.sum() == 4
    # category 0 (x<1.5) gets the two events in bin [0,1): 0.5 and 0.2
    assert hd.mc_hist[0, 0] == 2


def test_overlay_hist_data_fills_data_sample():
    bins = np.array([0.0, 1.0, 2.0])
    hd = OverlayHistData(var_save_name="v", breakdown_type="fake", bins=bins, n_cat=2)
    df = pd.DataFrame({"x": [0.1, 1.9]})  # no pot_weight -> data convention
    hd.fill_from_df(df, "x", "data")
    assert hd.has_data is True
    assert hd.data_hist.sum() == 2


def test_overlay_hist_data_iadd_merges_histograms():
    bins = np.array([0.0, 1.0, 2.0])
    a = OverlayHistData(var_save_name="v", breakdown_type="fake", bins=bins, n_cat=2)
    b = OverlayHistData(var_save_name="v", breakdown_type="fake", bins=bins, n_cat=2)
    df = pd.DataFrame({"x": [0.5], "pot_weight": [2.0]})
    a.fill_from_df(df, "x", "mc", get_cuts_fn=two_category_cuts)
    b.fill_from_df(df, "x", "mc", get_cuts_fn=two_category_cuts)
    a += b
    assert a.mc_hist.sum() == 4.0  # 2.0 + 2.0


def test_overlay_hist_data_iadd_rejects_mismatched_bins():
    a = OverlayHistData(var_save_name="v", breakdown_type="fake", bins=np.array([0., 1.]), n_cat=2)
    b = OverlayHistData(var_save_name="v", breakdown_type="fake", bins=np.array([0., 2.]), n_cat=2)
    with pytest.raises(AssertionError):
        a += b


# --------------------------------------------------------------------------
# BarBreakdown
# --------------------------------------------------------------------------
def test_bar_breakdown_fill_mc_and_total_count():
    bb = BarBreakdown.empty("fake", n_cat=2)
    df = pd.DataFrame({"x": [0.5, 1.5], "pot_weight": [1.0, 3.0]})
    bb.fill_mc(df, two_category_cuts)
    assert list(bb.mc_counts) == [1.0, 3.0]
    assert bb.total_count("mc") == 4.0


def test_bar_breakdown_total_count_unknown_sample_raises():
    bb = BarBreakdown.empty("fake", n_cat=2)
    with pytest.raises(ValueError):
        bb.total_count("nonsense")


# --------------------------------------------------------------------------
# EfficiencyAccumulator
# --------------------------------------------------------------------------
class _FakeVarConfig:
    var_save_name = "x"
    var_nu_col = "x"
    var_evt_reco_col = "x"
    var_evt_truth_col = "x"
    bins = np.array([0.0, 1.0, 2.0])


def test_efficiency_accumulator_numerator_and_scale():
    ea = EfficiencyAccumulator.empty(_FakeVarConfig())
    evt = pd.DataFrame({"x": [0.5, 1.5], "pot_weight": [1.0, 1.0]})
    mask = fake_signal_mask(evt)  # only x=1.5 passes (x>1.0)
    ea.fill_numerator_from_evt(evt, _FakeVarConfig(), mask)
    assert ea.n_signal_pot.sum() == 1.0
    ea.scale_pot_components(2.0)
    assert ea.n_signal_pot.sum() == 2.0
    # raw counts are untouched by POT scaling
    assert ea.n_signal_raw.sum() == 1.0


# --------------------------------------------------------------------------
# ChunkRunner end-to-end over a tiny 1-stage pipeline
# --------------------------------------------------------------------------
def _make_runner(sample="mc"):
    plot = PlotSpec(
        var_config=_FakeVarConfig(),
        breakdown_type="fake",
        selector=lambda state, sample: state.get("evt"),
    )
    stage = Stage(key="s0", label="Stage 0", plots=[plot], save_for_breakdown=True)
    return ChunkRunner(
        sample=sample,
        stages=[stage],
        efficiency_vars=[],
        breakdown_registry=FAKE_REGISTRY,
        signal_mask_fn=fake_signal_mask,
        bar_breakdown_types=("fake",),
    )


def test_chunk_runner_run_fills_histdata_and_bar():
    runner = _make_runner("mc")
    evt = pd.DataFrame({"x": [0.5, 1.5], "pot_weight": [1.0, 1.0]})
    runner.run({"evt": evt})
    assert len(runner.histdata) == 1
    hd = next(iter(runner.histdata.values()))
    assert hd.mc_hist.sum() == 2
    assert "s0" in runner.bar
    assert runner.bar["s0"]["fake"].mc_counts.sum() == 2


def test_chunk_runner_rejects_unknown_sample():
    with pytest.raises(ValueError):
        ChunkRunner(
            sample="bogus",
            stages=[],
            efficiency_vars=[],
            breakdown_registry=FAKE_REGISTRY,
            signal_mask_fn=fake_signal_mask,
        )


def test_chunk_runner_save_and_aggregate_roundtrip(tmp_path):
    runner = _make_runner("mc")
    evt = pd.DataFrame({"x": [0.5, 1.5], "pot_weight": [1.0, 1.0]})
    runner.run({"evt": evt})
    out_path = str(tmp_path / "chunk0.pkl")
    runner.save(out_path)

    agg = aggregate_chunk_files([out_path, out_path])  # sum the same chunk twice
    hd = next(iter(agg["histdata"].values()))
    assert hd.mc_hist.sum() == 4  # doubled


def test_aggregate_chunk_files_empty_list_raises():
    with pytest.raises(ValueError):
        aggregate_chunk_files([])


def test_merge_samples_infers_n_cat_from_seed():
    bins = np.array([0.0, 1.0, 2.0])
    hd = OverlayHistData(var_save_name="v", breakdown_type="fake", bins=bins, n_cat=2)
    df = pd.DataFrame({"x": [0.5], "pot_weight": [1.0]})
    hd.fill_from_df(df, "x", "mc", get_cuts_fn=two_category_cuts)
    samples = {
        "mc": {
            "sample": "mc", "stage_keys": ["s0"], "stage_labels": ["S0"],
            "histdata": {("s0", "k"): hd}, "bar": {}, "eff": {},
        }
    }
    merged = merge_samples(samples)
    merged_hd = merged["histdata"][("s0", "k")]
    assert merged_hd.mc_hist.shape[0] == 2
    assert merged_hd.mc_hist.sum() == 1


def test_sanitize_merged_histdata_finite_zeroes_nans():
    bins = np.array([0.0, 1.0, 2.0])
    hd = OverlayHistData(var_save_name="v", breakdown_type="fake", bins=bins, n_cat=2)
    hd.mc_hist[0, 0] = np.nan
    hd.mc_hist[0, 1] = np.inf
    merged = {"histdata": {("s", "k"): hd}, "bar": {}}
    n_hd, n_bar = sanitize_merged_histdata_finite(merged)
    assert n_hd == 1
    assert np.all(np.isfinite(hd.mc_hist))


def test_apply_global_exposure_scales_scales_mc_and_data_stays_raw():
    bins = np.array([0.0, 1.0, 2.0])
    hd = OverlayHistData(var_save_name="v", breakdown_type="fake", bins=bins, n_cat=2)
    df_mc = pd.DataFrame({"x": [0.5], "pot_weight": [1.0]})
    hd.fill_from_df(df_mc, "x", "mc", get_cuts_fn=two_category_cuts)
    df_data = pd.DataFrame({"x": [0.5]})
    hd.fill_from_df(df_data, "x", "data")

    merged = {"histdata": {("s", "k"): hd}, "bar": {}, "eff": {}}
    totals = ExposureTotals(data_pot=2.0, mc_pot=1.0)
    scales = apply_global_exposure_scales(merged, totals)
    assert scales["scale_mc"] == pytest.approx(2.0)
    assert hd.mc_hist.sum() == pytest.approx(2.0)   # scaled by data/mc POT ratio
    assert hd.data_hist.sum() == pytest.approx(1.0)  # data untouched
