"""The pi0 sideband is defined twice: in the df maker
(makedf/make_nueccdf.py:_apply_pi0_sideband_cuts) and as the "pi0_sideband" stages of the
event-selection pipeline (config/stages.py:build_pipeline). These tests run both on the
same synthetic slices and require them to keep exactly the same ones, and check the signal
selection still applies every cut, so the two definitions can't drift apart.
"""
import numpy as np
import pandas as pd
import pytest

import analysis_village.nueNp0Pi.config.stages as stages_mod
from analysis_village.nueNp0Pi.config.stages import build_pipeline

make_nueccdf = pytest.importorskip("analysis_village.nueNp0Pi.makedf.make_nueccdf")

# Every column any signal or sideband cut reads, with a value range that straddles its
# threshold so each cut removes some slices.
_BOOL_COLS = ("is_fiducial", "sbnd_fiducial_reco", "is_flash_matched", "is_contained",
              "muon_mask_reco", "pion_mask_reco", "photon_mask_reco", "ele_mask_reco",
              "proton_mask_reco")
_FLOAT_COLS = {
    "ele_softmax_reco": (0.0, 1.0),
    "ele_primary_reco": (0.9, 1.0),
    "proton_softmax_reco": (0.0, 1.0),
    "ele_vertex_distance_reco": (0.0, 10.0),
    "ele_dedx_reco": (0.0, 8.0),
    "photon_energy_reco": (0.0, 500.0),
    "subprim_photon_energy_reco": (0.0, 100.0),
}


def _slices(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    data = {}
    for c in _BOOL_COLS:
        # mostly passing so plenty of slices survive the whole chain
        p_true = 0.1 if c in ("muon_mask_reco", "pion_mask_reco", "photon_mask_reco") else 0.9
        data[("rec", "dlp", c)] = rng.random(n) < p_true
    for c, (lo, hi) in _FLOAT_COLS.items():
        data[("rec", "dlp", c)] = rng.uniform(lo, hi, n)
    df = pd.DataFrame(data)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def _run_stage_cuts(mode, df):
    prev = stages_mod.SELECTION_MODE
    try:
        stages_mod.set_selection_mode(mode)
        stages = build_pipeline()
    finally:
        stages_mod.set_selection_mode(prev)
    state = {"evt": df, "trk": None, "hdr": None, "mcnu": None}
    for st in stages:
        if st.cut is not None:
            state = st.cut(state, sample="mc")
    return state["evt"], [s.key for s in stages]


def test_pipeline_sideband_matches_df_maker_sideband():
    df = _slices()
    pipeline, _ = _run_stage_cuts("pi0_sideband", df)
    dfmaker = make_nueccdf._apply_pi0_sideband_cuts(df)
    assert len(dfmaker) > 0
    assert pipeline.index.equals(dfmaker.index), (
        f"pipeline keeps {len(pipeline)} slices, df maker keeps {len(dfmaker)} -- update "
        "config/stages.py:build_pipeline or make_nueccdf._apply_pi0_sideband_cuts so they match")


def test_sideband_and_signal_stage_lists():
    _, sideband_keys = _run_stage_cuts("pi0_sideband", _slices(10))
    _, signal_keys = _run_stage_cuts("signal", _slices(10))
    removed = {"no_photons", "electron_softmax", "electron_primary", "proton_softmax",
               "electron_vertex_distance", "electron_dedx"}
    assert removed <= set(signal_keys)
    assert not removed & set(sideband_keys)
    assert {"pi0_lead_photon", "pi0_sub_photon"} <= set(sideband_keys)
    # every stage present in only one mode is listed, so N-1 keys for it are skipped
    # (not rejected as unknown) in the other mode
    only_one = set(signal_keys) ^ set(sideband_keys)
    assert only_one <= stages_mod.MODE_SPECIFIC_STAGE_KEYS


def test_default_n_minus_1_keys_build_in_both_modes():
    for mode in stages_mod.SELECTION_MODES:
        prev = stages_mod.SELECTION_MODE
        try:
            stages_mod.set_selection_mode(mode)
            build_pipeline()   # must not raise on signal-only default N-1 keys
        finally:
            stages_mod.set_selection_mode(prev)
