"""Tests for config/stages.py's N_MINUS_1_* wiring: setting the module-level knobs
and calling build_pipeline() should attach the expected N-1 PlotSpecs (see
pyanalib.chunked_selection.n_minus_1_selector / mask_from_filter for the underlying
mechanism, tested independently -- with fake, non-physics cuts -- in
tests/test_chunked_selection_n_minus_1.py).

NOTE: like tests/test_nueNp0Pi_event_selection.py, this imports build_pipeline
directly, which pulls in the real makedf/uproot/SPINE-schema-adjacent dependency
chain -- run this in the full cafpyana environment, not a minimal sandbox.
"""
import pytest

from analysis_village.nueNp0Pi.config.plots import VariableConfig
import analysis_village.nueNp0Pi.config.stages as stages_mod
from analysis_village.nueNp0Pi.config.stages import build_pipeline
from pyanalib.chunked_selection import PlotSpec


@pytest.fixture(autouse=True)
def _reset_n_minus_1_knobs():
    """N_MINUS_1_* are plain module attributes (same notebook-configurable pattern as
    EVT_BREAKDOWN_*) -- save/restore them so tests don't leak state into each other or
    into a real notebook session that later imports this module.
    """
    saved = {
        "N_MINUS_1_STAGE_KEYS": stages_mod.N_MINUS_1_STAGE_KEYS,
        "N_MINUS_1_VARS": stages_mod.N_MINUS_1_VARS,
        "N_MINUS_1_DEFAULT_VARS": stages_mod.N_MINUS_1_DEFAULT_VARS,
        "N_MINUS_1_BREAKDOWN_TYPE": stages_mod.N_MINUS_1_BREAKDOWN_TYPE,
        "N_MINUS_1_DIR_NAME": stages_mod.N_MINUS_1_DIR_NAME,
        "N_MINUS_1_VLINES": stages_mod.N_MINUS_1_VLINES,
    }
    yield
    for k, v in saved.items():
        setattr(stages_mod, k, v)


def _n1_plots(stages):
    """PlotSpecs on the final stage whose name_suffix marks them as N-1 plots."""
    return [ps for ps in stages[-1].plots if (ps.name_suffix or "").startswith("n_minus_1_")]


def test_default_n_minus_1_stage_keys_is_empty_and_adds_no_plots():
    """Off by default -- no N_MINUS_1_STAGE_KEYS populated -> build_pipeline() attaches
    nothing extra (matches the module docstring: opt-in, unlike EVT_BREAKDOWN_VARS)."""
    assert stages_mod.N_MINUS_1_STAGE_KEYS == ()
    stages = build_pipeline()
    assert _n1_plots(stages) == []


def test_every_reco_cut_stage_has_a_mask_fn():
    """Every real cut stage (built via _evt_cut) should be N-1-eligible; only the
    cut-less 'allreco' stage should not."""
    stages = build_pipeline()
    for s in stages:
        if s.key == "allreco":
            assert s.mask_fn is None
        elif s.cut is not None:
            assert s.mask_fn is not None, f"stage {s.key!r} has a cut but no mask_fn"


def test_populating_stage_keys_and_vars_appends_expected_plotspecs():
    stages_mod.N_MINUS_1_STAGE_KEYS = ["electron_softmax", "electron_dedx"]
    stages_mod.N_MINUS_1_VARS = {
        "electron_softmax": [VariableConfig.electron_softmax_score()],
        "electron_dedx": [VariableConfig.electron_dedx()],
    }

    stages = build_pipeline()
    n1_plots = _n1_plots(stages)
    assert len(n1_plots) == 2

    by_suffix = {ps.name_suffix: ps for ps in n1_plots}
    assert set(by_suffix) == {"n_minus_1_electron_softmax", "n_minus_1_electron_dedx"}

    ps_softmax = by_suffix["n_minus_1_electron_softmax"]
    assert ps_softmax.var_config.var_save_name == "electron-softmax-score"
    assert ps_softmax.breakdown_type == stages_mod.N_MINUS_1_BREAKDOWN_TYPE
    # Pre-filled default vline for this cut (see N_MINUS_1_VLINES) should be threaded
    # through to save_kwargs so overlay_hists_from_histdata draws the cut-threshold line.
    assert ps_softmax.save_kwargs["vline"] == stages_mod.N_MINUS_1_VLINES["electron_softmax"]

    ps_dedx = by_suffix["n_minus_1_electron_dedx"]
    assert ps_dedx.var_config.var_save_name == "electron-dedx"
    assert ps_dedx.save_kwargs["vline"] == stages_mod.N_MINUS_1_VLINES["electron_dedx"]


def test_held_out_key_actually_excludes_that_cut_from_its_own_n1_selector():
    """The PlotSpec.selector for a held-out cut must not include that cut's own key in
    the AND -- spot check by calling n_minus_1_selector's returned closure directly
    against a fake state, without needing real event data."""
    stages_mod.N_MINUS_1_STAGE_KEYS = ["electron_softmax"]
    stages_mod.N_MINUS_1_VARS = {"electron_softmax": [VariableConfig.electron_softmax_score()]}

    stages = build_pipeline()
    ps = _n1_plots(stages)[0]

    import pandas as pd
    import numpy as np
    base = pd.DataFrame({"dummy": [1, 2, 3]})
    # Fake masks: electron_softmax mask would reject everything if it were (wrongly)
    # applied; some other eligible cut's mask keeps everything.
    fake_masks = {
        "electron_softmax": np.array([False, False, False]),
        "is_fiducial": np.array([True, True, True]),
    }
    state = {"_n1_base_evt": base, "_n1_masks": fake_masks}
    result = ps.selector(state, "mc")
    # electron_softmax's own (all-False) mask must be skipped since it's the held-out key.
    assert result is not None
    assert len(result) == 3


def test_var_missing_from_n_minus_1_vars_falls_back_to_default_vars():
    stages_mod.N_MINUS_1_STAGE_KEYS = ["good_electron"]
    stages_mod.N_MINUS_1_VARS = {}  # no entry for "good_electron"
    stages_mod.N_MINUS_1_DEFAULT_VARS = [VariableConfig.neutrino_energy()]

    stages = build_pipeline()
    n1_plots = _n1_plots(stages)
    assert len(n1_plots) == 1
    assert n1_plots[0].name_suffix == "n_minus_1_good_electron"
    # good_electron has no entry in N_MINUS_1_VLINES -> no vline kwarg forced on it.
    assert "vline" not in n1_plots[0].save_kwargs


def test_unknown_stage_key_raises():
    stages_mod.N_MINUS_1_STAGE_KEYS = ["not_a_real_stage"]
    with pytest.raises(ValueError):
        build_pipeline()


def test_stage_key_without_mask_fn_raises_with_helpful_message():
    """Guards against silently no-op'ing if someone points N_MINUS_1_STAGE_KEYS at a
    stage that was built with plain cut=_apply_to_evt(...) instead of **_evt_cut(...)."""
    # "allreco" is a real stage key but has cut=None / mask_fn=None.
    stages_mod.N_MINUS_1_STAGE_KEYS = ["allreco"]
    with pytest.raises(ValueError, match="mask_fn"):
        build_pipeline()
