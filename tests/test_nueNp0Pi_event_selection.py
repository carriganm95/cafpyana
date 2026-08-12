"""analysis_village/nueNp0Pi/event_selection.py: the nueNp0Pi-flavored
ChunkRunner subclass that defaults to this analysis's BREAKDOWN_REGISTRY
(analysis_village.nueNp0Pi.config.stages) / SIGNAL_MASK_FN
(analysis_village.nueNp0Pi.selections) so pre-existing call sites keep
working after the generic chunk-processing machinery moved to
pyanalib.chunked_selection, and the map-phase core was consolidated here
(formerly split across event_selection_batch_core.py and
selection_framework.py). The pipeline definition itself
(build_pipeline/EFFICIENCY_VARS/BREAKDOWN_REGISTRY) now lives in
analysis_village.nueNp0Pi.config.stages.
"""
import pandas as pd
import pytest

from analysis_village.nueNp0Pi.selections import SIGNAL_MASK_FN
from analysis_village.nueNp0Pi.config.stages import build_pipeline, BREAKDOWN_REGISTRY
from analysis_village.nueNp0Pi.event_selection import ChunkRunner, build_runner
from pyanalib.chunked_selection import multicol_get_series


def test_breakdown_registry_has_expected_keys():
    assert set(BREAKDOWN_REGISTRY.keys()) == {"topology", "genie", "genie_sb", "pdg"}
    for n_cat, get_cuts_fn in BREAKDOWN_REGISTRY.values():
        assert isinstance(n_cat, int) and n_cat > 0
        assert callable(get_cuts_fn)


def test_old_call_signature_still_works_and_uses_defaults():
    # This is the exact call signature that existed before the refactor
    # (sample, stages, efficiency_vars, mc_univ_syst_tags=...) with no
    # breakdown_registry/signal_mask_fn -- any pre-existing driver script
    # calling it this way must keep working.
    runner = ChunkRunner(sample="mc", stages=[], efficiency_vars=[], mc_univ_syst_tags=None)
    assert runner.breakdown_registry is BREAKDOWN_REGISTRY
    assert runner.signal_mask_fn is SIGNAL_MASK_FN
    # NOT "pdg": bar_breakdown_types feeds ChunkRunner._fill_breakdown, which
    # always operates on the event-level df (state["evt"]). get_pdg_category_spine
    # needs a per-track df (df.rec.dlp_true.particles.pdg_code only exists on
    # the trk-level table, confirmed against real data) -- see
    # tests/test_nueNp0Pi_categories.py and config/stages.py's
    # sel_primary_trks / PlotSpec(breakdown_type="pdg", ...) wiring in the
    # "no_photons" stage for how pdg breakdown is actually meant to work.
    assert runner.bar_breakdown_types == ("topology", "genie")


def test_explicit_override_is_respected():
    fake_registry = {"fake": (1, lambda df, ret_cuts=True: [df.index >= 0])}
    runner = ChunkRunner(
        sample="mc",
        stages=[],
        efficiency_vars=[],
        breakdown_registry=fake_registry,
        signal_mask_fn=lambda df: df.index >= 0,
    )
    assert runner.breakdown_registry is fake_registry
    assert runner.breakdown_registry is not BREAKDOWN_REGISTRY


def test_multicol_get_series_from_pyanalib():
    df = pd.DataFrame({"x": [1, 2, 3]})
    assert list(multicol_get_series(df, "x")) == [1, 2, 3]


def test_build_pipeline_and_build_runner_still_wire_up():
    # Regression guard for the event_selection_pipeline_def.py +
    # selection_framework.py -> event_selection.py consolidation: build_runner
    # must still return a ChunkRunner defaulting to this analysis's
    # BREAKDOWN_REGISTRY/SIGNAL_MASK_FN, built from build_pipeline()'s stages.
    stages = build_pipeline()
    assert len(stages) > 0
    runner = build_runner("mc")
    assert isinstance(runner, ChunkRunner)
    assert runner.breakdown_registry is BREAKDOWN_REGISTRY
    assert runner.signal_mask_fn is SIGNAL_MASK_FN
