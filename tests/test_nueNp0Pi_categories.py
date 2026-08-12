"""analysis_village/nueNp0Pi/categories.py: particle-type (pdg) breakdown
regression guards.

get_pdg_category_spine previously (a) had no ret_cuts parameter, so calling
it the same way BREAKDOWN_REGISTRY calls its topology/genie siblings
(get_cuts_fn(df, ret_cuts=True), see pyanalib/chunked_selection.py) raised
TypeError, and (b) reversed its returned cut list while pdg_labels was left
unreversed, silently mislabeling/miscoloring every category. Both are why
the particle-type breakdown plot didn't work while the genie-mode one did
(get_genie_category_spine has neither bug).
"""
import numpy as np
import pandas as pd
import pytest

from analysis_village.nueNp0Pi.selections import get_pdg_category_spine
from analysis_village.nueNp0Pi.config.settings import pdg_labels


def _fake_spine_df(pdg_codes):
    cols = pd.MultiIndex.from_tuples([("rec", "dlp_true", "particles", "pdg_code")])
    return pd.DataFrame([[c] for c in pdg_codes], columns=cols)


def test_accepts_ret_cuts_like_its_topology_and_genie_siblings():
    # This is the exact call signature pyanalib.chunked_selection.ChunkRunner
    # uses for every entry in BREAKDOWN_REGISTRY -- must not raise.
    df = _fake_spine_df([11, 22, 13, 2212, 211, 999])
    cuts = get_pdg_category_spine(df, ret_cuts=True)
    assert len(cuts) == len(pdg_labels)


def test_cuts_are_in_pdg_labels_order_not_reversed():
    # pdg_labels = [e, gamma, mu, p, pi, Other]. One event of each kind,
    # in that same order, must set exactly the diagonal entry true.
    df = _fake_spine_df([11, 22, 13, 2212, 211, 999])
    cuts = get_pdg_category_spine(df, ret_cuts=True)
    cuts = [np.asarray(c) for c in cuts]

    expected_true_index = [0, 1, 2, 3, 4, 5]  # e, gamma, mu, p, pi, other
    for row, expected_cat in enumerate(expected_true_index):
        hits = [i for i, c in enumerate(cuts) if c[row]]
        assert hits == [expected_cat], (
            f"row {row} (pdg={df.iloc[row, 0]}) matched category index(es) "
            f"{hits}, expected only {expected_cat} ({pdg_labels[expected_cat]})"
        )


def test_category_series_matches_cuts_when_ret_cuts_false():
    df = _fake_spine_df([11, 2212, 999])
    categ = get_pdg_category_spine(df, ret_cuts=False)
    assert list(categ) == [0, 3, 5]  # e -> 0, proton -> 3, other -> 5


def test_requires_track_level_df_not_event_level_df():
    # Confirmed against real .df data: df.rec.dlp_true.particles.pdg_code only
    # exists on the per-track table ("trk"), not the per-event table ("evt") --
    # the event table has no "particles" sub-level at all. get_pdg_category_spine
    # must therefore be driven off a PlotSpec(selector=sel_trks_concat, ...)-style
    # track-level dataframe, NOT wired into ChunkRunner.bar_breakdown_types
    # (whose _fill_breakdown always passes the event-level df -- see
    # test_nueNp0Pi_selection_framework.py's test_old_call_signature_...).
    evt_like_cols = pd.MultiIndex.from_tuples([("rec", "dlp_true", "lepton_pdg_code", "", "")])
    evt_like_df = pd.DataFrame([[11]], columns=evt_like_cols)
    with pytest.raises(AttributeError):
        get_pdg_category_spine(evt_like_df, ret_cuts=True)
