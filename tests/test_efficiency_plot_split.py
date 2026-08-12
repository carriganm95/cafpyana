"""render_efficiency_plots output-layout regression guard.

Originally ``render_efficiency_plots`` wrote a single combined dual-axis PNG
per variable to a flat directory (``<save_fig_dir>/efficiency-<var>.png``).
The user found the combined view too cluttered (event counts AND efficiency
error bars for every stage on one figure) and asked for three views per
variable -- combined (unchanged), event-counts-only, and efficiency-only.
These were first split into a per-variable subdirectory
(``<save_fig_dir>/<var_save_name>/{combined,event_counts,efficiency}/``),
then flipped the other way around on request: one directory per view
(``<save_fig_dir>/{combined,event_counts,efficiency}/``), with every
variable's plot as a file directly inside it.

Later, the whole output tree (across all render_* functions, not just this
one) was split again into a top-level ``image/``/``pkl/`` pair, mirroring the
same subdirectory structure under each -- so PNGs now live under
``<save_fig_dir>/image/{combined,event_counts,efficiency}/`` and each gets a
sibling pickled ``matplotlib.figure.Figure`` under
``<save_fig_dir>/pkl/{combined,event_counts,efficiency}/``.

These tests build a minimal synthetic ``merged`` payload (two stages, one
efficiency variable) and check the on-disk layout and the still-required
``eff_dict.pkl`` side output, without needing real HDF5 event data.
"""
import pickle

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from pyanalib.chunked_selection import EfficiencyAccumulator
from analysis_village.nueNp0Pi.config.stages import EFFICIENCY_VARS
from analysis_village.nueNp0Pi.event_selection_aggregate import render_efficiency_plots


def _make_merged_for_one_var(var_config, stage_keys, stage_labels):
    bins = var_config.bins
    n_bin = len(bins) - 1

    eff = {}
    for i, sk in enumerate(stage_keys):
        ea = EfficiencyAccumulator.empty(var_config)
        # Denominator only meaningful on the first stage.
        if i == 0:
            ea.n_truth_nu_pot[:] = 10.0
            ea.n_truth_nu_raw[:] = 10
            ea.n_total_truth_nu_int = 10.0 * n_bin
        ea.n_signal_pot[:] = 10.0 - i * 2.0
        ea.n_signal_raw[:] = 10 - i * 2
        ea.n_total_signal_int = float(np.sum(ea.n_signal_pot))
        eff[sk] = {var_config.var_save_name: ea}

    return {
        "stage_keys": stage_keys,
        "stage_labels": stage_labels,
        "eff": eff,
    }


@pytest.fixture
def one_var_merged():
    var_config = EFFICIENCY_VARS[0]
    stage_keys = ["allreco", "in_fv"]
    stage_labels = ["All reconstructed interactions", "In FV"]
    merged = _make_merged_for_one_var(var_config, stage_keys, stage_labels)
    return var_config, merged


def test_writes_three_view_dirs_with_variable_files_inside(tmp_path, one_var_merged):
    var_config, merged = one_var_merged
    out_dir = str(tmp_path)

    render_efficiency_plots(merged, out_dir, pot_str="1e20 POT", save_fig=True, show_fig=False)

    for subdir in ("combined", "event_counts", "efficiency"):
        f = tmp_path / "image" / subdir / f"efficiency-{var_config.var_save_name}.png"
        assert f.is_file(), f"missing {f}"

    # No per-variable subdirectory should exist anymore.
    assert not (tmp_path / "image" / var_config.var_save_name).exists()


def test_writes_matching_pkl_figures(tmp_path, one_var_merged):
    """Every PNG under image/ gets a sibling pickled Figure under pkl/, same
    relative path, same basename (just a .pkl extension).
    """
    var_config, merged = one_var_merged
    out_dir = str(tmp_path)

    render_efficiency_plots(merged, out_dir, pot_str="1e20 POT", save_fig=True, show_fig=False)

    for subdir in ("combined", "event_counts", "efficiency"):
        pkl_path = tmp_path / "pkl" / subdir / f"efficiency-{var_config.var_save_name}.pkl"
        assert pkl_path.is_file(), f"missing {pkl_path}"
        with open(pkl_path, "rb") as f:
            fig = pickle.load(f)
        assert fig.__class__.__name__ == "Figure"


def test_does_not_write_flat_legacy_path(tmp_path, one_var_merged):
    # Original (pre-split) behavior wrote directly into save_fig_dir with no
    # view subdirectory at all -- make sure that flat path is gone.
    var_config, merged = one_var_merged
    out_dir = str(tmp_path)

    render_efficiency_plots(merged, out_dir, pot_str="1e20 POT", save_fig=True, show_fig=False)

    legacy_flat_path = tmp_path / f"efficiency-{var_config.var_save_name}.png"
    assert not legacy_flat_path.exists()
    legacy_view_path = tmp_path / "combined" / f"efficiency-{var_config.var_save_name}.png"
    assert not legacy_view_path.exists()


def test_writes_eff_dict_pkl_with_per_variable_entry(tmp_path, one_var_merged):
    var_config, merged = one_var_merged
    out_dir = str(tmp_path)

    render_efficiency_plots(merged, out_dir, pot_str="1e20 POT", save_fig=True, show_fig=False)

    pkl_path = tmp_path / "pkl" / "eff_dict.pkl"
    assert pkl_path.is_file()
    with open(pkl_path, "rb") as f:
        eff_dict = pickle.load(f)
    assert var_config.var_save_name in eff_dict
    entry = eff_dict[var_config.var_save_name]
    assert "eff_list" in entry and "eff_err_list" in entry
    assert len(entry["eff_list"]) == 2  # one per stage


def test_no_efficiency_data_skips_without_raising(tmp_path):
    merged = {"stage_keys": [], "stage_labels": [], "eff": {}}
    # Should just print-and-return, not raise (e.g. NameError on eff_dict).
    render_efficiency_plots(merged, str(tmp_path), pot_str="1e20 POT",
                             save_fig=True, show_fig=False)
    assert list(tmp_path.iterdir()) == []
