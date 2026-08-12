"""pyanalib.variable_config: the generic VariableConfig record shape, and
analysis_village/nueNp0Pi/variable_configs.py's subclass built on top of it.
"""
import numpy as np
import pytest

from pyanalib.variable_config import (
    VariableConfig,
    INTEGRATED_VAR_SAVE_NAME,
    is_integrated_var_config,
)


def _make(var_save_name="v", **overrides):
    kwargs = dict(
        var_save_name=var_save_name,
        var_plot_name="V",
        var_labels=["V", "V reco", "V true"],
        bins=np.array([0.0, 1.0, 2.0, 3.0]),
        var_evt_reco_col="reco",
        var_evt_truth_col="truth",
        var_nu_col="nu",
        xsec_label="dsigma/dV",
    )
    kwargs.update(overrides)
    return VariableConfig(**kwargs)


def test_bin_centers_derived_from_bins():
    vc = _make(bins=np.array([0.0, 2.0, 4.0]))
    assert list(vc.bin_centers) == [1.0, 3.0]


def test_category_syst_var_save_name_optional_defaults_none():
    vc = _make()
    assert vc.category_syst_var_save_name is None
    vc2 = _make(category_syst_var_save_name="other")
    assert vc2.category_syst_var_save_name == "other"


def test_is_integrated_var_config():
    vc = _make(var_save_name=INTEGRATED_VAR_SAVE_NAME)
    assert is_integrated_var_config(vc) is True
    assert is_integrated_var_config(_make(var_save_name="not-integrated")) is False
    # anything without var_save_name (e.g. plain None) should not raise
    assert is_integrated_var_config(None) is False


def test_nueNp0Pi_variable_configs_subclasses_base():
    from analysis_village.nueNp0Pi.config.plots import VariableConfig as NueVC

    assert issubclass(NueVC, VariableConfig)


def test_nueNp0Pi_muon_momentum_has_expected_shape():
    from analysis_village.nueNp0Pi.config.plots import VariableConfig as NueVC

    vc = NueVC.muon_momentum()
    assert vc.var_save_name == "muon-p"
    assert len(vc.bin_centers) == len(vc.bins) - 1
    assert vc.xsec_label  # non-empty, inherited attribute the notebook copies lack


def test_nueNp0Pi_all_events_uses_shared_sentinel():
    from analysis_village.nueNp0Pi.config.plots import VariableConfig as NueVC

    vc = NueVC.all_events()
    assert vc.var_save_name == INTEGRATED_VAR_SAVE_NAME
    assert is_integrated_var_config(vc) is True


def test_nueNp0Pi_list_all_configs_enumerates_every_classmethod():
    from analysis_village.nueNp0Pi.config.plots import VariableConfig as NueVC

    names = NueVC.list_all_configs(print_summary=False)
    assert "muon_momentum" in names
    assert "proton_momentum" in names
    assert "all_events" in names
    assert len(names) > 60  # ~68 at time of writing; guards against silent mass-deletion
