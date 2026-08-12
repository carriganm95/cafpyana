"""Regression tests for the print() -> logger conversion in the systematics
loading section of analysis_village/nueNp0Pi/utils.py (get_syst_unc and the
category-summary / GENIE-SB covariance loaders).

The systematics-loading engine itself now lives in :mod:`pyanalib.syst_loading`
(analysis_village/nueNp0Pi/utils.py is a thin wrapper around it) -- so its debug
and info logging happens under the ``cafpyana.pyanalib.syst_loading`` logger,
not ``cafpyana.analysis_village.nueNp0Pi.utils``. ERROR/WARNING-level messages
still get captured under the nueNp0Pi logger name too, since the shared
``cafpyana`` root's default effective level (WARNING) lets them through
regardless of which specific descendant logger's threshold ``caplog.at_level``
adjusts; DEBUG/INFO messages don't clear that default threshold unless the
level is lowered on the *exact* logger doing the logging.
"""
import logging

import numpy as np
import pytest


def test_utils_has_a_module_logger():
    import analysis_village.nueNp0Pi.utils as u
    assert u.logger.name == "cafpyana.analysis_village.nueNp0Pi.utils"


def test_fail_syst_disk_logs_error_and_raises(caplog):
    import analysis_village.nueNp0Pi.utils as u

    with caplog.at_level(logging.ERROR, logger="cafpyana.analysis_village.nueNp0Pi.utils"):
        with pytest.raises(FileNotFoundError):
            u._fail_syst_disk("boom")
    assert any("boom" in rec.message for rec in caplog.records)


def test_resolve_syst_disk_root_missing_raises_via_logged_error(monkeypatch, caplog):
    import analysis_village.nueNp0Pi.utils as u

    monkeypatch.delenv(u.SYST_DISK_ENV, raising=False)
    with pytest.raises(FileNotFoundError):
        u.resolve_syst_disk_root(None)


def test_resolve_syst_disk_root_debug_logs_on_success(monkeypatch, caplog):
    import analysis_village.nueNp0Pi.utils as u

    monkeypatch.setenv(u.SYST_DISK_ENV, "/tmp/some-syst-disk")
    with caplog.at_level(logging.DEBUG, logger="cafpyana.pyanalib.syst_loading"):
        root = u.resolve_syst_disk_root(None)
    assert root  # resolved successfully
    assert any("resolved syst disk root" in rec.message for rec in caplog.records)


def test_resolve_category_syst_summary_path_warns_on_hardcoded_fallback(monkeypatch, caplog):
    import analysis_village.nueNp0Pi.utils as u

    monkeypatch.delenv(u.SYST_DISK_ENV, raising=False)
    with caplog.at_level(logging.WARNING, logger="cafpyana.analysis_village.nueNp0Pi.utils"):
        path = u.resolve_category_syst_summary_path()
    assert u._DEFAULT_SYST_DISK_ROOT in path
    assert any("falling back to hardcoded default root" in rec.message for rec in caplog.records)


def test_resolve_category_syst_summary_path_no_warning_when_env_set(monkeypatch, caplog):
    import analysis_village.nueNp0Pi.utils as u

    monkeypatch.setenv(u.SYST_DISK_ENV, "/tmp/my-syst-disk")
    with caplog.at_level(logging.WARNING, logger="cafpyana.analysis_village.nueNp0Pi.utils"):
        path = u.resolve_category_syst_summary_path()
    assert path.startswith("/tmp/my-syst-disk")
    assert not any("falling back" in rec.message for rec in caplog.records)


def test_get_syst_unc_flat_components_logs_summary(caplog):
    import analysis_village.nueNp0Pi.utils as u

    class _FakeVarConfig:
        var_save_name = "x"
        bin_centers = np.array([0.5, 1.5])
        bins = np.array([0.0, 1.0, 2.0])

    with caplog.at_level(logging.INFO, logger="cafpyana.pyanalib.syst_loading"):
        unc, cov = u.get_syst_unc(_FakeVarConfig(), syst_components=["pot", "ntargets"])
    assert unc.shape == (2,)
    assert any("mean frac. unc." in rec.message for rec in caplog.records)
