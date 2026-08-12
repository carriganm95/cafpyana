"""analysis_village/nueNp0Pi/dataset_locations.py regression guards.

Covers the paths_config.py fold-in (FLUX_FILE) and two bugs found and fixed
while reviewing newly-added files: a stale NUMUCC_SYST_DISK_ROOT env var name
that had drifted out of sync with pyanalib.syst_disk_layout.SYST_DISK_ENV,
and an EVENT_SELECTION_GLOBS["mc"] glob that silently ignored
SPRING_GEN1_ROOT overrides because it was joined as an already-absolute path.
"""
import os

from pyanalib.syst_disk_layout import SYST_DISK_ENV


def test_dataset_locations_imports_cleanly():
    import analysis_village.nueNp0Pi.config.datasets as dl
    assert hasattr(dl, "FLUX_FILE")


def test_flux_file_folded_in_from_paths_config():
    from analysis_village.nueNp0Pi.config.datasets import FLUX_FILE
    assert FLUX_FILE  # non-empty
    assert FLUX_FILE == os.environ.get(
        "NUENP0PI_FLUX_FILE",
        "/exp/sbnd/data/users/munjung/flux/sbnd_original_flux.root",
    )


def test_default_syst_disk_root_uses_current_env_var_name(monkeypatch):
    from analysis_village.nueNp0Pi.dataset_paths import default_syst_disk_root

    assert SYST_DISK_ENV == "SYST_DISK_ROOT"  # pin the name this test relies on
    monkeypatch.setenv(SYST_DISK_ENV, "/tmp/my-syst-disk")
    assert str(default_syst_disk_root()) == "/tmp/my-syst-disk"


def test_default_syst_disk_root_ignores_stale_env_var_name(monkeypatch):
    # Regression guard: this used to read NUMUCC_SYST_DISK_ROOT, a name that had
    # drifted out of sync with the already-renamed SYST_DISK_ENV constant.
    from analysis_village.nueNp0Pi.dataset_paths import default_syst_disk_root

    monkeypatch.delenv("SYST_DISK_ROOT", raising=False)
    monkeypatch.setenv("NUMUCC_SYST_DISK_ROOT", "/tmp/should-not-be-used")
    assert str(default_syst_disk_root()) != "/tmp/should-not-be-used"


def test_mc_glob_respects_spring_gen1_root_override(monkeypatch):
    # Regression guard: EVENT_SELECTION_GLOBS["mc"] used to be built by joining an
    # already-absolute path onto SPRING_GEN1_ROOT via pathlib's `/` operator, which
    # silently discards the left-hand side when the right side is absolute -- so
    # overriding NUMUCC_SPRING_GEN1_ROOT had no effect on this specific glob.
    monkeypatch.setenv("NUMUCC_SPRING_GEN1_ROOT", "/tmp/custom-root")
    # dataset_locations reads the env var at import time, so force a fresh import.
    import importlib
    import analysis_village.nueNp0Pi.config.datasets as dl
    importlib.reload(dl)
    try:
        assert dl.EVENT_SELECTION_GLOBS["mc"].startswith("/tmp/custom-root")
    finally:
        importlib.reload(dl)  # restore module state for later tests
