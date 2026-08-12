"""pyanalib.syst_disk_layout: path-resolution helpers for the precomputed
systematics-covariance directory tree. Pure path-string logic, no I/O other
than resolve_genie_disk_path's os.path.isfile probe.
"""
import os

from pyanalib.syst_disk_layout import (
    SYST_DISK_ENV,
    category_out_dir,
    category_summary_manifest_path,
    category_summary_npz_path,
    normalized_root,
    resolve_genie_disk_path,
    syst_disk_paths,
)


def test_syst_disk_env_name_is_generic():
    # Regression guard: this was renamed from NUMUCC_SYST_DISK_ROOT so the module
    # could be shared across analyses. Don't let it drift back to an analysis name.
    assert SYST_DISK_ENV == "SYST_DISK_ROOT"


def test_syst_disk_paths_layout(tmp_path):
    root = str(tmp_path)
    paths = syst_disk_paths(root)
    assert paths["root"] == normalized_root(root)
    assert paths["mcstat"].endswith(os.path.join("MCstat", "mcstat_syst_dict.npz"))
    assert paths["flux"].endswith(os.path.join("Flux", "flux_syst_dict.npz"))
    assert paths["g4"].endswith(os.path.join("G4", "g4_syst_dict.npz"))
    assert paths["genie"].endswith(os.path.join("GENIE", "cov_mat_dict.pkl"))
    assert paths["cosmics"].endswith(os.path.join("Cosmics", "cosmics_syst_dict.npz"))
    assert paths["detector"].endswith(os.path.join("Detector", "detector_syst_dict.npz"))
    assert paths["category_summary"] == category_summary_npz_path(root)


def test_normalized_root_expands_user_and_relative(tmp_path):
    # trailing separator should be stripped, path should become absolute
    assert normalized_root(str(tmp_path) + os.sep) == str(tmp_path)


def test_category_out_dir(tmp_path):
    assert category_out_dir(str(tmp_path), "Flux") == os.path.join(str(tmp_path), "Flux")


def test_category_summary_manifest_path(tmp_path):
    npz = category_summary_npz_path(str(tmp_path))
    manifest = category_summary_manifest_path(npz)
    assert manifest == os.path.join(os.path.dirname(npz), "category_syst_summary_manifest.json")


def test_resolve_genie_disk_path_missing_returns_none(tmp_path):
    assert resolve_genie_disk_path(str(tmp_path)) is None


def test_resolve_genie_disk_path_finds_pkl(tmp_path):
    genie_dir = tmp_path / "GENIE"
    genie_dir.mkdir()
    pkl = genie_dir / "cov_mat_dict.pkl"
    pkl.write_bytes(b"not a real pickle, just needs to exist")
    found = resolve_genie_disk_path(str(tmp_path))
    assert found == str(pkl)


def test_resolve_genie_disk_path_finds_npz_sidecar(tmp_path):
    genie_dir = tmp_path / "GENIE"
    genie_dir.mkdir()
    npz = genie_dir / "cov_mat_dict.pkl.npz"
    npz.write_bytes(b"also not real, just needs to exist")
    found = resolve_genie_disk_path(str(tmp_path))
    assert found == str(npz)
