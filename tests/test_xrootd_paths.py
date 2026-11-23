import json
import os

import pytest

from pyanalib.xrootd_paths import is_xrootd_url, pnfs_to_xrootd

DOOR = "root://fndcadoor.fnal.gov:1094/pnfs/fnal.gov/usr/"


@pytest.mark.parametrize("path, url", [
    ("/pnfs/sbnd/persistent/users/x/a.root", DOOR + "sbnd/persistent/users/x/a.root"),
    ("/pnfs/sbn/persistent/users/x/a.root", DOOR + "sbn/persistent/users/x/a.root"),
    ("/pnfs/fnal.gov/usr/sbnd/a.root", DOOR + "sbnd/a.root"),          # already full namespace
    ("xroot://host//a.root", "root://host//a.root"),
    ("root://host//a.root", "root://host//a.root"),
    ("  /pnfs/sbnd/a.root\n", DOOR + "sbnd/a.root"),
])
def test_pnfs_to_xrootd(path, url):
    assert pnfs_to_xrootd(path) == url
    assert is_xrootd_url(pnfs_to_xrootd(path))


@pytest.mark.parametrize("path", ["/exp/sbnd/data/a.root", "a.root", "/pnfsx/a.root"])
def test_non_pnfs_paths_are_rejected(path):
    with pytest.raises(ValueError):
        pnfs_to_xrootd(path)


def test_catalogue_entries_are_complete():
    """Every dataset has a /pnfs glob, a type and a known role. (Detector-variation entries
    are all role "detvar" for now; pairing them with their CVs comes with the detvar
    processing code.)"""
    cat_path = os.path.join(os.path.dirname(__file__), "..", "filelists", "datasets.json")
    cat = json.load(open(cat_path))
    roles = {"cv", "data", "offbeam", "intime", "dirt", "detvar"}
    for key, entry in cat.items():
        assert entry["path"].startswith("/pnfs/"), key
        assert entry["type"] in ("mc", "data"), key
        assert entry["role"] in roles, key
