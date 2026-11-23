#!/usr/bin/env python3
"""Check a .df production without loading its data.

For a sample of files it reads only the HDF5 metadata (table names, column names, row
counts) and checks that every table and column the event-selection pipeline needs is
there -- e.g. the pi0 columns that older productions lack.

    python -m analysis_village.nueNp0Pi.check_dfs /path/to/production_dir --sample mc
    python -m analysis_village.nueNp0Pi.check_dfs a.df b.df --sample data --all

Exit status 0 if every checked file passes.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Dict, List, Set, Tuple

import tables

# Columns (rec.dlp.<name>) the selection cuts and the pi0 sideband read.
RECO_COLS = (
    "is_fiducial", "sbnd_fiducial_reco", "is_flash_matched", "is_contained",
    "muon_mask_reco", "pion_mask_reco", "photon_mask_reco", "ele_mask_reco", "proton_mask_reco",
    "ele_softmax_reco", "ele_primary_reco", "proton_softmax_reco", "ele_vertex_distance_reco",
    "ele_dedx_reco", "photon_energy_reco", "subprim_photon_energy_reco", "pi0_mass_reco",
)
# Truth columns (rec.dlp_true.<name>) needed for MC: truth matching and the signal definition.
TRUTH_COLS = ("nu_id", "true_signal1p", "true_signalNp", "pi0_mass_true")
TABLES = {"mc": ("evt", "trk", "hdr", "mcnu"), "data": ("evt", "trk", "hdr")}


def _column_tuples(node) -> Set[Tuple[str, ...]]:
    """Column labels of a pandas 'fixed' frame, from its axis0 level/label arrays."""
    names = [n._v_name for n in node._f_iter_nodes()]
    if "axis0_level0" not in names:                     # single-level columns
        vals = node._f_get_child("axis0").read()
        return {(v.decode() if isinstance(v, bytes) else str(v),) for v in vals}
    levels = []
    i = 0
    while f"axis0_level{i}" in names:
        vals = [v.decode() if isinstance(v, bytes) else str(v)
                for v in node._f_get_child(f"axis0_level{i}").read()]
        codes = node._f_get_child(f"axis0_label{i}").read()
        levels.append([vals[c] if c >= 0 else "" for c in codes])
        i += 1
    return set(zip(*levels))


def _n_rows(node) -> int:
    for n in node._f_iter_nodes():
        if n._v_name == "axis1" or n._v_name == "axis1_label0":
            return int(n.shape[0])
    return -1


def check_file(path: str, sample: str) -> Tuple[List[str], Dict[str, int]]:
    problems: List[str] = []
    rows: Dict[str, int] = {}
    with tables.open_file(path, "r") as f:
        keys = {n._v_name for n in f.root._f_iter_nodes()}
        splits = sorted(int(k.split("_")[1]) for k in keys if k.startswith("evt_") and k[4:].isdigit())
        if not splits:
            return [f"no evt_* tables (keys: {sorted(keys)[:8]})"], rows
        for t in TABLES[sample]:
            missing = [i for i in splits if f"{t}_{i}" not in keys]
            if missing:
                problems.append(f"table {t}_{missing[0]} missing ({len(missing)} split(s))")
        for t in ("evt", "trk"):
            rows[t] = sum(_n_rows(f.get_node(f"/{t}_{i}")) for i in splits if f"{t}_{i}" in keys)
        cols = _column_tuples(f.get_node("/evt_0"))
    flat = {tuple(p for p in c if p != "") for c in cols}
    for name in RECO_COLS:
        if ("rec", "dlp", name) not in flat:
            problems.append(f"evt column rec.dlp.{name} missing")
    if sample == "mc":
        for name in TRUTH_COLS:
            if ("rec", "dlp_true", name) not in flat:
                problems.append(f"evt column rec.dlp_true.{name} missing")
    return problems, rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help=".df files, or one directory of them")
    ap.add_argument("--sample", choices=("mc", "data"), required=True)
    ap.add_argument("--all", action="store_true", help="check every file (default: first, middle, last)")
    args = ap.parse_args()

    files = (sorted(glob.glob(os.path.join(args.paths[0], "*.df")))
             if len(args.paths) == 1 and os.path.isdir(args.paths[0]) else sorted(args.paths))
    if not files:
        print("no .df files found")
        return 1
    total = sum(os.path.getsize(p) for p in files)
    print(f"{len(files)} .df file(s), {total / 1e9:.1f} GB")
    picks = files if args.all else sorted({files[0], files[len(files) // 2], files[-1]})

    n_bad = 0
    for p in picks:
        try:
            problems, rows = check_file(p, args.sample)
        except Exception as exc:  # unreadable / truncated file
            problems, rows = [f"cannot read: {type(exc).__name__}: {exc}"], {}
        status = "OK " if not problems else "BAD"
        print(f"[{status}] {os.path.basename(p)}  evt rows={rows.get('evt', '?')} trk rows={rows.get('trk', '?')}")
        for msg in problems:
            print(f"       - {msg}")
        n_bad += bool(problems)
    print(f"{len(picks) - n_bad}/{len(picks)} checked file(s) pass")
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
