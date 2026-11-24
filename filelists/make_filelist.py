#!/usr/bin/env python3
"""Write the CAF file list for one dataset in filelists/datasets.json.

    python filelists/make_filelist.py mc1e20                 # -> filelists/filelist_mc1e20.txt
    python filelists/make_filelist.py sce0x --max-files 1    # -> filelists/filelist_sce0x_test.txt
    python filelists/make_filelist.py --list                 # show the catalogue

Each line is an xrootd URL (root://fndcadoor.fnal.gov:1094/pnfs/fnal.gov/usr/...), the form
grid jobs need; bare /pnfs paths are never written. The glob is resolved in Python, so very
large samples (the in-time cosmics have ~200k files) work where a shell glob would not.
For a SAM definition instead of a catalogue entry use data/sample_lists/create_list.py.
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from pyanalib.xrootd_paths import is_xrootd_url, pnfs_to_xrootd  # noqa: E402

CATALOGUE = os.path.join(HERE, "datasets.json")


def load_catalogue(path=CATALOGUE):
    with open(path) as f:
        return json.load(f)


def resolve(pattern, max_files=None):
    files = []
    for p in glob.iglob(pattern):
        files.append(p)
    files.sort()
    if max_files:
        files = files[:max_files]
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("key", nargs="?", help="dataset key in datasets.json")
    ap.add_argument("--max-files", type=int, default=None,
                    help="keep only the first N files and write filelist_<key>_test.txt")
    ap.add_argument("--out", default=None, help="output path (default filelists/filelist_<key>[_test].txt)")
    ap.add_argument("--list", action="store_true", help="print the catalogue and exit")
    args = ap.parse_args()

    cat = load_catalogue()
    if args.list or not args.key:
        for k, v in cat.items():
            extra = f" (cv: {v['cv']})" if v.get("cv") else ""
            print(f"{k:18s} type={v['type']:5s} role={v.get('role', '?'):10s}{extra}  {v['description']}")
        return
    if args.key not in cat:
        sys.exit(f"unknown dataset {args.key!r}; known: {', '.join(cat)}")

    entry = cat[args.key]
    suffix = "_test" if args.max_files else ""
    out = args.out or os.path.join(HERE, f"filelist_{args.key}{suffix}.txt")
    if os.path.exists(out):
        sys.exit(f"{out} already exists; not overwriting (use --out for a new name)")
    files = resolve(entry["path"], args.max_files)
    if not files:
        sys.exit(f"no files match {entry['path']}")
    urls = [pnfs_to_xrootd(p) for p in files]
    bad = [u for u in urls if not is_xrootd_url(u)]
    if bad:
        sys.exit(f"{len(bad)} path(s) could not be converted to xrootd URLs, e.g. {bad[0]}")

    with open(out, "w") as f:
        f.write("\n".join(urls) + "\n")
    print(f"{len(urls)} file(s) from {entry['path']}\n  type={entry['type']} role={entry.get('role')}"
          f"{' cv=' + entry['cv'] if entry.get('cv') else ''}\n  -> {out}")


if __name__ == "__main__":
    main()
