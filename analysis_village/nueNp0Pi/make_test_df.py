#!/usr/bin/env python3
"""Cut a small test .df from the first split of a production file, for local tests before a
grid submission (a full MC .df needs ~5 GB of RAM; the interactive node has ~11 GB total).

    python -m analysis_village.nueNp0Pi.make_test_df <in.df> <out.df> [--rows 60000]

Keeps the first ~N slices (whole events only) of split 0, with their tracks, header and mcnu
rows, so POT and truth stay consistent. Refuses to overwrite <out.df>.
"""
import argparse
import os
import sys

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--rows", type=int, default=60000, help="approx. number of slices to keep")
    args = ap.parse_args()
    if os.path.exists(args.out):
        sys.exit(f"{args.out} exists; not overwriting")

    evt = pd.read_hdf(args.inp, "evt_0", start=0, stop=args.rows)
    lv = evt.index.names
    ntuple = evt.index.get_level_values("__ntuple")[0]
    entries = evt.index.get_level_values("entry").unique()
    keep = set(entries[:-1]) if len(entries) > 1 else set(entries)   # drop a partial last entry

    def sel(df):
        i = df.index
        return df[(i.get_level_values("__ntuple") == ntuple) & i.get_level_values("entry").isin(keep)]

    # tracks follow the slice order; read a generous head, then check it covers every kept event
    trk = pd.read_hdf(args.inp, "trk_0", start=0, stop=args.rows * 5)
    out = {"evt": sel(evt), "trk": sel(trk)}
    if out["trk"].index.get_level_values("entry").max() < max(keep):
        sys.exit("track head too short for the kept events; lower --rows")
    with pd.HDFStore(args.inp, "r") as st:
        keys = {k.strip("/") for k in st.keys()}
    for k in ("hdr", "mcnu"):
        if f"{k}_0" in keys:
            out[k] = sel(pd.read_hdf(args.inp, f"{k}_0"))
    for k, df in out.items():
        df.to_hdf(args.out, key=f"{k}_0", mode="a")
    pd.DataFrame({"n_split": [1]}).to_hdf(args.out, key="split", mode="a")
    print(f"{args.out}: " + ", ".join(f"{k} {len(df)} rows" for k, df in out.items())
          + f" ({len(keep)} events, index levels {list(lv)})")


if __name__ == "__main__":
    main()
