"""Generator-level (mcnu) truth distributions for the truth plots.

Selects true nue CC interactions with at least one primary electron and one
primary proton, with no kinetic-energy thresholds, and the true vertex inside the
signal's truth fiducial volume (``InFV(..., det="SBND_Gen1")``, as in
``make_nueNp0Pi_df``; SPINE's extra ``is_fiducial`` flag can't be computed from mcnu), and
histograms them in the same variables/bins as the truth plots made by
``event_selection_aggregate.render_truth_signal_plots``.

Usage (writes a pickle that the truth plots overlay):

    python -m analysis_village.nueNp0Pi.truth_mcnu \
        --manifest <run_dir>/manifest.json --out <run_dir>/plots/pkl/truth/mcnu_truth_hists.pkl
    (or --files '<dir>/*.df' instead of --manifest)
"""
from __future__ import annotations

import argparse
import json
import pickle
from multiprocessing import Pool
from os import path, makedirs

import numpy as np
import pandas as pd

ELECTRON_MASS_GEV = 0.000511

SELECTION_LABEL = r"True $\nu_e$ CC in FV, $\geq$1e $\geq$1p (no thresholds)"
TRUTH_FV_DET = "SBND_Gen1"   # same FV as sbnd_fiducial_true in make_nueNp0Pi_df


def select_true_nue_cc_ep(nu: pd.DataFrame) -> pd.Series:
    """nue/nuebar CC in the truth FV with >=1 primary electron and >=1 primary proton,
    no thresholds."""
    from makedf.util import InFV
    pos = pd.DataFrame({a: nu[("position", a, "")] for a in "xyz"})
    # column keys, not attributes: ``nu.ne`` is pandas' DataFrame.ne() method
    return ((nu[("iscc", "", "")] == 1) & (nu[("pdg", "", "")].abs() == 12)
            & (nu[("ne", "", "")] >= 1) & (nu[("np", "", "")] >= 1)
            & InFV(pos, det=TRUTH_FV_DET).fillna(False).astype(bool))


def _unit(nu, part):
    p = np.sqrt(nu[(part, "genp", "x")] ** 2 + nu[(part, "genp", "y")] ** 2
                + nu[(part, "genp", "z")] ** 2)
    return (nu[(part, "genp", "x")] / p, nu[(part, "genp", "y")] / p,
            nu[(part, "genp", "z")] / p)


def _tki_lp(nu):
    """Leading-proton TKI angles (deg), same formulas as get_tki_spine_lp."""
    lx, ly = nu[("e", "genp", "x")], nu[("e", "genp", "y")]
    px, py = nu[("p", "genp", "x")], nu[("p", "genp", "y")]
    l_pt = np.sqrt(lx ** 2 + ly ** 2)
    p_pt = np.sqrt(px ** 2 + py ** 2)
    tx, ty = lx + px, ly + py
    t = np.sqrt(tx ** 2 + ty ** 2)
    dphi = np.degrees(np.arccos(np.clip(-(lx * px + ly * py) / (l_pt * p_pt), -1, 1)))
    dalpha = np.degrees(np.arccos(np.clip(-(lx * tx + ly * ty) / (l_pt * t), -1, 1)))
    return dalpha, dphi


def _open_angle(nu):
    ex, ey, ez = _unit(nu, "e")
    px, py, pz = _unit(nu, "p")
    return np.clip(ex * px + ey * py + ez * pz, -1, 1)


# var_save_name -> value from mcnu, matching the units of the slice-level truth column.
# Not available from mcnu (left out): tki-del_Tp / tki-del_phi (summed over all
# hadrons) and secondary-proton-p (mcnu only keeps the leading proton).
MCNU_TRUTH_VARS = {
    "E_nu": lambda nu: nu[("E", "", "")],
    "electron-e": lambda nu: nu[("e", "genE", "")] - ELECTRON_MASS_GEV,   # KE [GeV]
    "proton-p": lambda nu: nu[("p", "totp", "")],                         # |p| [GeV]
    "num-protons": lambda nu: nu[("np", "", "")],
    "opening_angle": _open_angle,                                          # cos
    "opening_angle_beam": lambda nu: _unit(nu, "e")[2],                    # cos
    "tki-del_alpha_lp": lambda nu: _tki_lp(nu)[0],
    "tki-del_phi_lp": lambda nu: _tki_lp(nu)[1],
}


def _one_file(args):
    df_file, bins_by_var = args
    from analysis_village.nueNp0Pi.event_selection import hdr_chunk_pot

    # read every split each key actually has (some files have fewer hdr_N than mcnu_N)
    with pd.HDFStore(df_file, "r") as store:
        keys = [k.strip("/") for k in store.keys()]
        dfs = {}
        for base in ("hdr", "mcnu"):
            parts = sorted((k for k in keys if k.startswith(base + "_") and k[len(base) + 1:].isdigit()),
                           key=lambda k: int(k.split("_")[-1]))
            dfs[base] = pd.concat([store[k] for k in parts]) if parts else None
    if dfs["mcnu"] is None:
        return float(hdr_chunk_pot(dfs["hdr"])), 0, {k: np.zeros(len(b) - 1) for k, b in bins_by_var.items()}
    pot = float(hdr_chunk_pot(dfs["hdr"]))
    nu = dfs["mcnu"]
    nu = nu[select_true_nue_cc_ep(nu)]
    counts = {}
    for name, bins in bins_by_var.items():
        vals = np.asarray(MCNU_TRUTH_VARS[name](nu), dtype=float)
        vals = vals[np.isfinite(vals)]
        counts[name] = np.histogram(vals, bins=bins)[0].astype(float)
    return pot, len(nu), counts


def build_mcnu_truth_hists(df_files, var_configs, nproc: int = 8) -> dict:
    """Sum mcnu truth histograms over ``df_files`` (unweighted, at the files' simulated POT)."""
    bins_by_var = {vc.var_save_name: np.asarray(vc.bins, dtype=float)
                   for vc in var_configs if vc.var_save_name in MCNU_TRUTH_VARS}
    total = {k: np.zeros(len(b) - 1) for k, b in bins_by_var.items()}
    pot, n_int = 0.0, 0
    with Pool(nproc) as pool:
        for i, (p, n, c) in enumerate(pool.imap_unordered(
                _one_file, [(f, bins_by_var) for f in df_files])):
            pot += p
            n_int += n
            for k in total:
                total[k] += c[k]
            print(f"[truth_mcnu] {i + 1}/{len(df_files)} files", flush=True)
    return {
        "pot": pot,
        "n_interactions": n_int,
        "label": SELECTION_LABEL,
        "vars": {k: {"bins": bins_by_var[k], "counts": total[k]} for k in total},
    }


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", help="event-selection manifest.json (uses its mc files)")
    src.add_argument("--files", help="glob of MC .df files instead, e.g. a BlueArc copy "
                                     "(reading .df files through the /pnfs mount can hang)")
    ap.add_argument("--out", required=True, help="output pickle")
    ap.add_argument("--nproc", type=int, default=8)
    args = ap.parse_args()

    from analysis_village.nueNp0Pi.config.stages import EFFICIENCY_VARS

    if args.files:
        import glob
        files = sorted(glob.glob(args.files))
    else:
        manifest = json.load(open(args.manifest))
        files = [x["path"] for x in manifest["file_survey"] if x["sample"] == "mc"]
    if not files:
        raise SystemExit("no MC .df files found")
    res = build_mcnu_truth_hists(files, EFFICIENCY_VARS, nproc=args.nproc)
    makedirs(path.dirname(path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(res, f)
    print(f"[truth_mcnu] {res['n_interactions']} interactions, POT={res['pot']:.4e} -> {args.out}")


if __name__ == "__main__":
    main()
