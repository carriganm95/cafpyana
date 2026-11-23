"""Compare mcnu-derived truth variables with SPINE truth for the same true interaction.

Each slice matched to a true neutrino is linked to its mcnu row via
``rematch_mc_columns`` (key: __ntuple, entry, rec.dlp_true.nu_id). Duplicate slices of
the same true interaction are dropped. For true nue CC interactions with >=1 electron and
>=1 proton (the selection used for the red line in the truth plots), every variable in
``truth_mcnu.MCNU_TRUTH_VARS`` is computed from mcnu and compared with the slice's SPINE
truth column.

    python -m analysis_village.nueNp0Pi.validate_mcnu_truth --out <dir> <mc .df files...>
"""
from __future__ import annotations

import argparse
from os import makedirs, path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyanalib.split_df_helpers_new import load_dfs  # tolerates a stale n_split
from analysis_village.nueNp0Pi.event_selection import rematch_mc_columns
from analysis_village.nueNp0Pi.truth_mcnu import MCNU_TRUTH_VARS, select_true_nue_cc_ep

# var -> (SPINE truth column on evt, scale to mcnu units, agreement tolerance, kind)
# kind "rel": |mcnu/spine - 1| < tol ; "abs": |mcnu - spine| < tol ; "exact": equal
SPINE_COLS = {
    "electron-e":         (("rec", "dlp_true", "ele_energy_true"), 1e-3, 0.02, "rel"),
    "proton-p":           (("rec", "dlp_true", "proton_p_true"), 1e-3, 0.02, "rel"),
    "opening_angle":      (("rec", "dlp_true", "lp_open_angle_true"), 1.0, 0.01, "abs"),
    "opening_angle_beam": (("rec", "dlp_true", "lepton_beam_angle_true"), 1.0, 0.01, "abs"),
    "tki-del_alpha_lp":   (("rec", "dlp_true", "del_alpha_lp_true"), 1.0, 1.0, "abs"),
    "tki-del_phi_lp":     (("rec", "dlp_true", "del_phi_lp_true"), 1.0, 1.0, "abs"),
    # SPINE counts protons with KE >= 25 MeV; mcnu "np" has no threshold -> not expected to match
    "num-protons":        (("rec", "dlp_true", "proton_count_true"), 1.0, 0.0, "exact"),
}


def _col(df, key):
    cols = [c for c in df.columns if c[:len(key)] == key]
    return df[cols[0]] if cols else None


def matched_interactions(df_file):
    dfs = load_dfs(df_file, ["evt", "mcnu"])
    evt = rematch_mc_columns(dfs["evt"], dfs["mcnu"])
    nu_id = _col(evt, ("rec", "dlp_true", "nu_id"))
    evt = evt[(nu_id >= 0).fillna(False).to_numpy()]
    key = pd.MultiIndex.from_arrays(
        [evt.index.get_level_values("__ntuple"), evt.index.get_level_values("entry"),
         _col(evt, ("rec", "dlp_true", "nu_id")).to_numpy()])
    evt = evt[~key.duplicated()]

    # mcnu view of the same rows (drop the leading "mc" level and padding)
    nu = evt[[c for c in evt.columns if c[0] == "mc"]].copy()
    nu.columns = pd.MultiIndex.from_tuples([tuple(c[1:4]) for c in nu.columns])
    sel = select_true_nue_cc_ep(nu).to_numpy()
    return evt[sel], nu[sel]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()
    makedirs(args.out, exist_ok=True)

    evts, nus = [], []
    for f in args.files:
        e, n = matched_interactions(f)
        evts.append(e)
        nus.append(n)
        print(f"[validate] {path.basename(f)}: {len(e)} matched true nue CC >=1e >=1p", flush=True)
    evt = pd.concat(evts)
    nu = pd.concat(nus)

    rows = []
    for name, (col, scale, tol, kind) in SPINE_COLS.items():
        spine = _col(evt, col)
        if spine is None:
            print(f"[validate] {name}: SPINE column {col} not found")
            continue
        s = spine.to_numpy(dtype=float) * scale
        m = np.asarray(MCNU_TRUTH_VARS[name](nu), dtype=float)
        ok = np.isfinite(s) & np.isfinite(m)
        s, m = s[ok], m[ok]
        if kind == "rel":
            good = np.abs(np.divide(m, s, out=np.full_like(m, np.inf), where=s != 0) - 1) < tol
            resid = np.divide(m - s, s, out=np.zeros_like(m), where=s != 0)
            rlab = "(mcnu - SPINE) / SPINE"
        elif kind == "abs":
            good = np.abs(m - s) < tol
            resid = m - s
            rlab = "mcnu - SPINE"
        else:
            good = m == s
            resid = m - s
            rlab = "mcnu - SPINE"
        rows.append({"variable": name, "n_compared": int(ok.sum()),
                     "n_missing_spine_or_mcnu": int((~ok).sum()),
                     "frac_agree": float(good.mean()) if len(good) else np.nan,
                     "tolerance": f"{tol} ({kind})",
                     "median_resid": float(np.median(resid)) if len(resid) else np.nan})

        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
        lo, hi = np.nanpercentile(np.r_[s, m], [0.5, 99.5]) if len(s) else (0, 1)
        a1.hist2d(s, m, bins=60, range=[[lo, hi], [lo, hi]], cmin=1)
        a1.plot([lo, hi], [lo, hi], "r--", lw=1)
        a1.set_xlabel(f"SPINE truth: {col[-1]}")
        a1.set_ylabel(f"mcnu: {name}")
        a2.hist(np.clip(resid, *np.nanpercentile(resid, [1, 99])) if len(resid) else resid, bins=80)
        a2.set_xlabel(rlab)
        a2.set_title(f"{name}: {good.mean():.1%} agree ({tol} {kind})" if len(good) else name)
        fig.tight_layout()
        fig.savefig(path.join(args.out, f"validate-{name}.png"), dpi=120)
        plt.close(fig)

    table = pd.DataFrame(rows)
    table.to_csv(path.join(args.out, "validate_summary.csv"), index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
