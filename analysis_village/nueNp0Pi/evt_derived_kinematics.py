"""Derived per-event / per-track columns expected by :class:`config.plots.VariableConfig`.

Used by chunked syst drivers when reading pre-stored ``evt`` / ``mcnu`` tables that may omit
reco ``phi``, opening-angle helpers, truth-level ``(..., truth, p, phi, )``, or
``(mc, mu|p|trk1|trk2, phi, ...)`` on prefixed ``mcnu`` frames.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pyanalib.pandas_helpers import pad_column_name

from pyanalib.chunked_selection import multicol_resolve_column_key


def ensure_derived_trk_kinematics_cols(evtdf: pd.DataFrame) -> pd.DataFrame:
    """Add reco + truth kinematic columns used by :class:`~config.plots.VariableConfig`.

    Reco (CAF-style): ``theta_mu_p``, ``(mu|p).pfp.trk.phi`` from ``...dir.{x,y,z}``;
    truth: ``mc_theta_mu_p`` from ``mu``/``p`` ``...truth.p.dir.*`` (same ``arccos(dot)`` in rad);
    ``(mu|p).pfp.trk.truth.p.phi`` from ``...truth.p.dir.x/y`` in degrees (same ``arctan2`` as reco).
    ``opening_angle`` bins are radians on ``[0, \\pi]`` for both reco and truth openers.
    """
    if evtdf is None or len(evtdf) == 0:
        return evtdf
    if not isinstance(evtdf.columns, pd.MultiIndex):
        return evtdf

    def _col(df: pd.DataFrame, *parts: str) -> pd.Series | None:
        key = multicol_resolve_column_key(df, parts)
        if key is None:
            return None
        try:
            return df.loc[:, key]
        except Exception:
            return None

    df = evtdf
    add_theta = multicol_resolve_column_key(df, ("theta_mu_p", "", "", "", "", "", "")) is None
    add_mu_phi = multicol_resolve_column_key(df, ("mu", "pfp", "trk", "phi", "", "", "")) is None
    add_p_phi = multicol_resolve_column_key(df, ("p", "pfp", "trk", "phi", "", "", "")) is None
    add_mc_theta = multicol_resolve_column_key(df, ("mc_theta_mu_p", "", "", "", "", "", "")) is None
    add_mu_t_phi = multicol_resolve_column_key(df, ("mu", "pfp", "trk", "truth", "p", "phi", "")) is None
    add_p_t_phi = multicol_resolve_column_key(df, ("p", "pfp", "trk", "truth", "p", "phi", "")) is None

    dirs_ok = all(
        _col(df, "mu", "pfp", "trk", "dir", ax, "", "") is not None
        and _col(df, "p", "pfp", "trk", "dir", ax, "", "") is not None
        for ax in ("x", "y", "z")
    )
    mu_xy_ok = _col(df, "mu", "pfp", "trk", "dir", "x", "", "") is not None and _col(
        df, "mu", "pfp", "trk", "dir", "y", "", ""
    ) is not None
    p_xy_ok = _col(df, "p", "pfp", "trk", "dir", "x", "", "") is not None and _col(
        df, "p", "pfp", "trk", "dir", "y", "", ""
    ) is not None

    truth_dirs_ok = all(
        _col(df, "mu", "pfp", "trk", "truth", "p", "dir", ax) is not None
        and _col(df, "p", "pfp", "trk", "truth", "p", "dir", ax) is not None
        for ax in ("x", "y", "z")
    )
    mu_truth_xy_ok = _col(df, "mu", "pfp", "trk", "truth", "p", "dir", "x") is not None and _col(
        df, "mu", "pfp", "trk", "truth", "p", "dir", "y"
    ) is not None
    p_truth_xy_ok = _col(df, "p", "pfp", "trk", "truth", "p", "dir", "x") is not None and _col(
        df, "p", "pfp", "trk", "truth", "p", "dir", "y"
    ) is not None

    # SPINE's native-units DLP reco/truth kinematic columns are in MeV, but the
    # VariableConfig entries that plot them (config/plots.py's electron_energy(),
    # proton_momentum()) use GeV bins to match their generator-level truth
    # counterparts (mc.e.genE, mc.p.totp). Add "_GeV" siblings (raw * 1e-3) for
    # each so those VariableConfigs have something on the right scale to point
    # at. Confirmed against real data: e.g. rec.dlp.proton_p_reco has
    # mean/max ~O(100-1000), matching mc.p.totp's GeV-scale truth x1000.
    _mev_to_gev_cols = (
        (("rec", "dlp", "ele_energy_reco", "", ""), ("rec", "dlp", "ele_energy_reco_GeV", "", "")),
        (("rec", "dlp_true", "ele_energy_true", "", ""), ("rec", "dlp_true", "ele_energy_true_GeV", "", "")),
        (("rec", "dlp", "proton_p_reco", "", ""), ("rec", "dlp", "proton_p_reco_GeV", "", "")),
        (("rec", "dlp_true", "proton_p_true", "", ""), ("rec", "dlp_true", "proton_p_true_GeV", "", "")),
    )
    gev_conversions_needed = [
        (src, gev) for src, gev in _mev_to_gev_cols
        if multicol_resolve_column_key(df, gev) is None
        and multicol_resolve_column_key(df, src) is not None
    ]

    # Resolution columns for kinematics variables: (out_col, reco_col, true_col, mode).
    # "frac" -> (true - reco) / true (dimensionless); "abs" -> reco - true (same units).
    _kinematics_res_specs = [
        (("rec", "dlp", "ele_energy_res", "", ""),
         ("rec", "dlp", "ele_energy_reco", "", ""),
         ("rec", "dlp_true", "ele_energy_true", "", ""),
         "frac"),
        (("rec", "dlp", "proton_p_res", "", ""),
         ("rec", "dlp", "proton_p_reco", "", ""),
         ("rec", "dlp_true", "proton_p_true", "", ""),
         "frac"),
        (("rec", "dlp", "subprim_proton_p_res", "", ""),
         ("rec", "dlp", "subprim_proton_p_reco", "", ""),
         ("rec", "dlp_true", "subprim_proton_p_true", "", ""),
         "frac"),
        (("rec", "dlp", "del_Tp_res", "", ""),
         ("rec", "dlp", "del_Tp_reco", "", ""),
         ("rec", "dlp_true", "del_Tp_true", "", ""),
         "frac"),
        (("rec", "dlp", "del_Tp_lp_res", "", ""),
         ("rec", "dlp", "del_Tp_lp_reco", "", ""),
         ("rec", "dlp_true", "del_Tp_lp_true", "", ""),
         "frac"),
        (("rec", "dlp", "del_alpha_res", "", ""),
         ("rec", "dlp", "del_alpha_reco", "", ""),
         ("rec", "dlp_true", "del_alpha_true", "", ""),
         "abs"),
        (("rec", "dlp", "del_alpha_lp_res", "", ""),
         ("rec", "dlp", "del_alpha_lp_reco", "", ""),
         ("rec", "dlp_true", "del_alpha_lp_true", "", ""),
         "abs"),
        (("rec", "dlp", "del_phi_res", "", ""),
         ("rec", "dlp", "del_phi_reco", "", ""),
         ("rec", "dlp_true", "del_phi_true", "", ""),
         "abs"),
        (("rec", "dlp", "del_phi_lp_res", "", ""),
         ("rec", "dlp", "del_phi_lp_reco", "", ""),
         ("rec", "dlp_true", "del_phi_lp_true", "", ""),
         "abs"),
        (("rec", "dlp", "lp_open_angle_res", "", ""),
         ("rec", "dlp", "lp_open_angle_reco", "", ""),
         ("rec", "dlp_true", "lp_open_angle_true", "", ""),
         "abs"),
        (("rec", "dlp", "lepton_beam_angle_res", "", ""),
         ("rec", "dlp", "lepton_beam_angle_reco", "", ""),
         ("rec", "dlp_true", "lepton_beam_angle_true", "", ""),
         "abs"),
    ]
    # Only check whether any output column is missing; actual prerequisites are
    # re-evaluated on `out` after GeV conversions (some res cols depend on the
    # _GeV columns that are added in the same function call).
    _might_need_kinematics_res = any(
        multicol_resolve_column_key(df, out_col) is None
        for out_col, _, _, _ in _kinematics_res_specs
    )

    if not (
        (add_theta and dirs_ok)
        or (add_mu_phi and mu_xy_ok)
        or (add_p_phi and p_xy_ok)
        or (add_mc_theta and truth_dirs_ok)
        or (add_mu_t_phi and mu_truth_xy_ok)
        or (add_p_t_phi and p_truth_xy_ok)
        or gev_conversions_needed
        or _might_need_kinematics_res
    ):
        return df

    out = df.copy()
    if add_theta and dirs_ok:
        mx = np.asarray(_col(out, "mu", "pfp", "trk", "dir", "x", "", ""), dtype=float)
        my = np.asarray(_col(out, "mu", "pfp", "trk", "dir", "y", "", ""), dtype=float)
        mz = np.asarray(_col(out, "mu", "pfp", "trk", "dir", "z", "", ""), dtype=float)
        px = np.asarray(_col(out, "p", "pfp", "trk", "dir", "x", "", ""), dtype=float)
        py = np.asarray(_col(out, "p", "pfp", "trk", "dir", "y", "", ""), dtype=float)
        pz = np.asarray(_col(out, "p", "pfp", "trk", "dir", "z", "", ""), dtype=float)
        dot = mx * px + my * py + mz * pz
        dot = np.clip(dot, -1.0, 1.0)
        out.loc[:, pad_column_name(("theta_mu_p", "", "", "", "", "", ""), out)] = np.arccos(dot)
    if add_mu_phi and mu_xy_ok:
        mux = np.asarray(_col(out, "mu", "pfp", "trk", "dir", "x", "", ""), dtype=float)
        muy = np.asarray(_col(out, "mu", "pfp", "trk", "dir", "y", "", ""), dtype=float)
        out.loc[:, pad_column_name(("mu", "pfp", "trk", "phi", "", "", ""), out)] = np.degrees(
            np.arctan2(mux, muy)
        )
    if add_p_phi and p_xy_ok:
        px = np.asarray(_col(out, "p", "pfp", "trk", "dir", "x", "", ""), dtype=float)
        py = np.asarray(_col(out, "p", "pfp", "trk", "dir", "y", "", ""), dtype=float)
        out.loc[:, pad_column_name(("p", "pfp", "trk", "phi", "", "", ""), out)] = np.degrees(
            np.arctan2(px, py)
        )
    if add_mc_theta and truth_dirs_ok:
        mx = np.asarray(_col(out, "mu", "pfp", "trk", "truth", "p", "dir", "x"), dtype=float)
        my = np.asarray(_col(out, "mu", "pfp", "trk", "truth", "p", "dir", "y"), dtype=float)
        mz = np.asarray(_col(out, "mu", "pfp", "trk", "truth", "p", "dir", "z"), dtype=float)
        px = np.asarray(_col(out, "p", "pfp", "trk", "truth", "p", "dir", "x"), dtype=float)
        py = np.asarray(_col(out, "p", "pfp", "trk", "truth", "p", "dir", "y"), dtype=float)
        pz = np.asarray(_col(out, "p", "pfp", "trk", "truth", "p", "dir", "z"), dtype=float)
        dot = mx * px + my * py + mz * pz
        dot = np.clip(dot, -1.0, 1.0)
        out.loc[:, pad_column_name(("mc_theta_mu_p", "", "", "", "", "", ""), out)] = np.arccos(dot)
    if add_mu_t_phi and mu_truth_xy_ok:
        mux = np.asarray(_col(out, "mu", "pfp", "trk", "truth", "p", "dir", "x"), dtype=float)
        muy = np.asarray(_col(out, "mu", "pfp", "trk", "truth", "p", "dir", "y"), dtype=float)
        out.loc[:, pad_column_name(("mu", "pfp", "trk", "truth", "p", "phi", ""), out)] = np.degrees(
            np.arctan2(mux, muy)
        )
    if add_p_t_phi and p_truth_xy_ok:
        px = np.asarray(_col(out, "p", "pfp", "trk", "truth", "p", "dir", "x"), dtype=float)
        py = np.asarray(_col(out, "p", "pfp", "trk", "truth", "p", "dir", "y"), dtype=float)
        out.loc[:, pad_column_name(("p", "pfp", "trk", "truth", "p", "phi", ""), out)] = np.degrees(
            np.arctan2(px, py)
        )
    for src_parts, gev_parts in gev_conversions_needed:
        src_key = multicol_resolve_column_key(out, src_parts)
        out.loc[:, pad_column_name(gev_parts, out)] = out.loc[:, src_key] * 1e-3
    # Re-evaluate res specs on `out` (not df) so that GeV-derived prerequisite
    # columns added above (e.g. proton_p_reco_GeV) are visible to the lookup.
    for out_col, reco_col, true_col, mode in _kinematics_res_specs:
        if multicol_resolve_column_key(out, out_col) is not None:
            continue
        reco_key = multicol_resolve_column_key(out, reco_col)
        true_key = multicol_resolve_column_key(out, true_col)
        if reco_key is None or true_key is None:
            continue
        reco_vals = np.asarray(out.loc[:, reco_key], dtype=float)
        true_vals = np.asarray(out.loc[:, true_key], dtype=float)
        if mode == "frac":
            vals = (true_vals - reco_vals) / true_vals
        else:
            vals = reco_vals - true_vals
        out.loc[:, pad_column_name(out_col, out)] = vals
    return out


def _mcnu_series(df: pd.DataFrame, parts: tuple) -> pd.Series | None:
    key = multicol_resolve_column_key(df, parts)
    if key is None:
        return None
    try:
        return df.loc[:, key]
    except Exception:
        return None


def _mcnu_has_mc_branch_phi(df: pd.DataFrame, branch: str) -> bool:
    for probe in (
        ("mc", branch, "phi", "", "", "", ""),
        ("mc", branch, "phi", ""),
        ("mc", branch, "phi"),
    ):
        if multicol_resolve_column_key(df, probe) is not None:
            return True
    return False


def ensure_mc_level_phi_mcnu(mc_nu_df: pd.DataFrame) -> pd.DataFrame:
    """Add ``mc.<branch>.phi`` (degrees) from ``mc.<branch>.dir.{x,y}`` when missing.

    ``VariableConfig`` xsec nu columns use e.g. ``('mc', 'mu', 'phi', '', '', '', '')``.
    Call **after** :func:`get_systematics_genie._prefix_mcnu_columns` so the ``mc`` group exists.
    Same ``arctan2(dir.x, dir.y)`` convention as reco phi in the GENIE driver.
    """
    if mc_nu_df is None or len(mc_nu_df) == 0:
        return mc_nu_df
    if not isinstance(mc_nu_df.columns, pd.MultiIndex):
        return mc_nu_df

    df = mc_nu_df
    modified = False
    out = df

    for branch in ("mu", "p", "trk1", "trk2"):
        if _mcnu_has_mc_branch_phi(df, branch):
            continue
        mux = _mcnu_series(df, ("mc", branch, "dir", "x"))
        muy = _mcnu_series(df, ("mc", branch, "dir", "y"))
        if mux is None or muy is None:
            continue
        if not modified:
            out = df.copy()
            modified = True
        phi_col = pad_column_name(("mc", branch, "phi"), out)
        out.loc[:, phi_col] = np.degrees(
            np.arctan2(np.asarray(mux, dtype=float), np.asarray(muy, dtype=float))
        )
    return out if modified else df