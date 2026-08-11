import numpy as np
import pandas as pd
from makedf.util import *
from makedf.constants import *
import vector

from pyanalib.pandas_helpers import multicol_add, pad_column_name

_CC1PI0PI_TKI_VAR_NAMES = (
    "del_alpha",
    "del_phi",
    "del_Tp",
    "del_p",
    "del_Tp_x",
    "del_Tp_y",
)


def get_cc1p0pi_tki(lepdf, pdf, P_lep_col, P_p_col, lepton_mass=MUON_MASS):
    """
    Calculate TKI variables for CC Np0pi selected events.

    Inputs:
    - lepdf       : pandas df with charged lepton information. Must contain P_lep_col and dir.
    - pdf         : pandas df with leading-proton information. Must contain P_p_col and dir.
    - P_lep_col   : tuple of str -- column name in lepdf for the absolute lepton momentum.
    - P_p_col     : tuple of str -- column name in pdf for the absolute proton momentum.
    - lepton_mass : lepton mass in GeV (default MUON_MASS; use ELECTRON_MASS for nueCC).

    Returns:
    - A dictionary with one entry per TKI observable:
       - del_alpha: angle between transverse momentum of lepton and transverse momentum imbalance
       - del_phi:   angle between transverse momentum of lepton and transverse momentum of proton
       - del_Tp:    magnitude of the transverse momentum imbalance
       - del_Tp_x:  (p̂_ν × p̂_T^l) · δp⃗_T  with p̂_ν along +z
       - del_Tp_y:  -p̂_T^l · δp⃗_T
       - del_p:     magnitude of the 3D imbalance
    """

    lep_p = lepdf[P_lep_col]
    lep_p_x = lep_p * lepdf["dir"]["x"]
    lep_p_y = lep_p * lepdf["dir"]["y"]
    lep_p_z = lep_p * lepdf["dir"]["z"]
    lep_phi_x = lep_p_x/mag2d(lep_p_x, lep_p_y)
    lep_phi_y = lep_p_y/mag2d(lep_p_x, lep_p_y)

    p_p = pdf[P_p_col]
    p_p_x = p_p * pdf["dir"]["x"]
    p_p_y = p_p * pdf["dir"]["y"]
    p_p_z = p_p * pdf["dir"]["z"]

    lep_Tp_x = lepdf["dir"]["x"] * lep_p
    lep_Tp_y = lepdf["dir"]["y"] * lep_p
    lep_Tp = mag2d(lep_Tp_x, lep_Tp_y)

    p_Tp_x = pdf["dir"]["x"] * p_p
    p_Tp_y = pdf["dir"]["y"] * p_p
    p_Tp = mag2d(p_Tp_x, p_Tp_y)

    _del_Tp_x = lep_Tp_x + p_Tp_x
    _del_Tp_y = lep_Tp_y + p_Tp_y
    del_Tp = mag2d(_del_Tp_x, _del_Tp_y)

    del_alpha = np.arccos(-(lep_Tp_x*_del_Tp_x + lep_Tp_y*_del_Tp_y)/(lep_Tp*del_Tp))
    del_phi = np.arccos(-(lep_Tp_x*p_Tp_x + lep_Tp_y*p_Tp_y)/(lep_Tp*p_Tp))

    lep_E = mag2d(lep_p, lepton_mass)
    p_E = mag2d(p_p, PROTON_MASS)

    # R = MASS_A + lep_p_z + p_p_z - lep_E - p_E
    # del_Lp = 0.5*R - mag2d(MASS_Ap, del_Tp)**2/(2*R)
    e_cal = lep_E - lepton_mass + p_E - PROTON_MASS + 0.0309 # https://link.springer.com/article/10.1140/epjc/s10052-019-6750-3
    del_Lp = lep_p_z + p_p_z - e_cal
    del_p = mag2d(del_Tp, del_Lp)

    # δp_{T,x} = (p̂_ν × p̂_T^l) · δp⃗_T,  δp_{T,y} = -p̂_T^l · δp⃗_T  (p̂_ν = +z)
    del_Tp_x = -lep_phi_y * _del_Tp_x + lep_phi_x * _del_Tp_y
    del_Tp_y = -(lep_phi_x * _del_Tp_x + lep_phi_y * _del_Tp_y)

    return {
        "del_alpha": del_alpha * 180/np.pi,
        "del_phi": del_phi * 180/np.pi,
        "del_Tp": del_Tp,
        "del_Tp_x": del_Tp_x,
        "del_Tp_y": del_Tp_y,
        "del_p": del_p,
    }


_DEFAULT_LEPTON_P_COL = {"mu": "p_muon", "ele": "p_electron"}


def get_tki_spine(particles, lepton_idx, int_levels):
    """Vectorized TKI using summed hadronic transverse momentum (pid > 2).

    particles:   particle-level DataFrame (e.g. _ptmp.rec.dlp.particles)
    lepton_idx:  particle-level Series with the lepton particle index broadcast per interaction
    int_levels:  list of index levels identifying an interaction (all but the last)

    Returns a dict of interaction-level Series. NaN propagates automatically when
    lepton_idx is NaN (no lepton found for that interaction).
    """
    part_idx = pd.Series(particles.index.get_level_values(-1), index=particles.index)
    lep_mask = part_idx == lepton_idx  # True only for the lepton row per interaction

    lep_px = particles.momentum['x'].where(lep_mask).groupby(level=int_levels).first()
    lep_py = particles.momentum['y'].where(lep_mask).groupby(level=int_levels).first()

    had_mask = particles.pid > 2
    had_px = particles.momentum['x'].where(had_mask).groupby(level=int_levels).sum()
    had_py = particles.momentum['y'].where(had_mask).groupby(level=int_levels).sum()

    lep_pT   = np.sqrt(lep_px**2 + lep_py**2)
    had_pT   = np.sqrt(had_px**2 + had_py**2)
    del_Tp_x = lep_px + had_px
    del_Tp_y = lep_py + had_py
    del_Tp   = np.sqrt(del_Tp_x**2 + del_Tp_y**2)

    del_phi   = np.arccos(np.clip(-(lep_px*had_px   + lep_py*had_py)   / (lep_pT * had_pT), -1, 1)) * 180/np.pi
    del_alpha = np.arccos(np.clip(-(lep_px*del_Tp_x + lep_py*del_Tp_y) / (lep_pT * del_Tp), -1, 1)) * 180/np.pi

    return {"del_alpha": del_alpha, "del_phi": del_phi, "del_Tp": del_Tp}


def get_tki_spine_lp(particles, lepton_idx, proton_idx, int_levels):
    """Vectorized TKI using the leading proton only.

    particles:   particle-level DataFrame
    lepton_idx:  particle-level Series with the lepton particle index broadcast per interaction
    proton_idx:  particle-level Series with the leading proton particle index broadcast per interaction
    int_levels:  list of index levels identifying an interaction

    Returns a dict of interaction-level Series.
    """
    part_idx = pd.Series(particles.index.get_level_values(-1), index=particles.index)
    lep_mask = part_idx == lepton_idx
    pro_mask = part_idx == proton_idx

    lep_px = particles.momentum['x'].where(lep_mask).groupby(level=int_levels).first()
    lep_py = particles.momentum['y'].where(lep_mask).groupby(level=int_levels).first()
    pro_px = particles.momentum['x'].where(pro_mask).groupby(level=int_levels).first()
    pro_py = particles.momentum['y'].where(pro_mask).groupby(level=int_levels).first()

    lep_pT   = np.sqrt(lep_px**2 + lep_py**2)
    pro_pT   = np.sqrt(pro_px**2 + pro_py**2)
    del_Tp_x = lep_px + pro_px
    del_Tp_y = lep_py + pro_py
    del_Tp   = np.sqrt(del_Tp_x**2 + del_Tp_y**2)

    del_phi   = np.arccos(np.clip(-(lep_px*pro_px   + lep_py*pro_py)   / (lep_pT * pro_pT), -1, 1)) * 180/np.pi
    del_alpha = np.arccos(np.clip(-(lep_px*del_Tp_x + lep_py*del_Tp_y) / (lep_pT * del_Tp), -1, 1)) * 180/np.pi

    return {"del_alpha": del_alpha, "del_phi": del_phi, "del_Tp": del_Tp}

def add_reco_cc1p0pi_tki_evtdf(
    evtdf: pd.DataFrame,
    lepton_key: str = "mu",
    lepton_mass: float = MUON_MASS,
) -> pd.DataFrame:
    """Attach reco TKI columns ``del_*`` onto ``evtdf`` via :func:`get_cc1p0pi_tki`.

    ``lepton_key`` selects which lepton block to use (``"mu"`` for numuCC, ``"ele"`` for nueCC).
    Skips if ``del_Tp`` already appears at column level 0.
    """
    if evtdf is None or len(evtdf) == 0:
        return evtdf
    try:
        if "del_Tp" in evtdf.columns.get_level_values(0):
            return evtdf
    except Exception:
        pass
    slc_lepdf = evtdf[lepton_key].pfp.trk
    slc_pdf = evtdf.p.pfp.trk
    p_col_name = _DEFAULT_LEPTON_P_COL.get(lepton_key, f"p_{lepton_key}")
    slc_P_lep_col = pad_column_name(("P", p_col_name), slc_lepdf)
    slc_P_p_col = pad_column_name(("P", "p_proton"), slc_pdf)
    tki_reco = get_cc1p0pi_tki(slc_lepdf, slc_pdf, slc_P_lep_col, slc_P_p_col, lepton_mass=lepton_mass)
    for var_name in _CC1PI0PI_TKI_VAR_NAMES:
        evtdf = multicol_add(evtdf, tki_reco[var_name].rename(var_name))
    return evtdf


def add_mc_cc1p0pi_tki_mcnu(
    mc_nu_df: pd.DataFrame,
    lepton_key: str = "mu",
    lepton_mass: float = MUON_MASS,
) -> pd.DataFrame:
    """Attach MC-truth TKI ``del_*`` onto ``mcnu`` via :func:`get_cc1p0pi_tki`.

    ``lepton_key`` selects the lepton block (``"mu"`` for numuCC, ``"ele"`` for nueCC).
    Resolves lepton / proton blocks as:

    * ``mc.<lepton_key>`` / ``mc.p`` when truth columns live under a leading ``mc`` level.
    * Top-level ``<lepton_key>`` / ``p`` when the ``mc`` slice has no ``<lepton_key>``
      (mixed layout: GENIE weights under ``mc``, truth lepton/proton at top level).
    """
    if mc_nu_df is None or len(mc_nu_df) == 0:
        return mc_nu_df
    try:
        if "del_Tp" in mc_nu_df.columns.get_level_values(0):
            return mc_nu_df
    except Exception:
        pass
    if not isinstance(mc_nu_df.columns, pd.MultiIndex):
        return mc_nu_df

    levels0 = set(mc_nu_df.columns.get_level_values(0))
    mc_lepdf = None
    mc_pdf = None
    # Prefer truth nested under mc when that subtree actually contains lepton_key / p.
    if "mc" in levels0:
        mc_blk = mc_nu_df["mc"]
        sub0 = set(mc_blk.columns.get_level_values(0))
        if lepton_key in sub0 and "p" in sub0:
            mc_lepdf = mc_blk[lepton_key]
            mc_pdf = mc_blk["p"]
    # Mixed mcnu: lepton / proton stay at top level.
    if mc_lepdf is None and lepton_key in levels0 and "p" in levels0:
        mc_lepdf = mc_nu_df[lepton_key]
        mc_pdf = mc_nu_df["p"]
    if mc_lepdf is None or mc_pdf is None:
        return mc_nu_df

    try:
        mc_P_lep_col = pad_column_name(("totp",), mc_lepdf)
        mc_P_p_col = pad_column_name(("totp",), mc_pdf)
        tki_mc = get_cc1p0pi_tki(mc_lepdf, mc_pdf, mc_P_lep_col, mc_P_p_col, lepton_mass=lepton_mass)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return mc_nu_df
    for var_name in _CC1PI0PI_TKI_VAR_NAMES:
        mc_nu_df = multicol_add(mc_nu_df, tki_mc[var_name].rename("{}".format(var_name)))
    return mc_nu_df


def add_truth_cc1p0pi_tki_evtdf(
    evtdf: pd.DataFrame,
    lepton_key: str = "mu",
    lepton_mass: float = MUON_MASS,
) -> pd.DataFrame:
    """Truth-level TKI on ``evtdf`` under ``('mc', '<var>', '', ...)`` via :func:`get_cc1p0pi_tki`.

    ``lepton_key`` selects which lepton block to use (``"mu"`` for numuCC, ``"ele"`` for nueCC).
    """
    if evtdf is None or len(evtdf) == 0:
        return evtdf
    nlevel = evtdf.columns.nlevels
    key_mc_delTp = ("mc", "del_Tp") + ("",) * (nlevel - 2)
    try:
        if key_mc_delTp in evtdf.columns:
            return evtdf
    except Exception:
        pass
    try:
        slc_lepdf = evtdf[lepton_key].pfp.trk.truth.p
        slc_pdf = evtdf.p.pfp.trk.truth.p
    except (KeyError, AttributeError, TypeError):
        return evtdf
    slc_P_lep_col = pad_column_name(("totp",), slc_lepdf)
    slc_P_p_col = pad_column_name(("totp",), slc_pdf)
    tki_truth = get_cc1p0pi_tki(slc_lepdf, slc_pdf, slc_P_lep_col, slc_P_p_col, lepton_mass=lepton_mass)
    for var_name in _CC1PI0PI_TKI_VAR_NAMES:
        tname = ("mc", var_name) + ("",) * (nlevel - 2)
        evtdf = multicol_add(evtdf, tki_truth[var_name].rename(tname))
    return evtdf