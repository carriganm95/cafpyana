"""Selection logic: reconstruction-level cuts + truth-level categories.

This is where the actual cut/predicate/category FUNCTIONS live for this
analysis (thresholds and label/color settings they're built from live in
:mod:`analysis_village.nueNp0Pi.config.settings`, kept separate so that
file only has to be read to find what's configurable, not how it's used).

Also holds ``SIGNAL_MASK_FN`` -- this analysis's "what counts as signal"
definition, used by :class:`analysis_village.nueNp0Pi.event_selection.ChunkRunner`.
(``BREAKDOWN_REGISTRY`` -- "what breakdown categories exist" -- is assembled
in :mod:`analysis_village.nueNp0Pi.config.stages` instead, from the category
functions below; see the note near the bottom of this file.)

Formerly split across ``makedf/selections.py`` and ``categories.py``, then
briefly merged into ``config/selections.py`` before this settings/logic split.

Plot/variable definitions live in :mod:`analysis_village.nueNp0Pi.config.plots`.
Dataset paths and other config live in :mod:`analysis_village.nueNp0Pi.config.datasets`.
"""
import sys
from os import path as _path

import numpy as np
import pandas as pd

sys.path.append(_path.dirname(_path.dirname(_path.dirname(_path.abspath(__file__)))))
from makedf.util import *
from pyanalib.variable_calculator import *
from pyanalib.pandas_helpers import *
from makedf.constants import *

# Explicit (not wildcard) -- re-binds DETECTOR/PER_TPC_INCATHODE_CM to this
# analysis's values (shadowing the makedf.constants wildcard import above),
# and pulls in the threshold/label settings the functions below are built on.
from analysis_village.nueNp0Pi.config.settings import (
    DETECTOR, PER_TPC_INCATHODE_CM,
    NU_SCORE_TH, SAVE_NTRKS, TRACKSCORE_TH, VTXDIST_TH,
    MU_CHI2MU_TH, MU_CHI2P_TH, MU_LEN_TH, QUAL_TH, P_CHI2P_TH, P_LEN_TH,
    MU_PLO_TH, MU_PHI_TH, P_PLO_TH, P_PHI_TH,
)
from pyanalib.chunked_selection import multicol_resolve_column_key


# ===========================================================================
# Truth-level categories: fiducial/topology/interaction-mode/particle-type
# definitions used for signal/background breakdown plots and efficiency
# denominators. (Formerly ``analysis_village/nueNp0Pi/categories.py``.)
# ===========================================================================

def IsNu(df):
    return (np.abs(df.mc.pdg) == 14) | (np.abs(df.mc.pdg) == 12)

def IsCosmic(df):
    return ~IsNu(df)

def IsNuOutFV(df):
    return IsNu(df) & ~InFV(df.mc.position, det=DETECTOR)

def IsNuInFV(df):
    return IsNu(df) & InFV(df.mc.position, det=DETECTOR)

def IsNuInFV_NuOther(df):
    return IsNuInFV(df) & (df.mc.iscc == 1) & (df.mc.pdg != 14)

def IsNuInFV_NumuNC(df):
    return IsNuInFV(df) & (df.mc.iscc == 0)


def IsTruthCC1p0piPerTPCFV(df, incathode=PER_TPC_INCATHODE_CM):
    """Truth fiducial aligned with reco per-TPC cut (vertex + μ/p ends in same TPC).

    Reco uses ``slc.vertex`` and reconstructed track ends; truth uses ``mc.position``
    and true lepton end positions. Same ``SBND_TPC1`` / ``SBND_TPC2`` x-bands as
    ``InFV(..., det=\"SBND_TPC1|2\")`` in ``makedf.util``.
    """
    in_tpc1 = (
        InFV(df.mc.position, det="SBND_TPC1", incathode=incathode)
        & InFV(df.mc.e.end, det="SBND_TPC1", incathode=incathode)
        & InFV(df.mc.p.end, det="SBND_TPC1", incathode=incathode)
    )
    in_tpc2 = (
        InFV(df.mc.position, det="SBND_TPC2", incathode=incathode)
        & InFV(df.mc.e.end, det="SBND_TPC2", incathode=incathode)
        & InFV(df.mc.p.end, det="SBND_TPC2", incathode=incathode)
    )
    return in_tpc1 | in_tpc2

def IsTruthCC1p0piNominalFV(df, detector=DETECTOR):
    """Truth fiducial aligned with nominal ``SBND_nohighyz`` (μ/p start and end in volume)."""
    return (
        InFV(df.mc.e.start, det=detector)
        & InFV(df.mc.p.start, det=detector)
        & InFV(df.mc.e.end, det=detector)
        & InFV(df.mc.p.end, det=detector)
    )

def IsTruthCC1p0piFVSpine(df, detector=DETECTOR):
    """Fiducial volume selection directly from SPINE definition"""
    return (
        df.rec.dlp_true.is_fiducial
    )


# ---- numu CC in FV, breakdown in topology
def Is_1p0pi(df, detector=DETECTOR, signal_truth_fv="per_tpc"):
    """True CC 1p0π topology with configurable truth fiducial.

    signal_truth_fv : {'per_tpc', 'nominal', 'none'}
        ``per_tpc`` — ``IsTruthCC1p0piPerTPCFV`` (matches per-TPC reco selection).
        ``nominal`` — μ/p start and end in ``detector`` (default ``SBND_nohighyz``).
        ``none`` — topology only, no μ/p end containment requirement.
    """
    topo = (
        (df.mc.nmu_220MeVc == 1)
        & (df.mc.np_300MeVc == 1)
        & (df.mc.npi_70MeVc == 0)
        & (df.mc.npi0 == 0)
        & (np.sqrt(df.mc.mu.genp.x**2 + df.mc.mu.genp.y**2 + df.mc.mu.genp.z**2) < 1)
        & (np.sqrt(df.mc.p.genp.x**2 + df.mc.p.genp.y**2 + df.mc.p.genp.z**2) < 1)
    )
    if signal_truth_fv == "per_tpc":
        return topo & IsTruthCC1p0piPerTPCFV(df)
    if signal_truth_fv == "nominal":
        return topo & IsTruthCC1p0piNominalFV(df, detector=detector)
    if signal_truth_fv == "none":
        return topo
    raise ValueError(
        f"signal_truth_fv must be 'per_tpc', 'nominal', or 'none', got {signal_truth_fv!r}"
    )

def Is_Np0pi(df):
    return (df.mc.nmu_220MeVc == 1) & (df.mc.np_300MeVc > 1) & (df.mc.npi_70MeVc == 0) & (df.mc.npi0 == 0)

def IsNuInFV_NumuCC_Other(df, detector=DETECTOR, signal_truth_fv="per_tpc"):
    return IsNuInFV(df) & (df.mc.pdg == 14) & (df.mc.iscc == 1) &\
              ~Is_1p0pi(df, detector=detector, signal_truth_fv=signal_truth_fv) & ~Is_Np0pi(df)

def IsNuInFV_NumuCC_Np0pi(df):
    return IsNuInFV(df) & (df.mc.pdg == 14) & (df.mc.iscc == 1) &\
              Is_Np0pi(df)

def IsNuInFV_NumuCC_1p0pi(df, detector=DETECTOR, signal_truth_fv="per_tpc"):
    return IsNuInFV(df) & (df.mc.pdg == 14) & (df.mc.iscc == 1) &\
              Is_1p0pi(df, detector=detector, signal_truth_fv=signal_truth_fv)

def IsNuInFV_NueCC_1p0pi(df, detector=DETECTOR, signal_truth_fv="per_tpc"):
    return IsNuInFV(df) & (df.mc.pdg == 12) & (df.mc.iscc == 1) &\
              IsNueNp0pi(df, detector=detector, signal_truth_fv=signal_truth_fv)

# --- numu CC in FV, breakdown in interaction mode (GENIE)
def IsNuInFV_NumuCC_QE(df):
    return IsNuInFV(df) & (df.mc.pdg == 14) & (df.mc.iscc == 1) &\
              (df.mc.genie_mode == 0)

def IsNuInFV_NumuCC_MEC(df):
    return IsNuInFV(df) & (df.mc.pdg == 14) & (df.mc.iscc == 1) &\
              (df.mc.genie_mode == 10)

def IsNuInFV_NumuCC_RES(df):
    return IsNuInFV(df) & (df.mc.pdg == 14) & (df.mc.iscc == 1) &\
              (df.mc.genie_mode == 1)

def IsNuInFV_NumuCC_DIS(df):
    return IsNuInFV(df) & (df.mc.pdg == 14) & (df.mc.iscc == 1) &\
              (df.mc.genie_mode == 2)

# def IsNuInFV_NumuCC_COH(df):
#     return IsNuInFV(df) & (df.pdg == 14) & (df.iscc == 1) &\
#               (df.genie_mode == 3)

def IsNuInFV_NumuCC_OtherMode(df):
    return IsNuInFV(df) & (df.mc.pdg == 14) & (df.mc.iscc == 1) &\
              ~(df.mc.genie_mode == 0) & ~(df.mc.genie_mode == 1) & ~(df.mc.genie_mode == 2) & ~(df.mc.genie_mode == 10)


# --- numu CC in FV, breakdown in interaction mode (GiBUU)
def IsNuInFV_NumuCC_QE_GiBUU(df):
    return IsNuInFV(df) & (df.mc.pdg == 14) & (df.mc.iscc == 1) &\
              (df.mc.genie_mode == 1)

def IsNuInFV_NumuCC_MEC_GiBUU(df):
    return IsNuInFV(df) & (df.mc.pdg == 14) & (df.mc.iscc == 1) &\
              ((df.mc.genie_mode == 35) | (df.mc.genie_mode == 36))

def IsNuInFV_NumuCC_RES_GiBUU(df):
    return IsNuInFV(df) & (df.mc.pdg == 14) & (df.mc.iscc == 1) &\
              ((df.mc.genie_mode >= 2) & (df.mc.genie_mode <= 31))

def IsNuInFV_NumuCC_DIS_GiBUU(df):
    return IsNuInFV(df) & (df.mc.pdg == 14) & (df.mc.iscc == 1) &\
              ((df.mc.genie_mode == 32) | (df.mc.genie_mode == 33) | (df.mc.genie_mode == 34) | (df.mc.genie_mode == 37))

def IsNuInFV_NumuCC_OtherMode_GiBUU(df):
    return IsNuInFV(df) & (df.mc.pdg == 14) & (df.mc.iscc == 1) &\
              ~IsNuInFV_NumuCC_QE_GiBUU(df) & ~IsNuInFV_NumuCC_MEC_GiBUU(df) & ~IsNuInFV_NumuCC_RES_GiBUU(df) & ~IsNuInFV_NumuCC_DIS_GiBUU(df)


# --- for event topoloy breakdown ---
# NOTE: this redefines IsNu (see the first definition above) -- preserved
# exactly as found when this code was consolidated from the old
# makedf/selections.py + categories.py files; the SECOND definition below is
# the one actually in effect (Python keeps the last def). Flagging in case
# this divergence (mc.pdg-based vs isna-based) wasn't intentional.
def IsNu(df):
    return ~df.mc.pdg.isna()


def IsSignal(df): # definition
    is_fv = InFV(df.mc.position, det=DETECTOR)
    is_1mu1p0pi = (df.mc.nmu_220MeVc == 1) & (df.mc.npi_70MeVc == 0) & (df.mc.np_300MeVc == 1) & (df.mc.npi0 == 0) & (df.mc.mu.totp < 1) & (df.mc.p.totp < 1) # & (df.np_20MeVc == 1) : add with stubs
    return is_fv & is_1mu1p0pi


def Is1muNp0pi(df): # definition
    is_fv = InFV(df.mc.position, det=DETECTOR)
    is_1mu1p0pi = (df.mc.nmu_220MeVc == 1) & (df.mc.npi_70MeVc == 0) & (df.mc.np_300MeVc > 1) & (df.mc.npi0 == 0) & (df.mc.mu.totp < 1) & (df.mc.p.totp < 1) #& (df.mu.genE > 0.25) # & (df.np_20MeVc == 1) : add with stubs
    return is_fv & is_1mu1p0pi


def Is1muNcpi(df): # definition
    is_fv = InFV(df.mc.position, det=DETECTOR)
    is_1mu1p0pi = (df.mc.nmu_220MeVc == 1) & (df.mc.npi_70MeVc > 0) & (df.mc.npi0 == 0) #& (df.mu.genE > 0.25) # & (df.np_20MeVc == 1) : add with stubs
    return is_fv & is_1mu1p0pi

def IsNueNp0pi(df, detector=DETECTOR, signal_truth_fv="none"): # definition
    """True CC 1p0π topology with configurable truth fiducial.

    signal_truth_fv : {'per_tpc', 'nominal', 'none'}
        ``per_tpc`` — ``IsTruthCC1p0piPerTPCFV`` (matches per-TPC reco selection).
        ``nominal`` — μ/p start and end in ``detector`` (default ``SBND_nohighyz``).
        ``none`` — topology only, no μ/p end containment requirement.
    """
    topo = (
        (df.mc.ne_500MeV == 1)
        & (df.mc.np_40MeV == 1)
        & (df.mc.npi_25MeV == 0)
        & (df.mc.nmu_25MeV == 0)
        & (df.mc.ng_100MeV == 0)
    )
    if signal_truth_fv == "per_tpc":
        return topo & IsTruthCC1p0piPerTPCFV(df)
    if signal_truth_fv == "nominal":
        return topo & IsTruthCC1p0piNominalFV(df, detector=detector)
    if signal_truth_fv == "none":
        return topo
    raise ValueError(
        f"signal_truth_fv must be 'per_tpc', 'nominal', or 'none', got {signal_truth_fv!r}"
    )

# ==== functions to get breakdowns of event categories ====

def get_pdg_category_spine(df, ret_cuts=False, print_summary=False):
    """Reco-track truth-pdg category (SPINE), in ``pdg_labels`` order.

    Signature/ordering matches its topology/genie siblings
    (``get_topo_category_nueNp0Pi``, ``get_genie_category_spine``) so it can
    be called uniformly as ``get_cuts_fn(df, ret_cuts=True)`` from
    ``BREAKDOWN_REGISTRY``. NOTE: this categorizes individual PARTICLES
    (needs ``df.rec.dlp_true.particles.pdg_code``, only present on the
    per-track table), unlike topology/genie which categorize whole EVENTS --
    it must be driven off a track-level ``PlotSpec(selector=...)``, not
    ``ChunkRunner.bar_breakdown_types`` (which always uses the event-level df).
    """
    cut_ele = (np.abs(df.rec.dlp_true.particles.pdg_code) == 11)
    cut_photon = (df.rec.dlp_true.particles.pdg_code == 22)
    cut_muon = (np.abs(df.rec.dlp_true.particles.pdg_code) == 13)
    cut_proton = (df.rec.dlp_true.particles.pdg_code == 2212)
    cut_pion = (np.abs(df.rec.dlp_true.particles.pdg_code) == 211)
    cut_other = (~cut_ele & ~cut_photon & ~cut_muon & ~cut_proton & ~cut_pion)
    cuts = [cut_ele, cut_photon, cut_muon, cut_proton, cut_pion, cut_other]

    if print_summary:
        pdg_categ = pd.Series(len(cuts) - 1, index=df.index)
        for i, cut in enumerate(cuts):
            pdg_categ[cut] = i
        print(pdg_categ.value_counts())

    if ret_cuts:
        return cuts

    pdg_categ = pd.Series(len(cuts) - 1, index=df.index)
    for i, cut in enumerate(cuts):
        pdg_categ[cut] = i
    return pdg_categ

def get_topo_category_nueNp0Pi(df, ret_cuts=False, print_summary=False, detector=DETECTOR, signal_truth_fv="per_tpc"):

    all_flags = (
        df.rec.dlp_true.true_signal1p +
        df.rec.dlp_true.true_signalNp +
        df.rec.dlp_true.bkgd_oofv +
        df.rec.dlp_true.bkgd_oops +
        df.rec.dlp_true.bkgd_pi +
        df.rec.dlp_true.bkgd_mu +
        df.rec.dlp_true.bkgd_photon +
        df.rec.dlp_true.bkgd_nueOther +
        df.rec.dlp_true.bkgd_numu +
        df.rec.dlp_true.bkgd_nc +
        df.rec.dlp_true.bkgd_other
    )
    assert (all_flags != 1).sum() == 0, f"{(all_flags != 1).sum()} rows have overlapping or missing categories"

    # Index order below MUST match config/settings.py's topology_labels
    # (OOPS then OOFV, at indices 2/3) -- these two were previously swapped
    # here relative to that list, a pre-existing bug (confirmed by
    # cross-referencing against topology_labels, not just assumed) that
    # mislabeled OOPS events as "OOFV" and vice versa in breakdown plots.
    nuint_categ = pd.Series(10, index=df.index)
    nuint_categ[df.rec.dlp_true.true_signal1p == 1] = 0
    nuint_categ[df.rec.dlp_true.true_signalNp == 1] = 1
    nuint_categ[df.rec.dlp_true.bkgd_oops == 1] = 2
    nuint_categ[df.rec.dlp_true.bkgd_oofv == 1] = 3
    nuint_categ[df.rec.dlp_true.bkgd_pi == 1] = 4
    nuint_categ[df.rec.dlp_true.bkgd_mu == 1] = 5
    nuint_categ[df.rec.dlp_true.bkgd_photon == 1] = 6
    nuint_categ[df.rec.dlp_true.bkgd_nueOther == 1] = 7
    nuint_categ[df.rec.dlp_true.bkgd_numu == 1] = 8
    nuint_categ[df.rec.dlp_true.bkgd_nc == 1] = 9
    nuint_categ[df.rec.dlp_true.bkgd_other == 1] = 10

    if print_summary:
        print(nuint_categ.value_counts())

    if ret_cuts:
        return [
            df.rec.dlp_true.true_signal1p == 1,   # 0
            df.rec.dlp_true.true_signalNp == 1,    # 1
            df.rec.dlp_true.bkgd_oops == 1,        # 2
            df.rec.dlp_true.bkgd_oofv == 1,        # 3
            df.rec.dlp_true.bkgd_pi == 1,          # 4
            df.rec.dlp_true.bkgd_mu == 1,          # 5
            df.rec.dlp_true.bkgd_photon == 1,      # 6
            df.rec.dlp_true.bkgd_nueOther == 1,    # 7
            df.rec.dlp_true.bkgd_numu == 1,        # 8
            df.rec.dlp_true.bkgd_nc == 1,          # 9
            df.rec.dlp_true.bkgd_other == 1,       # 10
        ]

    return nuint_categ


def get_genie_category_spine(df, ret_cuts=False, print_summary=False):

    mode = df.rec.dlp_true.interaction_mode
    iscc = df.rec.dlp_true.cc_mask == True
    isnu = df.rec.dlp_true.nu_id >= 0

    good_nu = isnu & iscc

    genie_categ = pd.Series(10, index=df.index)
    genie_categ[good_nu & (mode == 0)] = 0  # QE
    genie_categ[good_nu & (mode == 1)] = 1  # RES
    genie_categ[good_nu & (mode == 2)] = 2  # DIS
    genie_categ[good_nu & (mode == 10)] = 3  # MEC
    genie_categ[good_nu & ((mode != 0) & (mode != 1) & (mode != 2) & (mode != 10))] = 4  # Other
    genie_categ[isnu & ~iscc] = -1
    genie_categ[~isnu] = -2

    if ret_cuts:
        cc_other = good_nu & ~(mode == 0) & ~(mode == 1) & ~(mode == 2) & ~(mode == 10)
        return [
            good_nu & (mode == 0),   # 0: QE
            good_nu & (mode == 1),   # 1: RES
            good_nu & (mode == 2),   # 2: DIS
            good_nu & (mode == 10),  # 3: MEC
            cc_other,                # 4: Other CC
            isnu & ~iscc,            # 5: NC
            ~isnu,                   # 6: Cosmic
        ]

    return genie_categ

def get_genie_sb_category(df, ret_cuts=False, print_summary=False, detector=DETECTOR):
    cut_cosmic = IsCosmic(df)
    cut_nu_other = (IsNuOutFV(df) | IsNuInFV_NuOther(df))
    cut_nu_infv_numu_nc = IsNuInFV_NumuNC(df)
    # break down numu modes into signal and background
    cut_nu_infv_numu_othermode_s = (IsNuInFV_NumuCC_OtherMode(df) | IsNuInFV_NumuCC_DIS(df)) & IsNuInFV_NumuCC_1p0pi(df, detector=detector)
    cut_nu_infv_numu_othermode_b = (IsNuInFV_NumuCC_OtherMode(df) | IsNuInFV_NumuCC_DIS(df)) & ~IsNuInFV_NumuCC_1p0pi(df, detector=detector)
    cut_nu_infv_numu_cc_res_s = IsNuInFV_NumuCC_RES(df) & IsNuInFV_NumuCC_1p0pi(df, detector=detector)
    cut_nu_infv_numu_cc_res_b = IsNuInFV_NumuCC_RES(df) & ~IsNuInFV_NumuCC_1p0pi(df, detector=detector)
    cut_nu_infv_numu_cc_mec_s = IsNuInFV_NumuCC_MEC(df) & IsNuInFV_NumuCC_1p0pi(df, detector=detector)
    cut_nu_infv_numu_cc_mec_b = IsNuInFV_NumuCC_MEC(df) & ~IsNuInFV_NumuCC_1p0pi(df, detector=detector)
    cut_nu_infv_numu_cc_qe_s = IsNuInFV_NumuCC_QE(df) & IsNuInFV_NumuCC_1p0pi(df, detector=detector)
    cut_nu_infv_numu_cc_qe_b = IsNuInFV_NumuCC_QE(df) & ~IsNuInFV_NumuCC_1p0pi(df, detector=detector)

    assert (cut_cosmic & cut_nu_other & cut_nu_infv_numu_nc & cut_nu_infv_numu_othermode_s & cut_nu_infv_numu_othermode_b & cut_nu_infv_numu_cc_res_s & cut_nu_infv_numu_cc_res_b & cut_nu_infv_numu_cc_mec_s & cut_nu_infv_numu_cc_mec_b & cut_nu_infv_numu_cc_qe_s & cut_nu_infv_numu_cc_qe_b).sum() == 0
    assert (cut_cosmic | cut_nu_other | cut_nu_infv_numu_nc | cut_nu_infv_numu_othermode_s | cut_nu_infv_numu_othermode_b | cut_nu_infv_numu_cc_res_s | cut_nu_infv_numu_cc_res_b | cut_nu_infv_numu_cc_mec_s | cut_nu_infv_numu_cc_mec_b | cut_nu_infv_numu_cc_qe_s | cut_nu_infv_numu_cc_qe_b).sum() == len(df)

    cuts = [cut_cosmic, cut_nu_other, cut_nu_infv_numu_nc,
            cut_nu_infv_numu_othermode_b, cut_nu_infv_numu_othermode_s,
            cut_nu_infv_numu_cc_res_b, cut_nu_infv_numu_cc_res_s,
            cut_nu_infv_numu_cc_mec_b, cut_nu_infv_numu_cc_mec_s,
            cut_nu_infv_numu_cc_qe_b, cut_nu_infv_numu_cc_qe_s]

    # if print_summary:
    #     print(genie_sb_categ.value_counts())

    if ret_cuts:
        return cuts

    # TODO: return category series
    return cuts


# ===========================================================================
# Reconstruction-level event-selection cuts.
# (Formerly ``analysis_village/nueNp0Pi/makedf/selections.py``.)
# ===========================================================================
def fiducial_volume(df):
    return df[df.rec.dlp.is_fiducial == 1]

def contained(df):
    return df[df.rec.dlp.is_contained == 1]

def flash_matched(df):
    return df[df.rec.dlp.is_flash_matched == 1]

def no_muons(df):
    return df[df.rec.dlp.muon_mask_reco == 0]

def no_pions(df):
    return df[df.rec.dlp.pion_mask_reco == 0]

def no_photons(df):
    return df[df.rec.dlp.photon_mask_reco == 0]

def good_electron(df):
    return df[df.rec.dlp.ele_mask_reco == 1]

def good_proton(df):
    return df[df.rec.dlp.proton_mask_reco == 1]

def electron_softmax(df):
    return df[df.rec.dlp.ele_softmax_reco > 0.9]

def electron_primary(df):
    return df[df.rec.dlp.ele_primary_reco > 0.99]

def proton_softmax(df):
    return df[df.rec.dlp.proton_softmax_reco > 0.75]

def electron_vertex_distance(df):
    return df[df.rec.dlp.ele_vertex_distance_reco < 3.5]

def electron_dedx(df):
    return df[df.rec.dlp.ele_dedx_reco < 4]



def cut_clear_cosmic(df):
    return df[df.slc.is_clear_cosmic == 0]


def _is_gen1_or_per_tpc_det(det: str) -> bool:
    """True when ``det`` selects Gen-1 / per-TPC reco containment (aliases)."""
    return det in ("SBND_Gen1", "perTPC", "per_tpc", "Gen1")


def vertex_in_gen1_fv(df):
    """Slice vertex inside the Gen-1 fiducial volume (pre-track stage)."""
    return df[InFV(df.slc.vertex, det="SBND_Gen1")]


def event_contained_per_tpc(df, incathode=PER_TPC_INCATHODE_CM):
    """True when the whole slice is fully contained in TPC1 **or** TPC2.

    Requires the slice vertex and both leading tracks' start **and** end points
    to pass the FULL ``SBND_Gen1`` fiducial volume (x, y, z) AND all be in the
    same TPC (all x < 0 or all x > 0).
    """
    points = [
        df.slc.vertex,
        df.trk1.pfp.trk.start,
        df.trk1.pfp.trk.end,
        df.trk2.pfp.trk.start,
        df.trk2.pfp.trk.end,
    ]

    # All points must be in the full Gen1 fiducial volume (x, y, z)
    all_in_fv = None
    for pt in points:
        m = InFV(pt, det="SBND_Gen1")
        all_in_fv = m if all_in_fv is None else (all_in_fv & m)

    # All points must be in the SAME TPC (all x < -cathode or all x > +cathode)
    all_in_tpc1 = None
    all_in_tpc2 = None
    for pt in points:
        m1 = pt.x < (-1 * incathode)
        m2 = pt.x > incathode
        all_in_tpc1 = m1 if all_in_tpc1 is None else (all_in_tpc1 & m1)
        all_in_tpc2 = m2 if all_in_tpc2 is None else (all_in_tpc2 & m2)

    return all_in_fv & (all_in_tpc1 | all_in_tpc2)


def cut_vertex_in_fv(df, det="SBND"):
    if _is_gen1_or_per_tpc_det(det):
        return vertex_in_gen1_fv(df)
    return df[InFV(df.slc.vertex, det=det)]

def cut_nu_score(df, th=0.5):
    return df[df.slc.nu_score > th]

def get_valid_trks(df):
    return df[df.pfp.trk.producer != 4294967295]

def cut_good_trks(trkdf):
    mask = (trkdf.pfp.trk.len > 0) &\
         (trkdf.pfp.pfochar.vtxdist < 100) #&\
    return trkdf[mask]


def _dedupe_event_level_index(trk_df: pd.DataFrame, n_event_levels: int) -> pd.DataFrame:
    """Drop duplicate slice keys before merging tracks onto ``evt``.

    ``groupby(...).nth(i)`` can return multiple rows per slice when track ordering
    ties; ``multicol_merge(..., validate='one_to_one')`` then raises and ``trk1`` /
    ``trk2`` never get attached.
    """
    if trk_df is None or len(trk_df) == 0:
        return trk_df
    if trk_df.index.duplicated().any():
        return trk_df[~trk_df.index.duplicated(keep="first")]
    return trk_df


def _is_merged_trk_block_name(name) -> bool:
    """True for per-slice track blocks added by :func:`get_trk_info` (incl. merge suffixes)."""
    s = str(name)
    if s in ("mu", "p"):
        return True
    if s.startswith("nocut_trk") or (s.startswith("trk") and len(s) > 3 and s[3].isdigit()):
        return True
    return False


def evt_has_trk1_trk2(evtdf: pd.DataFrame) -> bool:
    """True when per-slice ``trk1`` / ``trk2`` column blocks are present on ``evt``."""
    if evtdf is None or len(evtdf) == 0:
        return False
    try:
        top = evtdf.columns.get_level_values(0).unique()
    except Exception:
        return False
    return ("trk1" in top) and ("trk2" in top)


def evt_has_block(evtdf: pd.DataFrame, block_name: str) -> bool:
    """True when a top-level MultiIndex block (e.g. ``mu``, ``p``) is present on ``evt``."""
    if evtdf is None or len(evtdf) == 0:
        return False
    try:
        top = evtdf.columns.get_level_values(0).unique()
    except Exception:
        return False
    return block_name in top


def _merge_nan_pid_placeholder(evtdf, trks, nlevels, block_name):
    """Left-merge a NaN-filled pid block when no candidate rows exist to merge."""
    if evtdf is None or len(evtdf) == 0 or evt_has_block(evtdf, block_name):
        return evtdf
    placeholder = trks.groupby(level=list(range(nlevels))).nth(0)
    if len(placeholder) == 0:
        return evtdf
    placeholder = placeholder.copy()
    for col in placeholder.columns:
        parts = (col,) if isinstance(col, str) else tuple(c for c in col if c)
        if parts and parts[-1] == "producer":
            placeholder.loc[:, col] = np.nan
    placeholder.columns = pd.MultiIndex.from_tuples(
        [tuple([block_name] + list(c)) for c in placeholder.columns]
    )
    placeholder = _dedupe_event_level_index(placeholder, nlevels)
    return multicol_merge(
        evtdf, placeholder, left_index=True, right_index=True, how="left", validate="one_to_one"
    )


def _drop_merged_trk_blocks(evtdf: pd.DataFrame) -> pd.DataFrame:
    """Remove prior ``trk*`` / ``nocut_trk*`` / ``mu`` / ``p`` blocks before re-merging tracks.

    Re-running :func:`get_trk_info` without this leaves duplicate top-level names; pandas
    then suffixes columns (``trk1_x``) and ``evt.trk1`` attribute access breaks.
    """
    if evtdf is None or len(evtdf.columns) == 0:
        return evtdf
    if not isinstance(evtdf.columns, pd.MultiIndex):
        return evtdf
    lev0 = evtdf.columns.get_level_values(0)
    keep = ~pd.Index(lev0).map(_is_merged_trk_block_name)
    if keep.all():
        return evtdf
    return evtdf.loc[:, keep]


def get_trk_info(evtdf, trkdf, save_ntrks=3):
    if trkdf is None or len(trkdf) == 0:
        return evtdf
    good_trks = cut_good_trks(trkdf).copy()
    if len(good_trks) == 0:
        # Do not call _drop_merged_trk_blocks — would strip trk1/trk2 with nothing to re-merge.
        return evtdf
    evtdf = _drop_merged_trk_blocks(evtdf)
    nlevels = len(trkdf.index.names)
    ntrks = trkdf.pfp.id.groupby(level=list(range(nlevels-1))).count()
    ntrks.reindex(evtdf.index, fill_value=0)
    evtdf.loc[:, "n_trks"] = ntrks.copy()

    ntrks = good_trks.pfp.id.groupby(level=list(range(nlevels-1))).count()
    ntrks.reindex(evtdf.index, fill_value=0)
    evtdf.loc[:, "n_good_trks"] = ntrks.copy()

    trks_sorted = trkdf.sort_values(by=('pfp','trk','len'), ascending=False)
    good_trks_sorted = good_trks.sort_values(by=('pfp','trk','len'), ascending=False)
    # get 'ntrks' longest tracks
    evt_levels = nlevels - 1
    for i in range(save_ntrks):
        trk_i = good_trks_sorted.groupby(level=list(range(evt_levels))).nth(i)
        trk_i.columns = pd.MultiIndex.from_tuples([tuple(["nocut_trk" + str(i+1)] + list(c)) for c in trk_i.columns])
        trk_i = _dedupe_event_level_index(trk_i.droplevel(-1), evt_levels)
        evtdf = multicol_merge(evtdf, trk_i, left_index=True, right_index=True, how="left", validate="one_to_one")

        good_trk_i = good_trks_sorted.groupby(level=list(range(evt_levels))).nth(i)
        good_trk_i.columns = pd.MultiIndex.from_tuples([tuple(["trk" + str(i+1)] + list(c)) for c in good_trk_i.columns])
        good_trk_i = _dedupe_event_level_index(good_trk_i.droplevel(-1), evt_levels)
        evtdf = multicol_merge(evtdf, good_trk_i, left_index=True, right_index=True, how="left", validate="one_to_one")

    return evtdf


def cut_2prong(df):
    if df is None or len(df) == 0:
        return df
    if "n_good_trks" not in df.columns:
        return df.iloc[0:0]
    # return df[(df.n_good_trks == 2) & (df.n_trks <= 3)]
    return df[(df.n_good_trks == 2)]

def cut_2prong_contained(df, det="SBND"):
    if df is None or len(df) == 0:
        return df
    if not evt_has_trk1_trk2(df):
        return df.iloc[0:0]
    if _is_gen1_or_per_tpc_det(det):
        return df[event_contained_per_tpc(df)]

    return df[
        InFV(df.trk1.pfp.trk.start, det=det)
        & InFV(df.trk1.pfp.trk.end, det=det)
        & InFV(df.trk2.pfp.trk.start, det=det)
        & InFV(df.trk2.pfp.trk.end, det=det)
    ]

def cut_2prong_trackscore(df, trackscore_th=0.5):
    if df is None or len(df) == 0 or not evt_has_trk1_trk2(df):
        return df.iloc[0:0] if df is not None else df
    return df[(df.trk1.pfp.trackScore > trackscore_th) & (df.trk2.pfp.trackScore > trackscore_th)]

def cut_2prong_vtxdist(df, vtxdist_th=1.5):
    if df is None or len(df) == 0 or not evt_has_trk1_trk2(df):
        return df.iloc[0:0] if df is not None else df
    return df[(df.trk1.pfp.pfochar.vtxdist < vtxdist_th) & (df.trk2.pfp.pfochar.vtxdist < vtxdist_th)]

def get_mu_p_candidate(df,
                       mu_chi2mu_th=30, mu_chi2p_th=100, mu_len_th=50, qual_th=0.25,
                       p_chi2mu_th=30, p_chi2p_th=90, p_len_th=0, score_tag=""):

    if df is None or len(df) == 0:
        return df

    nlevels = len(df.index.names)

    if not evt_has_trk1_trk2(df):
        raise KeyError(
            "evt is missing trk1/trk2 columns required for mu/p PID — "
            "call get_trk_info(evt, trk) after matching tracks to the current slice table"
        )
    trks = pd.concat([df.trk1, df.trk2])

    chimu_avg = avg_chi2(trks, f"chi2_muon{score_tag}")
    chip_avg = avg_chi2(trks, f"chi2_proton{score_tag}")

    mcs_range_diff = np.abs((trks.pfp.trk.rangeP.p_muon - trks.pfp.trk.mcsP.fwdP_muon) / trks.pfp.trk.rangeP.p_muon)

    mu_cut = (chimu_avg > 0) & (chimu_avg < mu_chi2mu_th) & \
            (chip_avg > mu_chi2p_th) & \
            (trks.pfp.trk.len > mu_len_th) & \
            (mcs_range_diff < qual_th)

    mu_candidate = trks[mu_cut]
    mu_candidate = mu_candidate.groupby(level=list(range(nlevels))).nth(0)

    mu_candidate.columns = pd.MultiIndex.from_tuples([tuple(["mu"] + list(c)) for c in mu_candidate.columns])
    if len(mu_candidate) == 0:
        df = _merge_nan_pid_placeholder(df, trks, nlevels, "mu")
    else:
        df = multicol_merge(df, mu_candidate, left_index=True, right_index=True, how="left", validate="one_to_one")

    # TODO: keep & use original trk index?
    not_mu_candidate = pd.concat([trks[~mu_cut], trks[mu_cut].groupby(level=list(range(nlevels))).nth(1)])
    chip_avg = avg_chi2(not_mu_candidate, f"chi2_proton{score_tag}")
    p_candidate = not_mu_candidate[(chip_avg > 0) & (chip_avg < p_chi2p_th) & (not_mu_candidate.pfp.trk.len > p_len_th)]
    p_candidate = p_candidate.groupby(level=list(range(nlevels))).nth(0)

    p_candidate.columns = pd.MultiIndex.from_tuples([tuple(["p"] + list(c)) for c in p_candidate.columns])
    if len(p_candidate) == 0:
        df = _merge_nan_pid_placeholder(df, trks, nlevels, "p")
    else:
        df = multicol_merge(df, p_candidate, left_index=True, right_index=True, how="left", validate="one_to_one")

    return df

def cut_has_mu(df):
    if df is None or len(df) == 0 or not evt_has_block(df, "mu"):
        return df.iloc[0:0] if df is not None else df
    return df[~np.isnan(df.mu.pfp.trk.producer)]

def cut_has_p(df):
    if df is None or len(df) == 0 or not evt_has_block(df, "p"):
        return df.iloc[0:0] if df is not None else df
    return df[~np.isnan(df.p.pfp.trk.producer)]

def cut_mu_kinematics(df, mu_Plo_th=0.22, mu_Phi_th= 1):
    if df is None or len(df) == 0 or not evt_has_block(df, "mu"):
        return df.iloc[0:0] if df is not None else df
    return df[(df.mu.pfp.trk.rangeP.p_muon > mu_Plo_th) & (df.mu.pfp.trk.rangeP.p_muon < mu_Phi_th)]

def cut_p_kinematics(df, p_Plo_th=0.3, p_Phi_th= 1):
    if df is None or len(df) == 0 or not evt_has_block(df, "p"):
        return df.iloc[0:0] if df is not None else df
    return df[(df.p.pfp.trk.rangeP.p_proton > p_Plo_th) & (df.p.pfp.trk.rangeP.p_proton < p_Phi_th)]


# ===========================================================================
# Signal definition for the chunked event-selection pipeline.
# (Formerly ``analysis_village/nueNp0Pi/selection_framework.py``.)
#
# BREAKDOWN_REGISTRY (the breakdown-category dict built from
# get_topo_category_nueNp0Pi / get_genie_category_spine / get_pdg_category_spine
# above) now lives in :mod:`analysis_village.nueNp0Pi.config.stages` instead of
# here -- it's assembled there rather than in config/settings.py to avoid a
# circular import (settings.py has no imports from this file; stages.py
# already imports category functions from here with no cycle).
# ===========================================================================

# DLP truth signal keys — present in evt_df, absent in mcnu_df.
_DLP_SIGNAL1P_KEY = ('rec', 'dlp_true', 'true_signal1p', '', '')
_DLP_SIGNAL_NP_KEY = ('rec', 'dlp_true', 'true_signalNp', '', '')


def _mcnu_signal_mask_dlp_phase_space(df):
    """GENIE-level approximation of DLP true_signal1p|true_signalNp phase space.

    NueCC in FV with ≥1 primary electron (no KE threshold — DLP accepts electrons
    below 500 MeV), ≥1 proton KE>40 MeV, and no pions/muons/photons above threshold.
    Gives ~283 events per file vs ~160 DLP truth events (~57% DLP reco efficiency).

    To use all nue CC in FV as denominator instead (~484/file), replace with:
        IsNuInFV(df) & (df.mc.iscc == 1) & (df.mc.pdg.abs() == 12)
    """
    ne_key = multicol_resolve_column_key(df, ("mc", "ne", ""))
    np_key = multicol_resolve_column_key(df, ("mc", "np_40MeV", ""))
    npi_key = multicol_resolve_column_key(df, ("mc", "npi_25MeV", ""))
    nmu_key = multicol_resolve_column_key(df, ("mc", "nmu_25MeV", ""))
    ng_key = multicol_resolve_column_key(df, ("mc", "ng_100MeV", ""))
    if any(k is None for k in (ne_key, np_key, npi_key, nmu_key, ng_key)):
        return IsNuInFV(df) & (df.mc.iscc == 1) & (df.mc.pdg.abs() == 12)
    return (
        IsNuInFV(df)
        & (df.mc.iscc == 1)
        & (df.mc.pdg.abs() == 12)
        & (df.loc[:, ne_key] >= 1)
        & (df.loc[:, np_key] >= 1)
        & (df.loc[:, npi_key] == 0)
        & (df.loc[:, nmu_key] == 0)
        & (df.loc[:, ng_key] == 0)
    )


def SIGNAL_MASK_FN(df):
    """Signal mask dispatching on truth source.

    evt_df:  true_signal1p | true_signalNp (DLP particle-level truth matching).
    mcnu_df: GENIE nue CC in FV with DLP-compatible topology cuts (no electron
             KE threshold; proton KE>40 MeV; no pi/mu/gamma above threshold).
    """
    if _DLP_SIGNAL1P_KEY in df.columns:
        mask = df[_DLP_SIGNAL1P_KEY] == 1
        if _DLP_SIGNAL_NP_KEY in df.columns:
            mask = mask | (df[_DLP_SIGNAL_NP_KEY] == 1)
        return mask
    return _mcnu_signal_mask_dlp_phase_space(df)
