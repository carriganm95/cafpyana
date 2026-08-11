import numpy as np
import pandas as pd
import sys, os as _os
sys.path.append(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
from makedf.util import *


DETECTOR = "SBND_Gen1"
# DETECTOR = "SBND"

# Cathode inset (cm) for per-TPC x-fiducial; matches reco helpers in
# ``makedf.selections.event_contained_per_tpc`` and selected-events drivers.
PER_TPC_INCATHODE_CM = 10

# ==== definitions for event categories ===-

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

# ==== functions to get breadkdowns of event categories ====
def get_nu_cosmics_category(df, ret_cuts=False, print_summary=False):
    cut_cosmic = IsCosmic(df)
    cut_nu_outfv = IsNuOutFV(df)
    cut_nu_infv = IsNuInFV(df)
    cuts = [cut_cosmic, cut_nu_outfv, cut_nu_infv]
    return cuts

def get_pdg_category(df, ret_cuts=False, print_summary=False):
    # reco track truth pdg category
    cut_muon = (np.abs(df.pfp.trk.truth.p.pdg) == 13)
    cut_proton = (df.pfp.trk.truth.p.pdg == 2212)
    cut_pion = (np.abs(df.pfp.trk.truth.p.pdg) == 211)
    # cut_pion0 = (df.pfp.trk.truth.p.pdg == 111)
    cut_other = (~cut_muon & ~cut_proton & ~cut_pion)

    cuts = [cut_muon, cut_proton, cut_pion, cut_other]
    return cuts[::-1]

pdg_labels = [r"$e^{\pm}$", r"$\gamma$", r"$\mu^{\pm}$", r"$p$", r"$\pi^{\pm}$", r"Other"]
pdg_colors = ["C0", "C1", "C2", "C3", "C4", "C5"]

def get_pdg_category_spine(df):
    cut_ele = (np.abs(df.rec.dlp_true.particles.pdg_code) == 11)
    cut_photon = (df.rec.dlp_true.particles.pdg_code == 22)
    cut_muon = (np.abs(df.rec.dlp_true.particles.pdg_code) == 13)
    cut_proton = (df.rec.dlp_true.particles.pdg_code == 2212)
    cut_pion = (np.abs(df.rec.dlp_true.particles.pdg_code) == 211)
    cut_other = (~cut_ele & ~cut_photon & ~cut_muon & ~cut_proton & ~cut_pion)
    cuts = [cut_ele, cut_photon, cut_muon, cut_proton, cut_pion, cut_other]
    return cuts[::-1]

def get_topo_category(
    df,
    ret_cuts=False,
    print_summary=False,
    detector=DETECTOR,
    signal_truth_fv="per_tpc",
):
    cut_cosmic = IsCosmic(df)
    # cut_nu_outfv = IsNuOutFV(df)
    # cut_nu_infv_nu_other = IsNuInFV_NuOther(df)
    cut_nu_other = (IsNuOutFV(df) | IsNuInFV_NuOther(df))
    cut_nu_infv_numu_nc = IsNuInFV_NumuNC(df)
    cut_nu_infv_numu_cc_other = IsNuInFV_NumuCC_Other(
        df, detector=detector, signal_truth_fv=signal_truth_fv
    )
    cut_nu_infv_numu_cc_np0pi = IsNuInFV_NumuCC_Np0pi(df)
    cut_nu_infv_numu_cc_1p0pi = IsNuInFV_NumuCC_1p0pi(
        df, detector=detector, signal_truth_fv=signal_truth_fv
    )

    # assert there's no overlap between the categories, AND that all categories are covered just in case i messed something up...
    assert (cut_cosmic & cut_nu_other & cut_nu_infv_numu_nc & cut_nu_infv_numu_cc_other & cut_nu_infv_numu_cc_np0pi & cut_nu_infv_numu_cc_1p0pi).sum() == 0
    assert (cut_cosmic | cut_nu_other | cut_nu_infv_numu_nc | cut_nu_infv_numu_cc_other | cut_nu_infv_numu_cc_np0pi | cut_nu_infv_numu_cc_1p0pi).sum() == len(df)

    # category 1 NEEDS TO BE THE SIGNAL MODE
    nuint_categ = pd.Series(10, index=df.index)
    nuint_categ[cut_cosmic] = -1  # not nu
    nuint_categ[cut_nu_other] = 0  # nu out of FV
    nuint_categ[cut_nu_infv_numu_cc_1p0pi] = 1    # nu in FV, signal
    nuint_categ[cut_nu_infv_numu_cc_np0pi] = 2  # 1mu, 0cpi, Np, 0pi0
    nuint_categ[cut_nu_infv_numu_cc_other] = 3  # 1mu, Ncpi, 0pi0
    nuint_categ[cut_nu_infv_numu_nc] = 4  # nu in FV, numu NC
    # nuint_categ[cut_nu_infv_nu_other] = 5  # nu in FV, other

    if print_summary:
        print(nuint_categ.value_counts())

    if ret_cuts:
        cuts = [cut_cosmic, cut_nu_other, cut_nu_infv_numu_nc, 
                cut_nu_infv_numu_cc_other, cut_nu_infv_numu_cc_np0pi, cut_nu_infv_numu_cc_1p0pi]
        return cuts

    return nuint_categ

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

    nuint_categ = pd.Series(10, index=df.index)
    nuint_categ[df.rec.dlp_true.true_signal1p == 1] = 0
    nuint_categ[df.rec.dlp_true.true_signalNp == 1] = 1
    nuint_categ[df.rec.dlp_true.bkgd_oofv == 1] = 2
    nuint_categ[df.rec.dlp_true.bkgd_oops == 1] = 3
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
            df.rec.dlp_true.bkgd_oofv == 1,        # 2
            df.rec.dlp_true.bkgd_oops == 1,        # 3
            df.rec.dlp_true.bkgd_pi == 1,          # 4
            df.rec.dlp_true.bkgd_mu == 1,          # 5
            df.rec.dlp_true.bkgd_photon == 1,      # 6
            df.rec.dlp_true.bkgd_nueOther == 1,    # 7
            df.rec.dlp_true.bkgd_numu == 1,        # 8
            df.rec.dlp_true.bkgd_nc == 1,          # 9
            df.rec.dlp_true.bkgd_other == 1,       # 10
        ]

    return nuint_categ



def get_genie_category(df, ret_cuts=False, print_summary=False):
    cut_cosmic = IsCosmic(df)
    # cut_nu_outfv = IsNuOutFV(df)
    # cut_nu_infv_nu_other = IsNuInFV_NuOther(df)
    cut_nu_other = (IsNuOutFV(df) | IsNuInFV_NuOther(df))
    cut_nu_infv_numu_nc = IsNuInFV_NumuNC(df)
    # cut_nu_infv_numu_coh = IsNuInFV_NumuCC_COH(evtdf)
    cut_nu_infv_numu_othermode = (IsNuInFV_NumuCC_OtherMode(df) | IsNuInFV_NumuCC_DIS(df))
    # cut_nu_infv_numu_cc_dis = IsNuInFV_NumuCC_DIS(df)
    cut_nu_infv_numu_cc_res = IsNuInFV_NumuCC_RES(df)
    cut_nu_infv_numu_cc_me = IsNuInFV_NumuCC_MEC(df)
    cut_nu_infv_numu_cc_qe = IsNuInFV_NumuCC_QE(df)

    # assert (cut_cosmic & cut_nu_outfv & cut_nu_infv_nu_other & cut_nu_infv_numu_nc & cut_nu_infv_numu_othermode & cut_nu_infv_numu_cc_dis & cut_nu_infv_numu_cc_res & cut_nu_infv_numu_cc_me & cut_nu_infv_numu_cc_qe).sum() == 0
    # assert (cut_cosmic | cut_nu_outfv | cut_nu_infv_nu_other | cut_nu_infv_numu_nc | cut_nu_infv_numu_othermode | cut_nu_infv_numu_cc_dis | cut_nu_infv_numu_cc_res | cut_nu_infv_numu_cc_me | cut_nu_infv_numu_cc_qe).sum() == len(df)

    assert (cut_cosmic & cut_nu_other & cut_nu_infv_numu_nc & cut_nu_infv_numu_othermode & cut_nu_infv_numu_cc_res & cut_nu_infv_numu_cc_me & cut_nu_infv_numu_cc_qe).sum() == 0
    assert (cut_cosmic | cut_nu_other | cut_nu_infv_numu_nc | cut_nu_infv_numu_othermode | cut_nu_infv_numu_cc_res | cut_nu_infv_numu_cc_me | cut_nu_infv_numu_cc_qe).sum() == len(df)

    genie_categ = pd.Series(10, index=df.index)
    genie_categ[cut_cosmic] = -1  # not nu
    genie_categ[cut_nu_other] = 0  # nu out of FV
    genie_categ[cut_nu_infv_numu_cc_qe] = 1  # nu in FV, QE
    genie_categ[cut_nu_infv_numu_cc_me] = 2  # nu in FV, MEC
    genie_categ[cut_nu_infv_numu_cc_res] = 3  # nu in FV, RES
    # genie_categ[cut_nu_infv_numu_cc_dis] = 4  # nu in FV, DIS
    genie_categ[cut_nu_infv_numu_othermode] = 5  # nu in FV, other mode
    genie_categ[cut_nu_infv_numu_nc] = 6  # nu in FV, numu NC
    # genie_categ[cut_nu_infv_nu_other] = 7  # nu in FV, other

    if print_summary:
        print(genie_categ.value_counts())

    if ret_cuts:
        cuts = [cut_cosmic, cut_nu_other, cut_nu_infv_numu_nc, 
                # cut_nu_infv_numu_othermode, cut_nu_infv_numu_cc_dis, cut_nu_infv_numu_cc_res, 
                cut_nu_infv_numu_othermode, cut_nu_infv_numu_cc_res, 
                cut_nu_infv_numu_cc_me, cut_nu_infv_numu_cc_qe]
        return cuts

    return genie_categ

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


# --- colors & labelsfor plotting ---
# nu / cosmic breakdown
nu_cosmics_labels = ["Cosmic", r"Out-FV $\nu$", r"FV $\nu$"]
nu_cosmics_colors = ["gray", "C0", "C1"]

# signal / backgroundtopology breakdown
# the signal mode code MUST be the first item in the list for all the code below to work
topology_list = [0, 1, 2, 3, 4, 5,
                6, 7, 8, 9, 10]

topology_labels = [r'$\nu_e$ CC 1p0$\pi$',
                    r'$\nu_e$ CC Np0$\pi$', 
                    r'$\nu_e$ CC OOPS',
                    r'$\nu_e$ CC OOFV',
                    r'$\nu_e$ CC $\pi^{\pm}$',
                    r'$\nu_e$ CC $\mu^{\pm}$',
                    r'$\nu_e$ CC $\gamma$',
                    r'$\nu_e$ CC Other',
                    r'$\nu_{\mu}$ CC',
                    r'NC',
                    r'Other']

topology_colors = ['#00082E', 
                    '#001261', 
                    '#033E7D', 
                    '#1E6F9D', 
                    '#71A8C4', 
                    '#C9DDE7', 
                    '#EACEBD', 
                    '#D39774', 
                    '#BE6533', 
                    '#8B2706', 
                    '#590008', 
                    '#000000'] 

# --- GENIE interaction mode breakdown ---
genie_mode_list = [0, # CCQE
                   1, # CCRES 
                   2, # CCDIS
                   3, # CCMEC
                   4, # CC Other
                   -1, # NC
                   -2  # Cosmic
                   ]
# genie_mode_labels = [r'$\nu_{\mu}$ CC QE', r'$\nu_{\mu}$ CC MEC', r'$\nu_{\mu}$ CC RES', r'$\nu_{\mu}$ CC SIS/DIS', r'$\nu_{\mu}$ CC Other', 
genie_mode_labels = [ r'$\nu_e$ CC QE',
                        r'$\nu_e$ CC RES',
                        r'$\nu_e$ CC DIS', 
                        r'$\nu_e$ CC MEC', 
                        r'$\nu_e$ CC Other', 
                        r"$\nu$ NC", 
                        "Cosmic"]


# genie_mode_colors = ["#9b5580", "#390C1E", "#2c7c94", "#D88A3B", "#BFB17C", 
genie_mode_colors = ["#9b5580", "#390C1E", "#2c7c94", "#D88A3B", 
                     "darkgreen", 
                     "crimson", 
                    #  "sienna",
                     "gray"] 

# --- GENIE SB interaction mode breakdown ---
genie_sb_mode_labels = [r'$\nu_{\mu}$ CC QE', r'$\nu_{\mu}$ CC QE',
                     r'$\nu_{\mu}$ CC MEC', r'$\nu_{\mu}$ CC MEC', 
                     r'$\nu_{\mu}$ CC RES', r'$\nu_{\mu}$ CC RES', 
                     r'$\nu_{\mu}$ CC Other', r'$\nu_{\mu}$ CC Other',
                     r"$\nu$ NC", 
                     r"Other $\nu$", 
                     "Cosmic"]

genie_sb_mode_colors = ["#9b5580", "#9b5580",
                        "#390C1E", "#390C1E",
                        "#2c7c94", "#2c7c94",
                        "#D88A3B", "#D88A3B",
                        "darkgreen", 
                        "crimson", 
                        "gray"] 


# --- GiBUU interaction mode breakdown ---
gibuu_mode_labels = [r'$\nu_{\mu}$ CC QE', r'$\nu_{\mu}$ CC QE (2p2h)', r'$\nu_{\mu}$ CC RES', r'$\nu_{\mu}$ CC DIS', r'$\nu_{\mu}$ CC Other', 
                     r"$\nu$ NC", r"FV other $\nu$", r"Out-FV $\nu$", "Cosmic"]
gibuu_mode_colors = ["#9b5580", "#390C1E", "#2c7c94", "#D88A3B", "#BFB17C", 
                     "darkgreen", "crimson", "sienna","gray"] 


def get_category_cuts(breakdown_type, df, ret_cuts=False):
    if breakdown_type == "nu_cosmics":
        labels = nu_cosmics_labels
        colors = nu_cosmics_colors
        cuts = get_nu_cosmics_category(df, ret_cuts=ret_cuts)

    elif breakdown_type == "topology":
        labels = topology_labels[::-1]
        colors = topology_colors[::-1]
        cuts = get_topo_category_nueNp0Pi(df, ret_cuts=ret_cuts)

    elif breakdown_type == "genie":
        labels = genie_mode_labels[::-1]
        colors = genie_mode_colors[::-1]
        cuts = get_genie_category_spine(df, ret_cuts=ret_cuts)
    
    # TODO: GiBUU breakdown

    else:
        raise ValueError("Invalid breakdown_type: %s, please choose between [topology, genie, or genie_sb]" % breakdown_type)

    return cuts, labels, colors