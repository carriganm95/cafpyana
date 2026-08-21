from makedf.makedf import *
from pyanalib.pandas_helpers import *
from makedf.util import *
from pyanalib.variable_calculator import get_tki_spine, get_tki_spine_lp, get_lp_open_angle_spine, get_lepton_beam_angle_spine
from analysis_village.nueNp0Pi.config.settings import (
    DETECTOR, PER_TPC_INCATHODE_CM,
    ELE_SOFTMAX_TH, ELE_PRIMARY_TH, P_SOFTMAX_TH, ELE_VTXDIST_TH, ELE_DEDX_TH,
    ELE_KE_TH, P_KE_TH, MU_KE_TH, PI_KE_TH, GAM_KE_TH
)

## == For additional column in mcdf with primary particle multiplicities
## ==== "<column name>": ["<particle name>", <KE cut in GeV>]
## ==== <particle name> is used to collect PID and mass from the "PDG" dictionary
TRUE_KE_THRESHOLDS = {"nmu_25MeV": ["muon", MU_KE_TH],
                      "np_40MeV": ["proton", P_KE_TH],
                      "npi_25MeV": ["pipm", PI_KE_TH],
                      "ne_500MeV": ["electron", ELE_KE_TH],
                      'ng_100MeV': ["gamma", GAM_KE_TH]
                      }

def make_mcnudf_nuecc(f,**args):
    mcdf = make_mcnudf(f,**args)
    # # drop mcdf columns not relevant for this analysis
    # if 'mu'  in list(zip(*list(mcdf.columns)))[0]:  mcdf = mcdf.drop('mu', axis=1,level=0)
    # if 'p'   in list(zip(*list(mcdf.columns)))[0]:  mcdf = mcdf.drop('p',  axis=1,level=0)
    # if 'cpi' in list(zip(*list(mcdf.columns)))[0]:  mcdf = mcdf.drop('cpi',axis=1,level=0)

    # Add in primary particle info
    mcprimdf = loadbranches(f["recTree"], mcprimbranches)
    while mcprimdf.columns.nlevels > 2:
        mcprimdf.columns = mcprimdf.columns.droplevel(0)

    mcprimdf.index = mcprimdf.index.rename(mcdf.index.names[:2] + mcprimdf.index.names[2:])

    # particle counts w/ threshold
    for identifier, (particle, threshold) in TRUE_KE_THRESHOLDS.items():
        this_KE = mcprimdf[np.abs(mcprimdf.pdg)==PDG[particle][0]].genE - PDG[particle][2]
        mcdf = multicol_add(mcdf, ((np.abs(mcprimdf.pdg)==PDG[particle][0]) & (this_KE > threshold)).groupby(level=[0,1]).sum().rename(identifier))

    return mcdf

def make_nueccdf_mc_wgt(f):
    df = make_nueccdf_mc(f,include_weights=True)
    return df

def make_nueccdf_mc(f, include_weights=False,multisim_nuniv=100,slim=True):
    
    slcdf = make_nueccdf(f)
    mcdf = make_mcnudf_nuecc(f,include_weights=include_weights,multisim_nuniv=multisim_nuniv,slim=slim)
    mcdf.columns = pd.MultiIndex.from_tuples([tuple(["slc", "truth"] + list(c)) for c in mcdf.columns])
    df = multicol_merge(slcdf.reset_index(), 
                        mcdf.reset_index(),
                        left_on=[('entry', '', '', '', '', ''), 
                                ('slc', 'tmatch', 'idx', '', '', '')], 
                        right_on=[('entry', '', '', '', '', ''), 
                                ('rec.mc.nu..index', '', '')], 
                        how="left")
    df = df.set_index(slcdf.index.names, verify_integrity=True)
    return df

def make_nueccdf_data(f):
    slcdf = make_nueccdf(f)
    # drop truth cols for data
    slcdf = slcdf.drop('tmatch', axis=1,level=1) # slc level
    slcdf = slcdf.drop('truth',  axis=1,level=2) # pfp level
    
    ## keep the only relevant column (for now)
    framedf = make_framedf(f)[['frameApplyAtCaf']]
    
    df = multicol_merge(slcdf.reset_index(), 
                        framedf.reset_index(),
                        left_on=[('entry', '', '', '', '', '')],
                        right_on=[('entry', '', '', '', '', '')], 
                        how="left")
    df = df.set_index(slcdf.index.names, verify_integrity=True)
    return df

def broadcast_to_particles(interaction_mask, df):
    """Broadcast an interaction-level mask to particle level."""
    s = interaction_mask.reindex(df.index.droplevel(df.index.names[-1]))
    s.index = df.index
    return s

def get_highest_ke_particle_idx(pfpdf, pdg, truth=False):
    """Return the particles..index of the highest-KE particle with the given pdg
    per interaction, as a Series indexed by the interaction-level index.
    Uses dlp_true particles when truth=True, dlp particles otherwise."""
    branch = 'dlp_true' if truth else 'dlp'
    ke_col  = pad_column_name(('rec', branch, 'particles', 'ke'),       pfpdf)
    pdg_col = pad_column_name(('rec', branch, 'particles', 'pdg_code'), pfpdf)
    primary_col = pad_column_name(('rec', branch, 'particles', 'is_primary'), pfpdf)
    int_levels = list(range(pfpdf.index.nlevels - 1))
    empty = pd.Series(dtype=float, index=pd.MultiIndex.from_tuples([], names=pfpdf.index.names[:len(int_levels)]))
    filtered = pfpdf[(abs(pfpdf[pdg_col]) == pdg) & (pfpdf[ke_col] > 0) & (pfpdf[primary_col] == True)]
    if filtered.empty:
        return empty
    best_idx = filtered[ke_col].groupby(level=int_levels).idxmax()
    return best_idx.apply(lambda x: x[pfpdf.index.nlevels - 1])

def get_second_highest_ke_particle_idx(pfpdf, pdg, truth=False):
    """Return the particles..index of the second highest-KE particle with the given pdg
    per interaction, as a Series indexed by the interaction-level index.
    Interactions with fewer than 2 matching particles are excluded.
    Uses dlp_true particles when truth=True, dlp particles otherwise."""
    branch = 'dlp_true' if truth else 'dlp'
    ke_col  = pad_column_name(('rec', branch, 'particles', 'ke'),       pfpdf)
    pdg_col = pad_column_name(('rec', branch, 'particles', 'pdg_code'), pfpdf)
    primary_col = pad_column_name(('rec', branch, 'particles', 'is_primary'), pfpdf)
    int_levels = list(range(pfpdf.index.nlevels - 1))
    empty = pd.Series(dtype=float, index=pd.MultiIndex.from_tuples([], names=pfpdf.index.names[:len(int_levels)]))
    filtered = pfpdf[(abs(pfpdf[pdg_col]) == pdg) & (pfpdf[ke_col] > 0) & (pfpdf[primary_col] == True)]
    if filtered.empty:
        return empty
    ranks = filtered[ke_col].groupby(level=int_levels).rank(method='first', ascending=False)
    second = filtered[ke_col][ranks == 2]
    if second.empty:
        return empty
    best_idx = second.groupby(level=int_levels).idxmax()
    return best_idx.apply(lambda x: x[pfpdf.index.nlevels - 1])

def _remap_true_to_reco(series, slcdf):
    """Map a Series indexed by [entry, rec.dlp_true..index] to slcdf's [entry, rec.dlp..index].

    Uses the reco↔true interaction matching stored as a column in slcdf from make_spine_int_df.
    Left-joins so unmatched reco interactions (no true match) get NaN."""
    true_int_col = pad_column_name(("rec.dlp_true..index",), slcdf)
    mapping = slcdf[[true_int_col]].reset_index()
    mapping.columns = pd.Index(["entry", "reco_int", "true_int"])
    vals = series.reset_index()
    vals.columns = pd.Index(["entry", "true_int", "val"])
    result = (
        mapping.merge(vals, on=["entry", "true_int"], how="left")
               .set_index(["entry", "reco_int"])["val"]
    )
    result.index.names = slcdf.index.names
    return result


def _get_highest_ke_true_primary_idx(spinetpart_df, pdg):
    """Return the true particle index of the highest-KE primary particle with the given PDG
    per true interaction. Searches all true particles regardless of reco matching.
    Returns a Series indexed by [entry, rec.dlp_true..index]."""
    ke_col      = pad_column_name(('rec', 'dlp_true', 'particles', 'ke'),         spinetpart_df)
    pdg_col     = pad_column_name(('rec', 'dlp_true', 'particles', 'pdg_code'),   spinetpart_df)
    primary_col = pad_column_name(('rec', 'dlp_true', 'particles', 'is_primary'), spinetpart_df)
    int_levels  = list(range(spinetpart_df.index.nlevels - 1))
    empty = pd.Series(dtype=float,
                      index=pd.MultiIndex.from_tuples([], names=spinetpart_df.index.names[:len(int_levels)]))
    filtered = spinetpart_df[(abs(spinetpart_df[pdg_col]) == pdg) &
                             (spinetpart_df[ke_col] > 0) &
                             (spinetpart_df[primary_col] == True)]
    if filtered.empty:
        return empty
    best_idx = filtered[ke_col].groupby(level=int_levels).idxmax()
    return best_idx.apply(lambda x: x[spinetpart_df.index.nlevels - 1])


def _get_second_highest_ke_true_primary_idx(spinetpart_df, pdg):
    """Return the true particle index of the second highest-KE primary particle with the given PDG
    per true interaction. Searches all true particles regardless of reco matching.
    Returns a Series indexed by [entry, rec.dlp_true..index]."""
    ke_col      = pad_column_name(('rec', 'dlp_true', 'particles', 'ke'),         spinetpart_df)
    pdg_col     = pad_column_name(('rec', 'dlp_true', 'particles', 'pdg_code'),   spinetpart_df)
    primary_col = pad_column_name(('rec', 'dlp_true', 'particles', 'is_primary'), spinetpart_df)
    int_levels  = list(range(spinetpart_df.index.nlevels - 1))
    empty = pd.Series(dtype=float,
                      index=pd.MultiIndex.from_tuples([], names=spinetpart_df.index.names[:len(int_levels)]))
    filtered = spinetpart_df[(abs(spinetpart_df[pdg_col]) == pdg) &
                             (spinetpart_df[ke_col] > 0) &
                             (spinetpart_df[primary_col] == True)]
    if filtered.empty:
        return empty
    ranks = filtered[ke_col].groupby(level=int_levels).rank(method='first', ascending=False)
    second = filtered[ke_col][ranks == 2]
    if second.empty:
        return empty
    best_idx = second.groupby(level=int_levels).idxmax()
    return best_idx.apply(lambda x: x[spinetpart_df.index.nlevels - 1])


def make_nueNp0Pi_df(f):
    det = loadbranches(f["recTree"], ["rec.hdr.det"]).rec.hdr.det
    if (1 == det.unique()):
        DETECTOR = "SBND"
    else:
        DETECTOR = "ICARUS"

    assert DETECTOR == "SBND"

    pfpdf = make_spine_part_df(f)
    slcdf = make_spine_int_df(f)
    hdrdf = make_mchdrdf(f)

    mcdf = make_mcnudf_nuecc(f)

    # Load all true particles directly — used for truth computations to avoid the
    # reco-matching bias in pfpdf (which only contains true particles matched to a reco particle).
    spinetpart_df = loadbranches(f["recTree"], spinetpart_branches)
    rename_to_XYZ(spinetpart_df, ["momentum", "end_point", "start_point", "start_dir", "end_dir", "vertex"])

    ## add candidate particle indices to slcdf while it is still interaction-level (2-level index)
    ## so that multicol_merge aligns cleanly; the subsequent right merge with pfpdf
    ## broadcasts these interaction-level values down to every particle row
    nlevel = slcdf.columns.nlevels
    for pdg, label in [(11, 'primele'), (13, 'primmu'), (211, 'primpi'), (2212, 'primpr'), (22, 'primph')]:
        idx = get_highest_ke_particle_idx(pfpdf, pdg)
        idx.index.names = slcdf.index.names
        col = tuple(['rec', 'dlp', label] + [''] * (nlevel - 3))
        idx_df = idx.rename(col).to_frame()
        slcdf = multicol_merge(slcdf, idx_df, left_index=True, right_index=True, how='left')

    # Pre-compute truth candidate indices once from spinetpart_df; reused below for both
    # storing in slcdf and broadcasting into spinetpart_df particle space for masks/kinematics.
    _prim_ele_idx   = _get_highest_ke_true_primary_idx(spinetpart_df, 11)
    _prim_mu_idx    = _get_highest_ke_true_primary_idx(spinetpart_df, 13)
    _prim_pi_idx    = _get_highest_ke_true_primary_idx(spinetpart_df, 211)
    _prim_pr_idx    = _get_highest_ke_true_primary_idx(spinetpart_df, 2212)
    _prim_ph_idx    = _get_highest_ke_true_primary_idx(spinetpart_df, 22)
    _subprim_pr_idx = _get_second_highest_ke_true_primary_idx(spinetpart_df, 2212)

    for idx, label in [(_prim_ele_idx, 'true_primele'), (_prim_mu_idx, 'true_primmu'),
                       (_prim_pi_idx, 'true_primpi'),   (_prim_pr_idx, 'true_primpr'),
                       (_prim_ph_idx, 'true_primph')]:
        mapped = _remap_true_to_reco(idx, slcdf)
        mapped.index.names = slcdf.index.names
        col = tuple(['rec', 'dlp_true', label] + [''] * (nlevel - 3))
        slcdf = multicol_merge(slcdf, mapped.rename(col).to_frame(), left_index=True, right_index=True, how='left')

    # get subleading proton
    idx = get_second_highest_ke_particle_idx(pfpdf, 2212)
    idx.index.names = slcdf.index.names
    col = tuple(['rec', 'dlp', 'subprimpr'] + [''] * (nlevel - 3))
    idx_df = idx.rename(col).to_frame()
    slcdf = multicol_merge(slcdf, idx_df, left_index=True, right_index=True, how='left')

    mapped = _remap_true_to_reco(_subprim_pr_idx, slcdf)
    mapped.index.names = slcdf.index.names
    col = tuple(['rec', 'dlp_true', 'true_subprimpr'] + [''] * (nlevel - 3))
    slcdf = multicol_merge(slcdf, mapped.rename(col).to_frame(), left_index=True, right_index=True, how='left')

    # expand to particle level temporarily to compute truth category masks
    _ptmp = multicol_merge(slcdf, pfpdf, left_index=True, right_index=True, how="right", validate="one_to_many")

    # create truth categories — all masks reduced back to event level via groupby
    int_levels = list(range(_ptmp.index.nlevels - 1))  # all levels except particle (for reco)
    particle_idx       = pd.Series(_ptmp.index.get_level_values(_ptmp.index.names[-1]), index=_ptmp.index)

    # reco particle row selectors
    primele_rows_r       = particle_idx == _ptmp.rec.dlp.primele
    primproton_rows_r    = particle_idx == _ptmp.rec.dlp.primpr
    subprimproton_rows_r = particle_idx == _ptmp.rec.dlp.subprimpr
    primmuon_rows_r      = particle_idx == _ptmp.rec.dlp.primmu
    primpion_rows_r      = particle_idx == _ptmp.rec.dlp.primpi
    primphoton_rows_r    = particle_idx == _ptmp.rec.dlp.primph

    # truth particle row selectors — operate directly in spinetpart_df (all true particles,
    # not just those matched to a reco particle)
    _true_int_levels = list(range(spinetpart_df.index.nlevels - 1))  # [0, 1]
    _particle_idx_t = pd.Series(spinetpart_df.index.get_level_values(-1), index=spinetpart_df.index)

    def _bc(idx_series):
        """Broadcast a true-interaction-level Series to spinetpart_df's particle level."""
        result = idx_series.reindex(spinetpart_df.index.droplevel(-1))
        result.index = spinetpart_df.index
        return result

    primele_rows_t       = _particle_idx_t == _bc(_prim_ele_idx)
    primproton_rows_t    = _particle_idx_t == _bc(_prim_pr_idx)
    subprimproton_rows_t = _particle_idx_t == _bc(_subprim_pr_idx)
    primmuon_rows_t      = _particle_idx_t == _bc(_prim_mu_idx)
    primpion_rows_t      = _particle_idx_t == _bc(_prim_pi_idx)
    primphoton_rows_t    = _particle_idx_t == _bc(_prim_ph_idx)

    # interaction-level truth properties from the reco↔true interaction match already in slcdf.
    # .fillna(False).astype(bool) guards against float64 dtype (introduced when slcdf has reco
    # interactions with no matched true interaction, causing NaN in the joined columns).
    nu_mask         = (slcdf.rec.dlp_true.nu_id >= 0).fillna(False).astype(bool)
    cc_mask         = (slcdf.rec.dlp_true.current_type == 0).fillna(False).astype(bool)
    nupdg_mask      = (abs(slcdf.rec.dlp_true.pdg_code) == 12).fillna(False).astype(bool)
    fiducial_mask_t = (slcdf.rec.dlp_true.is_fiducial == 1).fillna(False).astype(bool)

    # spinetpart_df column name helpers
    _stp_ke        = pad_column_name(('rec', 'dlp_true', 'particles', 'ke'),           spinetpart_df)
    _stp_csda      = pad_column_name(('rec', 'dlp_true', 'particles', 'csda_ke'),      spinetpart_df)
    _stp_mcs       = pad_column_name(('rec', 'dlp_true', 'particles', 'mcs_ke'),       spinetpart_df)
    _stp_p         = pad_column_name(('rec', 'dlp_true', 'particles', 'p'),            spinetpart_df)
    _stp_contained = pad_column_name(('rec', 'dlp_true', 'particles', 'is_contained'), spinetpart_df)
    _stp_pdg       = pad_column_name(('rec', 'dlp_true', 'particles', 'pdg_code'),     spinetpart_df)
    _stp_valid     = pad_column_name(('rec', 'dlp_true', 'particles', 'is_valid'),     spinetpart_df)

    def _reduce_t(mask):
        """Reduce a boolean particle-level mask in spinetpart_df to reco-interaction level via any()."""
        return _remap_true_to_reco(mask.groupby(level=_true_int_levels).any(), slcdf).fillna(False).astype(bool)

    def _first_t(vals):
        """Reduce particle-level values in spinetpart_df to reco-interaction level via first()."""
        return _remap_true_to_reco(vals.groupby(level=_true_int_levels).first(), slcdf)

    ele_mask_t           = _reduce_t(primele_rows_t    & (spinetpart_df[_stp_ke] >= ELE_KE_TH))
    proton_mask_t        = _reduce_t(primproton_rows_t & (spinetpart_df[_stp_ke] >= P_KE_TH ))
    subprimproton_mask_t = _reduce_t(subprimproton_rows_t & (spinetpart_df[_stp_ke] >= P_KE_TH ))
    muon_mask_t          = _reduce_t(primmuon_rows_t   & (spinetpart_df[_stp_ke] >= MU_KE_TH ))
    pion_mask_t          = _reduce_t(primpion_rows_t   & (spinetpart_df[_stp_ke] >= PI_KE_TH ))
    photon_mask_t        = _reduce_t(primphoton_rows_t & (spinetpart_df[_stp_ke] >= GAM_KE_TH))

    ele_mask_noKE_t           = _reduce_t(primele_rows_t)
    proton_mask_noKE_t        = _reduce_t(primproton_rows_t)
    subprimproton_mask_noKE_t = _reduce_t(subprimproton_rows_t)
    muon_mask_noKE_t          = _reduce_t(primmuon_rows_t)
    pion_mask_noKE_t          = _reduce_t(primpion_rows_t)
    photon_mask_noKE_t        = _reduce_t(primphoton_rows_t)

    fiducial_mask_r = (_ptmp.rec.dlp.is_fiducial == 1).groupby(level=int_levels).first()
    flash_match_r = (_ptmp.rec.dlp.is_flash_matched == 1).groupby(level=int_levels).first()
    containment_r = (_ptmp.rec.dlp.is_contained == 1).groupby(level=int_levels).first()
    muon_mask_r = (primmuon_rows_r & (_ptmp.rec.dlp.particles.ke >= MU_KE_TH)).groupby(level=int_levels).any()
    pion_mask_r = (primpion_rows_r & (_ptmp.rec.dlp.particles.ke >= PI_KE_TH)).groupby(level=int_levels).any()
    proton_mask_r = (primproton_rows_r & (_ptmp.rec.dlp.particles.ke >= P_KE_TH)).groupby(level=int_levels).any()
    subprimproton_mask_r = (subprimproton_rows_r & (_ptmp.rec.dlp.particles.ke >= P_KE_TH)).groupby(level=int_levels).any()
    photon_mask_r = (primphoton_rows_r & (_ptmp.rec.dlp.particles.ke >= GAM_KE_TH)).groupby(level=int_levels).any()
    ele_mask_r = (primele_rows_r & (_ptmp.rec.dlp.particles.ke >= ELE_KE_TH)).groupby(level=int_levels).any()
    ele_softmax_r = (primele_rows_r & (_ptmp.rec.dlp.particles.pid_scores['I1'] >= ELE_SOFTMAX_TH)).groupby(level=int_levels).any()
    ele_primary_r = (primele_rows_r & (_ptmp.rec.dlp.particles.primary_scores['I1'] >= ELE_PRIMARY_TH)).groupby(level=int_levels).any()
    proton_softmax_r = (primproton_rows_r & (_ptmp.rec.dlp.particles.pid_scores['I4'] >= P_SOFTMAX_TH)).groupby(level=int_levels).any()
    vertex_distance_r = (primele_rows_r & (_ptmp.rec.dlp.particles.vertex_distance < ELE_VTXDIST_TH)).groupby(level=int_levels).any()
    ele_dedx_r = (primele_rows_r & (_ptmp.rec.dlp.particles.start_dedx < ELE_DEDX_TH)).groupby(level=int_levels).any()

    ele_energy_r = _ptmp.rec.dlp.particles.ke.where(primele_rows_r).groupby(level=int_levels).first()
    ele_dedx_r = _ptmp.rec.dlp.particles.start_dedx.where(primele_rows_r).groupby(level=int_levels).first()
    ele_softmax_r = _ptmp.rec.dlp.particles.pid_scores['I1'].where(primele_rows_r).groupby(level=int_levels).first()
    ele_primary_r = _ptmp.rec.dlp.particles.primary_scores['I1'].where(primele_rows_r).groupby(level=int_levels).first()
    ele_vertex_distance_r = _ptmp.rec.dlp.particles.vertex_distance.where(primele_rows_r).groupby(level=int_levels).first()
    muon_energy_r = _ptmp.rec.dlp.particles.ke.where(primmuon_rows_r).groupby(level=int_levels).first()
    pion_energy_r = _ptmp.rec.dlp.particles.ke.where(primpion_rows_r).groupby(level=int_levels).first()
    proton_energy_r = _ptmp.rec.dlp.particles.ke.where(primproton_rows_r).groupby(level=int_levels).first()
    muon_energy_csda_r = _ptmp.rec.dlp.particles.csda_ke.where(primmuon_rows_r).groupby(level=int_levels).first()
    pion_energy_csda_r = _ptmp.rec.dlp.particles.csda_ke.where(primpion_rows_r).groupby(level=int_levels).first()
    proton_energy_csda_r = _ptmp.rec.dlp.particles.csda_ke.where(primproton_rows_r).groupby(level=int_levels).first()
    muon_energy_mcs_r = _ptmp.rec.dlp.particles.mcs_ke.where(primmuon_rows_r).groupby(level=int_levels).first()
    pion_energy_mcs_r = _ptmp.rec.dlp.particles.mcs_ke.where(primpion_rows_r).groupby(level=int_levels).first()
    proton_energy_mcs_r = _ptmp.rec.dlp.particles.mcs_ke.where(primproton_rows_r).groupby(level=int_levels).first()
    muon_contained_r = _ptmp.rec.dlp.particles.is_contained.where(primmuon_rows_r).groupby(level=int_levels).first()
    pion_contained_r = _ptmp.rec.dlp.particles.is_contained.where(primpion_rows_r).groupby(level=int_levels).first()
    proton_contained_r = _ptmp.rec.dlp.particles.is_contained.where(primproton_rows_r).groupby(level=int_levels).first()
    ele_contained_r = _ptmp.rec.dlp.particles.is_contained.where(primele_rows_r).groupby(level=int_levels).first()
    photon_contained_r = _ptmp.rec.dlp.particles.is_contained.where(primphoton_rows_r).groupby(level=int_levels).first()
    subprim_proton_contained_r = _ptmp.rec.dlp.particles.is_contained.where(subprimproton_rows_r).groupby(level=int_levels).first()
    proton_p_r = _ptmp.rec.dlp.particles.p.where(primproton_rows_r).groupby(level=int_levels).first()
    proton_softmax_r = _ptmp.rec.dlp.particles.pid_scores['I4'].where(primproton_rows_r).groupby(level=int_levels).first()
    subprim_proton_energy_r = _ptmp.rec.dlp.particles.ke.where(subprimproton_rows_r).groupby(level=int_levels).first()
    subprim_proton_energy_csda_r = _ptmp.rec.dlp.particles.csda_ke.where(subprimproton_rows_r).groupby(level=int_levels).first()
    subprim_proton_energy_mcs_r = _ptmp.rec.dlp.particles.mcs_ke.where(subprimproton_rows_r).groupby(level=int_levels).first()
    subprim_proton_p_r = _ptmp.rec.dlp.particles.p.where(subprimproton_rows_r).groupby(level=int_levels).first()
    subprim_proton_softmax_r = _ptmp.rec.dlp.particles.pid_scores['I4'].where(subprimproton_rows_r).groupby(level=int_levels).first()
    photon_energy_r = _ptmp.rec.dlp.particles.ke.where(primphoton_rows_r).groupby(level=int_levels).first()

    ele_energy_t         = _first_t(spinetpart_df[_stp_ke].where(primele_rows_t))
    muon_energy_t        = _first_t(spinetpart_df[_stp_ke].where(primmuon_rows_t))
    pion_energy_t        = _first_t(spinetpart_df[_stp_ke].where(primpion_rows_t))
    proton_energy_t      = _first_t(spinetpart_df[_stp_ke].where(primproton_rows_t))
    photon_energy_t      = _first_t(spinetpart_df[_stp_ke].where(primphoton_rows_t))
    muon_energy_csda_t   = _first_t(spinetpart_df[_stp_csda].where(primmuon_rows_t))
    pion_energy_csda_t   = _first_t(spinetpart_df[_stp_csda].where(primpion_rows_t))
    proton_energy_csda_t = _first_t(spinetpart_df[_stp_csda].where(primproton_rows_t))
    muon_energy_mcs_t    = _first_t(spinetpart_df[_stp_mcs].where(primmuon_rows_t))
    pion_energy_mcs_t    = _first_t(spinetpart_df[_stp_mcs].where(primpion_rows_t))
    proton_energy_mcs_t  = _first_t(spinetpart_df[_stp_mcs].where(primproton_rows_t))
    proton_p_t           = _first_t(spinetpart_df[_stp_p].where(primproton_rows_t))
    subprim_proton_energy_t      = _first_t(spinetpart_df[_stp_ke].where(subprimproton_rows_t))
    subprim_proton_p_t           = _first_t(spinetpart_df[_stp_p].where(subprimproton_rows_t))
    subprim_proton_energy_csda_t = _first_t(spinetpart_df[_stp_csda].where(subprimproton_rows_t))
    subprim_proton_energy_mcs_t  = _first_t(spinetpart_df[_stp_mcs].where(subprimproton_rows_t))
    subprim_proton_contained_t   = _first_t(spinetpart_df[_stp_contained].where(subprimproton_rows_t))
    muon_contained_t     = _first_t(spinetpart_df[_stp_contained].where(primmuon_rows_t))
    pion_contained_t     = _first_t(spinetpart_df[_stp_contained].where(primpion_rows_t))
    proton_contained_t   = _first_t(spinetpart_df[_stp_contained].where(primproton_rows_t))
    ele_contained_t      = _first_t(spinetpart_df[_stp_contained].where(primele_rows_t))
    photon_contained_t   = _first_t(spinetpart_df[_stp_contained].where(primphoton_rows_t))

    # Interaction-level particle_counts.* summary is not populated in current SPINE CAF output;
    # compute equivalents here from particle-level is_valid + pdg_code (same semantics SPINE intends)
    _pdg_r = abs(_ptmp.rec.dlp.particles.pdg_code)
    _valid_r = _ptmp.rec.dlp.particles.is_valid
    photon_count_reco   = ((_pdg_r == 22)   & (_valid_r == 1)).groupby(level=int_levels).sum()
    electron_count_reco = ((_pdg_r == 11)   & (_valid_r == 1)).groupby(level=int_levels).sum()
    muon_count_reco     = ((_pdg_r == 13)   & (_valid_r == 1)).groupby(level=int_levels).sum()
    pion_count_reco     = ((_pdg_r == 211)  & (_valid_r == 1)).groupby(level=int_levels).sum()
    proton_count_reco   = ((_pdg_r == 2212) & (_valid_r == 1)).groupby(level=int_levels).sum()

    _pdg_t_abs = abs(spinetpart_df[_stp_pdg])
    _valid_t   = spinetpart_df[_stp_valid]
    photon_count_true   = _remap_true_to_reco(((_pdg_t_abs == 22)   & (_valid_t == 1)).groupby(level=_true_int_levels).sum(), slcdf).fillna(0)
    electron_count_true = _remap_true_to_reco(((_pdg_t_abs == 11)   & (_valid_t == 1)).groupby(level=_true_int_levels).sum(), slcdf).fillna(0)
    muon_count_true     = _remap_true_to_reco(((_pdg_t_abs == 13)   & (_valid_t == 1)).groupby(level=_true_int_levels).sum(), slcdf).fillna(0)
    pion_count_true     = _remap_true_to_reco(((_pdg_t_abs == 211)  & (_valid_t == 1)).groupby(level=_true_int_levels).sum(), slcdf).fillna(0)
    proton_count_true   = _remap_true_to_reco(((_pdg_t_abs == 2212) & (_valid_t == 1)).groupby(level=_true_int_levels).sum(), slcdf).fillna(0)

    tki_mc = get_tki_spine(_ptmp.rec.dlp.particles, _ptmp.rec.dlp.primele, int_levels)

    tki_lp_mc = get_tki_spine_lp(_ptmp.rec.dlp.particles, _ptmp.rec.dlp.primele, _ptmp.rec.dlp.primpr, int_levels)

    _tki_mc_true_pre = get_tki_spine(spinetpart_df.rec.dlp_true.particles, _bc(_prim_ele_idx), _true_int_levels)
    tki_mc_true = {k: _remap_true_to_reco(v, slcdf) for k, v in _tki_mc_true_pre.items()}

    _tki_lp_mc_true_pre = get_tki_spine_lp(spinetpart_df.rec.dlp_true.particles, _bc(_prim_ele_idx), _bc(_prim_pr_idx), _true_int_levels)
    tki_lp_mc_true = {k: _remap_true_to_reco(v, slcdf) for k, v in _tki_lp_mc_true_pre.items()}

    lp_open_angle = np.cos(get_lp_open_angle_spine(_ptmp.rec.dlp.particles, _ptmp.rec.dlp.primele, _ptmp.rec.dlp.primpr, int_levels))

    lepton_beam_angle = np.cos(get_lepton_beam_angle_spine(_ptmp.rec.dlp.particles, _ptmp.rec.dlp.primele, int_levels))

    lp_open_angle_true = np.cos(_remap_true_to_reco(get_lp_open_angle_spine(spinetpart_df.rec.dlp_true.particles, _bc(_prim_ele_idx), _bc(_prim_pr_idx), _true_int_levels), slcdf))

    lepton_beam_angle_true = np.cos(_remap_true_to_reco(get_lepton_beam_angle_spine(spinetpart_df.rec.dlp_true.particles, _bc(_prim_ele_idx), _true_int_levels), slcdf))

    sbnd_fiducial = InFV(slcdf.rec.dlp.vertex, det="SBND_Gen1")
    sbnd_fiducial_true = InFV(slcdf.rec.dlp_true.vertex, det="SBND_Gen1")

    true_signalNp = nu_mask & fiducial_mask_t & sbnd_fiducial_true & cc_mask & nupdg_mask & \
                  ele_mask_t & proton_mask_t & subprimproton_mask_t & ~muon_mask_t & ~pion_mask_t & ~photon_mask_t

    true_signal1p = ~true_signalNp & \
                (nu_mask & fiducial_mask_t & sbnd_fiducial_true & cc_mask & nupdg_mask & \
                  ele_mask_t & proton_mask_t & ~subprimproton_mask_t & ~muon_mask_t & ~pion_mask_t & ~photon_mask_t)

    bkgd_oofv = ~true_signal1p & ~true_signalNp & \
                (nu_mask & cc_mask & nupdg_mask & ele_mask_t & \
                  proton_mask_t & ~muon_mask_t & ~pion_mask_t & ~photon_mask_t)

    bkgd_oops = ~true_signal1p & ~true_signalNp & ~bkgd_oofv & \
                (nu_mask & cc_mask & nupdg_mask & ele_mask_noKE_t & \
                  proton_mask_noKE_t & ~muon_mask_t & ~pion_mask_t & ~photon_mask_t)

    bkgd_pi = ~true_signal1p & ~true_signalNp & ~bkgd_oofv & ~bkgd_oops & \
                (nu_mask & cc_mask & nupdg_mask & ele_mask_t & \
                  proton_mask_t & ~muon_mask_t & ~photon_mask_t)

    bkgd_mu = ~true_signal1p & ~true_signalNp & ~bkgd_oofv & ~bkgd_oops & ~bkgd_pi & \
                (nu_mask & cc_mask & nupdg_mask & ele_mask_t & \
                  proton_mask_t & ~pion_mask_t & ~photon_mask_t)
    
    bkgd_photon = ~true_signal1p & ~true_signalNp & ~bkgd_oofv & ~bkgd_oops & ~bkgd_pi & ~bkgd_mu & \
                (nu_mask & cc_mask & nupdg_mask & ele_mask_t & \
                  proton_mask_t & ~pion_mask_t & ~muon_mask_t)

    bkgd_nueOther = ~true_signal1p & ~true_signalNp & ~bkgd_oofv & ~bkgd_oops & \
                ~bkgd_pi & ~bkgd_mu & ~bkgd_photon & \
                (nu_mask & cc_mask & nupdg_mask)

    bkgd_numu = ~true_signal1p & ~true_signalNp & ~bkgd_oofv & ~bkgd_oops & \
                ~bkgd_pi & ~bkgd_mu & ~bkgd_photon & ~bkgd_nueOther & \
                (nu_mask & cc_mask & ~nupdg_mask)

    bkgd_nc = ~true_signal1p & ~true_signalNp & ~bkgd_oofv & ~bkgd_oops & \
                ~bkgd_pi & ~bkgd_mu & ~bkgd_photon & ~bkgd_nueOther & ~bkgd_numu & \
                (nu_mask & ~cc_mask)

    bkgd_other = ~true_signal1p & ~true_signalNp & ~bkgd_oofv & ~bkgd_oops & \
                ~bkgd_pi & ~bkgd_mu & ~bkgd_photon & ~bkgd_nueOther & ~bkgd_numu & ~bkgd_nc #& \
                #(~nu_mask)

    assert not (pion_mask_t & true_signal1p).any(), \
        f"pion_mask_t True on {(pion_mask_t & true_signal1p).sum()} rows also flagged true_signal1p"
    assert not (pion_mask_t & true_signalNp).any(), \
        f"pion_mask_t True on {(pion_mask_t & true_signalNp).sum()} rows also flagged true_signalNp"

    
    for label, mask, in [
                        ('nu_mask',       nu_mask),
                        ('cc_mask',       cc_mask),
                        ('nupdg_mask',    nupdg_mask),

                        ('fiducial_mask_true', fiducial_mask_t),
                        ('sbnd_fiducial_true', sbnd_fiducial_true),
                        ('ele_mask_true',      ele_mask_t),
                        ('proton_mask_true',   proton_mask_t),
                        ('subprimproton_mask_true', subprimproton_mask_t),
                        ('muon_mask_true',     muon_mask_t),
                        ('pion_mask_true',     pion_mask_t),
                        ('photon_mask_true',   photon_mask_t),
                        ('ele_energy_true',      ele_energy_t),
                        ('muon_energy_true',     muon_energy_t),
                        ('pion_energy_true',     pion_energy_t),
                        ('proton_energy_true',   proton_energy_t),
                        ('proton_p_true',        proton_p_t),
                        ('subprim_proton_energy_true', subprim_proton_energy_t),
                        ('subprim_proton_p_true', subprim_proton_p_t),
                        ('photon_energy_true',   photon_energy_t),

                        ('muon_energy_csda_true', muon_energy_csda_t),
                        ('pion_energy_csda_true', pion_energy_csda_t),
                        ('proton_energy_csda_true', proton_energy_csda_t),
                        ('subprim_proton_energy_csda_true', subprim_proton_energy_csda_t),
                        ('muon_energy_mcs_true', muon_energy_mcs_t),
                        ('pion_energy_mcs_true', pion_energy_mcs_t),
                        ('proton_energy_mcs_true', proton_energy_mcs_t),
                        ('subprim_proton_energy_mcs_true', subprim_proton_energy_mcs_t),
                        ('muon_contained_true', muon_contained_t),
                        ('pion_contained_true', pion_contained_t),
                        ('proton_contained_true', proton_contained_t),
                        ('subprim_proton_contained_true', subprim_proton_contained_t),
                        ('ele_contained_true', ele_contained_t),
                        ('photon_contained_true', photon_contained_t),

                        ('del_alpha_lp_true', tki_lp_mc_true['del_alpha']),
                        ('del_phi_lp_true',   tki_lp_mc_true['del_phi']),
                        ('del_Tp_lp_true',    tki_lp_mc_true['del_Tp']),
                        ('del_alpha_true', tki_mc_true['del_alpha']),
                        ('del_phi_true',   tki_mc_true['del_phi']),
                        ('del_Tp_true',    tki_mc_true['del_Tp']),

                        ('lp_open_angle_true', lp_open_angle_true),
                        ('lepton_beam_angle_true', lepton_beam_angle_true),

                        ("photon_count_true",   photon_count_true),
                        ("electron_count_true", electron_count_true),
                        ("muon_count_true",     muon_count_true),
                        ("pion_count_true",     pion_count_true),
                        ("proton_count_true",   proton_count_true),

                        ('true_signal1p', true_signal1p),
                        ('true_signalNp', true_signalNp),
                        ('bkgd_oofv',     bkgd_oofv),
                        ('bkgd_oops',     bkgd_oops),
                        ('bkgd_pi',       bkgd_pi),
                        ('bkgd_mu',       bkgd_mu),
                        ('bkgd_photon',   bkgd_photon),
                        ('bkgd_nueOther', bkgd_nueOther),
                        ('bkgd_numu',     bkgd_numu),
                        ('bkgd_nc',       bkgd_nc),
                        ('bkgd_other',    bkgd_other)]:
        col = pad_column_name(('rec', 'dlp_true', label), slcdf)
        slcdf[col] = mask

    for label, mask in [                        
                        ('fiducial_mask_reco', fiducial_mask_r),
                        ('sbnd_fiducial_reco', sbnd_fiducial),
                        ('flash_match_reco', flash_match_r),
                        ('containment_reco', containment_r),
                        ('subprimproton_mask_reco', subprimproton_mask_r),
                        ('muon_mask_reco',     muon_mask_r),
                        ('pion_mask_reco',     pion_mask_r),
                        ('photon_mask_reco',   photon_mask_r),
                        ('ele_mask_reco',      ele_mask_r),
                        ('proton_mask_reco',   proton_mask_r),
                        ('ele_energy_reco',      ele_energy_r),
                        ('ele_dedx_reco',         ele_dedx_r),
                        ('ele_softmax_reco',     ele_softmax_r),
                        ('ele_primary_reco',     ele_primary_r),
                        ('ele_vertex_distance_reco', ele_vertex_distance_r),
                        ('muon_energy_reco',     muon_energy_r),
                        ('pion_energy_reco',     pion_energy_r),
                        ('proton_energy_reco',   proton_energy_r),
                        ('proton_p_reco',        proton_p_r),
                        ('proton_softmax_reco',   proton_softmax_r),
                        ('subprim_proton_energy_reco', subprim_proton_energy_r),
                        ('subprim_proton_p_reco', subprim_proton_p_r),
                        ('subprim_proton_softmax_reco', subprim_proton_softmax_r),
                        ('photon_energy_reco',   photon_energy_r),

                        ('muon_energy_csda_reco', muon_energy_csda_r),
                        ('pion_energy_csda_reco', pion_energy_csda_r),
                        ('proton_energy_csda_reco', proton_energy_csda_r),
                        ('subprim_proton_energy_csda_reco', subprim_proton_energy_csda_r),
                        ('muon_energy_mcs_reco', muon_energy_mcs_r),
                        ('pion_energy_mcs_reco', pion_energy_mcs_r),
                        ('proton_energy_mcs_reco', proton_energy_mcs_r),
                        ('subprim_proton_energy_mcs_reco', subprim_proton_energy_mcs_r),
                        ('muon_contained_reco', muon_contained_r),
                        ('pion_contained_reco', pion_contained_r),
                        ('proton_contained_reco', proton_contained_r),
                        ('subprim_proton_contained_reco', subprim_proton_contained_r),
                        ('ele_contained_reco', ele_contained_r),
                        ('photon_contained_reco', photon_contained_r),

                        ('del_alpha_lp_reco', tki_lp_mc['del_alpha']),
                        ('del_phi_lp_reco',   tki_lp_mc['del_phi']),
                        ('del_Tp_lp_reco',    tki_lp_mc['del_Tp']),
                        ('del_alpha_reco', tki_mc['del_alpha']),
                        ('del_phi_reco',   tki_mc['del_phi']),
                        ('del_Tp_reco',    tki_mc['del_Tp']),
                        
                        ('lp_open_angle_reco', lp_open_angle),
                        ('lepton_beam_angle_reco', lepton_beam_angle),

                        ('photon_count_reco',   photon_count_reco),
                        ('electron_count_reco', electron_count_reco),
                        ('muon_count_reco',     muon_count_reco),
                        ('pion_count_reco',     pion_count_reco),
                        ('proton_count_reco',   proton_count_reco),

                        ]:
        col = pad_column_name(('rec', 'dlp', label), slcdf)
        slcdf[col] = mask

    # pre-selection cuts
    # slcdf = slcdf[slcdf.rec.dlp['is_fiducial'] == 1]
    # slcdf = slcdf[slcdf.rec.dlp['is_flash_matched'] == 1]
    # slcdf = slcdf[slcdf.rec.dlp['is_contained'] == 1]

    slcdf = multicol_merge(slcdf, hdrdf, left_index=True, right_index=True, how="left", validate="many_to_one")

    return truth_match_spine(slcdf, mcdf) 


def make_nueNp0Pi_selected_df(f):

    df = make_nueNp0Pi_df(f)

    # pre-selection cuts
    df = df[(df.rec.dlp['is_fiducial'] == 1) & (df.rec.dlp['sbnd_fiducial_reco'] == 1)]
    df = df[df.rec.dlp['is_flash_matched'] == 1]
    df = df[df.rec.dlp['is_contained'] == 1]

    # selection cuts
    df = df[df.rec.dlp['muon_mask_reco'] == False]
    df = df[df.rec.dlp['pion_mask_reco'] == False]
    df = df[df.rec.dlp['photon_mask_reco'] == False]
    df = df[df.rec.dlp['ele_mask_reco'] == True]
    df = df[df.rec.dlp['proton_mask_reco'] == True]
    df = df[df.rec.dlp['ele_primary_reco'] >= ELE_PRIMARY_TH]
    df = df[df.rec.dlp['ele_softmax_reco'] >= ELE_SOFTMAX_TH]
    df = df[df.rec.dlp['proton_softmax_reco'] >= P_SOFTMAX_TH]
    df = df[df.rec.dlp['ele_dedx_reco'] < ELE_DEDX_TH]
    df = df[df.rec.dlp['ele_vertex_distance_reco'] < ELE_VTXDIST_TH]

    return df
