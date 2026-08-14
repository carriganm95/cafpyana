from makedf.makedf import *
from pyanalib.pandas_helpers import *
from makedf.util import *
from pyanalib.variable_calculator import get_tki_spine, get_tki_spine_lp, get_lp_open_angle_spine, get_lepton_beam_angle_spine

## == For additional column in mcdf with primary particle multiplicities
## ==== "<column name>": ["<particle name>", <KE cut in GeV>]
## ==== <particle name> is used to collect PID and mass from the "PDG" dictionary
TRUE_KE_THRESHOLDS = {"nmu_25MeV": ["muon", 0.025],
                      "np_40MeV": ["proton", 0.040],
                      "npi_25MeV": ["pipm", 0.025],
                      "ne_500MeV": ["electron", 0.5],
                      'ng_100MeV': ["gamma", 0.1]
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
    int_levels = list(range(pfpdf.index.nlevels - 1))
    empty = pd.Series(dtype=float, index=pd.MultiIndex.from_tuples([], names=pfpdf.index.names[:len(int_levels)]))
    filtered = pfpdf[(abs(pfpdf[pdg_col]) == pdg) & (pfpdf[ke_col] > 0)]
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
    int_levels = list(range(pfpdf.index.nlevels - 1))
    empty = pd.Series(dtype=float, index=pd.MultiIndex.from_tuples([], names=pfpdf.index.names[:len(int_levels)]))
    filtered = pfpdf[(abs(pfpdf[pdg_col]) == pdg) & (pfpdf[ke_col] > 0)]
    if filtered.empty:
        return empty
    ranks = filtered[ke_col].groupby(level=int_levels).rank(method='first', ascending=False)
    second = filtered[ke_col][ranks == 2]
    if second.empty:
        return empty
    best_idx = second.groupby(level=int_levels).idxmax()
    return best_idx.apply(lambda x: x[pfpdf.index.nlevels - 1])

def make_nueNp0Pi_df(f):
    det = loadbranches(f["recTree"], ["rec.hdr.det"]).rec.hdr.det
    if (1 == det.unique()):
        DETECTOR = "SBND"
    else:
        DETECTOR = "ICARUS"

    assert DETECTOR == "SBND"

    pfpdf = make_spine_part_df(f)
    slcdf = make_spine_int_df(f)

    mcdf = make_mcnudf_nuecc(f)

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

    for pdg, label in [(11, 'true_primele'), (13, 'true_primmu'), (211, 'true_primpi'), (2212, 'true_primpr'), (22, 'true_primph')]:
        idx = get_highest_ke_particle_idx(pfpdf, pdg, truth=True)
        idx.index.names = slcdf.index.names
        col = tuple(['rec', 'dlp_true', label] + [''] * (nlevel - 3))
        idx_df = idx.rename(col).to_frame()
        slcdf = multicol_merge(slcdf, idx_df, left_index=True, right_index=True, how='left')

    # get subleading proton
    idx = get_second_highest_ke_particle_idx(pfpdf, 2212)
    idx.index.names = slcdf.index.names
    col = tuple(['rec', 'dlp', 'subprimpr'] + [''] * (nlevel - 3))
    idx_df = idx.rename(col).to_frame()
    slcdf = multicol_merge(slcdf, idx_df, left_index=True, right_index=True, how='left')

    idx = get_second_highest_ke_particle_idx(pfpdf, 2212, truth=True)
    idx.index.names = slcdf.index.names
    col = tuple(['rec', 'dlp_true', 'true_subprimpr'] + [''] * (nlevel - 3))
    idx_df = idx.rename(col).to_frame()
    slcdf = multicol_merge(slcdf, idx_df, left_index=True, right_index=True, how='left')

    # expand to particle level temporarily to compute truth category masks
    _ptmp = multicol_merge(slcdf, pfpdf, left_index=True, right_index=True, how="right", validate="one_to_many")

    # create truth categories — all masks reduced back to event level via groupby
    int_levels = list(range(_ptmp.index.nlevels - 1))  # all levels except particle
    particle_idx       = pd.Series(_ptmp.index.get_level_values(_ptmp.index.names[-1]), index=_ptmp.index)
    
    primele_rows_t       = particle_idx == _ptmp.rec.dlp_true.true_primele
    primproton_rows_t    = particle_idx == _ptmp.rec.dlp_true.true_primpr
    subprimproton_rows_t = particle_idx == _ptmp.rec.dlp_true.true_subprimpr
    primmuon_rows_t      = particle_idx == _ptmp.rec.dlp_true.true_primmu
    primpion_rows_t      = particle_idx == _ptmp.rec.dlp_true.true_primpi
    primphoton_rows_t    = particle_idx == _ptmp.rec.dlp_true.true_primph

    primele_rows_r       = particle_idx == _ptmp.rec.dlp.primele
    primproton_rows_r    = particle_idx == _ptmp.rec.dlp.primpr
    subprimproton_rows_r = particle_idx == _ptmp.rec.dlp.subprimpr
    primmuon_rows_r      = particle_idx == _ptmp.rec.dlp.primmu
    primpion_rows_r      = particle_idx == _ptmp.rec.dlp.primpi
    primphoton_rows_r    = particle_idx == _ptmp.rec.dlp.primph

    nu_mask       = (_ptmp.rec.dlp_true.nu_id >= 0).groupby(level=int_levels).first()
    cc_mask        = (_ptmp.rec.dlp_true.current_type == 0).groupby(level=int_levels).first()
    nupdg_mask     = (abs(_ptmp.rec.dlp_true.pdg_code) == 12).groupby(level=int_levels).first()
    fiducial_mask_t  = (_ptmp.rec.dlp_true.is_fiducial == 1).groupby(level=int_levels).first()
    ele_mask_t       = (primele_rows_t    & (_ptmp.rec.dlp_true.particles.calo_ke > 500.0)).groupby(level=int_levels).any()
    proton_mask_t    = (primproton_rows_t & (_ptmp.rec.dlp_true.particles.calo_ke > 40.0 )).groupby(level=int_levels).any()
    subprimproton_mask_t = (subprimproton_rows_t & (_ptmp.rec.dlp_true.particles.calo_ke > 40.0 )).groupby(level=int_levels).any()
    muon_mask_t      = (primmuon_rows_t   & (_ptmp.rec.dlp_true.particles.calo_ke > 25.0 )).groupby(level=int_levels).any()
    pion_mask_t      = (primpion_rows_t   & (_ptmp.rec.dlp_true.particles.calo_ke > 25.0 )).groupby(level=int_levels).any()
    photon_mask_t    = (primphoton_rows_t & (_ptmp.rec.dlp_true.particles.calo_ke > 100.0)).groupby(level=int_levels).any()

    ele_mask_noKE_t       = primele_rows_t.groupby(level=int_levels).any()
    proton_mask_noKE_t    = primproton_rows_t.groupby(level=int_levels).any()
    subprimproton_mask_noKE_t = subprimproton_rows_t.groupby(level=int_levels).any()
    muon_mask_noKE_t      = primmuon_rows_t.groupby(level=int_levels).any()
    pion_mask_noKE_t      = primpion_rows_t.groupby(level=int_levels).any()
    photon_mask_noKE_t    = primphoton_rows_t.groupby(level=int_levels).any()

    fiducial_mask_r = (_ptmp.rec.dlp.is_fiducial == 1).groupby(level=int_levels).first()
    flash_match_r = (_ptmp.rec.dlp.is_flash_matched == 1).groupby(level=int_levels).first()
    containment_r = (_ptmp.rec.dlp.is_contained == 1).groupby(level=int_levels).first()
    muon_mask_r = (primmuon_rows_r & (_ptmp.rec.dlp.particles.calo_ke > 25.0)).groupby(level=int_levels).any()
    pion_mask_r = (primpion_rows_r & (_ptmp.rec.dlp.particles.calo_ke > 25.0)).groupby(level=int_levels).any()
    proton_mask_r = (primproton_rows_r & (_ptmp.rec.dlp.particles.calo_ke > 40.0)).groupby(level=int_levels).any()
    subprimproton_mask_r = (subprimproton_rows_r & (_ptmp.rec.dlp.particles.calo_ke > 40.0)).groupby(level=int_levels).any()
    photon_mask_r = (primphoton_rows_r & (_ptmp.rec.dlp.particles.calo_ke > 100.0)).groupby(level=int_levels).any()
    ele_mask_r = (primele_rows_r & (_ptmp.rec.dlp.particles.calo_ke > 500.0)).groupby(level=int_levels).any()
    ele_softmax_r = (primele_rows_r & (_ptmp.rec.dlp.particles.pid_scores['I1'] > 0.9)).groupby(level=int_levels).any()
    ele_primary_r = (primele_rows_r & (_ptmp.rec.dlp.particles.primary_scores['I1'] > 0.99)).groupby(level=int_levels).any()
    proton_softmax_r = (primproton_rows_r & (_ptmp.rec.dlp.particles.pid_scores['I4'] > 0.75)).groupby(level=int_levels).any()
    vertex_distance_r = (primele_rows_r & (_ptmp.rec.dlp.particles.vertex_distance < 3.5)).groupby(level=int_levels).any()
    ele_dedx_r = (primele_rows_r & (_ptmp.rec.dlp.particles.start_dedx < 4.0)).groupby(level=int_levels).any()

    ele_energy_r = _ptmp.rec.dlp.particles.calo_ke.where(primele_rows_r).groupby(level=int_levels).first()
    ele_dedx_r = _ptmp.rec.dlp.particles.start_dedx.where(primele_rows_r).groupby(level=int_levels).first()
    ele_softmax_r = _ptmp.rec.dlp.particles.pid_scores['I1'].where(primele_rows_r).groupby(level=int_levels).first()
    ele_primary_r = _ptmp.rec.dlp.particles.primary_scores['I1'].where(primele_rows_r).groupby(level=int_levels).first()
    ele_vertex_distance_r = _ptmp.rec.dlp.particles.vertex_distance.where(primele_rows_r).groupby(level=int_levels).first()
    muon_energy_r = _ptmp.rec.dlp.particles.calo_ke.where(primmuon_rows_r).groupby(level=int_levels).first()
    pion_energy_r = _ptmp.rec.dlp.particles.calo_ke.where(primpion_rows_r).groupby(level=int_levels).first()
    proton_energy_r = _ptmp.rec.dlp.particles.calo_ke.where(primproton_rows_r).groupby(level=int_levels).first()
    proton_p_r = _ptmp.rec.dlp.particles.p.where(primproton_rows_r).groupby(level=int_levels).first()
    proton_softmax_r = _ptmp.rec.dlp.particles.pid_scores['I4'].where(primproton_rows_r).groupby(level=int_levels).first()
    subprim_proton_energy_r = _ptmp.rec.dlp.particles.calo_ke.where(subprimproton_rows_r).groupby(level=int_levels).first()
    subprim_proton_p_r = _ptmp.rec.dlp.particles.p.where(subprimproton_rows_r).groupby(level=int_levels).first()
    subprim_proton_softmax_r = _ptmp.rec.dlp.particles.pid_scores['I4'].where(subprimproton_rows_r).groupby(level=int_levels).first()

    ele_energy_t = _ptmp.rec.dlp_true.particles.calo_ke.where(primele_rows_t).groupby(level=int_levels).first()
    muon_energy_t = _ptmp.rec.dlp_true.particles.calo_ke.where(primmuon_rows_t).groupby(level=int_levels).first()
    pion_energy_t = _ptmp.rec.dlp_true.particles.calo_ke.where(primpion_rows_t).groupby(level=int_levels).first()
    proton_energy_t = _ptmp.rec.dlp_true.particles.calo_ke.where(primproton_rows_t).groupby(level=int_levels).first()
    proton_p_t = _ptmp.rec.dlp_true.particles.p.where(primproton_rows_t).groupby(level=int_levels).first()
    subprim_proton_energy_t = _ptmp.rec.dlp_true.particles.calo_ke.where(subprimproton_rows_t).groupby(level=int_levels).first()
    subprim_proton_p_t = _ptmp.rec.dlp_true.particles.p.where(subprimproton_rows_t).groupby(level=int_levels).first()

    tki_mc = get_tki_spine(_ptmp.rec.dlp.particles, _ptmp.rec.dlp.primele, int_levels)

    tki_lp_mc = get_tki_spine_lp(_ptmp.rec.dlp.particles, _ptmp.rec.dlp.primele, _ptmp.rec.dlp.primpr, int_levels)

    tki_mc_true = get_tki_spine(_ptmp.rec.dlp_true.particles, _ptmp.rec.dlp_true.true_primele, int_levels)

    tki_lp_mc_true = get_tki_spine_lp(_ptmp.rec.dlp_true.particles, _ptmp.rec.dlp_true.true_primele, _ptmp.rec.dlp_true.true_primpr, int_levels)

    lp_open_angle = np.cos(get_lp_open_angle_spine(_ptmp.rec.dlp.particles, _ptmp.rec.dlp.primele, _ptmp.rec.dlp.primpr, int_levels))

    lepton_beam_angle = np.cos(get_lepton_beam_angle_spine(_ptmp.rec.dlp.particles, _ptmp.rec.dlp.primele, int_levels))

    lp_open_angle_true = np.cos(get_lp_open_angle_spine(_ptmp.rec.dlp_true.particles, _ptmp.rec.dlp_true.true_primele, _ptmp.rec.dlp_true.true_primpr, int_levels))

    lepton_beam_angle_true = np.cos(get_lepton_beam_angle_spine(_ptmp.rec.dlp_true.particles, _ptmp.rec.dlp_true.true_primele, int_levels))

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

                        ('del_alpha_lp_true', tki_lp_mc_true['del_alpha']),
                        ('del_phi_lp_true',   tki_lp_mc_true['del_phi']),
                        ('del_Tp_lp_true',    tki_lp_mc_true['del_Tp']),
                        ('del_alpha_true', tki_mc_true['del_alpha']),
                        ('del_phi_true',   tki_mc_true['del_phi']),
                        ('del_Tp_true',    tki_mc_true['del_Tp']),

                        ('lp_open_angle_true', lp_open_angle_true),
                        ('lepton_beam_angle_true', lepton_beam_angle_true),

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
                        
                        ('del_alpha_lp_reco', tki_lp_mc['del_alpha']),
                        ('del_phi_lp_reco',   tki_lp_mc['del_phi']),
                        ('del_Tp_lp_reco',    tki_lp_mc['del_Tp']),
                        ('del_alpha_reco', tki_mc['del_alpha']),
                        ('del_phi_reco',   tki_mc['del_phi']),
                        ('del_Tp_reco',    tki_mc['del_Tp']),
                        
                        ('lp_open_angle_reco', lp_open_angle),
                        ('lepton_beam_angle_reco', lepton_beam_angle)

                        ]:
        col = pad_column_name(('rec', 'dlp', label), slcdf)
        slcdf[col] = mask

    # pre-selection cuts
    # slcdf = slcdf[slcdf.rec.dlp['is_fiducial'] == 1]
    # slcdf = slcdf[slcdf.rec.dlp['is_flash_matched'] == 1]
    # slcdf = slcdf[slcdf.rec.dlp['is_contained'] == 1]

    return truth_match_spine(slcdf, mcdf) 
