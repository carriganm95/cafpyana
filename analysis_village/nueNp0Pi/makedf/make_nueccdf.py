from makedf.makedf import *
from pyanalib.pandas_helpers import *
from makedf.util import *

def make_mcnudf_nuecc(f,**args):
    mcdf = make_mcnudf(f,**args)
    # drop mcdf columns not relevant for this analysis
    if 'mu'  in list(zip(*list(mcdf.columns)))[0]:  mcdf = mcdf.drop('mu', axis=1,level=0)
    if 'p'   in list(zip(*list(mcdf.columns)))[0]:  mcdf = mcdf.drop('p',  axis=1,level=0)
    if 'cpi' in list(zip(*list(mcdf.columns)))[0]:  mcdf = mcdf.drop('cpi',axis=1,level=0)
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

    for pdg, label in [(11, 'true_primele'), (13, 'true_pimmu'), (211, 'true_primpi'), (2212, 'true_primpr'), (22, 'true_primph')]:
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

    # merge pfpdf into slcdf to get particle-level info for the candidate particles
    slcdf = multicol_merge(slcdf, pfpdf, left_index=True, right_index=True, how="right", validate="one_to_many")

    # create truth categories
    int_levels = list(range(slcdf.index.nlevels - 1))  # all levels except particle
    particle_idx       = pd.Series(slcdf.index.get_level_values(slcdf.index.names[-1]), index=slcdf.index)
    primele_rows       = particle_idx == slcdf.rec.dlp_true.true_primele
    primproton_rows    = particle_idx == slcdf.rec.dlp_true.true_primpr
    subprimproton_rows = particle_idx == slcdf.rec.dlp_true.true_subprimpr
    primmuon_rows      = particle_idx == slcdf.rec.dlp_true.true_pimmu
    primpion_rows      = particle_idx == slcdf.rec.dlp_true.true_primpi
    primphoton_rows    = particle_idx == slcdf.rec.dlp_true.true_primph

    nu_mask        = slcdf.rec.dlp_true.nu_id >= 0
    fiducial_mask  = slcdf.rec.dlp_true.is_fiducial == 1
    cc_mask        = slcdf.rec.dlp_true.current_type == 0
    nupdg_mask     = abs(slcdf.rec.dlp_true.pdg_code) == 12
    ele_mask       = broadcast_to_particles((primele_rows    & (slcdf.rec.dlp_true.particles.calo_ke > 500.0)).groupby(level=int_levels).any(), slcdf)
    proton_mask    = broadcast_to_particles((primproton_rows & (slcdf.rec.dlp_true.particles.calo_ke > 40.0 )).groupby(level=int_levels).any(), slcdf)
    subproton_mask = broadcast_to_particles((subprimproton_rows & (slcdf.rec.dlp_true.particles.calo_ke > 40.0 )).groupby(level=int_levels).any(), slcdf)
    muon_mask      = broadcast_to_particles((primmuon_rows   & (slcdf.rec.dlp_true.particles.calo_ke > 25.0 )).groupby(level=int_levels).any(), slcdf)
    pion_mask      = broadcast_to_particles((primpion_rows   & (slcdf.rec.dlp_true.particles.calo_ke > 25.0 )).groupby(level=int_levels).any(), slcdf)
    photon_mask    = broadcast_to_particles((primphoton_rows & (slcdf.rec.dlp_true.particles.calo_ke > 100.0)).groupby(level=int_levels).any(), slcdf)

    ele_mask_noKE = broadcast_to_particles((primele_rows).groupby(level=int_levels).any(), slcdf)
    proton_mask_noKE = broadcast_to_particles(primproton_rows.groupby(level=int_levels).any(), slcdf)
    subproton_mask_noKE = broadcast_to_particles(subprimproton_rows.groupby(level=int_levels).any(), slcdf)
    muon_mask_noKE = broadcast_to_particles(primmuon_rows.groupby(level=int_levels).any(), slcdf)
    pion_mask_noKE = broadcast_to_particles(primpion_rows.groupby(level=int_levels).any(), slcdf)
    photon_mask_noKE = broadcast_to_particles(primphoton_rows.groupby(level=int_levels).any(), slcdf)

    true_signalNp = nu_mask & fiducial_mask & cc_mask & nupdg_mask & \
                  ele_mask & proton_mask & subproton_mask & ~muon_mask & ~pion_mask & ~photon_mask

    true_signal1p = ~true_signalNp & \
                (nu_mask & fiducial_mask & cc_mask & nupdg_mask & \
                  ele_mask & proton_mask & ~subproton_mask & ~muon_mask & ~pion_mask & ~photon_mask)

    bkgd_oofv = ~true_signal1p & ~true_signalNp & \
                (nu_mask & cc_mask & nupdg_mask & ele_mask & \
                  proton_mask & ~muon_mask & ~pion_mask & ~photon_mask)

    bkgd_oops = ~true_signal1p & ~true_signalNp & ~bkgd_oofv & \
                (nu_mask & cc_mask & nupdg_mask & ele_mask_noKE & \
                  proton_mask_noKE & ~muon_mask & ~pion_mask & ~photon_mask)

    bkgd_pi = ~true_signal1p & ~true_signalNp & ~bkgd_oofv & ~bkgd_oops & \
                (nu_mask & cc_mask & nupdg_mask & ele_mask & \
                  proton_mask & ~muon_mask & ~photon_mask)

    bkgd_mu = ~true_signal1p & ~true_signalNp & ~bkgd_oofv & ~bkgd_oops & ~bkgd_pi & \
                (nu_mask & cc_mask & nupdg_mask & ele_mask & \
                  proton_mask & ~pion_mask & ~photon_mask)
    
    bkgd_photon = ~true_signal1p & ~true_signalNp & ~bkgd_oofv & ~bkgd_oops & ~bkgd_pi & ~bkgd_mu & \
                (nu_mask & cc_mask & nupdg_mask & ele_mask & \
                  proton_mask & ~pion_mask & ~muon_mask)

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
                ~bkgd_pi & ~bkgd_mu & ~bkgd_photon & ~bkgd_nueOther & ~bkgd_numu & ~bkgd_nc & \
                (~nu_mask)
    
    for label, mask in [('nu_mask',       nu_mask),
                        ('fiducial_mask', fiducial_mask),
                        ('cc_mask',       cc_mask),
                        ('nupdg_mask',    nupdg_mask),
                        ('ele_mask',      ele_mask),
                        ('proton_mask',   proton_mask),
                        ('muon_mask',     muon_mask),
                        ('pion_mask',     pion_mask),
                        ('photon_mask',   photon_mask),
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
        slcdf[col] = mask.values

    # pre-selection cuts
    slcdf = slcdf[slcdf.rec.dlp['is_fiducial'] == 1]
    slcdf = slcdf[slcdf.rec.dlp['is_flash_matched'] == 1]
    slcdf = slcdf[slcdf.rec.dlp['is_contained'] == 1]

    return slcdf 
