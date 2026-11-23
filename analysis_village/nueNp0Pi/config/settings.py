"""Selection settings: thresholds, detector convention, and category labels/colors.

This is the single place to edit reconstruction-level cut thresholds, the
truth-fiducial detector convention, and the labels/colors used by
signal/background breakdown plots. It holds ONLY settings -- no cut or
category LOGIC. Named ``settings.py`` (not ``selections.py``) specifically to
disambiguate from :mod:`analysis_village.nueNp0Pi.selections` (the functions
that consume these settings: truth predicates, reco cuts, category/breakdown
functions, BREAKDOWN_REGISTRY, SIGNAL_MASK_FN), which imports its constants
from here -- two files both named "selections" was confusing.

Plot/variable definitions live in :mod:`analysis_village.nueNp0Pi.config.plots`.
Dataset paths and other config live in :mod:`analysis_village.nueNp0Pi.config.datasets`.
"""

# ===========================================================================
# Truth-fiducial / reco-fiducial detector convention.
# ===========================================================================
DETECTOR = "SBND_Gen1"
# DETECTOR = "SBND"

# Cathode inset (cm) for per-TPC x-fiducial; matches reco helpers in
# ``event_contained_per_tpc`` (analysis_village.nueNp0Pi.selections) and
# selected-events drivers.
PER_TPC_INCATHODE_CM = 10


# ===========================================================================
# Labels & colors for category/breakdown plots.
# ===========================================================================
# particle-type (pdg) breakdown
pdg_labels = [r"$e^{\pm}$", r"$\gamma$", r"$\mu^{\pm}$", r"$p$", r"$\pi^{\pm}$", r"Other"]
pdg_colors = ["C0", "C1", "C2", "C3", "C4", "C5"]

# nu / cosmic breakdown
nu_cosmics_labels = ["Cosmic", r"Out-FV $\nu$", r"FV $\nu$"]
nu_cosmics_colors = ["gray", "C0", "C1"]

# signal / background topology breakdown
# the signal mode code MUST be the first item in the list for all the code below to work
topology_list = [0, 1, 2, 3, 4, 5,
                6, 7, 8, 9, 10]

topology_labels = [r'$\nu_e$ CC 1p0$\pi$',
                    r'$\nu_e$ CC >1p0$\pi$',
                    r'$\nu_e$ CC OOPS',
                    r'$\nu_e$ CC OOFV',
                    r'$\nu_e$ CC $\pi^{\pm}$',
                    r'$\nu_e$ CC $\mu^{\pm}$',
                    r'$\nu_e$ CC $\gamma$',
                    r'$\nu_e$ CC Other',
                    r'$\nu_{\mu}$ CC',
                    r'NC',
                    r'Other']

topology_colors = ['#332288',  # νe CC 1p0π
                    '#117733',  # νe CC >1p0π
                    '#44AA99',  # νe CC OOPS
                    '#88CCEE',  # νe CC OOFV
                    '#DDCC77',  # νe CC π±
                    '#CC6677',  # νe CC μ±
                    '#AA4499',  # νe CC γ
                    '#882255',  # νe CC Other
                    '#6699CC',  # νμ CC
                    '#999933',  # NC
                    '#888888']  # Other

# --- GENIE interaction mode breakdown ---
genie_mode_list = [0, # CCQE
                   1, # CCRES
                   2, # CCDIS
                   3, # CCMEC
                   4, # CC Other (signal)
                   5, # non-signal CC
                   -1, # NC
                   -2  # Cosmic
                   ]
# genie_mode_labels = [r'$\nu_{\mu}$ CC QE', r'$\nu_{\mu}$ CC MEC', r'$\nu_{\mu}$ CC RES', r'$\nu_{\mu}$ CC SIS/DIS', r'$\nu_{\mu}$ CC Other',
genie_mode_labels = [ r'$\nu_e$ CC QE',
                        r'$\nu_e$ CC RES',
                        r'$\nu_e$ CC DIS',
                        r'$\nu_e$ CC MEC',
                        r'$\nu_e$ CC Other',
                        r'$\nu$ CC Other',
                        r"$\nu$ NC",
                        "Cosmic"]


genie_mode_colors = ['#228833',  # QE
                     '#EE6677',  # RES
                     '#4477AA',  # DIS
                     '#CCBB44',  # MEC
                     '#AA3377',  # CC Other (signal)
                     '#882255',  # non-signal CC
                     '#66CCEE',  # NC
                     '#BBBBBB']  # Cosmic

# --- GENIE SB interaction mode breakdown ---
genie_sb_mode_labels = [r'$\nu_{\mu}$ CC QE', r'$\nu_{\mu}$ CC QE',
                     r'$\nu_{\mu}$ CC MEC', r'$\nu_{\mu}$ CC MEC',
                     r'$\nu_{\mu}$ CC RES', r'$\nu_{\mu}$ CC RES',
                     r'$\nu_{\mu}$ CC Other', r'$\nu_{\mu}$ CC Other',
                     r"$\nu$ NC",
                     r"Other $\nu$",
                     "Cosmic"]

genie_sb_mode_colors = ['#228833', '#228833',  # QE pair
                        '#CCBB44', '#CCBB44',  # MEC pair
                        '#EE6677', '#EE6677',  # RES pair
                        '#AA3377', '#AA3377',  # Other pair
                        '#66CCEE',             # NC
                        '#999933',             # Other ν
                        '#BBBBBB']             # Cosmic


# --- GiBUU interaction mode breakdown ---
gibuu_mode_labels = [r'$\nu_{\mu}$ CC QE', r'$\nu_{\mu}$ CC QE (2p2h)', r'$\nu_{\mu}$ CC RES', r'$\nu_{\mu}$ CC DIS', r'$\nu_{\mu}$ CC Other',
                     r"$\nu$ NC", r"FV other $\nu$", r"Out-FV $\nu$", "Cosmic"]
gibuu_mode_colors = ['#228833',  # QE
                     '#44AA99',  # QE (2p2h)
                     '#EE6677',  # RES
                     '#4477AA',  # DIS
                     '#AA3377',  # CC Other
                     '#66CCEE',  # NC
                     '#CCBB44',  # FV other ν
                     '#999933',  # Out-FV ν
                     '#BBBBBB']  # Cosmic


# nueNp0Pi PID/kinematic selection cuts (config/stages.py's build_pipeline() reco-cut
# stages, in analysis_village.nueNp0Pi.selections). Named here (instead of literals in
# selections.py) so config/stages.py's N_MINUS_1_VLINES can draw the exact same cut
# value as a threshold marker on the corresponding N-1 diagnostic plot -- single source
# of truth for "what is this cut's value" shared between the selection logic and the
# plot that visualizes it.
ELE_SOFTMAX_TH  = 0.9   # electron_softmax(): ele_softmax_reco > this
ELE_PRIMARY_TH  = 0.99  # electron_primary(): ele_primary_reco > this
P_SOFTMAX_TH    = 0.75  # proton_softmax(): proton_softmax_reco > this
ELE_VTXDIST_TH  = 3.5   # electron_vertex_distance(): ele_vertex_distance_reco < this (cm)
ELE_DEDX_TH     = 4     # electron_dedx(): ele_dedx_reco < this (MeV/cm)
ELE_KE_TH       = 500    # electron_kinetic_energy(): ele_ke_reco > this (MeV)
P_KE_TH         = 40    # proton_kinetic_energy(): proton_ke_reco > this (MeV)
MU_KE_TH        = 25   # muon_kinetic_energy(): muon_ke_reco > this (MeV)
PI_KE_TH        = 25    # pion_kinetic_energy(): pion_ke_reco > this (MeV)
GAM_KE_TH       = 25    # gamma_kinetic_energy(): gamma_ke_reco > this (MeV)
# pi0 sideband (makedf/make_nueccdf.py: _apply_pi0_sideband_cuts)
PI0_LEAD_GAM_KE_TH = 175  # leading photon KE > this (MeV)
PI0_SUB_GAM_KE_TH  = 25   # subleading photon KE > this (MeV)
