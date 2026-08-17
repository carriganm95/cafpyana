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


# ===========================================================================
# Reconstruction-level cut thresholds.
# ===========================================================================
# slice cuts
NU_SCORE_TH   = 0.45
SAVE_NTRKS    = 2
# track quality cuts
TRACKSCORE_TH = 0.5
VTXDIST_TH    = 1.2
# pid cuts
MU_CHI2MU_TH  = 25
MU_CHI2P_TH   = 100
MU_LEN_TH     = 50
QUAL_TH       = 0.2
P_CHI2P_TH    = 90
P_LEN_TH      = 0
# kinematic cuts
MU_PLO_TH     = 0.22
MU_PHI_TH     = 1
P_PLO_TH      = 0.3
P_PHI_TH      = 1

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
