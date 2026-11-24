from analysis_village.nueNp0Pi.makedf.make_nueccdf import *

# Background data (off-beam): only slices passing FV + flash match + containment, and only
# their particles. The evt function must come before the trk function. For beam data use
# nuecc_data.py (keeps every slice).
DFS =   [make_nueNp0Pi_presel_data_df, make_spine_part_reco_presel_df, make_hdrdf, make_potdf_bnb]
NAMES = ["evt", "trk", "hdr", "pot"]
