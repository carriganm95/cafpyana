from analysis_village.nueNp0Pi.makedf.make_nueccdf import *

# Background MC (in-time cosmics, dirt): only slices passing FV + flash match + containment,
# and only their particles. The evt function must come before the trk function (it records
# which slices passed). Not for the nominal signal MC: the pipeline needs every slice there
# for the full cutflow and the efficiency denominators -- use nuecc_mc.py.
DFS =   [make_nueNp0Pi_presel_df, make_spine_part_presel_df, make_hdrdf, make_potdf_bnb, make_mcnudf_nuecc]
NAMES = ["evt", "trk", "hdr", "pot", "mcnu"]
