from analysis_village.nueNp0Pi.makedf.make_nueccdf import *

# SPINE-based DATA config: reco-only evt df (truth stripped) so the output
# keys (evt, trk, hdr) match the event-selection pipeline's KEYS2LOAD and
# config/datasets.py's "data" EVENT_SELECTION_GLOBS entry. make_hdrdf (not
# make_mchdrdf) and no mcnu, since data has no truth.
DFS =   [make_nueNp0Pi_data_df, make_spine_part_reco_df, make_hdrdf, make_potdf_bnb]
NAMES = ["evt", "trk", "hdr", "pot"]
