# Constants for this analysis
# Today: 2026-02-04

import uproot
import matplotlib.pyplot as plt
from makedf.constants import *
from os import path
import pandas as pd
from pyanalib.split_df_helpers import *
from pyanalib.pandas_helpers import *

plot = False

DETECTOR = "SBND_nohighyz"
EPSILON = 1e-6 # for clipping distributions at bin ranges

# xsec-unit calculation (flux integration, active volume, N targets) now lives in
# makedf.flux.get_xsec_unit; this analysis's flux-file path is in
# analysis_village.nueNp0Pi.paths_config.FLUX_FILE. This used to be a dead
# commented-out duplicate of analysis_village.nueNp0Pi.utils.get_xsec_unit -- removed.