# nueNp0Pi/df_maker_configs

Small config scripts for the repo-root **`.df`-production** CLI, `run_df_maker.py` — a different
tool from (and upstream of) the batched event-selection pipeline in `../event_selection.py`.
Each file here just sets two module-level names:

- `DFS` — a list of dataframe-maker functions (from `../makedf/make_nueccdf.py`, or in two
  cases a different analysis's maker, see below) to run against each input CAF ntuple.
- `NAMES` — the matching list of HDF5 key names each output dataframe gets saved under.

`run_df_maker.py` loads a config with `exec(open(args.config).read())` (see its `-c` flag),
so `DFS`/`NAMES` (and an optional `PREPROCESS` list, not used here) become globals it reads
directly — these files are not imported as normal Python modules.

Named `df_maker_configs/` (not `configs/`) specifically to disambiguate from `config/`
(singular) one directory up, which holds this analysis's downstream settings — unrelated,
different lifecycle, different naming was a source of confusion before the rename.

| File | `DFS` | `NAMES` | Notes |
|---|---|---|---|
| `nuecc.py` | `make_nueccdf_data, make_hdrdf, make_potdf_bnb` | `nuecc, hdr, pot` | Non-SPINE, data. |
| `nuecc_mc.py` | `make_nueNp0Pi_df, make_spine_part_df, make_hdrdf, make_potdf_bnb, make_mcnudf_nuecc` | `evt, trk, hdr, pot, mcnu` | **The current SPINE-based MC config** — output keys (`evt`, `trk`, `hdr`, `mcnu`) match what `../event_selection.py`'s `run_batch_selection` and `config/datasets.py`'s `KEYS2LOAD` expect. |
| `nuecc_mc_pi0_sideband.py` | `make_nueNp0Pi_pi0_sideband_df, make_spine_part_df, make_hdrdf, make_potdf_bnb, make_mcnudf_nuecc` | `evt, trk, hdr, pot, mcnu` | MC π⁰ sideband: selection without photon veto / electron dE/dx cut, plus 2 photons (leading > 175 MeV, subleading > 25 MeV). |
| `nuecc_data_pi0_sideband.py` | `make_nueNp0Pi_pi0_sideband_data_df, make_spine_part_reco_df, make_hdrdf, make_potdf_bnb` | `evt, trk, hdr, pot` | Data (reco-only) π⁰ sideband, same cuts. |
| `nuecc_debug.py` | `make_hdrdf, make_mcnu_nuecc` | `hdr, mcnu` | Imports from `analysis_village.nuecc.makedf.make_nueccdf` — a **different, older sibling analysis** (`nuecc`, not `nueNp0Pi`), not `../makedf/make_nueccdf.py`. |
| `nuecc_mc_wgt.py` | `make_nueccdf_mc_wgt, make_hdrdf, make_potdf_bnb, make_mcnudf` | `nuecc, hdr, pot, mcnu` | Also imports from `analysis_village.nuecc...`, not `../makedf/...`. Non-SPINE, with GENIE weights. |

If you're producing input for the current pipeline (`event_selection.py` / `efficiency.ipynb`),
use `nuecc_mc.py` for MC and `nuecc.py` for data — those are the two that call into this
analysis's own `../makedf/make_nueccdf.py`.

Example invocation (from the repo root, see `run_df_maker.py --help` for the full flag list):

```
python run_df_maker.py -c analysis_village/nueNp0Pi/df_maker_configs/nuecc_mc.py \
    -o my_output -i input_0.root,input_1.root
```
