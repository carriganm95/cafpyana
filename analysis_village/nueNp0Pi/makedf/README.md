# nueNp0Pi/makedf

Turns raw CAF ntuples (uproot-loaded `recTree`) into this analysis's `.df` dataframes. This is
the PRODUCTION step upstream of everything else in `nueNp0Pi/` — its output (`.df` HDF5 files)
is what `config/datasets.py`'s globs point at and what `event_selection.py` loads. Driven by
`../../../run_df_maker.py` via the `DFS`/`NAMES` lists defined in `../df_maker_configs/` (see
that directory's README).

## `make_nueccdf.py`

- `TRUE_KE_THRESHOLDS` — per-particle true kinetic-energy thresholds (muon 25 MeV, proton 40 MeV,
  charged pion 25 MeV, electron 500 MeV, photon 100 MeV) used to build primary-particle-multiplicity
  columns on the truth (`mcnu`) dataframe.
- `make_mcnudf_nuecc(f, **args)` — wraps `makedf.makedf.make_mcnudf`, adding the per-threshold
  particle-count columns above from the primary-particle branches.
- `make_nueccdf_mc(f, ...)` / `make_nueccdf_mc_wgt(f)` / `make_nueccdf_data(f)` — older,
  non-SPINE dataframe makers (slice-level + truth-matched MC, or data with a frame-quality
  column merged in). Used by some `df_maker_configs/` entries; largely superseded by
  `make_nueNp0Pi_df` below for the current SPINE-based pipeline.
- `broadcast_to_particles(interaction_mask, df)` — broadcasts an interaction-level boolean mask
  down to every particle row of a particle-level dataframe.
- `get_highest_ke_particle_idx` / `get_second_highest_ke_particle_idx` — per-interaction,
  return the particle index of the highest/second-highest-KE particle of a given PDG (reco
  `dlp` or truth `dlp_true` particles).
- **`make_nueNp0Pi_df(f)`** — the main SPINE-based dataframe maker actually used by
  `df_maker_configs/nuecc.py` / `nuecc_mc.py`. Asserts `DETECTOR == "SBND"` (reads
  `rec.hdr.det`), builds `slcdf` (interaction-level, via `make_spine_int_df`) and `pfpdf`
  (particle-level, via `make_spine_part_df`), tags the leading (and for protons, subleading)
  reco/truth particle of each species (electron, muon, pion, proton, photon) per interaction,
  then derives ~40 truth/reco boolean category columns (`true_signal1p`, `true_signalNp`,
  `bkgd_oofv`, `bkgd_oops`, `bkgd_pi`, `bkgd_mu`, `bkgd_photon`, `bkgd_nueOther`, `bkgd_numu`,
  `bkgd_nc`, `bkgd_other`, plus per-particle truth/reco energy/momentum/PID columns and TKI
  variables via `pyanalib.variable_calculator`). Applies loose pre-selection cuts
  (`is_fiducial`, `is_flash_matched`, `is_contained`) before returning. This is the function
  whose OUTPUT `event_selection.py` consumes downstream — if you change a column name or
  category definition here, check `../selections.py` and `../config/plots.py` for anything that
  reads it.

**Note**: `df_maker_configs/nuecc_debug.py` and `nuecc_mc_wgt.py` import from
`analysis_village.nuecc.makedf.make_nueccdf` (a *different*, older sibling analysis, `nuecc`,
not `nueNp0Pi`) rather than this file — flagged here since it's easy to miss, not something this
documentation pass changed.
