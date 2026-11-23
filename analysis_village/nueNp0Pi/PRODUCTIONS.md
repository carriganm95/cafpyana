# nueNp0Pi production log

One entry per production step, newest first, appended by the production skills
(`.claude/skills/make-dfs`, `make-skim`, `make-plots`) and by hand. Record what was run, from
which code, where every copy of the output lives, and whether it was validated. Paths here
are the source of truth for `SPRING_GEN1_ROOT` / `DATA_GEN1_ROOT` / `-work_base`.

Entry template:

```
## <YYYY-MM-DD> <stage>: <name>
- dataset / sample:
- command:
- code: <git commit> (<dirty?>)
- inputs:
- outputs: <every location, with file count and size>
- validation: <check_dfs / merge report / ...>
- status / notes:
```

---

## Current defaults (config/datasets.py)
- MC `.df`: `nueNp0Pi_mc_v4` — `/pnfs/sbnd/persistent/users/micarrig/nueNp0Pi/selection_v4/`
- data `.df`: `nueNp0Pi_data_v2` — `/pnfs/sbnd/persistent/users/micarrig/nueNp0Pi/selection_data_v2/`
- No skim of these yet.

---

## PLANNED v5: full reprocessing (decided 2026-11-23, not started)
One commit for every sample, all named `nueNp0Pi_<key>_v5`:
`mc1e20` (cv, `nuecc_mc.py`), `dev` (data, `nuecc_data.py`), `offbeam` (`nuecc_data.py`),
`cosmics` (in-time MC, `nuecc_mc.py`, ~200k CAFs), `dirt` (`nuecc_mc.py`).
Detector variations later. Before submitting: commit the working tree, one-file local test per
sample, explicit yes per submission (see `CLAUDE.md`).

## 2026-10-29 selection (full map, no skim): 20261029_v4
- command: `run_event_selection_map.py -submit_grid -selection both` (manifest created 2026-10-29 12:02)
- inputs: 400 files — `selection_v4` (mc, 200) + `selection_data_v2` (data, 200), pnfs persistent
- outputs: `/exp/sbnd/data/users/micarrig/nueNp0Pi/20261029_v4/` — `batches/` (50 mc + 1 data pickles),
  `batches_pi0_sideband/`, `plots/`, `plots_7cat/`, `agg_signal.log`, `agg_sideband.log`
- status: aggregated (signal exit 0). Made before the skim workflow and before the 2026-11-23
  sideband change (sideband dropped the PID-score / vertex cuts) — sideband plots here use the
  old sideband definition.

## 2026-10-28 dataframes: nueNp0Pi_mc_v4
- dataset: MC gen 1, 1e20 POT (`mc1e20` in `filelists/datasets.json`)
- command: `run_df_maker.py -ngrid 200 -o nueNp0Pi_mc_v4 ...` (config to confirm from
  `run_0.sh` in the log dir; expected `df_maker_configs/nuecc_mc.py`)
- code: committed HEAD at submission = `dc4d398` (pi0 columns). The df-maker truth-matching fix
  (`truth_match_spine_nu_id`, committed later in `6fdd248`) is NOT in these files; the pipeline's
  `rematch_mc_columns` corrects `mc.*` on load, so they are still usable.
- grid log dir: `/pnfs/sbnd/scratch/users/micarrig/cafpyana_out/logs/2026_10_28_084259__nueNp0Pi_mc_v4_log`
  (earlier attempts 2026_10_27_151509 / 184536 / 200157 also exist)
- outputs (200 files each):
  - scratch: `/pnfs/sbnd/scratch/users/micarrig/cafpyana_out/dfs/2026_10_28_084259__nueNp0Pi_mc_v4/` (433 GB)
  - persistent: `/pnfs/sbnd/persistent/users/micarrig/nueNp0Pi/selection_v4/`
  - BlueArc: `/exp/sbnd/data/users/micarrig/nueNp0Pi/2026_10_28/` (465 GB)
- validation (2026-11-23): `check_dfs` on the BlueArc copy, 3/3 files OK (all tables and
  columns incl. pi0). ~300–420k slices per file.

## 2026-10-27 dataframes: nueNp0Pi_data_v2
- dataset: BNB-light dev data (`dev`; list `filelists/filelist_data_bnblight.txt`, 1779 CAFs)
- command: `run_df_maker.py -ngrid 200 -o nueNp0Pi_data_v2 ...` (config to confirm; expected
  `df_maker_configs/nuecc_data.py`)
- code: committed HEAD at submission = `dc4d398`
- grid log dir: `/pnfs/sbnd/scratch/users/micarrig/cafpyana_out/logs/2026_10_27_200048__nueNp0Pi_data_v2_log`
- outputs (200 files each, 2.6 GB):
  - scratch: `/pnfs/sbnd/scratch/users/micarrig/cafpyana_out/dfs/2026_10_27_200048__nueNp0Pi_data_v2/`
  - persistent: `/pnfs/sbnd/persistent/users/micarrig/nueNp0Pi/selection_data_v2/`
  - BlueArc: `/exp/sbnd/data/users/micarrig/nueNp0Pi/selection_data_v2/`
- validation (2026-11-23): `check_dfs` on the BlueArc copy, 3/3 files OK (incl. pi0 columns).

---

## Superseded (no pi0 columns — don't use for the current pipeline)
- `selection_v3` (MC, 160 files, 427 GB, `/exp/sbnd/data/users/micarrig/nueNp0Pi/selection_v3/`,
  grid output `2026_08_21_112409__output_v1`) — `check_dfs`: missing `pi0_mass_reco`,
  `subprim_photon_energy_reco`.
- `selection_data_v1` (data, 180 files) — same missing columns.
- `selection_skim_v3` (160 files, 272 GB, plus `merged_output.h5`) — older skim of `selection_v3`
  made outside this workflow; purpose not recorded.
