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
`mc1e20` (cv, `nuecc_mc.py`), `dev` (data, `nuecc_data.py`), `offbeam` (`nuecc_data_presel.py`),
`cosmics` (in-time MC, `nuecc_mc_presel.py`, ~200k CAFs), `dirt` (`nuecc_mc_presel.py`).
Background samples use the preselected configs (decided 2026-11-24): only slices passing
FV + flash match + containment are written, so their cutflow starts at containment; header
POT/gates are complete.
Detector variations later. Before submitting: commit the working tree, one-file local test per
sample, explicit yes per submission (see `CLAUDE.md`).

- 2026-11-23 file lists (`filelists/make_filelist.py`): mc1e20 799, dev 1779 (identical to
  `filelist_data_bnblight.txt`), offbeam 34,923, dirt 113,066 (path
  `mc_MCP2025B_..._lowenergydirt_sbnd_CV_caf_sbnd/*/*/*ar23p.flat.caf.root`), cosmics 203,841.
- 2026-11-24 one-CAF local tests, code `2ed1802` (branch `nueNp0pi_v5`, df-maker code clean),
  outputs in `/exp/sbnd/data/users/micarrig/nueNp0Pi/v5_tests/<key>/` (`run_tests2.log`):
  all exit 0 and pass `check_dfs`; header normalisation present for each role.

  | key | config | time | peak RAM | output | slices | hdr |
  |---|---|---|---|---|---|---|
  | dev | nuecc_data | 14 s | 0.4 GB | 1.8 MB | 629 | POT 2.8e15, 736 BNB gates |
  | offbeam | nuecc_data | 9 s | 0.4 GB | 1.8 MB | 617 | 2545 off-beam gates |
  | dirt | nuecc_mc | 11 s | 0.4 GB | 2.2 MB | 171 | POT 3.0e15, mcnu 23 |
  | cosmics | nuecc_mc | 11 s | 0.4 GB | 2.2 MB | 163 | 500 generated events, mcnu empty |
  | mc1e20 | nuecc_mc | 102 s | 2.5 GB | 491 MB | 88,409 | POT 7.5e17, mcnu 13,297 |

- 2026-11-24 preselected configs (`nuecc_{mc,data}_presel.py`, uncommitted at test time), same
  CAFs, outputs in `v5_tests/<key>_presel/`: kept slices identical to applying the three cuts
  to the full test output, tracks only of kept slices, header unchanged; check_dfs OK.
  offbeam 617->0 slices, dirt 171->4, cosmics 163->0 per CAF. 20-CAF dirt test
  (`v5_tests/dirt_presel20/`): 3m30s, 0.4 GB peak, 1.75 MB (~0.02 MB per extra CAF) ->
  ~9 MB per ~380-CAF job; all three background samples ~10 GB total (vs ~760 GB unselected).

  Note: reading the CAF directly over xrootd (instead of a local copy, as grid jobs do) left
  the process hung at exit after finishing; local copies exit cleanly.
- 2026-11-24 first mc1e20 submission attempt (`2026_11_24_093021__nueNp0Pi_mc1e20_v5`) hung
  before `jobsub_submit` (writing the job bundle through the /pnfs mount blocks forever; the
  process was killed, 0 jobs submitted). Empty dirs left in `$CAFPYANA_GRID_OUT_DIR/logs/` and
  `/dfs/` under that name. Fix (uncommitted at test time): bundle + metadata now built in
  `$CAFPYANA_GRID_JOB_DIR` (`/exp/sbnd/data/users/$USER/cafpyana_grid`), and one `filelist.txt` +
  `job.env` replace the per-job `run_K.sh` (job K takes lines K, K+N, ...). Local test: 3-job
  dev bundle, `grid_executable.sh` run as PROCESS=1 in a clean env
  (`v5_tests/grid_worker_test/`): picked CAF #2 (run18255_2), exit 0, `gtest_1.df` 592 slices,
  `check_dfs` OK; `-resubmit_missing` on the bundle mapped renumbered jobs 0,1 -> 1,2 correctly.

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
