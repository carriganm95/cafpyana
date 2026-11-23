---
name: make-skim
description: Make the nueNp0Pi Parquet skim from a .df production on the Fermilab grid - submit run_event_selection_map.py -skim, pull the outputs back, resubmit missing jobs, merge into one Parquet file per sample, and log it. Use after a new or changed .df production, or after changing any cut at or before config.stages.SKIM_STAGE_KEY (FV, flash match, containment).
---

# make-skim: .df files -> Parquet skim (stage 3)

Run `grid-env` first. Follow the processing rules in `CLAUDE.md`: test before every full grid submission, get the user's explicit yes before submitting to or removing anything from the grid, and never move, overwrite or delete files without an explicit instruction. Details of the skim design:
`analysis_village/nueNp0Pi/README.md` ("Skim workflow") and `analysis_village/nueNp0Pi/skim.py`.

## When a new skim is needed
- a new `.df` production, or
- a change to a cut at or before `SKIM_STAGE_KEY` (`config/stages.py`, currently
  `is_contained`), or to the columns added in `event_selection.load_prepared_file`.
Anything after containment only needs `make-plots`.

## 1. Inputs and work base
- Inputs come from `SPRING_GEN1_ROOT` (MC) and `DATA_GEN1_ROOT` (data); defaults in
  `config/datasets.py` and `analysis_village/nueNp0Pi/PRODUCTIONS.md` ("Current defaults").
  They must be **pnfs** paths (grid jobs can't read `/exp/...`).
- Work base `W`: `/exp/sbnd/data/users/$USER/nueNp0Pi/skim_<mc name>_<data name>` (one per
  input combination). Detector variations / other samples get their own `W` (run them with
  `-samples mc` and `SPRING_GEN1_ROOT` pointing at that sample's `.df` directory).
- Grid jobs ship the **working tree** (uncommitted edits included); note `git status` and the
  commit in the log.

## 2. Test locally first (required before every submission)
One small file per sample, outputs in the session scratchpad (`T`). A full MC `.df` needs ~5 GB
of RAM, so cut a test file first; data files are small enough to use as they are:
```bash
cd $REPO && P=envs/venv_py310_cafpyana/bin/python
$P -m analysis_village.nueNp0Pi.make_test_df <one mc .df (BlueArc copy)> $T/in/test_mc.df
$P run_event_selection_map.py -skim -sample mc   -job_id 0 -out_dir $T/W/batches -files $T/in/test_mc.df
$P run_event_selection_map.py -skim -sample data -job_id 0 -out_dir $T/W/batches -files <one data .df>
$P run_event_selection_map.py -merge_skim -work_base $T/W
$P run_event_selection_map.py -from_skim $T/W/skim -out_dir $T/R -selection both
```
Report to the user: slices in -> kept per sample (expect ~2–3%), the merge summary, and that
from-skim ran for both selections. Only then ask to submit.

## 3. Submit (only after the user says yes to this submission)
```bash
cd $REPO && source setup.sh >/dev/null 2>&1 && \
SPRING_GEN1_ROOT=<mc dir> DATA_GEN1_ROOT=<data dir> \
python run_event_selection_map.py -skim -submit_grid -work_base <W> -samples mc,data -max_job_bytes_gb 10
```
It surveys the inputs into `<W>/manifest.json`, submits one job per batch, and prints the pnfs
output dir (`$CAFPYANA_GRID_OUT_DIR/event_sel_batches/<ts>`) and the exact pull-back commands.
Watch with the `grid-jobs` skill.

## 4. Pull outputs back
Use the commands printed at submission (both lines; `-skim` always writes signal and
pi0-sideband pickles):
```bash
ifdh cp -r <grid dir>/ <W>/batches/
ifdh cp -r <grid dir>_pi0_sideband/ <W>/batches_pi0_sideband/
```
(`ifdh` from cvmfs — see `bin/stage_to_persistent.sh` for the PATH lines.) Copy only; if a destination file already exists, stop and ask. Then
`-skim -resubmit_missing -work_base <W>` for anything missing (ask first), and pull again.

## 5. Merge
```bash
cd $REPO && envs/venv_py310_cafpyana/bin/python run_event_selection_map.py -merge_skim -work_base <W>
```
Refuses to run if any batch pickle or Parquet part is missing/corrupt. Writes
`<W>/skim/{sample}_{evt,trk}.parquet`, `<W>/skim/preskim/<mode>/<sample>__preskim.pkl` and
`<W>/skim/skim_manifest.json` (rows, POT, sizes per sample). Expect ~2.4% of slices kept and
roughly 6–7 GB for the full MC.

## 6. Log it
Append to `analysis_village/nueNp0Pi/PRODUCTIONS.md`: input `.df` productions, `W`, jobsub
id(s), commit + dirty state, numbers from `skim_manifest.json` (rows, POT, GB per sample),
resubmissions. Then hand over to `make-plots`.
