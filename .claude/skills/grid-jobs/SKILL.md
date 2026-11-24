---
name: grid-jobs
description: Monitor nueNp0Pi grid jobs and recover failures - jobsub_q status, find which df-maker or skim jobs produced no output, read their logs, and resubmit only the missing ones (run_df_maker.py -resubmit_missing, run_event_selection_map.py -skim -resubmit_missing). Use after make-dfs or make-skim submissions, or when the user asks about job status.
---

# grid-jobs: watch, diagnose, resubmit

Run `grid-env` first (token). Follow the processing rules in `CLAUDE.md`: test before every full grid submission, get the user's explicit yes before submitting to or removing anything from the grid, and never move, overwrite or delete files without an explicit instruction. Resubmitting missing jobs reuses an already
tested setup (no new test needed) but still needs a yes. Never `jobsub_rm` jobs or delete
outputs/logs on pnfs unless the user asks.

## Status
```bash
jobsub_q -G sbnd --user $USER                     # running / idle / held
jobsub_q -G sbnd --user $USER --hold              # held jobs (usually memory/disk/lifetime)
```
Jobs finishing isn't proof of success — always count outputs (below).

## Which jobs are missing

**df maker** (outputs `<name>_K.df`, K = 0..N-1):
```bash
LOG=$CAFPYANA_GRID_JOB_DIR/logs/<ts>__<name>_log   # before 2026-11-24: $CAFPYANA_GRID_OUT_DIR/logs/...
OUT=$CAFPYANA_GRID_OUT_DIR/dfs/<ts>__<name>
N=$(grep N_JOBS $LOG/job.env | cut -d= -f2); have=$(ls $OUT/*.df 2>/dev/null | wc -l); echo "$have / $N"
```
`run_df_maker.py -resubmit_missing $LOG` also prints the missing indices before submitting.

**skim** (`run_event_selection_map.py -skim -submit_grid`): outputs land in pnfs
`$CAFPYANA_GRID_OUT_DIR/event_sel_batches/<ts>` (+ `_pi0_sideband`); after pulling them into
`<W>/batches` (see `make-skim`), `run_event_selection_map.py -skim -resubmit_missing -work_base <W>`
checks every job in `<W>/manifest.json` and resubmits the missing/corrupt ones.

## Why did a job fail
- Per-job log: df maker copies `log_K.log` next to the outputs (`$OUT/log_K.log`); skim jobs
  copy theirs to `<grid out>/map_logs/`. Read it via xrootd or after copying it to the
  scratchpad (reading through the /pnfs mount can hang).
- Full condor log: `jobsub_fetchlog -G sbnd --jobid <cluster>.<process>@<schedd> --destdir <scratchpad>/logs`.
- Common causes: expired token at submission (auth errors on xrdcp/ifdh), an input CAF that
  no longer exists, memory (job held — data files are small, MC ones ~2.5 GB), code error
  (same traceback in every log -> fix, commit, resubmit; for the df maker a code fix means a
  NEW submission, since resubmission reuses the original code bundle).

## Resubmit missing jobs
**df maker** (reuses the original `cafpyana.tar.gz`, `filelist.txt` and output names; jobs are
renumbered and mapped back with `process_map.txt`; the new bundle goes in
`$CAFPYANA_GRID_JOB_DIR/logs/<ts>__<name>_resubmit_log`). Old submissions whose log dir is on
/pnfs can't be resubmitted from the gpvm (copying their bundle reads /pnfs, which hangs):
```bash
cd $REPO && source setup.sh >/dev/null 2>&1 && python run_df_maker.py -resubmit_missing $LOG
```
**skim**:
```bash
cd $REPO && source setup.sh >/dev/null 2>&1 && python run_event_selection_map.py -skim -resubmit_missing -work_base <W>
```
Note the new jobsub id and log dir in `analysis_village/nueNp0Pi/PRODUCTIONS.md` under the
production's entry (e.g. "resubmitted 7 jobs on <date>: <log dir>").
