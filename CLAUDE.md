# Rules for working in this repo

## Processing and grid rules (always apply)

1. **Test before every full grid submission.** Before any full `jobsub_submit`-backed run
   (`run_df_maker.py -ngrid`, `run_event_selection_map.py -submit_grid`, `-skim -submit_grid`),
   run the same command locally on a small input (one CAF / one `.df` file, outputs in the
   session scratchpad) and check the output (`check_dfs.py`, row counts, logs). Report the test
   result to the user before asking to submit. Resubmissions of missing jobs
   (`-resubmit_missing`) reuse an already-tested setup and don't need a new test.
2. **Explicit confirmation before submitting to, or removing from, the grid.** Never run
   `jobsub_submit` (directly or through `run_df_maker.py -ngrid`, `-submit_grid`,
   `-resubmit_missing`), `jobsub_rm`, or any delete on grid storage (`ifdh rm`, `rm` under
   `/pnfs`, `xrdfs rm`) unless the user has said yes to that specific action in the current
   conversation. Show the exact command, the number of jobs and the input/output locations
   when asking. A yes to one submission does not cover the next one.
3. **Never move, rename, overwrite or delete files without an explicit instruction.** This
   covers user data, productions, plots and logs anywhere (`/exp`, `/pnfs`, the repo). Copying
   (e.g. `bin/stage_to_persistent.sh`, `ifdh cp` of grid outputs into a work base) is allowed
   when it is part of a step the user asked for; it must never overwrite an existing file.
   Write new outputs to new locations. The tools enforce this: `skim.py` and
   `stage_to_persistent.sh` refuse to overwrite and never delete. Exceptions: temporary
   files in the session scratchpad, and grid worker scripts (`bin/grid_executable*.sh`)
   removing their own partial output before retrying a failed copy, inside a job the user
   approved.

When a rule blocks a step, stop and ask; don't look for a workaround.

## Where things are
- Processing chain (CAF -> plots) and skills: `analysis_village/nueNp0Pi/README.md`
  ("Processing chain"); skills in `.claude/skills/`.
- Production log: `analysis_village/nueNp0Pi/PRODUCTIONS.md`. Add an entry for every
  production step (df maker, staging copy, skim, plots) — command, commit, every output
  location, validation.
- Python: `envs/venv_py310_cafpyana/bin/python`; grid steps need
  `source setup.sh` in the same command (see the `grid-env` skill).
