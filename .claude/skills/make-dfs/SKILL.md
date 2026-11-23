---
name: make-dfs
description: Produce nueNp0Pi dataframe (.df) files from CAF files on the Fermilab grid - build the xrootd file list from filelists/datasets.json, check the df-maker code is committed, run a one-file local test, submit run_df_maker.py -ngrid, stage the output to pnfs persistent, and log the production. Use when making a new MC, data, cosmic, dirt, or detector-variation production.
---

# make-dfs: CAF files -> .df files (stages 1-2 of the processing chain)

See `analysis_village/nueNp0Pi/README.md` ("Processing chain") for where this sits. Run the
`grid-env` skill first. Follow the processing rules in `CLAUDE.md`: test before every full grid submission, get the user's explicit yes before submitting to or removing anything from the grid, and never move, overwrite or delete files without an explicit instruction.

## 1. Pick the dataset and build its list
```bash
cd $REPO && envs/venv_py310_cafpyana/bin/python filelists/make_filelist.py --list
envs/venv_py310_cafpyana/bin/python filelists/make_filelist.py <key>                 # filelists/filelist_<key>.txt
envs/venv_py310_cafpyana/bin/python filelists/make_filelist.py <key> --max-files 1   # filelist_<key>_test.txt
```
- `type` (mc/data) picks the config: mc -> `analysis_village/nueNp0Pi/df_maker_configs/nuecc_mc.py`,
  data -> `.../nuecc_data.py`. `role` (cv, data, offbeam, intime, dirt, detvar)
  says how the sample is used later. Every detector-variation sample (including its CVs) is
  `detvar`; there is no processing code for those yet, so don't produce them unless asked.
- For a SAM definition instead: `python data/sample_lists/create_list.py <definition> -e sbnd > filelists/filelist_<key>.txt`.
- Every line must start with `root://` (bare `/pnfs` paths fail on the grid):
  `grep -vc '^root://' filelists/filelist_<key>.txt` must print 0.
- Huge samples (in-time cosmics ~200k files) take minutes to list; that's expected.

## 2. Check the code that will run
`run_df_maker.py -ngrid` ships `git archive HEAD` — **committed code only**.
```bash
git status --porcelain analysis_village/nueNp0Pi/makedf analysis_village/nueNp0Pi/df_maker_configs makedf pyanalib
git log -1 --format='%h %s'
```
If anything is modified, stop and ask the user to commit (or confirm the committed version
is what they want). Record the commit hash.

## 3. One-file local test (required before every submission)
```bash
cd $REPO && source setup.sh >/dev/null 2>&1 && \
python run_df_maker.py -c analysis_village/nueNp0Pi/df_maker_configs/nuecc_<mc|data>.py \
  -o <scratchpad>/test_<key> -l filelists/filelist_<key>_test.txt -ncpu 1
python -m analysis_village.nueNp0Pi.check_dfs <scratchpad>/test_<key>.df --sample <mc|data>
```
Use the session scratchpad for the output. `check_dfs` must report OK (all tables, pi0 and
truth columns). Run long commands in the background.

## 4. Submit (only after the user says yes to this submission)
Choose `-ngrid` so each job gets ~4–10 CAFs (history: 799 MC CAFs -> 200 jobs; 1779 data CAFs
-> 200 jobs). Name: `nueNp0Pi_<key>_v<N>` (next unused version in `PRODUCTIONS.md`).
```bash
cd $REPO && source setup.sh >/dev/null 2>&1 && \
python run_df_maker.py -ngrid <N> -c analysis_village/nueNp0Pi/df_maker_configs/nuecc_<mc|data>.py \
  -o nueNp0Pi_<key>_v<N> -l filelists/filelist_<key>.txt
```
It prints the jobsub id, writes job scripts + `submit_info.json` to
`$CAFPYANA_GRID_OUT_DIR/logs/<ts>__<name>_log/`, and outputs go to
`$CAFPYANA_GRID_OUT_DIR/dfs/<ts>__<name>/<name>_K.df`. Then use the `grid-jobs` skill to watch
and recover failed jobs.

## 5. Stage to persistent storage (scratch gets purged; grid can't read /exp)
When all `<name>_K.df` exist (grid-jobs reports none missing):
```bash
cd $REPO && source setup.sh >/dev/null 2>&1 && \
bash bin/stage_to_persistent.sh $CAFPYANA_GRID_OUT_DIR/dfs/<ts>__<name> /pnfs/sbnd/persistent/users/$USER/nueNp0Pi/<name>
```
Copies only (never moves or deletes); skips files already copied and reports size
mismatches instead of overwriting. Run it in the background (hundreds of GB). The scratch
copy stays where it is — only remove it if the user asks.
Then validate with the `validate-dfs` skill.

## 6. Log it
Append an entry to `analysis_village/nueNp0Pi/PRODUCTIONS.md` (template at the top): dataset
key, command, commit, list file, grid log dir, every output location with count/size,
validation result. If this production should become the default input, update
`SPRING_GEN1_ROOT` / `DATA_GEN1_ROOT` defaults in `analysis_village/nueNp0Pi/config/datasets.py`
and the "Current defaults" section of the log — ask the user first.
