---
name: make-plots
description: Make the nueNp0Pi selection plots from an existing Parquet skim - run the selection over the skim for the signal and pi0 sideband selections, optionally the generator-level truth histograms, then aggregate and render plots and tables, and log the run. Use whenever a cut after containment, a plot, a binning, or a breakdown changes, or the user asks for updated plots.
---

# make-plots: skim -> histograms -> plots (stages 4-5)

All local; takes minutes. Needs a merged skim (`make-skim`); find the current one in
`analysis_village/nueNp0Pi/PRODUCTIONS.md`. If the change touches a cut at or before
`SKIM_STAGE_KEY` (FV, flash match, containment), a new skim is needed instead.

Names: `W` = skim work base (holds `skim/`), `R` = histogram output dir, `P` = plots dir.
Suggested: `R=<W>/runs/<YYYYMMDD>_<label>`, `P=<R>/plots` (sideband: `<R>_pi0_sideband`,
`<P>_pi0_sideband`). Never overwrite an earlier run's `R`/`P`: pick a new label (`-from_skim` refuses to write
into a directory that already has pickles). Follow the processing rules in `CLAUDE.md`: test before every full grid submission, get the user's explicit yes before submitting to or removing anything from the grid, and never move, overwrite or delete files without an explicit instruction.

## 1. Selection over the skim
```bash
cd $REPO && envs/venv_py310_cafpyana/bin/python run_event_selection_map.py \
  -from_skim <W>/skim -out_dir <R> -selection both -nproc 4
```
Writes `<R>/<sample>__skim_NNNN.pkl` + `<sample>__preskim.pkl`, and the same under
`<R>_pi0_sideband/`. `-selection signal` or `pi0_sideband` for just one. Keep `-nproc` ≤ 4
on the interactive node (~11 GB RAM). If it fails with a `KeyError` on a column, the
`.df` production behind the skim predates that column (check with `validate-dfs`).

## 2. Generator-level truth (optional, once per MC production)
Adds a second "true νe CC ≥1e ≥1p" line to the truth plots. Skip if
`<P>/pkl/truth/mcnu_truth_hists.pkl` already exists for this MC production (copy it from an
earlier run's `P` instead). Needs the original MC `.df` files — use the BlueArc copy listed in
`PRODUCTIONS.md` (reading through /pnfs can hang):
```bash
cd $REPO && envs/venv_py310_cafpyana/bin/python -m analysis_village.nueNp0Pi.truth_mcnu \
  --files '<BlueArc mc dir>/*.df' --out <P>/pkl/truth/mcnu_truth_hists.pkl --nproc 4
```
Must run before step 3 (the aggregate step picks the pickle up from `<P>`).

## 3. Aggregate and render
```bash
cd $REPO && envs/venv_py310_cafpyana/bin/python -m analysis_village.nueNp0Pi.event_selection_aggregate \
  --in_dir <R> --out_dir <P>
cd $REPO && envs/venv_py310_cafpyana/bin/python -m analysis_village.nueNp0Pi.event_selection_aggregate \
  --in_dir <R>_pi0_sideband --out_dir <P>_pi0_sideband
```
The selection mode is read from the pickles. Systematic bands are off by default
(`config.stages.DISABLE_SYST_BANDS`); pass `--syst` only once real nueNp0Pi systematics exist.
Check the log for `exit`/tracebacks and that the POT line matches the skim's POT
(`<W>/skim/skim_manifest.json`).

## 4. Report and log
- Tell the user where the plots are (`<P>`, `<P>_pi0_sideband`) and show 2–3 key ones
  (e.g. the final-stage selection plots and the cutflow table) with the Read tool.
- Append to `analysis_village/nueNp0Pi/PRODUCTIONS.md`: skim used (`W`), `R`, `P`, commit +
  `git status --short` summary (plots depend on uncommitted config edits), what changed
  (e.g. "proton KE binning"), POT.
