# nueNp0Pi

SBND ν_e CC N-proton, 0-pion (`nueNp0Pi`) cross-section analysis. This directory has three
layers: **config** (what's configurable), **logic/runner** (how the pipeline actually runs), and
**scratch/production tooling** (notebooks, the ntuple → `.df` dataframe maker).

If you're new here and just want to run the event-selection pipeline, read the "What you'll
probably need to change" section below first, then open `notebooks/efficiency.ipynb`.

## Directory map

```
nueNp0Pi/
├── config/                    Settings + pipeline definition (see config/README.md)
├── makedf/                    Turns raw CAF ntuples into this analysis's .df dataframes (see makedf/README.md)
├── df_maker_configs/          DFS/NAMES snippets exec()'d by ../../run_df_maker.py (see df_maker_configs/README.md)
├── notebooks/                 efficiency.ipynb (maintained) + personal scratch notebooks (see notebooks/README.md)
├── dataset_paths.py           Thin wiring: binds config/datasets.py's globs to pyanalib.dataset_paths
├── selections.py              Cut/predicate/truth-category FUNCTIONS (the logic; settings live in config/settings.py)
├── event_selection.py         Pipeline RUNNER: map-phase core + EventSelectionPipeline factory
├── event_selection_aggregate.py   Reduce/render phase: merges chunk histograms into plots/tables
├── evt_derived_kinematics.py  Adds derived reco/truth kinematic columns (phi, opening angles, …) some VariableConfigs expect
└── utils.py                   Thin wiring: binds pyanalib's syst_loading/overlay_plotting engines to this analysis's config
```

Canonical `VariableConfig` sets for final-sample plots/efficiency curves used to live in a
separate `final_selected_evt_vars.py`; they're now part of `config/plots.py` (see
`config/README.md`) — there was no longer a good reason to split "variable definitions" across
two files.

## What each top-level file does

**`dataset_paths.py`** — Binds this analysis's glob dicts (from `config/datasets.py`) and
env-var names to the generic path-resolution helpers in `pyanalib/dataset_paths.py`. Exposes
`iter_event_selection_df_paths`, `iter_genie_group_df_paths`, `iter_detvar_chunk_jobs`,
`iter_cosmics_chunk_df_paths`, `iter_multisim_*`, `sorted_glob`, and one `default_*_work_root()`
function per systematics workflow (event selection, GENIE, cosmics, multisim, detvar, joint
GENIE/multisim CC, syst-disk root). Run `python -m analysis_village.nueNp0Pi.dataset_paths` for
a human-readable summary of every configured dataset/glob and how many files each resolves to
right now — useful for sanity-checking a path change.

**`selections.py`** — Every cut/predicate/truth-category FUNCTION used by the pipeline:
fiducial-volume checks, PID/kinematic cuts (`no_muons`, `good_electron`, `electron_softmax`, …),
truth-topology categorization (`get_topo_category_nueNp0Pi`, `get_pdg_category_spine`,
`get_genie_category_spine`), and `SIGNAL_MASK_FN` (the signal definition consumed by
`pyanalib.chunked_selection.ChunkRunner`). `BREAKDOWN_REGISTRY` (which breakdown categories
exist, and which of this file's functions computes each one) lives in `config/stages.py` instead
of here — see that module's docstring for why (a circular-import constraint: it needs both this
file's category functions and settings.py's label lists). The numeric thresholds these functions
use as default args (e.g. `NU_SCORE_TH`, `MU_CHI2MU_TH`) live in `config/settings.py`, imported
explicitly at the top of this file — **edit the threshold there, not the default-arg value
here.** Note: this file has two `def IsNu(df):` definitions (a pre-existing quirk, not a bug
introduced during refactoring) — Python keeps the second one (`~df.mc.pdg.isna()`-based); see the
comment above it.

**`event_selection.py`** — The pipeline RUNNER (how stages get applied, not what they are — that
lives in `config/stages.py`). Contains:
  - `ChunkRunner` — nueNp0Pi-flavored subclass of `pyanalib.chunked_selection.ChunkRunner`,
    defaulting to `selections.BREAKDOWN_REGISTRY`/`SIGNAL_MASK_FN`.
  - `build_runner(sample)` — builds a `ChunkRunner` from `config.stages.build_pipeline()`.
  - `run_batch_selection(...)` — the actual map-phase: loads a batch of `.df` files, runs the
    pipeline, writes one pickle. Also holds the header/weight/kinematics helpers it needs
    (`hdr_chunk_pot`, `intrinsic_weight_series`, `ensure_phi_and_kinematics_cols`, …).
  - `build_event_selection_pipeline(cfg)` — factory that wires `run_batch_selection`,
    `dataset_paths.iter_event_selection_df_paths`, and
    `event_selection_aggregate.aggregate_and_render` into a generic
    `pyanalib.event_selection_pipeline.EventSelectionPipeline`. **This is the function
    notebooks call.**

**`event_selection_aggregate.py`** — The reduce/render phase: merges per-chunk histogram
pickles across all batches (`aggregate_chunk_files`/`merge_samples` from
`pyanalib.chunked_selection`), applies global POT/exposure scaling, and renders every plot
(overlay stacks, efficiency curves, breakdown bar charts, chi2 tables). `render_efficiency_plots`
writes three views under `<plots_dir>/`: `combined/`, `event_counts/`, `efficiency/`, each
holding one plot file per variable. Kept as a separate file from `event_selection.py` deliberately:
it's a big, distinct concern (turns histograms into matplotlib figures), not part of "how do
I run the selection."

**`evt_derived_kinematics.py`** — `ensure_derived_trk_kinematics_cols` / `ensure_mc_level_phi_mcnu`
add reco/truth kinematic columns (`phi`, opening angles, …) that some `config/plots.py`
`VariableConfig`s expect but that older/pre-stored `.df` files may not have computed yet. Called
once per loaded batch in `event_selection.py`'s map phase.

**`utils.py`** — A thin wiring file (NOT the systematics/plotting logic itself anymore — that
moved to `pyanalib/syst_loading.py` and `pyanalib/overlay_plotting.py`, see
`../../pyanalib/README.md`). Binds those two generic engines to this analysis's own config:
category label/color registry (`BREAKDOWN_DISPLAY`, built from `config/settings.py`'s
`pdg_labels`/`topology_labels`/`genie_mode_labels`/`genie_sb_mode_labels` and colors), the
`_DEFAULT_SYST_DISK_ROOT` fallback, and three analysis-specific loader hooks that wrap imports of
modules that don't currently exist in this repo (see the module docstring's "pre-existing bugs"
list — preserved as-is from before the split, not introduced by it, and not silently fixed).
Re-exports `get_syst_unc`, `overlay_hists_from_histdata`, `get_clipped_evts`,
`get_pot_str`/`generate_tags`, and the category-summary/GENIE-SB covariance loaders under their
original names, so existing imports (including the notebook's
`from analysis_village.nueNp0Pi.utils import *`) keep working. Deliberately re-imports
`DETECTOR`/`EPSILON`/`save_fig_base_dir` from `config/datasets.py` **after** a wildcard import of
`config/settings.py`, to shadow `config/settings.py`'s own `DETECTOR` — this is intentional and
tested (`tests/test_nueNp0Pi_utils.py::test_detector_constant_not_silently_shadowed_incorrectly`),
not a leftover bug.

## What you'll probably need to change

Everything below lives in `config/` — see `config/README.md` for the full settings reference.
This is just the "if you're getting started, here's what to look at" shortlist:

1. **Where your input `.df` files are** — `config/datasets.py`: `SPRING_GEN1_ROOT` for MC and
   `DATA_GEN1_ROOT` for data (env vars of the same names; the historical
   `NUMUCC_SPRING_GEN1_ROOT` is still accepted for MC — see note below), and the
   `EVENT_SELECTION_GLOBS` / `SELECTED_EVENTS_GLOBS` / `MULTISIM_SYST_GLOBS_*` /
   `GENIE_GROUP_GLOBS` / `DETVAR_WIREMOD_GLOBS` dicts that build actual glob patterns from it.
   Easiest override without editing the file: `export SPRING_GEN1_ROOT=/your/path`.
2. **Where plots get saved** — `config/datasets.py`: `PLOTS_BASE` (env var
   `NUMUCC_PLOTS_BASE`) and `save_fig_base_dir`. Also `EventSelectionPipelineConfig.plots_dir`
   if you're calling `build_event_selection_pipeline` directly from a notebook/script.
3. **Selection cut thresholds** (electron energy, proton energy, PID scores, vertex distance,
   dE/dx, …) — `config/settings.py`, the `*_TH` constants. The actual cuts that use them are in
   `selections.py`, but you should only ever need to touch the number in `config/settings.py`.
4. **Detector / fiducial-volume convention** — `DETECTOR` is defined TWICE on purpose:
   `config/settings.py` (`"SBND_Gen1"`, used by truth/reco cut functions in `selections.py`) and
   `config/datasets.py` (`"SBND_nohighyz"`, shadowed into `utils.py`'s systematics code). If you
   change the detector geometry tag, check both.
5. **What cuts/stages run and what plots each stage produces** — `config/stages.py`:
   `build_pipeline()`. Add a `Stage(...)` to add/remove a cut; append a `PlotSpec(...)` to a
   stage's `plots` list to add a plot at that stage. For a stacked event-distribution breakdown
   plot (as opposed to `sel_primary_trks`'s per-particle plots), use the
   `evt_breakdown_plot(var_config, breakdown_type)` helper (also in `config/stages.py`) and
   append its result to whichever stage's `plots` list you want the post-cut distribution from —
   nothing is wired in by default yet, since the variables/breakdown types to use for these
   aren't decided.
6. **Category/breakdown labels & colors** (particle-type, topology, GENIE mode, …) —
   `config/settings.py`: `pdg_labels`/`colors`, `topology_labels`/`colors`,
   `genie_mode_labels`/`colors`, etc.
7. **Batch-pipeline output/work directories** (where map-phase pickles land before aggregation)
   — the `default_*_work_root()` functions in `dataset_paths.py`, each overridable by its own
   env var (`NUMUCC_EVENT_SELECTION_WORK_BASE`, `NUMUCC_GENIE_SYST_WORK_BASE`, etc.), or pass
   `work_base=` directly on `EventSelectionPipelineConfig`.
8. **Systematics-covariance disk root** — the `SYST_DISK_ROOT` env var (see
   `pyanalib/syst_disk_layout.py`), read by `utils.get_syst_unc` and
   `dataset_paths.default_syst_disk_root()`.

**Historical naming note:** many env-var names and default work-root paths in
`config/datasets.py` / `dataset_paths.py` still say `NUMUCC_*` / `numucc_1p0pi` even though this
is the nueNp0Pi analysis — this file was originally copied from the `numucc_1p0pi` analysis and
the names were deliberately left as-is (documented in `config/datasets.py`'s own docstring) to
avoid breaking anyone's existing shell/grid setup that already sets them. A full rename is
possible but hasn't been done. `FLUX_FILE` is the one path that already uses a `NUENP0PI_*`
env var name.

## Processing chain: CAF files to plots

![Processing chain from CAF files to plots](docs/processing_chain.png)

Each band is one stage: a processor (the code that runs) produces an output (the files it
saves), which feeds the next stage. The image is drawn by `docs/make_processing_chain.py`;
edit the `STAGES` list there and rerun it to update the picture.

Notes:
- The df maker's grid jobs ship **committed** code only (`git archive HEAD`); the skim
  jobs ship the working tree. Commit `makedf/` changes before making `.df` files.
- Each sample (nominal MC, data, cosmics, each detector variation) is its own pass
  through this chain, with its own file list, `.df` directory and skim work base.
- `truth_mcnu.py` reads `mcnu` from the original `.df` files (it isn't in the skim), so it
  must run before those are removed, and before the aggregate step into the same `P`.
- Legacy path, not drawn: the full event selection straight from the `.df` files
  (`run_event_selection_map.py -submit_grid` without `-skim`, or
  `notebooks/efficiency.ipynb`) produces the same histogram pickles without a skim, but
  re-reads every input file on each run.

### Running it: skills and the production log

Each stage has a Claude Code skill in `.claude/skills/` (invoke with `/<name>` or by asking for
the step); they hold the exact commands, checks and gotchas:

| Skill | Stages | Does |
|---|---|---|
| `grid-env` | all | environment, token, `ifdh`/xrootd checks before any grid step |
| `make-dfs` | 1–2 | file list from `filelists/datasets.json` (`filelists/make_filelist.py`), one-file test, `run_df_maker.py -ngrid`, copy to pnfs persistent (`bin/stage_to_persistent.sh <src> <dst>`) |
| `grid-jobs` | 2–3 | `jobsub_q`, find missing outputs, logs, `-resubmit_missing` (df maker and skim) |
| `validate-dfs` | 2→3 | `check_dfs.py` (tables/columns, metadata only), `validate_mcnu_truth.py` |
| `make-skim` | 3 | `-skim -submit_grid`, pull back, `-merge_skim` |
| `make-plots` | 4–5 | `-from_skim -selection both`, optional `truth_mcnu.py`, aggregate both modes |

Every production (`.df` files, copies, skims, plot runs) is recorded in
[`PRODUCTIONS.md`](PRODUCTIONS.md): command, commit, every output location, validation.
Check it for the current inputs before running anything.

`filelists/datasets.json` entries have a `type` (`mc`/`data`: picks the df-maker config) and a
`role` (`cv`, `data`, `offbeam`, `intime`, `dirt`, `detvar`: how the sample is used; every
detector-variation sample, CVs included, is `detvar` until code to process them exists).
Each entry is its own production.

## Skim workflow (Parquet)

Running the selection over all the `.df` inputs re-reads every file each time a cut or plot
changes. Instead, make a **skim** once per production and rerun from it:

```bash
# 1. once per production: skim on the grid (keeps slices passing FV + flash match +
#    containment, i.e. every stage up to config.stages.SKIM_STAGE_KEY)
python run_event_selection_map.py -skim -submit_grid -work_base <W> -samples mc,data
#    pull the outputs back as printed (both <W>/batches and <W>/batches_pi0_sideband);
#    -skim -resubmit_missing -work_base <W> resubmits failed jobs
# 2. merge into one Parquet file per sample/table under <W>/skim
python run_event_selection_map.py -merge_skim -work_base <W>
# 3. every time a later cut/plot/binning changes (local, minutes)
python run_event_selection_map.py -from_skim <W>/skim -out_dir <R> -selection both -nproc 4
python -m analysis_village.nueNp0Pi.event_selection_aggregate --in_dir <R> --out_dir <plots>
```

`<W>/skim` holds `<sample>_evt.parquet` / `<sample>_trk.parquet` (one chunk per input file),
`preskim/<mode>/<sample>__preskim.pkl` (histograms, cutflow, efficiency denominators and
POT for the stages up to the skim cut) and `skim_manifest.json`. A `-from_skim` run fills only
the stages after the skim cut and copies the preskim pickle next to its own, so the output
directory aggregates exactly like a normal batches directory. On test files the two routes
give identical histograms, cutflows, efficiencies and plots.

**Re-skim when** a cut at or before `SKIM_STAGE_KEY` changes, the df maker changes, or a
column computed in `event_selection.load_prepared_file` changes. Everything after the skim
cut only needs step 3.

Reading the skim directly (e.g. in a notebook), one chunk at a time:

```python
from pyanalib.parquet_skim import iter_skim_chunks, SkimFile
for chunk_id, evt, trk in iter_skim_chunks("<W>/skim/mc_evt.parquet", "<W>/skim/mc_trk.parquet",
                                           evt_columns=[("rec", "dlp"), ("pot_weight",)]):
    ...   # evt/trk are the usual MultiIndex frames
evt_all = SkimFile("<W>/skim/mc_evt.parquet").read(columns=[("rec", "dlp", "ele_energy_reco")])
```

`__ntuple` is rewritten to `chunk_id * 10000 + __ntuple` so (`__ntuple`, `entry`) stays unique
across input files. `mcnu` isn't in the skim (`truth_mcnu.py` reads the original files).

## Everything not listed above

`config/plots.py` holds ~70 `VariableConfig` factories (one per plotted variable) — see
`config/README.md`. Generic, reusable machinery this analysis builds on
(`ChunkRunner`/histogram accumulation, the survey→map→aggregate orchestrator, dataset-path
helpers, `.df` file I/O) lives in `pyanalib/`, not here — see `../../pyanalib/README.md`.
