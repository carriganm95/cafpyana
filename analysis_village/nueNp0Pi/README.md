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

1. **Where your input `.df` files are** — `config/datasets.py`: `SPRING_GEN1_ROOT` (defaults to
   an env var `NUMUCC_SPRING_GEN1_ROOT`, historical name — see note below), and the
   `EVENT_SELECTION_GLOBS` / `SELECTED_EVENTS_GLOBS` / `MULTISIM_SYST_GLOBS_*` /
   `GENIE_GROUP_GLOBS` / `DETVAR_WIREMOD_GLOBS` dicts that build actual glob patterns from it.
   Easiest override without editing the file: `export NUMUCC_SPRING_GEN1_ROOT=/your/path`.
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

## Everything not listed above

`config/plots.py` holds ~70 `VariableConfig` factories (one per plotted variable) — see
`config/README.md`. Generic, reusable machinery this analysis builds on
(`ChunkRunner`/histogram accumulation, the survey→map→aggregate orchestrator, dataset-path
helpers, `.df` file I/O) lives in `pyanalib/`, not here — see `../../pyanalib/README.md`.
