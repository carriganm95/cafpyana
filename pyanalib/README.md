# pyanalib

Generic, analysis-agnostic machinery shared across `analysis_village/*` analyses. Nothing in
here should import from a specific `analysis_village.<analysis>` package — if a file does,
that's a bug (see the `dataset_paths.py` history below for one that got fixed). Analyses wire
these generic pieces to their own settings/functions via small "hooks"/config objects passed in
by the caller — see `analysis_village/nueNp0Pi/README.md` for a worked example of an analysis
built on top of this directory.

**One deliberate exception**: `overlay_plotting.py` imports plot-style helpers directly from
`analysis_village.plot_style.sbnd_style` (`get_textloc_x`, `add_approval_text`, `add_pot_text`,
`add_chi2_text`, `add_genie_version_text`, `format_singlebin_plot`). That module is treated as
shared repo-wide infrastructure (used by more than one analysis), the same tier as `makedf` —
not owned by any single analysis, so importing it here doesn't reintroduce an analysis-specific
dependency.

## Event-selection / histogramming pipeline

**`chunked_selection.py`** — The core memory-efficient, chunked event-selection and
histogram-accumulation framework. Turns an all-in-memory event-selection notebook into a
pipeline that loads one input file at a time, runs the full selection, bins variables into
per-category histograms at each "plot point" instead of plotting immediately, and saves the
histograms (one pickle per chunk) — so a full sample never needs to fit in RAM at once. Key
pieces:
  - `Stage` / `PlotSpec` — a pipeline stage (a cut + the plots to make at that point).
  - `ChunkRunner` — runs a list of `Stage`s over one loaded file, accumulating histograms.
    Analyses subclass it (or pass `breakdown_registry`/`signal_mask_fn` directly) to supply
    their own signal definition and category breakdown.
  - `OverlayHistData`, `BarBreakdown`, `EfficiencyAccumulator` — the histogram accumulators
    themselves (overlay stack, category breakdown bar chart, efficiency curve).
  - `aggregate_chunk_files` / `merge_samples` / `apply_global_exposure_scales` — the reduce-phase
    functions: sum per-chunk histograms, merge across samples, apply global POT/gates scaling.
  - `multicol_get_series` / `multicol_resolve_column_key` — MultiIndex-column-tolerant column
    lookup (handles depth mismatches / dropped intermediate `p` levels between on-disk schemas).

**`event_selection_pipeline.py`** — The analysis-agnostic survey → bin-pack → map → aggregate
orchestration layer built on top of `chunked_selection.py`. Lists input `.df` files, packs them
into ≤N-byte batch jobs, runs the map phase (optionally across a `multiprocessing.Pool`), and
drives the aggregate/render phase — none of it knows anything about a specific analysis, only
about the three callables (`iter_df_paths`, `run_batch_selection`, `aggregate_and_render`) an
analysis supplies via `EventSelectionHooks`. `EventSelectionPipeline` is the class analyses
instantiate; `EventSelectionPipelineConfig` holds the run-time knobs (work dirs, batch size,
sample list, `n_workers`, …). See `analysis_village/nueNp0Pi/event_selection.py`'s
`build_event_selection_pipeline` for how an analysis wires this up.

**`dataset_paths.py`** — Generic dataset-path resolution: glob listing (`sorted_glob`, which
shells out to `ls` to handle PNFS/dCache mounts that `glob.glob`/`os.listdir` silently fail on),
keyed/grouped/tagged glob iteration (`iter_keyed_df_paths`, `iter_tagged_chunk_jobs`,
`iter_ordered_group_tasks`, `iter_unique_df_paths`), and default work-root path computation
(`default_work_root`, `default_syst_disk_root`, `sibling_dir`). No hardcoded analysis-specific
globs, env-var names, or paths anywhere in this file — every analysis supplies its own glob
dicts / env-var names / work-base path segment as arguments. `analysis_village/nueNp0Pi/dataset_paths.py`
is a thin wiring example.

## Systematics

**`syst_disk_layout.py`** — Defines the on-disk directory layout precomputed systematic
covariances get written to and read from (`MCstat/`, `Flux/`, `G4/`, `GENIE/`, `Cosmics/`,
`Detector/`, `CategorySummary/`, each with its own expected filename). Controlled by the
`SYST_DISK_ROOT` env var (`SYST_DISK_ENV`). Any analysis can point this at its own tree; loaders
fail loudly if an expected file is missing (no silent fallback).

**`covariance.py`** — Small covariance/correlation-matrix utilities: `cov_from_fraccov` /
`fraccov_from_cov` (convert between absolute and fractional covariance given per-bin central
values), `corr_from_fraccov` (correlation matrix from a fractional covariance), and
`get_covariance_matrix(univ_events, cv_events)` (build cov/frac-cov/corr from a
multi-universe × bin array of event counts against a central-value array — used for multisim
systematics).

**`stat_helpers.py`** — `return_data_stat_err(data_array)`: Garwood/Poisson asymmetric error
bars (68.27% CL, via `scipy.stats.gamma`) for data histogram counts — the standard "data point
error bar" convention for count data, including empty bins.

**`syst_loading.py`** — Generic systematics-covariance loading: reads the `syst_disk_layout.py`
tree (`get_syst_unc`), a `category_syst_summary.npz` export (`load_overlay_syst_cov_frac`,
`get_category_summary_syst_unc`), and a GENIE signal/background covariance pickle
(`load_genie_sb_bkgd_rate_cov_frac`). No hardcoded analysis-specific paths or defaults — callers
supply their own fallback root (`default_root`/`default_root_base`) and, for the two loaders with
an analysis-specific data source (`get_syst_unc`'s `cosmics` component,
`load_overlay_syst_cov_frac`'s category-summary export), an explicit hook callable
(`cosmics_flat_uncorrelated_cov_frac`, `syst_category_summary_loader`). See
`analysis_village/nueNp0Pi/utils.py` for a worked wiring example, including how it preserves two
pre-existing bugs (imports of modules that don't exist anywhere in this repo) by leaving them
inside its own hook functions rather than "fixing" them as part of the move.

**`overlay_plotting.py`** — Generic overlay-histogram rendering: turns a precomputed
`chunked_selection.OverlayHistData` into the stacked MC/data comparison plot
(`overlay_hists_from_histdata`) — legend rows, syst. bands, chi2 text, ratio panel, POT/approval
annotations. Callers supply a `BreakdownDisplay` dict (`{breakdown_type: (labels, colors,
hatches_or_None)}`) for whatever breakdown categories they use, plus optional hooks for the
category-summary syst band and GENIE-SB background syst band (same pattern as `syst_loading.py`).
Also holds `get_clipped_evts` (bin-range clipping + POT weights, with an analysis-configurable
single-bin "integrated" sentinel) and small formatting helpers (`get_pot_str`, `generate_tags`).
Legend rows for MC categories include both the POT-scaled predicted event count and percentage of
the plotted total, e.g. `"label (142, 34.2%)"`. Two chi2/covariance-decomposition code paths
(`textchi2=True`, `syst_decomp=True`) call `get_chi2`/`get_chi2_shape`/`Matrix_Decomp`, which
aren't defined or imported anywhere in this repo — a pre-existing bug (see the module/function
docstrings), dormant unless those paths are actually exercised.

## `.df` file I/O

**`split_df_helpers.py`** — Original helpers for reading chunked `.df` HDF5 files (each dataset
key stored as N smaller per-split tables, so large samples can be handled without one giant
in-memory concat): `get_n_split`, `print_keys`, `load_dfs`. Still used by several other
analyses (`gump`, `unfolding`) and `run_ttree_maker.py`.

**`split_df_helpers_new.py`** — A newer, more careful version of the same idea, currently only
consumed by `analysis_village/nueNp0Pi/event_selection.py`. Adds: cross-key ntuple-index
deduplication/remapping when concatenating splits (`_remap_ntuple_index`,
`_unique_ntuple_values_across_keys`), a `tqdm` progress bar, structured logging via
`logging_utils`, and `dfs_from_dir` for loading a whole directory of split files at once. Not a
drop-in replacement for the older module — some function signatures differ (e.g.
`get_n_split(file, reference_key="evt")`).

## Column / dataframe utilities

**`pandas_helpers.py`** — Generic helpers for the MultiIndex-column convention `.df` files use:
`broadcast` (repeat an interaction-level Series to particle-level rows matching a target index),
`multicol_concat` / `multicol_add` / `multicol_merge` (concat/join/merge that pad differing
column-tuple depths to a common `nlevels` first), `loadbranches` (load `uproot` tree branches,
including jagged/vector branches, into a flat/MultiIndex dataframe), `pad_column_name`,
`add_upper_level_to_df`, `rename_to_XYZ`.

**`ntuple_glob.py`** — `NTupleGlob` / `NTupleProc`: multiprocessing-pool-based driver that opens
a list of ROOT ntuple files (with retry-on-open-failure), applies a list of dataframe-maker
functions (the `DFS` from a `df_maker_configs`-style config) to each, and yields the resulting
dataframes. This is what `run_df_maker.py` actually calls to drive production.

**`sbnanaobj_enums.py`** — Loads C++ enums from the `sbnanaobj` header
(`SREnums.h`, pinned to a specific CVMFS-hosted version) into Python dicts via the ROOT
interpreter, at import time. Requires ROOT and CVMFS access; fails soft (prints an error,
doesn't raise) if either is unavailable.

**`logging_utils.py`** — `get_logger(name)`: one shared logging setup for the whole codebase,
controlled by a single env var, instead of every module hand-rolling its own `print()`/verbose
flags.

## Physics variable calculators (analysis-specific despite living in pyanalib)

**`variable_config.py`** — The generic `VariableConfig` record shape (reco/truth/nu column
paths, bin edges, plot labels) that analyses subclass to define their own plotted-variable
factories — see `analysis_village/nueNp0Pi/config/plots.py` for ~70 examples. Also the shared
`INTEGRATED_VAR_SAVE_NAME` sentinel / `is_integrated_var_config` convention for single-bin
"all events" measurements.

**`variable_calculator.py`** — TKI (transverse kinematic imbalance) variable calculators:
`get_cc1p0pi_tki`, `get_tki_spine` / `get_tki_spine_lp` (SPINE-schema variants, with/without a
leading-proton definition), plus `add_reco_cc1p0pi_tki_evtdf` / `add_mc_cc1p0pi_tki_mcnu` /
`add_truth_cc1p0pi_tki_evtdf` to attach the computed TKI columns onto event/truth dataframes.
Used by `analysis_village/nueNp0Pi/makedf/make_nueccdf.py` and (per its name) originally written
for CC1p0π-style analyses generally.

**`cc2p_reco_var.py`** — Reconstruction-variable helpers (fiducial volume, signal/topology
definitions like `cc2pNpi`/`cc1p0pi`/`cc0p0pi`, PID/containment helpers, reco-imbalance
calculation) written for and currently only consumed by `analysis_village/cc2p/makedf/make_cc2pdf.py`
— despite living in `pyanalib`, this one is CC2p-analysis-specific, not a generic shared utility.
Also defines its own `PDG` dict (pdg code / name / mass), separate from `makedf.makedf.PDG` —
check which one a given function is actually using if you're editing particle masses/codes.

## `__init__.py`

Empty — `pyanalib` is a plain namespace package, no package-level exports.
