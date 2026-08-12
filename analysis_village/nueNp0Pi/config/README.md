# nueNp0Pi/config

Every editable knob for the nueNp0Pi analysis lives in one of these four files. The rule of
thumb: **settings/definitions here, logic that consumes them lives one directory up** (in
`../selections.py`, `../dataset_paths.py`, `../event_selection.py`).

| File | Holds | Logic that consumes it |
|---|---|---|
| `settings.py` | Cut thresholds, detector convention, category labels/colors | `../selections.py` |
| `plots.py` | ~70 `VariableConfig` variable definitions, plus the curated efficiency/final-sample `VariableConfig` sets | `../event_selection_aggregate.py`, `stages.py` |
| `datasets.py` | Input file globs, output paths, small constants | `../dataset_paths.py` |
| `stages.py` | The pipeline definition itself (stages/cuts/plots + efficiency vars) | `../event_selection.py` |

## `settings.py`

Reconstruction-level cut thresholds, the truth-fiducial detector convention, and
category/breakdown labels & colors. **No functions** — every name here is a constant.

- `DETECTOR = "SBND_Gen1"` — truth-fiducial / reco-fiducial convention used by cut/predicate
  functions in `../selections.py` (their default args bind to this at import time).
- `PER_TPC_INCATHODE_CM = 10` — cathode inset for the per-TPC x-fiducial cut.
- Category labels/colors: `pdg_labels`/`pdg_colors` (particle type), `nu_cosmics_labels`/`colors`,
  `topology_list`/`topology_labels`/`topology_colors` (11-entry signal/background topology
  breakdown — **the signal mode code must stay first in `topology_list`**, other code relies on
  it), `genie_mode_list`/`labels`/`colors`, `genie_sb_mode_labels`/`colors`,
  `gibuu_mode_labels`/`colors`.
- Reconstruction cut thresholds: `NU_SCORE_TH`, `SAVE_NTRKS`, `TRACKSCORE_TH`, `VTXDIST_TH`,
  `MU_CHI2MU_TH`, `MU_CHI2P_TH`, `MU_LEN_TH`, `QUAL_TH`, `P_CHI2P_TH`, `P_LEN_TH`, `MU_PLO_TH`,
  `MU_PHI_TH`, `P_PLO_TH`, `P_PHI_TH`.

Named `settings.py` rather than `selections.py` specifically to disambiguate from
`../selections.py` (the functions file) — two files both called "selections" was confusing.

## `plots.py`

`VariableConfig` (subclassing the generic record in `pyanalib.variable_config`) plus ~70
`@classmethod` factories — one per plotted physics variable (`electron_energy`,
`muon_momentum`, `proton_direction`, the `tki_del_*`/`tki_del_*_lp` TKI variables, `vertex_x/y/z`,
`opening_angle`, `neutrino_energy`, …). Each classmethod bundles the reco/truth/nu dataframe
column paths, bin edges, and plot labels for that variable — this **is** the variable's config,
so unlike the other three files this one isn't split into "settings vs. logic."

**To add a new plotted variable**: add a new `@classmethod` here following the pattern of an
existing one, then reference it from `config/stages.py` (a `PlotSpec`) or this file's own
`CORE_SELECTED_EVT_VARIABLE_CONFIGS`/`FINAL_SELECTED_EVT_VARIABLE_CONFIGS` (the efficiency-curve
variable lists — see below).

`CORE_SELECTED_EVT_VARIABLE_CONFIGS` (baseline: integrated + electron/proton kinematics + TKI
variables) and `FINAL_SELECTED_EVT_VARIABLE_CONFIGS` (extra vertex/φ components) are merged
without duplicates via `with_final_selected_evt_variables`. These used to live in a separate
`../final_selected_evt_vars.py`; merged into this file since there was no longer a good reason to
split "variable definitions" across two files.

`INTEGRATED_HIST_DUMMY = 500.0` is a production-convention dummy value for the single-bin
"integrated" measurement (matches the WireMod/SCE covariance production convention) —
analysis-specific, which is why it isn't in the shared `pyanalib.variable_config` base.

## `datasets.py`

Input file locations/globs, output paths, and small standalone constants — **settings only, no
path-computation logic** (that lives in `../dataset_paths.py`).

- `FLUX_FILE` — absolute path to the flux input ROOT file (env var `NUENP0PI_FLUX_FILE`).
- `SPRING_GEN1_ROOT` / `SPRING_GEN1_ROOT_EAF` — base directory `.df` files are globbed from
  (env var `NUMUCC_SPRING_GEN1_ROOT` — see the historical-naming note below).
- `PLOTS_BASE` — base output directory for plots (env var `NUMUCC_PLOTS_BASE`).
- `EVENT_SELECTION_GLOBS` — one glob pattern per sample (`mc`, `data`, `intime`, `offbeam`,
  `dirt`) for the main batched event-selection pipeline. **This is the one you'll edit most
  often** — uncomment/edit the entry for whatever sample you're running.
- `SELECTED_EVENTS_GLOBS` / `SELECTED_EVENTS_TAG` — globs for already-final-selected bundles
  (env var `NUMUCC_SELECTED_EVENTS_TAG`, default `"sel_mup"`).
- `MULTISIM_SYST_GLOBS_FINAL` / `_SEL_ALL` — one glob per neutrino multisim systematic
  (`MCstat`, `Flux`, `G4`, `GENIE`); keys must match `NEUTRINO_SYST_ORDER` below.
- `NEUTRINO_SYST_ORDER = ("MCstat", "Flux", "G4")` — this analysis's own copy (added when a
  cross-analysis import bug was fixed — this used to import from
  `analysis_village.numucc_1p0pi.syst_multisim_common`, a module that doesn't exist in this
  repo).
- `DETVAR_WIREMOD_GLOBS` — `(tag, glob)` pairs for detector-variation systematics.
- `GENIE_GROUP_GLOBS` / `_SEL_ALL` / `GENIE_GROUP_ORDER` / `GENIE_GROUP_KNOBS` — GENIE reweight
  sample globs, one per knob group (CCQE, MEC, RES, nonRES, DIS, Other, Ar23p).
- `DETECTOR = "SBND_nohighyz"` — a **second**, deliberately different `DETECTOR` value from
  `settings.py`'s (see "Detector convention" in `../README.md`) — used by `../utils.py`'s
  systematics code.
- `EPSILON`, `save_fig_base_dir`, `KEYS2LOAD` — small standalone constants.

**Historical naming note** (from this file's own docstring): many internal names here
(`NUMUCC_*` env vars, "numu CC 1p0pi workflows" in comments) are historical — this module was
originally copied from the `numucc_1p0pi` analysis and adapted for nueNp0Pi. Left as-is
deliberately to avoid breaking existing shell/grid-submission setups that already set these env
vars; a full rename would be a separate cleanup.

## `stages.py`

**The pipeline definition** — what cuts run, in what order, and what plots each stage produces.
This is the single file to edit to add/remove a cut or plot, or change the efficiency-curve
variable list.

- `build_pipeline() -> List[Stage]` — the 14-stage nueNp0Pi selection (all-reconstructed → FV →
  flash-matched → contained → no muons/pions → electron/proton PID cuts → vertex distance →
  dE/dx), each stage a `pyanalib.chunked_selection.Stage`. **Add a cut**: append a new
  `Stage(key=..., cut=_apply_to_evt(some_function_from_selections_py), ...)`. **Add a plot at a
  stage**: append a `PlotSpec(...)` to that stage's `plots` list.
- `EFFICIENCY_VARS` — the variable list the efficiency curve is computed against; built from this
  file's own `CORE_SELECTED_EVT_VARIABLE_CONFIGS` (imported from `plots.py`) plus
  `VariableConfig.neutrino_energy()`.
- `BREAKDOWN_REGISTRY` — which breakdown categories exist (`"topology"`, `"genie"`, `"pdg"`;
  `"genie_sb"` commented out pending a `get_genie_sb_category` function) and which
  `../selections.py` function computes each one. Consumed by `../event_selection.py`'s
  `ChunkRunner` (default `breakdown_registry`) and any `PlotSpec` whose `breakdown_type` names a
  key here. Lives here rather than in `settings.py` to avoid a circular import — see the comment
  above its definition for the full reasoning.
- `_apply_to_evt(fn)` — wraps a `df -> df` cut function (from `../selections.py`) into a
  `(state, sample) -> state` stage-cut.
- `sel_primary_trks(state, sample)` — selector used by the one `PlotSpec` in the pipeline
  (particle-KE breakdown by true pdg at the `no_photons` stage) to pick primary reconstructed
  particles from the per-track table.
- `sel_evt(state, sample)` / `evt_breakdown_plot(var_config, breakdown_type, ...)` — the
  event-level counterpart to `sel_primary_trks`: `sel_evt` selects `state["evt"]` as-is (for
  event-DISTRIBUTION breakdown plots, not per-particle ones), and `evt_breakdown_plot` builds a
  `PlotSpec` using it, appendable to any stage's `plots` list.
- `EVT_BREAKDOWN_VARS` / `EVT_BREAKDOWN_TYPE` / `EVT_BREAKDOWN_STAGE_KEYS` / `EVT_BREAKDOWN_DIR_NAME`
  — notebook-configurable knobs `build_pipeline()`/`../event_selection_aggregate.py` read to
  auto-attach and render `evt_breakdown_plot(...)` calls: which variables (default:
  `EFFICIENCY_VARS` minus 4 entries — see below), which `BREAKDOWN_REGISTRY` key (default
  `"topology"`), which stage key(s) (default: the final stage only), and which output subdirectory
  (default `"selection"`). Set these as plain attributes on the imported `config.stages` module
  from a notebook cell — see `../notebooks/README.md` — rather than editing this file. Set
  `EVT_BREAKDOWN_VARS = []` to turn these plots off. `verify_evt_breakdown_vars(evt_df)`
  sanity-checks a variable list's reco columns against a real event dataframe without running the
  full pipeline — handy before a long batched job.

  **Rendering more than one `EVT_BREAKDOWN_TYPE` into the same plots dir**: `selection_<var>.png`
  filenames don't encode `breakdown_type`, so a second run with a different `EVT_BREAKDOWN_TYPE`
  overwrites the first run's files unless you also set `EVT_BREAKDOWN_DIR_NAME` to something
  distinct, e.g.
  `stages_mod.EVT_BREAKDOWN_DIR_NAME = f"selection_{stages_mod.EVT_BREAKDOWN_TYPE}"` before each
  run. `../event_selection_aggregate.py`'s `render_overlay_plots` reads this via
  `stages_mod.EVT_BREAKDOWN_DIR_NAME` (module attribute, not a plain import) so it picks up
  whatever a notebook cell most recently set, the same way the other `EVT_BREAKDOWN_*` knobs work.
  Note this dir choice must match what was in effect when the underlying batch pickles were
  produced (map phase) if you're aggregating pre-existing pickles for more than one
  `breakdown_type` — the histdata itself is keyed by the `breakdown_type` that was active when
  `run_batch_selection` ran, not by whatever `EVT_BREAKDOWN_TYPE` is set to at render time.

  These plots (built via `sel_evt`/`evt_breakdown_plot`) render to
  `<plots_dir>/selection/selection_<var_save_name>.png` — a dedicated subdirectory and simpler
  name, set in `../event_selection_aggregate.py`'s `render_overlay_plots` by special-casing
  `ps.selector is sel_evt`. Other `PlotSpec`-driven plots (e.g. the `no_photons` particle-KE
  breakdown below, which uses `sel_primary_trks`) keep the older
  `<stage_key>__<breakdown_type>__<var_save_name>.png` naming in `plots_dir` directly, since those
  can legitimately appear at more than one stage for the same variable.

  `EVT_BREAKDOWN_VARS`'s default excludes `"integrated"` (the OverlayHistData fill path these
  plots go through doesn't special-case its single-bin sentinel the way the efficiency curve's
  does) and `"vertex_x"`/`"vertex_y"`/`"vertex_z"` (their `var_evt_reco_col` in `plots.py` is
  stale — doesn't match the real on-disk SPINE schema, confirmed against `data/test.df`; a
  pre-existing bug never previously exercised, since the efficiency curve itself doesn't read that
  field). There are 3 plausible real candidate columns for each of `vertex_x/y/z`
  (`rec.dlp.vertex.*`, `rec.dlp_true.vertex.*`, `rec.dlp_true.reco_vertex.*`) and no clear winner
  without domain knowledge, so these stay excluded pending someone confirming which is correct.
  `"electron-e"` was in this exclusion set too for a while, but that was based on a wrong
  diagnosis — see the MeV-vs-GeV note below — and it's included by default now, fixed properly.

  **MeV-vs-GeV unit bug (electron-e / proton-p), fixed**: `electron_energy()`'s and
  `proton_momentum()`'s `var_evt_reco_col`/`var_evt_truth_col` must point at the `_GeV`-suffixed
  columns (`rec.dlp.ele_energy_reco_GeV`, `rec.dlp.proton_p_reco_GeV`, etc.) that
  `../evt_derived_kinematics.py`'s `ensure_derived_trk_kinematics_cols` computes at runtime
  (`raw_MeV_column * 1e-3`) — the raw `rec.dlp.ele_energy_reco`/`rec.dlp.proton_p_reco` columns on
  disk are in MeV (SPINE's native units), but these `VariableConfig`s use GeV bins to match their
  generator-truth counterparts (`mc.e.genE`, `mc.p.totp`). Pointing a `var_evt_reco_col` at the
  raw MeV column without the `_GeV` suffix silently piles ~all events into the last bin (values
  like 560 "GeV" get clipped to a 0–3 GeV axis). A prior pass here mistakenly treated the `_GeV`
  columns as stale (because they don't exist on undressed `data/test.df` — they only appear after
  `ensure_derived_trk_kinematics_cols` runs, which a raw-file column search skips) and pointed
  `electron_energy()` at the raw MeV column instead, silently breaking its units; that's been
  reverted. `proton_momentum()` had no derived `_GeV` column at all before this round (only
  electron energy did) — `ensure_derived_trk_kinematics_cols` was generalized to derive
  `proton_p_reco_GeV`/`proton_p_true_GeV` the same way. Also fixed in passing: `proton_momentum()`'s
  truth column was named `proton_p_truth` (extra "h"); the real column is `proton_p_true`.

  **Top-right POT annotation didn't match the y-axis label, fixed**: the top-right corner text
  (`pot_annotation_text` in `pyanalib.overlay_plotting.overlay_hists_from_histdata`) defaults to a
  hardcoded placeholder string (`"8.8 x10^19 POT"`) when the caller doesn't pass a real one — this
  default was never actually derived from the plotted data. `../event_selection_aggregate.py`'s
  `render_overlay_plots` never set it, so every rendered plot showed that fixed placeholder in the
  top-right regardless of the run's actual exposure, while the y-axis label
  (`"Events / Bin (POT=...)"`) right next to it correctly used the dynamically computed `pot_str`
  — a user screenshot caught the two disagreeing (8.8e19 vs. 1.42e19). Fixed by having
  `render_overlay_plots` pass `pot_annotation_text=f"{pot_str} POT"` explicitly. Note this only
  fixes the pipeline's auto-rendered plots; direct notebook calls to
  `overlay_hists_from_histdata`/`utils.overlay_hists_from_histdata` still need `pot_annotation_text`
  passed explicitly to reflect real exposure (the function's docstring says so).

  **Output layout, `render_breakdown_independent`, and the `category_syst_summary` warning**:
  all plot output (from every `render_*` function in `../event_selection_aggregate.py`, not just
  these `EVT_BREAKDOWN_*` plots) now splits into a top-level `image/` and `pkl/` under the plots
  dir, mirroring the same subdirectory structure under each — e.g.
  `image/selection/selection_proton-p.png` pairs with `pkl/selection/selection_proton-p.pkl`.
  Every PNG gets a sibling `.pkl` holding the actual `matplotlib.figure.Figure` object
  (`pickle.dump(fig, ...)`, not a data summary) — `pickle.load` it back later and you get the
  same interactive figure without re-running aggregation. `eff_dict.pkl`/`merged_histdata.pkl`
  (aggregate-level, not per-plot) live at `pkl/eff_dict.pkl`/`pkl/merged_histdata.pkl`.
  `aggregate_and_render(..., render_breakdown_independent=True)` — set `False` to skip
  `render_summary_breakdown_plot`/`render_efficiency_plots`/`render_cutflow_table`/
  `render_breakdown_table`; their content doesn't depend on `EVT_BREAKDOWN_TYPE` (built from the
  `eff`/`bar` accumulators, filled the same way regardless), so if you're only re-running to pick
  up a different `EVT_BREAKDOWN_TYPE`'s `selection_<var>.png` plots, those four would otherwise
  get needlessly recomputed and rewritten with identical content every time. Also fixed the
  `"could not load category_syst_summary (No module named ...)"` warning that used to print once
  per plot on every batch render — `../utils.py` now only wires in that (permanently missing,
  see the module docstring's bug #2) loader as a default if the module actually exists, instead of
  always trying and catching the resulting `ModuleNotFoundError` every time.

  **GENIE-version watermark can now be turned off**: `overlay_hists_from_histdata` (and the
  nueNp0Pi `utils.py` wrapper, which passes `**kwargs` straight through) takes a
  `show_genie_label: bool = True` kwarg. Pass `show_genie_label=False` directly, or set it in a
  `PlotSpec`'s `save_kwargs` (e.g. inside `evt_breakdown_plot()`) to suppress it for
  pipeline-rendered plots.

  **Legend overflow / GENIE-text overlap, fixed**: `pyanalib/overlay_plotting.py`'s breakdown
  legend used a fixed `ncol=3` at `fontsize=12`, which runs past the right edge of the axes once a
  breakdown has more than a handful of long category names (topology has up to 12, including
  Dirt) — confirmed visually. `ncol`/`fontsize` now scale down with entry count so the legend
  stays inside the axes box. The GENIE-version watermark (`add_genie_version_text`) also moved
  from a fixed upper-left slot (which ended up rendered underneath a tall legend) to bottom-right,
  which stays clear of the legend regardless of its row count.

  **Important pre-existing bug fixed in an earlier round**: `pyanalib/overlay_plotting.py`'s
  `overlay_hists_from_histdata` reversed the breakdown legend's labels/colors (to match its
  signal-drawn-last stacking convention) but never reversed the matching per-category data
  (`weights_categ`/`var_categ`, built from `histdata.mc_hist` in `get_cuts_fn`'s signal-first
  "cuts order") — so every legend label except Dirt was paired with the wrong category's actual
  counts. Confirmed against real data: a topology-breakdown plot at the final selection stage was
  showing mostly "NC"/"Other" despite the sample being >90% signal. Now fixed (data reversed to
  match), with regression tests in `../../../tests/test_overlay_plotting_breakdown_order.py`. A
  smaller, related bug — OOPS/OOFV swapped at indices 2/3 in `../selections.py`'s
  `get_topo_category_nueNp0Pi` relative to `topology_labels`' order — was fixed alongside it. Both
  bugs predate this refactor and affected every historical topology/genie/pdg breakdown plot, not
  just the new `EVT_BREAKDOWN_*` ones.

`../event_selection.py` (the runner) just calls `build_pipeline()` and applies whatever `Stage`
list comes back — it doesn't know what stages exist, only how to run them.

## `__init__.py`

Just a docstring pointing back to this README's structure — no code.
