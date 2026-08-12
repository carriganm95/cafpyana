# nueNp0Pi/notebooks

## `efficiency.ipynb` — the maintained notebook

Runs the full batched event-selection pipeline end to end via
`analysis_village.nueNp0Pi.event_selection.build_event_selection_pipeline` and
`pyanalib.event_selection_pipeline.EventSelectionPipelineConfig`, then displays the resulting
efficiency curves / breakdown plots / overlay stacks. This is the one to open if you want to
run the current pipeline or see how `build_event_selection_pipeline` is meant to be called.
If you change `config/datasets.py`'s globs or `config/stages.py`'s pipeline definition, re-run
this notebook to sanity-check the change against real data before trusting it.

Right before `pipeline.run_full()` there's a cell for configuring the stacked
event-distribution breakdown plots (one per `EFFICIENCY_VARS` variable, by
topology, at the final stage by default) via `config/stages.py`'s
`EVT_BREAKDOWN_TYPE` / `EVT_BREAKDOWN_STAGE_KEYS` / `EVT_BREAKDOWN_VARS`
module attributes — set them there to change breakdown type, which stage(s)
get a plot, or which variables, without editing `config/stages.py` itself.

## `presentation.mplstyle`

Matplotlib style sheet applied by `../utils.py` at import time
(`plt.style.use(.../notebooks/presentation.mplstyle)`) — controls fonts/sizes/colors for every
plot this analysis renders. Edit this to change the look of all plots analysis-wide.

## `example.ipynb`, `test.ipynb` — personal scratch notebooks, not part of the maintained pipeline

Both hardcode absolute paths belonging to individual users (`example.ipynb`:
`/exp/sbnd/data/users/lynnt/...`; `test.ipynb`: `/nashome/m/micarrig/...`) and predate the
current batched-pipeline structure — they load `.df` files directly and explore/select by hand
rather than going through `event_selection.py`. Kept intentionally (confirmed with Mike: these
are personal working notebooks for running/exploring the event selection, not dead code to
clean up) — don't expect them to run as-is on a different machine, and don't treat them as
documentation of the current pipeline API (use `efficiency.ipynb` for that).
