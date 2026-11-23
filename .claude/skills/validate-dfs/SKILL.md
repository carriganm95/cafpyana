---
name: validate-dfs
description: Validate a nueNp0Pi .df production before using it - file count and size, required tables and columns (including the pi0 columns) via check_dfs.py, and truth matching via validate_mcnu_truth.py on a couple of MC files. Use after make-dfs, before make-skim, or when deciding whether an existing production is usable.
---

# validate-dfs: is this production usable?

Cheap checks first; nothing here loads a whole file into memory.

## 1. Structure (seconds)
```bash
cd $REPO && envs/venv_py310_cafpyana/bin/python -m analysis_village.nueNp0Pi.check_dfs <dir> --sample <mc|data>
```
Reads only HDF5 metadata of the first/middle/last file (`--all` for every file): tables
`evt/trk/hdr` (+`mcnu` for MC) in every split, and every column the selection and the pi0
sideband need. Older productions fail on `pi0_mass_reco` / `subprim_photon_energy_reco`.
- Run it on a **BlueArc copy** (`/exp/sbnd/data/...`) or a file copied to the scratchpad with
  `xrdcp` — reading through the `/pnfs` mount can hang. To copy one file:
  `cd $REPO && source setup.sh >/dev/null 2>&1 && xrdcp root://fndcadoor.fnal.gov:1094/pnfs/fnal.gov/usr/<path under /pnfs/> <scratchpad>/`
- Expect the file count to equal the number of grid jobs, and sizes to be similar across files
  (a much smaller file usually means a job died part way).

## 2. Truth matching (MC only, a few minutes)
```bash
cd $REPO && envs/venv_py310_cafpyana/bin/python -m analysis_village.nueNp0Pi.validate_mcnu_truth \
  --out <scratchpad>/validate_<name> <one or two MC .df files>
```
Compares mcnu-derived truth with the SPINE truth on the same interaction; plots land in
`--out`. Look for agreement within the per-variable tolerances it prints.

## 3. Glob summary (what the pipeline will pick up)
```bash
cd $REPO && SPRING_GEN1_ROOT=<mc dir> DATA_GEN1_ROOT=<data dir> \
  envs/venv_py310_cafpyana/bin/python -m analysis_village.nueNp0Pi.dataset_paths
```

## Record
Add the result to the production's entry in `analysis_village/nueNp0Pi/PRODUCTIONS.md`
("validation: check_dfs 3/3 OK on <copy>; validate_mcnu_truth OK").
