---
name: grid-env
description: Check and set up the cafpyana environment and Fermilab grid credentials (setup.sh, venv, bearer token, CAFPYANA_GRID_OUT_DIR, ifdh) before any nueNp0Pi grid submission, file copy, or xrootd access. Use before make-dfs, make-skim, grid-jobs, or whenever a grid/pnfs/xrootd command fails with auth errors.
---

# grid-env: environment and credentials

Run this before any step that submits grid jobs, copies files with `ifdh`, or reads files
over xrootd. It is quick (~40 s) and fixes the usual failures up front.

## Facts about this setup
- Repo root: the directory holding `setup.sh` (call it `$REPO`). Every Bash call starts a
  fresh shell, so **each command that needs the environment must start with**
  `cd $REPO && source setup.sh >/dev/null 2>&1 && ...`
- `setup.sh` loads spack (ROOT, HDF5, xrootd), activates `envs/venv_py310_cafpyana`,
  exports `CAFPYANA_GRID_OUT_DIR=/pnfs/sbnd/scratch/users/$USER/cafpyana_out`, and **renews the
  bearer token** (`htgettoken`, stored at `/run/user/$(id -u)/bt_u$(id -u)`).
- Tokens last ~3 h. `httokendecode -H /run/user/$(id -u)/bt_u$(id -u) | grep exp` shows expiry.
- `ifdh` is NOT on PATH after `setup.sh`. Scripts that need it (`bin/stage_to_persistent.sh`,
  the grid workers) add the EL9 build from
  `/cvmfs/fermilab.opensciencegrid.org/packages/common/ifdhc/2.8.0/...` themselves.
- `CAFPYANA_WD` is optional: `run_df_maker.py` / `run_event_selection_map.py` default it to
  the repo root.
- Listing `/pnfs` works on the interactive node, but **reading file contents through the
  /pnfs mount can hang for minutes** — read pnfs files with xrootd (`root://` URLs) or check
  a BlueArc (`/exp/sbnd/data/...`) copy instead.
- The node has ~11 GB RAM: never load a whole `.df` file interactively.

## Check
Run once and report each line to the user:
```bash
cd $REPO && source setup.sh >/dev/null 2>&1; \
echo "python: $(which python)"; python -c "import pyarrow, pandas, tables; print('python deps ok')"; \
echo "grid out: ${CAFPYANA_GRID_OUT_DIR:-UNSET}"; \
httokendecode -H /run/user/$(id -u)/bt_u$(id -u) | grep '"exp"'; \
which jobsub_submit jobsub_q; \
BEARER_TOKEN_FILE=/run/user/$(id -u)/bt_u$(id -u) python -c "from XRootD import client; \
st,_=client.FileSystem('root://fndcadoor.fnal.gov:1094').stat('/pnfs/fnal.gov/usr/sbnd/persistent/users/$USER'); print('xrootd:', st.ok, st.message.strip())"
```
Everything should be present, the token expiry in the future, and `xrootd: True`.

## Fixes
- Token missing/expired and `setup.sh` didn't renew it (e.g. no Kerberos ticket): ask the
  user to run `! kinit` and then `! htgettoken -a htvaultprod.fnal.gov -i sbnd` themselves
  (interactive login — don't try to drive it).
- `python deps` fails on pyarrow: `envs/venv_py310_cafpyana/bin/pip install -r envs/pip_requirements.txt`.
- `CAFPYANA_GRID_OUT_DIR` unset: `setup.sh` only sets it on hosts whose name contains `sbnd`
  or `icarus`; use an sbnd gpvm.
