#!/bin/bash
# Copy a production's .df files to durable storage (pnfs scratch gets purged; grid jobs
# can read pnfs persistent but not /exp/... BlueArc).
#
# Usage (after `source setup.sh` and a valid token):
#   bash bin/stage_to_persistent.sh <src_dir> <dst_dir>
#   e.g. bash bin/stage_to_persistent.sh \
#          /pnfs/sbnd/scratch/users/$USER/cafpyana_out/dfs/2026_10_28_084259__nueNp0Pi_mc_v4 \
#          /pnfs/sbnd/persistent/users/$USER/nueNp0Pi/nueNp0Pi_mc_v4
#
# Copies only: never moves or deletes anything. Files already at the destination with the
# same size are skipped (safe to rerun after an interruption); a destination file with a
# different size is reported, not overwritten. Ends with a count check; exits non-zero if
# anything is missing or mismatched.
set -u
SRC=${1:?usage: stage_to_persistent.sh <src_dir> <dst_dir>}
DST=${2:?usage: stage_to_persistent.sh <src_dir> <dst_dir>}

if ! command -v ifdh >/dev/null 2>&1; then
  # setup.sh doesn't put ifdh on PATH; use the EL9 build the grid scripts use
  _FC=/cvmfs/fermilab.opensciencegrid.org/packages/common
  _IFDHC=${_FC}/ifdhc/2.8.0/linux-almalinux9-x86_64_v2-gcc-11.4.1-6u3yv5pf5euibfyxqjrnxptk65m3zfu7
  export IFDHC_CONFIG_DIR=${_FC}/ifdhc-config/2.8.0/linux-almalinux9-x86_64_v2-gcc-11.4.1-pm3ua2jppmteolvjpfuuypy4dr6funfx
  export PATH=${_IFDHC}/bin:${IFDHC_CONFIG_DIR}/bin:${PATH}
  export LD_LIBRARY_PATH=${_IFDHC}/lib:${LD_LIBRARY_PATH:-}
fi
command -v ifdh >/dev/null 2>&1 || { echo "[stage] ifdh not found"; exit 1; }

shopt -s nullglob
files=("${SRC}"/*.df)
if [ ${#files[@]} -eq 0 ]; then
  echo "[stage] no .df files in ${SRC}"; exit 1
fi
echo "[stage] ${#files[@]} .df file(s): ${SRC} -> ${DST}"
ifdh mkdir_p "${DST}" 2>/dev/null || mkdir -p "${DST}"

n_copied=0; n_skipped=0; n_failed=0
for f in "${files[@]}"; do
  dst="${DST}/$(basename "${f}")"
  if [ -f "${dst}" ] && [ "$(stat -c %s "${dst}")" = "$(stat -c %s "${f}")" ]; then
    n_skipped=$((n_skipped + 1)); continue
  fi
  if [ -f "${dst}" ]; then
    # size differs: probably a partial copy. Never delete or overwrite it here -- report it
    # and let the user decide.
    echo "[stage] SIZE MISMATCH, not overwritten: ${dst}"; n_failed=$((n_failed + 1)); continue
  fi
  if ifdh cp "${f}" "${dst}"; then
    n_copied=$((n_copied + 1))
  else
    echo "[stage] FAILED: $(basename "${f}")"; n_failed=$((n_failed + 1))
  fi
done

n_dst=$(ls "${DST}"/*.df 2>/dev/null | wc -l)
echo "[stage] copied ${n_copied}, already there ${n_skipped}, failed ${n_failed}; ${n_dst}/${#files[@]} at destination"
[ ${n_failed} -eq 0 ] && [ "${n_dst}" -ge ${#files[@]} ]
