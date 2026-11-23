#!/bin/bash
# Stage event-selection input .df files from BlueArc to pnfs/persistent
# so grid workers can ifdh cp them. Run after sourcing setup.sh.
#
# Usage: source setup.sh && bash bin/stage_to_persistent.sh
#
# MC files: ~458 GB -- runs in background; check progress with:
#   ls /pnfs/sbnd/persistent/users/micarrig/nueNp0Pi/selection_v3/*.df | wc -l
# Data files: ~2.6 GB -- runs in foreground (fast).

set -e

MC_SRC="/exp/sbnd/data/users/micarrig/nueNp0Pi/selection_v3"
MC_DST="/pnfs/sbnd/persistent/users/micarrig/nueNp0Pi/selection_v3"
DATA_SRC="/exp/sbnd/data/users/micarrig/nueNp0Pi/selection_data_v1"
DATA_DST="/pnfs/sbnd/persistent/users/micarrig/nueNp0Pi/selection_data_v1"

echo "[stage] creating pnfs persistent directories..."
ifdh mkdir_p ${MC_DST}
ifdh mkdir_p ${DATA_DST}

# ---- Data files (2.6 GB, fast) ----
echo "[stage] staging data files..."
for f in ${DATA_SRC}/*.df; do
    dst="${DATA_DST}/$(basename ${f})"
    ifdh ls ${dst} > /dev/null 2>&1 && echo "  skip (exists): $(basename ${f})" || \
        { echo "  cp: $(basename ${f})"; ifdh cp ${f} ${dst}; }
done
echo "[stage] data files done."

# ---- MC files (458 GB, background) ----
echo "[stage] staging MC files in background (log: /tmp/stage_mc.log)..."
(
    for f in ${MC_SRC}/*.df; do
        dst="${MC_DST}/$(basename ${f})"
        ifdh ls ${dst} > /dev/null 2>&1 && echo "  skip (exists): $(basename ${f})" || \
            { echo "  cp: $(basename ${f})"; ifdh cp ${f} ${dst}; }
    done
    echo "[stage] MC files done."
) >> /tmp/stage_mc.log 2>&1 &

MC_PID=$!
echo "[stage] MC staging PID=${MC_PID} -- check progress:"
echo "  tail -f /tmp/stage_mc.log"
echo "  ls ${MC_DST}/*.df 2>/dev/null | wc -l   (expect 160)"
echo ""
echo "[stage] Once staging is complete, resubmit with:"
echo "  export SPRING_GEN1_ROOT=${MC_DST}"
echo "  export DATA_GEN1_ROOT=${DATA_DST}"
echo "  python run_event_selection_map.py -submit_grid \\"
echo "      -work_base /exp/sbnd/data/users/micarrig/nueNp0Pi/\$(date +%Y%m%d) \\"
echo "      -samples mc,data -max_job_bytes_gb 10"
