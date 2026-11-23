# Setup grid submission -- event-selection map phase (one BatchJob per grid job).
# Companion to bin/grid_executable.sh (the df-maker grid worker); same tarball/ifdh
# plumbing, different inner command (run_event_selection_map.py instead of run_df_maker.py)
# and output (<sample>__batch_NNNN.pkl instead of a .df file).
outDir=$1
echo "@@ outDir : ${outDir}"

nProcess=$PROCESS
echo "@@ nProcess : "${nProcess}

# spack-fnal-v1.0.0's spack tool broke on 2026-10-27 (empty lib/spack); v1.1.1 resolves
# the same hdf5/xrootd/root builds (same hashes as setup.sh). Python is 3.10.16.
source /cvmfs/larsoft.opensciencegrid.org/spack-fnal-v1.1.1/setup-env.sh

echo "@@ ls -alh"
ls -alh

spack load cmake/rencrd2 || true
spack load hdf5/bnmkgjf
spack load xrootd/wrvy2k7
spack load root/vgs6mjs

# ifdh: v1.1.1 only has x86_64_v3 ifdhc builds, so use the EL9 x86_64_v2 build from
# the Fermilab common area directly.
_FC=/cvmfs/fermilab.opensciencegrid.org/packages/common
_IFDHC=${_FC}/ifdhc/2.8.0/linux-almalinux9-x86_64_v2-gcc-11.4.1-6u3yv5pf5euibfyxqjrnxptk65m3zfu7
export IFDHC_CONFIG_DIR=${_FC}/ifdhc-config/2.8.0/linux-almalinux9-x86_64_v2-gcc-11.4.1-pm3ua2jppmteolvjpfuuypy4dr6funfx
export PATH=${_IFDHC}/bin:${IFDHC_CONFIG_DIR}/bin:${PATH}
export LD_LIBRARY_PATH=${_IFDHC}/lib:${LD_LIBRARY_PATH}
echo "@@ python: $(which python) $(python --version 2>&1)  ifdh: $(which ifdh)"

thisOutputCreationDir=`pwd`
filesFromSender=${CONDOR_DIR_INPUT}/bin_dir/

echo "@@ copy cafpyana.tar.gz and extract"
cp ${filesFromSender}/cafpyana.tar.gz ./
mkdir cafpyana
mv cafpyana.tar.gz cafpyana
cd cafpyana
tar -xvf cafpyana.tar.gz
echo "@@ ls -alh"
ls -alh

echo "@@ run init_grid.sh"
source ./bin/init_grid.sh
echo "@@ ls -alh"
ls -alh

echo "@@ check filesFromSender dir"
ls -alh ${filesFromSender}

echo "@@ Setup xrootd"
echo $VIRTUAL_ENV
cp -r ${filesFromSender}/XRootD $VIRTUAL_ENV/lib/python3.10/site-packages/
cp -r ${filesFromSender}/pyxrootd $VIRTUAL_ENV/lib/python3.10/site-packages/
cp -r ${filesFromSender}/xrootd.libs $VIRTUAL_ENV/lib/python3.10/site-packages/

export IFDH_CP_MAXRETRIES=2

echo "@@ outDir : "${outDir}
echo "@@ ifdh  mkdir_p "${outDir}
ifdh mkdir_p ${outDir} || true

echo "@@ source ${filesFromSender}/run_"${nProcess}".sh "
ls -ltr

cp ${filesFromSender}/run_${nProcess}.sh ./
source run_${nProcess}.sh  &> log_${nProcess}.log
ls -ltr

echo "@@ print log"
more log_${nProcess}.log

echo "@@ Check output dir"
ls -alh ./out

ifdh cp log_${nProcess}.log ${outDir}/../map_logs/log_${nProcess}.log 2>/dev/null || \
  ifdh mkdir_p ${outDir}/../map_logs && ifdh cp log_${nProcess}.log ${outDir}/../map_logs/log_${nProcess}.log

shopt -s nullglob
outFILES=(${thisOutputCreationDir}/cafpyana/out/*.pkl)
if [ ${#outFILES[@]} -gt 0 ]; then
  for f in "${outFILES[@]}"; do
    dest=${outDir}/$(basename ${f})
    ok=0
    for attempt in 1 2 3; do
      echo "ifdh cp ${f} ${dest} (attempt ${attempt})"
      if ifdh cp ${f} ${dest}; then ok=1; break; fi
      # a timed-out transfer can leave a partial file that blocks the retry ("File exists")
      ifdh rm ${dest} 2>/dev/null || true
      sleep $((60 * attempt))
    done
    if [ ${ok} -ne 1 ]; then
      echo "@@ FAILED to copy ${f} to ${dest}"
      exit 1
    fi
  done
  echo "@@ Done!"
else
  echo "@@ No output .pkl found under ./out -- see log_${nProcess}.log (already copied above)"
fi

# -selection both: pi0 sideband pickles go to a sibling dir (<outDir>_pi0_sideband)
sbFILES=(${thisOutputCreationDir}/cafpyana/out_pi0_sideband/*.pkl)
if [ ${#sbFILES[@]} -gt 0 ]; then
  sbDir=${outDir}_pi0_sideband
  ifdh mkdir_p ${sbDir} || true
  for f in "${sbFILES[@]}"; do
    dest=${sbDir}/$(basename ${f})
    ok=0
    for attempt in 1 2 3; do
      echo "ifdh cp ${f} ${dest} (attempt ${attempt})"
      if ifdh cp ${f} ${dest}; then ok=1; break; fi
      ifdh rm ${dest} 2>/dev/null || true
      sleep $((60 * attempt))
    done
    if [ ${ok} -ne 1 ]; then
      echo "@@ FAILED to copy ${f} to ${dest}"
      exit 1
    fi
  done
  echo "@@ Done (pi0 sideband)!"
fi
