# Setup grid submission -- event-selection map phase (one BatchJob per grid job).
# Companion to bin/grid_executable.sh (the df-maker grid worker); same tarball/ifdh
# plumbing, different inner command (run_event_selection_map.py instead of run_df_maker.py)
# and output (<sample>__batch_NNNN.pkl instead of a .df file).
outDir=$1
echo "@@ outDir : ${outDir}"

nProcess=$PROCESS
echo "@@ nProcess : "${nProcess}

source /cvmfs/larsoft.opensciencegrid.org/spack-fnal-v1.0.0/setup-env.sh

echo "@@ ls -alh"
ls -alh

spack load cmake@3.31.8
spack load ifdhc@2.8.0

spack load hdf5@1.12.2%gcc@12.5.0 arch=linux-almalinux9-x86_64_v2
spack load xrootd@5.6.9%gcc@12.5.0 arch=linux-almalinux9-x86_64_v2
# this sets Python to 3.10.16
spack load root arch=linux-almalinux9-x86_64_v2

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
    echo "ifdh cp ${f} ${outDir}/$(basename ${f})"
    ifdh cp ${f} ${outDir}/$(basename ${f})
  done
  echo "@@ Done!"
else
  echo "@@ No output .pkl found under ./out -- see log_${nProcess}.log (already copied above)"
fi
