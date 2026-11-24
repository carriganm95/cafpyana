# Setup grid submission
outDir=$1
echo "@@ outDir : ${outDir}"
DFPREFIX=$2

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

# A -resubmit_missing submission renumbers its jobs 0..M-1; process_map.txt maps each one
# back to the original job index K, so the output keeps its original name <prefix>_K.df.
outIdx=${nProcess}
if [ -f ${filesFromSender}/process_map.txt ]; then
  outIdx=$(sed -n "$((nProcess + 1))p" ${filesFromSender}/process_map.txt)
fi
echo "@@ output index : ${outIdx}"

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

echo "@@ mkdir output"
mkdir output
echo "@@ Done!"
echo "@@ check filesFromSender dir"
ls -alh ${filesFromSender}

echo "@@ Setup xrootd"
echo $VIRTUAL_ENV
ls envs/
ls envs/venv_py310_cafpyana/
ls envs/venv_py310_cafpyana/lib/
ls envs/venv_py310_cafpyana/lib/python3.10
ls envs/venv_py310_cafpyana/lib/python3.10/site-packages/
cp -r ${filesFromSender}/XRootD $VIRTUAL_ENV/lib/python3.10/site-packages/
cp -r ${filesFromSender}/pyxrootd $VIRTUAL_ENV/lib/python3.10/site-packages/
cp -r ${filesFromSender}/xrootd.libs $VIRTUAL_ENV/lib/python3.10/site-packages/

export IFDH_CP_MAXRETRIES=2

echo "@@ outDir : "${outDir}
echo "@@ ifdh  mkdir_p "${outDir}
ifdh mkdir_p ${outDir} || true

ls -ltr
jobLog=log_${outIdx}.log

if [ -f ${filesFromSender}/job.env ]; then
  # Current layout: one filelist.txt for the whole submission. Job K (= outIdx) processes
  # lines K, K+N_JOBS, K+2*N_JOBS, ... (0-based), as run_df_maker.py assigns them.
  source ${filesFromSender}/job.env
  echo "@@ config ${DF_CONFIG}, prefix ${DF_PREFIX}, job ${outIdx} of ${N_JOBS}"
  mapfile -t myInputs < <(awk -v k=${outIdx} -v n=${N_JOBS} 'NF && (NR-1)%n==k' ${filesFromSender}/filelist.txt)
  {
    if [ ${#myInputs[@]} -eq 0 ]; then
      echo "@@ no input files for job ${outIdx}"
    else
      names=()
      for i in "${!myInputs[@]}"; do
        echo "[job ${outIdx}] input ${i} : ${myInputs[$i]}"
        xrdcp ${myInputs[$i]} .
        names+=("$(basename ${myInputs[$i]})")
      done
      ls -alh
      python run_df_maker.py -c ${DF_CONFIG} -o ${DF_PREFIX}_${outIdx}.df -ncpu 1 -i $(IFS=,; echo "${names[*]}")
    fi
  } &> ${jobLog}
else
  # Older submissions (and their resubmissions): one run_K.sh per job
  echo "@@ source ${filesFromSender}/run_"${nProcess}".sh "
  cp ${filesFromSender}/run_${nProcess}.sh ./
  source run_${nProcess}.sh  &> ${jobLog}
fi
ls -ltr

echo "@@ print log"
more ${jobLog}
echo "@@ Check output : ${DFPREFIX}_${outIdx}.df"
ls -alh ${DFPREFIX}_${outIdx}.df

outFILE=${thisOutputCreationDir}/cafpyana/${DFPREFIX}_${outIdx}.df
if [ -f "$outFILE" ]; then
  echo "ifdh cp ${outFILE} ${outDir}/${DFPREFIX}_${outIdx}.df"
  ifdh cp ${outFILE} ${outDir}/${DFPREFIX}_${outIdx}.df
  echo "ifdh cp ${thisOutputCreationDir}/cafpyana/${jobLog} ${outDir}/log_${outIdx}.log"
  ifdh cp ${thisOutputCreationDir}/cafpyana/${jobLog} ${outDir}/log_${outIdx}.log
  echo "@@ Done!"
else
  ifdh cp ${thisOutputCreationDir}/cafpyana/${jobLog} ${outDir}/log_${outIdx}.log
  echo "File not exist"
fi
