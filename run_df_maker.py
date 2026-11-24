#!/usr/bin/env python3 
import os,sys,time
import json
import re
import shlex
import shutil
import subprocess
import datetime
import pathlib
#from TimeTools import *
import argparse
import tables
from pyanalib.ntuple_glob import NTupleGlob
import pandas as pd
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=tables.exceptions.NaturalNameWarning)
pd.set_option('future.no_silent_downcasting', True)

## Arguments
parser = argparse.ArgumentParser(
    description="Data frame maker command: process input flatcaf files and generate output dataframes.",
    epilog="""\
Examples:

  -- Use Pool
  $ python run_df_maker.py -c ./configs/cohpi_slcdf.py -o test_cohpi_slcdf -i input_0.root,input_1.root,...

  -- Use Grid (adding -ngrid to an integer > 0 will automatically submit grid jobs)
  $ python run_df_maker.py -ngrid 2 -c ./configs/cohpi_slcdf.py -o test_cohpi_slcdf -i input_0.root,input_1.root,...

  -- Note!!
  Output df files are sent to $CAFPYANA_GRID_OUT_DIR/dfs (/pnfs scratch) in Grid mode; the job
  bundle and submission metadata are kept in $CAFPYANA_GRID_JOB_DIR/logs (local disk).
""",
    formatter_class=argparse.RawTextHelpFormatter  # Ensures line breaks are preserved
)
parser.add_argument('-c', dest='config', default="", help="Path to the data frame maker configuration file in ./configs, i.e.) -c ./configs/mcnu.py.")
parser.add_argument('-o', dest='output', default="", help="output data frame name prefix")
parser.add_argument('-i', dest='inputfiles', default="", help="input root file path, you can submit multiple files using comma, i.e.) -i input_0.root,input_1.root")
parser.add_argument('-l', dest='inputfilelist', default="", help="a file of list for input root files")
parser.add_argument('-ncpu', dest='NCPU', default=-1, type=int, help="Number of CPUs to run on. Default is set to number on server.")
parser.add_argument('-ngrid', dest='NGridJobs', default=0, type=int, help="Number of grid jobs. Default = 0, no grid submission.")
parser.add_argument('-nfile', dest='NFiles', default=0, type=int, help="Number of files to run. Default = 0, run all input files.")
parser.add_argument('-split', dest='SplitSize', default=1.0, type=float, help="Split size in GB before writing to HDF5. Default = 1.0 GB.")
parser.add_argument('-resubmit_missing', dest='ResubmitDir', default="",
                    help="Job dir of an earlier -ngrid submission ($CAFPYANA_GRID_JOB_DIR/logs/<ts>__<name>_log):\n"
                         "resubmit only the jobs whose <name>_K.df is missing, with the same code bundle.")

args = parser.parse_args()

def run_pool(output, inputs, nproc):
    os.nice(10)
    ntuples = NTupleGlob(inputs, None)

    # if PREPROCESS doesn't exist, set it to None
    global PREPROCESS
    try:
        PREPROCESS
    except:
        PREPROCESS = []

    dfss = ntuples.dataframes(nproc=nproc, fs=DFS, preprocess=PREPROCESS)
    output = pathlib.Path(output).with_suffix('.df')
    k_idx = 0
    split_margin = args.SplitSize
    with pd.HDFStore(output) as hdf_pd:
        NAMES.append("histpotdf")
        NAMES.append("histgenevtdf")
        size_counters = {k: 0 for k in NAMES}
        df_buffers = {k: [] for k in NAMES}

        for dfs in dfss:
            this_NAMES = NAMES
            if len(dfs) == 2: ## no or empty recTree
                this_NAMES = ["histpotdf", "histgenevtdf"]

            for k, df in zip(reversed(this_NAMES), reversed(dfs)):
                this_key = k + "_" + str(k_idx)
                size_bytes = df.memory_usage(deep=True).sum() if df is not None else 0
                size_gb = size_bytes / (1024**3)
                if len(dfs) == 2: ## no or empty recTree
                    size_counters["histpotdf"] += size_gb
                    df_buffers["histpotdf"].append(df)

                    size_counters["histgenevtdf"] += size_gb
                    df_buffers["histgenevtdf"].append(df)
                else:
                    size_counters[k] += size_gb
                    if df is not None:
                        df_buffers[k].append(df)  # accumulate

                #print(f"{k}_{k_idx}: added {size_gb:.4f} GB (total {size_counters[k]:.4f} GB)")

                del df

            if any(val > split_margin for val in size_counters.values()):
                # Concatenate and save accumulated DataFrames
                for k, buffer in df_buffers.items():
                    if buffer:  # only if buffer has data
                        concat_df = pd.concat(buffer, ignore_index=False)
                        this_key = k + "_" + str(k_idx)
                        try:
                            hdf_pd.put(key=this_key, value=concat_df, format="fixed")
                            print(f"Saved {this_key}: {concat_df.memory_usage(deep=True).sum() / (1024**3):.4f} GB")
                        except Exception as e:
                            print(f"Table {this_key} failed to save, skipping. Exception: {str(e)}")
                        del concat_df
                # Reset counters and buffers
                k_idx += 1
                size_counters = {k: 0 for k in this_NAMES}
                df_buffers = {k: [] for k in this_NAMES}

        for k, buffer in df_buffers.items():
            if buffer:
                concat_df = pd.concat(buffer, ignore_index=False)
                this_key = k + "_" + str(k_idx)
                try:
                    hdf_pd.put(key=this_key, value=concat_df, format="fixed")
                    print(f"Saved {this_key}: {concat_df.memory_usage(deep=True).sum() / (1024**3):.4f} GB")
                except Exception as e:
                    print(f"Table {this_key} failed to save, skipping. Exception: {str(e)}")
                del concat_df

        # Save the split count metadata
        split_df = pd.DataFrame({"n_split": [k_idx + 1]})  # +1 because k_idx is 0-based
        hdf_pd.put(key="split", value=split_df, format="fixed")
        print(f"Saved split info: {split_df.iloc[0]['n_split']} total splits")

def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    except Exception:
        return ""


GIT_COMMIT = _git_commit()


def submit_jobs(njobs, OutputDir, prefix):
    """jobsub_submit from the current directory (a prepared MasterJobDir holding bin_dir.tar)."""
    submitCMD = '''jobsub_submit \\
-G sbnd \\
--auth-methods="token" \\
-e LC_ALL=C \\
--role=Analysis \\
--resource-provides="usage_model=DEDICATED,OPPORTUNISTIC" \\
-l '+SingularityImage=\\"/cvmfs/singularity.opensciencegrid.org/fermilab/fnal-wn-el9\:9.7\\"' \\
--lines '+FERMIHTC_AutoRelease=True' --lines '+FERMIHTC_GraceMemory=1000' --lines '+FERMIHTC_GraceLifetime=3600' \\
--append_condor_requirements='(TARGET.HAS_SINGULARITY=?=true)' \\
--tar_file_name "dropbox://$(pwd)/bin_dir.tar" \\
-N %d \\
--disk 100GB \\
--cpu 1 \\
--memory 10GB \\
--expected-lifetime 24h \\
"file://$(pwd)/grid_executable.sh" \\
"%s" \\
"%s"'''%(njobs,OutputDir,prefix)

    print(submitCMD)
    os.system(submitCMD)


def grid_job_dir():
    """Local directory where grid submissions are prepared and their metadata kept.

    Not on /pnfs: writing file contents through the /pnfs NFS mount hangs on the gpvms.
    Grid jobs never read this directory -- jobsub uploads bin_dir.tar at submission time.
    """
    d = os.environ.get('CAFPYANA_GRID_JOB_DIR', '')
    if not d:
        sys.exit("CAFPYANA_GRID_JOB_DIR is not set (source setup.sh)")
    if d.startswith('/pnfs'):
        sys.exit(f"CAFPYANA_GRID_JOB_DIR={d} is on /pnfs; writes there hang -- use a local disk")
    return d


def write_job_inputs(job_dir, inputfiles, njobs, prefix, config):
    """The two files every job reads to find its work (see bin/grid_executable.sh).

    filelist.txt: all input CAFs; job K processes lines K, K+njobs, K+2*njobs, ... (0-based).
    job.env:      shell variables DF_CONFIG, DF_PREFIX, N_JOBS.
    """
    with open(os.path.join(job_dir, 'filelist.txt'), 'w') as f:
        f.write('\n'.join(inputfiles) + '\n')
    with open(os.path.join(job_dir, 'job.env'), 'w') as f:
        f.write(f"DF_CONFIG={shlex.quote(config)}\nDF_PREFIX={shlex.quote(prefix)}\nN_JOBS={int(njobs)}\n")


def run_grid(inputfiles):
    # 1) dir/file name style
    JobStartTime = datetime.datetime.now()
    timestamp =  JobStartTime.strftime('%Y_%m_%d_%H%M%S')

    # 2) MasterJobDir (job bundle + metadata) on local disk; outputs on /pnfs scratch
    CAFPYANA_GRID_OUT_DIR = os.environ['CAFPYANA_GRID_OUT_DIR']
    MasterJobDir = grid_job_dir() + "/logs/" + timestamp + '__' + args.output + "_log"
    OutputDir = CAFPYANA_GRID_OUT_DIR + "/dfs/" + timestamp + '__' + args.output
    os.makedirs(MasterJobDir)
    # create the output dir here too: ifdh mkdir_p on the worker can fail silently
    # (mkdir on /pnfs is a metadata operation and works; writing file contents does not)
    os.system('mkdir -p ' + OutputDir)

    # 3) grid job is based on number of files
    ngrid = args.NGridJobs
    if(len(inputfiles) <= ngrid):
        ngrid = len(inputfiles)

    NInputfiles = len(inputfiles)
    print("Number of Grid Jobs: %d, number of input caf files: %d" % (ngrid, NInputfiles))
    print("Job dir: " + MasterJobDir)

    # 4) one input list for all jobs; each job selects its own CAFs by index
    write_job_inputs(MasterJobDir, inputfiles, ngrid, args.output, args.config)

    os.system('cp ./bin/grid_executable.sh %s' %MasterJobDir)

    # 5) prepare a package for xrootd
    CAFPYANA_WD = os.environ.get('CAFPYANA_WD', os.path.dirname(os.path.abspath(__file__)))
    _site = CAFPYANA_WD + "/envs/venv_py310_cafpyana/lib/python3.10/site-packages"
    cp_XRootD = "cp -r " + _site + "/XRootD " + MasterJobDir
    cp_pyxrootd = "cp -r " + _site + "/pyxrootd " + MasterJobDir
    os.system(cp_XRootD)
    os.system(cp_pyxrootd)
    # pyxrootd links its bundled libssl/libcrypto etc. from ../xrootd.libs
    os.system("cp -r " + _site + "/xrootd.libs " + MasterJobDir)

    # 6) git archive of the current branch's last commit
    archive_repo = "git archive -o " + MasterJobDir + "/cafpyana.tar.gz HEAD"
    os.system(archive_repo)

    # 7) move to MasterJobDir to submit jobs
    os.chdir(MasterJobDir)
    tar_cmd = 'tar cf bin_dir.tar ./'
    os.system(tar_cmd)

    with open('submit_info.json', 'w') as f:
        json.dump({
            "output": args.output, "config": args.config, "output_dir": OutputDir,
            "job_dir": MasterJobDir,
            "n_jobs": ngrid, "n_inputs": NInputfiles, "submitted": timestamp,
            "git_commit": GIT_COMMIT,
        }, f, indent=2)
    submit_jobs(ngrid, OutputDir, args.output)
    
    # go back to working dir
    os.chdir(CAFPYANA_WD)
    
def resubmit_missing(log_dir):
    """Resubmit the jobs of an earlier -ngrid submission whose output .df is missing.

    Reuses the original code bundle (cafpyana.tar.gz) and per-job scripts, so the rerun
    uses exactly the code and inputs of the first submission. Jobs are renumbered 0..M-1;
    process_map.txt maps each back to its original index K so outputs keep their names.
    """
    log_dir = os.path.abspath(log_dir.rstrip('/'))
    info_path = os.path.join(log_dir, 'submit_info.json')
    if os.path.isfile(info_path):
        info = json.load(open(info_path))
        prefix, OutputDir = info['output'], info['output_dir']
    else:   # older submissions: <ts>__<name>_log, outputs in $CAFPYANA_GRID_OUT_DIR/dfs/<ts>__<name>
        m = re.match(r'^(\d{4}_\d{2}_\d{2}_\d{6})__(.+)_log$', os.path.basename(log_dir))
        if not m:
            sys.exit(f"can't parse submission name from {log_dir} (and no submit_info.json)")
        prefix = m.group(2)
        OutputDir = os.environ['CAFPYANA_GRID_OUT_DIR'] + '/dfs/' + m.group(1) + '__' + prefix
    # Job indices of the original submission. Current layout: filelist.txt + job.env (N_JOBS is
    # the original job count, also in a resubmit dir). Older layout: one run_K.sh per job.
    env_path = os.path.join(log_dir, 'job.env')
    new_layout = os.path.isfile(env_path)
    if new_layout:
        env = dict(shlex.split(l)[0].split('=', 1) for l in open(env_path) if '=' in l)
        jobs = list(range(int(env['N_JOBS'])))
    else:
        jobs = sorted((int(re.match(r'run_(\d+)\.sh$', f).group(1)) for f in os.listdir(log_dir)
                       if re.match(r'run_(\d+)\.sh$', f)))
        if not jobs:
            sys.exit(f"neither job.env nor run_K.sh scripts in {log_dir}")
    missing = [k for k in jobs
               if not (os.path.isfile(f"{OutputDir}/{prefix}_{k}.df")
                       and os.path.getsize(f"{OutputDir}/{prefix}_{k}.df") > 0)]
    print(f"{len(jobs)} job(s) in {log_dir}; outputs in {OutputDir}; {len(missing)} missing: {missing}")
    if not missing:
        return

    timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H%M%S')
    new_dir = grid_job_dir() + '/logs/' + timestamp + '__' + prefix + '_resubmit_log'
    os.makedirs(new_dir)
    for name in ('cafpyana.tar.gz', 'XRootD', 'pyxrootd', 'xrootd.libs'):
        src = os.path.join(log_dir, name)
        if not os.path.exists(src):
            sys.exit(f"{src} missing -- can't reproduce the original job environment")
        (shutil.copytree if os.path.isdir(src) else shutil.copy2)(src, os.path.join(new_dir, name))
    if new_layout:
        for name in ('filelist.txt', 'job.env'):
            shutil.copy2(os.path.join(log_dir, name), os.path.join(new_dir, name))
    else:
        for j, k in enumerate(missing):
            shutil.copy2(os.path.join(log_dir, f'run_{k}.sh'), os.path.join(new_dir, f'run_{j}.sh'))
    with open(os.path.join(new_dir, 'process_map.txt'), 'w') as f:
        f.write('\n'.join(str(k) for k in missing) + '\n')
    CAFPYANA_WD = os.environ.get('CAFPYANA_WD', os.path.dirname(os.path.abspath(__file__)))
    shutil.copy2(os.path.join(CAFPYANA_WD, 'bin', 'grid_executable.sh'), new_dir)
    with open(os.path.join(new_dir, 'submit_info.json'), 'w') as f:
        json.dump({"output": prefix, "output_dir": OutputDir, "n_jobs": len(missing),
                   "resubmit_of": log_dir, "process_map": missing, "submitted": timestamp}, f, indent=2)
    os.chdir(new_dir)
    os.system('tar cf bin_dir.tar ./')
    submit_jobs(len(missing), OutputDir, prefix)
    os.chdir(CAFPYANA_WD)
    print(f"resubmitted {len(missing)} job(s); logs in {new_dir}")


if __name__ == "__main__":
    if args.ResubmitDir:
        resubmit_missing(args.ResubmitDir)
        sys.exit(0)
    printhelp = ((args.inputfiles == "" and args.inputfilelist == "") or args.config == "" or args.output == "")
    if printhelp:
        parser.print_help()
        print(parser.epilog)
        sys.exit(1)
        
    else:
        ### Organize input list 
        InputSamples = []
        StringForHash = ""
        if args.inputfilelist != "":
            lines = open(args.inputfilelist)
            for line in lines:
                if "#" in line:
                    continue
                line = line.strip('\n')
                InputSamples.append(line)
                StringForHash += line
        else:
            split_inputfiles = args.inputfiles.split(",")
            for split_inputfile in split_inputfiles:
                InputSamples.append(split_inputfile)
                StringForHash += args.inputfiles

        if(args.NFiles > 0 and len(InputSamples) > args.NFiles):
            InputSamples = InputSamples[:args.NFiles]
                
        ### check if it is grid mode for pool mode
        if args.NGridJobs == 0:
            print("Running Pool mode");
            exec(open(args.config).read())
            run_pool(args.output, InputSamples, "auto" if args.NCPU < 0 else args.NCPU)

        elif args.NGridJobs > 0:
            print("Running Grid mode");
            run_grid(InputSamples)
            
        else:
            print("-ngrid must be greater than 0.");
