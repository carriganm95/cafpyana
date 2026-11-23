#!/usr/bin/env python3
"""Run the nueNp0Pi event-selection MAP phase for one batch job, or submit one
grid job per batch job.

This is the standalone-CLI counterpart to ``pyanalib.event_selection_pipeline.
EventSelectionPipeline.process_one_job`` -- same file/pickle contract as the
notebook's local (in-process ``multiprocessing.Pool``) map phase, so output
pickles from this script and from a notebook run drop into the same
``batches_dir`` and are aggregated identically by
``analysis_village.nueNp0Pi.event_selection_aggregate`` /
``EventSelectionPipeline.run_aggregate()`` (which cross-checks every expected
job against ``manifest.json`` via ``check_batches()`` before merging -- see
that module for the missing/corrupt-batch report).

Examples:

  -- Run one batch job locally (what each grid job below also runs)
  $ python run_event_selection_map.py \\
        -sample mc -job_id 0 -out_dir /path/to/batches \\
        -files /path/output_v1_0.df,/path/output_v1_1.df

  -- Submit one grid job per batch job (survey + bin-pack happens here,
     locally, via EventSelectionPipeline.discover_jobs() -- same manifest.json
     a local run would write)
  $ python run_event_selection_map.py -submit_grid \\
        -work_base /exp/sbnd/data/users/$USER/nueNp0Pi/20261020 \\
        -samples mc,data -max_files_per_sample mc=5 -max_job_bytes_gb 10

  -- Resubmit only the missing/corrupt batch jobs from a previous -submit_grid run
     (reads existing manifest.json -- does NOT re-survey or overwrite it):
  $ python run_event_selection_map.py -resubmit_missing \\
        -work_base /exp/sbnd/data/users/$USER/nueNp0Pi/20261020

  -- Note!!
  Grid-mode output pickles are written directly into <work_base>/batches (ifdh cp'd
  back from the worker), the same directory a local/notebook run would use -- so
  EventSelectionPipeline.run_aggregate()/check_batches() against that work_base see
  grid- and locally-produced batches identically.
"""
import argparse
import datetime
import os
import sys

parser = argparse.ArgumentParser(
    description="nueNp0Pi event-selection map-phase runner / grid submitter.",
    epilog=__doc__,
    formatter_class=argparse.RawTextHelpFormatter,
)
# ---- single-job mode ----
parser.add_argument("-sample", default="", help="Sample name: mc/data/dirt/intime/offbeam.")
parser.add_argument("-files", default="", help="Comma-separated input .df file paths.")
parser.add_argument("-filelist", default="", help="Path to a newline-delimited list of input .df files.")
parser.add_argument("-job_id", default=0, type=int, help="Batch job id (-> tag batch_%%04d). Default 0.")
parser.add_argument("-out_dir", default="", help="Directory to write <sample>__batch_NNNN.pkl into.")
parser.add_argument("-skip_existing", action="store_true", default=False,
                     help="Skip (don't overwrite) if the output pickle already exists.")
# ---- shared map-phase options ----
parser.add_argument("-use_mc_genweight", action="store_true", default=False,
                     help="Apply the GENIE central-value event weight to MC (default off).")
parser.add_argument("-mc_univ_syst", default="", help="Comma-separated multi-universe syst tags, e.g. Flux,G4,GENIE.")
parser.add_argument("-trace", action="store_true", default=False, help="Verbose per-stage pipeline tracing.")
parser.add_argument("-selection", default="signal", choices=("signal", "pi0_sideband", "both"),
                     help="Which selection to run: the nueNp0Pi signal selection (default), the pi0 "
                          "sideband, or both. 'both' reads each input file once and writes signal "
                          "pickles to <out_dir> (grid: <work_base>/batches) and sideband pickles to "
                          "<out_dir>" + "_pi0_sideband" + ". Passed on to every grid job.")
# ---- grid-submission mode ----
parser.add_argument("-submit_grid", action="store_true", default=False,
                     help="Survey+bin-pack input files and submit one grid job per batch job, instead of "
                          "running a single job locally.")
parser.add_argument("-resubmit_missing", action="store_true", default=False,
                     help="Read existing manifest.json under -work_base, check which batch pkls are "
                          "missing or corrupt, and submit only those jobs to the grid (no re-survey, "
                          "no manifest overwrite).")
parser.add_argument("-work_base", default="", help="(grid mode) Work dir: batches/ and manifest.json live here.")
parser.add_argument("-samples", default="mc,data", help="(grid mode) Comma-separated sample list. Default mc,data.")
parser.add_argument("-max_job_bytes_gb", default=10.0, type=float,
                     help="(grid mode) Max input size per batch job, in GiB. Default 10.")
parser.add_argument("-max_files_per_sample", default="",
                     help="(grid mode) Either a plain int applied to every sample, or comma-separated "
                          "sample=N pairs (e.g. 'mc=5' or 'mc=5,data=50') -- a sample not listed is unlimited.")

args = parser.parse_args()
# Must be set before the pipeline modules are imported (they're imported lazily below).
# "both" builds each mode's runner explicitly (see run_single_job).
os.environ["NUENP0PI_SELECTION"] = "signal" if args.selection == "both" else args.selection
SIDEBAND_DIR_SUFFIX = "_pi0_sideband"


def _sideband_dir(d):
    """Where 'both' puts the sideband pickles: a sibling of the signal dir."""
    return str(d).rstrip("/") + SIDEBAND_DIR_SUFFIX


def _parse_max_files_per_sample(raw: str):
    if raw == "":
        return None
    if "=" not in raw:
        return int(raw)
    out = {}
    for part in raw.split(","):
        k, v = part.split("=")
        out[k.strip()] = int(v)
    return out


def run_single_job(files):
    from pyanalib.event_selection_pipeline import BatchJob
    from analysis_village.nueNp0Pi.event_selection import build_event_selection_pipeline
    from pyanalib.event_selection_pipeline import EventSelectionPipelineConfig

    cfg = EventSelectionPipelineConfig(
        use_mc_genweight=args.use_mc_genweight,
        mc_univ_syst=tuple(t for t in args.mc_univ_syst.split(",") if t),
        trace=args.trace,
    )
    pipeline = build_event_selection_pipeline(cfg)

    sizes = []
    for f in files:
        sizes.append(os.path.getsize(f) if os.path.isfile(f) else 0)
    job = BatchJob(sample=args.sample, job_id=args.job_id, files=files, total_bytes=sum(sizes))

    if args.selection != "both":
        out = pipeline.process_one_job(job, args.out_dir, skip_existing=args.skip_existing)
        print(f"[run_event_selection_map] wrote {out}", flush=True)
        return

    # Both selections from one read of the input files.
    from pathlib import Path
    from analysis_village.nueNp0Pi.event_selection import run_batch_selection_modes
    out_paths = {}
    for mode, d in (("signal", args.out_dir), ("pi0_sideband", _sideband_dir(args.out_dir))):
        Path(d).mkdir(parents=True, exist_ok=True)
        out_paths[mode] = str(pipeline._batch_out_path(Path(d), job.sample, job))
    if args.skip_existing and all(os.path.isfile(p) for p in out_paths.values()):
        print(f"[run_event_selection_map] {list(out_paths.values())} exist; skipping", flush=True)
        return
    run_batch_selection_modes(
        job.sample, job.files, out_paths, job_id=job.tag,
        use_mc_genweight=cfg.use_mc_genweight,
        mc_univ_syst_tags=tuple(cfg.mc_univ_syst) if job.sample == "mc" else (),
        pipeline_trace=(lambda msg: print(msg, flush=True)) if cfg.trace else None,
    )
    for p in out_paths.values():
        print(f"[run_event_selection_map] wrote {p}", flush=True)


def submit_grid():
    from analysis_village.nueNp0Pi.event_selection import build_event_selection_pipeline
    from pyanalib.event_selection_pipeline import EventSelectionPipelineConfig

    cfg = EventSelectionPipelineConfig(
        work_base=args.work_base,
        samples=tuple(s.strip() for s in args.samples.split(",") if s.strip()),
        max_job_bytes=int(args.max_job_bytes_gb * 1024 ** 3),
        max_files_per_sample=_parse_max_files_per_sample(args.max_files_per_sample),
        use_mc_genweight=args.use_mc_genweight,
        mc_univ_syst=tuple(t for t in args.mc_univ_syst.split(",") if t),
    )
    pipeline = build_event_selection_pipeline(cfg)

    # Survey + bin-pack happens HERE, locally/interactively (cheap -- just os.path.getsize
    # + globbing, no data read) -- writes the same manifest.json a local run_map() would,
    # so run_aggregate()/check_batches() later cross-check grid output against it exactly
    # like a local run's output.
    records, jobs, manifest_path = pipeline.discover_jobs()
    print(f"[run_event_selection_map] {len(jobs)} batch job(s), manifest={manifest_path}", flush=True)

    work, batches_dir, _ = pipeline.resolve_paths()

    JobStartTime = datetime.datetime.now()
    timestamp = JobStartTime.strftime("%Y_%m_%d_%H%M%S")

    CAFPYANA_GRID_OUT_DIR = os.environ["CAFPYANA_GRID_OUT_DIR"]
    MasterJobDir = CAFPYANA_GRID_OUT_DIR + "/logs/" + timestamp + "__event_selection_map_log"
    os.system("mkdir -p " + MasterJobDir)

    # Grid nodes can only write to /pnfs/ via ifdh -- /exp/sbnd/data/ is NFS and
    # ifdh/gfal can't mkdir_p or cp there from the grid.  Use a pnfs staging area;
    # after jobs finish, pull with:
    #   ifdh cp -r <grid_batches_dir>/ <work_base>/batches/
    grid_batches_dir = CAFPYANA_GRID_OUT_DIR + "/event_sel_batches/" + timestamp

    # One grid job per BatchJob -- discover_jobs() already size-capped each one via
    # max_job_bytes, so (unlike run_df_maker.py's -ngrid, which round-robins a
    # user-chosen worker count across files) the job count isn't a free parameter here.
    for i, job in enumerate(jobs):
        out = open(MasterJobDir + "/run_%d.sh" % i, "w")
        out.write("#!/bin/bash\n")
        for f in job.files:
            out.write('echo "[run_%d.sh] input : %s"\n' % (i, f))
            out.write("ifdh cp " + f + " .\n")
        local_files = ",".join(os.path.basename(f) for f in job.files)
        cmd = (
            "python run_event_selection_map.py"
            " -sample %s -job_id %d -out_dir ./out -skip_existing"
            " -files %s" % (job.sample, job.job_id, local_files)
        )
        if args.use_mc_genweight:
            cmd += " -use_mc_genweight"
        if args.mc_univ_syst:
            cmd += " -mc_univ_syst " + args.mc_univ_syst
        cmd += " -selection " + args.selection
        out.write("mkdir -p ./out\n")
        out.write(cmd + "\n")
        out.close()

    os.system("cp ./bin/grid_executable_event_selection.sh %s" % MasterJobDir)

    CAFPYANA_WD = os.environ["CAFPYANA_WD"]
    _site = CAFPYANA_WD + "/envs/venv_py310_cafpyana/lib/python3.10/site-packages"
    cp_XRootD = "cp -r " + _site + "/XRootD " + MasterJobDir
    cp_pyxrootd = "cp -r " + _site + "/pyxrootd " + MasterJobDir
    os.system(cp_XRootD)
    os.system(cp_pyxrootd)
    # pyxrootd links its bundled libssl/libcrypto etc. from ../xrootd.libs
    os.system("cp -r " + _site + "/xrootd.libs " + MasterJobDir)

    # ship the working tree (tracked files incl. uncommitted edits), not just HEAD
    archive_repo = "git archive -o " + MasterJobDir + "/cafpyana.tar.gz $(git stash create | grep . || echo HEAD)"
    os.system(archive_repo)

    os.chdir(MasterJobDir)
    os.system("tar cf bin_dir.tar ./")

    submitCMD = '''jobsub_submit \\
-G sbnd \\
--auth-methods="token" \\
-e LC_ALL=C \\
--role=Analysis \\
--resource-provides="usage_model=DEDICATED,OPPORTUNISTIC,OFFSITE" \\
-l '+SingularityImage=\\"/cvmfs/singularity.opensciencegrid.org/fermilab/fnal-wn-el9\\:9.7\\"' \\
--lines '+FERMIHTC_AutoRelease=True' --lines '+FERMIHTC_GraceMemory=1000' --lines '+FERMIHTC_GraceLifetime=3600' \\
--append_condor_requirements='(TARGET.HAS_SINGULARITY=?=true)' \\
--tar_file_name "dropbox://$(pwd)/bin_dir.tar" \\
-N %d \\
--disk 100GB \\
--cpu 1 \\
--memory 10GB \\
--expected-lifetime 24h \\
"file://$(pwd)/grid_executable_event_selection.sh" \\
"%s"''' % (len(jobs), grid_batches_dir)

    print(submitCMD)
    os.system(submitCMD)

    os.chdir(CAFPYANA_WD)
    print(
        f"[run_event_selection_map] submitted {len(jobs)} job(s)\n"
        f"[run_event_selection_map] grid output (pnfs) -> {grid_batches_dir}\n"
        f"[run_event_selection_map] once jobs land, pull to work_base with:\n"
        f"  ifdh cp -r {grid_batches_dir}/ {batches_dir}/\n"
        + (f"  ifdh cp -r {_sideband_dir(grid_batches_dir)}/ {_sideband_dir(batches_dir)}/\n"
           if args.selection == "both" else "")
        + f"then cross-check completeness with pipeline.check_batches() before aggregating.",
        flush=True,
    )


def resubmit_missing():
    from analysis_village.nueNp0Pi.event_selection import build_event_selection_pipeline
    from pyanalib.event_selection_pipeline import EventSelectionPipelineConfig

    cfg = EventSelectionPipelineConfig(
        work_base=args.work_base,
        use_mc_genweight=args.use_mc_genweight,
        mc_univ_syst=tuple(t for t in args.mc_univ_syst.split(",") if t),
    )
    pipeline = build_event_selection_pipeline(cfg)

    work, batches_dir, _ = pipeline.resolve_paths()
    report = pipeline.check_batches(batches_dir=batches_dir)
    jobs = list(report.needs_resubmit)
    if args.selection == "both":
        # a job is redone if either its signal or its sideband pickle is missing/corrupt
        sb_report = pipeline.check_batches(batches_dir=_sideband_dir(batches_dir))
        have = {(j.sample, j.job_id) for j in jobs}
        jobs += [j for j in sb_report.needs_resubmit if (j.sample, j.job_id) not in have]
    if not jobs:
        print("[run_event_selection_map] no missing or corrupt batches — nothing to resubmit.")
        return

    print(f"[run_event_selection_map] {len(jobs)} batch job(s) to resubmit (missing/corrupt)", flush=True)

    JobStartTime = datetime.datetime.now()
    timestamp = JobStartTime.strftime("%Y_%m_%d_%H%M%S")

    CAFPYANA_GRID_OUT_DIR = os.environ["CAFPYANA_GRID_OUT_DIR"]
    MasterJobDir = CAFPYANA_GRID_OUT_DIR + "/logs/" + timestamp + "__event_selection_resubmit_log"
    os.system("mkdir -p " + MasterJobDir)

    grid_batches_dir = CAFPYANA_GRID_OUT_DIR + "/event_sel_batches/" + timestamp

    for i, job in enumerate(jobs):
        out = open(MasterJobDir + "/run_%d.sh" % i, "w")
        out.write("#!/bin/bash\n")
        for f in job.files:
            out.write('echo "[run_%d.sh] input : %s"\n' % (i, f))
            out.write("ifdh cp " + f + " .\n")
        local_files = ",".join(os.path.basename(f) for f in job.files)
        cmd = (
            "python run_event_selection_map.py"
            " -sample %s -job_id %d -out_dir ./out -skip_existing"
            " -files %s" % (job.sample, job.job_id, local_files)
        )
        if args.use_mc_genweight:
            cmd += " -use_mc_genweight"
        if args.mc_univ_syst:
            cmd += " -mc_univ_syst " + args.mc_univ_syst
        cmd += " -selection " + args.selection
        out.write("mkdir -p ./out\n")
        out.write(cmd + "\n")
        out.close()

    os.system("cp ./bin/grid_executable_event_selection.sh %s" % MasterJobDir)

    CAFPYANA_WD = os.environ["CAFPYANA_WD"]
    _site = CAFPYANA_WD + "/envs/venv_py310_cafpyana/lib/python3.10/site-packages"
    os.system("cp -r " + _site + "/XRootD " + MasterJobDir)
    os.system("cp -r " + _site + "/pyxrootd " + MasterJobDir)
    os.system("cp -r " + _site + "/xrootd.libs " + MasterJobDir)

    # ship the working tree (tracked files incl. uncommitted edits), not just HEAD
    os.system("git archive -o " + MasterJobDir + "/cafpyana.tar.gz $(git stash create | grep . || echo HEAD)")

    os.chdir(MasterJobDir)
    os.system("tar cf bin_dir.tar ./")

    submitCMD = '''jobsub_submit \\
-G sbnd \\
--auth-methods="token" \\
-e LC_ALL=C \\
--role=Analysis \\
--resource-provides="usage_model=DEDICATED,OPPORTUNISTIC,OFFSITE" \\
-l '+SingularityImage=\\"/cvmfs/singularity.opensciencegrid.org/fermilab/fnal-wn-el9\\:9.7\\"' \\
--lines '+FERMIHTC_AutoRelease=True' --lines '+FERMIHTC_GraceMemory=1000' --lines '+FERMIHTC_GraceLifetime=3600' \\
--append_condor_requirements='(TARGET.HAS_SINGULARITY=?=true)' \\
--tar_file_name "dropbox://$(pwd)/bin_dir.tar" \\
-N %d \\
--disk 100GB \\
--cpu 1 \\
--memory 10GB \\
--expected-lifetime 24h \\
"file://$(pwd)/grid_executable_event_selection.sh" \\
"%s"''' % (len(jobs), grid_batches_dir)

    print(submitCMD)
    os.system(submitCMD)

    os.chdir(CAFPYANA_WD)
    print(
        f"[run_event_selection_map] submitted {len(jobs)} missing/corrupt job(s)\n"
        f"[run_event_selection_map] grid output (pnfs) -> {grid_batches_dir}\n"
        f"[run_event_selection_map] once jobs land, pull to work_base with:\n"
        f"  ifdh cp -r {grid_batches_dir}/ {batches_dir}/\n"
        + (f"  ifdh cp -r {_sideband_dir(grid_batches_dir)}/ {_sideband_dir(batches_dir)}/\n"
           if args.selection == "both" else "")
        + f"then re-check completeness with pipeline.check_batches() before aggregating.",
        flush=True,
    )


if __name__ == "__main__":
    if args.submit_grid:
        if args.work_base == "":
            parser.print_help()
            sys.exit(1)
        submit_grid()
    elif args.resubmit_missing:
        if args.work_base == "":
            parser.print_help()
            sys.exit(1)
        resubmit_missing()
    else:
        if args.sample == "" or args.out_dir == "" or (args.files == "" and args.filelist == ""):
            parser.print_help()
            sys.exit(1)
        if args.filelist:
            with open(args.filelist) as f:
                files = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        else:
            files = [f for f in args.files.split(",") if f]
        run_single_job(files)
