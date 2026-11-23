"""nueNp0Pi Parquet skim: make it once, then rerun the selection/plots from it.

The event-selection pipeline (``config/stages.py``) is split at
``config.stages.SKIM_STAGE_KEY``:

* **Making the skim** (:func:`run_batch_skim`, one call per batch job, grid or local):
  reads the full ``.df`` inputs, runs the stages up to and including the skim stage,
  saves their histograms/cutflow/efficiency-denominator as an ordinary batch pickle (with
  the real POT/gates metadata, so ``check_batches`` works unchanged), and writes the
  surviving ``evt`` slices and their ``trk`` rows to Parquet "part" files.
* **Merging** (:func:`merge_skim`): concatenates the parts into
  ``<skim_dir>/<sample>_{evt,trk}.parquet`` and sums the batch pickles into one
  ``<skim_dir>/preskim/<mode>/<sample>__preskim.pkl`` per selection mode.
* **Running over the skim** (:func:`run_skim_selection` / :func:`run_from_skim`): reads the
  skim one chunk (input file) at a time, runs the full stage list but fills only the
  stages after the skim stage, and writes ``<sample>__skim_NNNN.pkl`` pickles next to a
  copy of the preskim pickle. That directory aggregates exactly like a normal batches
  directory (``event_selection_aggregate``).

What needs a new skim: any change to a cut at or before ``SKIM_STAGE_KEY``, to the df
maker, or to the columns computed in :func:`event_selection.load_prepared_file`. Everything
after the skim stage (cuts, plots, binning, N-1, breakdowns) only needs a rerun from the
skim.

Chunk ids are ``job_id * 1000 + file index`` and ``__ntuple`` is rewritten to
``chunk_id * 10000 + __ntuple`` so (``__ntuple``, ``entry``) stays unique across input files
when several chunks are read together.
"""
from __future__ import annotations

import glob
import json
import os
import pickle
import subprocess
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

import analysis_village.nueNp0Pi.config.stages as stages_mod
from analysis_village.nueNp0Pi.config.datasets import KEYS2LOAD
from analysis_village.nueNp0Pi.event_selection import (
    accumulate_hdr_meta,
    build_runner,
    file_hdr_meta,
    hdf_has_mcnu,
    load_prepared_file,
)
from pyanalib.chunked_selection import aggregate_chunk_files
from pyanalib.logging_utils import get_logger
from pyanalib.parquet_skim import SkimFile, SkimWriter, iter_skim_chunks, merge_skim_files

logger = get_logger(__name__)

SAMPLES = ("mc", "data", "intime", "offbeam", "dirt")
FILES_PER_JOB_MAX = 1000      # chunk_id = job_id * 1000 + file index
NTUPLE_STRIDE = 10000         # new __ntuple = chunk_id * 10000 + old __ntuple
META_SUM_KEYS = ("chunk_pot", "chunk_gates_bnb", "chunk_cosmic_gates_intime",
                 "chunk_cosmic_gates_offbeam", "n_evt", "n_files")


# ===========================================================================
# helpers
# ===========================================================================
def part_paths(out_dir: str | Path, sample: str, job_tag: str) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    return {kind: out_dir / f"{sample}_{kind}__{job_tag}.parquet" for kind in ("evt", "trk")}


def skim_paths(skim_dir: str | Path, sample: str) -> Dict[str, Path]:
    skim_dir = Path(skim_dir)
    return {kind: skim_dir / f"{sample}_{kind}.parquet" for kind in ("evt", "trk")}


def preskim_pickle_path(skim_dir: str | Path, mode: str, sample: str) -> Path:
    return Path(skim_dir) / "preskim" / mode / f"{sample}__preskim.pkl"


def _stages_for_mode(mode: str):
    prev = stages_mod.SELECTION_MODE
    try:
        stages_mod.set_selection_mode(mode)
        return stages_mod.build_pipeline()
    finally:
        stages_mod.set_selection_mode(prev)


def _runner_for_mode(mode: str, sample: str, stages, fill_stage_keys):
    prev = stages_mod.SELECTION_MODE
    try:
        stages_mod.set_selection_mode(mode)
        return build_runner(sample, stages=stages, fill_stage_keys=fill_stage_keys)
    finally:
        stages_mod.set_selection_mode(prev)


def _refuse_existing(paths, what: str) -> None:
    """Never overwrite or delete existing outputs: stop and name them instead. Rerun with a
    new output location, or remove the old files yourself first."""
    existing = [str(p) for p in paths if Path(p).exists()]
    if existing:
        raise FileExistsError(
            f"{what}: {len(existing)} output file(s) already exist and will not be overwritten, "
            f"e.g. {existing[:3]} -- choose a new output location or remove them yourself")


def _save_runner(runner, out_path: str | Path, full_stages, meta: Dict[str, Any]) -> None:
    """Save like ``ChunkRunner.save`` but always record the FULL stage list, so pickles
    holding only some stages still merge into a complete cutflow."""
    d = runner.to_dict()
    d["stage_keys"] = [s.key for s in full_stages]
    d["stage_labels"] = [s.label for s in full_stages]
    d["meta"] = meta
    _refuse_existing([out_path], "histogram pickle")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(out_path) + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(d, f)
    os.replace(tmp, out_path)


def _remap_ntuple(df: pd.DataFrame, chunk_id: int) -> pd.DataFrame:
    if df is None or "__ntuple" not in (df.index.names or []):
        return df
    lvl = df.index.names.index("__ntuple")
    old = np.asarray(df.index.get_level_values(lvl), dtype=np.int64)
    if len(old) and (old.min() < 0 or old.max() >= NTUPLE_STRIDE):
        raise ValueError(f"__ntuple outside [0, {NTUPLE_STRIDE}) -- raise NTUPLE_STRIDE")
    new = old + np.int64(chunk_id) * NTUPLE_STRIDE
    if isinstance(df.index, pd.MultiIndex):
        df.index = df.index.set_levels(
            df.index.levels[lvl].astype(np.int64) + np.int64(chunk_id) * NTUPLE_STRIDE,
            level=lvl, verify_integrity=False)
    else:
        df.index = pd.Index(new, name="__ntuple")
    return df


def _tracks_of(trk: pd.DataFrame, evt: pd.DataFrame) -> pd.DataFrame:
    """Rows of ``trk`` whose slice (all index levels but the last) is in ``evt``."""
    if trk is None or len(trk) == 0:
        return trk
    return trk[trk.index.droplevel(-1).isin(evt.index)]


def _git_hash() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    except Exception:
        return ""


# ===========================================================================
# 1. make the skim (one batch job)
# ===========================================================================
def run_batch_skim(
    sample: str,
    df_files: Sequence[str],
    out_paths: Dict[str, str],
    parts_dir: str,
    *,
    job_id: int,
    use_mc_genweight: bool = False,
    max_splits_per_file: int | None = None,
    pipeline_trace=None,
) -> Dict[str, Any]:
    """Skim one batch of ``.df`` files.

    ``out_paths``: ``{selection_mode: pickle path}`` for the pre-skim histograms (the same
    paths a normal map job would write, so ``check_batches``/``-resubmit_missing`` work).
    ``parts_dir``: where ``<sample>_{evt,trk}__batch_NNNN.parquet`` are written.
    """
    if len(df_files) >= FILES_PER_JOB_MAX:
        raise ValueError(f"at most {FILES_PER_JOB_MAX - 1} files per skim job")
    job_tag = f"batch_{job_id:04d}"
    modes = list(out_paths)

    full_stages = {m: _stages_for_mode(m) for m in modes}
    pre_keys = {m: stages_mod.split_at_skim(full_stages[m])[0] for m in modes}
    if len({tuple(k) for k in pre_keys.values()}) != 1:
        raise ValueError(f"pre-skim stages differ between modes: {pre_keys}")
    n_pre = len(pre_keys[modes[0]])
    # Fresh stage objects for the runners (only the pre-skim part is run here).
    runners = {
        m: _runner_for_mode(m, sample, _stages_for_mode(m)[:n_pre], None) for m in modes
    }

    load_mcnu = sample == "mc" and any(hdf_has_mcnu(f) for f in df_files)
    keys_load = list(KEYS2LOAD) + (["mcnu"] if load_mcnu else [])

    paths = part_paths(parts_dir, sample, job_tag)
    _refuse_existing(list(paths.values()) + list(out_paths.values()), f"skim {job_tag}")
    Path(parts_dir).mkdir(parents=True, exist_ok=True)
    tmp = {k: Path(str(p) + ".tmp") for k, p in paths.items()}
    pot, bnb, intime, offbeam = [0.0], [0.0], [0.0], [0.0]
    per_file, chunk_ids = [], []
    n_in = n_evt_out = n_trk_out = 0

    with SkimWriter(str(tmp["evt"])) as w_evt, SkimWriter(str(tmp["trk"])) as w_trk:
        for i, df_file in enumerate(df_files):
            chunk_id = job_id * FILES_PER_JOB_MAX + i
            evt, trk, hdr, mcnu = load_prepared_file(
                df_file, sample, keys_load, use_mc_genweight=use_mc_genweight,
                max_splits_per_file=max_splits_per_file)
            fmeta = file_hdr_meta(sample, hdr, df_file)
            accumulate_hdr_meta(sample, hdr, pot, bnb, intime, offbeam)
            n_in += len(evt)

            survived = None
            for m, runner in runners.items():
                st = runner.run({"evt": evt, "trk": trk, "hdr": hdr, "mcnu": mcnu},
                                pipeline_trace=pipeline_trace)
                if survived is None:
                    survived = st["evt"]
                elif len(st["evt"]) != len(survived):
                    raise RuntimeError("pre-skim cuts kept different slices in different modes")

            evt_s = _remap_ntuple(survived.copy(), chunk_id)
            trk_s = _remap_ntuple(_tracks_of(trk, survived).copy(), chunk_id)
            w_evt.write(evt_s, chunk_id)
            w_trk.write(trk_s, chunk_id)
            n_evt_out += len(evt_s)
            n_trk_out += len(trk_s)
            fmeta.update(chunk_id=chunk_id, n_evt_in=len(evt), n_evt_skim=len(evt_s),
                         n_trk_skim=len(trk_s))
            per_file.append(fmeta)
            chunk_ids.append(chunk_id)
            logger.info("skim %s: %d -> %d slices, %d tracks", df_file, len(evt),
                        len(evt_s), len(trk_s))
            del evt, trk, hdr, mcnu, survived, evt_s, trk_s
    for k in tmp:
        if tmp[k].exists():
            os.replace(tmp[k], paths[k])

    meta = {
        "weight_scheme": "intrinsic",
        "use_mc_genweight": bool(use_mc_genweight),
        "mc_univ_syst_tags": [],
        "df_files": list(df_files),
        "per_file": per_file,
        "sample": sample,
        "job_id": job_tag,
        "chunk_pot": pot[0],
        "chunk_gates_bnb": bnb[0],
        "chunk_cosmic_gates_intime": intime[0],
        "chunk_cosmic_gates_offbeam": offbeam[0],
        "n_evt": n_in,
        "n_files": len(df_files),
        "workflow": "skim",
        "skim_stage_key": stages_mod.SKIM_STAGE_KEY,
        "skim_chunk_ids": chunk_ids,
        "n_evt_skim": n_evt_out,
        "n_trk_skim": n_trk_out,
        "skim_parts": {k: os.path.basename(p) for k, p in paths.items()},
        "mc_efficiency_enabled": load_mcnu,
    }
    for m, runner in runners.items():
        _save_runner(runner, out_paths[m], full_stages[m], dict(meta, selection_mode=m))
    return meta


# ===========================================================================
# 2. merge grid parts into the skim directory
# ===========================================================================
def _sum_pickles(files: Sequence[str]) -> Dict[str, Any]:
    out = aggregate_chunk_files(list(files))
    metas = []
    for cf in files:
        with open(cf, "rb") as f:
            metas.append(pickle.load(f).get("meta") or {})
    m = dict(metas[0])
    for k in META_SUM_KEYS:
        m[k] = sum(float(x.get(k, 0.0) or 0.0) for x in metas)
    m["n_evt"] = int(m["n_evt"])
    m["n_files"] = int(m["n_files"])
    m["df_files"] = [f for x in metas for f in x.get("df_files", [])]
    m["per_file"] = [p for x in metas for p in x.get("per_file", [])]
    m["job_id"] = "preskim"
    m["n_batches"] = len(files)
    out["meta"] = m
    return out


def merge_skim(batch_dirs: Dict[str, str], skim_dir: str) -> Dict[str, Any]:
    """Merge skim job outputs into ``skim_dir``.

    ``batch_dirs``: ``{selection_mode: dir}`` holding that mode's ``<sample>__batch_NNNN.pkl``
    pre-skim pickles; the Parquet parts are read from the first mode's dir.
    Every pickle must have its two parts (and vice versa), or this raises.
    """
    skim_dir = Path(skim_dir)
    skim_dir.mkdir(parents=True, exist_ok=True)
    modes = list(batch_dirs)
    parts_dir = Path(batch_dirs[modes[0]])
    manifest: Dict[str, Any] = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "git": _git_hash(),
        "skim_stage_key": stages_mod.SKIM_STAGE_KEY,
        "batch_dirs": {m: str(d) for m, d in batch_dirs.items()},
        "samples": {},
    }
    for sample in SAMPLES:
        pkls = {m: sorted(glob.glob(os.path.join(d, f"{sample}__batch_*.pkl")))
                for m, d in batch_dirs.items()}
        if not any(pkls.values()):
            continue
        tags = {m: [Path(p).stem.split("__", 1)[1] for p in v] for m, v in pkls.items()}
        if len({tuple(t) for t in tags.values()}) != 1:
            raise RuntimeError(f"{sample}: modes have different batch pickles: "
                               f"{ {m: len(t) for m, t in tags.items()} }")
        job_tags = tags[modes[0]]
        parts = {k: [] for k in ("evt", "trk")}
        missing = []
        for tag in job_tags:
            for k, p in part_paths(parts_dir, sample, tag).items():
                (parts[k] if p.is_file() else missing).append(str(p))
        if missing:
            raise RuntimeError(f"{sample}: {len(missing)} Parquet part(s) missing, e.g. {missing[:3]}")
        extra = sorted(set(glob.glob(str(parts_dir / f"{sample}_evt__batch_*.parquet")))
                       - set(parts["evt"]))
        if extra:
            logger.warning("%s: %d evt part(s) without a pickle ignored, e.g. %s",
                           sample, len(extra), extra[:3])

        out = skim_paths(skim_dir, sample)
        _refuse_existing(list(out.values()) + [preskim_pickle_path(skim_dir, m, sample) for m in modes],
                         f"merge {sample}")
        n_rows = {}
        for k in ("evt", "trk"):
            tmp = str(out[k]) + ".tmp"
            n_rows[k] = merge_skim_files(parts[k], tmp)
            os.replace(tmp, out[k])
        pot = None
        for m in modes:
            summed = _sum_pickles(pkls[m])
            dest = preskim_pickle_path(skim_dir, m, sample)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                pickle.dump(summed, f)
            pot = summed["meta"]["chunk_pot"]
        manifest["samples"][sample] = {
            "n_jobs": len(job_tags),
            "n_evt_rows": n_rows["evt"],
            "n_trk_rows": n_rows["trk"],
            "chunk_pot": pot,
            "evt_bytes": os.path.getsize(out["evt"]),
            "trk_bytes": os.path.getsize(out["trk"]),
        }
        print(f"[skim] {sample}: {len(job_tags)} jobs -> {n_rows['evt']} slices, "
              f"{n_rows['trk']} tracks, "
              f"{(manifest['samples'][sample]['evt_bytes'] + manifest['samples'][sample]['trk_bytes']) / 1e9:.2f} GB",
              flush=True)
    with open(skim_dir / "skim_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


# ===========================================================================
# 3. run the selection over the skim
# ===========================================================================
def run_skim_selection(
    sample: str,
    skim_dir: str,
    out_paths: Dict[str, str],
    *,
    chunk_ids: Optional[Sequence[int]] = None,
    pipeline_trace=None,
) -> Dict[str, Any]:
    """Run the post-skim stages over (some of) a sample's skim chunks.

    ``out_paths``: ``{selection_mode: pickle path}``. The pickles hold only the stages
    after ``SKIM_STAGE_KEY`` and ``chunk_pot = 0``; aggregate them together with
    ``preskim/<mode>/<sample>__preskim.pkl`` (which :func:`run_from_skim` copies in).
    """
    paths = skim_paths(skim_dir, sample)
    modes = list(out_paths)
    full_stages, runners = {}, {}
    for m in modes:
        full_stages[m] = _stages_for_mode(m)
        _, post = stages_mod.split_at_skim(full_stages[m])
        runners[m] = _runner_for_mode(m, sample, _stages_for_mode(m), post)

    n_evt = 0
    ids: List[int] = []
    for cid, evt, trk in iter_skim_chunks(str(paths["evt"]), str(paths["trk"]),
                                          chunk_ids=chunk_ids):
        for runner in runners.values():
            runner.run({"evt": evt, "trk": trk, "hdr": None, "mcnu": None},
                       pipeline_trace=pipeline_trace)
        n_evt += len(evt)
        ids.append(cid)

    meta = {
        "sample": sample,
        "workflow": "from_skim",
        "skim_dir": str(skim_dir),
        "skim_chunk_ids": ids,
        "chunk_pot": 0.0,          # exposure comes from the preskim pickle
        "chunk_gates_bnb": 0.0,
        "chunk_cosmic_gates_intime": 0.0,
        "chunk_cosmic_gates_offbeam": 0.0,
        "n_evt": n_evt,
        "n_files": 0,
        "df_files": [],
    }
    for m, runner in runners.items():
        _save_runner(runner, out_paths[m], full_stages[m], dict(meta, selection_mode=m))
    return meta


def _skim_task(task):
    sample, skim_dir, out_paths, chunk_ids, mode_env = task
    os.environ["NUENP0PI_SELECTION"] = mode_env
    run_skim_selection(sample, skim_dir, out_paths, chunk_ids=chunk_ids)
    return sample, list(out_paths.values())


def run_from_skim(
    skim_dir: str,
    out_dirs: Dict[str, str],
    *,
    samples: Optional[Sequence[str]] = None,
    nproc: int = 1,
    chunks_per_task: int = 10,
) -> Dict[str, List[str]]:
    """Run the selection over a whole skim directory.

    ``out_dirs``: ``{selection_mode: output dir}``; each ends up with
    ``<sample>__skim_NNNN.pkl`` plus ``<sample>__preskim.pkl`` and can be passed straight
    to ``event_selection_aggregate`` as its input dir.
    """
    skim_dir = str(skim_dir)
    if samples is None:
        samples = [s for s in SAMPLES if skim_paths(skim_dir, s)["evt"].is_file()]
    for d in out_dirs.values():
        _refuse_existing(glob.glob(os.path.join(d, "*.pkl")), f"from-skim output dir {d}")
        Path(d).mkdir(parents=True, exist_ok=True)

    tasks = []
    for sample in samples:
        for m, d in out_dirs.items():
            src = preskim_pickle_path(skim_dir, m, sample)
            if not src.is_file():
                raise FileNotFoundError(f"{src} missing -- run -merge_skim first")
            with open(src, "rb") as f_in, open(os.path.join(d, f"{sample}__preskim.pkl"), "wb") as f_out:
                f_out.write(f_in.read())
        ids = SkimFile(str(skim_paths(skim_dir, sample)["evt"])).chunk_ids
        for k, i in enumerate(range(0, len(ids), chunks_per_task)):
            outs = {m: os.path.join(d, f"{sample}__skim_{k:04d}.pkl") for m, d in out_dirs.items()}
            tasks.append((sample, skim_dir, outs, ids[i:i + chunks_per_task],
                          stages_mod.SELECTION_MODE))
    print(f"[skim] {len(tasks)} task(s) over {list(samples)} with nproc={nproc}", flush=True)

    written: Dict[str, List[str]] = {s: [] for s in samples}
    if nproc > 1:
        with Pool(nproc) as pool:
            for sample, outs in pool.imap_unordered(_skim_task, tasks):
                written[sample] += outs
    else:
        for t in tasks:
            sample, outs = _skim_task(t)
            written[sample] += outs
    return written
