"""Generic survey -> bin-pack -> map -> aggregate/render orchestration for
chunked event-selection pipelines built on :mod:`pyanalib.chunked_selection`.

This is the analysis-agnostic half of what used to be
``analysis_village/nueNp0Pi/event_selection_batched.py`` (now folded into
``analysis_village/nueNp0Pi/event_selection.py``): listing input
``.df`` files, packing them into <=N-byte jobs, running the map phase
(optionally across a ``multiprocessing.Pool``), and driving the
aggregate/render phase. None of that logic actually depends on nueNp0Pi --
only on being handed:

* ``iter_df_paths``: ``sample -> Iterator[str]`` of input file paths.
* ``run_batch_selection``: ``(sample, files, out_path, *, job_id, ...) -> dict``
  meta, the analysis's own map-phase function (loads files, runs its
  ``ChunkRunner``, writes one pickle). Needs ``meta["n_evt"]`` and
  ``meta["chunk_pot"]`` for progress logging.
* ``aggregate_and_render``: ``(batches_dir, plots_dir, **kwargs) -> dict``
  payload, the analysis's own reduce/render function. Needs
  ``payload["pot_str"]`` and ``payload["data_pot"]``.

An analysis wires these up once (see
``analysis_village.nueNp0Pi.event_selection.build_event_selection_pipeline``
for the nueNp0Pi example) and gets survey/map/aggregate/run_full for free.

Typical usage::

    pipeline = EventSelectionPipeline(cfg, hooks)
    result = pipeline.run_full()
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

DEFAULT_MAX_JOB_BYTES = 1 << 30  # 1 GiB


@dataclass
class FileRecord:
    sample: str
    path: str
    size_bytes: int

    @property
    def size_gb(self) -> float:
        return self.size_bytes / (1024.0 ** 3)


@dataclass
class BatchJob:
    sample: str
    job_id: int
    files: List[str]
    total_bytes: int

    @property
    def tag(self) -> str:
        return f"batch_{self.job_id:04d}"


@dataclass
class EventSelectionPipelineConfig:
    work_base: Path | str | None = None
    batches_dir: Path | str | None = None
    plots_dir: Path | str | None = None
    max_job_bytes: int = DEFAULT_MAX_JOB_BYTES
    # Optional map-phase bookkeeping only (not used to form overlay bands).
    mc_univ_syst: Sequence[str] = ()
    use_mc_genweight: bool = False
    skip_existing_batches: bool = True
    aggregate_only: bool = False
    skip_aggregate: bool = False
    cosmic_estimate: str = "intime"
    f_offbeam_frac: float = 0.08
    save_fig: bool = True
    show_fig: bool = False
    syst_disk_root: Path | str | None = None
    syst_disk_env_var: str = "SYST_DISK_ROOT"
    syst_tag: str = ""
    samples: Sequence[str] = ("mc",)
    max_files_per_sample: int | None = None
    trace: bool = False
    # >1 -> run batch jobs in a multiprocessing.Pool (maxtasksperchild=1, so
    # each job still gets a fresh worker process torn down after -- same
    # per-job memory isolation a subprocess-per-job design gives, without the
    # CLI-args/pickle round-trip of actually spawning `python script.py`.
    # Default 1 (sequential, in the calling process) has the smallest memory
    # footprint; raise it if you have RAM to spare for real parallelism.
    n_workers: int = 1


@dataclass
class EventSelectionPipelineResult:
    work_base: Path
    batches_dir: Path
    plots_dir: Path
    batch_paths: Dict[str, List[str]]
    failed: List[Tuple[str, str, str]]
    merged_payload: dict | None
    pot_str: str
    data_pot: float
    manifest_path: Path


@dataclass
class EventSelectionHooks:
    """The analysis-specific callables the generic pipeline needs.

    All three are required. ``default_work_root`` / ``default_syst_disk_root``
    are optional convenience defaults -- if omitted, the corresponding
    :class:`EventSelectionPipelineConfig` field must be set explicitly
    (``work_base``, ``syst_disk_root``, or the ``syst_disk_env_var`` env var).
    """
    iter_df_paths: Callable[[str], Iterator[str]]
    run_batch_selection: Callable[..., Dict[str, Any]]
    aggregate_and_render: Callable[..., Dict[str, Any]]
    default_work_root: Optional[Callable[[Optional[str]], Path]] = None
    default_syst_disk_root: Optional[Callable[[], Path]] = None
    # Only used for a friendlier "no files found" message; safe to omit.
    describe_glob: Optional[Callable[[str], str]] = None


class EventSelectionPipeline:
    """Survey -> bin-pack -> map -> aggregate/render, parameterized by hooks."""

    def __init__(self, cfg: EventSelectionPipelineConfig, hooks: EventSelectionHooks):
        self.cfg = cfg
        self.hooks = hooks

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    def resolve_paths(self) -> Tuple[Path, Path, Path]:
        cfg = self.cfg
        today = datetime.now().strftime("%Y%m%d")
        work_base = cfg.work_base
        if work_base is None:
            if self.hooks.default_work_root is None:
                raise ValueError(
                    "EventSelectionPipelineConfig.work_base is unset and no "
                    "hooks.default_work_root was provided -- set one or the other."
                )
            work_base = self.hooks.default_work_root(today)
        work = Path(work_base).expanduser()
        batches = Path(cfg.batches_dir or work / "batches").expanduser()
        plots = Path(cfg.plots_dir or work / "plots").expanduser()
        return work, batches, plots

    def _resolve_syst_disk_root(self) -> Optional[Path]:
        cfg = self.cfg
        syst_root = cfg.syst_disk_root
        if syst_root is None:
            syst_root = os.environ.get(cfg.syst_disk_env_var)
        if syst_root is None and self.hooks.default_syst_disk_root is not None:
            syst_root = self.hooks.default_syst_disk_root()
        if syst_root is None:
            return None
        return Path(syst_root).expanduser()

    # ------------------------------------------------------------------
    # Survey / bin-pack
    # ------------------------------------------------------------------
    def survey_files(self) -> List[FileRecord]:
        """List input ``.df`` files with on-disk sizes."""
        cfg = self.cfg
        records: List[FileRecord] = []
        for sample in cfg.samples:
            where = self.hooks.describe_glob(sample) if self.hooks.describe_glob else "configured glob"
            print(f"[event_selection] {sample}: looking in {where}", flush=True)
            paths = list(self.hooks.iter_df_paths(sample))
            print(f"[event_selection] {sample}: found {len(paths)} file(s)", flush=True)
            if cfg.max_files_per_sample is not None:
                paths = paths[: max(0, int(cfg.max_files_per_sample))]
            for p in paths:
                try:
                    size = os.path.getsize(p)
                except OSError:
                    continue
                records.append(FileRecord(sample=sample, path=p, size_bytes=size))
        return records

    def group_files_into_jobs(self, records: Sequence[FileRecord]) -> List[BatchJob]:
        """Greedy bin-packing: group files per sample without exceeding ``max_job_bytes``."""
        max_bytes = self.cfg.max_job_bytes
        by_sample: Dict[str, List[FileRecord]] = {}
        for rec in records:
            by_sample.setdefault(rec.sample, []).append(rec)

        jobs: List[BatchJob] = []
        for sample in sorted(by_sample):
            files = sorted(by_sample[sample], key=lambda r: r.path)
            batch_files: List[str] = []
            batch_bytes = 0
            job_id = 0
            for rec in files:
                if rec.size_bytes > max_bytes:
                    if batch_files:
                        jobs.append(
                            BatchJob(sample=sample, job_id=job_id, files=batch_files, total_bytes=batch_bytes)
                        )
                        job_id += 1
                        batch_files, batch_bytes = [], 0
                    jobs.append(
                        BatchJob(
                            sample=sample,
                            job_id=job_id,
                            files=[rec.path],
                            total_bytes=rec.size_bytes,
                        )
                    )
                    job_id += 1
                    continue
                if batch_files and batch_bytes + rec.size_bytes > max_bytes:
                    jobs.append(
                        BatchJob(sample=sample, job_id=job_id, files=batch_files, total_bytes=batch_bytes)
                    )
                    job_id += 1
                    batch_files, batch_bytes = [], 0
                batch_files.append(rec.path)
                batch_bytes += rec.size_bytes
            if batch_files:
                jobs.append(
                    BatchJob(sample=sample, job_id=job_id, files=batch_files, total_bytes=batch_bytes)
                )
        return jobs

    def write_manifest(
        self,
        work_dir: Path | str,
        records: Sequence[FileRecord],
        jobs: Sequence[BatchJob],
    ) -> Path:
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "created": datetime.now().isoformat(),
            "max_job_bytes": self.cfg.max_job_bytes,
            "file_survey": [asdict(r) for r in records],
            "jobs": [
                {
                    "sample": j.sample,
                    "job_id": j.job_id,
                    "tag": j.tag,
                    "files": j.files,
                    "total_bytes": j.total_bytes,
                    "total_gb": j.total_bytes / (1024.0 ** 3),
                }
                for j in jobs
            ],
        }
        out = work_dir / "manifest.json"
        with open(out, "w") as f:
            json.dump(manifest, f, indent=2)
        return out

    def print_survey_summary(self, records: Sequence[FileRecord], jobs: Sequence[BatchJob]) -> None:
        by_sample: Dict[str, List[FileRecord]] = {}
        for r in records:
            by_sample.setdefault(r.sample, []).append(r)
        for sample, recs in sorted(by_sample.items()):
            tot = sum(r.size_bytes for r in recs)
            print(
                f"[event_selection] sample={sample}  files={len(recs)}  total={tot / (1024**3):.2f} GiB",
                flush=True,
            )
        print(f"[event_selection] {len(jobs)} job(s) under size budget", flush=True)
        for j in jobs[:12]:
            print(
                f"  {j.sample} {j.tag}: {len(j.files)} file(s), {j.total_bytes / (1024**3):.3f} GiB",
                flush=True,
            )
        if len(jobs) > 12:
            print(f"  ... and {len(jobs) - 12} more", flush=True)

    def discover_jobs(self) -> Tuple[List[FileRecord], List[BatchJob], Path]:
        work, _, _ = self.resolve_paths()
        records = self.survey_files()
        if not records:
            raise RuntimeError("No input .df files matched -- check hooks.iter_df_paths' glob(s)")
        jobs = self.group_files_into_jobs(records)
        manifest_path = self.write_manifest(work, records, jobs)
        self.print_survey_summary(records, jobs)
        return records, jobs, manifest_path

    # ------------------------------------------------------------------
    # Map phase
    # ------------------------------------------------------------------
    @staticmethod
    def _batch_out_path(batches_dir: Path, sample: str, job: BatchJob) -> Path:
        return batches_dir / f"{sample}__{job.tag}.pkl"

    def process_one_job(
        self,
        job: BatchJob,
        out_dir: Path | str,
        *,
        skip_existing: bool = True,
    ) -> Optional[str]:
        """Run one batch job in-process via ``hooks.run_batch_selection``."""
        cfg = self.cfg
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_pkl = self._batch_out_path(out_dir, job.sample, job)
        if skip_existing and out_pkl.is_file():
            print(f"[event_selection] {out_pkl} exists; skipping", flush=True)
            return str(out_pkl)

        print(
            f"[event_selection] sample={job.sample} {job.tag}  files={len(job.files)}  "
            f"size={job.total_bytes / (1024**3):.3f} GiB",
            flush=True,
        )
        mc_univ = tuple(cfg.mc_univ_syst) if job.sample == "mc" else ()
        pipeline_trace = (lambda msg: print(msg, flush=True)) if cfg.trace else None

        meta = self.hooks.run_batch_selection(
            job.sample,
            job.files,
            str(out_pkl),
            job_id=job.tag,
            use_mc_genweight=cfg.use_mc_genweight,
            mc_univ_syst_tags=mc_univ,
            pipeline_trace=pipeline_trace,
        )
        print(
            f"[event_selection] done sample={job.sample} {job.tag}  n_evt={meta['n_evt']}  "
            f"pot={meta['chunk_pot']:.3e}",
            flush=True,
        )
        return str(out_pkl)

    def _process_one_job_safe(
        self,
        job: BatchJob,
        out_dir: Path | str,
        skip_existing: bool,
    ) -> Tuple[BatchJob, Optional[str], Optional[str]]:
        """``process_one_job`` wrapped to return (job, out_path, error) instead of
        raising -- so one bad job doesn't abort the rest of the pool/loop, whether
        running sequentially or via ``multiprocessing.Pool.map``.
        """
        try:
            out = self.process_one_job(job, out_dir, skip_existing=skip_existing)
            return job, out, None
        except Exception as exc:  # noqa: BLE001 -- deliberately broad, see docstring
            return job, None, f"{type(exc).__name__}: {exc}"

    def run_map(
        self,
        jobs: Sequence[BatchJob] | None = None,
    ) -> Tuple[Path, Dict[str, List[str]], List[Tuple[str, str, str]], Path]:
        cfg = self.cfg
        work, batches_dir, _ = self.resolve_paths()
        batches_dir.mkdir(parents=True, exist_ok=True)
        if jobs is None:
            _, jobs, manifest_path = self.discover_jobs()
        else:
            records = self.survey_files()
            manifest_path = self.write_manifest(work, records, jobs)

        batch_paths: Dict[str, List[str]] = {s: [] for s in cfg.samples}
        failed: List[Tuple[str, str, str]] = []

        print(f"[event_selection] WORK_BASE={work}", flush=True)
        print(f"[event_selection] BATCHES_DIR={batches_dir}", flush=True)

        worker = partial(
            self._process_one_job_safe,
            out_dir=batches_dir,
            skip_existing=cfg.skip_existing_batches,
        )

        if cfg.n_workers <= 1 or len(jobs) <= 1:
            results = [worker(job) for job in jobs]
        else:
            n_workers = min(cfg.n_workers, len(jobs))
            print(f"[event_selection] running {len(jobs)} job(s) across {n_workers} worker process(es)", flush=True)
            with mp.Pool(processes=n_workers, maxtasksperchild=1) as pool:
                results = pool.map(worker, jobs)

        for job, out, err in results:
            tag = f"{job.sample}/{job.tag}"
            if err is not None:
                print(f"[event_selection] FAILED {tag} ({err})", flush=True)
                failed.append((job.sample, tag, err))
            elif out:
                batch_paths.setdefault(job.sample, []).append(out)

        for sample, paths in batch_paths.items():
            print(f"[event_selection] sample={sample} wrote {len(paths)} batch pickle(s)", flush=True)
        return batches_dir, batch_paths, failed, manifest_path

    # ------------------------------------------------------------------
    # Aggregate/render phase
    # ------------------------------------------------------------------
    def run_aggregate(self, batches_dir: Path | str | None = None) -> EventSelectionPipelineResult:
        """Reduce phase: aggregate batch pickles and render plots via
        ``hooks.aggregate_and_render``.
        """
        cfg = self.cfg
        work, default_batches, plots_dir = self.resolve_paths()
        batches_dir = Path(batches_dir or default_batches)
        plots_dir.mkdir(parents=True, exist_ok=True)

        syst_root = self._resolve_syst_disk_root()
        syst_disk_arg = str(syst_root) if syst_root is not None and syst_root.is_dir() else None

        merged_payload = self.hooks.aggregate_and_render(
            str(batches_dir),
            str(plots_dir),
            cosmic_estimate=cfg.cosmic_estimate,
            f_offbeam_frac=cfg.f_offbeam_frac,
            syst_disk_root=syst_disk_arg,
            save_fig=cfg.save_fig,
            show_fig=cfg.show_fig,
        )

        batch_paths = {s: sorted(str(p) for p in batches_dir.glob(f"{s}__*.pkl")) for s in cfg.samples}
        manifest_path = work / "manifest.json"
        return EventSelectionPipelineResult(
            work_base=work,
            batches_dir=batches_dir,
            plots_dir=plots_dir,
            batch_paths=batch_paths,
            failed=[],
            merged_payload=merged_payload,
            pot_str=merged_payload["pot_str"],
            data_pot=merged_payload["data_pot"],
            manifest_path=manifest_path,
        )

    def run_full(self) -> EventSelectionPipelineResult:
        cfg = self.cfg
        work, batches_dir, plots_dir = self.resolve_paths()
        manifest_path = work / "manifest.json"

        failed: List[Tuple[str, str, str]] = []
        if not cfg.aggregate_only:
            _, _, failed, manifest_path = self.run_map()
            if failed:
                print(f"[event_selection] WARN: {len(failed)} batch job(s) failed", flush=True)
        else:
            print("[event_selection] aggregate_only=True — skipping map phase", flush=True)

        if cfg.skip_aggregate:
            print("[event_selection] skip_aggregate=True — map pickles only", flush=True)
            return EventSelectionPipelineResult(
                work_base=work,
                batches_dir=batches_dir,
                plots_dir=plots_dir,
                batch_paths={},
                failed=failed,
                merged_payload=None,
                pot_str="",
                data_pot=0.0,
                manifest_path=manifest_path,
            )

        result = self.run_aggregate(batches_dir)
        result.failed = failed
        print(f"[event_selection] DONE plots -> {result.plots_dir}", flush=True)
        return result

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    @staticmethod
    def show_saved_plots(plots_dir: Path | str, max_images: int = 20) -> None:
        from IPython.display import Image, display

        plots_dir = Path(plots_dir)
        paths = sorted(plots_dir.glob("*.png"))[:max_images]
        if not paths:
            print(f"[event_selection] no plots matched under {plots_dir}", flush=True)
            return
        print(f"[event_selection] displaying {len(paths)} plot(s) from {plots_dir}", flush=True)
        for p in paths:
            display(Image(filename=str(p)))
