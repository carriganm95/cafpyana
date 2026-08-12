"""pyanalib/event_selection_pipeline.py: the generic survey -> bin-pack ->
map -> aggregate/render orchestration extracted from
analysis_village/nueNp0Pi/event_selection_batched.py during the
config/pipeline restructure. These tests use fake hooks (no real .df files,
no real analysis-specific ChunkRunner) to prove the orchestration itself
doesn't depend on nueNp0Pi -- see analysis_village/nueNp0Pi/event_selection.py
for the real hook wiring, and tests/test_nueNp0Pi_event_selection.py for the
nueNp0Pi-specific pieces.
"""
from pathlib import Path

import pytest

from pyanalib.event_selection_pipeline import (
    BatchJob,
    EventSelectionHooks,
    EventSelectionPipeline,
    EventSelectionPipelineConfig,
    FileRecord,
)


def _hooks(**overrides):
    defaults = dict(
        iter_df_paths=lambda sample: iter([]),
        run_batch_selection=lambda *a, **k: {"n_evt": 0, "chunk_pot": 0.0},
        aggregate_and_render=lambda *a, **k: {"pot_str": "0", "data_pot": 0.0},
    )
    defaults.update(overrides)
    return EventSelectionHooks(**defaults)


def test_resolve_paths_requires_work_base_or_default_hook():
    cfg = EventSelectionPipelineConfig()  # work_base unset
    pipeline = EventSelectionPipeline(cfg, _hooks())
    with pytest.raises(ValueError):
        pipeline.resolve_paths()


def test_resolve_paths_uses_default_work_root_hook(tmp_path):
    cfg = EventSelectionPipelineConfig()
    hooks = _hooks(default_work_root=lambda tag: tmp_path / f"work-{tag}")
    pipeline = EventSelectionPipeline(cfg, hooks)
    work, batches, plots = pipeline.resolve_paths()
    assert str(work).startswith(str(tmp_path / "work-"))
    assert batches == work / "batches"
    assert plots == work / "plots"


def test_resolve_paths_explicit_work_base_wins(tmp_path):
    cfg = EventSelectionPipelineConfig(work_base=str(tmp_path / "explicit"))
    pipeline = EventSelectionPipeline(cfg, _hooks())
    work, batches, plots = pipeline.resolve_paths()
    assert work == tmp_path / "explicit"


def test_group_files_into_jobs_bins_by_size_budget():
    cfg = EventSelectionPipelineConfig(max_job_bytes=100)
    pipeline = EventSelectionPipeline(cfg, _hooks())
    # Named so alphabetical path-sort order (what group_files_into_jobs uses)
    # matches intended processing order: a, b, c, then the oversized file.
    records = [
        FileRecord(sample="mc", path="/a.df", size_bytes=40),
        FileRecord(sample="mc", path="/b.df", size_bytes=40),
        FileRecord(sample="mc", path="/c.df", size_bytes=40),  # doesn't fit with a+b
        FileRecord(sample="mc", path="/z_over.df", size_bytes=500),  # over budget alone
    ]
    jobs = pipeline.group_files_into_jobs(records)
    assert len(jobs) == 3
    assert [j.files for j in jobs] == [["/a.df", "/b.df"], ["/c.df"], ["/z_over.df"]]
    assert jobs[2].total_bytes == 500  # oversized single-file job kept whole, not split


def test_survey_files_respects_max_files_per_sample():
    cfg = EventSelectionPipelineConfig(samples=("mc",), max_files_per_sample=1)
    hooks = _hooks(iter_df_paths=lambda sample: iter(["/x.df", "/y.df"]))
    pipeline = EventSelectionPipeline(cfg, hooks)
    # os.path.getsize will fail for these fake paths -- survey_files skips
    # unreadable paths (OSError) rather than raising, so this should yield 0
    # records, not crash.
    records = pipeline.survey_files()
    assert records == []


def test_run_map_sequential_writes_batches_and_reports_failures(tmp_path):
    calls = []

    def run_batch_selection(sample, files, out_path, **kwargs):
        if sample == "mc" and files == ["/fail.df"]:
            raise RuntimeError("boom")
        calls.append((sample, tuple(files)))
        Path(out_path).write_text("ok")
        return {"n_evt": 1, "chunk_pot": 1.0}

    cfg = EventSelectionPipelineConfig(work_base=str(tmp_path), samples=("mc",))
    hooks = _hooks(run_batch_selection=run_batch_selection)
    pipeline = EventSelectionPipeline(cfg, hooks)

    jobs = [
        BatchJob(sample="mc", job_id=0, files=["/ok.df"], total_bytes=1),
        BatchJob(sample="mc", job_id=1, files=["/fail.df"], total_bytes=1),
    ]
    batches_dir, batch_paths, failed, manifest_path = pipeline.run_map(jobs)

    assert len(batch_paths["mc"]) == 1
    assert len(failed) == 1
    assert failed[0][0] == "mc"
    assert "boom" in failed[0][2]
    assert manifest_path.is_file()


def test_run_map_skips_existing_batch_by_default(tmp_path):
    cfg = EventSelectionPipelineConfig(work_base=str(tmp_path), samples=("mc",))
    calls = []

    def run_batch_selection(sample, files, out_path, **kwargs):
        calls.append(files)
        return {"n_evt": 1, "chunk_pot": 1.0}

    hooks = _hooks(run_batch_selection=run_batch_selection)
    pipeline = EventSelectionPipeline(cfg, hooks)
    job = BatchJob(sample="mc", job_id=0, files=["/ok.df"], total_bytes=1)

    # Pre-create the expected output pickle -- skip_existing_batches defaults True.
    out_dir = tmp_path / "batches"
    out_dir.mkdir()
    (out_dir / "mc__batch_0000.pkl").write_text("already there")

    out = pipeline.process_one_job(job, out_dir)
    assert out is not None
    assert calls == []  # never called -- skipped because file already existed


def test_run_aggregate_delegates_to_hooks(tmp_path):
    seen_kwargs = {}

    def aggregate_and_render(batches_dir, plots_dir, **kwargs):
        seen_kwargs.update(kwargs)
        return {"pot_str": "1.23e20", "data_pot": 1.23e20}

    cfg = EventSelectionPipelineConfig(
        work_base=str(tmp_path), samples=("mc",),
        cosmic_estimate="offbeam", f_offbeam_frac=0.5,
    )
    hooks = _hooks(aggregate_and_render=aggregate_and_render)
    pipeline = EventSelectionPipeline(cfg, hooks)

    result = pipeline.run_aggregate(tmp_path / "batches")
    assert result.pot_str == "1.23e20"
    assert result.data_pot == 1.23e20
    assert seen_kwargs["cosmic_estimate"] == "offbeam"
    assert seen_kwargs["f_offbeam_frac"] == 0.5


def test_run_full_skip_aggregate_returns_early(tmp_path):
    cfg = EventSelectionPipelineConfig(
        work_base=str(tmp_path), samples=("mc",),
        aggregate_only=True, skip_aggregate=True,
    )
    pipeline = EventSelectionPipeline(cfg, _hooks())
    result = pipeline.run_full()
    assert result.merged_payload is None
    assert result.pot_str == ""
