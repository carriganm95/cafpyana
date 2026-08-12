"""pyanalib.logging_utils: shared CAFPYANA_LOG_LEVEL-controlled logging, and its
wiring into pyanalib.chunked_selection.ChunkRunner's pipeline_trace hook.
"""
import logging

import numpy as np
import pandas as pd
import pytest

from pyanalib.logging_utils import ENV_VAR, DEFAULT_LEVEL, get_logger
from pyanalib.chunked_selection import ChunkRunner, PlotSpec, Stage


def test_default_level_is_warning(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    log = get_logger("test_default")
    assert logging.getLevelName(log.getEffectiveLevel()) == DEFAULT_LEVEL


def test_env_var_controls_level(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "DEBUG")
    log = get_logger("test_debug")
    assert log.getEffectiveLevel() == logging.DEBUG

    monkeypatch.setenv(ENV_VAR, "ERROR")
    log2 = get_logger("test_error")
    assert log2.getEffectiveLevel() == logging.ERROR


def test_invalid_level_raises(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "NOT_A_LEVEL")
    with pytest.raises(ValueError):
        get_logger("test_invalid")


def test_loggers_share_the_cafpyana_root():
    a = get_logger("module_a")
    b = get_logger("module_b")
    assert a.name == "cafpyana.module_a"
    assert b.name == "cafpyana.module_b"
    assert a.parent is b.parent  # both children of the shared "cafpyana" logger


def test_chunk_runner_pipeline_trace_falls_back_to_logger(monkeypatch, caplog):
    # Regression guard: ChunkRunner.run's pipeline_trace hook used to be a pure
    # no-op when the caller didn't pass a callback -- meaning debugging a stalled
    # pipeline required threading a print callback through by hand. It should
    # now default to this module's logger at DEBUG level.
    monkeypatch.setenv(ENV_VAR, "DEBUG")

    class _FakeVarConfig:
        var_save_name = "x"
        var_evt_reco_col = "x"
        bins = np.array([0.0, 1.0, 2.0])

    plot = PlotSpec(
        var_config=_FakeVarConfig(),
        breakdown_type="fake",
        selector=lambda state, sample: state.get("evt"),
    )
    stage = Stage(key="s0", label="Stage 0", plots=[plot])
    runner = ChunkRunner(
        sample="mc",
        stages=[stage],
        efficiency_vars=[],
        breakdown_registry={"fake": (1, lambda df, ret_cuts=True: [df.index >= 0])},
        signal_mask_fn=lambda df: df.index >= 0,
    )
    evt = pd.DataFrame({"x": [0.5], "pot_weight": [1.0]})
    with caplog.at_level(logging.DEBUG, logger="cafpyana.pyanalib.chunked_selection"):
        runner.run({"evt": evt})
    assert any("stage='s0'" in rec.message or "s0" in rec.message for rec in caplog.records)
