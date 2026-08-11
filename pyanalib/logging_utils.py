"""Shared logging setup, controlled by a single environment variable.

Instead of each module hand-rolling its own ``print()`` calls and ad-hoc
``verbose``/``verbose_hist`` boolean flags that don't talk to each other and
don't propagate into nested calls, modules should get a logger from
``get_logger(__name__)`` here and log through it. Verbosity for the WHOLE
codebase is then controlled by one environment variable::

    export CAFPYANA_LOG_LEVEL=DEBUG   # or INFO / WARNING / ERROR / CRITICAL

or, inside a notebook cell, before importing anything else from this repo::

    import os
    os.environ["CAFPYANA_LOG_LEVEL"] = "DEBUG"

Default level is ``WARNING`` (quiet) -- set ``INFO`` for high-level progress
(POT/exposure summaries, stage transitions, file counts) or ``DEBUG`` for
detailed tracing (e.g. ``ChunkRunner``'s per-stage pipeline trace).

Python's ``logging`` module already supports hierarchy (a logger named
``cafpyana.analysis_village.nueNp0Pi.utils`` is a child of ``cafpyana``), so if
one module needs to be louder than the rest, that's still possible via
``logging.getLogger("cafpyana.<dotted.module.name>").setLevel(...)`` -- no
need for a bespoke per-class/per-object verbosity system for that either.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

ENV_VAR = "CAFPYANA_LOG_LEVEL"
DEFAULT_LEVEL = "WARNING"

_ROOT_NAME = "cafpyana"


def _resolve_level() -> int:
    level_name = os.environ.get(ENV_VAR, DEFAULT_LEVEL).upper()
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        raise ValueError(
            "%s=%r is not a valid logging level; expected one of "
            "DEBUG, INFO, WARNING, ERROR, CRITICAL" % (ENV_VAR, level_name)
        )
    return level


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the shared ``cafpyana`` hierarchy for ``name`` (pass ``__name__``).

    Re-reads ``CAFPYANA_LOG_LEVEL`` every call (cheap -- this is normally called
    once per module at import time to bind a module-level ``logger``), so
    changing the env var and re-importing/re-calling picks up the new level
    without restarting the interpreter.
    """
    root = logging.getLogger(_ROOT_NAME)
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
        root.addHandler(handler)
        root.propagate = False
    root.setLevel(_resolve_level())
    return logging.getLogger("%s.%s" % (_ROOT_NAME, name))
