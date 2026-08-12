"""Generic dataset-path resolution: glob listing, keyed/grouped/tagged glob
iteration, and default work-root path computation.

No hardcoded analysis-specific globs, env-var names, or paths live here --
analysis packages (e.g. ``analysis_village.nueNp0Pi.dataset_paths``) supply
their own glob dicts / env-var names / work-base path segments and call these
generic helpers. This mirrors the ``pyanalib.event_selection_pipeline`` split:
this module is the reusable machinery, the analysis package is the wiring.
"""

from __future__ import annotations

import glob
import os
import subprocess as _sp
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


def sorted_glob(pattern: str) -> List[str]:
    """Sorted glob that shells out to handle PNFS/dCache paths.

    Python's glob.glob uses os.listdir() which fails silently on many PNFS/dCache
    mounts even when 'ls' works fine. Shell out to get the real listing.
    """
    result = _sp.run(
        f"ls -1 -- {pattern} 2>/dev/null",
        shell=True, capture_output=True, text=True,
    )
    candidates = sorted(p for p in result.stdout.splitlines() if p)
    if not candidates:
        candidates = sorted(glob.glob(pattern))
    out: List[str] = []
    for p in candidates:
        try:
            resolved = os.path.realpath(p)
        except OSError:
            continue
        if os.path.isdir(resolved):
            continue
        out.append(p)
    return out


def iter_keyed_df_paths(key: str, globs: Dict[str, str]) -> Iterator[str]:
    """Yield sorted ``.df`` paths for ``globs[key]``."""
    if key not in globs:
        raise KeyError("unknown key %r; choose from %s" % (key, tuple(globs)))
    yield from sorted_glob(globs[key])


def iter_tagged_chunk_jobs(tagged_globs: Sequence[Tuple[str, str]]) -> Iterator[Tuple[str, str]]:
    """Yield ``(tag, df_path)`` for each ``(tag, glob_pattern)`` pair, paths sorted per glob."""
    for tag, pattern in tagged_globs:
        for p in sorted_glob(pattern):
            yield tag, p


def iter_ordered_group_tasks(
    group_globs: Dict[str, str],
    group_order: Sequence[str] = (),
) -> Iterator[Tuple[str, str]]:
    """Yield ``(group_tag, df_path)`` -- ``group_order`` entries first (if present in
    ``group_globs``), then any remaining keys in sorted order (so ad-hoc entries not
    listed in ``group_order`` are still included).
    """
    seen: set = set()
    for tag in group_order:
        if tag not in group_globs:
            continue
        seen.add(tag)
        for p in sorted_glob(group_globs[tag]):
            yield tag, p
    for tag in sorted(k for k in group_globs if k not in seen):
        for p in sorted_glob(group_globs[tag]):
            yield tag, p


def iter_unique_df_paths(tasks: Iterable[Tuple[str, str]]) -> Iterator[str]:
    """Yield unique ``.df`` paths from a ``(tag, path)`` task stream, first-seen order."""
    seen: set = set()
    for _, p in tasks:
        if p not in seen:
            seen.add(p)
            yield p


def default_work_root(
    env_var: str,
    work_base: str,
    subdir_stem: str,
    tag: Optional[str] = None,
    *,
    chunked: bool = True,
) -> Path:
    """Default ``/exp/sbnd/data/users/<user>/<work_base>/<subdir_stem>[-chunked]-<tag>``,
    overridable by setting ``env_var``. ``tag`` defaults to today (``%Y%m%d``).
    """
    t = tag or datetime.now().strftime("%Y%m%d")
    base = os.environ.get(env_var)
    if base:
        return Path(base)
    user = os.environ.get("USER", "user")
    suffix = f"-chunked-{t}" if chunked else f"-{t}"
    return Path(f"/exp/sbnd/data/users/{user}/{work_base}/{subdir_stem}{suffix}")


def default_syst_disk_root(env_var: str, work_base: str, dirname: str = "syst_disk") -> Path:
    """Default unified ``syst_disk_layout`` root, overridable by setting ``env_var``."""
    env = os.environ.get(env_var)
    if env:
        return Path(env).expanduser()
    return Path(f"/exp/sbnd/data/users/{os.environ.get('USER', 'user')}/{work_base}/{dirname}")


def sibling_dir(base: Path, sibling_name: str) -> Path:
    """``<base.parent>/<sibling_name>`` -- e.g. a ``_CC`` (joint/cross-variable) variant
    tree living next to ``base``.
    """
    return base.parent / sibling_name


def print_summary(lines: Iterable[str]) -> None:
    for line in lines:
        print(line)
