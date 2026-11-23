"""Convert dCache paths to xrootd URLs.

Grid jobs read their inputs with xrootd (``xrdcp``), so file lists must hold URLs, not
``/pnfs`` paths. The standard Fermilab dCache door form (what samweb returns for
``schema=root``) is::

    /pnfs/<rest>  ->  root://fndcadoor.fnal.gov:1094/pnfs/fnal.gov/usr/<rest>
"""
from __future__ import annotations

XROOTD_DOOR = "root://fndcadoor.fnal.gov:1094"
_PNFS_PREFIX = "/pnfs/"
_DCACHE_NAMESPACE = "/pnfs/fnal.gov/usr/"


def pnfs_to_xrootd(path: str) -> str:
    """Return the xrootd URL for a ``/pnfs/...`` path; URLs are returned unchanged.

    Raises ``ValueError`` for anything else (a local path can't be read on the grid).
    """
    p = path.strip()
    if p.startswith("root://"):
        return p
    if p.startswith("xroot://"):                    # same protocol, other spelling
        return "root://" + p[len("xroot://"):]
    if p.startswith(_DCACHE_NAMESPACE):              # already the full dCache namespace
        return XROOTD_DOOR + p
    if p.startswith(_PNFS_PREFIX):
        return XROOTD_DOOR + _DCACHE_NAMESPACE + p[len(_PNFS_PREFIX):]
    raise ValueError(f"not a /pnfs path or xrootd URL: {path!r}")


def is_xrootd_url(path: str) -> bool:
    return path.strip().startswith("root://")
