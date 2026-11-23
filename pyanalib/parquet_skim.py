"""Parquet storage for skimmed cafpyana event/track dataframes.

A skim is a pair of Parquet files per sample (``<sample>_evt.parquet`` and
``<sample>_trk.parquet``) holding the slices (and their tracks) that passed a loose
preselection. They are written once from the full ``.df`` inputs and then read back one
*chunk* at a time, so memory stays bounded no matter how large the skim is.

Format
------
* **Chunks.** Every row carries an ``__skim_chunk`` id (one id per input ``.df`` file).
  The writer puts each chunk in its own row group(s); the reader groups row groups by the
  min/max statistics of ``__skim_chunk``, so ``evt`` and ``trk`` rows of the same chunk are
  read together without having to line up row-group numbers between the two files.
* **Columns.** cafpyana frames use tuple (MultiIndex) column names, which Parquet can't
  store. Columns are written under flat names (``"|".join(tuple)``) and the original
  tuples, plus the index level names, are kept as JSON in the schema metadata
  (``cafpyana.skim``), so :func:`read_chunk` returns exactly the frame that was written.
* **Index.** MultiIndex levels are written as ordinary columns (``__index__<i>``) and
  restored on read.
* **object-bool columns.** Some columns (e.g. ``rec.dlp.*_mask_reco``) are ``object``
  dtype holding bools mixed with NaN. They are written as nullable bool and restored to
  ``object`` with ``np.nan`` for nulls, so selection code sees the same values.
* **Fixed schema.** The schema comes from the first chunk written; later chunks are
  conformed to it (missing columns -> nulls, dtypes cast), so an all-NaN chunk can't change
  a column's type part-way through the file.
"""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pyanalib.logging_utils import get_logger

logger = get_logger(__name__)

CHUNK_COL = "__skim_chunk"
META_KEY = b"cafpyana.skim"
_COLSEP = "|"
_INDEX_PREFIX = "__index__"
FORMAT_VERSION = 1


# ===========================================================================
# Column-name flattening
# ===========================================================================
def _flat_name(col: Any) -> str:
    if isinstance(col, tuple):
        return _COLSEP.join(str(p) for p in col)
    return str(col)


def _strip_trailing_empty(col: Tuple) -> Tuple:
    parts = list(col)
    while parts and parts[-1] == "":
        parts.pop()
    return tuple(parts)


def _is_object_bool(s: pd.Series) -> bool:
    """True for an ``object`` column whose non-null values are all bools."""
    if s.dtype != object:
        return False
    vals = s.to_numpy(dtype=object)
    na = pd.isna(vals)
    return all(isinstance(v, (bool, np.bool_)) for v in vals[~na])


def _object_bool_to_arrow(s: pd.Series) -> pa.Array:
    vals = s.to_numpy(dtype=object)
    na = pd.isna(vals)
    out = np.zeros(len(vals), dtype=bool)
    if (~na).any():
        out[~na] = vals[~na].astype(bool)
    return pa.array(out, type=pa.bool_(), mask=na)


# ===========================================================================
# DataFrame <-> Arrow
# ===========================================================================
def _frame_meta(df: pd.DataFrame) -> Dict[str, Any]:
    cols = list(df.columns)
    is_multi = isinstance(df.columns, pd.MultiIndex)
    object_bool = [_flat_name(c) for c in cols if _is_object_bool(df[c])]
    return {
        "version": FORMAT_VERSION,
        "columns": [list(c) if is_multi else c for c in cols],
        "flat_columns": [_flat_name(c) for c in cols],
        "multiindex_columns": is_multi,
        "column_level_names": list(df.columns.names) if is_multi else None,
        "index_names": list(df.index.names),
        "object_bool": object_bool,
    }


def frame_to_table(
    df: pd.DataFrame,
    chunk_id: int,
    meta: Optional[Dict[str, Any]] = None,
) -> Tuple[pa.Table, Dict[str, Any]]:
    """Convert ``df`` to an Arrow table with flat column names, index columns and
    ``__skim_chunk``. Returns ``(table, meta)``; pass ``meta`` back in for later chunks so
    the same object-bool columns are encoded the same way."""
    if meta is None:
        meta = _frame_meta(df)
    obj_bool = set(meta["object_bool"])

    arrays: List[pa.Array] = []
    names: List[str] = []
    for i, name in enumerate(meta["index_names"]):
        arrays.append(pa.array(np.asarray(df.index.get_level_values(i))))
        names.append(f"{_INDEX_PREFIX}{i}")
    arrays.append(pa.array(np.full(len(df), chunk_id, dtype=np.int64)))
    names.append(CHUNK_COL)

    for col in df.columns:
        flat = _flat_name(col)
        s = df[col]
        if flat in obj_bool:
            arr = _object_bool_to_arrow(s)
        else:
            arr = pa.array(s.to_numpy(), from_pandas=True)
        arrays.append(arr)
        names.append(flat)
    return pa.Table.from_arrays(arrays, names=names), meta


def _conform(table: pa.Table, schema: pa.Schema) -> pa.Table:
    """Reorder/cast ``table`` to ``schema``; missing columns become nulls."""
    have = set(table.column_names)
    extra = have - set(schema.names)
    if extra:
        logger.warning("skim chunk has %d column(s) not in the file schema; dropped: %s",
                       len(extra), sorted(extra)[:10])
    cols = []
    for field in schema:
        if field.name in have:
            col = table.column(field.name)
            if col.type != field.type:
                col = col.cast(field.type)
            cols.append(col)
        else:
            logger.warning("skim chunk is missing column %r; filled with nulls", field.name)
            cols.append(pa.nulls(table.num_rows, type=field.type))
    return pa.Table.from_arrays(cols, schema=schema)


# ===========================================================================
# Writing
# ===========================================================================
class SkimWriter:
    """Append chunks of one frame type (evt or trk) to a Parquet file.

    The file is created on the first :meth:`write`. A chunk with zero rows is recorded
    in nothing but the caller's bookkeeping -- readers return an empty frame for it.
    """

    def __init__(self, path: str, compression: str = "zstd", compression_level: int = 3):
        self.path = path
        self.compression = compression
        self.compression_level = compression_level
        self._writer: Optional[pq.ParquetWriter] = None
        self._meta: Optional[Dict[str, Any]] = None
        self._schema: Optional[pa.Schema] = None
        self.n_rows = 0

    def write(self, df: pd.DataFrame, chunk_id: int) -> None:
        table, meta = frame_to_table(df, chunk_id, self._meta)
        if self._writer is None:
            self._meta = meta
            schema = table.schema.with_metadata({META_KEY: json.dumps(meta).encode()})
            self._schema = schema
            self._writer = pq.ParquetWriter(
                self.path, schema,
                compression=self.compression,
                compression_level=self.compression_level,
            )
        table = _conform(table, self._schema)
        if table.num_rows:
            # One row group per chunk (pyarrow would split very large ones further,
            # which the reader handles).
            self._writer.write_table(table, row_group_size=max(table.num_rows, 1))
            self.n_rows += table.num_rows

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def merge_skim_files(inputs: Sequence[str], out_path: str,
                     compression: str = "zstd", compression_level: int = 3) -> int:
    """Concatenate skim part files (same frame type) into one file, row group by row
    group, without going through pandas. Chunk ids must already be unique across parts.
    Returns the number of rows written."""
    writer = None
    schema = None
    n = 0
    seen_chunks: set = set()
    try:
        for path in inputs:
            pf = pq.ParquetFile(path)
            if schema is None:
                schema = pf.schema_arrow
                writer = pq.ParquetWriter(out_path, schema, compression=compression,
                                          compression_level=compression_level)
            part_chunks = set(_row_group_chunks(pf).keys())
            dup = part_chunks & seen_chunks
            if dup:
                raise ValueError(f"{path}: chunk ids {sorted(dup)[:5]} already in another part")
            seen_chunks |= part_chunks
            for i in range(pf.num_row_groups):
                t = _conform(pf.read_row_group(i), schema)
                writer.write_table(t, row_group_size=max(t.num_rows, 1))
                n += t.num_rows
    finally:
        if writer is not None:
            writer.close()
    return n


# ===========================================================================
# Reading
# ===========================================================================
def read_meta(pf: pq.ParquetFile) -> Dict[str, Any]:
    raw = (pf.schema_arrow.metadata or {}).get(META_KEY)
    if raw is None:
        raise ValueError("not a cafpyana skim file (no cafpyana.skim metadata)")
    return json.loads(raw)


def _row_group_chunks(pf: pq.ParquetFile) -> Dict[int, List[int]]:
    """``{chunk_id: [row_group, ...]}`` from the ``__skim_chunk`` column statistics."""
    idx = pf.schema_arrow.get_field_index(CHUNK_COL)
    out: Dict[int, List[int]] = defaultdict(list)
    for rg in range(pf.num_row_groups):
        st = pf.metadata.row_group(rg).column(idx).statistics
        if st is None or not st.has_min_max or st.min != st.max:
            raise ValueError(f"row group {rg} does not hold a single {CHUNK_COL}")
        out[int(st.min)].append(rg)
    return dict(out)


def _resolve_columns(meta: Dict[str, Any], columns: Optional[Iterable[Any]]) -> Optional[List[str]]:
    """Map requested column keys (tuples, with or without trailing ''; or a tuple prefix
    selecting every column under it) to flat names."""
    if columns is None:
        return None
    tuples = [tuple(c) if isinstance(c, list) else c for c in meta["columns"]]
    flat = meta["flat_columns"]
    out: List[str] = []
    for want in columns:
        if not isinstance(want, tuple):
            want = (want,)
        w = _strip_trailing_empty(want)
        hits = [f for t, f in zip(tuples, flat)
                if (isinstance(t, tuple) and _strip_trailing_empty(t)[:len(w)] == w) or t == want]
        if not hits:
            raise KeyError(want)
        out.extend(h for h in hits if h not in out)
    return out


def table_to_frame(table: pa.Table, meta: Dict[str, Any]) -> pd.DataFrame:
    flat_to_col = {
        f: (tuple(c) if isinstance(c, list) else c)
        for f, c in zip(meta["flat_columns"], meta["columns"])
    }
    obj_bool = set(meta["object_bool"])
    n_idx = len(meta["index_names"])
    index_arrays = [table.column(f"{_INDEX_PREFIX}{i}").to_numpy() for i in range(n_idx)]

    data_names = [n for n in table.column_names if n in flat_to_col]
    data: Dict[str, Any] = {}
    for name in data_names:
        col = table.column(name)
        if name in obj_bool:
            vals = np.asarray(col.to_pylist(), dtype=object)
            vals[pd.isna(vals)] = np.nan
            data[name] = vals
        else:
            data[name] = col.to_pandas()
    if n_idx == 1:
        index = pd.Index(index_arrays[0], name=meta["index_names"][0])
    else:
        index = pd.MultiIndex.from_arrays(index_arrays, names=meta["index_names"])
    df = pd.DataFrame({k: np.asarray(v) if not isinstance(v, pd.Series) else v.to_numpy()
                       for k, v in data.items()}, index=index)
    if meta["multiindex_columns"]:
        df.columns = pd.MultiIndex.from_tuples(
            [flat_to_col[n] for n in data_names], names=meta["column_level_names"])
    else:
        df.columns = [flat_to_col[n] for n in data_names]
    return df


class SkimFile:
    """Read access to one skim Parquet file (evt or trk)."""

    def __init__(self, path: str):
        self.path = path
        self.pf = pq.ParquetFile(path)
        self.meta = read_meta(self.pf)
        self.chunks = _row_group_chunks(self.pf)

    @property
    def chunk_ids(self) -> List[int]:
        return sorted(self.chunks)

    @property
    def n_rows(self) -> int:
        return self.pf.metadata.num_rows

    def _read_columns(self, columns) -> Optional[List[str]]:
        cols = _resolve_columns(self.meta, columns)
        if cols is None:
            return None
        n_idx = len(self.meta["index_names"])
        return [f"{_INDEX_PREFIX}{i}" for i in range(n_idx)] + cols

    def empty_frame(self, columns=None) -> pd.DataFrame:
        cols = self._read_columns(columns)
        schema = self.pf.schema_arrow
        names = cols if cols is not None else schema.names
        empty = pa.Table.from_arrays(
            [pa.array([], type=schema.field(n).type) for n in names], names=names)
        return table_to_frame(empty, self.meta)

    def read_chunk(self, chunk_id: int, columns=None) -> pd.DataFrame:
        rgs = self.chunks.get(chunk_id)
        if not rgs:
            return self.empty_frame(columns)
        table = self.pf.read_row_groups(rgs, columns=self._read_columns(columns))
        return table_to_frame(table, self.meta)

    def read(self, columns=None, chunk_ids: Optional[Sequence[int]] = None) -> pd.DataFrame:
        """Read several chunks (default: all) into one frame. Mind the memory."""
        ids = self.chunk_ids if chunk_ids is None else list(chunk_ids)
        rgs = [rg for c in ids for rg in self.chunks.get(c, [])]
        if not rgs:
            return self.empty_frame(columns)
        return table_to_frame(
            self.pf.read_row_groups(rgs, columns=self._read_columns(columns)), self.meta)


def iter_skim_chunks(
    evt_path: str,
    trk_path: Optional[str] = None,
    *,
    evt_columns=None,
    trk_columns=None,
    chunk_ids: Optional[Sequence[int]] = None,
) -> Iterator[Tuple[int, pd.DataFrame, Optional[pd.DataFrame]]]:
    """Yield ``(chunk_id, evt_df, trk_df)`` one chunk at a time.

    ``*_columns``: optional list of column keys (tuples, trailing '' optional; a shorter
    tuple selects every column under that prefix). ``None`` reads every column.
    ``chunk_ids``: restrict to these chunks (default: every chunk in the evt file).
    """
    evt = SkimFile(evt_path)
    trk = SkimFile(trk_path) if trk_path else None
    ids = evt.chunk_ids if chunk_ids is None else list(chunk_ids)
    for cid in ids:
        e = evt.read_chunk(cid, evt_columns)
        t = trk.read_chunk(cid, trk_columns) if trk is not None else None
        yield cid, e, t
