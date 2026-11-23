import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyarrow")

from pyanalib.parquet_skim import (
    SkimFile,
    SkimWriter,
    iter_skim_chunks,
    merge_skim_files,
)


def _evt(n, ntuple, seed):
    rng = np.random.default_rng(seed)
    cols = pd.MultiIndex.from_tuples([
        ("rec", "dlp", "energy", "", ""),
        ("rec", "dlp", "nhit", "", ""),
        ("rec", "dlp", "is_good", "", ""),
        ("rec", "dlp", "mask_reco", "", ""),        # object: bool mixed with NaN
        ("rec", "dlp_true", "pdg", "", ""),
        ("pot_weight", "", "", "", ""),
    ])
    idx = pd.MultiIndex.from_arrays(
        [np.full(n, ntuple), np.arange(n) // 2, np.arange(n) % 2],
        names=["__ntuple", "entry", "rec.dlp..index"],
    )
    mask = np.array([True, False, np.nan] * n, dtype=object)[:n]
    df = pd.DataFrame(index=idx, columns=cols)
    df[cols[0]] = rng.normal(size=n).astype(np.float32)
    df[cols[1]] = rng.integers(0, 100, n).astype(np.int32)
    df[cols[2]] = rng.random(n) > 0.5
    df[cols[3]] = mask
    df[cols[4]] = np.where(rng.random(n) > 0.3, 11.0, np.nan)
    df[cols[5]] = 1.0
    return df


def _trk(evt):
    rows = []
    for (nt, e, s) in evt.index:
        for p in range(2):
            rows.append((nt, e, s, p))
    idx = pd.MultiIndex.from_tuples(
        rows, names=["__ntuple", "entry", "rec.dlp..index", "rec.dlp.particles..index"])
    cols = pd.MultiIndex.from_tuples([("rec", "dlp", "particles", "ke", "", ""),
                                      ("pot_weight", "", "", "", "", "")])
    return pd.DataFrame({cols[0]: np.arange(len(idx), dtype=float), cols[1]: 1.0},
                        index=idx)[cols]


def test_round_trip_multiple_chunks(tmp_path):
    chunks = {3: _evt(10, 0, 1), 4: _evt(0, 0, 2), 5: _evt(7, 1, 3)}
    p_evt, p_trk = tmp_path / "e.parquet", tmp_path / "t.parquet"
    with SkimWriter(str(p_evt)) as we, SkimWriter(str(p_trk)) as wt:
        for cid, df in chunks.items():
            we.write(df, cid)
            wt.write(_trk(df), cid)

    sf = SkimFile(str(p_evt))
    assert sf.chunk_ids == [3, 5]          # empty chunk writes no row group
    got = {cid: (e, t) for cid, e, t in iter_skim_chunks(str(p_evt), str(p_trk))}
    for cid in (3, 5):
        pd.testing.assert_frame_equal(got[cid][0], chunks[cid])
        pd.testing.assert_frame_equal(got[cid][1], _trk(chunks[cid]))
    # object-bool column keeps NaN (not None) for missing values
    col = ("rec", "dlp", "mask_reco", "", "")
    assert got[3][0][col].dtype == object
    assert any(isinstance(v, float) and np.isnan(v) for v in got[3][0][col])
    # an empty chunk reads back as an empty frame with the right columns
    assert len(sf.read_chunk(4)) == 0
    assert list(sf.read_chunk(4).columns) == list(chunks[3].columns)


def test_column_selection(tmp_path):
    df = _evt(6, 0, 0)
    p = tmp_path / "e.parquet"
    with SkimWriter(str(p)) as w:
        w.write(df, 1)
    sf = SkimFile(str(p))
    one = sf.read_chunk(1, columns=[("rec", "dlp", "energy")])
    assert list(one.columns) == [("rec", "dlp", "energy", "", "")]
    prefix = sf.read_chunk(1, columns=[("rec", "dlp")])
    assert len(prefix.columns) == 4
    with pytest.raises(KeyError):
        sf.read_chunk(1, columns=[("nope",)])


def test_schema_fixed_by_first_chunk(tmp_path):
    a, b = _evt(5, 0, 0), _evt(5, 1, 1)
    b[("rec", "dlp_true", "pdg", "", "")] = np.nan       # all-NaN chunk
    b[("rec", "dlp", "mask_reco", "", "")] = np.nan       # object-bool all NaN
    p = tmp_path / "e.parquet"
    with SkimWriter(str(p)) as w:
        w.write(a, 0)
        w.write(b, 1)
    back = SkimFile(str(p)).read_chunk(1)
    assert back[("rec", "dlp_true", "pdg", "", "")].isna().all()
    assert back[("rec", "dlp", "mask_reco", "", "")].isna().all()


def test_merge(tmp_path):
    parts = []
    for i, cid in enumerate((10, 20)):
        p = tmp_path / f"part{i}.parquet"
        with SkimWriter(str(p)) as w:
            w.write(_evt(4, cid, cid), cid)
        parts.append(str(p))
    out = tmp_path / "merged.parquet"
    assert merge_skim_files(parts, str(out)) == 8
    sf = SkimFile(str(out))
    assert sf.chunk_ids == [10, 20]
    pd.testing.assert_frame_equal(sf.read_chunk(20), _evt(4, 20, 20))
    with pytest.raises(ValueError):
        merge_skim_files(parts + [parts[0]], str(tmp_path / "dup.parquet"))
