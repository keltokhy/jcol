"""Table IO and row serialization. Source columns stay intact in outputs."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import polars as pl


FORMATS = ("csv", "tsv", "parquet", "jsonl")


def table_format(path=None, format: str | None = None) -> str:
    format = format or (Path(path).suffix.lower().lstrip(".") if isinstance(path, (str, Path)) else None)
    if format == "ndjson":
        format = "jsonl"
    if format not in FORMATS:
        raise ValueError("table format must be csv, tsv, parquet or jsonl; specify a format for streams")
    return format


def read_table(path, limit: int | None = None, *, format: str | None = None) -> pl.DataFrame:
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    readers = {"parquet": pl.read_parquet, "csv": pl.read_csv,
               "tsv": lambda p: pl.read_csv(p, separator="\t"), "jsonl": pl.read_ndjson}
    df = readers[table_format(path, format)](path)
    return df.head(limit) if limit is not None else df


def select_inputs(df: pl.DataFrame, inputs: str | list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if inputs is None:
        strings = [c for c, t in df.schema.items() if t == pl.String]
        if not strings:
            raise ValueError("table has no text column to read")
        inputs = [max(strings, key=lambda c: df[c].str.len_chars().mean() or 0)]
    if isinstance(inputs, str):
        inputs = [inputs]
    if not inputs or len(set(inputs)) != len(inputs):
        raise ValueError("select at least one input column, without duplicates")
    for name in inputs:
        if name not in df.columns:
            raise ValueError(f"table has no column {name!r}; its columns are {df.columns}")
    return tuple(inputs)


def row_texts(df: pl.DataFrame, inputs: tuple[str, ...]) -> list[str]:
    if len(inputs) == 1:
        return df[inputs[0]].cast(pl.String).fill_null("").to_list()
    return [json.dumps(row, ensure_ascii=False, default=str) for row in df.select(inputs).iter_rows(named=True)]


def identity(df: pl.DataFrame, inputs: tuple[str, ...], **settings) -> dict:
    # IPC buffers can encode the same logical table differently after a Parquet
    # round trip (offsets, unused null bytes, dictionary layout). Hash values instead.
    digest = hashlib.sha256()
    digest.update(json.dumps([(name, str(dtype)) for name, dtype in df.schema.items()]).encode())
    physical = df.select([_physical_temporals(pl.col(name), dtype).alias(name) for name, dtype in df.schema.items()])
    for row in physical.iter_rows():
        digest.update(b"\n")
        digest.update(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":")).encode())
    return {"format": 2, "table_sha256": digest.hexdigest(),
            "rows": df.height, "inputs": list(inputs), **settings}


def _physical_temporals(expr: pl.Expr, dtype: pl.DataType) -> pl.Expr:
    # Python datetime conversion drops nanoseconds; preserve temporal integers,
    # including nested values, while retaining the original schema in the hash.
    if dtype.is_temporal():
        return expr.to_physical()
    if isinstance(dtype, pl.List):
        return expr.list.eval(_physical_temporals(pl.element(), dtype.inner))
    if isinstance(dtype, pl.Array):
        return expr.arr.to_list().list.eval(_physical_temporals(pl.element(), dtype.inner))
    if isinstance(dtype, pl.Struct):
        return pl.struct([_physical_temporals(expr.struct.field(f.name), f.dtype).alias(f.name) for f in dtype.fields])
    return expr


def append_columns(df: pl.DataFrame, columns) -> pl.DataFrame:
    series, names = [], set(df.columns)
    for column in columns:
        name = column.spec.name
        for candidate in (name, name + "__confidence"):
            if candidate in names:
                raise ValueError(f"output column {candidate!r} would overwrite an existing column")
            names.add(candidate)
        dtype = pl.String if column.spec.kind == "choice" else pl.Float64
        cells = [column.values.get(r, (None, None)) for r in range(df.height)]
        series.extend([pl.Series(name, [v for v, _ in cells], dtype=dtype),
                       pl.Series(name + "__confidence", [p for _, p in cells], dtype=pl.Float64)])
    return df.with_columns(series)


@contextmanager
def atomic_path(path: str | Path):
    """Replace a file only after writing succeeds; keep temporary data beside its destination."""
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        yield temporary
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_table(df: pl.DataFrame, path, *, format: str | None = None) -> None:
    format = table_format(path, format)

    def write(target):
        if format == "parquet":
            df.write_parquet(target)
        elif format == "jsonl":
            df.write_ndjson(target)
        else:
            df.write_csv(target, separator="\t" if format == "tsv" else ",")

    if isinstance(path, (str, Path)):
        with atomic_path(path) as temporary:
            write(temporary)
    else:
        write(path)
