#!/usr/bin/env python3
"""Convert Tenrec CSV to Parquet in streaming mode."""

import argparse
import time
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Convert Tenrec CSV to Parquet")
    p.add_argument("--data_dir", type=str, default="data/tenrec/Tenrec")
    p.add_argument("--scenario", type=str, default="ctr_data_1M")
    p.add_argument("--chunk_size", type=int, default=1_000_000)
    p.add_argument("--compression", type=str, default="zstd", choices=["zstd", "snappy", "gzip", "brotli", "none"])
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    csv_path = data_dir / f"{args.scenario}.csv"
    parquet_path = data_dir / f"{args.scenario}.parquet"

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    if parquet_path.exists() and not args.overwrite:
        print(f"Parquet already exists: {parquet_path}")
        print("Use --overwrite to rebuild.")
        return

    compression = None if args.compression == "none" else args.compression

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required. Install with: python3 -m pip install pyarrow") from exc

    start = time.time()
    total_rows = 0
    writer = None

    if parquet_path.exists() and args.overwrite:
        parquet_path.unlink()

    reader = pd.read_csv(
        csv_path,
        na_values=["\\N", ""],
        engine="c",
        low_memory=False,
        chunksize=args.chunk_size,
    )

    for idx, chunk in enumerate(reader, start=1):
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(parquet_path, table.schema, compression=compression)
        writer.write_table(table)
        total_rows += len(chunk)

        if idx % 10 == 0:
            print(f"chunks={idx}, rows={total_rows}")

    if writer is not None:
        writer.close()

    elapsed = time.time() - start
    size_gb = parquet_path.stat().st_size / (1024 ** 3)
    print(f"Done. rows={total_rows}, parquet={parquet_path}, size={size_gb:.2f}GB, elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    main()
