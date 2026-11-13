"""Convert the 16B Pason 10-second CSV (inside 16B_Pason.zip) to one Parquet file.

Step 1 of 2: no filtering. Every channel is kept, renamed to snake_case,
-999.25 (and similar null sentinels) -> NaN, numeric columns as float64, text columns as strings.
Also writes a channel report so we can choose channels and build the
on-bottom / new-hole flags in step 2.

Reads straight from the zip, in chunks; nothing is extracted to disk.

Usage:
    python ingest/pason_to_parquet.py                 # full file
    python ingest/pason_to_parquet.py --max-rows 200000   # quick trial
"""
import argparse
import re
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

ZIP = config.RAW_16B / "16B_Pason.zip"
MEMBER_PATTERN = re.compile(r"10\s*second.*\.csv$", re.I)
NULLS = [-999.25, -999.0, -9999.0, -99999.0]
CHUNK = 200_000


def snake(name: str) -> str:
    s = name.strip().lower()
    s = s.replace("%", "pct").replace("#", "num").replace("°", "deg")
    s = re.sub(r"[^0-9a-z]+", "_", s).strip("_")
    return s or "unnamed"


def unique(names):
    seen, out = {}, []
    for n in names:
        k = n
        i = 1
        while k in seen:
            i += 1
            k = f"{n}_{i}"
        seen[k] = True
        out.append(k)
    return out


def find_member(zf: zipfile.ZipFile) -> str:
    names = [n for n in zf.namelist() if MEMBER_PATTERN.search(n)]
    if not names:
        sys.exit(f"No '10 Second Data.csv' in {ZIP}. Members:\n  " + "\n  ".join(zf.namelist()))
    return names[0]


DATE_RE = re.compile(r"^\d{4}[/-]\d{1,2}[/-]\d{1,2}|^\d{1,2}/\d{1,2}/\d{2,4}")
CLOCK_RE = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?")


class PlainFile:
    """Minimal stand-in for ZipFile so a plain CSV goes through the same code path."""
    def __init__(self, path):
        self.path = path

    def open(self, member):
        return open(self.path, "rb")


def sniff(zf, member):
    """Return (header, units row or None, rows to skip, sample DataFrame of data rows)."""
    with zf.open(member) as f:
        head = pd.read_csv(f, nrows=2000, dtype=str, encoding="latin-1")
    first = head.iloc[0]
    vals = first.dropna().astype(str).str.strip()
    vals = vals[vals != ""]
    numeric = pd.to_numeric(vals, errors="coerce").notna().mean() if len(vals) else 1.0
    if len(vals) >= 0.5 * len(first) and numeric < 0.3:   # second line is units, not data
        return list(head.columns), list(first.fillna("")), 1, head.iloc[1:]
    return list(head.columns), None, 0, head


def time_columns(raw_cols, sample):
    """Date/time columns, found from their values (header names vary)."""
    out = []
    for c in raw_cols:
        v = sample[c].dropna().astype(str).str.strip()
        if len(v) and (v.str.match(DATE_RE).mean() > 0.9 or v.str.match(CLOCK_RE).mean() > 0.9):
            out.append(c)
    return out


def text_columns(cols, raw_cols, sample, skip):
    """Columns that are mostly non-numeric text in the sample: kept as strings."""
    out = set()
    for c, r in zip(cols, raw_cols):
        if c in skip:
            continue
        v = sample[r].dropna().astype(str).str.strip()
        v = v[v != ""]
        if len(v) and pd.to_numeric(v, errors="coerce").isna().mean() > 0.05:
            out.add(c)
    return out


def build_timestamp(df, tcols_snake):
    if not tcols_snake:
        return None
    if len(tcols_snake) == 1:
        s = df[tcols_snake[0]]
    else:
        s = df[tcols_snake[0]].astype(str) + " " + df[tcols_snake[1]].astype(str)
    return pd.to_datetime(s, errors="coerce", format="mixed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--src", type=Path, default=ZIP, help="Pason zip (16B) or plain CSV (e.g. 16A)")
    ap.add_argument("--name", default="16b", help="output name: pason_<name>_10s.parquet")
    args = ap.parse_args()

    config.OUT.mkdir(parents=True, exist_ok=True)
    out_pq = config.OUT / f"pason_{args.name}_10s.parquet"
    out_rep = config.OUT / f"pason_{args.name}_channels.csv"

    if args.src.suffix.lower() == ".zip":
        zf = zipfile.ZipFile(args.src)
        member = find_member(zf)
        size = zf.getinfo(member).file_size
    else:
        zf, member, size = PlainFile(args.src), args.src.name, args.src.stat().st_size
    print(f"Reading {args.src.name} :: {member}  ({size/1e9:.2f} GB uncompressed)")

    raw_cols, units, skip, sample = sniff(zf, member)
    cols = unique([snake(c) for c in raw_cols])
    raw_t = time_columns(raw_cols, sample)
    tcols = [cols[raw_cols.index(c)] for c in raw_t]
    txt = text_columns(cols, raw_cols, sample, set(tcols))
    print(f"Columns: {len(cols)}   units row: {'yes' if units else 'no'}   time columns: {raw_t or 'NONE'}")
    print(f"Text columns kept as strings: {sorted(txt) or 'none'}")

    stats = {c: {"n": 0, "nonnull": 0, "text_lost": 0, "min": np.inf, "max": -np.inf} for c in cols}
    writer = None
    n_total, n_bad_ts = 0, 0
    t0 = time.time()

    with zf.open(member) as f:
        reader = pd.read_csv(
            f, header=0, names=cols, skiprows=range(1, 1 + skip),
            dtype=str, chunksize=CHUNK, encoding="latin-1",
            nrows=args.max_rows, low_memory=False,
        )
        for chunk in reader:
            out = {"row": chunk.index.to_numpy()}     # file order; 16A stamps are only to the minute
            ts = build_timestamp(chunk, tcols)
            if ts is not None:
                n_bad_ts += int(ts.isna().sum())
                out["timestamp"] = ts
            for c in cols:
                if c in tcols:
                    continue
                s = chunk[c].str.strip()
                st = stats[c]
                st["n"] += len(s)
                if c in txt:
                    s = s.replace("", None)
                    st["nonnull"] += int(s.notna().sum())
                    out[c] = s.astype("string")
                    continue
                v = pd.to_numeric(s, errors="coerce")
                st["text_lost"] += int((v.isna() & s.notna() & (s != "")).sum())
                v = v.mask(v.isin(NULLS)).astype("float64")
                nn = v.notna()
                st["nonnull"] += int(nn.sum())
                if nn.any():
                    st["min"] = min(st["min"], float(v.min()))
                    st["max"] = max(st["max"], float(v.max()))
                out[c] = v
            out = pd.DataFrame(out, index=chunk.index)
            table = pa.Table.from_pandas(out, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_pq, table.schema, compression="zstd")
            writer.write_table(table)
            n_total += len(out)
            print(f"  {n_total:>10,} rows  {time.time()-t0:6.0f} s", end="\r")
    if writer:
        writer.close()
    print()

    rep = []
    for i, c in enumerate(cols):
        if c in tcols:
            continue
        st = stats[c]
        rep.append({
            "column": c, "raw_name": raw_cols[i], "units": units[i] if units else "",
            "pct_nonnull": round(100 * st["nonnull"] / max(st["n"], 1), 1),
            "type": "text" if c in txt else "num",
            "min": st["min"] if st["nonnull"] and c not in txt else None,
            "max": st["max"] if st["nonnull"] and c not in txt else None,
            "text_values_dropped": st["text_lost"],
        })
    rep = pd.DataFrame(rep)
    rep.to_csv(out_rep, index=False)

    print(f"Rows: {n_total:,}   unparsed timestamps: {n_bad_ts:,}")
    print(f"Wrote {out_pq}  ({out_pq.stat().st_size/1e6:.0f} MB)")
    print(f"Wrote {out_rep}")
    with pd.option_context("display.max_rows", 500, "display.width", 200):
        print(rep.to_string(index=False))


if __name__ == "__main__":
    main()
