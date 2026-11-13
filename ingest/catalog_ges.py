"""Load the GES downhole-geophone catalog of the April 2022 16A(78)-32 stimulation
(Dyer & Karvounis, "Preliminary Event Catalogue", one xlsx per stage) into the project frame,
and measure each event's closest approach to 16B.

Input : raw/catalogs/ges_2022/FORGE16AApril22AllStage {1,2,3}.xlsx
        processed/survey_16b.parquet
Output: processed/catalog_ges.parquet

Catalog frame (sheet header): origin (0,0,0) = ground level at well 16A, X / Y / Depth in feet,
Depth = TVD below origin, positive down. X is taken as east and Y as north (16A heads ESE,
and the events sit at X ~ +3,500 ft, Y ~ -1,000 ft).
Origin used here: the 16A wellhead from the 16A survey report (UTM 12N E 334639.65,
N 4263442.78 m, ground 1650.19 m). X/Y are grid east/north: against the 16A survey the
three check shots fit best at -0.25..0 deg and clearly worse at the -1.17 deg grid
convergence (survey_16a.py). --rotate-deg is kept for sensitivity tests.

Usage:  python ingest/catalog_ges.py [--max-error-ft 200] [--rotate-deg 0]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from catalog_uuss import RADII_M, well_points, closest  # noqa: E402

# Origin = 16A wellhead from the 16A survey report (SHL E 1097896.92 / N 13987645.20 usft,
# ground 5414 usft). The check shots fit the 16A survey best in this frame with no rotation.
REF_E = 1097896.92 * config.USFT_TO_M
REF_N = 13987645.20 * config.USFT_TO_M
REF_ELEV = 5414.0 * config.USFT_TO_M

SRC_DIR = config.RAW / "catalogs" / "ges_2022"
SURVEY = config.OUT / "survey_16b.parquet"
OUT = config.OUT / "catalog_ges.parquet"
FT = 0.3048

# Cored footage from the End of Well Report bit record (ft MD)
CORED = [(9800, 9817), (9823, 9853), (10250, 10256), (10264, 10271),
         (10274, 10304), (10430, 10460), (10462, 10493)]


def read_stage(path):
    raw = pd.read_excel(path, header=None)
    hdr = raw.index[raw[0].astype(str).str.strip().eq("Source")][-1]   # last "Source" = table header
    df = raw.iloc[hdr + 1:].copy()
    df.columns = [str(c).strip() for c in raw.iloc[hdr]]
    df = df.dropna(subset=["Source"])
    df["file"] = path.name
    return df


def load():
    files = sorted(SRC_DIR.rglob("*.xlsx"))
    if not files:
        sys.exit(f"No xlsx in {SRC_DIR}")
    df = pd.concat([read_stage(f) for f in files], ignore_index=True)
    for c in ("Source", "Status", "Cluster", "Y", "X", "Depth", "MomMag", "PGV", "Stage",
              "P S/N", "S S/N", "Quality", "Error"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["origin_utc"] = pd.to_datetime(df["Origin Date"].astype(str).str.strip() + " " +
                                      df["Origin Time"].astype(str).str.strip(),
                                      format="%d/%m/%Y %H:%M:%S.%f", utc=True)
    df = df.rename(columns={"Source": "source", "Status": "status", "Cluster": "cluster",
                            "X": "x_ft", "Y": "y_ft", "Depth": "depth_ft", "MomMag": "mag",
                            "Stage": "stage", "Quality": "quality", "Error": "error_ft",
                            "Location": "loc_method"})
    df["loc_method"] = df["loc_method"].astype(str).str.strip()
    print(f"Read {len(files)} files: {len(df):,} rows "
          f"({(df.status == 2).sum():,} events, {(df.status == 4).sum()} check shots)")
    return df


def to_project_frame(df, rotate_deg):
    a = np.radians(rotate_deg)
    e = df.x_ft * np.cos(a) - df.y_ft * np.sin(a)
    n = df.x_ft * np.sin(a) + df.y_ft * np.cos(a)
    df["utm_e_m"] = REF_E + e * FT
    df["utm_n_m"] = REF_N + n * FT
    df["elev_m"] = REF_ELEV - df.depth_ft * FT
    df["error_m"] = df.error_ft * FT
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-error-ft", type=float, default=200.0)
    ap.add_argument("--rotate-deg", type=float, default=0.0,
                    help="counter-clockwise rotation of catalog X/Y into UTM grid (deg)")
    args = ap.parse_args()

    df = to_project_frame(load(), args.rotate_deg)
    sv = pd.read_parquet(SURVEY)
    md, w_xyz = well_points(sv)
    dist, md_close = closest(df[["utm_e_m", "utm_n_m", "elev_m"]].to_numpy(), md, w_xyz)
    df["dist_16b_m"] = dist
    df["md_closest_ft"] = md_close
    df["good"] = (df.status == 2) & (df.error_ft <= args.max_error_ft)
    df.to_parquet(OUT, index=False)
    print(f"Wrote {OUT}")

    print("\nCheck shots (perf shots, known 16A positions):")
    print(df[df.status == 4][["stage", "origin_utc", "x_ft", "y_ft", "depth_ft", "error_ft",
                              "utm_e_m", "utm_n_m", "elev_m"]].to_string(index=False))

    ev = df[df.status == 2]
    print(f"\nError (ft) by stage, events: ")
    print(ev.groupby("stage").error_ft.describe(percentiles=[.25, .5, .75, .9]).round(0).to_string())
    print(f"Kept (error <= {args.max_error_ft:g} ft): {df.good.sum():,} of {len(ev):,}")

    g = df[df.good]
    print(f"\nDistance to 16B (good events): min {g.dist_16b_m.min():.0f} m, median {g.dist_16b_m.median():.0f} m")
    for st, gs in g.groupby("stage"):
        print(f"  stage {st:.0f}: {len(gs):>5,} events, median distance {gs.dist_16b_m.median():.0f} m, "
              f"closest-approach MD median {gs.md_closest_ft.median():,.0f} ft")
    for r in RADII_M:
        near = g[g.dist_16b_m <= r]
        if len(near):
            print(f"  within {r:>3} m: {len(near):>5,}   closest-approach MD "
                  f"{near.md_closest_ft.quantile(.05):,.0f} .. {near.md_closest_ft.quantile(.95):,.0f} ft (5-95%)")
        else:
            print(f"  within {r:>3} m: 0")

    near = g[g.dist_16b_m <= 100].copy()
    if len(near):
        near["md_bin"] = (near.md_closest_ft // 50 * 50).astype(int)
        cored = lambda m: any(a <= m < b for a, b in CORED) or any(m <= a < m + 50 for a, b in CORED)
        t = near.groupby("md_bin").agg(events=("source", "size"),
                                       med_dist_m=("dist_16b_m", "median"))
        t["has_core"] = [cored(m) for m in t.index]
        print("\nGood events within 100 m of 16B, by 50-ft MD bin of closest approach:")
        print(t.to_string())


if __name__ == "__main__":
    main()
