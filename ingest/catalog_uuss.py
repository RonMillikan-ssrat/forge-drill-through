"""Load the UUSS surface-network catalog (Niemz et al. 2024, GDR 1584) into the project frame,
and measure each event's closest approach to 16B.

Input : raw/catalogs/uuss_surface/*.csv      (semicolon-separated, see ReadMe)
        processed/survey_16b.parquet
Output: processed/catalog_uuss.parquet

Catalog frame (from the ReadMe): east/north/depth in metres relative to the 16A wellhead,
  ref UTM 12N NAD83(2011) E 334641.1891, N 4263443.693, elevation 1650.0249 m, depth positive down.
Project frame: UTM 12N metres (NAD83; HARN vs 2011 differ by cm here), elevation m above MSL.

Usage:  python ingest/catalog_uuss.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

SRC_DIR = config.RAW / "catalogs" / "uuss_surface"
SURVEY = config.OUT / "survey_16b.parquet"
OUT = config.OUT / "catalog_uuss.parquet"

REF_E, REF_N, REF_ELEV = 334641.1891, 4263443.693, 1650.0249
DRILL_START_16B = pd.Timestamp("2023-04-26", tz="UTC")   # first Pason row; events after this post-date the bit
RADII_M = (25, 50, 100, 200)
FT_TO_M = config.FT_TO_M


def load():
    files = sorted(SRC_DIR.rglob("*.csv"))
    if not files:
        sys.exit(f"No CSV in {SRC_DIR}")
    df = pd.read_csv(files[0], sep=";", skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"time[UTC]": "time_utc", "east[m]": "east_m", "north[m]": "north_m",
                            "depth[m]": "depth_m", "magnitude[M_FORK]": "mag"})
    for c in ("evid", "loc_type"):
        df[c] = df[c].astype(str).str.strip()
    df["origin_utc"] = pd.to_datetime(df["date"].astype(str).str.strip() + " " +
                                      df["time_utc"].astype(str).str.strip(), utc=True, format="mixed")
    df["phase"] = df["evid"].str.split("_").str[0]
    df["utm_e_m"] = REF_E + df["east_m"]
    df["utm_n_m"] = REF_N + df["north_m"]
    df["elev_m"] = REF_ELEV - df["depth_m"]
    df["before_16b"] = df["origin_utc"] < DRILL_START_16B
    print(f"Read {files[0].name}: {len(df):,} events")
    return df


def well_points(sv, step_ft=1.0):
    md = np.arange(sv.md_ft.min(), sv.md_ft.max() + step_ft, step_ft)
    xyz = np.column_stack([np.interp(md, sv.md_ft, sv[c]) for c in ("utm_e_m", "utm_n_m", "elev_m")])
    return md, xyz


def closest(ev_xyz, md, w_xyz, chunk=2000):
    dmin = np.empty(len(ev_xyz))
    imin = np.empty(len(ev_xyz), dtype=int)
    for i in range(0, len(ev_xyz), chunk):
        e = ev_xyz[i:i + chunk]
        d2 = ((e[:, None, :] - w_xyz[None, :, :]) ** 2).sum(-1)
        j = d2.argmin(1)
        imin[i:i + chunk] = j
        dmin[i:i + chunk] = np.sqrt(d2[np.arange(len(e)), j])
    return dmin, md[imin]


def main():
    df = load()
    sv = pd.read_parquet(SURVEY)
    md, w_xyz = well_points(sv)
    dist, md_close = closest(df[["utm_e_m", "utm_n_m", "elev_m"]].to_numpy(), md, w_xyz)
    df["dist_16b_m"] = dist
    df["md_closest_ft"] = md_close
    df.to_parquet(OUT, index=False)
    print(f"Wrote {OUT}")

    print("\nCounts (phase x final_cat x loc_type):")
    print(df.groupby(["phase", "final_cat", "loc_type"]).size().to_string())

    for ph, g in df[df.final_cat == 1].groupby("phase"):
        print(f"\n{ph}, final_cat=1: {len(g):,} events   "
              f"{g.origin_utc.min():%Y-%m-%d %H:%M} .. {g.origin_utc.max():%Y-%m-%d %H:%M} UTC   "
              f"before 16B spud: {g.before_16b.sum():,}")
        print(f"  mag {g.mag.min():.2f} .. {g.mag.max():.2f}   elev {g.elev_m.min():.0f} .. {g.elev_m.max():.0f} m   "
              f"E {g.utm_e_m.min():.0f}..{g.utm_e_m.max():.0f}  N {g.utm_n_m.min():.0f}..{g.utm_n_m.max():.0f}")
        print(f"  distance to 16B: min {g.dist_16b_m.min():.0f} m, median {g.dist_16b_m.median():.0f} m")
        for r in RADII_M:
            near = g[g.dist_16b_m <= r]
            if len(near):
                print(f"  within {r:>3} m: {len(near):>5,} events   closest-approach MD "
                      f"{near.md_closest_ft.quantile(.05):,.0f} .. {near.md_closest_ft.quantile(.95):,.0f} ft (5-95%)")
            else:
                print(f"  within {r:>3} m: 0")

    stim = df[(df.final_cat == 1) & (df.phase == "STIM")]
    near = stim[stim.dist_16b_m <= 200]
    if len(near):
        print("\nSTIM events within 200 m of 16B, by 250-ft MD bin of closest approach "
              "(cored: 9,750-10,000 and 10,250-10,500 ft, approx.):")
        b = (near.md_closest_ft // 250 * 250).astype(int)
        print(b.value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
