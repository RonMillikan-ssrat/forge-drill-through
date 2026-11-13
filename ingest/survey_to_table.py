#!/usr/bin/env python3
"""
Parse the 16B(78)-32 gyro survey (Final Survey Report.txt) into a table that
maps measured depth (MD) to UTM 12N metres and elevation, the frame the 2022
microseismic catalogs use.

What it does
  1. Reads the header (map system, datum, wellhead, KB elevation, north ref).
  2. Reads every survey station: MD, INC, AZI, TVD, SSTVD, N/S, NORTHING, E/W, EASTING, VSEC, DLS.
  3. Re-computes the well path from MD/INC/AZI with minimum curvature and compares it with
     the contractor's N/S, E/W, TVD. Agreement to ~0.01 ft means the parse is right
     and the azimuths are grid azimuths (as the header says).
  4. Converts to metres: UTM easting/northing (US survey ft -> m), elevation above sea level,
     TVD below KB.
  5. Writes data/processed/survey_16b.csv and .parquet.

md_to_xyz(md_ft) is provided for other scripts: linear interpolation between 25-ft stations.
With dogleg severity of a few deg/100 ft, the chord-vs-arc error over 25 ft is millimetres.

Run:
  python ingest/survey_to_table.py                 # uses config.SURVEY_16B_TXT
  python ingest/survey_to_table.py path/to/report.txt
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

COLS = ["md_ft", "inc_deg", "azi_deg", "tvd_ft", "sstvd_ft", "ns_ft",
        "northing_usft", "ew_ft", "easting_usft", "vsec_ft", "dls_deg100ft"]


def parse_survey(path: Path):
    header, rows = {}, []
    for line in path.read_text(errors="replace").splitlines():
        if ":" in line and not rows:
            k, _, v = line.partition(":")
            header[k.strip()] = v.strip()
            continue
        toks = line.split()
        if len(toks) != len(COLS):
            continue
        try:
            rows.append([float(t) for t in toks])
        except ValueError:
            continue  # unit row "(ft) (deg) ..." or column titles
    if not rows:
        sys.exit(f"No survey stations parsed from {path}")
    df = pd.DataFrame(rows, columns=COLS)
    return header, df


def header_number(header, key):
    m = re.search(r"-?\d+(\.\d+)?", header.get(key, ""))
    return float(m.group()) if m else None


def min_curvature(md, inc_deg, azi_deg):
    """Return cumulative (north, east, tvd) by the minimum-curvature method."""
    inc = np.radians(inc_deg)
    azi = np.radians(azi_deg)
    n = np.zeros_like(md)
    e = np.zeros_like(md)
    v = np.zeros_like(md)
    for i in range(1, len(md)):
        dmd = md[i] - md[i - 1]
        i1, i2, a1, a2 = inc[i - 1], inc[i], azi[i - 1], azi[i]
        cos_dl = np.cos(i2 - i1) - np.sin(i1) * np.sin(i2) * (1 - np.cos(a2 - a1))
        dl = np.arccos(np.clip(cos_dl, -1.0, 1.0))
        rf = 1.0 if dl < 1e-9 else (2.0 / dl) * np.tan(dl / 2.0)
        n[i] = n[i - 1] + dmd / 2 * (np.sin(i1) * np.cos(a1) + np.sin(i2) * np.cos(a2)) * rf
        e[i] = e[i - 1] + dmd / 2 * (np.sin(i1) * np.sin(a1) + np.sin(i2) * np.sin(a2)) * rf
        v[i] = v[i - 1] + dmd / 2 * (np.cos(i1) + np.cos(i2)) * rf
    return n, e, v


def build(path: Path) -> pd.DataFrame:
    header, df = parse_survey(path)
    print("Header:")
    for k in ("MAP SYSTEM", "GEODIC DATUM", "MAP ZONE", "NORTH REF", "CALCULATION METHOD",
              "WELL EASTING", "WELL NORTHING", "KB ELEV", "GL ELEV", "DATE"):
        print(f"  {k:20s}: {header.get(k)}")
    print(f"Stations: {len(df)}  MD {df.md_ft.min():.2f} .. {df.md_ft.max():.2f} ft  "
          f"step median {np.median(np.diff(df.md_ft)):.2f} ft")

    if "Feet" not in header.get("MAP SYSTEM", "") or "12N" not in header.get("MAP ZONE", ""):
        print("WARNING: map system/zone not 'UTM US Survey Feet, Zone 12N' - check units before trusting x,y")

    # 3. Independent check of the contractor's numbers
    n, e, v = min_curvature(df.md_ft.to_numpy(), df.inc_deg.to_numpy(), df.azi_deg.to_numpy())
    dn, de, dv = n - df.ns_ft, e - df.ew_ft, v - df.tvd_ft
    print("Min-curvature recompute vs report (ft): "
          f"max |dN|={np.abs(dn).max():.3f}  max |dE|={np.abs(de).max():.3f}  max |dTVD|={np.abs(dv).max():.3f}")

    # Which foot? The header says "US Survey Feet", but the End of Well Report's UTM line
    # matches the international foot. The two differ by ~8.5 m in northing here.
    # Compare these lat/lon values with the first station of the "Geographic" survey report.
    try:
        from pyproj import Transformer
        to_ll = Transformer.from_crs("EPSG:26912", "EPSG:4269", always_xy=True)  # NAD83 UTM12N m -> NAD83 lat/lon
        e0, n0 = df.easting_usft.iloc[0], df.northing_usft.iloc[0]
        for name, f in (("US survey ft", config.USFT_TO_M), ("international ft", config.FT_TO_M)):
            lon, lat = to_ll.transform(e0 * f, n0 * f)
            print(f"Wellhead as {name:16s}: lat {lat:.6f}  lon {lon:.6f}")
    except ImportError:
        print("pyproj not installed; skipping lat/lon check")
    map_ft = config.MAP_FT_TO_M
    print(f"Using {'US survey' if map_ft == config.USFT_TO_M else 'international'} foot for map coordinates "
          "(config.MAP_FT_TO_M)")

    kb = header_number(header, "KB ELEV")
    out = df.copy()
    out["utm_e_m"] = out.easting_usft * map_ft
    out["utm_n_m"] = out.northing_usft * map_ft
    # SSTVD = TVD - KB elevation (positive below sea level), so elevation = -SSTVD
    out["elev_m"] = -out.sstvd_ft * config.USFT_TO_M
    out["tvd_m"] = out.tvd_ft * config.USFT_TO_M
    out["md_m"] = out.md_ft * config.FT_TO_M
    if kb is not None:
        chk = np.abs((out.tvd_ft - kb) - out.sstvd_ft).max()
        print(f"SSTVD = TVD - KB check: max error {chk:.3f} ft  (KB = {kb} ft)")
    return out


_SURVEY = None


def md_to_xyz(md_ft):
    """Interpolate UTM easting/northing (m) and elevation (m asl) at measured depth(s) in ft."""
    global _SURVEY
    if _SURVEY is None:
        _SURVEY = pd.read_parquet(config.OUT / "survey_16b.parquet")
    s = _SURVEY
    md = np.asarray(md_ft, dtype=float)
    return (np.interp(md, s.md_ft, s.utm_e_m),
            np.interp(md, s.md_ft, s.utm_n_m),
            np.interp(md, s.md_ft, s.elev_m))


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else config.SURVEY_16B_TXT
    out = build(path)
    config.OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.OUT / "survey_16b.csv", index=False)
    out.to_parquet(config.OUT / "survey_16b.parquet", index=False)
    last = out.iloc[-1]
    print(f"Wrote {config.OUT / 'survey_16b.parquet'}")
    print(f"Wellhead: E {out.utm_e_m.iloc[0]:.2f} m  N {out.utm_n_m.iloc[0]:.2f} m  "
          f"KB elev {out.elev_m.iloc[0]:.2f} m")
    print(f"Last station: MD {last.md_ft:.2f} ft  E {last.utm_e_m:.2f} m  N {last.utm_n_m:.2f} m  "
          f"elev {last.elev_m:.2f} m  TVD {last.tvd_ft:.2f} ft  INC {last.inc_deg:.2f}  AZI {last.azi_deg:.2f}")


if __name__ == "__main__":
    main()
