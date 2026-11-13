"""16A(78)-32 survey -> table, then compare the GES check shots with the 16A trajectory.

Input : raw/16A/survey/16A(78)-32 Survey.xlsx   (GDR 1283)
        processed/catalog_ges.parquet          (catalog_ges.py; check shots = status 4)
Output: processed/survey_16a.parquet

Spreadsheet columns, identified from the tail rows (closure distance = hypot(N, E) and
closure azimuth = atan2(E, N) both check out):
  1 station, 2 MD, 3 inc, 4 azi (grid), 5 course length, 6 TVD below KB, 7 vertical section,
  8 N/S, 9 E/W, 10 closure distance, 11 closure azimuth, 12 DLS, 13 tool. Units: US survey ft.
Survey report header: SHL N 13987645.20 / E 1097896.92 usft, KB = GE 5414 + 30 = 5444 usft,
azimuths to grid north.

Check shots: GES frame is X east, Y north, depth below ground at 16A, in feet. Ground at 16A
is 30 ft below KB, so catalog depth = survey TVD - 30. For each candidate rotation of the
catalog X/Y (and with/without a free vertical shift) the script reports how far each shot
lies from the 16A trajectory.

Usage:  python ingest/survey_16a.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

SRC = config.RAW / "16A" / "survey" / "16A(78)-32 Survey.xlsx"
CAT = config.OUT / "catalog_ges.parquet"
OUT = config.OUT / "survey_16a.parquet"

USFT = config.USFT_TO_M
SHL_E_USFT, SHL_N_USFT = 1097896.92, 13987645.20
KB_USFT, GE_USFT = 5444.0, 5414.0
COLS = {1: "station", 2: "md_ft", 3: "inc_deg", 4: "azi_deg", 5: "cl_ft", 6: "tvd_ft",
        7: "vsec_ft", 8: "ns_ft", 9: "ew_ft", 10: "closure_ft", 11: "closure_azi", 12: "dls", 13: "tool"}


def min_curvature(md, inc, azi):
    i, a = np.radians(inc), np.radians(azi)
    n = np.zeros(len(md)); e = np.zeros(len(md)); v = np.zeros(len(md))
    for k in range(1, len(md)):
        dmd = md[k] - md[k - 1]
        cosdl = (np.cos(i[k] - i[k - 1]) - np.sin(i[k - 1]) * np.sin(i[k]) * (1 - np.cos(a[k] - a[k - 1])))
        dl = np.arccos(np.clip(cosdl, -1, 1))
        rf = 1.0 if dl < 1e-9 else 2 / dl * np.tan(dl / 2)
        n[k] = n[k - 1] + dmd / 2 * (np.sin(i[k - 1]) * np.cos(a[k - 1]) + np.sin(i[k]) * np.cos(a[k])) * rf
        e[k] = e[k - 1] + dmd / 2 * (np.sin(i[k - 1]) * np.sin(a[k - 1]) + np.sin(i[k]) * np.sin(a[k])) * rf
        v[k] = v[k - 1] + dmd / 2 * (np.cos(i[k - 1]) + np.cos(i[k])) * rf
    return n, e, v


def load_survey():
    raw = pd.read_excel(SRC, header=None)
    df = raw[list(COLS)].rename(columns=COLS)
    for c in COLS.values():
        if c != "tool":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["md_ft", "inc_deg", "azi_deg", "tvd_ft"]).sort_values("md_ft").reset_index(drop=True)
    print(f"16A survey: {len(df)} stations, MD {df.md_ft.min():,.0f} .. {df.md_ft.max():,.0f} ft")
    print(df.head(4).to_string(index=False))
    return df


def main():
    sv = load_survey()
    n, e, v = min_curvature(sv.md_ft.to_numpy(), sv.inc_deg.to_numpy(), sv.azi_deg.to_numpy())
    v0 = sv.tvd_ft.iloc[0] - v[0]
    print(f"Min-curvature recompute vs sheet (ft): max |dN| {np.abs(n - sv.ns_ft).max():.2f}  "
          f"max |dE| {np.abs(e - sv.ew_ft).max():.2f}  max |dTVD| {np.abs(v + v0 - sv.tvd_ft).max():.2f}")

    sv["utm_e_m"] = (SHL_E_USFT + sv.ew_ft) * USFT
    sv["utm_n_m"] = (SHL_N_USFT + sv.ns_ft) * USFT
    sv["elev_m"] = (KB_USFT - sv.tvd_ft) * USFT
    sv["depth_below_ground_ft"] = sv.tvd_ft - (KB_USFT - GE_USFT)
    sv.to_parquet(OUT, index=False)
    print(f"Wrote {OUT}   wellhead E {sv.utm_e_m.iloc[0]:.2f} N {sv.utm_n_m.iloc[0]:.2f} m   "
          f"TD: MD {sv.md_ft.iloc[-1]:,.0f} ft, E {sv.utm_e_m.iloc[-1]:.1f} N {sv.utm_n_m.iloc[-1]:.1f} "
          f"elev {sv.elev_m.iloc[-1]:.1f} m")

    # check shots vs trajectory (local frame, ft)
    cat = pd.read_parquet(CAT)
    shots = cat[cat.status == 4].sort_values("stage")
    md = np.arange(sv.md_ft.min(), sv.md_ft.max(), 0.5)
    W = np.column_stack([np.interp(md, sv.md_ft, sv[c]) for c in ("ew_ft", "ns_ft", "depth_below_ground_ft")])
    rows = []
    for rot in np.arange(-3.0, 3.01, 0.25):
        a = np.radians(rot)
        for _, s in shots.iterrows():
            x = s.x_ft * np.cos(a) - s.y_ft * np.sin(a)
            y = s.x_ft * np.sin(a) + s.y_ft * np.cos(a)
            d = np.sqrt((W[:, 0] - x) ** 2 + (W[:, 1] - y) ** 2 + (W[:, 2] - s.depth_ft) ** 2)
            i = d.argmin()
            rows.append({"rot_deg": rot, "stage": int(s.stage), "shot_err_ft": s.error_ft, "md_closest_ft": md[i],
                         "miss_3d_ft": d[i], "horiz_ft": np.hypot(W[i, 0] - x, W[i, 1] - y),
                         "dz_ft": s.depth_ft - W[i, 2]})
    r = pd.DataFrame(rows)
    print("\nCheck shots vs 16A trajectory at rotation 0 (catalog X/Y = grid E/N):")
    print(r[r.rot_deg == 0].round(1).to_string(index=False))
    summ = r.groupby("rot_deg").agg(sum_miss_ft=("miss_3d_ft", "sum"), max_horiz_ft=("horiz_ft", "max"),
                                    mean_dz_ft=("dz_ft", "mean"))
    print("\nTotal miss by rotation (deg, + = counter-clockwise):")
    print(summ.round(1).to_string())
    best = summ.sum_miss_ft.idxmin()
    print(f"\nBest rotation: {best:+.2f} deg   (grid convergence here is -1.17 deg)"
          + ("   WARNING: at the edge of the search range" if abs(best) >= 3.0 else ""))
    print(r[r.rot_deg == best].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
