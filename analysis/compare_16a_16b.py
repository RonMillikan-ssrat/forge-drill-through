"""Does 16B's drilling response follow the 2022 cloud once 16A's pre-stimulation profile is allowed for?

Input : processed/drill_vs_cloud_10ft.parquet   16B bins + event counts (drill_vs_cloud.py)
        processed/pason_16a_bins_10ft.parquet   16A bins + within-run z-scores (pason16a_drilling.py)
        processed/survey_16a.parquet            for the common-axis origin
Output: processed/compare_16a_16b_10ft.parquet

Steps
 1. 16B: robust z-score of MSE, ROP and mud dT within each PDC bit run below 8,500 ft
    (bins cut by one run only). Cores, TCI stubs and mixed bins are left out.
 2. Common axis: s_m = horizontal distance along azimuth 105 deg from the 16A wellhead.
    For each 16B bin, the 16A value is taken at the same s_m (linear, only where 16A has
    data within 15 m of s). The 3-D gap between the two wells at that s is reported.
 3. Residual r = z16B - beta * z16A (beta by least squares). r is what 16A's
    pre-stimulation rock profile does not explain.
 4. Spearman rho of r (and of raw z16B) against event count n_R, R = 25/50/100 m, with the
    circular-shift p-value from drill_vs_cloud.py run over the whole 16B series.

Pairs: 16B mse_ksi (surface + motor) with 16A mse_downhole_ksi (at the bit);
       16B rop_avg_fph with 16A rop_avg_fph. 16A has no mud dT, so dT is tested on 16B alone.

Usage:  python analysis/compare_16a_16b.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from drill_vs_cloud import shift_test, spearman, lag1  # noqa: E402

B16 = config.OUT / "drill_vs_cloud_10ft.parquet"
A16 = config.OUT / "pason_16a_bins_10ft.parquet"
SV16A = config.OUT / "survey_16a.parquet"
OUT = config.OUT / "compare_16a_16b_10ft.parquet"
AXIS_AZ_DEG = 105.0
MAX_GAP_S_M = 15.0
PAIRS = [("mse_ksi", "mse_downhole_ksi_z"), ("rop_avg_fph", "rop_avg_fph_z"), ("temp_dt_f", None)]


def robust_z(df, col, by):
    g = df.groupby(by)[col]
    med = g.transform("median")
    mad = g.transform(lambda x: 1.4826 * (x - x.median()).abs().median())
    return (df[col] - med) / mad.replace(0, np.nan)


def main():
    b = pd.read_parquet(B16)
    a = pd.read_parquet(A16)
    sv = pd.read_parquet(SV16A)
    e0, n0 = sv.utm_e_m.iloc[0], sv.utm_n_m.iloc[0]
    az = np.radians(AXIS_AZ_DEG)
    b["s_m"] = (b.utm_e_m - e0) * np.sin(az) + (b.utm_n_m - n0) * np.cos(az)

    keep = (b.md_bin_ft >= 8500) & (b.bit_type == "PDC") & (b.n_bit_runs == 1)
    b = b[keep].sort_values("md_bin_ft").reset_index(drop=True)
    for c, _ in PAIRS:
        b[f"{c}_z"] = robust_z(b, c, "bit_run")

    a = a[(a.md_bin_ft >= 7390) & (a.bit_type == "PDC") & (a.n_bit_runs == 1)].sort_values("s_m")
    sa = a.s_m.to_numpy()
    near = np.array([np.abs(sa - s).min() <= MAX_GAP_S_M for s in b.s_m])
    for c16b, c16a in PAIRS:
        if c16a:
            v = np.interp(b.s_m, sa, a[c16a].to_numpy())
            b[f"a16_{c16a}"] = np.where(near, v, np.nan)
    for c in ("utm_e_m", "utm_n_m", "elev_m"):
        b[f"a16_{c}"] = np.where(near, np.interp(b.s_m, sa, a[c].to_numpy()), np.nan)
    b["well_gap_m"] = np.sqrt((b.utm_e_m - b.a16_utm_e_m) ** 2 + (b.utm_n_m - b.a16_utm_n_m) ** 2 +
                              (b.elev_m - b.a16_elev_m) ** 2)
    b["well_dz_m"] = b.elev_m - b.a16_elev_m

    print(f"16B PDC bins (>= 8,500 ft, single run): {len(b)}   with a 16A match: {near.sum()}")
    print(f"16B s_m {b.s_m.min():.0f}..{b.s_m.max():.0f} m   16A s_m {sa.min():.0f}..{sa.max():.0f} m")
    print(f"16B - 16A separation at the same s: median {b.well_gap_m.median():.0f} m "
          f"(range {b.well_gap_m.min():.0f}..{b.well_gap_m.max():.0f}), 16B above 16A by "
          f"{b.well_dz_m.median():.0f} m (median)")
    print("\nRuns used:", b.groupby("bit_run", sort=False).md_bin_ft.agg(["min", "max", "size"]).to_string())

    rows = []
    for c16b, c16a in PAIRS:
        y = b[f"{c16b}_z"].to_numpy(float)
        if c16a:
            x16a = b[f"a16_{c16a}"].to_numpy(float)
            ok = np.isfinite(y) & np.isfinite(x16a)
            beta = float(np.sum(y[ok] * x16a[ok]) / np.sum(x16a[ok] ** 2)) if ok.sum() > 5 else np.nan
            rho_ab, _ = spearman(y, x16a)
            resid = y - beta * x16a
            b[f"{c16b}_resid"] = resid
        else:
            beta, rho_ab, resid = np.nan, np.nan, y
        for R in (25, 50, 100):
            nR = b[f"n{R}"].to_numpy(float)
            for label, series in (("z16B", y), ("resid", resid)):
                if label == "resid" and not c16a:
                    continue
                rho, p, n = shift_test(series, nR)
                rows.append({"response": c16b, "series": label, "R_m": R, "n": n, "rho": rho, "p_shift": p,
                             "beta_16A": beta if label == "resid" else np.nan,
                             "rho_16B_16A": rho_ab if label == "resid" else np.nan,
                             "lag1": lag1(series)})
    res = pd.DataFrame(rows)
    b.to_parquet(OUT, index=False)
    print(f"\nWrote {OUT}")
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(res.to_string(index=False))
    print("\nrho_16B_16A: does 16B's within-run profile follow 16A's at the same position? "
          "(positive = shared rock signal)\n"
          "resid rows: 16B after removing beta * 16A. p_shift as in drill_vs_cloud.py; "
          f"{len(res)} tests, so expect ~{0.05*len(res):.1f} with p < 0.05 by chance.")


if __name__ == "__main__":
    main()
