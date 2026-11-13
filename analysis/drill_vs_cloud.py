"""First look: drilling response along 16B vs 2022 event density (GES downhole catalog).

Input : processed/pason_16b_bins_10ft.parquet   (pason_drilling.py)
        processed/catalog_ges.parquet          (catalog_ges.py; uses events with good == True)
Output: processed/drill_vs_cloud_10ft.parquet  bins + event counts within R of each bin midpoint

Event density: number of good events within R metres (3-D sphere) of each bin's midpoint,
for R in RADII_M. Also distance to the nearest event.

Test (within one bit run, default bit 22, 9,863-10,250 ft):
  * Spearman rho between each drilling response and n_R.
  * Significance by circular shift: slide the event-count series along the run by every
    possible offset (wrapping around), recompute rho, and ask how often |rho| >= observed.
    This keeps the autocorrelation of both series, which ordinary p-values ignore.
    With N bins there are only N-1 shifts, so the smallest attainable p is about 1/N.
  * Lag-1 autocorrelation and an effective sample size n_eff ~ N (1-r1)/(1+r1) per series.

Usage:  python analysis/drill_vs_cloud.py [--run 22] [--radius 50]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

BINS = config.OUT / "pason_16b_bins_10ft.parquet"
CAT = config.OUT / "catalog_ges.parquet"
OUT = config.OUT / "drill_vs_cloud_10ft.parquet"
RADII_M = (25, 50, 100)
IN_MIN = 5          # "in cloud": at least this many events within R; "out": none
RESPONSES = ["mse_ksi", "rop_avg_fph", "temp_dt_f", "pvt_change_bbl", "flow_gl_pct",
             "torque_kftlb", "diffp_psi", "wob_klbs"]


def spearman(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5:
        return np.nan, int(ok.sum())
    ra = pd.Series(a[ok]).rank().to_numpy()
    rb = pd.Series(b[ok]).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return np.nan, int(ok.sum())
    return float(np.corrcoef(ra, rb)[0, 1]), int(ok.sum())


def shift_test(y, x):
    rho, n = spearman(y, x)
    if not np.isfinite(rho):
        return rho, np.nan, n
    null = np.array([spearman(y, np.roll(x, k))[0] for k in range(1, len(x))])
    null = null[np.isfinite(null)]
    p = (1 + np.sum(np.abs(null) >= abs(rho))) / (1 + len(null))
    return rho, p, n


def lag1(v):
    v = v[np.isfinite(v)]
    if len(v) < 4 or v.std() == 0:
        return np.nan
    return float(np.corrcoef(v[:-1], v[1:])[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="22")
    ap.add_argument("--radius", type=int, default=50, choices=RADII_M)
    args = ap.parse_args()

    b = pd.read_parquet(BINS)
    ev = pd.read_parquet(CAT)
    ev = ev[ev.good]
    E = ev[["utm_e_m", "utm_n_m", "elev_m"]].to_numpy()
    P = b[["utm_e_m", "utm_n_m", "elev_m"]].to_numpy()
    d = np.sqrt(((P[:, None, :] - E[None, :, :]) ** 2).sum(-1))       # bins x events
    for r in RADII_M:
        b[f"n{r}"] = (d <= r).sum(1)
    b["nearest_event_m"] = d.min(1)
    b.to_parquet(OUT, index=False)
    print(f"Wrote {OUT}  ({len(b)} bins, {len(ev):,} good events)")

    deep = b[b.md_bin_ft >= 9200]
    cols = ["md_bin_ft", "bit_run", "bit_type", "bit_dia_in", "mse_ksi", "rop_avg_fph", "temp_dt_f",
            "pvt_change_bbl", "flow_gl_pct", "n25", "n50", "n100", "nearest_event_m"]
    print("\nBins below 9,200 ft MD:")
    with pd.option_context("display.max_rows", 300, "display.width", 220,
                           "display.float_format", "{:.1f}".format):
        print(deep[cols].to_string(index=False))

    print("\nBy bit run (>= 8,500 ft), medians:")
    runs = b[b.md_bin_ft >= 8500].groupby("bit_run", sort=False).agg(
        md_from=("md_bin_ft", "min"), md_to=("md_bin_ft", "max"), bins=("md_bin_ft", "size"),
        type=("bit_type", "first"), mse=("mse_ksi", "median"), rop=("rop_avg_fph", "median"),
        dt=("temp_dt_f", "median"), n50_sum=("n50", "sum"))
    with pd.option_context("display.width", 200, "display.float_format", "{:.1f}".format):
        print(runs.to_string())

    run = b[(b.bit_run == args.run) & (b.n_bit_runs == 1)].sort_values("md_bin_ft")
    x = run[f"n{args.radius}"].to_numpy(float)
    print(f"\nBit run {args.run}: {len(run)} single-run bins, MD {run.md_bin_ft.min():,.0f}.."
          f"{run.md_bin_ft.max():,.0f} ft; in cloud (n{args.radius} >= {IN_MIN}): {(x >= IN_MIN).sum()} bins, "
          f"out (n{args.radius} = 0): {(x == 0).sum()} bins; "
          f"lag-1 autocorr of n{args.radius} = {lag1(x):.2f}")
    rows = []
    for c in RESPONSES:
        y = run[c].to_numpy(float)
        rho, p, n = shift_test(y, x)
        r1 = lag1(y)
        neff = n * (1 - r1) / (1 + r1) if np.isfinite(r1) and r1 > -1 else np.nan
        inside, outside = y[x >= IN_MIN], y[x == 0]
        rows.append({"response": c, "n": n, "spearman_rho": rho, "p_shift": p, "lag1": r1,
                     "n_eff": neff, "median_in_cloud": np.nanmedian(inside) if len(inside) else np.nan,
                     "median_out": np.nanmedian(outside) if len(outside) else np.nan})
    res = pd.DataFrame(rows)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(res.to_string(index=False))
    print("\np_shift: share of circular shifts with |rho| >= observed. Smallest possible ~ 1/N. "
          "Several responses are tested, so treat p < 0.05 on one of them as a lead, not a finding.")


if __name__ == "__main__":
    main()
