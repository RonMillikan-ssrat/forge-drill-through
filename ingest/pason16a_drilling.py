"""16A(78)-32 (drilled Oct 2020 - Jan 2021, before the 2022 stimulation): new-hole rows and 10-ft bins.

Input : processed/pason_16a_10s.parquet  (pason_to_parquet.py --src ... --name 16a)
        processed/survey_16a.parquet     (survey_16a.py)
Output: processed/pason_16a_bins_10ft.parquet

Differences from 16B (pason_drilling.py):
  * No bit-depth channel: new hole = On Bottom == 1 and hole depth at its running maximum.
    Hole depth has spikes (up to 735,423 ft), so it is range-clipped and median-filtered
    before the running maximum is taken.
  * Rows are ordered by file row: timestamps are only to the minute.
  * Torque is in ft-lb. 16A has Bit RPM and a near-bit Downhole Torque channel, so two MSEs:
      mse_surface_ksi  : top-drive RPM x top-drive torque (comparable to 16B mse_surface_ksi)
      mse_downhole_ksi : bit RPM x downhole torque (energy delivered at the bit)
  * Mud temperature is one channel (no in/out difference).
Bit runs (BHA numbers) are from the 16A Daily Operations summary (Composite-C); every run
below the curve is 8.75 in. Runs above 7,390 ft are labelled "upper".

Each bin also gets s_m: horizontal distance along azimuth 105 deg from the 16A wellhead,
the common axis used to line up 16A and 16B positions.

Usage:  python ingest/pason16a_drilling.py [--bin-ft 10]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from pason_drilling import md_to_xyz  # noqa: E402

SRC = config.OUT / "pason_16a_10s.parquet"
SURVEY = config.OUT / "survey_16a.parquet"
OUT_BIN = config.OUT / "pason_16a_bins_{}ft.parquet"
DT_S = 10.0
MIN_ROP = 0.1
AXIS_AZ_DEG = 105.0
TD_DRILLED_FT = 10987.0      # daily ops: drilled to 10,955 ft, cored to 10,987 ft
MAX_NEW_HOLE_FT = 10995.0    # Pason hole depth creeps to 11,087 ft during post-TD cleanouts/logging

KEEP = {
    "row": "row", "timestamp": "timestamp", "total_depth": "hole_depth_ft", "on_bottom": "on_bottom",
    "weight_on_bit": "wob_klbs", "top_drive_rpm": "rpm_topdrive", "bit_rpm": "rpm_bit",
    "top_drive_torque_ft_lbs": "torque_ftlb", "downhole_torque": "torque_dh_ftlb",
    "rop_depth_hour": "rop_pason_fph", "differential_pressure": "diffp_psi",
    "pump_pressure": "spp_psi", "flow_in": "flow_in", "return_flow": "flow_out_pct",
    "pit_g_l_active": "pit_gl_bbl", "mud_temp": "temp_mud_f", "bit_diameter": "bit_dia_pason_in",
    "hookload": "hookload_klbs",
}
VALID = {
    "hole_depth_ft": (0, 11100), "wob_klbs": (0, 120), "rpm_topdrive": (0, 250), "rpm_bit": (0, 400),
    "torque_ftlb": (0, 40000), "torque_dh_ftlb": (0, 20000), "rop_pason_fph": (0, 500),
    "diffp_psi": (-500, 3000), "temp_mud_f": (32, 300), "hookload_klbs": (0, 800),
}
# (BHA, type, MD in, MD out) from the Daily Operations summary
BIT_RUNS = [
    ("30", "PDC", 7390, 8024), ("34", "PDC", 8024, 8241), ("35", "PDC", 8241, 8535),
    ("38", "PDC", 8535, 9064), ("39", "PDC", 9064, 9748), ("41", "PDC", 9748, 10490),
    ("42", "PDC", 10490, 10955), ("47", "Core", 10955, 10971), ("48", "PDC", 10971, 10973),
    ("49", "Core", 10973, 10987),
]


def assign_run(md):
    lab = np.full(len(md), "upper", dtype=object)
    typ = np.full(len(md), "PDC", dtype=object)
    for b, t, a, z in BIT_RUNS:
        m = (md > a) & (md <= z)
        lab[m], typ[m] = b, t
    lab[md > BIT_RUNS[-1][3]] = BIT_RUNS[-1][0]
    return lab, typ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-ft", type=float, default=10.0)
    args = ap.parse_args()
    B = args.bin_ft

    df = pd.read_parquet(SRC, columns=list(KEEP)).rename(columns=KEEP).sort_values("row").reset_index(drop=True)
    print(f"Read {len(df):,} rows  {df.timestamp.min()} .. {df.timestamp.max()}")
    for c, (lo, hi) in VALID.items():
        n_bad = (~df[c].between(lo, hi) & df[c].notna()).sum()
        df[c] = df[c].where(df[c].between(lo, hi))
        if n_bad:
            print(f"  {c}: {n_bad:,} values outside [{lo}, {hi}] set to null")

    hd = df["hole_depth_ft"]
    hd_med = hd.rolling(7, center=True, min_periods=3).median()
    spike = (hd - hd_med).abs() > 2.0
    runmax = hd_med.ffill().cummax()
    new = (df.on_bottom == 1) & hd.notna() & ~spike & (hd >= runmax - 0.1) & (hd <= MAX_NEW_HOLE_FT)
    past = (df.on_bottom == 1) & (hd > MAX_NEW_HOLE_FT)
    print(f"On-bottom rows deeper than {MAX_NEW_HOLE_FT:,.0f} ft (post-TD depth creep), dropped: {past.sum():,}")
    print(f"on_bottom: {(df.on_bottom == 1).sum():,}   new hole: {new.sum():,} ({new.sum()*DT_S/3600:.1f} h)   "
          f"depth spikes removed: {spike.sum():,}")
    print(f"Max hole depth on new-hole rows: {hd[new].max():,.1f} ft  (drilled TD {TD_DRILLED_FT:,.0f} ft per daily ops, "
          f"difference {hd[new].max() - TD_DRILLED_FT:+.1f} ft)")

    d = df[new].copy()
    d["bit_run"], d["bit_type"] = assign_run(d.hole_depth_ft.to_numpy())
    d["bit_dia_in"] = 8.75
    up = d.hole_depth_ft <= 7390
    d.loc[up, "bit_dia_in"] = d.loc[up, "bit_dia_pason_in"]        # upper hole: Pason value, not used below
    area = np.pi * (d.bit_dia_in / 2) ** 2
    rop = d.rop_pason_fph.where(d.rop_pason_fph >= MIN_ROP)
    thrust = d.wob_klbs * 1e3 / area
    d["mse_surface_ksi"] = (thrust + 120 * np.pi * d.rpm_topdrive * d.torque_ftlb / (area * rop)) / 1e3
    d["mse_downhole_ksi"] = (thrust + 120 * np.pi * d.rpm_bit * d.torque_dh_ftlb / (area * rop)) / 1e3

    d["md_bin_ft"] = np.floor(d.hole_depth_ft / B) * B
    g = d.groupby("md_bin_ft")
    med = ["wob_klbs", "rpm_topdrive", "rpm_bit", "torque_ftlb", "torque_dh_ftlb", "rop_pason_fph", "diffp_psi",
           "spp_psi", "flow_out_pct", "temp_mud_f", "mse_surface_ksi", "mse_downhole_ksi", "bit_dia_in"]
    b = g[med].median()
    b["n_samples"] = g.size()
    b["hours_on_bottom"] = b.n_samples * DT_S / 3600
    b["ft_drilled"] = g.hole_depth_ft.max() - g.hole_depth_ft.min()
    b["rop_avg_fph"] = b.ft_drilled / b.hours_on_bottom
    b["pit_change_bbl"] = g.pit_gl_bbl.last() - g.pit_gl_bbl.first()
    b["bit_run"] = g.bit_run.agg(lambda x: x.mode().iat[0])
    b["bit_type"] = g.bit_type.agg(lambda x: x.mode().iat[0])
    b["n_bit_runs"] = g.bit_run.nunique()
    b["t_first"] = g.timestamp.min()
    b = b.reset_index()
    b["md_mid_ft"] = b.md_bin_ft + B / 2

    sv = pd.read_parquet(SURVEY)
    for c, v in md_to_xyz(b.md_mid_ft.to_numpy(), sv).items():
        b[c] = v
    e0, n0 = sv.utm_e_m.iloc[0], sv.utm_n_m.iloc[0]
    az = np.radians(AXIS_AZ_DEG)
    b["s_m"] = (b.utm_e_m - e0) * np.sin(az) + (b.utm_n_m - n0) * np.cos(az)

    # z-score of each response within its bit run (PDC runs below 7,390 ft only)
    deep = (b.md_bin_ft >= 7390) & (b.bit_type == "PDC") & (b.n_bit_runs == 1)
    for c in ("mse_surface_ksi", "mse_downhole_ksi", "rop_avg_fph", "torque_ftlb", "temp_mud_f"):
        grp = b[deep].groupby("bit_run")[c]
        b.loc[deep, f"{c}_z"] = (b.loc[deep, c] - grp.transform("median")) / grp.transform(
            lambda x: 1.4826 * (x - x.median()).abs().median()).replace(0, np.nan)
    out = Path(str(OUT_BIN).format(f"{B:g}"))
    b.to_parquet(out, index=False)
    print(f"Wrote {out}  ({len(b):,} bins, {b.md_bin_ft.min():.0f}..{b.md_bin_ft.max()+B:.0f} ft)")

    summ = b[b.md_bin_ft >= 7390].groupby("bit_run", sort=False).agg(
        md_from=("md_bin_ft", "min"), md_to=("md_bin_ft", "max"), bins=("md_bin_ft", "size"),
        hours=("hours_on_bottom", "sum"), rop=("rop_avg_fph", "median"), wob=("wob_klbs", "median"),
        rpm_bit=("rpm_bit", "median"), mse_surf=("mse_surface_ksi", "median"),
        mse_dh=("mse_downhole_ksi", "median"), temp=("temp_mud_f", "median"),
        s_from=("s_m", "min"), s_to=("s_m", "max"))
    with pd.option_context("display.width", 220, "display.float_format", "{:.1f}".format):
        print("\nBelow 7,390 ft, by BHA:")
        print(summ.to_string())


if __name__ == "__main__":
    main()
