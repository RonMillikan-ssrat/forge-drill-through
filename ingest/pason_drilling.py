"""Step 2: new-hole, on-bottom drilling rows from the 16B Pason Parquet, then per-foot bins.

Input : processed/pason_16b_10s.parquet  (from pason_to_parquet.py)
        processed/survey_16b.parquet     (from survey_to_table.py)
Output: processed/pason_16b_drilling.parquet  10-s rows that cut new hole, with Teale MSE
        processed/pason_16b_bins_<B>ft.parquet MD bins (default 10 ft) + survey x/y/elevation

A row "cuts new hole" when
  - Pason says on bottom (on_bottom_unitless == 1),
  - the bit is within BIT_TOL_FT of hole depth, and
  - hole depth is at (or within NEW_TOL_FT of) its running maximum.
Reaming / repeat passes have bit depth above hole depth, so they drop out
without date-based filters.

Teale (1965) MSE, psi:  WOB/A + 120*pi*(Ns*Ts + Nm*Tm) / (A*ROP)
  WOB lbf, A bit area in^2, N rpm, T ft-lbf, ROP ft/hr.
  mse_surface_ksi : surface terms only (Ns*Ts) -- misses the motor's power
  mse_ksi         : surface + mud motor, motor torque Tm = diffP * (rated torque / rated diffP),
                    motor rpm Nm from Pason "Motor RPM". Use this one.

Bins: BIN_FT along MD (default 10 ft ~ 3 m). At ~100 ft/hr a 10-s log has only
1-2 samples per foot, so 1-ft bins quantise ROP and leave gaps.

Usage:  python ingest/pason_drilling.py [--bin-ft 10]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

SRC = config.OUT / "pason_16b_10s.parquet"
SURVEY = config.OUT / "survey_16b.parquet"
OUT_ROWS = config.OUT / "pason_16b_drilling.parquet"
OUT_BIN = config.OUT / "pason_16b_bins_{}ft.parquet"

BIT_TOL_FT = 1.0          # bit depth within this of hole depth
NEW_TOL_FT = 0.05         # hole depth within this of running max
MIN_ROP = 0.1             # ft/hr; below this MSE blows up
DT_S = 10.0               # sample interval

# Bit record, End of Well Report Table 1: (bit label, size in, type, MD in, MD out).
# Runs with 0 ft drilled are omitted. Bit size from here replaces Pason's hand-set hole diameter.
BIT_RUNS = [
    ("1", 22.0, "PDC", 115, 1146), ("2", 14.75, "MillTooth", 1146, 1181),
    ("3", 14.75, "PDC", 1181, 4353), ("4", 14.75, "PDC", 4353, 4845),
    ("5", 9.5, "TCI", 4845, 4855), ("6", 8.75, "Core", 4855, 4871), ("7", 8.75, "Core", 4871, 4878),
    ("8", 9.5, "Particle", 4878, 4910), ("9", 9.5, "Particle", 4910, 4978), ("5/R2", 9.5, "TCI", 4978, 4980),
    ("10", 9.5, "PDC", 4980, 5269), ("10/R2", 9.5, "PDC", 5269, 5957), ("11", 9.5, "PDC", 5957, 6545),
    ("12", 9.5, "PDC", 6545, 6610), ("13", 9.5, "PDC", 6610, 6950), ("14", 9.5, "PDC", 6950, 7584),
    ("15", 9.5, "PDC", 7584, 8085), ("16", 9.5, "PDC", 8085, 8585), ("17", 9.5, "PDC", 8585, 9255),
    ("12/R2", 9.5, "PDC", 9255, 9800), ("18", 8.75, "Core", 9800, 9817), ("19", 8.75, "TCI", 9817, 9823),
    ("20", 8.75, "Core", 9823, 9853), ("21", 9.5, "TCI", 9853, 9863), ("22", 9.5, "PDC", 9863, 10250),
    ("23", 8.75, "Core", 10250, 10256), ("24", 8.75, "TCI", 10256, 10264), ("25", 8.75, "Core", 10264, 10271),
    ("24/R2", 8.75, "TCI", 10271, 10274), ("26", 8.75, "Core", 10274, 10304), ("27", 8.75, "PDC", 10304, 10430),
    ("28", 8.75, "Core", 10430, 10460), ("24/R3", 8.75, "TCI", 10460, 10462), ("29", 8.75, "Core", 10462, 10493),
    ("21/R2", 9.5, "TCI", 10493, 10503), ("30", 9.5, "PDC", 10503, 10947),
]
EOWR_TD_FT = 10947.0


def assign_bit(md):
    """Bit run for each hole depth: in < md <= out; past TD -> last run."""
    edges = np.array([r[4] for r in BIT_RUNS], dtype=float)
    i = np.searchsorted(edges, md, side="left").clip(0, len(BIT_RUNS) - 1)
    lab = np.array([r[0] for r in BIT_RUNS], dtype=object)[i]
    size = np.array([r[1] for r in BIT_RUNS])[i]
    typ = np.array([r[2] for r in BIT_RUNS], dtype=object)[i]
    return lab, size, typ

# raw Pason column -> short name
KEEP = {
    "timestamp": "timestamp",
    "hole_depth_feet": "hole_depth_ft",
    "bit_depth_feet": "bit_depth_ft",
    "on_bottom_unitless": "on_bottom",
    "weight_on_bit_klbs": "wob_klbs",
    "rotary_rpm_rpm": "rpm_surface",
    "top_drive_rotary_rpm": "rpm_topdrive",
    "bit_rpm_rpm": "rpm_bit",
    "top_drive_torque_kft_lb": "torque_kftlb",
    "convertible_torque_kft_lb": "torque_conv_kftlb",
    "on_bottom_rop_ft_per_hr": "rop_pason_fph",
    "differential_pressure_psi": "diffp_psi",
    "standpipe_pressure_psi": "spp_psi",
    "total_pump_output_gal_per_min": "pump_gpm",
    "flow_flow_percent": "flow_out_pct",
    "flow_1_gain_loss_percent": "flow_gl_pct",
    "pvt_total_mud_gain_loss_barrels": "pvt_gl_bbl",
    "total_mud_volume_barrels": "mud_vol_bbl",
    "temp_in_manifold_degrees": "temp_in_f",
    "temp_out_flow_degrees": "temp_out_f",
    "hole_diameter_in": "bit_dia_in",
    "mud_motor_used_unitless": "mud_motor",
    "dexponent_log": "dexp",
    "mechanical_specific_energy_ksi": "mse_pason_ksi",
    "total_mse_ksi": "mse_total_pason_ksi",
    "surface_stick_slip_index_percent": "stick_slip_pct",
    "hook_load_klbs": "hookload_klbs",
    "inclination_degrees": "inc_deg",
    "16a_wellhd_psi_psi": "whp_16a_psi",
    "motor_rpm_rpm": "rpm_motor",
    "motor_max_diff_psi": "motor_max_diffp_psi",
    "motor_max_torque_ft_lb": "motor_max_torque_ftlb",
}

# Plausible ranges; outside -> NaN (Pason computed channels carry 1e12-style spikes)
VALID = {
    "temp_in_f": (32, 300), "temp_out_f": (32, 400),
    "mse_pason_ksi": (0, 2000), "mse_total_pason_ksi": (0, 2000),
    "dexp": (0, 5), "wob_klbs": (0, 150), "torque_kftlb": (0, 60),
    "bit_dia_in": (4, 30),
    "rop_pason_fph": (0, 2000),
}


def md_to_xyz(md_ft, sv):
    """Interpolate survey (MD ft) to UTM E/N m and elevation m; extrapolate the tail linearly."""
    md = sv["md_ft"].to_numpy()
    out = {}
    for c in ("utm_e_m", "utm_n_m", "elev_m", "inc_deg"):
        y = sv[c].to_numpy()
        v = np.interp(md_ft, md, y)
        tail = md_ft > md[-1]
        if tail.any():
            slope = (y[-1] - y[-2]) / (md[-1] - md[-2])
            v[tail] = y[-1] + slope * (md_ft[tail] - md[-1])
        out[c] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-ft", type=float, default=10.0)
    args = ap.parse_args()
    B = args.bin_ft
    out_bin = Path(str(OUT_BIN).format(f"{B:g}"))

    cols = [c for c in KEEP if c != "timestamp"]
    df = pd.read_parquet(SRC, columns=["timestamp"] + cols).rename(columns=KEEP)
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    n0 = len(df)
    print(f"Read {n0:,} rows  {df.timestamp.min()} .. {df.timestamp.max()}")

    for c, (lo, hi) in VALID.items():
        df[c] = df[c].where(df[c].between(lo, hi))

    runmax = df["hole_depth_ft"].cummax()
    on_bot = df["on_bottom"] == 1
    bit_ok = (df["hole_depth_ft"] - df["bit_depth_ft"]).abs() <= BIT_TOL_FT
    at_max = df["hole_depth_ft"] >= runmax - NEW_TOL_FT
    new = on_bot & bit_ok & at_max
    print(f"on_bottom: {on_bot.sum():,}   + bit at bottom: {(on_bot & bit_ok).sum():,}   "
          f"+ new hole: {new.sum():,}  ({new.sum()*DT_S/3600:.1f} h)")
    excluded = on_bot & bit_ok & ~at_max
    if excluded.any():
        days = df.loc[excluded, "timestamp"].dt.date.value_counts().sort_index()
        print("On-bottom rows above prior TD (repeat passes), by day:")
        for d, n in days.items():
            if n * DT_S / 3600 >= 0.5:
                print(f"  {d}  {n*DT_S/3600:5.1f} h")

    d = df[new].copy()
    lab, size, typ = assign_bit(d["hole_depth_ft"].to_numpy())
    d["bit_run"], d["bit_type"] = lab, typ
    d["bit_dia_pason_in"] = d["bit_dia_in"]
    d["bit_dia_in"] = size
    d["cored"] = d["bit_type"] == "Core"
    print(f"Pason max hole depth {d.hole_depth_ft.max():,.1f} ft vs EOWR TD {EOWR_TD_FT:,.0f} ft "
          f"(difference {d.hole_depth_ft.max() - EOWR_TD_FT:+.1f} ft)")
    mism = (d.bit_dia_pason_in - d.bit_dia_in).abs() > 0.1
    print(f"Rows where Pason hole diameter differs from bit record: {mism.sum():,} "
          f"(MD {d.loc[mism, 'hole_depth_ft'].min():,.0f}..{d.loc[mism, 'hole_depth_ft'].max():,.0f} ft)")

    # Teale MSE (psi -> ksi), surface only and surface + motor
    area = np.pi * (d["bit_dia_in"] / 2) ** 2
    rop = d["rop_pason_fph"].where(d["rop_pason_fph"] >= MIN_ROP)
    rpm_s = d["rpm_topdrive"].fillna(d["rpm_surface"])
    p_surf = rpm_s * d["torque_kftlb"] * 1e3                         # rpm * ft-lbf
    k_motor = (d["motor_max_torque_ftlb"] / d["motor_max_diffp_psi"]).where(d["motor_max_diffp_psi"] > 0)
    t_motor = (d["diffp_psi"].clip(lower=0) * k_motor).fillna(0.0)    # ft-lbf
    on_motor = d["mud_motor"] == 1
    p_motor = (d["rpm_motor"].fillna(0) * t_motor).where(on_motor, 0.0)
    d["torque_motor_ftlb"] = t_motor.where(on_motor)
    thrust = (d["wob_klbs"] * 1e3) / area
    d["mse_surface_ksi"] = (thrust + 120 * np.pi * p_surf / (area * rop)) / 1e3
    d["mse_ksi"] = (thrust + 120 * np.pi * (p_surf + p_motor) / (area * rop)) / 1e3
    d["motor_power_frac"] = (p_motor / (p_surf + p_motor)).where((p_surf + p_motor) > 0)
    d["temp_dt_f"] = d["temp_out_f"] - d["temp_in_f"]
    d.to_parquet(OUT_ROWS, index=False)
    print(f"Wrote {OUT_ROWS}  ({len(d):,} rows)")

    # MD bins
    d["md_bin_ft"] = np.floor(d["hole_depth_ft"] / B) * B
    med_cols = ["wob_klbs", "rpm_topdrive", "rpm_motor", "rpm_bit", "torque_kftlb", "torque_motor_ftlb",
                "rop_pason_fph", "diffp_psi", "spp_psi", "pump_gpm", "flow_out_pct", "flow_gl_pct",
                "temp_in_f", "temp_out_f", "temp_dt_f", "bit_dia_in", "dexp", "mse_ksi",
                "mse_surface_ksi", "motor_power_frac", "mse_pason_ksi", "mse_total_pason_ksi",
                "stick_slip_pct", "whp_16a_psi"]
    g = d.groupby("md_bin_ft")
    ft = g[med_cols].median()
    ft["n_samples"] = g.size()
    ft["hours_on_bottom"] = ft["n_samples"] * DT_S / 3600
    ft["ft_drilled"] = g["hole_depth_ft"].max() - g["hole_depth_ft"].min()
    ft["rop_avg_fph"] = ft["ft_drilled"] / ft["hours_on_bottom"]
    ft["pvt_change_bbl"] = g["pvt_gl_bbl"].last() - g["pvt_gl_bbl"].first()
    ft["mud_motor"] = g["mud_motor"].mean()
    ft["cored"] = g["cored"].mean() > 0.5
    ft["bit_run"] = g["bit_run"].agg(lambda x: x.mode().iat[0])
    ft["bit_type"] = g["bit_type"].agg(lambda x: x.mode().iat[0])
    ft["n_bit_runs"] = g["bit_run"].nunique()
    ft["t_first"] = g["timestamp"].min()
    ft = ft.reset_index()
    ft["md_mid_ft"] = ft["md_bin_ft"] + B / 2

    sv = pd.read_parquet(SURVEY)
    xyz = md_to_xyz(ft["md_mid_ft"].to_numpy(), sv)
    for c, v in xyz.items():
        ft[c] = v
    ft.to_parquet(out_bin, index=False)

    top, bot = ft.md_bin_ft.min(), ft.md_bin_ft.max()
    span = bot - top + B
    span = span / B
    print(f"Wrote {out_bin}  ({len(ft):,} {B:g}-ft bins, {top:.0f}..{bot+B:.0f} ft, "
          f"{100*len(ft)/span:.1f}% of span has data)")
    beyond = (ft.md_mid_ft > sv.md_ft.max()).sum()
    if beyond:
        print(f"  {beyond} bins below last survey station ({sv.md_ft.max():.0f} ft): position extrapolated")

    # Per-section summary
    edges = list(range(0, 11001, 1000))
    ft["section"] = pd.cut(ft.md_bin_ft, edges, right=False)
    for label, sub in (("All bins", ft), ("Excluding cored bins", ft[~ft.cored])):
        summ = sub.groupby("section", observed=True).agg(
            bins=("md_bin_ft", "size"), bit_in=("bit_dia_in", "median"),
            hours=("hours_on_bottom", "sum"), rop_avg=("rop_avg_fph", "median"),
            rop_pason=("rop_pason_fph", "median"), mse_surf=("mse_surface_ksi", "median"),
            mse=("mse_ksi", "median"), motor_pwr=("motor_power_frac", "median"),
            mse_pason=("mse_pason_ksi", "median"), mse_tot_pason=("mse_total_pason_ksi", "median"),
            tin_f=("temp_in_f", "median"), dt_f=("temp_dt_f", "median"),
            motor=("mud_motor", "mean"), cored=("cored", "sum"),
        )
        print(f"\n{label}:")
        with pd.option_context("display.width", 250, "display.float_format", "{:.1f}".format):
            print(summ.to_string())
    m = d[d.mud_motor == 1]
    if len(m):
        r = (m.rpm_bit - m.rpm_topdrive.fillna(m.rpm_surface) - m.rpm_motor).abs()
        print(f"\nCheck on motor rows: median |bit rpm - (surface + motor rpm)| = {r.median():.1f} rpm  "
              f"(0 means Pason bit RPM = surface + motor)")
        print(f"Motor rated torque/diffP, median: {(m.motor_max_torque_ftlb/m.motor_max_diffp_psi).median():.2f} ft-lbf/psi")


if __name__ == "__main__":
    main()
