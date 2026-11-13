# FORGE Drill-Through

Did the drill bit that cut Utah FORGE production well **16B(78)-32** (April–June 2023) "feel" the
microseismic cloud left by the April 2022 stimulation of injection well **16A(78)-32**?

Drilling response along 16B (mechanical specific energy, rate of penetration, mud temperature
rise, mud volume change) is compared with the density of 2022 microseismic events around each
10-ft interval of the well. 16A, drilled in 2020 before any stimulation, is used as a
pre-stimulation control.

## Result

**No detectable relationship.** 16B passes through the event cloud for about 1,100 ft
(9,700–10,700 ft MD, nearest event 8–41 m), but drilling response does not track event density.

| Test | Bins | MSE | ROP | Mud ΔT |
|---|---|---|---|---|
| Single bit run (Baker D406VX, 9,863–10,250 ft), in vs out of cloud at 25 m | 15 vs 19 | 75 vs 77 ksi, p 0.84 | 193 vs 223 ft/hr, p 0.55 | p 0.95 |
| All PDC runs below 8,500 ft, within-run z-scores, R = 25/50/100 m | 223 | \|ρ\| ≤ 0.14, p ≥ 0.26 | ρ −0.10 to −0.18, p ≥ 0.12 | \|ρ\| ≤ 0.08, p ≥ 0.52 |
| Same, after removing 16A's profile at the same position | 193 | p ≥ 0.57 | p ≥ 0.18 | – |

Spearman ρ against event count within R metres; p from circular shifts of the event-count series,
which keeps the along-hole autocorrelation of both series. Effective sample size is about 220 (MSE)
and 67 (ROP), so correlations of about 0.15 (MSE) or 0.25 (ROP) would have reached p < 0.05.

16A does not predict 16B's within-run profile (wells 104 m apart, 16B about 102 m above 16A;
different rig, bits and drilling practice; 16A ROP was 4–5× lower), so it adds no control.

## Pipeline

Run from the repo root. Raw data go in `raw/` and outputs in `processed/` (both git-ignored).

| Step | Script | Output |
|---|---|---|
| 1 | `ingest/survey_to_table.py` | 16B survey in UTM 12N m (US survey foot confirmed against the geographic report) |
| 2 | `ingest/pason_to_parquet.py` | 16B Pason 10-s data, all 412 channels, plus a channel report |
| 3 | `ingest/pason_drilling.py` | New-hole rows, bit runs and sizes from the End of Well Report, Teale MSE with mud-motor term, 10-ft bins |
| 4 | `ingest/catalog_uuss.py` | UUSS surface-network catalog (Niemz et al. 2024) and each event's closest approach to 16B |
| 5 | `ingest/catalog_ges.py` | GES downhole-geophone catalog of the 2022 stimulation, in the project frame |
| 6 | `ingest/survey_16a.py` | 16A survey, and the check-shot test of the catalog frame |
| 7 | `ingest/pason_to_parquet.py --src raw/16A/16A_time_data_10s.csv --name 16a` | 16A Pason data |
| 8 | `ingest/pason16a_drilling.py` | 16A new-hole bins, BHA runs from the daily operations summary |
| 9 | `analysis/drill_vs_cloud.py` | Event density per 16B bin; single-run test |
| 10 | `analysis/compare_16a_16b.py` | Whole-well test with and without the 16A control |

## Checks along the way

- **16B survey:** recomputed minimum curvature matches the report within 0.007 ft. The End of
  Well Report's UTM coordinates use the international foot and sit about 8.5 m south.
- **MSE:** the Teale MSE with a mud-motor term (motor torque = differential pressure × rated
  torque ÷ rated differential pressure) matches Pason's own Total MSE within 1–3 ksi per 1,000 ft.
- **Bit size:** Pason's hole-diameter setting lags bit changes by about one run. Sizes come from
  the bit record instead.
- **Catalog frame:** the three GES check shots fit the 16A survey best with no rotation (X/Y are
  grid east/north). They miss the trajectory by 6.7, 14 and 21 m, with vertical errors of opposite
  sign, so no single depth correction applies.
- **Depth:** Pason TD is 8.4 ft deeper than the bit record on 16B, and 1.3 ft on 16A.

## Data (Utah FORGE, DOE Geothermal Data Repository)

| Data | Source |
|---|---|
| 16B Pason, survey, End of Well Report | https://gdr.openei.org/submissions/1516 |
| GES 2022 stimulation event catalogs | https://gdr.openei.org/submissions/1558 (preliminary sheets used here) |
| UUSS surface-network catalogs | https://gdr.openei.org/submissions/1584 |
| 16A survey, Pason 10-s data, daily operations | https://gdr.openei.org/submissions/1283 |

## Open items

- Core fracture counts in the 16B cores (9,800–9,853 and 10,250–10,493 ft) vs event density.
- Robustness to event-location jitter and to the +8.4 ft depth offset.
- Published GES catalog (GDR 1558) vs the preliminary sheets.
- Rebuild on GCP: BigQuery, BigQuery ML, Colab Enterprise, Vertex/Agent Platform Pipelines.
