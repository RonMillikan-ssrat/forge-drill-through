"""Shared settings for the FORGE Drill-Through project. Lives at the repo root."""
from pathlib import Path

GCP_PROJECT = "forge-drill-through"
GCS_BUCKET = "forge-drill-through-data"
BQ_DATASET = "forge"
BQ_LOCATION = "us-central1"

REPO = Path(__file__).resolve().parent          # /home/ron/devl/forge
RAW = REPO / "raw"
RAW_16B = RAW / "16B"
OUT = REPO / "processed"                        # script outputs (git-ignored)

SURVEY_16B_TXT = RAW_16B / "survey" / "16B(78)-32 Final Survey Report.txt"

# US survey foot -> metre
USFT_TO_M = 1200.0 / 3937.0
# Foot used for the survey's map coordinates (UTM easting/northing).
# Confirmed US survey feet: the Geographic report's site lat/lon (38.50017100, -112.88221000)
# matches its grid coords to <1 cm with 1200/3937; the international foot is off by 8.5 m north.
# The End of Well Report's UTM metres used the international foot and are ~8.5 m south.
MAP_FT_TO_M = USFT_TO_M
# International foot -> metre (Pason depths are rig measured depth in feet;
# over ~11,000 ft the two feet differ by ~2 cm, far below catalog location error)
FT_TO_M = 0.3048
