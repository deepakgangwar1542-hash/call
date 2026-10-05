"""Shared config, STATS19 code maps and metric helpers."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
RAW_CSV = ROOT / "data/raw/dft-road-casualty-statistics-collision-2023.csv"
CLEAN_CSV = ROOT / "data/processed/clean_2023.csv"
MODEL_PATH = ROOT / "models/best_model.joblib"
META_PATH = ROOT / "models/best_model_meta.json"
DB_PATH = ROOT / "calls.db"

K_DEFAULT = 300          # shift capacity K (enhanced responses per shift) - change as you like
SEED = 42

# 5 usable features (known at call time) -> SRS FR-1.1
FEATURES = ["road_type", "speed_limit", "weather", "light_condition", "junction_detail"]
VALID_SPEED = [20, 30, 40, 50, 60, 70]

# STATS19 code -> readable label ("Unknown" kept as an explicit category, not dropped)
ROAD_TYPE = {1: "Roundabout", 2: "One way street", 3: "Dual carriageway", 6: "Single carriageway",
             7: "Slip road", 12: "One way street/Slip road", 9: "Unknown", -1: "Unknown"}
WEATHER = {1: "Fine, no high winds", 2: "Raining, no high winds", 3: "Snowing, no high winds",
           4: "Fine, high winds", 5: "Raining, high winds", 6: "Snowing, high winds",
           7: "Fog or mist", 8: "Other", 9: "Unknown", -1: "Unknown"}
LIGHT = {1: "Daylight", 4: "Darkness - lights lit", 5: "Darkness - lights unlit",
         6: "Darkness - no lighting", 7: "Darkness - lighting unknown", -1: "Unknown"}
# 2023 file uses the NEW junction coding (STATS20 2024 spec). Code 0 also covers roundabout, mini-roundabout
# and slip-road collisions (old codes 1, 2, 5 were merged into 0); -1 is the old "99 Unknown".
JUNCTION = {0: "Not at junction / roundabout / slip road", 13: "T or staggered junction", 16: "Crossroads",
            17: "More than 4 arms (not roundabout)", 18: "Private drive or entrance",
            19: "Other junction", 99: "Unknown", -1: "Unknown"}

CHOICES = {  # dropdown options for the UI / validation
    "road_type": sorted(set(ROAD_TYPE.values())),
    "weather": sorted(set(WEATHER.values())),
    "light_condition": sorted(set(LIGHT.values())),
    "junction_detail": sorted(set(JUNCTION.values())),
}


def precision_at_k(y_true, scores, k, seed=SEED):
    """Of the top-k rows ranked by score, fraction that are truly 1.
    Ties are broken randomly (important for baselines that give many equal scores)."""
    y_true, scores = np.asarray(y_true), np.asarray(scores, dtype=float)
    if k <= 0:
        raise ValueError("K must be > 0")
    k = min(k, len(y_true))
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(len(scores)), -scores))   # sort by -score, random tie-break
    return float(y_true[order[:k]].mean())


def load_raw():
    """Load STATS19 CSV; supports old (accident_*) and new 2023 (collision_*) column names."""
    df = pd.read_csv(RAW_CSV, low_memory=False)
    df = df.rename(columns={"collision_severity": "accident_severity",
                            "collision_index": "accident_index",
                            "collision_year": "accident_year"})
    return df