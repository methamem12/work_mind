"""
ml/csv_import.py — CSV/drag-drop import pipeline.

Accepts a CSV exported from any GPS platform (Catapult, STATSports, Polar,
or a generic export) and normalises column names to the internal schema.

Usage:
    importer = CsvImporter()
    df, warnings = importer.load("export.csv")
    # df has exactly the columns in RAW_FEATURE_COLUMNS (NaN where missing)
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ml.injury_model import RAW_FEATURE_COLUMNS

# ── Column alias map (external name → internal name) ──────────────────────────
# Covers the most common GPS/wearable export column names.
COLUMN_ALIASES: Dict[str, str] = {
    # Distance
    "total_distance": "distance_km",
    "total distance": "distance_km",
    "distance":       "distance_km",
    "dist_km":        "distance_km",
    "distance (km)":  "distance_km",
    # Sprints
    "sprint_count":   "sprints_count",
    "num_sprints":    "sprints_count",
    "sprints":        "sprints_count",
    "sprint count":   "sprints_count",
    # HID
    "hi_distance":    "hid_km",
    "high_intensity_distance": "hid_km",
    "hid":            "hid_km",
    "high intensity distance": "hid_km",
    # Acceleration
    "max_accel":      "acceleration_max",
    "peak_acceleration": "acceleration_max",
    "max acceleration": "acceleration_max",
    # Player load
    "load":           "player_load",
    "pl":             "player_load",
    "playerload":     "player_load",
    "player load":    "player_load",
    # ACWR
    "acute_chronic":  "acwr",
    "a:c ratio":      "acwr",
    "ac_ratio":       "acwr",
    # HRV
    "hrv":            "hrv_rmssd",
    "rmssd":          "hrv_rmssd",
    "hrv_ms":         "hrv_rmssd",
    # HR
    "resting_hr":     "fc_repos",
    "rest_hr":        "fc_repos",
    "resting hr":     "fc_repos",
    "hr_rest":        "fc_repos",
    # Sleep
    "sleep_hours":    "sommeil_h",
    "sleep hours":    "sommeil_h",
    "sleep_duration": "sommeil_h",
    "sleep quality":  "sommeil_qualite",
    "sleep_score":    "sommeil_qualite",
    # Fatigue / wellness
    "fatigue_score":  "fatigue",
    "perceived_fatigue": "fatigue",
    "wellness":       "wellness_score",
    "wellness score": "wellness_score",
    # Hydration
    "hydration":      "hydratation_score",
    "hydration_score": "hydratation_score",
    # Stress
    "stress_score":   "stress",
    "mental_load":    "charge_mentale",
    # VO2
    "vo2_max":        "vo2max",
    "vo2max (ml/kg/min)": "vo2max",
    # CMJ
    "jump_height":    "cmj_cm",
    "cmj":            "cmj_cm",
    # Force
    "peak_force":     "force_n",
    "max_force":      "force_n",
    # CK
    "creatine_kinase": "ck_post",
    "ck":             "ck_post",
    "ck (u/l)":       "ck_post",
    # SpO2
    "spo2 (%)":       "spo2",
    "oxygen_saturation": "spo2",
    # Motivation / reaction
    "motivation_score": "motivation",
    "reaction_time":  "reaction_ms",
    "reaction time (ms)": "reaction_ms",
    # Perf index
    "performance_index": "perf_index",
    "perf":           "perf_index",
    # Days since match
    "days_since_match": "jours_depuis_match",
    "days since match": "jours_depuis_match",
}


def _normalise_col(name: str) -> str:
    """Lower, strip, collapse whitespace."""
    return re.sub(r"\s+", " ", str(name).strip().lower())


class CsvImporter:
    """Load, normalise and validate a GPS/wearable CSV export."""

    def load(self, path: str | Path) -> Tuple[pd.DataFrame, List[str]]:
        """
        Returns:
            df       : DataFrame with columns = RAW_FEATURE_COLUMNS (+ date/player_id if present)
            warnings : list of human-readable warning strings
        """
        path = Path(path)
        ext  = path.suffix.lower()

        if ext == ".csv":
            raw = pd.read_csv(path, skipinitialspace=True)
        elif ext in (".xlsx", ".xls"):
            raw = pd.read_excel(path)
        else:
            raise ValueError(f"Format non supporté : {ext}. Utilisez .csv ou .xlsx.")

        warns: List[str] = []
        # Normalise column names
        raw.columns = [_normalise_col(c) for c in raw.columns]

        # Apply alias map
        rename = {}
        for col in list(raw.columns):
            if col in COLUMN_ALIASES:
                rename[col] = COLUMN_ALIASES[col]
        raw = raw.rename(columns=rename)

        # Build output with all RAW_FEATURE_COLUMNS
        out = pd.DataFrame(index=raw.index)

        # Carry over date & player columns if present
        for meta in ("date", "player_id", "player", "player_name",
                     "sport", "session_type"):
            if meta in raw.columns:
                out[meta] = raw[meta]

        missing = []
        for col in RAW_FEATURE_COLUMNS:
            if col in raw.columns:
                out[col] = pd.to_numeric(raw[col], errors="coerce")
            else:
                out[col] = np.nan
                missing.append(col)

        if missing:
            warns.append(
                f"{len(missing)} colonne(s) absente(s) dans le fichier et remplies "
                f"avec NaN : {', '.join(missing[:8])}"
                + (" …" if len(missing) > 8 else "")
            )

        # Check for mostly-NaN columns
        nan_pct = out[RAW_FEATURE_COLUMNS].isna().mean()
        bad = nan_pct[nan_pct > 0.5].index.tolist()
        if bad:
            warns.append(
                f"⚠ {len(bad)} colonne(s) avec >50 % de NaN "
                f"(les valeurs médianes seront utilisées) : {', '.join(bad[:5])}"
            )

        # Clip obvious outliers
        CLIPS = {
            "acwr":        (0.0, 3.0),
            "hrv_rmssd":   (10.0, 200.0),
            "fc_repos":    (30.0, 120.0),
            "sommeil_h":   (0.0, 14.0),
            "fatigue":     (0.0, 10.0),
            "spo2":        (70.0, 100.0),
            "stress":      (0.0, 10.0),
            "hydratation_score": (0.0, 10.0),
        }
        for col, (lo, hi) in CLIPS.items():
            if col in out.columns:
                out[col] = out[col].clip(lo, hi)

        return out, warns

    @staticmethod
    def column_report(df: pd.DataFrame) -> pd.DataFrame:
        """Return a summary of which columns are populated vs. missing."""
        coverage = (df[RAW_FEATURE_COLUMNS].notna().mean() * 100).round(1)
        report = pd.DataFrame({
            "Colonne":    RAW_FEATURE_COLUMNS,
            "Couverture": coverage.values,
            "Statut":     ["✅" if c >= 80 else "⚠" if c >= 40 else "❌"
                           for c in coverage.values],
        })
        return report.sort_values("Couverture", ascending=False)
