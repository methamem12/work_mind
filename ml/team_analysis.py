"""
ml/team_analysis.py — Batch team risk analysis + heatmap renderer.

Provides:
  • run_team_analysis(db_path, model, feature_names) → DataFrame
  • render_heatmap(ax, df, title)
"""
from __future__ import annotations
import sqlite3
from typing import Optional
import numpy as np
import pandas as pd
import joblib

from ml.injury_model import (
    build_features, FEATURE_COLUMNS, load_sport_model,
)

DB_PATH    = "data/athlete.db"
MODEL_PATH = "ml/injury_model.pkl"

BODY_ZONES = [
    "head","chest","abdomen","lower_back","shoulders",
    "left_quad","right_quad","left_hamstring","right_hamstring",
    "left_knee","right_knee","left_calf","right_calf",
    "left_ankle","right_ankle",
]


def run_team_analysis(
    db_path: str = DB_PATH,
    global_model=None,
    sport_filter: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute current injury risk + 7-day trend for every player.
    Returns a DataFrame sorted by descending risk with columns:
      player_id, name, sport, injury_prob, risk_level,
      trend_7d, trend_arrow, fatigue, acwr, last_date
    
    Optimized: vectorized predictions, model caching, minimal filtering.
    """
    if global_model is None:
        global_model = joblib.load(MODEL_PATH)

    df = build_features(db_path)
    df = df.dropna(subset=["distance_km", "player_load", "acwr"])

    with sqlite3.connect(db_path) as cx:
        players = pd.read_sql(
            "SELECT id, name, sport FROM players ORDER BY name", cx)

    if sport_filter and sport_filter != "Tous":
        players = players[players["sport"] == sport_filter]

    # Cache sport models to avoid repeated disk I/O
    sport_model_cache = {}
    for sport in players["sport"].unique():
        model = load_sport_model(sport)
        sport_model_cache[sport] = model or global_model

    results = []
    player_group = df.groupby("player_id", sort=False)
    
    for _, row in players.iterrows():
        pid   = int(row["id"])
        pname = str(row["name"])
        sport = str(row["sport"])

        if pid not in player_group.groups:
            continue
            
        ph = player_group.get_group(pid).sort_values("date")
        model = sport_model_cache[sport]

        # Vectorize all predictions at once (last + recent + prior)
        indices_to_pred = []
        pred_indices = {}
        
        # Latest
        pred_indices["last"] = len(indices_to_pred)
        indices_to_pred.append(-1)
        
        # Last 7 days
        if len(ph) >= 7:
            pred_indices["recent_start"] = len(indices_to_pred)
            pred_indices["recent_count"] = min(7, len(ph))
            indices_to_pred.extend(range(-7, 0))
            
            # Prior 7 days
            if len(ph) >= 14:
                pred_indices["prior_start"] = len(indices_to_pred)
                pred_indices["prior_count"] = 7
                indices_to_pred.extend(range(-14, -7))

        # Extract all rows for prediction at once
        X_all = ph.iloc[indices_to_pred][FEATURE_COLUMNS].fillna(
            ph[FEATURE_COLUMNS].median())
        
        # Single batch prediction call
        proba_all = model.predict_proba(X_all)[:, 1]
        
        # Extract results from batch
        prob_now = float(proba_all[pred_indices["last"]])
        
        trend = 0.0
        if len(ph) >= 7:
            p_rec = float(proba_all[
                pred_indices["recent_start"]:
                pred_indices["recent_start"] + pred_indices["recent_count"]
            ].mean())
            if len(ph) >= 14:
                p_pri = float(proba_all[
                    pred_indices["prior_start"]:
                    pred_indices["prior_start"] + pred_indices["prior_count"]
                ].mean())
            else:
                p_pri = p_rec
            trend = p_rec - p_pri

        arrow = "↑" if trend > 0.03 else ("↓" if trend < -0.03 else "→")

        level = ("🔴 ÉLEVÉ" if prob_now >= 0.66
                 else "🟡 MODÉRÉ" if prob_now >= 0.33
                 else "🟢 FAIBLE")

        last = ph.iloc[[-1]]
        results.append({
            "player_id":   pid,
            "name":        pname,
            "sport":       sport,
            "injury_prob": prob_now,
            "risk_level":  level,
            "trend_7d":    trend,
            "trend_arrow": arrow,
            "fatigue":     float(last["fatigue"].iloc[0]) if "fatigue" in last else np.nan,
            "acwr":        float(last["acwr"].iloc[0])    if "acwr"    in last else np.nan,
            "last_date":   str(last["date"].iloc[0])[:10] if "date"    in last else "",
        })

    out = pd.DataFrame(results).sort_values("injury_prob", ascending=False)
    return out.reset_index(drop=True)


def render_heatmap(ax, team_df: pd.DataFrame, title: str = "Risque équipe") -> None:
    """
    Draw a player × metric heatmap on `ax`.
    Rows = players (sorted by risk desc), cols = key metrics.
    """
    ax.clear()
    ax.set_facecolor("#0B0F14")

    if team_df.empty:
        ax.text(0.5, 0.5, "Aucune donnée équipe",
                transform=ax.transAxes, ha="center", va="center",
                color="#8B97A8", fontsize=13)
        ax.set_axis_off()
        return

    metrics    = ["injury_prob", "fatigue", "acwr", "trend_7d"]
    col_labels = ["Risque (%)", "Fatigue", "ACWR", "Tendance 7j"]
    players    = team_df["name"].tolist()

    # Build matrix, normalise each column 0-1 for colour
    mat = team_df[metrics].fillna(0.0).values.astype(float)
    mat_norm = mat.copy()
    for j in range(mat.shape[1]):
        col = mat[:, j]
        rng = col.max() - col.min()
        mat_norm[:, j] = (col - col.min()) / rng if rng > 1e-9 else col * 0

    # Custom diverging palette: green → amber → red
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "risk", ["#22C55E", "#F59E0B", "#EF4444"])

    im = ax.imshow(mat_norm, aspect="auto", cmap=cmap,
                   vmin=0, vmax=1, interpolation="nearest")

    # Annotations
    for i in range(len(players)):
        for j in range(len(metrics)):
            raw = mat[i, j]
            if metrics[j] == "injury_prob":
                txt = f"{raw:.0%}"
            elif metrics[j] == "trend_7d":
                txt = f"{raw:+.1%}"
            elif metrics[j] == "acwr":
                txt = f"{raw:.2f}"
            else:
                txt = f"{raw:.1f}"
            brightness = mat_norm[i, j]
            text_color = "#0B0F14" if 0.3 < brightness < 0.85 else "#E6EDF3"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=9, color=text_color, fontweight="bold")

    # Labels
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(col_labels, fontsize=10, color="#E6EDF3", fontweight="bold")
    ax.set_yticks(range(len(players)))
    ax.set_yticklabels(players, fontsize=9, color="#E6EDF3")
    ax.tick_params(length=0)
    ax.set_title(title, color="#E6EDF3", fontsize=12,
                 fontweight="bold", pad=8)
    ax.spines[:].set_visible(False)

    # Risk-level colour strip on left margin
    for i, row in team_df.iterrows():
        prob = row["injury_prob"]
        strip_c = ("#EF4444" if prob >= 0.66
                   else "#F59E0B" if prob >= 0.33
                   else "#22C55E")
        ax.add_patch(__import__("matplotlib").patches.Rectangle(
            (-0.5 - 0.18, i - 0.5), 0.15, 1.0,
            transform=ax.transData, color=strip_c, clip_on=False))
