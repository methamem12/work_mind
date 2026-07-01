"""
ml/comparison.py — Session/player comparison utilities.

Two comparison modes:
  1. compare_sessions(s1, s2)   — diff two specific sessions of the SAME player,
                                   with SHAP delta per feature.
  2. compare_players_season(p1_id, p2_id) — overlay weekly risk curves of TWO
                                   different players across the season.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import pandas as pd
import joblib

from ml.injury_model import FEATURE_COLUMNS, RAW_FEATURE_COLUMNS, build_features
from ml.longitudinal import weekly_summary

MODEL_PATH = "ml/injury_model.pkl"


@dataclass
class FeatureDelta:
    feature: str
    label: str
    value_a: float
    value_b: float
    delta: float
    shap_a: float
    shap_b: float
    shap_delta: float


@dataclass
class SessionComparison:
    player_id: int
    date_a: str
    date_b: str
    prob_a: float
    prob_b: float
    deltas: List[FeatureDelta]


def compare_sessions(
    player_id: int,
    date_a: str,
    date_b: str,
    db_path: str = "data/athlete.db",
) -> SessionComparison:
    """Compare two sessions of the same player with a SHAP delta breakdown."""
    import shap
    from ml.performance_advisor import FEATURE_LABELS_FR

    df = build_features(db_path)
    ph = df[df["player_id"] == player_id].copy()
    ph["date_str"] = ph["date"].dt.strftime("%Y-%m-%d")

    row_a = ph[ph["date_str"] == date_a]
    row_b = ph[ph["date_str"] == date_b]
    if row_a.empty or row_b.empty:
        raise ValueError("Une des deux dates de séance est introuvable pour ce joueur.")

    model = joblib.load(MODEL_PATH)
    lr     = model.named_steps["clf"]
    scaler = model.named_steps["scaler"]
    imputer= model.named_steps["imputer"]

    X_a = row_a[FEATURE_COLUMNS].iloc[[0]]
    X_b = row_b[FEATURE_COLUMNS].iloc[[0]]

    Xa_imp = imputer.transform(X_a); Xa_sc = scaler.transform(Xa_imp)
    Xb_imp = imputer.transform(X_b); Xb_sc = scaler.transform(Xb_imp)

    explainer = shap.LinearExplainer(lr, Xa_sc, feature_names=FEATURE_COLUMNS)
    shap_a = explainer(Xa_sc).values[0]
    shap_b = explainer(Xb_sc).values[0]

    prob_a = float(model.predict_proba(X_a)[0, 1])
    prob_b = float(model.predict_proba(X_b)[0, 1])

    deltas = []
    for i, feat in enumerate(FEATURE_COLUMNS):
        va = float(X_a[feat].iloc[0])
        vb = float(X_b[feat].iloc[0])
        deltas.append(FeatureDelta(
            feature=feat, label=FEATURE_LABELS_FR.get(feat, feat),
            value_a=va, value_b=vb, delta=vb - va,
            shap_a=float(shap_a[i]), shap_b=float(shap_b[i]),
            shap_delta=float(shap_b[i] - shap_a[i]),
        ))
    deltas.sort(key=lambda d: -abs(d.shap_delta))

    return SessionComparison(
        player_id=player_id, date_a=date_a, date_b=date_b,
        prob_a=prob_a, prob_b=prob_b, deltas=deltas,
    )


def compare_players_season(player_a_id: int, player_b_id: int) -> dict:
    """
    Return weekly summaries for two players, aligned by week index
    (not calendar date) so seasons of different length can still overlay.
    """
    wa = weekly_summary(player_a_id)
    wb = weekly_summary(player_b_id)
    return {"player_a": wa, "player_b": wb}


def render_comparison_chart(ax, comparison: dict, name_a: str, name_b: str) -> None:
    """Overlay two players' weekly injury risk curves on one matplotlib Axes."""
    ax.clear()
    ax.set_facecolor("#0B0F14")

    wa = comparison["player_a"]
    wb = comparison["player_b"]

    if wa.empty and wb.empty:
        ax.text(0.5, 0.5, "Pas de données", transform=ax.transAxes,
                ha="center", va="center", color="#8B97A8")
        return

    if not wa.empty:
        xa = np.arange(len(wa))
        ax.plot(xa, wa["injury_prob"], color="#3B82F6", lw=2.2, marker="o",
                markersize=3, label=f"{name_a} — risque")
        ax.fill_between(xa, 0, wa["injury_prob"], color="#3B82F6", alpha=0.12)

    if not wb.empty:
        xb = np.arange(len(wb))
        ax.plot(xb, wb["injury_prob"], color="#F59E0B", lw=2.2, marker="s",
                markersize=3, label=f"{name_b} — risque")
        ax.fill_between(xb, 0, wb["injury_prob"], color="#F59E0B", alpha=0.12)

    ax.axhspan(0.66, 1.0, color="#EF4444", alpha=0.05)
    ax.axhspan(0.33, 0.66, color="#F59E0B", alpha=0.04)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Probabilité de blessure", color="#E6EDF3", fontsize=9)
    ax.set_xlabel("Semaine de saison", color="#E6EDF3", fontsize=9)
    ax.set_title(f"Comparaison saison — {name_a} vs {name_b}",
                color="#E6EDF3", fontsize=11, fontweight="bold", pad=6)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.3,
             facecolor="#1E2A42", edgecolor="#243042", labelcolor="white")
    ax.tick_params(colors="#8B97A8")
    ax.spines[:].set_edgecolor("#243042")
