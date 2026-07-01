"""
ml/longitudinal.py — Season-level longitudinal analytics per athlete.

Provides:
  • weekly_summary(player_id)  → weekly agg of risk, fatigue, ACWR, perf
  • season_chart(ax, summary)  → multi-line season chart on a mpl Axes
"""
from __future__ import annotations
import sqlite3
from typing import Optional
import numpy as np
import pandas as pd
import joblib

from ml.injury_model import (
    build_features, FEATURE_COLUMNS, SPORT_MODELS_DIR, load_sport_model,
)

DB_PATH = "data/athlete.db"
MODEL_PATH = "ml/injury_model.pkl"


def weekly_summary(player_id: int, db_path: str = DB_PATH) -> pd.DataFrame:
    """
    Return a weekly aggregated DataFrame for one player with columns:
      week, sessions, injury_prob_mean, fatigue_mean, acwr_mean,
      perf_index_mean, injury_count, load_sum
    """
    df = build_features(db_path)
    ph = df[df["player_id"] == player_id].copy()
    if ph.empty:
        return pd.DataFrame()

    model = joblib.load(MODEL_PATH)
    X = ph[FEATURE_COLUMNS].fillna(ph[FEATURE_COLUMNS].median())
    ph["injury_prob"] = model.predict_proba(X)[:, 1]

    ph["week"] = ph["date"].dt.to_period("W").apply(lambda p: p.start_time)
    weekly = ph.groupby("week").agg(
        sessions       = ("injury_prob", "count"),
        injury_prob    = ("injury_prob", "mean"),
        fatigue        = ("fatigue", "mean"),
        acwr           = ("acwr", "mean"),
        perf_index     = ("perf_index", "mean"),
        injury_count   = ("injury_label", "sum"),
        load_sum       = ("player_load", "sum"),
    ).reset_index()
    return weekly


def season_chart(ax_risk, ax_load, summary: pd.DataFrame,
                 player_name: str = "") -> None:
    """
    Plot the season dashboard onto two matplotlib Axes.
      ax_risk: top chart — injury probability + ACWR
      ax_load: bottom chart — weekly load + fatigue
    """
    if summary.empty:
        ax_risk.text(0.5, 0.5, "Pas de données", transform=ax_risk.transAxes,
                     ha="center", va="center", color="#8B97A8")
        return

    weeks = summary["week"]
    x     = np.arange(len(weeks))

    # ── Top: risk probability + ACWR ──
    ax_risk.clear()
    ax_risk.set_facecolor("#0B0F14")
    # Risk zones
    ax_risk.axhspan(0.66, 1.0, color="#EF4444", alpha=0.08)
    ax_risk.axhspan(0.33, 0.66, color="#F59E0B", alpha=0.07)
    ax_risk.axhspan(0.0, 0.33, color="#22C55E", alpha=0.06)

    ax_risk.plot(x, summary["injury_prob"], color="#EF4444", lw=2.2,
                 label="Risque moyen (%)", marker="o", markersize=3)
    ax_risk.fill_between(x, 0, summary["injury_prob"],
                         color="#EF4444", alpha=0.18)

    ax2 = ax_risk.twinx()
    ax2.plot(x, summary["acwr"], color="#F59E0B", lw=1.5, ls="--",
             label="ACWR", marker="s", markersize=2.5)
    ax2.axhline(1.3, color="#F59E0B", alpha=0.4, lw=0.8, ls=":")
    ax2.axhline(0.8, color="#22C55E", alpha=0.4, lw=0.8, ls=":")
    ax2.set_ylabel("ACWR", color="#F59E0B", fontsize=9)
    ax2.tick_params(colors="#F59E0B")
    ax2.set_ylim(0, 3.0)

    # Injury events
    inj_mask = summary["injury_count"] > 0
    ax_risk.scatter(x[inj_mask], summary["injury_prob"][inj_mask],
                    color="#EF4444", s=60, zorder=5, marker="X",
                    label="Session blessure")

    ax_risk.set_ylabel("Prob. blessure", color="#E6EDF3", fontsize=9)
    ax_risk.set_ylim(0, 1.05)
    ax_risk.set_xlim(-0.5, len(x) - 0.5)
    ax_risk.set_xticks([])
    ax_risk.set_title(f"Saison — {player_name} | risque & ACWR hebdomadaires",
                      color="#E6EDF3", fontsize=10, fontweight="bold", pad=4)
    lines1, labels1 = ax_risk.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax_risk.legend(lines1+lines2, labels1+labels2, fontsize=8,
                   loc="upper left", framealpha=0.3,
                   facecolor="#1E2A42", edgecolor="#243042", labelcolor="white")
    ax_risk.tick_params(colors="#8B97A8")
    ax_risk.spines[:].set_edgecolor("#243042")

    # ── Bottom: load + fatigue ──
    ax_load.clear()
    ax_load.set_facecolor("#0B0F14")
    bar_colors = [
        "#EF4444" if r >= 0.66 else "#F59E0B" if r >= 0.33 else "#22C55E"
        for r in summary["injury_prob"]
    ]
    ax_load.bar(x, summary["load_sum"], color=bar_colors, alpha=0.7,
                width=0.7, label="Charge hebdo (Player Load)")
    ax3 = ax_load.twinx()
    ax3.plot(x, summary["fatigue"], color="#A78BFA", lw=1.8, marker="^",
             markersize=3, label="Fatigue moyenne")
    ax3.set_ylabel("Fatigue", color="#A78BFA", fontsize=9)
    ax3.tick_params(colors="#A78BFA")

    week_labels = [w.strftime("%d %b") for w in weeks]
    ax_load.set_xticks(x)
    ax_load.set_xticklabels(week_labels, rotation=45, ha="right",
                            fontsize=7, color="#8B97A8")
    ax_load.set_ylabel("Player Load total", color="#E6EDF3", fontsize=9)
    ax_load.set_title("Charge hebdomadaire & fatigue",
                      color="#E6EDF3", fontsize=10, fontweight="bold", pad=4)
    ax_load.tick_params(colors="#8B97A8")
    ax_load.spines[:].set_edgecolor("#243042")
    lines4, labels4 = ax_load.get_legend_handles_labels()
    lines5, labels5 = ax3.get_legend_handles_labels()
    ax_load.legend(lines4+lines5, labels4+labels5, fontsize=8,
                   loc="upper left", framealpha=0.3,
                   facecolor="#1E2A42", edgecolor="#243042", labelcolor="white")
