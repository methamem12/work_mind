"""
ml/rtp_protocol.py — Return-to-Play (RTP) protocol engine.

Generates a 5-week graduated load progression for a player coming back
from injury (injury_label == 1 on their most recent session).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import pandas as pd

# Standard RTP load progression (% of normal training load)
RTP_WEEKS = [20, 40, 60, 80, 100]

RTP_CRITERIA = {
    1: "Mobilité complète sans douleur · marche/jogging léger uniquement",
    2: "Course continue à allure modérée · pas de changement de direction",
    3: "Changements de direction à 70 % vitesse · exercices techniques individuels",
    4: "Entraînement collectif partiel · sprints à 90 % vitesse max",
    5: "Entraînement complet sans restriction · contact complet (sports collectifs)",
}


@dataclass
class RtpWeekStatus:
    week: int
    target_pct: int
    criteria: str
    status: str          # "completed" | "current" | "upcoming" | "flagged"
    actual_load_pct: Optional[float] = None


@dataclass
class RtpPlan:
    player_id: int
    is_injured: bool
    weeks: List[RtpWeekStatus] = field(default_factory=list)
    current_week: int = 0
    progress_pct: float = 0.0
    status_message: str = ""


def is_player_injured(history: pd.DataFrame, player_id: int) -> bool:
    """True if the player's most recent session has injury_label == 1."""
    ph = history[history["player_id"] == player_id].sort_values("date")
    if ph.empty:
        return False
    return bool(ph.iloc[-1].get("injury_label", 0) == 1)


def build_rtp_plan(
    history: pd.DataFrame,
    player_id: int,
    normal_load_baseline: Optional[float] = None,
) -> RtpPlan:
    """
    Build a 5-week RTP plan for a player, comparing their actual post-injury
    load trajectory against the standard 20/40/60/80/100% progression.
    """
    ph = history[history["player_id"] == player_id].sort_values("date").copy()
    if ph.empty:
        return RtpPlan(player_id=player_id, is_injured=False)

    injured = is_player_injured(history, player_id)
    if not injured:
        return RtpPlan(player_id=player_id, is_injured=False,
                       status_message="Joueur disponible — pas de protocole RTP actif.")

    # Find the date of the most recent injury session
    injury_rows = ph[ph["injury_label"] == 1]
    injury_date = injury_rows["date"].max()

    # Baseline = median load over the 8 weeks BEFORE injury
    if normal_load_baseline is None:
        pre = ph[ph["date"] < injury_date].tail(56)  # ~8 weeks daily
        normal_load_baseline = float(pre["player_load"].median()) if not pre.empty else \
                               float(ph["player_load"].median())
        if normal_load_baseline <= 0:
            normal_load_baseline = 1.0

    # Sessions since injury, grouped into 7-day windows
    post = ph[ph["date"] >= injury_date].copy()
    post["days_since"] = (post["date"] - injury_date).dt.days
    post["rtp_week"]   = (post["days_since"] // 7) + 1
    post = post[post["rtp_week"] >= 1]

    weeks: List[RtpWeekStatus] = []
    current_week = 1
    for i, target in enumerate(RTP_WEEKS, start=1):
        wk_data = post[post["rtp_week"] == i]
        if wk_data.empty:
            actual_pct = None
            status = "upcoming"
        else:
            actual_load = float(wk_data["player_load"].mean())
            actual_pct  = round(100 * actual_load / normal_load_baseline, 1)
            current_week = i
            # Flag if actual load deviates >15pp from target (too much or too little)
            if abs(actual_pct - target) > 15:
                status = "flagged"
            else:
                status = "completed"

        weeks.append(RtpWeekStatus(
            week=i, target_pct=target, criteria=RTP_CRITERIA[i],
            status=status, actual_load_pct=actual_pct,
        ))

    # If we've gone past week 5, mark as fully returned
    n_weeks_elapsed = post["rtp_week"].max() if not post.empty else 0
    if n_weeks_elapsed >= 5:
        status_message = "✅ Protocole RTP terminé — joueur en charge normale."
        progress = 100.0
    else:
        active = weeks[min(current_week - 1, 4)]
        if active.status == "flagged":
            status_message = (
                f"⚠ Semaine {active.week} : charge réelle "
                f"{active.actual_load_pct:.0f}% vs cible {active.target_pct}% "
                f"— écart à surveiller.")
        else:
            status_message = (
                f"Semaine {active.week}/5 du protocole RTP — "
                f"cible {active.target_pct}% de la charge normale.")
        progress = min(100.0, (current_week / 5) * 100)

    return RtpPlan(
        player_id=player_id, is_injured=True, weeks=weeks,
        current_week=current_week, progress_pct=progress,
        status_message=status_message,
    )
