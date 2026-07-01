"""
ml/trend_utils.py — Shared trend-arrow utilities for both apps.
"""
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import pandas as pd


def compute_trend(
    series: pd.Series,
    window: int = 7,
) -> Tuple[float, str]:
    """
    Compare the mean of the last `window` values vs the prior `window` values.
    Returns (delta, arrow) where arrow ∈ {"↑","↓","→"}.
    """
    s = series.dropna()
    if len(s) < 2:
        return 0.0, "→"
    recent = s.tail(window)
    prior  = s.iloc[-(2*window):-window] if len(s) >= 2*window else s.iloc[:-len(recent)]
    if prior.empty:
        return 0.0, "→"
    delta = float(recent.mean() - prior.mean())
    threshold = 0.03 * (abs(prior.mean()) + 1e-6)
    if delta > threshold:
        return delta, "↑"
    if delta < -threshold:
        return delta, "↓"
    return delta, "→"


def trend_color(arrow: str, higher_is_worse: bool = True) -> str:
    """Return a hex colour for a trend arrow given semantic direction."""
    if arrow == "→":
        return "#8B97A8"
    bad = (arrow == "↑") if higher_is_worse else (arrow == "↓")
    return "#EF4444" if bad else "#22C55E"


def adaptive_threshold(
    player_history_probs: np.ndarray,
    global_threshold: float = 0.25,
    min_samples: int = 10,
) -> float:
    """
    Personal adaptive threshold: blends the global model threshold with the
    player's own historical risk distribution (mean + 1 std), so naturally
    high-load athletes don't trigger constant false alarms.
    """
    probs = np.asarray(player_history_probs)
    probs = probs[~np.isnan(probs)]
    if len(probs) < min_samples:
        return global_threshold
    personal = float(np.mean(probs) + np.std(probs))
    # Blend 60% personal / 40% global, clipped to sane bounds
    blended = 0.6 * personal + 0.4 * global_threshold
    return float(np.clip(blended, 0.15, 0.85))
