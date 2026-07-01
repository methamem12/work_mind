"""
ml/pdf_report.py — PDF export of risk report using ReportLab.
Embeds 3D body snapshot + all tables.
"""
from __future__ import annotations
import io, os, tempfile
from datetime import datetime
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, HRFlowable,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

W, H = A4
DARK   = colors.HexColor("#0B0F14")
NAVY   = colors.HexColor("#141C2E")
TEAL   = colors.HexColor("#00C9B1")
AMBER  = colors.HexColor("#F59E0B")
RED    = colors.HexColor("#EF4444")
GREEN  = colors.HexColor("#22C55E")
WHITE  = colors.white
MUTED  = colors.HexColor("#8B97A8")
OFF_W  = colors.HexColor("#E6EDF3")


def _risk_color(val: float):
    if val >= 0.66: return RED
    if val >= 0.33: return AMBER
    return GREEN


def _body_snapshot_bytes(body_risk: Dict[str, float]) -> bytes:
    """Render 3D body to PNG bytes."""
    from ml.body3d import render_body
    fig = plt.figure(figsize=(4.5, 6.5), facecolor="#0B0F14")
    ax  = fig.add_subplot(111, projection="3d")
    render_body(ax, body_risk, title="", reset_view=True, show_chains=True)
    ax.view_init(elev=12, azim=-72)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor="#0B0F14")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _season_chart_bytes(weekly_summary, player_name: str) -> bytes | None:
    if weekly_summary is None or weekly_summary.empty:
        return None
    from ml.longitudinal import season_chart
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5),
                                    facecolor="#0B0F14")
    season_chart(ax1, ax2, weekly_summary, player_name)
    fig.tight_layout(pad=1.0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor="#0B0F14")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def export_pdf(
    path: str,
    player_name: str,
    player_sport: str,
    report,                    # AdvisorReport
    weekly_summary=None,       # optional DataFrame
    sim_mode: bool = False,
) -> str:
    """
    Generate a full PDF report and save to `path`.
    Returns the path.
    """
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("title", fontSize=20, textColor=TEAL,
                                 spaceAfter=4, fontName="Helvetica-Bold",
                                 alignment=TA_CENTER)
    h1_style = ParagraphStyle("h1", fontSize=13, textColor=TEAL,
                               spaceBefore=10, spaceAfter=4,
                               fontName="Helvetica-Bold")
    body_style = ParagraphStyle("body", fontSize=9, textColor=OFF_W,
                                leading=13, fontName="Helvetica")
    muted_style = ParagraphStyle("muted", fontSize=8, textColor=MUTED,
                                 fontName="Helvetica-Oblique")

    doc = SimpleDocTemplate(
        path, pagesize=A4,
        leftMargin=1.8*cm, rightMargin=1.8*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )
    story = []

    # ── Header ─────────────────────────────────────────────────────────────
    mode_str = " [MODE SIMULATION]" if sim_mode else ""
    story.append(Paragraph(f"Rapport de Performance — {player_name}{mode_str}", title_style))
    story.append(Paragraph(
        f"Sport : {player_sport}  ·  Généré le {datetime.now().strftime('%d/%m/%Y %H:%M')}",
        muted_style))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=TEAL, spaceAfter=8))

    # ── KPIs ───────────────────────────────────────────────────────────────
    risk_c = _risk_color(report.injury_prob)
    kpi_data = [
        ["Index de performance", "Risque de blessure"],
        [f"{report.perf_index:.1f}", f"{report.injury_prob:.1%}"],
    ]
    kpi_table = Table(kpi_data, colWidths=[8*cm, 8*cm])
    kpi_table.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0), NAVY),
        ("BACKGROUND",  (0,1), (0,1), NAVY),
        ("BACKGROUND",  (1,1), (1,1), NAVY),
        ("TEXTCOLOR",   (0,0), (-1,0), MUTED),
        ("TEXTCOLOR",   (0,1), (0,1), TEAL),
        ("TEXTCOLOR",   (1,1), (1,1), risk_c),
        ("FONTNAME",    (0,1), (-1,1), "Helvetica-Bold"),
        ("FONTSIZE",    (0,1), (-1,1), 22),
        ("FONTSIZE",    (0,0), (-1,0), 9),
        ("ALIGN",       (0,0), (-1,-1), "CENTER"),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [NAVY]),
        ("BOX",         (0,0), (-1,-1), 0.5, TEAL),
        ("INNERGRID",   (0,0), (-1,-1), 0.3, colors.HexColor("#243042")),
        ("TOPPADDING",  (0,0), (-1,-1), 8),
        ("BOTTOMPADDING",(0,0), (-1,-1), 8),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 10))

    # ── 3D body + risk table side by side ─────────────────────────────────
    story.append(Paragraph("Zones anatomiques à risque", h1_style))
    img_bytes = _body_snapshot_bytes(report.body_risk)
    img = RLImage(io.BytesIO(img_bytes), width=6*cm, height=8.5*cm)

    # Body risk table
    from ml.performance_advisor import BODY_PART_LABELS_FR
    ranked = sorted(report.body_risk.items(), key=lambda x: -x[1])
    br_data = [["Zone", "Risque"]] + [
        [BODY_PART_LABELS_FR.get(z, z), f"{v:.0%}"] for z,v in ranked[:10]
    ]
    br_table = Table(br_data, colWidths=[5*cm, 2.5*cm])
    br_ts = [
        ("BACKGROUND", (0,0), (-1,0), NAVY),
        ("TEXTCOLOR",  (0,0), (-1,0), TEAL),
        ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0), (-1,-1), 8),
        ("ALIGN",      (1,0), (1,-1), "CENTER"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#141C2E"), colors.HexColor("#1A2336")]),
        ("TEXTCOLOR",  (0,1), (-1,-1), OFF_W),
        ("BOX",        (0,0), (-1,-1), 0.4, colors.HexColor("#243042")),
        ("TOPPADDING", (0,0), (-1,-1), 3),
        ("BOTTOMPADDING",(0,0),(-1,-1),3),
    ]
    for i, (z, v) in enumerate(ranked[:10], start=1):
        br_ts.append(("TEXTCOLOR", (1,i), (1,i), _risk_color(v)))
    br_table.setStyle(TableStyle(br_ts))
    combined = Table([[img, br_table]], colWidths=[7*cm, 9*cm])
    combined.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP")]))
    story.append(combined)
    story.append(Spacer(1, 8))

    # ── Causes table ───────────────────────────────────────────────────────
    story.append(Paragraph("Causes du risque & actions préventives", h1_style))
    from ml.performance_advisor import BODY_PART_LABELS_FR
    c_data = [["Attribut", "Écart (σ)", "Contrib.", "Zones", "Action préventive"]]
    for c in report.causes:
        zones = ", ".join(BODY_PART_LABELS_FR.get(z,z) for z in c.zones[:2])
        c_data.append([
            c.label, f"{c.current_z:+.2f}", f"{c.risk_contrib:.0%}",
            zones, c.action,
        ])
    c_table = Table(c_data, colWidths=[4*cm, 1.6*cm, 1.5*cm, 3.5*cm, 5.4*cm])
    c_ts = [
        ("BACKGROUND", (0,0), (-1,0), NAVY),
        ("TEXTCOLOR",  (0,0), (-1,0), TEAL),
        ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0), (-1,-1), 7.5),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#141C2E"), colors.HexColor("#1A2336")]),
        ("TEXTCOLOR",  (0,1), (-1,-1), OFF_W),
        ("BOX",        (0,0), (-1,-1), 0.4, colors.HexColor("#243042")),
        ("WORDWRAP",   (4,1), (4,-1), True),
        ("TOPPADDING", (0,0), (-1,-1), 3),
        ("BOTTOMPADDING",(0,0),(-1,-1),3),
        ("ALIGN",      (1,0), (2,-1), "CENTER"),
    ]
    for i, c in enumerate(report.causes, start=1):
        c_ts.append(("TEXTCOLOR", (2,i), (2,i), _risk_color(c.risk_contrib)))
    c_table.setStyle(TableStyle(c_ts))
    story.append(c_table)
    story.append(Spacer(1, 8))

    # ── FOCUS / CAUTION ────────────────────────────────────────────────────
    for cat, color_h in [("FOCUS", GREEN), ("CAUTION", AMBER)]:
        items = [r for r in report.recommendations if r.category == cat]
        if not items:
            continue
        story.append(Paragraph(f"Recommandations — {cat}", h1_style))
        r_data = [["Attribut", "Sens", "Impact perf", "Poids blessure"]]
        for r in items:
            r_data.append([
                r.label,
                "▲ augmenter" if r.direction > 0 else "▼ réduire",
                f"{r.perf_impact:+.2f}",
                f"{r.injury_weight:+.2f}",
            ])
        r_table = Table(r_data, colWidths=[6*cm, 3*cm, 2.5*cm, 2.5*cm])
        r_table.setStyle(TableStyle([
            ("BACKGROUND",  (0,0), (-1,0), NAVY),
            ("TEXTCOLOR",   (0,0), (-1,0), color_h),
            ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",    (0,0), (-1,-1), 8),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#141C2E"), colors.HexColor("#1A2336")]),
            ("TEXTCOLOR",   (0,1), (-1,-1), OFF_W),
            ("BOX",         (0,0), (-1,-1), 0.4, colors.HexColor("#243042")),
            ("TOPPADDING",  (0,0), (-1,-1), 3),
            ("BOTTOMPADDING",(0,0),(-1,-1),3),
        ]))
        story.append(r_table)
        story.append(Spacer(1, 6))

    # ── Season chart ───────────────────────────────────────────────────────
    season_bytes = _season_chart_bytes(weekly_summary, player_name)
    if season_bytes:
        story.append(Paragraph("Suivi saisonnier", h1_style))
        story.append(RLImage(io.BytesIO(season_bytes), width=16*cm, height=9*cm))

    # ── Footer ─────────────────────────────────────────────────────────────
    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=0.5, color=MUTED))
    story.append(Paragraph(
        "Athlete AI Platform — Rapport confidentiel généré automatiquement. "
        "Ce document est destiné au staff médical et technique uniquement.",
        muted_style))

    doc.build(story)
    return path
