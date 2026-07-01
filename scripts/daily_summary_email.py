"""
scripts/daily_summary_email.py — Automated daily high-risk summary email.

Run this every morning (Task Scheduler / cron) to:
  1. Score every player with the current model.
  2. Generate a PDF report for every player above the risk threshold.
  3. Email the PDFs to the medical staff distribution list.

Configuration via environment variables (recommended) or the CONFIG dict below:
    SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, EMAIL_FROM, EMAIL_TO

Usage:
    python scripts/daily_summary_email.py
    python scripts/daily_summary_email.py --threshold 0.5 --dry-run
"""
from __future__ import annotations
import os, sys, argparse, smtplib, ssl, tempfile
from email.message import EmailMessage
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import joblib
import pandas as pd

from ml.team_analysis import run_team_analysis
from ml.performance_advisor import PerformanceAdvisor
from ml.pdf_report import export_pdf
from ml.longitudinal import weekly_summary

CONFIG = {
    "SMTP_HOST":     os.environ.get("SMTP_HOST", "smtp.gmail.com"),
    "SMTP_PORT":     int(os.environ.get("SMTP_PORT", "587")),
    "SMTP_USER":     os.environ.get("SMTP_USER", ""),
    "SMTP_PASSWORD": os.environ.get("SMTP_PASSWORD", ""),
    "EMAIL_FROM":    os.environ.get("EMAIL_FROM", ""),
    "EMAIL_TO":      os.environ.get("EMAIL_TO", ""),  # comma-separated
}

DB_PATH    = "data/athlete.db"
MODEL_PATH = "ml/injury_model.pkl"


def generate_high_risk_pdfs(threshold: float, outdir: str) -> list[str]:
    """Generate one PDF per player above threshold. Returns list of file paths."""
    model = joblib.load(MODEL_PATH)
    team_df = run_team_analysis(db_path=DB_PATH, global_model=model)
    high_risk = team_df[team_df["injury_prob"] >= threshold]

    advisor = PerformanceAdvisor(db_path=DB_PATH)
    paths = []
    for _, row in high_risk.iterrows():
        pid = int(row["player_id"])
        try:
            report = advisor.advise(pid)
        except Exception as e:
            print(f"  ⚠ Skip player {pid} ({row['name']}): {e}")
            continue
        weekly = weekly_summary(pid)
        fname  = os.path.join(outdir, f"risque_{row['name'].replace(' ','_')}.pdf")
        export_pdf(
            path=fname, player_name=row["name"], player_sport=row["sport"],
            report=report, weekly_summary=weekly, sim_mode=False,
        )
        paths.append(fname)
        print(f"  ✅ PDF généré : {row['name']} ({row['injury_prob']:.0%})")
    return paths


def send_email(pdf_paths: list[str], threshold: float, dry_run: bool = False) -> None:
    if not pdf_paths:
        print("Aucun joueur au-dessus du seuil — pas d'email envoyé.")
        return

    subject = f"⚠ Athlete AI — {len(pdf_paths)} joueur(s) à risque élevé ({datetime.now():%d/%m/%Y})"
    body = (
        f"Résumé automatique du {datetime.now():%d/%m/%Y %H:%M}\n\n"
        f"{len(pdf_paths)} joueur(s) au-dessus du seuil de risque ({threshold:.0%}).\n"
        f"Rapports PDF détaillés en pièce jointe.\n\n"
        f"— Athlete AI Platform (envoi automatique, ne pas répondre)"
    )

    if dry_run:
        print("\n[DRY RUN] Email qui aurait été envoyé :")
        print(f"  À      : {CONFIG['EMAIL_TO']}")
        print(f"  Objet  : {subject}")
        print(f"  Pièces : {len(pdf_paths)} PDF(s)")
        return

    if not all([CONFIG["SMTP_USER"], CONFIG["SMTP_PASSWORD"],
                CONFIG["EMAIL_FROM"], CONFIG["EMAIL_TO"]]):
        print("⚠ Configuration SMTP incomplète — définissez les variables "
              "d'environnement SMTP_USER, SMTP_PASSWORD, EMAIL_FROM, EMAIL_TO.")
        return

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"]    = CONFIG["EMAIL_FROM"]
    msg["To"]      = CONFIG["EMAIL_TO"]
    msg.set_content(body)

    for path in pdf_paths:
        with open(path, "rb") as f:
            msg.add_attachment(
                f.read(), maintype="application", subtype="pdf",
                filename=os.path.basename(path))

    context = ssl.create_default_context()
    with smtplib.SMTP(CONFIG["SMTP_HOST"], CONFIG["SMTP_PORT"]) as server:
        server.starttls(context=context)
        server.login(CONFIG["SMTP_USER"], CONFIG["SMTP_PASSWORD"])
        server.send_message(msg)
    print(f"✅ Email envoyé à {CONFIG['EMAIL_TO']}")


def main():
    parser = argparse.ArgumentParser(description="Résumé quotidien des risques de blessure")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Seuil de risque pour inclusion (défaut 0.5)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Génère les PDFs mais n'envoie pas l'email")
    args = parser.parse_args()

    print(f"── Résumé quotidien Athlete AI — {datetime.now():%d/%m/%Y %H:%M} ──")
    print(f"Seuil de risque : {args.threshold:.0%}\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_paths = generate_high_risk_pdfs(args.threshold, tmpdir)
        send_email(pdf_paths, args.threshold, dry_run=args.dry_run)

    print("\nTerminé.")


if __name__ == "__main__":
    main()
