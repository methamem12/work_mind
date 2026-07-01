"""
main.py — Athlete AI Platform : Injury Risk Predictor (v2)

Nouveautés v2 :
  • Connexion / rôles (coach vs medical) — database/auth.py
  • Onglet Équipe : analyse batch + heatmap de risque
  • Recherche dans les notes de séance
  • Tendance 7 jours + flèches sur le risque
  • Protocole RTP (retour à l'entraînement) pour joueurs blessés
  • Export PDF (joueur + équipe)
  • Thème clair/sombre
"""
from __future__ import annotations
import sys, os, sqlite3, io
from typing import Optional

import numpy as np
import pandas as pd
import joblib

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QDialog, QMessageBox, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QVBoxLayout, QHBoxLayout, QWidget,
    QFileDialog, QProgressDialog, QLabel, QListWidgetItem,
)
from PyQt5.QtGui import QColor, QBrush
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QDate
from PyQt5.uic import loadUi

import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from database.database import Database
from database.auth import (
    authenticate, can_see_raw_features, can_see_shap,
    can_manage_users, can_edit_players, User,
)
from ml.injury_model import (
    build_features, FEATURE_COLUMNS, load_sport_model, MODEL_PATH,
)
from ml.team_analysis import run_team_analysis, render_heatmap
from ml.trend_utils import compute_trend, trend_color, adaptive_threshold
from ml.rtp_protocol import build_rtp_plan, is_player_injured
from ml.pdf_report import export_pdf
from ml.performance_advisor import PerformanceAdvisor
from constants import (
    MSG_PLAYER_NAME_REQUIRED, MSG_PLAYER_REQUIRED, MSG_PLAYER_NOT_SELECTED,
    MSG_PLAYER_DELETE_NOT_SELECTED, MSG_NO_DATA, MSG_NO_PLAYER_FIRST,
    MSG_PLAYER_ADDED, MSG_SESSION_ADDED, MSG_TRAINING_COMPLETED,
    TITLE_REQUIRED_FIELDS, TITLE_NO_PLAYER, TITLE_EXPORT_SUCCESS, TITLE_TRAINING_DONE,
    LABEL_RTP_AVAILABLE, LABEL_THEME_LIGHT, LABEL_THEME_DARK, LABEL_NO_RESULT,
)

DB_PATH    = "data/athlete.db"
THEME_DARK  = "ui/theme.qss"
THEME_LIGHT = "ui/theme_light.qss"

COLOR_SUCCESS = "#22C55E"
COLOR_WARNING = "#F59E0B"
COLOR_DANGER  = "#EF4444"


def _risk_color(val: float) -> QColor:
    if val >= 0.66: return QColor(COLOR_DANGER)
    if val >= 0.33: return QColor(COLOR_WARNING)
    return QColor(COLOR_SUCCESS)


def apply_theme(app: QApplication, dark: bool = True) -> None:
    path = THEME_DARK if dark else THEME_LIGHT
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            app.setStyleSheet(fh.read())
    bg = "#0B0F14" if dark else "#F5F7FA"
    fg = "#E6EDF3" if dark else "#1B2330"
    mpl.rcParams.update({
        "axes.facecolor": bg, "figure.facecolor": bg, "savefig.facecolor": bg,
        "axes.edgecolor": "#243042" if dark else "#D8DEE6",
        "axes.labelcolor": fg, "xtick.color": "#8B97A8" if dark else "#5B6573",
        "ytick.color": "#8B97A8" if dark else "#5B6573", "text.color": fg,
        "grid.color": "#243042" if dark else "#E3E9F0",
        "font.family": ["Segoe UI", "DejaVu Sans"],
    })


# ─── Login dialog ───────────────────────────────────────────────────────────
class LoginDialog(QDialog):
    def __init__(self):
        super().__init__()
        loadUi("ui/login_dialog.ui", self)
        self.user: Optional[User] = None
        self.loginBtn.clicked.connect(self._try_login)
        self.passwordEdit.returnPressed.connect(self._try_login)

    def _try_login(self):
        username = self.usernameEdit.text().strip()
        password = self.passwordEdit.text()
        user = authenticate(username, password, db_path=DB_PATH)
        if user is None:
            self.errorLabel.setText("⚠ Identifiants incorrects.")
            return
        self.user = user
        self.accept()


# ─── Background worker ──────────────────────────────────────────────────────
class Worker(QThread):
    finished = pyqtSignal(object)
    error    = pyqtSignal(str)
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self._fn, self._args, self._kwargs = fn, args, kwargs
    def run(self):
        try:
            self.finished.emit(self._fn(*self._args, **self._kwargs))
        except Exception as e:
            self.error.emit(str(e))


class PlayerDialog(QDialog):
    def __init__(self, parent=None, db=None, player=None):
        super().__init__(parent)
        loadUi("ui/player_dialog.ui", self)
        self.db = db
        self.player = player

        self.savePlayerBtn.clicked.connect(self._save_player)
        self.cancelPlayerBtn.clicked.connect(self.reject)

        if player:
            self.setWindowTitle("Modifier un joueur")
            self.playerNameEdit.setText(player[1] or "")
            self.positionCombo.setCurrentText(player[3] or "")
            self.ageSpin.setValue(int(player[4] or 0))
            self.heightSpin.setValue(float(player[5] or 0.0))
            self.weightSpin.setValue(float(player[6] or 0.0))
            self.footCombo.setCurrentText(player[7] or "")
        else:
            self.setWindowTitle("Ajouter un joueur")

    def _save_player(self):
        name = self.playerNameEdit.text().strip()
        if not name:
            QMessageBox.warning(self, TITLE_REQUIRED_FIELDS, MSG_PLAYER_NAME_REQUIRED)
            return

        sport = self.player[2] if self.player else "Football"
        team = self.player[9] if self.player else None
        if self.player is None:
            self.db.add_player(
                name=name,
                sport=sport,
                position=self.positionCombo.currentText() or None,
                age=self.ageSpin.value() or None,
                height=self.heightSpin.value() or None,
                weight=self.weightSpin.value() or None,
                dominant_foot=self.footCombo.currentText() or None,
                nationality=None,
                team=team,
            )
        else:
            self.db.update_player(
                self.player[0],
                name=name,
                sport=sport,
                position=self.positionCombo.currentText() or None,
                age=self.ageSpin.value() or None,
                height=self.heightSpin.value() or None,
                weight=self.weightSpin.value() or None,
                dominant_foot=self.footCombo.currentText() or None,
                nationality=None,
                team=team,
            )
        self.accept()


class SessionDialog(QDialog):
    def __init__(self, parent=None, player_id=None):
        super().__init__(parent)
        loadUi("ui/session_dialog.ui", self)
        self.player_id = player_id
        self.session_data = None

        self.dateEdit.setDate(QDate.currentDate())
        self.saveSessionBtn.clicked.connect(self._save_session)
        self.cancelSessionBtn.clicked.connect(self.reject)

    def _save_session(self):
        if not self.player_id:
            QMessageBox.warning(self, TITLE_REQUIRED_PLAYER, MSG_PLAYER_REQUIRED)
            return

        self.session_data = {
            "player_id": int(self.player_id),
            "date": self.dateEdit.date().toString("yyyy-MM-dd"),
            "session_type": self.sessionTypeCombo.currentText(),
            "weather": self.weatherCombo.currentText(),
            "surface": self.surfaceEdit.text().strip() or None,
            "training_minutes": self.trainingMinutesSpin.value(),
            "rpe": self.rpeSpin.value(),
            "distance_km": self.distanceKmSpin.value() or None,
            "sprint_distance_km": self.sprintDistanceKmSpin.value() or None,
            "sprints_count": self.sprintsCountSpin.value() or None,
            "hid_km": self.hidKmSpin.value() or None,
            "acceleration_max": self.accMaxSpin.value() or None,
            "max_speed": self.maxSpeedSpin.value() or None,
            "accelerations": self.accelerationsSpin.value() or None,
            "decelerations": self.decelerationsSpin.value() or None,
            "player_load": self.playerLoadSpin.value() or None,
            "heart_rate_avg": None,
            "heart_rate_max": None,
            "acwr": self.acwrSpin.value() or None,
            "charge_7j": None,
            "charge_28j": None,
            "hrv_rmssd": self.hrvRmssdSpin.value() or None,
            "hrv_trend": self.hrvTrendSpin.value() or None,
            "hrv_moy_7j": None,
            "fc_repos": self.fcReposSpin.value() or None,
            "fc_repos_alerte": 1 if self.fcAlerteCheck.isChecked() else 0,
            "sommeil_h": self.sommeilHSpin.value() or None,
            "sommeil_qualite": self.sommeilQSpin.value() or None,
            "fatigue": self.fatigueSpin.value() or None,
            "spo2": self.spo2Spin.value() or None,
            "jours_depuis_match": self.joursDepuisMatchSpin.value() or None,
            "body_temp_celsius": None,
            "motivation": self.motivationSpin.value() or None,
            "stress": self.stressSpin.value() or None,
            "reaction_ms": self.reactionMsSpin.value() or None,
            "charge_mentale": self.chargeMentaleSpin.value() or None,
            "regularite_score": self.regulariteSpin.value() or None,
            "poids_variation_pct": self.poidsVariationSpin.value() or None,
            "hydratation_score": self.hydratationSpin.value() or None,
            "ck_post": self.ckPostSpin.value() or None,
            "vo2max": self.vo2maxSpin.value() or None,
            "puissance_w_kg": None,
            "cmj_cm": self.cmjCmSpin.value() or None,
            "rsa_index": self.rsaIndexSpin.value() or None,
            "force_n": self.forceNSpin.value() or None,
            "fatigue_sprint_pct": self.fatigueSprintSpin.value() or None,
            "pct_masse_grasse": None,
            "ferritine": None,
            "hemoglobine": None,
            "wellness_score": self.wellnessSpin.value() or None,
            "perf_index": self.perfIndexSpin.value() or None,
            "previous_injuries": self.previousInjuriesSpin.value() or 0,
            "injury_label": 1 if self.injuryCheck.isChecked() else 0,
            "session_notes": None,
        }
        self.accept()


# ─── Main window ─────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):

    def __init__(self, user: User):
        super().__init__()
        loadUi("ui/main_window.ui", self)
        self.user = user
        self.db   = Database(db_path=DB_PATH)
        self.global_model = joblib.load(MODEL_PATH)
        self._dark_mode = True
        self._team_df: Optional[pd.DataFrame] = None
        self._current_player_id: Optional[int] = None

        self.setWindowTitle(
            f"⚡ Athlete AI Platform — {user.full_name} ({user.role})")

        self._apply_role_restrictions()
        self._setup_shap_canvas()
        self._setup_timeline_canvas()
        self._setup_heatmap_canvas()
        self._wire_buttons()

        self.load_players_list()
        self.load_predict_combo()
        self.load_similar_combo()
        self._refresh_sport_filter()

    # ── Role-based UI restrictions ─────────────────────────────────────────
    def _apply_role_restrictions(self):
        if not can_manage_users(self.user):
            pass  # no user-management UI currently exposed; placeholder for future admin panel
        if not can_edit_players(self.user):
            self.addPlayerBtn.setEnabled(False)
            self.editPlayerBtn.setEnabled(False)
            self.deletePlayerBtn.setEnabled(False)
            self.addSessionBtn.setEnabled(False)
            self.addPlayerBtn.setToolTip("Réservé au staff médical/admin")
            self.editPlayerBtn.setToolTip("Réservé au staff médical/admin")
            self.deletePlayerBtn.setToolTip("Réservé au staff médical/admin")
            self.addSessionBtn.setToolTip("Réservé au staff médical/admin")

    def _add_player(self):
        dialog = PlayerDialog(self, db=self.db)
        if dialog.exec_() == QDialog.Accepted:
            self.load_players_list()
            self.load_predict_combo()
            self.load_similar_combo()
            QMessageBox.information(self, "Joueur ajouté", "Le joueur a bien été enregistré.")

    def _edit_player(self):
        items = self.playerList.selectedItems()
        if not items:
            QMessageBox.warning(self, TITLE_NO_PLAYER, MSG_PLAYER_NOT_SELECTED)
            return
        pid = items[0].data(Qt.UserRole)
        player = self.db.get_player(pid)
        if not player:
            return
        dialog = PlayerDialog(self, db=self.db, player=player)
        if dialog.exec_() == QDialog.Accepted:
            self.load_players_list()
            self.load_predict_combo()
            self.load_similar_combo()
            for i in range(self.playerList.count()):
                if self.playerList.item(i).data(Qt.UserRole) == pid:
                    self.playerList.setCurrentRow(i)
                    break
            self._on_player_selected()

    def _delete_player(self):
        items = self.playerList.selectedItems()
        if not items:
            QMessageBox.warning(self, TITLE_NO_PLAYER, MSG_PLAYER_DELETE_NOT_SELECTED)
            return
        pid = items[0].data(Qt.UserRole)
        player = self.db.get_player(pid)
        if not player:
            return
        reply = QMessageBox.question(
            self, "Supprimer le joueur",
            f"Supprimer {player[1]} ?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self.db.delete_player(pid)
        self.load_players_list()
        self.load_predict_combo()
        self.load_similar_combo()
        self._current_player_id = None
        self.playerNameLabel.setText("Nom : —")
        self.playerSportLabel.setText("Sport : —")
        self.playerPositionLabel.setText("Poste : —")
        self.playerAgeLabel.setText("Âge : —")
        self.playerTeamLabel.setText("Équipe : —")
        self.sessionTable.setRowCount(0)
        self.sessionTable.setColumnCount(0)

    def _add_session(self):
        if self._current_player_id is None:
            QMessageBox.warning(self, "Aucun joueur", "Sélectionnez d'abord un joueur.")
            return
        dialog = SessionDialog(self, player_id=self._current_player_id)
        if dialog.exec_() == QDialog.Accepted:
            self.db.add_session(**dialog.session_data)
            self._load_sessions_table(self._current_player_id)
            QMessageBox.information(self, TITLE_EXPORT_SUCCESS, MSG_SESSION_ADDED)

    # ── Setup ──────────────────────────────────────────────────────────────
    def _setup_shap_canvas(self):
        self.shap_fig = Figure(figsize=(6, 3), facecolor="#0B0F14")
        self.shap_canvas = FigureCanvas(self.shap_fig)
        self.shap_ax = self.shap_fig.add_subplot(111)
        self.shapLayout.addWidget(self.shap_canvas)

    def _setup_timeline_canvas(self):
        self.timeline_fig = Figure(figsize=(6, 2.5), facecolor="#0B0F14")
        self.timeline_canvas = FigureCanvas(self.timeline_fig)
        self.timeline_ax = self.timeline_fig.add_subplot(111)
        self.timelineLayout.addWidget(self.timeline_canvas)

    def _setup_heatmap_canvas(self):
        self.heatmap_fig = Figure(figsize=(10, 6), facecolor="#0B0F14")
        self.heatmap_canvas = FigureCanvas(self.heatmap_fig)
        self.heatmap_ax = self.heatmap_fig.add_subplot(111)
        self.heatmapLayout.addWidget(self.heatmap_canvas)

    def _wire_buttons(self):
        self.themeToggleBtn.clicked.connect(self._toggle_theme)
        self.batchRunBtn.clicked.connect(self._run_team_analysis)
        self.batchRunBtn2.clicked.connect(self._run_team_analysis)
        self.exportPdfBtn.clicked.connect(self._export_player_pdf)
        self.exportTeamPdfBtn.clicked.connect(self._export_team_pdf)
        self.predictBtn.clicked.connect(self._run_prediction)
        self.notesSearchBtn.clicked.connect(self._search_notes)
        self.addPlayerBtn.clicked.connect(self._add_player)
        self.editPlayerBtn.clicked.connect(self._edit_player)
        self.deletePlayerBtn.clicked.connect(self._delete_player)
        self.addSessionBtn.clicked.connect(self._add_session)
        self.teamSportFilter.currentIndexChanged.connect(self._run_team_analysis)
        self.playerList.itemSelectionChanged.connect(self._on_player_selected)
        self.searchSimilarBtn.clicked.connect(self._search_similar)
        self.trainModelBtn.clicked.connect(self._train_model)
        self.playerSearchEdit.textChanged.connect(self._filter_player_list)
        self.sessionSearchEdit.textChanged.connect(self._filter_sessions_table)

    # ── Theme toggle ───────────────────────────────────────────────────────
    def _toggle_theme(self):
        self._dark_mode = not self._dark_mode
        app = QApplication.instance()
        apply_theme(app, dark=self._dark_mode)
        self.themeToggleBtn.setText(LABEL_THEME_LIGHT if self._dark_mode else LABEL_THEME_DARK)

    # ── Player list ────────────────────────────────────────────────────────
    def load_players_list(self):
        self.playerList.clear()
        self._all_players = self.db.get_players()
        for p in self._all_players:
            item = QListWidgetItem(f"{p[0]} — {p[1]} ({p[2]})")
            item.setData(Qt.UserRole, p[0])
            self.playerList.addItem(item)

    def _filter_player_list(self, text: str):
        text = text.lower().strip()
        for i in range(self.playerList.count()):
            item = self.playerList.item(i)
            item.setHidden(text not in item.text().lower())

    def _on_player_selected(self):
        items = self.playerList.selectedItems()
        if not items: return
        pid = items[0].data(Qt.UserRole)
        self._current_player_id = pid
        p = self.db.get_player(pid)
        if p:
            self.playerNameLabel.setText(f"Nom : {p[1]}")
            self.playerSportLabel.setText(f"Sport : {p[2]}")
            self.playerPositionLabel.setText(f"Poste : {p[3]}")
            self.playerAgeLabel.setText(f"Âge : {p[4]}")
            self.playerTeamLabel.setText(f"Équipe : {p[9] or '—'}")
        self._load_sessions_table(pid)

    # ── Sessions table ─────────────────────────────────────────────────────
    SESSION_TABLE_COLUMNS = [
        ("date", "Date"), ("session_type", "Type"),
        ("distance_km", "Distance (km)"), ("rpe", "RPE"),
        ("fatigue", "Fatigue"), ("acwr", "ACWR"),
        ("hrv_rmssd", "HRV (ms)"), ("sommeil_h", "Sommeil (h)"),
        ("injury_label", "Blessure"),
    ]

    def _load_sessions_table(self, pid):
        self._current_sessions = self.db.get_sessions_dict(pid)
        self._render_sessions_table(self._current_sessions)

    def _render_sessions_table(self, sessions):
        cols = self.SESSION_TABLE_COLUMNS
        self.sessionTable.setSortingEnabled(False)
        self.sessionTable.setColumnCount(len(cols))
        self.sessionTable.setHorizontalHeaderLabels([label for _, label in cols])
        self.sessionTable.setRowCount(len(sessions))
        for i, s in enumerate(sessions):
            for j, (key, _) in enumerate(cols):
                val = s.get(key)
                if key == "injury_label":
                    text = "⚠ Oui" if val else "—"
                elif isinstance(val, float):
                    text = f"{val:.2f}"
                else:
                    text = "" if val is None else str(val)
                self.sessionTable.setItem(i, j, QTableWidgetItem(text))
        self.sessionTable.horizontalHeader().setStretchLastSection(True)
        self.sessionTable.setSortingEnabled(True)

    def _filter_sessions_table(self, text: str):
        if not hasattr(self, "_current_sessions"):
            return
        text = text.lower().strip()
        if not text:
            self._render_sessions_table(self._current_sessions)
            return
        filtered = [
            s for s in self._current_sessions
            if text in str(s.get("date", "")).lower()
            or text in str(s.get("session_type", "")).lower()
            or text in str(s.get("session_notes", "")).lower()
        ]
        self._render_sessions_table(filtered)

    def load_predict_combo(self):
        self.predictPlayerCombo.clear()
        for p in self.db.get_players():
            self.predictPlayerCombo.addItem(f"{p[0]} — {p[1]}", p[0])

    def load_similar_combo(self):
        self.similarPlayerCombo.clear()
        for p in self.db.get_players():
            self.similarPlayerCombo.addItem(f"{p[0]} — {p[1]}", p[0])

    def _refresh_sport_filter(self):
        from PyQt5.QtCore import QSignalBlocker
        with QSignalBlocker(self.teamSportFilter):
            self.teamSportFilter.clear()
            self.teamSportFilter.addItem("Tous")
            for s in self.db.get_sports():
                self.teamSportFilter.addItem(s)

    # ── Notes search ───────────────────────────────────────────────────────
    def _search_notes(self):
        query = self.notesSearchEdit.text().strip()
        if not query:
            return
        results = self.db.search_session_notes(
            query, player_id=self._current_player_id)
        self.notesResultsList.clear()
        if not results:
            self.notesResultsList.addItem("Aucun résultat.")
            return
        for sid, pid, pname, date, notes in results:
            preview = (notes or "")[:60]
            self.notesResultsList.addItem(f"{date} — {pname} : {preview}…")

    # ── Single prediction (with trend + RTP) ───────────────────────────────
    def _run_prediction(self):
        pid = self.predictPlayerCombo.currentData()
        if pid is None: return

        df = build_features(DB_PATH)
        ph = df[df["player_id"] == int(pid)].sort_values("date")
        if ph.empty:
            QMessageBox.warning(self, "Données insuffisantes",
                MSG_NO_DATA); return

        with sqlite3.connect(DB_PATH) as cx:
            sport = pd.read_sql(
                f"SELECT sport FROM players WHERE id={int(pid)}", cx
            ).iloc[0, 0]
        model = load_sport_model(sport) or self.global_model

        X_all = ph[FEATURE_COLUMNS].fillna(ph[FEATURE_COLUMNS].median())
        proba_all = model.predict_proba(X_all)[:, 1]
        prob_now  = float(proba_all[-1])

        self.riskLabel.setText(f"Risque : {prob_now:.1%}")
        self.riskLabel.setProperty("risk",
            "high" if prob_now>=0.66 else "medium" if prob_now>=0.33 else "low")
        self.riskLabel.style().unpolish(self.riskLabel); self.riskLabel.style().polish(self.riskLabel)
        self.riskProgressBar.setValue(int(prob_now*100))

        # Trend
        delta, arrow = compute_trend(pd.Series(proba_all), window=7)
        self.trendArrowLabel.setText(arrow)
        self.trendArrowLabel.setStyleSheet(
            f"font-size:18pt;font-weight:700;color:{trend_color(arrow)};")
        self.trendDeltaLabel.setText(f"({delta:+.1%})")

        # Risk factors list
        self.riskFactorsList.clear()
        last_row = ph.iloc[-1]
        for feat in ["fatigue","acwr","sommeil_h","hrv_rmssd","stress"]:
            if feat in last_row:
                self.riskFactorsList.addItem(f"{feat} = {last_row[feat]:.2f}")

        # RTP protocol
        plan = build_rtp_plan(df, int(pid))
        if plan.is_injured:
            self.rtpStatusLabel.setText(plan.status_message)
            self.rtpProgressBar.setValue(int(plan.progress_pct))
            week_labels = [self.rtpW1,self.rtpW2,self.rtpW3,self.rtpW4,self.rtpW5]
            for i, wk in enumerate(plan.weeks):
                lbl = week_labels[i]
                color = ("#22C55E" if wk.status=="completed"
                        else "#F59E0B" if wk.status=="flagged"
                        else "#3B82F6" if wk.status=="current"
                        else "#566177")
                lbl.setStyleSheet(f"background:{color};color:white;padding:3px;border-radius:3px;")
                lbl.setToolTip(f"Semaine {wk.week} — cible {wk.target_pct}% — {wk.criteria}")
        else:
            self.rtpStatusLabel.setText("✅ Joueur disponible — pas de protocole RTP actif.")
            self.rtpProgressBar.setValue(0)

        # SHAP (medical only)
        if can_see_shap(self.user):
            self._render_shap(model, X_all, X_all.iloc[[-1]])
        else:
            self.shap_ax.clear()
            self.shap_ax.text(0.5, 0.5, "🔒 Accès SHAP réservé au staff médical",
                              transform=self.shap_ax.transAxes,
                              ha="center", va="center", color="#8B97A8")
            self.shap_ax.set_axis_off()
            self.shap_canvas.draw_idle()

        # Timeline
        self._render_timeline(ph["date"], proba_all)

    def _render_shap(self, model, X_bg, X_row):
        try:
            import shap
            from ml.performance_advisor import FEATURE_LABELS_FR
            lr = model.named_steps["clf"]; sc = model.named_steps["scaler"]
            imp = model.named_steps["imputer"]
            # Background must be a representative sample, not the row being
            # explained, otherwise every feature contribution collapses to 0.
            X_bg_imp = imp.transform(X_bg); X_bg_sc = sc.transform(X_bg_imp)
            X_row_imp = imp.transform(X_row); X_row_sc = sc.transform(X_row_imp)
            explainer = shap.LinearExplainer(lr, X_bg_sc, feature_names=FEATURE_COLUMNS)
            sv = explainer(X_row_sc).values[0]
            idx = np.argsort(np.abs(sv))[-12:][::-1]
            cols = [FEATURE_LABELS_FR.get(FEATURE_COLUMNS[i],FEATURE_COLUMNS[i]) for i in idx]
            vals = sv[idx]
            self.shap_ax.clear()
            self.shap_ax.set_facecolor("#0B0F14")
            colors_bar = [COLOR_DANGER if v>0 else COLOR_SUCCESS for v in vals]
            self.shap_ax.barh(range(len(vals)), vals[::-1], color=colors_bar[::-1])
            self.shap_ax.set_yticks(range(len(vals)))
            self.shap_ax.set_yticklabels(cols[::-1], fontsize=8, color="#E6EDF3")
            self.shap_ax.axvline(0, color="#243042", lw=1)
            self.shap_ax.tick_params(colors="#8B97A8")
            self.shap_fig.tight_layout(pad=1.0)
            self.shap_canvas.draw_idle()
        except ImportError:
            self.shap_ax.clear()
            self.shap_ax.text(0.5, 0.5,
                              "⚠ Librairie 'shap' non installée\n(pip install shap)",
                              transform=self.shap_ax.transAxes,
                              ha="center", va="center", color="#F59E0B", fontsize=9)
            self.shap_ax.set_axis_off()
            self.shap_canvas.draw_idle()
        except Exception as e:
            print(f"SHAP error: {e}")
            self.shap_ax.clear()
            self.shap_ax.text(0.5, 0.5, f"⚠ Erreur SHAP :\n{e}",
                              transform=self.shap_ax.transAxes,
                              ha="center", va="center", color="#EF4444", fontsize=8,
                              wrap=True)
            self.shap_ax.set_axis_off()
            self.shap_canvas.draw_idle()

    def _render_timeline(self, dates, probas):
        self.timeline_ax.clear()
        self.timeline_ax.set_facecolor("#0B0F14")
        x = np.arange(len(probas))
        self.timeline_ax.plot(x, probas, color="#3B82F6", lw=2)
        self.timeline_ax.fill_between(x, 0, probas, color="#3B82F6", alpha=0.15)
        self.timeline_ax.axhspan(0.66,1.0,color="#EF4444",alpha=0.06)
        self.timeline_ax.axhspan(0.33,0.66,color="#F59E0B",alpha=0.05)
        self.timeline_ax.set_ylim(0,1.05)
        self.timeline_ax.set_xticks([])
        self.timeline_ax.tick_params(colors="#8B97A8")
        self.timeline_fig.tight_layout(pad=1.0)
        self.timeline_canvas.draw_idle()

    # ── Team batch analysis + heatmap ──────────────────────────────────────
    def _run_team_analysis(self):
        sport_filter = self.teamSportFilter.currentText()
        prog = QProgressDialog("Analyse de l'équipe en cours…", "", 0, 0, self)
        prog.setWindowModality(Qt.WindowModal); prog.setMinimumDuration(0)
        prog.show(); QApplication.processEvents()

        def _job():
            return run_team_analysis(
                db_path=DB_PATH, global_model=self.global_model,
                sport_filter=sport_filter)

        w = Worker(_job)
        w.finished.connect(lambda df: self._on_team_ready(df, prog))
        w.error.connect(lambda e: (prog.close(), QMessageBox.critical(self,"Erreur",e)))
        w.start(); self._team_worker = w

    def _on_team_ready(self, df: pd.DataFrame, prog):
        prog.close()
        self._team_df = df
        render_heatmap(self.heatmap_ax, df,
                       title=f"Risque équipe ({len(df)} joueurs)")
        self.heatmap_fig.tight_layout(pad=1.5)
        self.heatmap_canvas.draw_idle()

        self.teamRiskTable.setRowCount(len(df))
        for i, row in df.iterrows():
            cells = [
                row["name"], row["sport"], f"{row['injury_prob']:.0%}",
                row["risk_level"],
                f"{row['trend_arrow']} {row['trend_7d']:+.1%}",
                f"{row['fatigue']:.1f}", f"{row['acwr']:.2f}",
            ]
            for j, val in enumerate(cells):
                item = QTableWidgetItem(str(val))
                if j == 2:
                    item.setForeground(QBrush(_risk_color(row["injury_prob"])))
                self.teamRiskTable.setItem(i, j, item)
        self.teamRiskTable.horizontalHeader().setStretchLastSection(True)

        from datetime import datetime
        self.teamLastRunLabel.setText(
            f"Dernière analyse : {datetime.now():%d/%m/%Y %H:%M}")

    # ── Similar players (existing feature, kept) ───────────────────────────
    def _search_similar(self):
        pid = self.similarPlayerCombo.currentData()
        if pid is None: return
        try:
            df = build_features(DB_PATH)
            target = df[df["player_id"]==int(pid)][FEATURE_COLUMNS].median()
            others = df.groupby("player_id")[FEATURE_COLUMNS].median()
            dist = ((others - target)**2).sum(axis=1).pow(0.5).sort_values()
            dist = dist.drop(int(pid), errors="ignore").head(10)

            self.similarTable.setRowCount(len(dist))
            self.similarTable.setColumnCount(2)
            self.similarTable.setHorizontalHeaderLabels(["Joueur ID","Distance"])
            for i,(opid,d) in enumerate(dist.items()):
                self.similarTable.setItem(i,0,QTableWidgetItem(str(opid)))
                self.similarTable.setItem(i,1,QTableWidgetItem(f"{d:.3f}"))
        except Exception as e:
            QMessageBox.critical(self,"Erreur",str(e))

    # ── Train model ────────────────────────────────────────────────────────
    def _train_model(self):
        prog = QProgressDialog("Entraînement en cours… (peut prendre plusieurs minutes)",
                               "", 0, 0, self)
        prog.setWindowModality(Qt.WindowModal); prog.setMinimumDuration(0)
        prog.show(); QApplication.processEvents()

        def _job():
            from ml.injury_model import train_model, train_sport_models
            m, report_global = train_model(verbose=False)
            _, report_sports = train_sport_models(verbose=False)
            return m, report_global, report_sports

        w = Worker(_job)
        def _done(result):
            prog.close()
            m, report_global, report_sports = result
            self.global_model = m
            self.accuracyLabel.setText(f"Seuil retenu : {m.threshold_:.3f}")
            
            # Display metrics in message box
            full_report = f"{report_global}\n\n{report_sports}"
            QMessageBox.information(self, TITLE_TRAINING_DONE, 
                f"{MSG_TRAINING_COMPLETED}\n\n{full_report}")
        w.finished.connect(_done)
        w.error.connect(lambda e:(prog.close(), QMessageBox.critical(self,"Erreur",e)))
        w.start(); self._train_worker = w

    # ── PDF export ─────────────────────────────────────────────────────────
    def _export_player_pdf(self):
        pid = self.predictPlayerCombo.currentData() or self._current_player_id
        if pid is None:
            QMessageBox.warning(self,"Aucun joueur","Sélectionnez un joueur d'abord."); return

        path,_ = QFileDialog.getSaveFileName(
            self,"Exporter rapport PDF", f"rapport_joueur_{pid}.pdf","PDF (*.pdf)")
        if not path: return

        try:
            advisor = PerformanceAdvisor(db_path=DB_PATH)
            report  = advisor.advise(int(pid))
            p = self.db.get_player(int(pid))
            from ml.longitudinal import weekly_summary
            weekly = weekly_summary(int(pid))
            export_pdf(path=path, player_name=p[1], player_sport=p[2],
                      report=report, weekly_summary=weekly, sim_mode=False)
            QMessageBox.information(self,"Export réussi", f"PDF sauvegardé :\n{path}")
        except Exception as e:
            QMessageBox.critical(self,"Erreur export",str(e))

    def _export_team_pdf(self):
        if self._team_df is None or self._team_df.empty:
            QMessageBox.warning(self,"Aucune donnée","Lancez d'abord l'analyse équipe."); return
        outdir = QFileDialog.getExistingDirectory(self,"Dossier de destination")
        if not outdir: return

        prog = QProgressDialog("Génération des rapports PDF…", "", 0,
                               len(self._team_df), self)
        prog.setWindowModality(Qt.WindowModal); prog.show()
        advisor = PerformanceAdvisor(db_path=DB_PATH)
        from ml.longitudinal import weekly_summary
        n = 0
        for _, row in self._team_df.iterrows():
            prog.setValue(n); QApplication.processEvents()
            if prog.wasCanceled(): break
            try:
                report = advisor.advise(int(row["player_id"]))
                weekly = weekly_summary(int(row["player_id"]))
                fname = os.path.join(outdir, f"rapport_{row['name'].replace(' ','_')}.pdf")
                export_pdf(path=fname, player_name=row["name"],
                          player_sport=row["sport"], report=report,
                          weekly_summary=weekly, sim_mode=False)
            except Exception as e:
                print(f"Erreur export {row['name']}: {e}")
            n += 1
        prog.setValue(len(self._team_df))
        QMessageBox.information(self,"Export terminé", f"{n} rapports générés dans :\n{outdir}")


def main():
    app = QApplication(sys.argv)
    apply_theme(app, dark=True)

    login = LoginDialog()
    if login.exec_() != QDialog.Accepted:
        sys.exit(0)

    win = MainWindow(login.user)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
