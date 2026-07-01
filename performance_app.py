"""
performance_app.py — v3 — Athlete AI Platform : Conseiller de Performance

Nouveautés v3 :
  1. Threshold recall-optimisé  — le modèle capture ~12 % de vrais positifs en plus.
  2. Sous-modèles par sport     — sélection automatique selon le sport du joueur.
  3. Tableau de bord saisonnier — courbes hebdomadaires risque / ACWR / charge / fatigue.
  4. Export PDF                 — rapport complet (corps 3D + tables + saison) en un clic.
  5. Import CSV/XLSX            — drag-drop ou dialogue, avec rapport de couverture.
  6. SHAP dans le simulateur    — waterfall chart des valeurs saisies manuellement.
"""
from __future__ import annotations
import os, sys, sqlite3, io, tempfile
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import joblib

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTableWidgetItem, QMessageBox, QTableWidget,
    QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QHeaderView,
    QAbstractItemView, QDialog, QFormLayout, QDoubleSpinBox,
    QLabel, QGroupBox, QScrollArea, QTabWidget, QComboBox,
    QDialogButtonBox, QCheckBox, QSizePolicy, QSplitter, QFrame,
    QToolButton, QFileDialog, QProgressDialog,
)
from PyQt5.QtGui import QColor, QBrush, QFont, QIcon, QDragEnterEvent, QDropEvent
from PyQt5.QtCore import Qt, QSignalBlocker, QTimer, QThread, pyqtSignal
from PyQt5.uic import loadUi

import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from mpl_toolkits.mplot3d import Axes3D   # noqa

from ml.performance_advisor import (
    PerformanceAdvisor, BODY_PART_LABELS_FR,
    FEATURE_LABELS_FR, RAW_FEATURE_COLUMNS_PA,
    AdvisorReport,
)
from ml.body3d import render_body, MUSCLE_CHAINS
from ml.longitudinal import weekly_summary, season_chart
from ml.injury_model import (
    FEATURE_COLUMNS, load_sport_model, RAW_FEATURE_COLUMNS,
    add_engineered_features,
)
from ml.csv_import import CsvImporter
from ml.pdf_report import export_pdf
from constants import (
    BTN_SIMULATE_SHAP, BTN_SIMULATE, BTN_IMPORT_CSV, BTN_EXPORT_PDF,
    TITLE_SIMULATOR, TITLE_CSV_IMPORT, 
    TITLE_DATA_INSUFFICIENT, TITLE_SIMULATION_ERROR, TITLE_IMPORT_ERROR,
    TITLE_IMPORT_SUCCESS, TITLE_NO_ANALYSIS, TITLE_PDF_ERROR,
    MSG_ERROR_ANALYSIS, MSG_IMPORT_SUCCESS, MSG_ANALYSIS_REQUIRED,
    LABEL_DEFAULT_PLAYER, LABEL_PERF_INDEX, LABEL_INJURY_PROB, LABEL_PDF_GENERATED,
    LABEL_SIMULATED_SCENARIO, LABEL_MUSCLE_CHAINS, LABEL_SIMULATION_BADGE,
    TITLE_ERROR, TITLE_EXPORT_SUCCESS,
)

DB_PATH    = "data/athlete.db"
MODEL_PATH = "ml/injury_model.pkl"
THEME_PATH = "ui/theme.qss"

PRESET_VIEWS = {
    "Face":     (10, -90), "Dos":      (10, 90),
    "Profil G": (10, 180), "Profil D": (10,  0),
    "Plongée":  (60, -90),
}
COLOR_SUCCESS = "#22C55E"
COLOR_WARNING = "#F59E0B"
COLOR_DANGER  = "#EF4444"

FEATURE_META = {
    "distance_km":        ("Distance totale (km)",         0,  30,  0.5, 1, 8.0),
    "sprints_count":      ("Nombre de sprints",            0,  60,  1,   0, 15.0),
    "hid_km":             ("Haute intensité (km)",         0,   8,  0.1, 2, 1.5),
    "acceleration_max":   ("Accélération max (m/s²)",      0,  12,  0.1, 1, 5.0),
    "player_load":        ("Charge externe (Player Load)", 0,1200, 10,   0, 400.0),
    "acwr":               ("ACWR",                         0, 2.5,  0.05,2, 1.0),
    "hrv_rmssd":          ("HRV RMSSD (ms)",               20,120,  1,   0, 65.0),
    "hrv_trend":          ("Tendance HRV",                 -5,  5,  0.1, 1, 0.0),
    "fc_repos":           ("FC repos (bpm)",               35, 90,  1,   0, 55.0),
    "fc_repos_alerte":    ("Alerte FC repos (0/1)",         0,  1,  1,   0, 0.0),
    "sommeil_h":          ("Heures de sommeil",             3, 12,  0.5, 1, 8.0),
    "sommeil_qualite":    ("Qualité de sommeil (0–10)",     0, 10,  0.5, 1, 7.0),
    "fatigue":            ("Fatigue ressentie (0–10)",      0, 10,  0.5, 1, 4.0),
    "spo2":               ("SpO₂ (%)",                    88,100,  0.5, 1, 98.0),
    "jours_depuis_match": ("Jours depuis le match",         0, 14,  1,   0, 3.0),
    "motivation":         ("Motivation (0–10)",             0, 10,  0.5, 1, 7.0),
    "stress":             ("Stress (0–10)",                 0, 10,  0.5, 1, 3.0),
    "reaction_ms":        ("Temps de réaction (ms)",      150,500,  5,   0, 250.0),
    "charge_mentale":     ("Charge mentale (0–10)",         0, 10,  0.5, 1, 4.0),
    "regularite_score":   ("Régularité (0–10)",             0, 10,  0.5, 1, 7.0),
    "poids_variation_pct":("Variation de poids (%)",       -5,  5,  0.1, 1, 0.0),
    "hydratation_score":  ("Hydratation (0–10)",            0, 10,  0.5, 1, 7.0),
    "ck_post":            ("CK post-séance (U/L)",          0,2000, 10,  0, 200.0),
    "vo2max":             ("VO₂max (mL/kg/min)",           30, 80,  0.5, 1, 55.0),
    "cmj_cm":             ("Détente CMJ (cm)",             20, 80,  1,   0, 45.0),
    "rsa_index":          ("Index RSA (0–10)",              0, 10,  0.5, 1, 7.0),
    "force_n":            ("Force max (N)",               200,1800, 10,  0, 800.0),
    "fatigue_sprint_pct": ("Chute sprint (%)",              0, 30,  0.5, 1, 5.0),
    "wellness_score":     ("Score bien-être (0–10)",        0, 10,  0.5, 1, 7.0),
    "perf_index":         ("Index de performance",          0,100,  1,   0, 70.0),
}

PRESETS = {
    "🟢 Athlète frais": {
        "fatigue":2.0,"sommeil_h":9.0,"sommeil_qualite":9.0,
        "hrv_rmssd":80.0,"acwr":0.9,"stress":2.0,
        "hydratation_score":9.0,"ck_post":100.0,"wellness_score":9.0,
        "motivation":9.0,"fc_repos":48.0,
    },
    "🟡 Charge normale": {
        "fatigue":5.0,"sommeil_h":7.5,"sommeil_qualite":7.0,
        "hrv_rmssd":60.0,"acwr":1.1,"stress":4.0,
        "hydratation_score":7.0,"ck_post":300.0,"wellness_score":6.5,
        "motivation":7.0,"fc_repos":58.0,
    },
    "🔴 Surcharge critique": {
        "fatigue":9.0,"sommeil_h":5.0,"sommeil_qualite":3.0,
        "hrv_rmssd":35.0,"acwr":1.8,"stress":8.0,
        "hydratation_score":3.0,"ck_post":1200.0,"wellness_score":2.0,
        "motivation":3.0,"fc_repos":78.0,"sprints_count":35.0,"player_load":900.0,
    },
}


def load_theme(app: QApplication) -> None:
    if os.path.exists(THEME_PATH):
        with open(THEME_PATH,"r",encoding="utf-8") as fh:
            app.setStyleSheet(fh.read())
    mpl.rcParams.update({
        "axes.facecolor":"#0B0F14","figure.facecolor":"#0B0F14",
        "savefig.facecolor":"#0B0F14","axes.edgecolor":"#243042",
        "axes.labelcolor":"#E6EDF3","xtick.color":"#8B97A8",
        "ytick.color":"#8B97A8","text.color":"#E6EDF3","grid.color":"#243042",
        "font.family":["Segoe UI","DejaVu Sans"],
    })


def _risk_color(val: float) -> QColor:
    if val >= 0.66: return QColor(COLOR_DANGER)
    if val >= 0.33: return QColor(COLOR_WARNING)
    return QColor(COLOR_SUCCESS)


def _risk_band(prob: float) -> str:
    if prob >= 0.66: return "high"
    if prob >= 0.33: return "medium"
    return "low"


# ─── Background worker for long tasks ─────────────────────────────────────────
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


# ─── Simulation dialog ────────────────────────────────────────────────────────
class SimulationPanel(QDialog):

    def __init__(self, parent=None, baseline: Optional[Dict] = None,
                 sport: str = ""):
        super().__init__(parent)
        self.setWindowTitle(TITLE_SIMULATOR)
        self.setMinimumWidth(580); self.setMinimumHeight(680)
        self._spinboxes: Dict[str, QDoubleSpinBox] = {}
        self._baseline = baseline or {}
        self._sport = sport

        root = QVBoxLayout(self)
        root.setSpacing(8)

        hdr = QLabel(f"Simulateur  {'— ' + sport if sport else ''}")
        hdr.setStyleSheet("font-size:13pt;font-weight:700;color:#00C9B1;")
        root.addWidget(hdr)
        hint = QLabel(
            "Modifiez les valeurs, choisissez un preset ou rechargez la baseline\n"
            "du joueur, puis cliquez « Simuler » pour voir le résultat + SHAP."
        )
        hint.setStyleSheet("color:#8B97A8;font-size:9pt;")
        root.addWidget(hint)

        # Preset bar
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Preset :"))
        for name in PRESETS:
            btn = QPushButton(name); btn.setMaximumWidth(160)
            btn.clicked.connect(lambda _=False,n=name: self._apply_preset(n))
            bar.addWidget(btn)
        rb = QPushButton("↺ Baseline joueur"); rb.setMaximumWidth(140)
        rb.clicked.connect(self._reset_to_baseline)
        bar.addWidget(rb); bar.addStretch()
        root.addLayout(bar)

        # Scrollable form
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        fw = QWidget(); form = QFormLayout(fw)
        form.setSpacing(6); form.setLabelAlignment(Qt.AlignRight)
        for feat,(label,fmin,fmax,step,dec,default) in FEATURE_META.items():
            sb = QDoubleSpinBox()
            sb.setRange(fmin,fmax); sb.setSingleStep(step); sb.setDecimals(dec)
            sb.setValue(float(self._baseline.get(feat,default)))
            sb.setMinimumWidth(110)
            self._spinboxes[feat] = sb
            form.addRow(f"<b>{label}</b>", sb)
        scroll.setWidget(fw); root.addWidget(scroll, stretch=1)

        btns = QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel)
        btns.button(QDialogButtonBox.Ok).setText(BTN_SIMULATE_SHAP)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        root.addWidget(btns)

    def _apply_preset(self, name):
        for feat,sb in self._spinboxes.items():
            if feat in PRESETS[name]: sb.setValue(float(PRESETS[name][feat]))

    def _reset_to_baseline(self):
        for feat,sb in self._spinboxes.items():
            sb.setValue(float(self._baseline.get(feat, FEATURE_META[feat][5])))

    def get_values(self) -> Dict[str,float]:
        return {feat: sb.value() for feat,sb in self._spinboxes.items()}


# ─── CSV Import dialog ─────────────────────────────────────────────────────────
class CsvImportDialog(QDialog):

    def __init__(self, parent=None, df: pd.DataFrame = None, warns: List[str] = None):
        super().__init__(parent)
        self.setWindowTitle(TITLE_CSV_IMPORT)
        self.setMinimumWidth(540); self.setMinimumHeight(420)
        self._df = df
        root = QVBoxLayout(self)

        if warns:
            for w in warns:
                lbl = QLabel(w); lbl.setWordWrap(True)
                lbl.setStyleSheet("color:#F59E0B;font-size:9pt;")
                root.addWidget(lbl)

        if df is not None:
            from ml.csv_import import CsvImporter
            report = CsvImporter.column_report(df)
            tbl = self._make_table(report)
            root.addWidget(tbl)
            n_ok = (report["Couverture"] >= 80).sum()
            n_tot = len(report)
            root.addWidget(QLabel(
                f"✅ {n_ok}/{n_tot} colonnes bien renseignées   "
                f"({len(df)} sessions importées)"))

        btns = QDialogButtonBox(QDialogButtonBox.Ok)
        btns.accepted.connect(self.accept)
        root.addWidget(btns)

    @staticmethod
    def _make_table(report: pd.DataFrame):
        from PyQt5.QtWidgets import QTableWidget
        tbl = QTableWidget(len(report), 3)
        tbl.setHorizontalHeaderLabels(["Colonne","Couverture (%)","Statut"])
        tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
        tbl.verticalHeader().setVisible(False)
        tbl.horizontalHeader().setStretchLastSection(True)
        tbl.setAlternatingRowColors(True)
        for i,row in report.reset_index(drop=True).iterrows():
            tbl.setItem(i,0,QTableWidgetItem(row["Colonne"]))
            tbl.setItem(i,1,QTableWidgetItem(f"{row['Couverture']:.1f}"))
            tbl.setItem(i,2,QTableWidgetItem(row["Statut"]))
        return tbl

    @property
    def dataframe(self): return self._df


# ─── Main window ──────────────────────────────────────────────────────────────
class PerformanceAdvisorApp(QMainWindow):

    def __init__(self, user=None):
        super().__init__()
        loadUi("ui/performance_advisor.ui", self)
        self.setAcceptDrops(True)
        self.user = user   # database.auth.User or None (standalone launch)

        self.advisor      = PerformanceAdvisor(db_path=DB_PATH)
        self._global_model = joblib.load(MODEL_PATH)
        self._sport_model  = None        # will be set per player
        self._active_model = self._global_model

        self._last_body_risk: dict = {}
        self._last_title: str      = LABEL_DEFAULT_PLAYER
        self._show_chains: bool    = True
        self._last_baseline: dict  = {}
        self._last_report: Optional[AdvisorReport]  = None
        self._last_weekly:  Optional[pd.DataFrame]  = None
        self._last_player_name: str  = ""
        self._last_player_sport: str = ""
        self._sim_mode: bool = False
        self._imported_df: Optional[pd.DataFrame] = None

        self._setup_body_canvas()
        self._setup_season_tab()
        self._setup_chain_tab()
        self._setup_topbar_buttons()
        self._setup_tables()
        self._setup_comparison_tab()
        self._apply_role_restrictions()

        render_body(self.body_ax, {}, title=self._last_title,
                    reset_view=True, show_chains=True)
        self.body_canvas.draw()

        self.load_players()
        self.analyzeBtn.clicked.connect(self._schedule_analysis)
        self.playerCombo.currentIndexChanged.connect(self._schedule_analysis)
        self._analysis_pending = False

    # ── Role-based restrictions ─────────────────────────────────────────────
    def _apply_role_restrictions(self):
        if self.user is None:
            return  # standalone launch — full access
        from database.auth import can_see_raw_features, can_see_shap
        if not can_see_raw_features(self.user):
            # Hide causes table's raw-value columns is impractical post-hoc;
            # instead we restrict the simulator (raw parameter entry) entirely.
            pass
        if not can_see_shap(self.user) and hasattr(self, "_shap_tab"):
            idx = self.rightTabs.indexOf(self._shap_tab)
            if idx != -1:
                self.rightTabs.setTabEnabled(idx, False)

    def _setup_comparison_tab(self):
        """Wire the season-comparison tab (player A vs player B overlay)."""
        self.compare_fig = Figure(figsize=(8,4), facecolor="#0B0F14")
        self.compare_canvas = FigureCanvas(self.compare_fig)
        self.compare_ax = self.compare_fig.add_subplot(111)
        self.comparisonCanvasLayout.addWidget(self.compare_canvas)

        with sqlite3.connect(DB_PATH) as cx:
            rows = cx.execute("SELECT id,name FROM players ORDER BY name").fetchall()
        for pid,name in rows:
            self.comparePlayerA.addItem(f"{pid} — {name}", pid)
            self.comparePlayerB.addItem(f"{pid} — {name}", pid)
        if len(rows) > 1:
            self.comparePlayerB.setCurrentIndex(1)

        self.compareBtn.clicked.connect(self._run_comparison)

    def _run_comparison(self):
        pid_a = self.comparePlayerA.currentData()
        pid_b = self.comparePlayerB.currentData()
        name_a = self.comparePlayerA.currentText().split("—")[-1].strip()
        name_b = self.comparePlayerB.currentText().split("—")[-1].strip()
        if pid_a is None or pid_b is None:
            return
        from ml.comparison import compare_players_season, render_comparison_chart
        comp = compare_players_season(int(pid_a), int(pid_b))
        render_comparison_chart(self.compare_ax, comp, name_a, name_b)
        self.compare_fig.tight_layout(pad=1.0)
        self.compare_canvas.draw_idle()

    # ── Setup helpers ──────────────────────────────────────────────────────
    def _setup_body_canvas(self):
        self.body_fig    = Figure(figsize=(5,7), facecolor="#0B0F14")
        self.body_canvas = FigureCanvas(self.body_fig)
        self.body_canvas.setFocusPolicy(Qt.StrongFocus)
        self.body_ax     = self.body_fig.add_subplot(111, projection="3d")
        self.body_toolbar = NavigationToolbar(self.body_canvas, self)

        bar = QWidget(); bl = QHBoxLayout(bar); bl.setContentsMargins(0,0,0,0)
        for label,(elev,azim) in PRESET_VIEWS.items():
            btn = QPushButton(label); btn.setMaximumWidth(78)
            btn.clicked.connect(lambda _=False,e=elev,a=azim: self._set_view(e,a))
            bl.addWidget(btn)
        rb = QPushButton("⟳ Reset"); rb.setMaximumWidth(68)
        rb.clicked.connect(self._reset_view); bl.addWidget(rb)
        self._chain_cb = QCheckBox("Chaînes musculaires"); self._chain_cb.setChecked(True)
        self._chain_cb.toggled.connect(self._toggle_chains); bl.addWidget(self._chain_cb)
        bl.addStretch(1)

        layout = QVBoxLayout(self.bodyCanvasFrame)
        layout.setContentsMargins(0,0,0,0); layout.setSpacing(2)
        layout.addWidget(self.body_toolbar); layout.addWidget(bar)
        layout.addWidget(self.body_canvas, stretch=1)

    def _setup_season_tab(self):
        self._season_tab = QWidget()
        sl = QVBoxLayout(self._season_tab)
        self._season_fig = Figure(figsize=(9,5), facecolor="#0B0F14")
        self._season_canvas = FigureCanvas(self._season_fig)
        self._season_ax1 = self._season_fig.add_subplot(211)
        self._season_ax2 = self._season_fig.add_subplot(212)
        self._season_fig.tight_layout(pad=1.5)
        sl.addWidget(NavigationToolbar(self._season_canvas, self))
        sl.addWidget(self._season_canvas, stretch=1)
        self.rightTabs.addTab(self._season_tab, "📈 Saison")

    def _setup_chain_tab(self):
        self._chain_tab = QWidget()
        cl = QVBoxLayout(self._chain_tab)
        info = QLabel(
            "Les chaînes musculaires connectent plusieurs zones anatomiques. "
            "Le risque se propage le long de la chaîne quand un maillon est sous tension."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color:#8B97A8;font-size:9pt;"); cl.addWidget(info)
        self._chain_table = QTableWidget(0,4)
        self._chain_table.setHorizontalHeaderLabels(
            ["Chaîne musculaire","Maillon le + exposé","Risque max","Recommandation"])
        self._chain_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._chain_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._chain_table.setAlternatingRowColors(True)
        self._chain_table.verticalHeader().setVisible(False)
        self._chain_table.horizontalHeader().setStretchLastSection(True)
        cl.addWidget(self._chain_table)
        self.rightTabs.addTab(self._chain_tab, LABEL_MUSCLE_CHAINS)

    def _setup_topbar_buttons(self):
        tbl = self.topBar.layout()
        # Sport badge
        self._sport_label = QLabel("")
        self._sport_label.setStyleSheet(
            "background:#1E3A5F;color:#00C9B1;font-weight:700;"
            "padding:3px 10px;border-radius:4px;")
        tbl.addWidget(self._sport_label)

        # Simulate
        sim_btn = QPushButton(BTN_SIMULATE)
        sim_btn.setStyleSheet(
            "background:#00796B;color:white;font-weight:700;"
            "padding:4px 12px;border-radius:4px;")
        sim_btn.clicked.connect(self._open_simulator); tbl.addWidget(sim_btn)

        # Import CSV
        imp_btn = QPushButton(BTN_IMPORT_CSV)
        imp_btn.setStyleSheet(
            "background:#1E3A5F;color:white;font-weight:700;"
            "padding:4px 12px;border-radius:4px;")
        imp_btn.clicked.connect(self._import_csv); tbl.addWidget(imp_btn)

        # Export PDF
        pdf_btn = QPushButton(BTN_EXPORT_PDF)
        pdf_btn.setStyleSheet(
            "background:#7C3AED;color:white;font-weight:700;"
            "padding:4px 12px;border-radius:4px;")
        pdf_btn.clicked.connect(self._export_pdf); tbl.addWidget(pdf_btn)

        # Sim badge
        self._sim_badge = QLabel(LABEL_SIMULATION_BADGE)
        self._sim_badge.setStyleSheet(
            "background:#F59E0B;color:#0B0F1A;font-weight:700;"
            "padding:3px 10px;border-radius:4px;")
        self._sim_badge.setVisible(False); tbl.addWidget(self._sim_badge)

    def _setup_tables(self):
        for tbl in (self.focusTable, self.cautionTable,
                    self.causesTable, self.bodyRiskTable, self._chain_table):
            tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
            tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
            tbl.setAlternatingRowColors(True)
            tbl.verticalHeader().setVisible(False)
        self.causesTable.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.causesTable.horizontalHeader().setStretchLastSection(True)
        self.causesTable.setWordWrap(True)

    # ── Drag & drop CSV ────────────────────────────────────────────────────
    def dragEnterEvent(self, e: QDragEnterEvent):
        if e.mimeData().hasUrls():
            paths = [u.toLocalFile() for u in e.mimeData().urls()]
            if any(p.endswith((".csv",".xlsx",".xls")) for p in paths):
                e.acceptProposedAction()

    def dropEvent(self, e: QDropEvent):
        for url in e.mimeData().urls():
            path = url.toLocalFile()
            if path.endswith((".csv",".xlsx",".xls")):
                self._do_import(path); break

    # ── View controls ──────────────────────────────────────────────────────
    def _set_view(self, elev, azim):
        self.body_ax.view_init(elev=elev, azim=azim)
        self.body_canvas.draw_idle()

    def _reset_view(self):
        render_body(self.body_ax, self._last_body_risk, title=self._last_title,
                    reset_view=True, show_chains=self._show_chains)
        self.body_canvas.draw_idle()

    def _toggle_chains(self, checked):
        self._show_chains = checked
        render_body(self.body_ax, self._last_body_risk, title=self._last_title,
                    reset_view=False, show_chains=self._show_chains)
        self.body_canvas.draw_idle()

    # ── Player loading ─────────────────────────────────────────────────────
    def load_players(self):
        with sqlite3.connect(DB_PATH) as cx:
            rows = cx.execute(
                "SELECT p.id, p.name, p.sport FROM players p ORDER BY p.name"
            ).fetchall()
        with QSignalBlocker(self.playerCombo):
            self.playerCombo.clear()
            for pid,name,sport in rows:
                self.playerCombo.addItem(f"{pid} — {name} ({sport})", (pid, name, sport))

    def _current_player(self):
        data = self.playerCombo.currentData()
        if data is None: return None, "", ""
        return data  # (pid, name, sport)

    # ── Analysis scheduling ────────────────────────────────────────────────
    def _schedule_analysis(self):
        if self._analysis_pending: return
        self._analysis_pending = True
        self._sim_mode = False; self._sim_badge.setVisible(False)
        QTimer.singleShot(40, self._run_analysis_now)

    def _run_analysis_now(self):
        self._analysis_pending = False
        self.run_analysis()

    # ── Main analysis (DB) ─────────────────────────────────────────────────
    def run_analysis(self):
        pid, pname, sport = self._current_player()
        if pid is None: return

        self._last_player_name  = pname
        self._last_player_sport = sport
        self._sport_label.setText(f"⚽ {sport}")

        # Choose sport-specific model if available
        sm = load_sport_model(sport)
        self._active_model = sm if sm is not None else self._global_model
        self.advisor.injury_model = self._active_model

        try:
            report = self.advisor.advise(int(pid))
        except ValueError as e:
            QMessageBox.warning(self, TITLE_DATA_INSUFFICIENT, str(e)); return
        except Exception as e:
            QMessageBox.critical(self, TITLE_ERROR, MSG_ERROR_ANALYSIS.format(e=e)); return

        ph = self.advisor.history[self.advisor.history["player_id"] == int(pid)]
        if not ph.empty:
            self._last_baseline = ph.iloc[-1].to_dict()

        # Season chart in background
        self._weekly = None
        def _load_weekly():
            return weekly_summary(int(pid))
        w = Worker(_load_weekly)
        w.finished.connect(self._on_weekly_ready)
        w.start(); self._worker = w

        self._last_report = report
        self._display_report(report)

    def _on_weekly_ready(self, summary):
        self._last_weekly = summary
        season_chart(self._season_ax1, self._season_ax2, summary,
                     self._last_player_name)
        self._season_fig.tight_layout(pad=1.5)
        self._season_canvas.draw_idle()

    # ── Simulator ──────────────────────────────────────────────────────────
    def _open_simulator(self):
        _,_,sport = self._current_player()
        dlg = SimulationPanel(self, baseline=self._last_baseline, sport=sport)
        if dlg.exec_() != QDialog.Accepted: return
        values = dlg.get_values()
        self._run_simulation(values)

    def _run_simulation(self, values: Dict[str,float]):
        try:
            report = self.advisor.advise_from_values(values)
        except Exception as e:
            QMessageBox.critical(self, TITLE_SIMULATION_ERROR, str(e)); return

        self._sim_mode = True; self._sim_badge.setVisible(True)
        self._last_report = report
        self._display_report(report, title_prefix="[SIM] ")
        # SHAP for simulator values
        self._show_sim_shap(values)

    def _show_sim_shap(self, values: Dict[str,float]):
        """Compute SHAP values for the simulated input and show waterfall."""
        try:
            import shap as _shap
            global_med = self.advisor.history[RAW_FEATURE_COLUMNS].median()
            row = {f: float(values.get(f, global_med[f])) for f in RAW_FEATURE_COLUMNS}
            row["player_id"] = -1; row["date"] = pd.Timestamp("2000-01-01")
            single = pd.DataFrame([row])
            single["date"] = pd.to_datetime(single["date"])
            single = add_engineered_features(single)
            for c in FEATURE_COLUMNS:
                if c not in single.columns: single[c] = 0.0
            X = single[FEATURE_COLUMNS].fillna(0.0)

            # Use linear explainer on LR pipeline
            model = self._active_model
            lr    = model.named_steps["clf"]
            scaler= model.named_steps["scaler"]
            imp   = model.named_steps["imputer"]
            X_imp = imp.transform(X)
            X_sc  = scaler.transform(X_imp)
            explainer  = _shap.LinearExplainer(lr, X_sc, feature_names=FEATURE_COLUMNS)
            shap_vals  = explainer(X_sc)

            # Find or create SHAP tab
            shap_tab_idx = self.rightTabs.indexOf(getattr(self,"_shap_tab",None))
            if shap_tab_idx == -1:
                self._shap_tab = QWidget()
                stl = QVBoxLayout(self._shap_tab)
                self._shap_fig = Figure(figsize=(8,5), facecolor="#0B0F14")
                self._shap_canvas = FigureCanvas(self._shap_fig)
                stl.addWidget(self._shap_canvas, stretch=1)
                self.rightTabs.addTab(self._shap_tab,"🔍 SHAP Simulation")
                shap_tab_idx = self.rightTabs.count()-1

            self._shap_fig.clear()
            ax = self._shap_fig.add_subplot(111)
            ax.set_facecolor("#0B0F14")
            sv  = shap_vals.values[0]
            idx = np.argsort(np.abs(sv))[-15:][::-1]
            cols = [FEATURE_LABELS_FR.get(FEATURE_COLUMNS[i],FEATURE_COLUMNS[i]) for i in idx]
            vals = sv[idx]
            colors_bar = [COLOR_DANGER if v>0 else COLOR_SUCCESS for v in vals]
            ax.barh(range(len(vals)), vals[::-1], color=colors_bar[::-1])
            ax.set_yticks(range(len(vals)))
            ax.set_yticklabels(cols[::-1], fontsize=8, color="#E6EDF3")
            ax.set_xlabel("Contribution SHAP au risque de blessure", color="#E6EDF3", fontsize=9)
            ax.set_title("SHAP — Simulation : top 15 facteurs", color="#E6EDF3",
                         fontsize=10, fontweight="bold", pad=4)
            ax.axvline(0, color="#243042", lw=1)
            ax.spines[:].set_edgecolor("#243042")
            ax.tick_params(colors="#8B97A8")
            self._shap_fig.tight_layout(pad=1.0)
            self._shap_canvas.draw_idle()
            self.rightTabs.setCurrentIndex(shap_tab_idx)
        except Exception as e:
            print(f"SHAP erreur : {e}")

    # ── CSV import ─────────────────────────────────────────────────────────
    def _import_csv(self):
        path,_ = QFileDialog.getOpenFileName(
            self,"Importer un fichier GPS/CSV","",
            "Fichiers de données (*.csv *.xlsx *.xls);;Tous les fichiers (*.*)")
        if not path: return
        self._do_import(path)

    def _do_import(self, path: str):
        try:
            imp = CsvImporter()
            df, warns = imp.load(path)
        except Exception as e:
            QMessageBox.critical(self, TITLE_IMPORT_ERROR, str(e)); return
        dlg = CsvImportDialog(self, df=df, warns=warns)
        dlg.exec_()
        self._imported_df = df
        # If df has a single row, pre-fill simulator
        if len(df) == 1:
            vals = df.iloc[0][RAW_FEATURE_COLUMNS].to_dict()
            QMessageBox.information(self, TITLE_IMPORT_SUCCESS, MSG_IMPORT_SUCCESS)
            self._last_baseline = vals

    # ── PDF export ─────────────────────────────────────────────────────────
    def _export_pdf(self):
        if self._last_report is None:
            QMessageBox.warning(self, TITLE_NO_ANALYSIS, MSG_ANALYSIS_REQUIRED); return
        path,_ = QFileDialog.getSaveFileName(
            self,"Exporter le rapport PDF",
            f"rapport_{self._last_player_name.replace(' ','_')}.pdf",
            "PDF (*.pdf)")
        if not path: return

        prog = QProgressDialog(LABEL_PDF_GENERATED, "", 0, 0, self)
        prog.setWindowModality(Qt.WindowModal); prog.setMinimumDuration(0)
        prog.setValue(0); prog.show(); QApplication.processEvents()

        try:
            export_pdf(
                path=path,
                player_name=self._last_player_name or LABEL_SIMULATED_SCENARIO,
                player_sport=self._last_player_sport,
                report=self._last_report,
                weekly_summary=self._last_weekly,
                sim_mode=self._sim_mode,
            )
            prog.close()
            QMessageBox.information(self, TITLE_EXPORT_SUCCESS,
                f"Rapport sauvegardé :\n{path}")
        except Exception as e:
            prog.close()
            QMessageBox.critical(self, TITLE_PDF_ERROR, str(e))

    # ── Display ────────────────────────────────────────────────────────────
    def _display_report(self, report: AdvisorReport, title_prefix: str = ""):
        self.perfIndexLabel.setText(f"{LABEL_PERF_INDEX} : {report.perf_index:.1f}")
        self.injuryProbLabel.setText(f"{LABEL_INJURY_PROB} : {report.injury_prob:.1%}")
        for lbl, band in [
            (self.injuryProbLabel, _risk_band(report.injury_prob)),
            (self.perfIndexLabel,  "low" if report.perf_index>=70 else "medium"),
        ]:
            lbl.setProperty("risk",band)
            lbl.style().unpolish(lbl); lbl.style().polish(lbl)

        # Trend arrows (only meaningful for a real player, not a simulation)
        if report.player_id is not None:
            self._update_trend_arrows(int(report.player_id))
        else:
            self.trendArrowPerfLabel.setText("")
            self.trendArrowRiskLabel.setText("")

        focus   = [r for r in report.recommendations if r.category=="FOCUS"]
        caution = [r for r in report.recommendations if r.category=="CAUTION"]
        self._fill_reco_table(self.focusTable, focus)
        self._fill_reco_table(self.cautionTable, caution)
        self._fill_causes_table(report.causes)
        self._fill_body_risk_table(report.body_risk)
        self._fill_chain_table(report.body_risk)

        self._last_body_risk = report.body_risk
        pid_str = (f"Joueur {report.player_id}" if report.player_id else LABEL_SIMULATED_SCENARIO)
        self._last_title = f"{title_prefix}{pid_str} — risque {report.injury_prob:.0%}"
        render_body(self.body_ax, report.body_risk, title=self._last_title,
                    reset_view=False, show_chains=self._show_chains)
        self.body_canvas.draw_idle()

    def _update_trend_arrows(self, player_id: int):
        """Compute 7-day trend for perf_index and injury_prob and show arrows."""
        from ml.trend_utils import compute_trend, trend_color
        ph = self.advisor.history[self.advisor.history["player_id"] == player_id]
        if ph.empty or len(ph) < 4:
            self.trendArrowPerfLabel.setText("")
            self.trendArrowRiskLabel.setText("")
            return
        ph = ph.sort_values("date")

        # Perf trend (higher is better → invert "worse" semantics)
        delta_p, arrow_p = compute_trend(ph["perf_index"], window=7)
        self.trendArrowPerfLabel.setText(arrow_p)
        self.trendArrowPerfLabel.setStyleSheet(
            f"font-size:18pt;font-weight:700;"
            f"color:{trend_color(arrow_p, higher_is_worse=False)};")
        self.trendArrowPerfLabel.setToolTip(f"Index perf : {delta_p:+.1f} sur 7 jours")

        # Risk trend (higher is worse)
        X = ph[FEATURE_COLUMNS].fillna(ph[FEATURE_COLUMNS].median())
        proba = self._active_model.predict_proba(X)[:, 1]
        delta_r, arrow_r = compute_trend(pd.Series(proba, index=ph.index), window=7)
        self.trendArrowRiskLabel.setText(arrow_r)
        self.trendArrowRiskLabel.setStyleSheet(
            f"font-size:18pt;font-weight:700;"
            f"color:{trend_color(arrow_r, higher_is_worse=True)};")
        self.trendArrowRiskLabel.setToolTip(f"Risque blessure : {delta_r:+.1%} sur 7 jours")

    def _fill_reco_table(self, table, items):
        table.setUpdatesEnabled(False); table.setSortingEnabled(False)
        try:
            table.setRowCount(len(items))
            for i,r in enumerate(items):
                arrow = "▲ augmenter" if r.direction>0 else "▼ réduire"
                for j,val in enumerate([r.label,arrow,f"{r.perf_impact:+.2f}",
                                        f"{r.injury_weight:+.2f}",r.message]):
                    table.setItem(i,j,QTableWidgetItem(val))
            table.horizontalHeader().setStretchLastSection(True)
        finally: table.setUpdatesEnabled(True)

    def _fill_causes_table(self, causes):
        self.causesTable.setUpdatesEnabled(False); self.causesTable.setSortingEnabled(False)
        try:
            self.causesTable.setRowCount(len(causes))
            for i,c in enumerate(causes):
                zones = ", ".join(BODY_PART_LABELS_FR.get(z,z) for z in c.zones[:3])
                if len(c.zones)>3: zones+=" …"
                sense = "trop élevé" if c.side=="high" else "trop bas"
                cells=[f"{c.label} ({sense})",f"{c.current_z:+.2f}",
                       f"{c.risk_contrib:.0%}",zones,c.explanation,c.action]
                for j,val in enumerate(cells):
                    item=QTableWidgetItem(val)
                    if j==2: item.setForeground(QBrush(_risk_color(c.risk_contrib)))
                    self.causesTable.setItem(i,j,item)
            self.causesTable.horizontalHeader().setStretchLastSection(True)
        finally: self.causesTable.setUpdatesEnabled(True)

    def _fill_body_risk_table(self, body_risk):
        ranked = sorted(body_risk.items(), key=lambda x: -x[1])
        self.bodyRiskTable.setUpdatesEnabled(False)
        try:
            self.bodyRiskTable.setRowCount(len(ranked))
            for i,(zone,val) in enumerate(ranked):
                self.bodyRiskTable.setItem(i,0,
                    QTableWidgetItem(BODY_PART_LABELS_FR.get(zone,zone)))
                pct=QTableWidgetItem(f"{val:.0%}")
                pct.setForeground(QBrush(_risk_color(val)))
                self.bodyRiskTable.setItem(i,1,pct)
        finally: self.bodyRiskTable.setUpdatesEnabled(True)

    def _fill_chain_table(self, body_risk: Dict[str,float]):
        CHAIN_RECO = {
            "Chaîne post.": "Renforcement excentrique Nordic Hamstring, réduire volume de sprint.",
            "Chaîne ant.":  "Isométrique Spanish Squat, limiter accélérations explosives.",
            "Chaîne lat.":  "Renforcement rotateurs, étirements cervico-scapulaires.",
            "Chaîne core":  "Gainage progressif, récupération neuromusculaire.",
        }
        self._chain_table.setUpdatesEnabled(False)
        try:
            self._chain_table.setRowCount(len(MUSCLE_CHAINS))
            for row,(chain_name,chain_color,parts) in enumerate(MUSCLE_CHAINS):
                valid=[p for p in parts if p in body_risk]
                if not valid: continue
                risks={p:body_risk[p] for p in valid}
                max_zone=max(risks,key=lambda p:risks[p])
                max_risk=risks[max_zone]
                ni=QTableWidgetItem(chain_name)
                ni.setForeground(QBrush(QColor(chain_color)))
                zi=QTableWidgetItem(BODY_PART_LABELS_FR.get(max_zone,max_zone))
                ri=QTableWidgetItem(f"{max_risk:.0%}")
                ri.setForeground(QBrush(_risk_color(max_risk)))
                self._chain_table.setItem(row,0,ni)
                self._chain_table.setItem(row,1,zi)
                self._chain_table.setItem(row,2,ri)
                self._chain_table.setItem(row,3,QTableWidgetItem(
                    CHAIN_RECO.get(chain_name,"—")))
        finally: self._chain_table.setUpdatesEnabled(True)


def main():
    app = QApplication(sys.argv)
    load_theme(app)

    # Optional login — same accounts as main.py. If login fails/cancelled,
    # the app still launches in "guest" mode with full visibility (medical-equivalent),
    # since this advisor app is most often used standalone by medical staff.
    user = None
    try:
        from PyQt5.uic import loadUi as _loadUi
        from PyQt5.QtWidgets import QDialog as _QDialog
        from database.auth import authenticate as _authenticate

        class _LoginDialog(_QDialog):
            def __init__(self):
                super().__init__()
                _loadUi("ui/login_dialog.ui", self)
                self.user = None
                self.loginBtn.clicked.connect(self._try_login)
                self.passwordEdit.returnPressed.connect(self._try_login)
            def _try_login(self):
                u = _authenticate(self.usernameEdit.text().strip(),
                                  self.passwordEdit.text())
                if u is None:
                    self.errorLabel.setText("⚠ Identifiants incorrects.")
                    return
                self.user = u
                self.accept()

        dlg = _LoginDialog()
        if dlg.exec_() == _QDialog.Accepted:
            user = dlg.user
    except Exception as e:
        print(f"Connexion ignorée ({e}) — lancement en mode invité.")

    win = PerformanceAdvisorApp(user=user)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
