import sqlite3
from pathlib import Path
import pandas as pd


class Database:
    def __init__(self, db_path="data/athlete.db"):
        Path("data").mkdir(exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.cursor = self.conn.cursor()
        self.create_tables()
        self._migrate()

    # ================================================================
    # SCHEMA
    # ================================================================

    def create_tables(self):
        self.cursor.executescript("""

        -- ── Players ─────────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS players (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            name            TEXT    NOT NULL,
            sport           TEXT    NOT NULL DEFAULT 'Football',
            position        TEXT,
            age             INTEGER,
            height          REAL,
            weight          REAL,
            dominant_foot   TEXT,
            nationality     TEXT,
            team            TEXT
        );

        -- ── Sessions ─────────────────────────────────────────────────
        -- One row per training session or match.
        -- Dimensions: GPS load | Biometry | Mental | Nutrition (daily)
        -- Slow-varying features (physical tests, blood work) live in
        -- separate tables and are joined / interpolated at query time.
        CREATE TABLE IF NOT EXISTS sessions (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id           INTEGER NOT NULL,
            date                TEXT    NOT NULL,
            session_type        TEXT    DEFAULT 'Training',
            weather             TEXT    DEFAULT 'Normal',
            surface             TEXT,
            training_minutes    INTEGER,
            rpe                 INTEGER,

            -- ── Dimension GPS / Charge physique ──────────────────────
            distance_km         REAL,           -- distance totale GPS (km)
            sprint_distance_km  REAL,           -- distance en sprint (km)
            sprints_count       INTEGER,        -- nb sprints > 25 km/h
            hid_km              REAL,           -- distance haute intensité > 19.8 km/h (km)
            acceleration_max    REAL,           -- pic d'accélération (m/s²)
            max_speed           REAL,           -- vitesse maximale (km/h)
            accelerations       INTEGER,        -- nb accélérations totales
            decelerations       INTEGER,        -- nb décélérations totales
            player_load         REAL,           -- charge mécanique composite (UA)
            heart_rate_avg      REAL,           -- fréquence cardiaque moyenne (bpm)
            heart_rate_max      REAL,           -- fréquence cardiaque maximale (bpm)
            -- Computed GPS features (stored after preprocessing)
            acwr                REAL,           -- ratio charge aiguë / chronique
            charge_7j           REAL,           -- distance moyenne 7 jours (km)
            charge_28j          REAL,           -- distance moyenne 28 jours (km)

            -- ── Dimension Biométrie & Récupération ───────────────────
            hrv_rmssd           REAL,           -- HRV RMSSD (ms)
            hrv_trend           REAL,           -- tendance HRV 7j (%)
            hrv_moy_7j          REAL,           -- HRV moyenne mobile 7j (ms)
            fc_repos            INTEGER,        -- FC repos matin (bpm)
            fc_repos_alerte     INTEGER DEFAULT 0, -- 1 si élévation > 5 bpm baseline
            sommeil_h           REAL,           -- durée sommeil (heures)
            sommeil_qualite     INTEGER,        -- qualité sommeil 1–5
            fatigue             INTEGER,        -- fatigue perçue 1–10
            spo2                REAL,           -- saturation O2 (%)
            jours_depuis_match  INTEGER,        -- jours depuis dernier match
            body_temp_celsius   REAL,           -- température corporelle (°C)

            -- ── Dimension Mental & Cognitif ───────────────────────────
            motivation          INTEGER,        -- motivation 1–10
            stress              INTEGER,        -- stress perçu 1–10
            reaction_ms         REAL,           -- temps de réaction (ms)
            charge_mentale      INTEGER,        -- charge mentale 1–10
            regularite_score    REAL,           -- régularité routines 0–1

            -- ── Dimension Nutrition & Hydratation (quotidien) ────────
            poids_variation_pct REAL,           -- variation poids vs référence (%)
            hydratation_score   INTEGER,        -- score urinaire 1–8
            ck_post             REAL,           -- créatine kinase post-effort (U/L)

            -- ── Features composées (calculées en preprocessing) ──────
            wellness_score      REAL,           -- score bien-être composite 0–10
            perf_index          REAL,           -- indice performance composite 0–100

            -- ── Modèle blessure ───────────────────────────────────────
            previous_injuries   INTEGER DEFAULT 0,
            injury_label        INTEGER DEFAULT 0,

            session_notes       TEXT,

            FOREIGN KEY(player_id) REFERENCES players(id) ON DELETE CASCADE
        );

        -- ── Physical tests (monthly) ──────────────────────────────────
        -- Interpolated linearly between tests in the preprocessing pipeline.
        CREATE TABLE IF NOT EXISTS physical_tests (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id       INTEGER NOT NULL,
            date_test       TEXT    NOT NULL,
            test_type       TEXT    DEFAULT 'Monthly',
            vo2max          REAL,           -- VO2max (ml/kg/min)
            puissance_w_kg  REAL,           -- puissance crête relative (W/kg)
            cmj_cm          REAL,           -- hauteur CMJ (cm)
            rsa_index       REAL,           -- indice RSA (%)
            force_n         REAL,           -- force isométrique (N)
            fatigue_sprint_pct REAL,        -- indice fatigue sprint (%)
            notes           TEXT,
            FOREIGN KEY(player_id) REFERENCES players(id) ON DELETE CASCADE
        );

        -- ── Nutrition & blood work (quarterly + daily body weight) ───
        CREATE TABLE IF NOT EXISTS nutrition_labs (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id           INTEGER NOT NULL,
            date                TEXT    NOT NULL,
            record_type         TEXT    DEFAULT 'quarterly', -- quarterly | daily
            -- Quarterly blood work
            pct_masse_grasse    REAL,       -- masse grasse (%)
            ferritine           REAL,       -- ferritine (ng/mL)
            hemoglobine         REAL,       -- hémoglobine (g/dL)
            -- Daily body weight variation (can also be in sessions)
            poids_kg            REAL,       -- poids du jour (kg)
            notes               TEXT,
            FOREIGN KEY(player_id) REFERENCES players(id) ON DELETE CASCADE
        );

        -- ── Sport-specific performance metrics ────────────────────────
        CREATE TABLE IF NOT EXISTS sport_metrics (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  INTEGER NOT NULL,
            sport       TEXT    NOT NULL,
            passes_completed    INTEGER,  pass_accuracy_pct   REAL,
            shots               INTEGER,  tackles             INTEGER,
            duels_won           INTEGER,  position_zone       TEXT,
            minutes_played      REAL,     points              INTEGER,
            rebounds            INTEGER,  assists             INTEGER,
            turnovers           INTEGER,  field_goal_pct      REAL,
            three_point_pct     REAL,     defensive_rating    REAL,
            sets_played         INTEGER,  games_played        INTEGER,
            aces                INTEGER,  double_faults       INTEGER,
            first_serve_pct     REAL,     winners             INTEGER,
            unforced_errors     INTEGER,
            laps                INTEGER,  avg_lap_time_s      REAL,
            stroke_type         TEXT,     stroke_rate         REAL,
            turn_efficiency_pct REAL,     pool_length_m       INTEGER,
            carries             INTEGER,  tackles_made        INTEGER,
            tackles_missed      INTEGER,  lineouts_won        INTEGER,
            metres_gained       REAL,     scrums              INTEGER,
            penalties_conceded  INTEGER,
            event_type          TEXT,     time_seconds        REAL,
            wind_ms             REAL,     altitude_m          REAL,
            discipline          TEXT,     round_count         INTEGER,
            strikes_landed      INTEGER,  strikes_attempted   INTEGER,
            strike_accuracy_pct REAL,     takedowns           INTEGER,
            submission_attempts INTEGER,
            avg_power_watts     REAL,     max_power_watts     REAL,
            avg_cadence_rpm     REAL,     elevation_gain_m    REAL,
            avg_speed_kmh       REAL,     ftp_watts           REAL,
            FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
        );

        -- ── Injury records ────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS injuries (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id       INTEGER NOT NULL,
            session_id      INTEGER,
            date            TEXT    NOT NULL,
            injury_type     TEXT,
            body_part       TEXT,
            severity        TEXT,
            mechanism       TEXT,
            days_to_return  INTEGER,
            notes           TEXT,
            FOREIGN KEY(player_id)  REFERENCES players(id)  ON DELETE CASCADE,
            FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE SET NULL
        );
        """)
        self.conn.commit()

    def _migrate(self):
        """Add missing columns to existing DBs without destroying data."""
        self.cursor.execute("PRAGMA table_info(physical_tests)")
        existing_pt = {row[1] for row in self.cursor.fetchall()}
        if "test_type" not in existing_pt:
            self.cursor.execute(
                "ALTER TABLE physical_tests ADD COLUMN test_type TEXT DEFAULT 'Monthly'"
            )

        self.cursor.execute("PRAGMA table_info(sessions)")
        existing = {row[1] for row in self.cursor.fetchall()}

        migrations = [
            # GPS computed
            ("acwr",             "REAL"),
            ("charge_7j",        "REAL"),
            ("charge_28j",       "REAL"),
            # Biometry extras
            ("hrv_trend",        "REAL"),
            ("hrv_moy_7j",       "REAL"),
            ("fc_repos_alerte",  "INTEGER DEFAULT 0"),
            # Physical tests (slow — may be NULL for most sessions)
            ("vo2max",           "REAL"),
            ("puissance_w_kg",   "REAL"),
            ("cmj_cm",           "REAL"),
            ("rsa_index",        "REAL"),
            ("force_n",          "REAL"),
            ("fatigue_sprint_pct","REAL"),
            # Biometry (heart rate)
            ("heart_rate_avg",   "REAL"),
            ("heart_rate_max",   "REAL"),
            # Nutrition extras
            ("pct_masse_grasse", "REAL"),
            ("ferritine",        "REAL"),
            ("hemoglobine",      "REAL"),
            # Computed wellness
            ("wellness_score",   "REAL"),
            ("perf_index",       "REAL"),
        ]
        for col, dtype in migrations:
            if col not in existing:
                self.cursor.execute(
                    f"ALTER TABLE sessions ADD COLUMN {col} {dtype}"
                )
        self.conn.commit()

    # ================================================================
    # PLAYERS
    # ================================================================

    def add_player(self, name, sport, position, age, height, weight,
                   dominant_foot=None, nationality=None, team=None):
        self.cursor.execute("""
            INSERT INTO players
                (name, sport, position, age, height, weight,
                 dominant_foot, nationality, team)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (name, sport, position, age, height, weight,
              dominant_foot, nationality, team))
        self.conn.commit()

    def get_players(self, sport=None):
        if sport:
            self.cursor.execute(
                "SELECT * FROM players WHERE sport=? ORDER BY name", (sport,))
        else:
            self.cursor.execute("SELECT * FROM players ORDER BY name")
        return self.cursor.fetchall()

    def get_player(self, player_id):
        return self.cursor.execute(
            "SELECT * FROM players WHERE id=?", (player_id,)
        ).fetchone()

    def update_player(self, player_id, name, sport, position, age, height,
                      weight, dominant_foot, nationality, team):
        self.cursor.execute("""
            UPDATE players
            SET name=?, sport=?, position=?, age=?, height=?, weight=?,
                dominant_foot=?, nationality=?, team=?
            WHERE id=?
        """, (name, sport, position, age, height, weight,
              dominant_foot, nationality, team, player_id))
        self.conn.commit()

    def delete_player(self, player_id):
        self.cursor.execute("DELETE FROM players WHERE id=?", (player_id,))
        self.conn.commit()

    # ================================================================
    # SESSIONS
    # ================================================================

    def add_session(self, player_id, date, session_type="Training",
                    weather="Normal", surface=None, training_minutes=None,
                    rpe=None,
                    # GPS
                    distance_km=None, sprint_distance_km=None,
                    sprints_count=None, hid_km=None,
                    acceleration_max=None, max_speed=None,
                    accelerations=None, decelerations=None, player_load=None,
                    heart_rate_avg=None, heart_rate_max=None,
                    # GPS computed
                    acwr=None, charge_7j=None, charge_28j=None,
                    # Biometry
                    hrv_rmssd=None, hrv_trend=None, hrv_moy_7j=None,
                    fc_repos=None, fc_repos_alerte=0,
                    sommeil_h=None, sommeil_qualite=None,
                    fatigue=None, spo2=None, jours_depuis_match=None,
                    body_temp_celsius=None,
                    # Mental
                    motivation=None, stress=None, reaction_ms=None,
                    charge_mentale=None, regularite_score=None,
                    # Nutrition daily
                    poids_variation_pct=None, hydratation_score=None,
                    ck_post=None,
                    # Physical tests (interpolated values)
                    vo2max=None, puissance_w_kg=None, cmj_cm=None,
                    rsa_index=None, force_n=None, fatigue_sprint_pct=None,
                    # Nutrition labs (interpolated values)
                    pct_masse_grasse=None, ferritine=None, hemoglobine=None,
                    # Composite scores
                    wellness_score=None, perf_index=None,
                    # Model
                    previous_injuries=0, injury_label=0,
                    session_notes=None):

        self.cursor.execute("""
            INSERT INTO sessions (
                player_id, date, session_type, weather, surface,
                training_minutes, rpe,
                distance_km, sprint_distance_km, sprints_count, hid_km,
                acceleration_max, max_speed, accelerations, decelerations,
                player_load, acwr, charge_7j, charge_28j,
                heart_rate_avg, heart_rate_max,
                hrv_rmssd, hrv_trend, hrv_moy_7j,
                fc_repos, fc_repos_alerte,
                sommeil_h, sommeil_qualite, fatigue, spo2,
                jours_depuis_match, body_temp_celsius,
                motivation, stress, reaction_ms,
                charge_mentale, regularite_score,
                poids_variation_pct, hydratation_score, ck_post,
                vo2max, puissance_w_kg, cmj_cm, rsa_index,
                force_n, fatigue_sprint_pct,
                pct_masse_grasse, ferritine, hemoglobine,
                wellness_score, perf_index,
                previous_injuries, injury_label, session_notes
            ) VALUES (
                ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,
                ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
            )
        """, (
            player_id, date, session_type, weather, surface,
            training_minutes, rpe,
            distance_km, sprint_distance_km, sprints_count, hid_km,
            acceleration_max, max_speed, accelerations, decelerations,
            player_load, acwr, charge_7j, charge_28j,
            heart_rate_avg, heart_rate_max,
            hrv_rmssd, hrv_trend, hrv_moy_7j,
            fc_repos, fc_repos_alerte,
            sommeil_h, sommeil_qualite, fatigue, spo2,
            jours_depuis_match, body_temp_celsius,
            motivation, stress, reaction_ms,
            charge_mentale, regularite_score,
            poids_variation_pct, hydratation_score, ck_post,
            vo2max, puissance_w_kg, cmj_cm, rsa_index,
            force_n, fatigue_sprint_pct,
            pct_masse_grasse, ferritine, hemoglobine,
            wellness_score, perf_index,
            previous_injuries, injury_label, session_notes
        ))
        self.conn.commit()
        return self.cursor.lastrowid

    def get_sessions(self, player_id):
        self.cursor.execute("""
            SELECT * FROM sessions
            WHERE player_id=? ORDER BY date DESC
        """, (player_id,))
        return self.cursor.fetchall()

    def get_sessions_dict(self, player_id):
        """Return sessions as a list of dicts (column-name access)."""
        cur = self.conn.cursor()
        cur.row_factory = sqlite3.Row
        cur.execute("""
            SELECT * FROM sessions
            WHERE player_id=? ORDER BY date DESC
        """, (player_id,))
        return [dict(r) for r in cur.fetchall()]

    def get_session_count(self):
        return self.cursor.execute(
            "SELECT COUNT(*) FROM sessions").fetchone()[0]

    def add_sport_metrics(self, session_id, sport, **kwargs):
        cols = ["session_id", "sport"] + list(kwargs.keys())
        vals = [session_id, sport] + list(kwargs.values())
        ph   = ",".join(["?"] * len(vals))
        self.cursor.execute(
            f"INSERT INTO sport_metrics ({','.join(cols)}) VALUES ({ph})", vals)
        self.conn.commit()

    def get_session_with_metrics(self, session_id):
        s = self.cursor.execute(
            "SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        m = self.cursor.execute(
            "SELECT * FROM sport_metrics WHERE session_id=?", (session_id,)).fetchone()
        return s, m

    # ================================================================
    # PHYSICAL TESTS  (monthly)
    # ================================================================

    def add_physical_test(self, player_id, date,
                          vo2max=None, puissance_w_kg=None, cmj_cm=None,
                          rsa_index=None, force_n=None,
                          fatigue_sprint_pct=None, test_type="Monthly", notes=None):
        self.cursor.execute("""
            INSERT INTO physical_tests
                (player_id, date_test, test_type, vo2max, puissance_w_kg, cmj_cm,
                 rsa_index, force_n, fatigue_sprint_pct, notes)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (player_id, date, test_type, vo2max, puissance_w_kg, cmj_cm,
              rsa_index, force_n, fatigue_sprint_pct, notes))
        self.conn.commit()

    def get_physical_tests(self, player_id):
        return self.cursor.execute(
            "SELECT * FROM physical_tests WHERE player_id=? ORDER BY date_test DESC",
            (player_id,)).fetchall()

    def get_latest_physical_test(self, player_id):
        return self.cursor.execute(
            "SELECT * FROM physical_tests WHERE player_id=? ORDER BY date_test DESC LIMIT 1",
            (player_id,)).fetchone()

    # ================================================================
    # NUTRITION LABS  (quarterly + daily)
    # ================================================================

    def add_nutrition_lab(self, player_id, date, record_type="quarterly",
                          pct_masse_grasse=None, ferritine=None,
                          hemoglobine=None, poids_kg=None, notes=None):
        self.cursor.execute("""
            INSERT INTO nutrition_labs
                (player_id, date, record_type, pct_masse_grasse,
                 ferritine, hemoglobine, poids_kg, notes)
            VALUES (?,?,?,?,?,?,?,?)
        """, (player_id, date, record_type, pct_masse_grasse,
              ferritine, hemoglobine, poids_kg, notes))
        self.conn.commit()

    def get_nutrition_labs(self, player_id, record_type=None):
        if record_type:
            return self.cursor.execute(
                "SELECT * FROM nutrition_labs WHERE player_id=? AND record_type=? ORDER BY date DESC",
                (player_id, record_type)).fetchall()
        return self.cursor.execute(
            "SELECT * FROM nutrition_labs WHERE player_id=? ORDER BY date DESC",
            (player_id,)).fetchall()

    def add_body_composition(self, player_id, date, measure_type="Quarterly",
                             pct_masse_grasse=None, ferritine=None,
                             hemoglobine=None, poids_kg=None, notes=None):
        """Body composition measurements are stored in nutrition_labs."""
        self.add_nutrition_lab(
            player_id, date, record_type=measure_type,
            pct_masse_grasse=pct_masse_grasse, ferritine=ferritine,
            hemoglobine=hemoglobine, poids_kg=poids_kg, notes=notes)

    def get_body_composition(self, player_id):
        return self.get_nutrition_labs(player_id)

    # ================================================================
    # INJURIES
    # ================================================================

    def add_injury(self, player_id, date, injury_type, body_part, severity,
                   mechanism, days_to_return=None, session_id=None, notes=None):
        self.cursor.execute("""
            INSERT INTO injuries
                (player_id, session_id, date, injury_type, body_part,
                 severity, mechanism, days_to_return, notes)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (player_id, session_id, date, injury_type, body_part,
              severity, mechanism, days_to_return, notes))
        self.conn.commit()

    def get_injuries(self, player_id):
        return self.cursor.execute(
            "SELECT * FROM injuries WHERE player_id=? ORDER BY date DESC",
            (player_id,)).fetchall()

    def get_injury_count(self, player_id):
        return self.cursor.execute(
            "SELECT COUNT(*) FROM injuries WHERE player_id=?",
            (player_id,)).fetchone()[0]

    # ================================================================
    # DATAFRAMES FOR ML
    # ================================================================

    def get_all_sessions_df(self):
        """Full feature set for model training — all 25 doc features + computed."""
        return pd.read_sql_query("""
            SELECT
                -- GPS
                distance_km, sprints_count, hid_km,
                acceleration_max, player_load,
                acwr,
                -- Biometry
                hrv_rmssd, hrv_trend, fc_repos, fc_repos_alerte,
                sommeil_h, sommeil_qualite, fatigue, spo2,
                jours_depuis_match,
                -- Mental
                motivation, stress, reaction_ms,
                charge_mentale, regularite_score,
                -- Nutrition daily
                poids_variation_pct, hydratation_score, ck_post,
                -- Physical tests (interpolated)
                vo2max, cmj_cm, rsa_index, force_n, fatigue_sprint_pct,
                -- Nutrition labs (interpolated)
                pct_masse_grasse, ferritine, hemoglobine,
                -- Composite
                wellness_score, perf_index,
                -- Target
                previous_injuries, injury_label
            FROM sessions
            WHERE distance_km IS NOT NULL
        """, self.conn)

    def get_all_sessions_df_with_player_id(self):
        return pd.read_sql_query("""
            SELECT player_id, date,
                   distance_km, sprints_count, hid_km, acceleration_max, player_load,
                   acwr, hrv_rmssd, hrv_trend, fc_repos, fc_repos_alerte,
                   sommeil_h, sommeil_qualite, fatigue, spo2, jours_depuis_match,
                   motivation, stress, reaction_ms, charge_mentale, regularite_score,
                   poids_variation_pct, hydratation_score, ck_post,
                   vo2max, cmj_cm, rsa_index, force_n, fatigue_sprint_pct,
                   wellness_score, perf_index
            FROM sessions WHERE distance_km IS NOT NULL
            ORDER BY player_id, date
        """, self.conn)

    def get_sessions_ordered(self):
        """Full sessions ordered by player+date for feature engineering."""
        return pd.read_sql_query("""
            SELECT * FROM sessions ORDER BY player_id, date
        """, self.conn)

    # ================================================================
    # PREPROCESSING — compute derived features
    # ================================================================

    def run_preprocessing(self):
        """
        Compute and store all derived features defined in the doc:
          - charge_7j, charge_28j, acwr
          - hrv_moy_7j, hrv_trend
          - fc_repos_alerte  (elevation > 5 bpm vs personal baseline)
          - wellness_score   (composite well-being 0–10)
          - perf_index       (composite performance 0–100)

        Should be called after bulk inserts (e.g. after generate_db).
        """
        df = pd.read_sql_query(
            "SELECT * FROM sessions ORDER BY player_id, date", self.conn)

        if df.empty:
            return

        rows_updated = 0

        # Pre-load nutrition labs for interpolation
        labs_df = pd.read_sql_query(
            "SELECT player_id, date, pct_masse_grasse, ferritine, hemoglobine "
            "FROM nutrition_labs ORDER BY player_id, date",
            self.conn
        )
        labs_df["date"] = pd.to_datetime(labs_df["date"])

        for pid, grp in df.groupby("player_id"):
            grp = grp.sort_values("date").reset_index(drop=True)
            grp["date"] = pd.to_datetime(grp["date"])

            # ── Interpolate quarterly lab values into daily sessions ──────────
            player_labs = labs_df[labs_df["player_id"] == pid].copy()
            if not player_labs.empty:
                # Merge on date, forward-fill + backward-fill between quarterly tests
                merged = pd.merge_asof(
                    grp[["id", "date"]].sort_values("date"),
                    player_labs[["date", "pct_masse_grasse", "ferritine", "hemoglobine"]].sort_values("date"),
                    on="date", direction="nearest"
                )
                # Add gentle noise so rows aren't all identical
                import numpy as _np
                merged["pct_masse_grasse"] = merged["pct_masse_grasse"].apply(
                    lambda v: round(v + _np.random.normal(0, abs(v)*0.015), 1) if pd.notna(v) else v)
                merged["ferritine"] = merged["ferritine"].apply(
                    lambda v: round(v + _np.random.normal(0, abs(v)*0.02), 1) if pd.notna(v) else v)
                merged["hemoglobine"] = merged["hemoglobine"].apply(
                    lambda v: round(v + _np.random.normal(0, abs(v)*0.01), 2) if pd.notna(v) else v)

                for _, lab_row in merged.iterrows():
                    self.cursor.execute("""
                        UPDATE sessions SET
                            pct_masse_grasse = ?,
                            ferritine        = ?,
                            hemoglobine      = ?
                        WHERE id = ?
                    """, (lab_row["pct_masse_grasse"], lab_row["ferritine"],
                          lab_row["hemoglobine"], int(lab_row["id"])))

            # Rolling GPS
            dist = grp["distance_km"].fillna(0)
            c7  = dist.rolling(7,  min_periods=1).mean()
            c28 = dist.rolling(28, min_periods=1).mean()
            acwr_series = c7 / c28.replace(0, 1)

            # HRV rolling
            hrv     = grp["hrv_rmssd"].ffill()
            hrv_7j  = hrv.rolling(7, min_periods=1).mean()
            hrv_trend_series = ((hrv - hrv_7j) / hrv_7j.replace(0, 1) * 100).round(2)

            # FC repos baseline (rolling 28-day median per player)
            fc = grp["fc_repos"].ffill()
            fc_base = fc.rolling(28, min_periods=3).median()
            fc_alerte = ((fc - fc_base) > 5).astype(int)

            # Wellness score: mean of normalised (fatigue_inv, sommeil, stress_inv, motivation)
            def norm01(s, lo, hi):
                return ((s - lo) / (hi - lo)).clip(0, 1)

            fat_inv  = 1 - norm01(grp["fatigue"].fillna(5),   1, 10)
            som_norm =     norm01(grp["sommeil_h"].fillna(7),  4, 10)
            str_inv  = 1 - norm01(grp["stress"].fillna(5),     1, 10)
            mot_norm =     norm01(grp["motivation"].fillna(5), 1, 10)
            wellness = ((fat_inv + som_norm + str_inv + mot_norm) / 4 * 10).round(2)

            # perf_index: weighted composite of acwr, hrv_trend, vo2max, cmj
            # Normalised to 0–100. Weights from doc (equal if not specified by position)
            acwr_norm   = (1 - (acwr_series - 1.05).abs() / 1.05).clip(0, 1) * 100
            hrv_t_norm  = norm01(hrv_trend_series.clip(-20, 20), -20, 20) * 100
            vo2_norm    = norm01(grp["vo2max"].fillna(45),   30, 70) * 100
            cmj_norm    = norm01(grp["cmj_cm"].fillna(35),   20, 60) * 100
            perf        = (0.35*acwr_norm + 0.25*hrv_t_norm +
                           0.25*vo2_norm  + 0.15*cmj_norm).round(2)

            for i, row in grp.iterrows():
                self.cursor.execute("""
                    UPDATE sessions SET
                        charge_7j       = ?,
                        charge_28j      = ?,
                        acwr            = ?,
                        hrv_moy_7j      = ?,
                        hrv_trend       = ?,
                        fc_repos_alerte = ?,
                        wellness_score  = ?,
                        perf_index      = ?
                    WHERE id = ?
                """, (
                    round(float(c7.iloc[i]),  3),
                    round(float(c28.iloc[i]), 3),
                    round(float(acwr_series.iloc[i]), 3),
                    round(float(hrv_7j.iloc[i]),  2),
                    round(float(hrv_trend_series.iloc[i]), 2),
                    int(fc_alerte.iloc[i]),
                    round(float(wellness.iloc[i]), 2),
                    round(float(perf.iloc[i]),    2),
                    int(row["id"])
                ))
                rows_updated += 1

        self.conn.commit()
        print(f"  Preprocessing: {rows_updated} sessions updated.")

    # ================================================================
    # HELPERS
    # ================================================================

    def get_sports(self):
        return [r[0] for r in self.cursor.execute(
            "SELECT DISTINCT sport FROM players ORDER BY sport").fetchall()]

    def search_session_notes(self, query: str, player_id=None):
        """
        Recherche plein-texte (LIKE) dans session_notes.
        Si player_id est fourni, restreint la recherche à ce joueur.
        Retourne une liste de tuples (session_id, player_id, player_name, date, session_notes).
        """
        like = f"%{query}%"
        if player_id is not None:
            rows = self.cursor.execute("""
                SELECT s.id, s.player_id, p.name, s.date, s.session_notes
                FROM sessions s JOIN players p ON s.player_id = p.id
                WHERE s.player_id = ? AND s.session_notes LIKE ?
                ORDER BY s.date DESC
            """, (player_id, like)).fetchall()
        else:
            rows = self.cursor.execute("""
                SELECT s.id, s.player_id, p.name, s.date, s.session_notes
                FROM sessions s JOIN players p ON s.player_id = p.id
                WHERE s.session_notes LIKE ?
                ORDER BY s.date DESC
            """, (like,)).fetchall()
        return rows

    def close(self):
        self.conn.close()
