"""
ml/performance_advisor.py
-------------------------------------------------------------------------------
Modèle d'aide à la décision : "Sur quoi se concentrer pour augmenter la
performance du joueur, qu'améliorer SANS faire monter le risque de blessure,
QUELLES sont les causes du risque actuel et COMMENT les corriger ?"

Le modèle s'appuie sur :
  - les coefficients standardisés de la régression logistique de blessure ;
  - l'historique des séances pour estimer la corrélation attribut ↔ perf_index
    et le z-score du joueur par rapport à sa propre baseline 30 jours.

Trois sorties principales :

  • Recommendation  → catégorisation FOCUS / CAUTION / MAINTAIN par attribut.
  • InjuryCause     → liste ordonnée des attributs qui poussent le risque
                      actuel à la hausse + zones anatomiques touchées.
  • PreventionAction→ contre-mesure concrète par attribut (sommeil, charge,
                      hydratation, etc.) dérivée d'un référentiel expert.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Mapping attribut → zones anatomiques
# -----------------------------------------------------------------------------
BODY_PARTS: List[str] = [
    "head", "heart", "chest", "abdomen", "lower_back",
    "shoulders",
    "left_arm", "right_arm",
    "left_quad", "right_quad",
    "left_hamstring", "right_hamstring",
    "left_knee", "right_knee",
    "left_calf", "right_calf",
    "left_ankle", "right_ankle",
]

_LEGS_QUAD = ["left_quad", "right_quad"]
_LEGS_HAM = ["left_hamstring", "right_hamstring"]
_LEGS_CALF = ["left_calf", "right_calf"]
_LEGS_KNEE = ["left_knee", "right_knee"]
_LEGS_ANKLE = ["left_ankle", "right_ankle"]
_ARMS = ["left_arm", "right_arm"]

FEATURE_BODY_MAP: Dict[str, List[str]] = {
    "distance_km":        _LEGS_QUAD + _LEGS_CALF + _LEGS_HAM,
    "sprints_count":      _LEGS_HAM + _LEGS_CALF,
    "hid_km":             _LEGS_HAM + _LEGS_QUAD,
    "acceleration_max":   _LEGS_QUAD + _LEGS_KNEE + _LEGS_ANKLE,
    "player_load":        _LEGS_QUAD + _LEGS_HAM + ["lower_back"],
    "acwr":               ["lower_back"] + _LEGS_HAM + _LEGS_KNEE,
    "hrv_rmssd":          ["heart"],
    "hrv_trend":          ["heart"],
    "fc_repos":           ["heart"],
    "fc_repos_alerte":    ["heart"],
    "spo2":               ["chest"],
    "sommeil_h":          ["head"],
    "sommeil_qualite":    ["head"],
    "motivation":         ["head"],
    "stress":             ["head", "heart"],
    "reaction_ms":        ["head"],
    "charge_mentale":     ["head"],
    "regularite_score":   ["head"],
    "wellness_score":     ["head"],
    "fatigue":            ["lower_back"] + _LEGS_QUAD + _LEGS_HAM,
    "jours_depuis_match": ["lower_back"],
    "fatigue_trend_7d":   ["lower_back"] + _LEGS_HAM,
    "poids_variation_pct": ["abdomen"],
    "hydratation_score":   ["abdomen"] + _LEGS_CALF,
    "ck_post":             _LEGS_QUAD + _LEGS_HAM,
    "vo2max":              ["chest", "heart"],
    "cmj_cm":              _LEGS_QUAD + _LEGS_CALF,
    "rsa_index":           _LEGS_HAM + _LEGS_CALF,
    "force_n":             _LEGS_QUAD + ["shoulders"],
    "fatigue_sprint_pct":  _LEGS_HAM,
    "perf_index":          ["chest"],
    # --- features dérivées (v2) ---
    "acwr_squared":        ["lower_back"] + _LEGS_HAM + _LEGS_KNEE,
    "load_per_sleep":      _LEGS_QUAD + _LEGS_HAM + ["head"],
    "hrv_delta_14d":       ["heart"],
    "fatigue_accel":       ["lower_back"] + _LEGS_HAM,
    "recovery_index":      ["head", "abdomen"],
}

FEATURE_LABELS_FR: Dict[str, str] = {
    "distance_km": "Distance totale (km)",
    "sprints_count": "Nombre de sprints",
    "hid_km": "Haute intensité (km)",
    "acceleration_max": "Accélération max",
    "player_load": "Charge externe (Player Load)",
    "acwr": "Ratio aigu/chronique (ACWR)",
    "hrv_rmssd": "Variabilité cardiaque (HRV)",
    "hrv_trend": "Tendance HRV",
    "fc_repos": "Fréquence cardiaque au repos",
    "fc_repos_alerte": "Alerte FC repos",
    "sommeil_h": "Heures de sommeil",
    "sommeil_qualite": "Qualité de sommeil",
    "fatigue": "Fatigue ressentie",
    "spo2": "Saturation O₂",
    "jours_depuis_match": "Jours depuis le match",
    "motivation": "Motivation",
    "stress": "Stress",
    "reaction_ms": "Temps de réaction (ms)",
    "charge_mentale": "Charge mentale",
    "regularite_score": "Régularité",
    "poids_variation_pct": "Variation de poids (%)",
    "hydratation_score": "Hydratation",
    "ck_post": "CK post-séance",
    "vo2max": "VO₂max",
    "cmj_cm": "Détente verticale (CMJ)",
    "rsa_index": "Index RSA",
    "force_n": "Force max (N)",
    "fatigue_sprint_pct": "Chute sprint (%)",
    "wellness_score": "Score bien-être",
    "perf_index": "Index de performance",
    "fatigue_trend_7d": "Tendance fatigue 7j",
    "acwr_squared": "Charge non-linéaire (ACWR²)",
    "load_per_sleep": "Charge / heure de sommeil",
    "hrv_delta_14d": "Écart HRV vs baseline 14j",
    "fatigue_accel": "Accélération de la fatigue",
    "recovery_index": "Index de récupération",
}

BODY_PART_LABELS_FR: Dict[str, str] = {
    "head": "Tête / SNC",
    "heart": "Cœur",
    "chest": "Thorax / poumons",
    "abdomen": "Abdomen",
    "lower_back": "Bas du dos",
    "shoulders": "Épaules",
    "left_arm": "Bras gauche", "right_arm": "Bras droit",
    "left_quad": "Quadriceps gauche", "right_quad": "Quadriceps droit",
    "left_hamstring": "Ischio-jambier gauche", "right_hamstring": "Ischio-jambier droit",
    "left_knee": "Genou gauche", "right_knee": "Genou droit",
    "left_calf": "Mollet gauche", "right_calf": "Mollet droit",
    "left_ankle": "Cheville gauche", "right_ankle": "Cheville droite",
}


# -----------------------------------------------------------------------------
# Référentiel expert : causes physiologiques + actions de prévention
# -----------------------------------------------------------------------------
# Pour chaque attribut on décrit :
#   high  → ce qu'il se passe quand la valeur est trop HAUTE
#   low   → ce qu'il se passe quand la valeur est trop BASSE
# Chaque entrée = (cause physiologique, action préventive concrète).
CAUSES_DB: Dict[str, Dict[str, tuple]] = {
    "distance_km": {
        "high": ("Volume kilométrique excessif → fatigue musculaire cumulée "
                 "des chaînes postérieures.",
                 "Réduire le volume de 15–20 % sur 3 jours, intercaler une "
                 "séance régénérative (vélo, natation)."),
    },
    "sprints_count": {
        "high": ("Trop de sprints maximaux → micro-lésions des ischio-jambiers "
                 "et mollets (risque grade I).",
                 "Plafonner à 12–15 sprints/séance, renforcement excentrique "
                 "Nordic Hamstring 2×/sem."),
    },
    "hid_km": {
        "high": ("Distance haute intensité au-dessus du seuil de tolérance "
                 "individuel.",
                 "Programmer 48 h de récupération active après la séance, "
                 "monitoring CK le lendemain."),
    },
    "acceleration_max": {
        "high": ("Pics d'accélération répétés → contrainte sur tendon "
                 "rotulien et chaîne antérieure du genou.",
                 "Travail isométrique quadriceps (Spanish squat 3×45 s), "
                 "limiter les départs explosifs hors séance dédiée."),
    },
    "player_load": {
        "high": ("Charge externe globale excessive (accumulation accélérations "
                 "+ chocs).",
                 "Décharge programmée : -30 % de player load sur la prochaine "
                 "session, dialogue staff médical."),
    },
    "acwr": {
        "high": ("ACWR > 1,3 : la charge récente dépasse l'adaptation "
                 "chronique → risque blessure ×2-4.",
                 "Ramener l'ACWR vers 0,8–1,3 : alléger la prochaine séance "
                 "OU étaler la charge sur la semaine."),
        "low":  ("ACWR < 0,8 : sous-charge prolongée → désentraînement et "
                 "fragilité au retour.",
                 "Réintroduire progressivement la charge (+10 %/semaine), "
                 "éviter le pic brutal au retour."),
    },
    "hrv_rmssd": {
        "low":  ("HRV basse → système nerveux autonome sous stress, "
                 "récupération incomplète.",
                 "Séance allégée, exercices de cohérence cardiaque 2×10 min, "
                 "vérifier sommeil et hydratation."),
    },
    "hrv_trend": {
        "low":  ("Tendance HRV baissière sur 7 jours → sur-entraînement "
                 "naissant.",
                 "Jour OFF complet ou cross-training léger, réévaluer la "
                 "charge hebdomadaire."),
    },
    "fc_repos": {
        "high": ("FC repos élevée → fatigue, déshydratation ou état infectieux "
                 "latent.",
                 "Bilan médical court, hydratation +500 ml/j, séance technique "
                 "à intensité réduite."),
    },
    "fc_repos_alerte": {
        "high": ("Alerte FC repos déclenchée (>10 bpm au-dessus du baseline).",
                 "Repos médical 24 h, recontrôler le lendemain matin avant "
                 "toute reprise."),
    },
    "spo2": {
        "low":  ("SpO₂ basse → oxygénation insuffisante, signe de fatigue "
                 "respiratoire ou altitude.",
                 "Séance aérobie modérée uniquement, contrôle médical si "
                 "<94 % à répétition."),
    },
    "sommeil_h": {
        "low":  ("Sommeil < 7 h → récupération neuromusculaire incomplète, "
                 "risque blessure +1,7×.",
                 "Cible 8–9 h, coucher avant 23 h, pas d'écran 45 min avant, "
                 "sieste 20 min si possible."),
    },
    "sommeil_qualite": {
        "low":  ("Sommeil fragmenté → cortisol matinal élevé, baisse de "
                 "vigilance.",
                 "Hygiène de sommeil : chambre 18–19 °C, obscurité totale, "
                 "magnésium en soirée si validé staff médical."),
    },
    "fatigue": {
        "high": ("Fatigue ressentie élevée → décalage entre charge perçue "
                 "et capacité de récupération.",
                 "Recovery actif (10 min vélo + étirements + 12 min sauna), "
                 "alléger la séance du lendemain."),
    },
    "fatigue_trend_7d": {
        "high": ("Fatigue moyenne 7 j en hausse → accumulation chronique.",
                 "Semaine de décharge planifiée : -25 % de volume sur 5 jours."),
    },
    "jours_depuis_match": {
        "low":  ("Récupération post-match incomplète (< 48 h).",
                 "Protocole J+1 : récup active 30 min, J+2 : technique basse "
                 "intensité avant reprise normale."),
    },
    "motivation": {
        "low":  ("Motivation basse → engagement musculaire réduit, "
                 "compensations à risque.",
                 "Échange individuel avec le préparateur mental, varier les "
                 "contenus d'entraînement."),
    },
    "stress": {
        "high": ("Stress psychologique élevé → tension musculaire chronique, "
                 "raideur cervico-lombaire.",
                 "Cohérence cardiaque 3×5 min/jour, sophrologie hebdomadaire, "
                 "limiter les sollicitations extra-sportives."),
    },
    "reaction_ms": {
        "high": ("Temps de réaction allongé → vigilance abaissée, risque de "
                 "mauvais placement / contact.",
                 "Réveil neuro pré-séance (10 min jeux visuels), vérifier "
                 "sommeil et hydratation."),
    },
    "charge_mentale": {
        "high": ("Charge mentale élevée → fatigue centrale, baisse de qualité "
                 "technique.",
                 "Alléger les meetings vidéo, créneau récup mentale 30 min/j."),
    },
    "regularite_score": {
        "low":  ("Régularité d'entraînement insuffisante → adaptation "
                 "physiologique incomplète.",
                 "Respecter 4 séances/semaine minimum, planning verrouillé "
                 "avec le staff."),
    },
    "wellness_score": {
        "low":  ("Score bien-être global bas (sommeil + humeur + courbatures).",
                 "Séance allégée, focus mobilité + récupération sur 48 h."),
    },
    "poids_variation_pct": {
        "high": ("Variation de poids rapide (>1,5 %) → déshydratation ou "
                 "déficit énergétique aigu.",
                 "Réhydratation 1,5 L sur 4 h (eau + électrolytes), repas "
                 "complet glucides/protéines dans l'heure."),
    },
    "hydratation_score": {
        "low":  ("Hydratation insuffisante → baisse de performance et "
                 "augmentation des crampes.",
                 "+500 ml d'eau toutes les 2 h, contrôle urine couleur paille "
                 "avant séance."),
    },
    "ck_post": {
        "high": ("CK post-séance élevée → dégât musculaire important, "
                 "récupération incomplète.",
                 "Récup passive 48 h, massage drainant, contrôle CK à 24 h "
                 "et à 48 h."),
    },
    "vo2max": {
        "low":  ("VO₂max sous le seuil de tolérance aérobie du poste.",
                 "Cycle 4 semaines d'intermittent court (15-15) 2×/sem en "
                 "période de base."),
    },
    "cmj_cm": {
        "low":  ("Détente verticale en baisse → fatigue neuromusculaire des "
                 "extenseurs.",
                 "48 h de récupération avant tout travail pliométrique, "
                 "réévaluation CMJ à J+2."),
    },
    "rsa_index": {
        "low":  ("Capacité à répéter les sprints dégradée → fatigue "
                 "métabolique.",
                 "Bloc RSA structuré (6×30 m / r=20 s) 1×/sem en période "
                 "spécifique uniquement."),
    },
    "force_n": {
        "low":  ("Force maximale en baisse → moins de protection articulaire "
                 "et tendineuse.",
                 "Renforcement lourd 3-5 RM (squat, hip-thrust) 2×/sem hors "
                 "période de compétition."),
    },
    "fatigue_sprint_pct": {
        "high": ("Chute de vitesse intra-séance importante → fatigue "
                 "spécifique élevée.",
                 "Réduire le nombre de répétitions, allonger les récupérations "
                 "(>= 4× temps d'effort)."),
    },
    # ---------- Features dérivées (v2) ----------
    "acwr_squared": {
        "high": ("Charge non-linéaire trop élevée : ACWR² amplifie le risque "
                 "au-delà de 1.3.",
                 "Lissage immédiat de la charge sur 7 jours, suppression des "
                 "sessions intenses non programmées."),
    },
    "load_per_sleep": {
        "high": ("Charge externe disproportionnée par rapport au sommeil → "
                 "déficit récupération critique.",
                 "Soit alléger la charge (-20 %), soit garantir 8 h de sommeil "
                 "avant la prochaine séance intense."),
    },
    "hrv_delta_14d": {
        "low":  ("HRV en chute vs baseline 14 j → système autonome saturé.",
                 "Jour de récupération active, cohérence cardiaque, recontrôler "
                 "HRV à J+1 avant reprise."),
    },
    "fatigue_accel": {
        "high": ("Fatigue qui s'accélère brutalement vs tendance 7 j → "
                 "signal précoce de surcharge.",
                 "Réduire l'intensité de la prochaine séance, monitoring "
                 "wellness quotidien sur 5 jours."),
    },
    "recovery_index": {
        "low":  ("Index de récupération composite faible (sommeil + hydratation "
                 "vs CK).",
                 "Protocole récupération renforcée : 9 h sommeil cible, "
                 "2,5 L eau + électrolytes, massage drainant."),
    },
}


# -----------------------------------------------------------------------------
# Structures de sortie
# -----------------------------------------------------------------------------
@dataclass
class Recommendation:
    feature: str
    label: str
    category: str          # "FOCUS" | "CAUTION" | "MAINTAIN"
    perf_impact: float
    injury_weight: float
    direction: int         # +1 (augmenter) / -1 (diminuer) pour gagner en perf
    current_z: float
    message: str


@dataclass
class InjuryCause:
    feature: str
    label: str
    current_z: float        # écart à la baseline (en σ)
    injury_weight: float    # coefficient LR standardisé
    risk_contrib: float     # contribution normalisée [0, 1]
    side: str               # "high" | "low"
    zones: List[str]        # zones anatomiques touchées
    explanation: str        # cause physiologique
    action: str             # action de prévention concrète


@dataclass
class AdvisorReport:
    player_id: int
    perf_index: float
    injury_prob: float
    recommendations: List[Recommendation] = field(default_factory=list)
    causes: List[InjuryCause] = field(default_factory=list)
    body_risk: Dict[str, float] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Cœur du modèle
# -----------------------------------------------------------------------------
class PerformanceAdvisor:

    def __init__(
        self,
        db_path: str = "data/athlete.db",
        injury_model_path: str = "ml/injury_model.pkl",
        feature_names_path: str = "ml/feature_names.pkl",
    ):
        self.db_path = db_path
        self.injury_model = joblib.load(injury_model_path)
        self.feature_names: List[str] = joblib.load(feature_names_path)
        self.history = self._load_history()
        self.injury_weights = self._extract_injury_weights()
        self.perf_impact = self._compute_perf_impact()

    # ---- chargement -----------------------------------------------------
    def _load_history(self) -> pd.DataFrame:
        with sqlite3.connect(self.db_path) as cx:
            cursor = cx.cursor()
            cursor.execute("PRAGMA table_info(sessions)")
            db_columns = {row[1] for row in cursor.fetchall()}

        # On charge toutes les colonnes connues présentes en base, on calcule
        # les features dérivées localement (compatible v2 du modèle).
        from ml.injury_model import RAW_FEATURE_COLUMNS, add_engineered_features

        raw_present = [c for c in RAW_FEATURE_COLUMNS if c in db_columns]
        cols = ", ".join(raw_present + ["player_id", "date", "injury_label"])
        with sqlite3.connect(self.db_path) as cx:
            df = pd.read_sql_query(
                f"SELECT {cols} FROM sessions WHERE distance_km IS NOT NULL "
                "ORDER BY player_id, date", cx)
        df["date"] = pd.to_datetime(df["date"])

        # Sécurité : si une feature attendue est absente en SQL, on l'initialise
        # à la médiane pour ne pas casser le pipeline.
        for c in RAW_FEATURE_COLUMNS:
            if c not in df.columns:
                df[c] = 0.0

        df = add_engineered_features(df)

        # S'assurer que toutes les features du modèle existent.
        for c in self.feature_names:
            if c not in df.columns:
                df[c] = 0.0
        return df

    def _extract_injury_weights(self) -> Dict[str, float]:
        # Compat v1 (Pipeline LR) et v2 (InjuryEnsembleModel)
        clf = self.injury_model.named_steps["clf"]
        return dict(zip(self.feature_names, clf.coef_[0]))

    def _compute_perf_impact(self) -> Dict[str, float]:
        impacts: Dict[str, float] = {}
        target = self.history["perf_index"].astype(float)
        for f in self.feature_names:
            if f == "perf_index":
                impacts[f] = 1.0
                continue
            s = self.history[f].astype(float)
            if s.nunique() < 3:
                impacts[f] = 0.0
                continue
            impacts[f] = float(s.corr(target))
        return impacts

    # ---- API principale -------------------------------------------------
    def advise(self, player_id: int, focus_top: int = 5,
               caution_top: int = 5, causes_top: int = 6) -> AdvisorReport:
        ph = self.history[self.history["player_id"] == player_id]
        if ph.empty:
            raise ValueError(f"Aucune séance trouvée pour le joueur {player_id}")

        last = ph.iloc[-1]
        baseline = ph.tail(30) if len(ph) >= 5 else ph
        mean = baseline[self.feature_names].mean()
        std = baseline[self.feature_names].std().replace(0, 1.0)

        X_last = last[self.feature_names].to_frame().T.astype(float)
        injury_prob = float(self.injury_model.predict_proba(X_last)[0, 1])

        recos: List[Recommendation] = []
        causes_raw: List[tuple] = []   # (raw_contrib, InjuryCause)
        body_risk: Dict[str, float] = {p: 0.0 for p in BODY_PARTS}

        for f in self.feature_names:
            perf = float(self.perf_impact.get(f, 0.0))
            weight = float(self.injury_weights.get(f, 0.0))
            direction = 1 if perf >= 0 else -1
            z = float((last[f] - mean[f]) / std[f]) if std[f] else 0.0

            # ---- Contribution au risque actuel -----------------------
            raw_contrib = weight * z          # signé
            if raw_contrib > 0:               # tire le risque vers le haut
                for zone in FEATURE_BODY_MAP.get(f, []):
                    body_risk[zone] += raw_contrib

                side = "high" if z > 0 else "low"
                entry = CAUSES_DB.get(f, {}).get(side)
                if entry is None:
                    # fallback générique
                    if side == "high":
                        entry = (f"Valeur élevée de « {FEATURE_LABELS_FR.get(f, f)} » "
                                 f"corrélée à un risque accru.",
                                 "Réduire ou compenser cet attribut sur les "
                                 "prochaines séances et surveiller l'évolution.")
                    else:
                        entry = (f"Valeur basse de « {FEATURE_LABELS_FR.get(f, f)} » "
                                 f"associée à un risque accru.",
                                 "Travailler spécifiquement cet attribut pour le "
                                 "ramener dans la norme du joueur.")
                cause = InjuryCause(
                    feature=f,
                    label=FEATURE_LABELS_FR.get(f, f),
                    current_z=z,
                    injury_weight=weight,
                    risk_contrib=raw_contrib,     # normalisé plus bas
                    side=side,
                    zones=list(FEATURE_BODY_MAP.get(f, [])),
                    explanation=entry[0],
                    action=entry[1],
                )
                causes_raw.append((raw_contrib, cause))

            # ---- Recommandation FOCUS / CAUTION / MAINTAIN -----------
            risky_to_push = (direction * weight) > 0.05
            high_perf = abs(perf) >= 0.10

            if high_perf and risky_to_push:
                category = "CAUTION"
                arrow = "↑" if direction > 0 else "↓"
                msg = (f"Pousser {arrow} cet attribut augmenterait la "
                       f"performance MAIS aussi le risque de blessure "
                       f"(coef LR = {weight:+.2f}).")
            elif high_perf:
                if direction * z < 0.25:
                    category = "FOCUS"
                    arrow = "↑" if direction > 0 else "↓"
                    msg = (f"Levier sûr : {arrow} cet attribut améliore la "
                           f"perf (r={perf:+.2f}) sans accroître le risque.")
                else:
                    category = "MAINTAIN"
                    msg = "Niveau déjà au-dessus de la moyenne — maintenir."
            else:
                category = "MAINTAIN"
                msg = "Impact performance faible — pas une priorité."

            recos.append(Recommendation(
                feature=f, label=FEATURE_LABELS_FR.get(f, f),
                category=category, perf_impact=perf, injury_weight=weight,
                direction=direction, current_z=z, message=msg,
            ))

        # ---- Normalisation des risques anatomiques ---------------------
        max_r = max(body_risk.values()) or 1.0
        body_risk = {k: min(v / max_r, 1.0) for k, v in body_risk.items()}

        # ---- Top causes triées + contribution normalisée ---------------
        causes_raw.sort(key=lambda x: -x[0])
        max_c = causes_raw[0][0] if causes_raw else 1.0
        causes: List[InjuryCause] = []
        for raw, c in causes_raw[:causes_top]:
            c.risk_contrib = float(min(raw / max_c, 1.0)) if max_c else 0.0
            causes.append(c)

        focus = sorted([r for r in recos if r.category == "FOCUS"],
                       key=lambda r: -abs(r.perf_impact))[:focus_top]
        caution = sorted([r for r in recos if r.category == "CAUTION"],
                         key=lambda r: -abs(r.injury_weight * r.perf_impact))[:caution_top]
        maintain = [r for r in recos if r.category == "MAINTAIN"]

        return AdvisorReport(
            player_id=player_id,
            perf_index=float(last["perf_index"]),
            injury_prob=injury_prob,
            recommendations=focus + caution + maintain,
            causes=causes,
            body_risk=body_risk,
        )


# -----------------------------------------------------------------------------
# CLI debug : `python ml/performance_advisor.py 1`
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    pid = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    adv = PerformanceAdvisor()
    rep = adv.advise(pid)
    print(f"Joueur {rep.player_id} | perf_index={rep.perf_index:.1f} | "
          f"risque blessure={rep.injury_prob:.1%}\n")
    print("--- CAUSES du risque (top) ---")
    for c in rep.causes:
        zones = ", ".join(BODY_PART_LABELS_FR.get(z, z) for z in c.zones[:3])
        print(f"  • {c.label} (z={c.current_z:+.2f}, contrib={c.risk_contrib:.0%})")
        print(f"      Pourquoi   : {c.explanation}")
        print(f"      Que faire  : {c.action}")
        print(f"      Zones      : {zones}\n")
    print("--- FOCUS (à pousser) ---")
    for r in [x for x in rep.recommendations if x.category == "FOCUS"]:
        print(f"  {r.label:<30s} perf={r.perf_impact:+.2f} risk_w={r.injury_weight:+.2f}")
    print("\n--- CAUTION (gain perf MAIS risque) ---")
    for r in [x for x in rep.recommendations if x.category == "CAUTION"]:
        print(f"  {r.label:<30s} perf={r.perf_impact:+.2f} risk_w={r.injury_weight:+.2f}")


# ── Alias for performance_app import ─────────────────────────────────────────
from ml.injury_model import RAW_FEATURE_COLUMNS as RAW_FEATURE_COLUMNS_PA


# ── Monkey-patch: advise_from_values ─────────────────────────────────────────
def _advise_from_values(self, values: dict,
                        focus_top: int = 5,
                        caution_top: int = 5,
                        causes_top: int = 6) -> "AdvisorReport":
    """
    Run the advisor with manually-supplied feature values (simulation mode).
    Missing features are filled from the global median of the history dataset.
    """
    import pandas as pd
    from ml.injury_model import add_engineered_features

    # Build a one-row DataFrame with all raw features
    from ml.injury_model import RAW_FEATURE_COLUMNS
    row: dict = {}
    global_med = self.history[RAW_FEATURE_COLUMNS].median()
    for f in RAW_FEATURE_COLUMNS:
        row[f] = float(values.get(f, global_med[f]))

    # Add dummy columns needed for engineered features
    row["player_id"] = -1
    row["date"]      = pd.Timestamp("2000-01-01")
    single = pd.DataFrame([row])
    single["date"] = pd.to_datetime(single["date"])
    single = add_engineered_features(single)

    # Ensure all model features present
    for c in self.feature_names:
        if c not in single.columns:
            single[c] = 0.0

    last = single.iloc[0]

    # Use global stats as baseline (no player-specific history)
    mean = self.history[self.feature_names].mean()
    std  = self.history[self.feature_names].std().replace(0, 1.0)

    X_last = last[self.feature_names].to_frame().T.astype(float)
    injury_prob = float(self.injury_model.predict_proba(X_last)[0, 1])

    from ml.performance_advisor import (
        BODY_PARTS, FEATURE_BODY_MAP, CAUSES_DB, FEATURE_LABELS_FR,
        Recommendation, InjuryCause, AdvisorReport,
    )

    recos: list = []
    causes_raw: list = []
    body_risk: dict = {p: 0.0 for p in BODY_PARTS}

    for f in self.feature_names:
        perf      = float(self.perf_impact.get(f, 0.0))
        weight    = float(self.injury_weights.get(f, 0.0))
        direction = 1 if perf >= 0 else -1
        z = float((last[f] - mean[f]) / std[f]) if std[f] else 0.0

        raw_contrib = weight * z
        if raw_contrib > 0:
            for zone in FEATURE_BODY_MAP.get(f, []):
                body_risk[zone] += raw_contrib

            side  = "high" if z > 0 else "low"
            entry = CAUSES_DB.get(f, {}).get(side)
            if entry is None:
                if side == "high":
                    entry = (f"Valeur élevée de « {FEATURE_LABELS_FR.get(f, f)} ».",
                             "Réduire ou compenser cet attribut sur les prochaines séances.")
                else:
                    entry = (f"Valeur basse de « {FEATURE_LABELS_FR.get(f, f)} ».",
                             "Travailler cet attribut pour le ramener dans la norme.")
            cause = InjuryCause(
                feature=f, label=FEATURE_LABELS_FR.get(f, f),
                current_z=z, injury_weight=weight,
                risk_contrib=raw_contrib, side=side,
                zones=list(FEATURE_BODY_MAP.get(f, [])),
                explanation=entry[0], action=entry[1],
            )
            causes_raw.append((raw_contrib, cause))

        risky_to_push = (direction * weight) > 0.05
        high_perf     = abs(perf) >= 0.10
        if high_perf and risky_to_push:
            category = "CAUTION"
            arrow = "↑" if direction > 0 else "↓"
            msg = (f"Pousser {arrow} cet attribut augmenterait la performance "
                   f"MAIS aussi le risque (coef={weight:+.2f}).")
        elif high_perf:
            if direction * z < 0.25:
                category = "FOCUS"
                arrow = "↑" if direction > 0 else "↓"
                msg = (f"Levier sûr : {arrow} cet attribut améliore la perf "
                       f"(r={perf:+.2f}) sans accroître le risque.")
            else:
                category = "MAINTAIN"
                msg = "Niveau déjà au-dessus de la moyenne — maintenir."
        else:
            category = "MAINTAIN"
            msg = "Impact performance faible — pas une priorité."

        recos.append(Recommendation(
            feature=f, label=FEATURE_LABELS_FR.get(f, f),
            category=category, perf_impact=perf, injury_weight=weight,
            direction=direction, current_z=z, message=msg,
        ))

    max_r = max(body_risk.values()) or 1.0
    body_risk = {k: min(v / max_r, 1.0) for k, v in body_risk.items()}

    causes_raw.sort(key=lambda x: -x[0])
    max_c = causes_raw[0][0] if causes_raw else 1.0
    causes: list = []
    for raw, c in causes_raw[:causes_top]:
        c.risk_contrib = float(min(raw / max_c, 1.0)) if max_c else 0.0
        causes.append(c)

    focus   = sorted([r for r in recos if r.category == "FOCUS"],
                     key=lambda r: -abs(r.perf_impact))[:focus_top]
    caution = sorted([r for r in recos if r.category == "CAUTION"],
                     key=lambda r: -abs(r.injury_weight * r.perf_impact))[:caution_top]
    maintain = [r for r in recos if r.category == "MAINTAIN"]

    return AdvisorReport(
        player_id=None,
        perf_index=float(last.get("perf_index", values.get("perf_index", 70.0))),
        injury_prob=injury_prob,
        recommendations=focus + caution + maintain,
        causes=causes,
        body_risk=body_risk,
    )


PerformanceAdvisor.advise_from_values = _advise_from_values
