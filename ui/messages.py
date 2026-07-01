"""
messages.py — Centralized UI messages and labels.

Separates all hardcoded UI strings from business logic, making it easier to:
- Maintain consistent messaging
- Translate to other languages
- Update UI text in one place
"""

# ── DIALOG MESSAGES ─────────────────────────────────────────────────────────
MSG_PLAYER_NAME_REQUIRED = "Le nom du joueur est obligatoire."
MSG_PLAYER_REQUIRED = "Sélectionnez d'abord un joueur."
MSG_PLAYER_NOT_SELECTED = "Sélectionnez un joueur à modifier."
MSG_PLAYER_DELETE_NOT_SELECTED = "Sélectionnez un joueur à supprimer."
MSG_NO_DATA = "Pas assez de données pour ce joueur."
MSG_NO_PLAYER_FIRST = "Sélectionnez d'abord un joueur."
MSG_NO_TEAM_DATA = "Lancez d'abord l'analyse équipe."

# ── SUCCESS MESSAGES ────────────────────────────────────────────────────────
MSG_PLAYER_ADDED = "Le joueur a bien été enregistré."
MSG_SESSION_ADDED = "La séance a bien été ajoutée."
MSG_TRAINING_COMPLETED = "Modèle global + sous-modèles par sport mis à jour."
MSG_EXPORT_SUCCESS = "PDF sauvegardé :"
MSG_EXPORT_BATCH_SUCCESS = "rapports générés dans :"

# ── ERROR MESSAGES ──────────────────────────────────────────────────────────
MSG_ERROR_GENERIC = "Erreur"
MSG_ERROR_EXPORT = "Erreur export"
MSG_ERROR_INSUFFICIENT_DATA = "Données insuffisantes"
MSG_ERROR_NO_DATA = "Aucune donnée"
MSG_ERROR_TITLE = "Erreur"

# ── DIALOG TITLES ───────────────────────────────────────────────────────────
TITLE_REQUIRED_FIELDS = "Champs requis"
TITLE_REQUIRED_PLAYER = "Joueur requis"
TITLE_NO_PLAYER = "Aucun joueur"
TITLE_TRAINING_DONE = "Entraînement terminé"
TITLE_EXPORT_SUCCESS = "Export réussi"
TITLE_EXPORT_ERROR = "Erreur export"
TITLE_EXPORT_DONE = "Export terminé"
TITLE_ERROR = "Erreur"
TITLE_NO_DATA = "Aucune donnée"

# ── LABEL DEFAULTS ──────────────────────────────────────────────────────────
LABEL_PLAYER_NAME = "Nom : —"
LABEL_PLAYER_SPORT = "Sport : —"
LABEL_PLAYER_POSITION = "Poste : —"
LABEL_PLAYER_AGE = "Âge : —"
LABEL_PLAYER_TEAM = "Équipe : —"
LABEL_RTP_AVAILABLE = "✅ Joueur disponible — pas de protocole RTP actif."
LABEL_TEAM_ANALYSIS = "Analyse de l'équipe en cours…"
LABEL_TRAINING_IN_PROGRESS = "Entraînement en cours… (peut prendre plusieurs minutes)"
LABEL_NO_RESULT = "Aucun résultat."
LABEL_THEME_LIGHT = "☀ Light"
LABEL_THEME_DARK = "🌙 Dark"

# ── BUTTON TEXTS ────────────────────────────────────────────────────────────
BTN_SAVE = "Save"
BTN_CANCEL = "Cancel"
BTN_CONFIRM_DELETE = "Confirmer"

# ── POSITION CHOICES ────────────────────────────────────────────────────────
POSITIONS = ["GK", "DEF", "MID", "FW"]

# ── FOOT CHOICES ────────────────────────────────────────────────────────────
FOOT_CHOICES = ["Right", "Left", "Both"]
