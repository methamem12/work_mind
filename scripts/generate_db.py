"""
scripts/generate_db.py
Run from the project root: python scripts/generate_db.py

Generates a realistic multi-sport athlete database.
All 25 raw features + 4 computed features from the IA module documentation are populated.

Dimensions covered:
  GPS / Charge physique     → distance_km, sprints_count, hid_km, acceleration_max,
                               player_load  (+computed: acwr, charge_7j, charge_28j)
  Biométrie & Récupération  → hrv_rmssd, fc_repos, sommeil_h, sommeil_qualite,
                               fatigue, spo2, jours_depuis_match
                               (+computed: hrv_trend, hrv_moy_7j, fc_repos_alerte)
  Mental & Cognitif         → motivation, stress, reaction_ms, charge_mentale,
                               regularite_score
  Nutrition & Hydratation   → poids_variation_pct, hydratation_score, ck_post
                               (+labs: pct_masse_grasse, ferritine, hemoglobine)
  Capacités physiques       → vo2max, puissance_w_kg, cmj_cm, rsa_index,
                               force_n, fatigue_sprint_pct  (monthly tests)
  Composées                 → wellness_score, perf_index
"""

import sys, math, random, datetime
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from database.database import Database

random.seed(42)
np.random.seed(42)

# ── Identity ─────────────────────────────────────────────────────────────────
FIRST_NAMES = [
    "Yassine","Mohamed","Karim","Amir","Bilal","Hamza","Sami","Omar",
    "Nabil","Anas","Achraf","Ilyes","Rayan","Mehdi","Sofiane","Ayoub",
    "Samir","Lotfi","Houssem","Fares","Ghaith","Seifeddine","Bassem",
    "Wissem","Maher","Tarek","Skander","Aziz","Oussama","Walid",
    "Liam","Noah","Ethan","James","Oliver","Lucas","Mason","Carlos",
    "Diego","Mateo","Sergio","Ivan","Emil","Lars","Finn","Aiden",
]
LAST_NAMES = [
    "Ben Ali","Gharbi","Msakni","Sliti","Khazri","Badri","Jebali",
    "Chaalali","Laifi","Sassi","Derbali","Amri","Khedira","Jaziri",
    "Trabelsi","Ferchichi","Bouzid","Mansour","Dridi","Hamdi",
    "Smith","Johnson","Williams","Brown","Jones","Garcia","Martinez",
    "Anderson","Taylor","Thomas","Jackson","White","Harris","Martin",
]
NATIONALITIES = ["Tunisian","Algerian","Moroccan","French","Spanish","German",
                 "Brazilian","Argentine","British","Italian","Dutch","Portuguese"]
FOOTS = ["Left","Right","Both"]

# ── Sport configuration ──────────────────────────────────────────────────────
SPORT_CONFIG = {
    "Football": {
        "positions": ["Goalkeeper","Defender","Midfielder","Forward"],
        "surfaces":  ["Grass","Artificial"],
        "session_types": ["Training","Match","Recovery","Gym"],
        "load_profiles": {
            "Goalkeeper": dict(d=(5.5,0.6), sc=(8,3),  hid=(0.4,0.1), am=(6.5,0.8), pl=380),
            "Defender":   dict(d=(9.2,0.9), sc=(18,5), hid=(1.1,0.3), am=(7.5,0.9), pl=620),
            "Midfielder": dict(d=(10.8,1.0),sc=(22,6), hid=(1.4,0.4), am=(7.0,0.8), pl=720),
            "Forward":    dict(d=(9.6,0.95),sc=(30,8), hid=(1.6,0.5), am=(8.0,1.0), pl=680),
        },
    },
    "Basketball": {
        "positions": ["Point Guard","Shooting Guard","Small Forward","Power Forward","Center"],
        "surfaces":  ["Hardwood","Indoor"],
        "session_types": ["Training","Match","Recovery","Gym"],
        "load_profiles": {
            "Point Guard":    dict(d=(6.0,0.7), sc=(12,4), hid=(0.9,0.2), am=(5.0,0.6), pl=420),
            "Shooting Guard": dict(d=(5.5,0.65),sc=(10,3), hid=(0.8,0.2), am=(4.8,0.6), pl=390),
            "Small Forward":  dict(d=(5.8,0.68),sc=(9,3),  hid=(0.8,0.2), am=(4.6,0.6), pl=400),
            "Power Forward":  dict(d=(4.8,0.60),sc=(6,2),  hid=(0.5,0.1), am=(4.0,0.5), pl=340),
            "Center":         dict(d=(4.2,0.55),sc=(4,2),  hid=(0.3,0.1), am=(3.5,0.5), pl=300),
        },
    },
    "Tennis": {
        "positions": ["Singles","Doubles"],
        "surfaces":  ["Clay","Grass","Hard","Indoor Hard"],
        "session_types": ["Training","Match","Recovery"],
        "load_profiles": {
            "Singles": dict(d=(8.0,1.5), sc=(15,5), hid=(1.2,0.4), am=(7.0,1.0), pl=500),
            "Doubles": dict(d=(5.5,1.0), sc=(9,3),  hid=(0.7,0.2), am=(6.0,0.8), pl=350),
        },
    },
    "Swimming": {
        "positions": ["Freestyle","Backstroke","Breaststroke","Butterfly","Individual Medley"],
        "surfaces":  ["Pool","Open Water"],
        "session_types": ["Training","Competition","Recovery"],
        "load_profiles": {
            "Freestyle":         dict(d=(4.0,0.8), sc=(0,0), hid=(1.0,0.3), am=(2.0,0.2), pl=280),
            "Backstroke":        dict(d=(3.5,0.7), sc=(0,0), hid=(0.8,0.2), am=(1.8,0.2), pl=250),
            "Breaststroke":      dict(d=(3.2,0.65),sc=(0,0), hid=(0.6,0.2), am=(1.6,0.2), pl=220),
            "Butterfly":         dict(d=(2.8,0.60),sc=(0,0), hid=(0.8,0.2), am=(1.9,0.2), pl=260),
            "Individual Medley": dict(d=(3.8,0.75),sc=(0,0), hid=(0.9,0.3), am=(1.9,0.2), pl=270),
        },
    },
    "Rugby": {
        "positions": ["Prop","Hooker","Lock","Flanker","Number 8",
                      "Scrum-half","Fly-half","Centre","Winger","Fullback"],
        "surfaces":  ["Grass","Artificial"],
        "session_types": ["Training","Match","Recovery","Gym"],
        "load_profiles": {
            "Prop":      dict(d=(5.0,0.6),sc=(3,1),  hid=(0.4,0.1), am=(6.0,0.8), pl=450),
            "Hooker":    dict(d=(5.2,0.62),sc=(4,2), hid=(0.5,0.1), am=(6.0,0.8), pl=460),
            "Lock":      dict(d=(6.0,0.7), sc=(5,2), hid=(0.6,0.2), am=(6.5,0.8), pl=490),
            "Flanker":   dict(d=(7.5,0.8), sc=(8,3), hid=(0.9,0.2), am=(7.5,0.9), pl=590),
            "Number 8":  dict(d=(7.0,0.75),sc=(7,3), hid=(0.8,0.2), am=(7.3,0.9), pl=560),
            "Scrum-half":dict(d=(7.2,0.78),sc=(9,3), hid=(1.0,0.3), am=(6.8,0.8), pl=550),
            "Fly-half":  dict(d=(7.0,0.75),sc=(10,3),hid=(1.1,0.3), am=(7.0,0.8), pl=540),
            "Centre":    dict(d=(7.8,0.82),sc=(13,4),hid=(1.3,0.3), am=(8.0,1.0), pl=580),
            "Winger":    dict(d=(7.5,0.8), sc=(18,5),hid=(1.5,0.4), am=(8.5,1.0), pl=570),
            "Fullback":  dict(d=(7.6,0.81),sc=(15,4),hid=(1.4,0.4), am=(8.2,1.0), pl=575),
        },
    },
    "Athletics": {
        "positions": ["Sprinter","Middle Distance","Long Distance",
                      "Hurdles","Jumps","Throws","Decathlon"],
        "surfaces":  ["Track","Road","Cross Country","Indoor Track"],
        "session_types": ["Training","Competition","Recovery"],
        "load_profiles": {
            "Sprinter":        dict(d=(3.0,0.5),  sc=(25,8), hid=(1.5,0.4), am=(9.5,1.0), pl=320),
            "Middle Distance": dict(d=(8.0,1.2),  sc=(15,5), hid=(2.5,0.6), am=(7.0,0.8), pl=480),
            "Long Distance":   dict(d=(15.0,3.0), sc=(5,2),  hid=(3.0,0.8), am=(5.0,0.6), pl=580),
            "Hurdles":         dict(d=(4.0,0.7),  sc=(20,6), hid=(1.8,0.5), am=(9.0,1.0), pl=350),
            "Jumps":           dict(d=(3.5,0.6),  sc=(18,5), hid=(1.3,0.4), am=(9.2,1.0), pl=330),
            "Throws":          dict(d=(2.5,0.4),  sc=(3,1),  hid=(0.3,0.1), am=(8.0,1.0), pl=280),
            "Decathlon":       dict(d=(6.0,1.0),  sc=(15,5), hid=(2.0,0.5), am=(8.5,1.0), pl=420),
        },
    },
    "MMA/Boxing": {
        "positions": ["MMA Fighter","Boxer","BJJ Specialist","Wrestler"],
        "surfaces":  ["Mat","Ring","Cage"],
        "session_types": ["Training","Competition","Recovery","Gym"],
        "load_profiles": {
            "MMA Fighter":    dict(d=(3.5,0.6),sc=(8,3), hid=(0.7,0.2), am=(8.5,1.0), pl=520),
            "Boxer":          dict(d=(3.0,0.5),sc=(6,2), hid=(0.5,0.1), am=(8.0,1.0), pl=480),
            "BJJ Specialist": dict(d=(2.5,0.4),sc=(4,2), hid=(0.3,0.1), am=(6.0,0.8), pl=420),
            "Wrestler":       dict(d=(3.0,0.5),sc=(5,2), hid=(0.4,0.1), am=(7.0,0.9), pl=460),
        },
    },
    "Cycling": {
        "positions": ["Road Cyclist","Track Cyclist","Mountain Biker","Triathlete"],
        "surfaces":  ["Road","Track","Trail","Velodrome"],
        "session_types": ["Training","Race","Recovery"],
        "load_profiles": {
            "Road Cyclist":  dict(d=(80,20),  sc=(0,0), hid=(20,6),  am=(3.0,0.5), pl=900),
            "Track Cyclist": dict(d=(40,10),  sc=(0,0), hid=(25,7),  am=(4.0,0.6), pl=750),
            "Mountain Biker":dict(d=(30,8),   sc=(0,0), hid=(12,4),  am=(5.0,0.7), pl=820),
            "Triathlete":    dict(d=(60,15),  sc=(0,0), hid=(18,5),  am=(3.5,0.5), pl=850),
        },
    },
}

# ── Physical test baselines per sport ────────────────────────────────────────
PHYS_BASELINES = {
    "Football":   dict(vo2=(55,5), pw=(14,2), cmj=(42,5), rsa=(92,3), fn=(350,50), fsp=(8,3)),
    "Basketball": dict(vo2=(52,5), pw=(13,2), cmj=(55,6), rsa=(90,3), fn=(320,50), fsp=(9,3)),
    "Tennis":     dict(vo2=(53,5), pw=(12,2), cmj=(38,5), rsa=(88,3), fn=(280,40), fsp=(10,3)),
    "Swimming":   dict(vo2=(58,6), pw=(11,2), cmj=(30,4), rsa=(85,3), fn=(260,40), fsp=(7,2)),
    "Rugby":      dict(vo2=(52,5), pw=(16,2), cmj=(40,5), rsa=(89,3), fn=(450,60), fsp=(10,3)),
    "Athletics":  dict(vo2=(62,6), pw=(18,3), cmj=(50,6), rsa=(94,3), fn=(380,55), fsp=(6,2)),
    "MMA/Boxing": dict(vo2=(55,5), pw=(14,2), cmj=(38,5), rsa=(88,3), fn=(360,50), fsp=(9,3)),
    "Cycling":    dict(vo2=(65,6), pw=(22,3), cmj=(32,4), rsa=(86,3), fn=(300,45), fsp=(8,2)),
}

INJURY_PROFILES = {
    "Football":   [("Muscle strain","Hamstring","Non-contact"),("Ligament sprain","Knee","Contact"),("Contusion","Ankle","Contact"),("Overuse","Knee","Overuse")],
    "Basketball": [("Ligament sprain","Ankle","Non-contact"),("Muscle strain","Quad","Non-contact"),("Overuse","Knee","Overuse")],
    "Tennis":     [("Overuse","Shoulder","Overuse"),("Muscle strain","Calf","Non-contact"),("Overuse","Elbow","Overuse")],
    "Swimming":   [("Overuse","Shoulder","Overuse"),("Overuse","Back","Overuse")],
    "Rugby":      [("Contusion","Shoulder","Contact"),("Ligament sprain","Knee","Contact"),("Fracture","Rib","Contact")],
    "Athletics":  [("Muscle strain","Hamstring","Non-contact"),("Overuse","Shin","Overuse"),("Ligament sprain","Ankle","Non-contact")],
    "MMA/Boxing": [("Contusion","Face","Contact"),("Ligament sprain","Shoulder","Contact"),("Fracture","Hand","Contact")],
    "Cycling":    [("Overuse","Knee","Overuse"),("Contusion","Hip","Contact"),("Fracture","Collarbone","Contact")],
}

SEVERITY_DAYS = {"Minor":(3,14),"Moderate":(14,42),"Severe":(42,180)}


def _clamp(v, lo, hi): return max(lo, min(hi, v))
def _gc(mu, sigma, lo, hi, dec=1):
    return round(_clamp(random.gauss(mu, sigma), lo, hi), dec)


def sample_session_load(sport, position, scale=1.0):
    p = SPORT_CONFIG[sport]["load_profiles"][position]
    d   = _gc(p["d"][0]*scale,  p["d"][1]*scale,  0.5, 200)
    sc  = max(0, int(random.gauss(p["sc"][0]*scale, max(p["sc"][1],0.1))))
    hid = _gc(p["hid"][0]*scale, p["hid"][1]*scale, 0, d*0.5)
    am  = _gc(p["am"][0],        p["am"][1],         1, 15)
    pl  = _gc(p["pl"]*scale,     p["pl"]*0.12,       10, 2000)
    tm  = int(_clamp(random.gauss(d/0.095 if sport != "Cycling" else 120, 10), 15, 300))
    ms  = _gc(28 if sport in ("Football","Rugby","Athletics") else 20, 3, 8, 45)
    acc = max(0, int(random.gauss(40*scale, 8)))
    dec = max(0, int(random.gauss(38*scale, 8)))
    hra = _gc(148*min(scale,1.1), 12, 80, 210)
    hrm = _gc(182*min(scale,1.05), 7, 120, 220)
    # sprint_distance derived from sprints_count × ~20m avg sprint
    spr_km = round(sc * _gc(0.022, 0.005, 0.010, 0.040), 3)
    return dict(training_minutes=tm, distance_km=d, sprint_distance_km=spr_km,
                sprints_count=sc, hid_km=hid, acceleration_max=am, max_speed=ms,
                accelerations=acc, decelerations=dec, player_load=pl,
                heart_rate_avg=hra, heart_rate_max=hrm)


def sample_biometrics(avg_load, match_last_n_days=None):
    """All biometry + mental + nutrition daily features."""
    # Recovery / sleep improve with rest, degrade with high load
    load_factor = min(avg_load / 700, 1.5)
    hrv    = _gc(68 - load_factor*15, 12, 20, 120)
    fc_r   = int(_clamp(random.gauss(52 + load_factor*8, 5), 35, 90))
    slp_h  = _gc(7.5 - load_factor*0.8, 0.7, 4, 10)
    slp_q  = int(_clamp(random.gauss(4.0 - load_factor*0.6, 0.8), 1, 5))
    fat    = int(_clamp(random.gauss(3 + load_factor*3, 1.5), 1, 10))
    spo2   = _gc(98.0 - load_factor*0.5, 0.5, 92, 100)
    jdm    = match_last_n_days if match_last_n_days is not None else random.randint(1, 7)
    btemp  = _gc(36.6, 0.2, 36.0, 38.5)
    # Mental
    mot    = int(_clamp(random.gauss(7.0 - load_factor*1.2, 1.2), 1, 10))
    str_   = int(_clamp(random.gauss(3.0 + load_factor*1.5, 1.5), 1, 10))
    rxn    = _gc(220 + load_factor*30, 20, 150, 400)
    cmt    = int(_clamp(random.gauss(5.0 + load_factor, 1.5), 1, 10))
    reg    = round(_clamp(random.gauss(0.82 - load_factor*0.1, 0.1), 0, 1), 2)
    # Nutrition daily
    pv_pct = _gc(load_factor*0.8, 0.5, -3, 4)      # weight variation %
    hyd    = int(_clamp(random.gauss(2.5 + load_factor, 1.0), 1, 8))
    ck     = _gc(200 + load_factor*300, 80, 50, 3000)  # CK post-effort
    # Derived simple wellness (will be overwritten by preprocessing but stored for speed)
    fat_inv = (10 - fat) / 9.0
    som_n   = (slp_h - 4) / 6.0
    str_inv = (10 - str_) / 9.0
    mot_n   = (mot - 1) / 9.0
    wellness = round(min(10, ((fat_inv + som_n + str_inv + mot_n) / 4) * 10), 2)
    return dict(hrv_rmssd=hrv, fc_repos=fc_r, sommeil_h=slp_h, sommeil_qualite=slp_q,
                fatigue=fat, spo2=spo2, jours_depuis_match=jdm, body_temp_celsius=btemp,
                motivation=mot, stress=str_, reaction_ms=rxn, charge_mentale=cmt,
                regularite_score=reg, poids_variation_pct=pv_pct,
                hydratation_score=hyd, ck_post=ck, wellness_score=wellness)


def sample_physical_test(sport, age):
    """Monthly physical test values with age-related degradation."""
    b = PHYS_BASELINES[sport]
    age_factor = max(0, (age - 28) * 0.008)   # slight decline after 28
    return dict(
        vo2max          = _gc(b["vo2"][0] - age_factor*b["vo2"][0], b["vo2"][1], 25, 85, 1),
        puissance_w_kg  = _gc(b["pw"][0]  - age_factor*b["pw"][0],  b["pw"][1],  4, 35, 2),
        cmj_cm          = _gc(b["cmj"][0] - age_factor*b["cmj"][0], b["cmj"][1], 10, 80, 1),
        rsa_index       = _gc(b["rsa"][0] - age_factor*2,           b["rsa"][1], 70,100, 1),
        force_n         = _gc(b["fn"][0]  - age_factor*b["fn"][0],  b["fn"][1],  100,700,0),
        fatigue_sprint_pct = _gc(b["fsp"][0] + age_factor*3,       b["fsp"][1],  0, 40, 1),
    )


def compute_acwr(loads):
    if len(loads) < 3: return 1.0
    a = float(np.mean(loads[-7:]))
    c = float(np.mean(loads[-28:]))
    return a / c if c > 0.1 else 1.0


def injury_probability(session, bio, prev_inj, acwr, age):
    log_odds = -1.90
    if   acwr < 0.8:  log_odds += (0.8 - acwr) * 1.2
    elif acwr <= 1.3: pass
    elif acwr <= 1.5: log_odds += (acwr - 1.3) * 3.0
    else:             log_odds += 0.6 + (acwr - 1.5) * 5.0
    log_odds += (bio["fatigue"] - 5) / 10 * 1.8
    log_odds += (5.0 - bio["wellness_score"] / 2) / 10 * 1.4
    log_odds += prev_inj * 0.55
    if age > 28: log_odds += (age - 28) * 0.04
    if session.get("sprints_count", 0) > 25:
        log_odds += (session["sprints_count"] - 25) / 40 * 0.8
    if session.get("player_load", 0) > 900:
        log_odds += (session["player_load"] - 900) / 600 * 0.8
    return _clamp(1 / (1 + math.exp(-log_odds)), 0.01, 0.99)


def generate_players(db, n=80):
    sports = list(SPORT_CONFIG.keys())
    per_sport = n // len(sports)
    meta = {}; used = set()

    for sport in sports:
        for _ in range(per_sport):
            for _ in range(20):
                name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
                if name not in used: used.add(name); break
            pos = random.choice(SPORT_CONFIG[sport]["positions"])
            age = random.randint(17, 38)
            db.add_player(name=name, sport=sport, position=pos, age=age,
                          height=round(random.uniform(1.60, 2.05), 2),
                          weight=random.randint(55, 120),
                          dominant_foot=random.choice(FOOTS) if sport in ("Football","Rugby","Athletics") else None,
                          nationality=random.choice(NATIONALITIES),
                          team=f"Team {random.choice('ABCDEF')}")

    for p in db.get_players():
        meta[p[0]] = {"sport": p[2], "position": p[3], "age": p[4],
                      "prev_inj": random.choices([0,1,2,3], weights=[.50,.30,.15,.05])[0]}
    print(f"  ✓ {len(meta)} players across {len(sports)} sports")
    return meta


def generate_physical_tests(db, players_meta, n_weeks=36):
    """Monthly physical tests — stored in physical_tests table."""
    start = datetime.date(2025, 7, 1)
    for pid, meta in players_meta.items():
        latest_test = sample_physical_test(meta["sport"], meta["age"])
        for month in range(0, (n_weeks // 4) + 1):
            test_date = start + datetime.timedelta(weeks=month * 4)
            # Add some variation between tests
            t = {k: round(v + random.gauss(0, abs(v)*0.04), 2) for k, v in latest_test.items()}
            db.add_physical_test(pid, str(test_date), **t)
    print(f"  ✓ Physical tests generated")


def generate_nutrition_labs(db, players_meta, n_weeks=36):
    """Quarterly blood work — stored in nutrition_labs table."""
    start = datetime.date(2025, 7, 1)
    for pid, meta in players_meta.items():
        for quarter in range(0, (n_weeks // 12) + 1):
            lab_date = start + datetime.timedelta(weeks=quarter * 12)
            db.add_nutrition_lab(
                pid, str(lab_date), record_type="quarterly",
                pct_masse_grasse = _gc(14 if meta["age"] < 28 else 16, 3, 5, 30),
                ferritine        = _gc(80, 30, 10, 300),
                hemoglobine      = _gc(15.2, 1.0, 11, 18),
            )
    print(f"  ✓ Nutrition labs generated")


def generate_sessions(db, players_meta, n_weeks=36):
    start = datetime.date(2025, 7, 1)
    load_history = {pid: [] for pid in players_meta}
    match_dates  = {pid: None for pid in players_meta}
    total = injuries_count = 0

    # Pre-fetch latest physical tests per player for session-level interpolation
    phys_cache = {}
    for pid, meta in players_meta.items():
        phys_cache[pid] = sample_physical_test(meta["sport"], meta["age"])

    for week in range(n_weeks):
        for pid, meta in players_meta.items():
            sport = meta["sport"]; cfg = SPORT_CONFIG[sport]
            n_sess = random.choices([3,4,5], weights=[.20,.50,.30])[0]
            days   = sorted(random.sample(range(7), n_sess))

            for day_off in days:
                date = start + datetime.timedelta(weeks=week, days=day_off)

                s_types = cfg["session_types"]
                match_kw = ("Match","Competition","Race")
                is_match_type = any(k in t for t in s_types for k in match_kw)
                weights = []
                for st in s_types:
                    if any(k in st for k in match_kw): weights.append(0.15)
                    elif st == "Recovery":              weights.append(0.10)
                    elif st == "Gym":                   weights.append(0.10)
                    else:                               weights.append(0.65)
                session_type = random.choices(s_types, weights=weights)[0]
                surface      = random.choice(cfg["surfaces"])
                is_match     = any(k in session_type for k in match_kw)
                if is_match: match_dates[pid] = date

                scale = {"Training":1.0, "Match":1.30, "Competition":1.35,
                         "Race":1.30, "Recovery":0.50, "Gym":0.60}.get(session_type, 1.0)

                s   = sample_session_load(sport, meta["position"], scale)
                recent = load_history[pid][-7:] if load_history[pid] else [500]
                avg_l  = float(np.mean(recent))

                jdm = (date - match_dates[pid]).days if match_dates[pid] else random.randint(1, 14)
                bio = sample_biometrics(avg_l, match_last_n_days=jdm)

                # Current physical test values (simplest: use cached monthly value + noise)
                phys = {k: round(v + random.gauss(0, abs(v)*0.02), 2)
                        for k, v in phys_cache[pid].items()}

                acwr   = compute_acwr(load_history[pid])
                p_inj  = injury_probability(s, bio, meta["prev_inj"], acwr, meta["age"])
                label  = 1 if random.random() < p_inj else 0
                if label: injuries_count += 1

                rpe = int(_clamp(random.gauss(scale * 6, 1.5), 1, 10))
                weather = random.choice(["Normal","Hot","Cold","Rainy","Indoor"])
                c7  = float(np.mean(load_history[pid][-7:]))  if len(load_history[pid])>=7  else avg_l
                c28 = float(np.mean(load_history[pid][-28:])) if len(load_history[pid])>=28 else avg_l

                sid = db.add_session(
                    player_id=pid, date=str(date),
                    session_type=session_type, weather=weather, surface=surface,
                    training_minutes=s["training_minutes"], rpe=rpe,
                    # GPS
                    distance_km=s["distance_km"], sprint_distance_km=s["sprint_distance_km"],
                    sprints_count=s["sprints_count"], hid_km=s["hid_km"],
                    acceleration_max=s["acceleration_max"], max_speed=s["max_speed"],
                    accelerations=s["accelerations"], decelerations=s["decelerations"],
                    player_load=s["player_load"],
                    # GPS computed
                    acwr=round(acwr, 3), charge_7j=round(c7/1000,3), charge_28j=round(c28/1000,3),
                    # Biometry
                    hrv_rmssd=bio["hrv_rmssd"], fc_repos=bio["fc_repos"],
                    sommeil_h=bio["sommeil_h"], sommeil_qualite=bio["sommeil_qualite"],
                    fatigue=bio["fatigue"], spo2=bio["spo2"],
                    jours_depuis_match=jdm, body_temp_celsius=bio["body_temp_celsius"],
                    # Mental
                    motivation=bio["motivation"], stress=bio["stress"],
                    reaction_ms=bio["reaction_ms"], charge_mentale=bio["charge_mentale"],
                    regularite_score=bio["regularite_score"],
                    # Nutrition daily
                    poids_variation_pct=bio["poids_variation_pct"],
                    hydratation_score=bio["hydratation_score"], ck_post=bio["ck_post"],
                    # Physical tests (interpolated monthly value)
                    vo2max=phys["vo2max"], puissance_w_kg=phys["puissance_w_kg"],
                    cmj_cm=phys["cmj_cm"], rsa_index=phys["rsa_index"],
                    force_n=phys["force_n"], fatigue_sprint_pct=phys["fatigue_sprint_pct"],
                    # Wellness composite
                    wellness_score=bio["wellness_score"],
                    # Labels
                    previous_injuries=meta["prev_inj"], injury_label=label,
                )

                if label and random.random() < 0.8:
                    inj = random.choice(INJURY_PROFILES.get(sport, [("Muscle strain","Other","Non-contact")]))
                    sev = random.choices(["Minor","Moderate","Severe"], weights=[.55,.35,.10])[0]
                    lo, hi = SEVERITY_DAYS[sev]
                    db.add_injury(pid, str(date), inj[0], inj[1], sev, inj[2],
                                  random.randint(lo, hi), session_id=sid)

                load_history[pid].append(s["player_load"])
                total += 1

    print(f"  ✓ {total} sessions | injury rate: {injuries_count/total:.1%}")


if __name__ == "__main__":
    import os
    db_path = "data/athlete.db"
    if os.path.exists(db_path): os.remove(db_path); print("  ↺ DB removed")

    db = Database(db_path)
    print("Generating multi-sport athlete database (all IA module features)...")
    meta = generate_players(db, n=80)
    generate_physical_tests(db, meta, n_weeks=36)
    generate_nutrition_labs(db, meta, n_weeks=36)
    generate_sessions(db, meta, n_weeks=36)

    print("\n  Running preprocessing (ACWR, HRV trend, wellness, perf_index)...")
    db.run_preprocessing()

    print(f"\n  Players        : {len(db.get_players())}")
    print(f"  Sessions       : {db.get_session_count()}")

    import sqlite3, pandas as pd
    conn = sqlite3.connect(db_path)
    pt = pd.read_sql_query("SELECT COUNT(*) n FROM physical_tests", conn).iloc[0,0]
    nl = pd.read_sql_query("SELECT COUNT(*) n FROM nutrition_labs",  conn).iloc[0,0]
    inj= pd.read_sql_query("SELECT COUNT(*) n FROM injuries",        conn).iloc[0,0]
    print(f"  Physical tests : {pt}")
    print(f"  Nutrition labs : {nl}")
    print(f"  Injuries       : {inj}")
    conn.close()
    db.close()
    print("\nDone.")
