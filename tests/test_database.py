"""
tests/test_database.py  —  aligned with Feature Doc v1.0
Run: python tests/test_database.py
"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from database.database import Database

TEST_DB = "data/test_athlete.db"

def fresh():  return Database(TEST_DB)
def cleanup(db):
    db.close()
    if os.path.exists(TEST_DB): os.remove(TEST_DB)
def assert_(c, msg=""):
    if not c: raise AssertionError(msg or "Assertion failed")

# Minimal session with all Doc v1.0 fields
SESSION = dict(
    date="2026-06-01", session_type="Training", weather="Normal",
    surface="Grass", training_minutes=90, rpe=7,
    # GPS
    distance_km=10.5, sprint_distance_km=0.8, sprints_count=12,
    hid_km=2.1, acceleration_max=4.2, max_speed=32.0,
    accelerations=25, decelerations=18, player_load=320.0,
    heart_rate_avg=165.0, heart_rate_max=190.0,
    # Biometrics
    hrv_rmssd=65.0, fc_repos=52, sommeil_h=7.5, sommeil_qualite=4,
    fatigue=3, spo2=98.5, jours_depuis_match=2, body_temp_celsius=36.6,
    # Mental
    motivation=8, stress=3, reaction_ms=280.0, charge_mentale=6,
    # Nutrition
    poids_variation_pct=-0.5, ck_post=280.0, hydratation_score=2,
    previous_injuries=1, injury_label=0,
)

PHYS_TEST = dict(date="2026-05-01", test_type="Monthly",
                 vo2max=58.0, puissance_w_kg=14.5, cmj_cm=40.0,
                 rsa_index=92.0, force_n=380.0, fatigue_sprint_pct=7.5)

BODY_COMP = dict(date="2026-04-01", measure_type="Quarterly",
                 pct_masse_grasse=12.5, ferritine=85.0, hemoglobine=15.2)

def add_player(db, sport="Football"):
    db.add_player("Test Player", sport, "Midfielder", 24, 1.78, 72,
                  "Right", "Tunisian", "Team A")
    return db.get_players()[0][0]

passed = failed = 0

def run(name, fn):
    global passed, failed
    db = fresh()
    try:
        fn(db)
        print(f"  PASS  {name}")
        passed += 1
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        failed += 1
    finally:
        cleanup(db)

# ── Player tests ──────────────────────────────────────────────
run("player add/get", lambda db: assert_(
    len((add_player(db), db.get_players())[1]) == 1))

run("sport filter", lambda db: (
    add_player(db, "Football"), add_player(db, "Basketball"),
    assert_(len(db.get_players("Football")) == 1 and len(db.get_players()) == 2)))

run("player update", lambda db: (
    db.update_player(add_player(db), "New", "Basketball","G",25,1.88,85,None,"FR","B"),
    assert_(db.get_player(db.get_players()[0][0])[1] == "New")))

run("cascade delete sessions", lambda db: (
    db.add_session(player_id=add_player(db), **SESSION),
    assert_(db.get_session_count() == 1),
    db.delete_player(db.get_players()[0][0]),
    assert_(db.get_session_count() == 0)))

# ── Session / Doc §2-§6 field tests ──────────────────────────
run("GPS fields stored", lambda db: (
    db.add_session(player_id=add_player(db), **SESSION),
    (lambda s, cols: (
        assert_(dict(zip(cols, s))["sprints_count"] == 12),
        assert_(dict(zip(cols, s))["hid_km"] == 2.1),
        assert_(dict(zip(cols, s))["acceleration_max"] == 4.2),
    ))(db.get_sessions(db.get_players()[0][0])[0],
       [d[0] for d in db.cursor.execute("SELECT * FROM sessions LIMIT 1").description])))

run("biometric fields stored", lambda db: (
    db.add_session(player_id=add_player(db), **SESSION),
    (lambda s, cols: (
        assert_(dict(zip(cols, s))["hrv_rmssd"] == 65.0),
        assert_(dict(zip(cols, s))["spo2"] == 98.5),
        assert_(dict(zip(cols, s))["jours_depuis_match"] == 2),
        assert_(dict(zip(cols, s))["sommeil_qualite"] == 4),
    ))(db.get_sessions(db.get_players()[0][0])[0],
       [d[0] for d in db.cursor.execute("SELECT * FROM sessions LIMIT 1").description])))

run("mental fields stored", lambda db: (
    db.add_session(player_id=add_player(db), **SESSION),
    (lambda s, cols: (
        assert_(dict(zip(cols, s))["motivation"] == 8),
        assert_(dict(zip(cols, s))["stress"] == 3),
        assert_(dict(zip(cols, s))["reaction_ms"] == 280.0),
        assert_(dict(zip(cols, s))["charge_mentale"] == 6),
    ))(db.get_sessions(db.get_players()[0][0])[0],
       [d[0] for d in db.cursor.execute("SELECT * FROM sessions LIMIT 1").description])))

run("nutrition fields stored", lambda db: (
    db.add_session(player_id=add_player(db), **SESSION),
    (lambda s, cols: (
        assert_(dict(zip(cols, s))["poids_variation_pct"] == -0.5),
        assert_(dict(zip(cols, s))["ck_post"] == 280.0),
        assert_(dict(zip(cols, s))["hydratation_score"] == 2),
    ))(db.get_sessions(db.get_players()[0][0])[0],
       [d[0] for d in db.cursor.execute("SELECT * FROM sessions LIMIT 1").description])))

# ── Physical tests (Doc §4) ───────────────────────────────────
run("physical test add/get", lambda db: (
    db.add_physical_test(player_id=add_player(db), **PHYS_TEST),
    (lambda tests: (
        assert_(len(tests) == 1),
        assert_(tests[0][4] == 58.0),    # vo2max
        assert_(tests[0][6] == 40.0),    # cmj_cm
    ))(db.get_physical_tests(db.get_players()[0][0]))))

run("physical test latest", lambda db: (
    db.add_physical_test(player_id=add_player(db), **{**PHYS_TEST, "date":"2026-01-01"}),
    db.add_physical_test(player_id=add_player(db) if False else db.get_players()[0][0],
                         **{**PHYS_TEST, "date":"2026-06-01", "vo2max":61.0}),
    assert_(db.get_latest_physical_test(db.get_players()[0][0])[4] == 61.0)))

run("cascade delete physical tests", lambda db: (
    (lambda pid: (
        db.add_physical_test(player_id=pid, **PHYS_TEST),
        db.delete_player(pid),
        assert_(len(db.get_physical_tests(pid)) == 0)
    ))(add_player(db))))

# ── Body composition (Doc §5) ─────────────────────────────────
run("body composition add/get", lambda db: (
    db.add_body_composition(player_id=add_player(db), **BODY_COMP),
    (lambda rows: (
        assert_(len(rows) == 1),
        assert_(rows[0][4] == 12.5),   # pct_masse_grasse
        assert_(rows[0][5] == 85.0),   # ferritine
        assert_(rows[0][6] == 15.2),   # hemoglobine
    ))(db.get_body_composition(db.get_players()[0][0]))))

# ── Injuries ──────────────────────────────────────────────────
run("injury add/get", lambda db: (
    db.add_injury(add_player(db), "2026-06-01", "Muscle strain",
                  "Hamstring", "Moderate", "Non-contact", 21),
    assert_(db.get_injury_count(db.get_players()[0][0]) == 1)))

run("injury cascade delete", lambda db: (
    (lambda pid: (
        db.add_injury(pid,"2026-06-01","Strain","Hamstring","Minor","Non-contact"),
        db.delete_player(pid),
        assert_(db.get_injury_count(pid) == 0)
    ))(add_player(db))))

# ── Sport metrics ─────────────────────────────────────────────
run("sport metrics add/get", lambda db: (
    (lambda pid: (
        (lambda sid: (
            db.add_sport_metrics(sid, "Basketball", points=18, rebounds=7, assists=5),
            assert_(db.get_session_with_metrics(sid)[1] is not None)
        ))(db.add_session(player_id=pid, **SESSION))
    ))(add_player(db, "Basketball"))))

print(f"\n{'─'*40}")
print(f"  {passed} passed  |  {failed} failed")
