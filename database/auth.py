"""
database/auth.py — Local login & role system (no server required).

Roles:
    "coach"   — sees recommendations + risk levels only (no raw feature values)
    "medical" — sees everything (raw features, SHAP, causes, RTP details)

Passwords are stored salted+hashed (PBKDF2-HMAC-SHA256), never in plaintext.
"""
from __future__ import annotations
import sqlite3, os, hashlib, binascii
from dataclasses import dataclass
from typing import Optional

DB_PATH = "data/athlete.db"
ROLES = ("coach", "medical", "admin")


@dataclass
class User:
    id: int
    username: str
    role: str
    full_name: str


def _hash_password(password: str, salt: Optional[bytes] = None) -> str:
    if salt is None:
        salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return binascii.hexlify(salt).decode() + ":" + binascii.hexlify(dk).decode()


def _verify_password(password: str, stored: str) -> bool:
    try:
        salt_hex, hash_hex = stored.split(":")
        salt = binascii.unhexlify(salt_hex)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
        return binascii.hexlify(dk).decode() == hash_hex
    except Exception:
        return False


def ensure_users_table(db_path: str = DB_PATH) -> None:
    with sqlite3.connect(db_path) as cx:
        cx.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                username    TEXT UNIQUE NOT NULL,
                password    TEXT NOT NULL,
                role        TEXT NOT NULL CHECK(role IN ('coach','medical','admin')),
                full_name   TEXT NOT NULL,
                created_at  TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cx.commit()
        # Seed default accounts if table is empty
        n = cx.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        if n == 0:
            defaults = [
                ("coach",   "coach123",   "coach",   "Entraîneur Principal"),
                ("medical", "medical123", "medical", "Staff Médical"),
                ("admin",   "admin123",   "admin",   "Administrateur"),
            ]
            for username, pwd, role, full_name in defaults:
                cx.execute(
                    "INSERT INTO users (username,password,role,full_name) VALUES (?,?,?,?)",
                    (username, _hash_password(pwd), role, full_name))
            cx.commit()


def authenticate(username: str, password: str, db_path: str = DB_PATH) -> Optional[User]:
    ensure_users_table(db_path)
    with sqlite3.connect(db_path) as cx:
        row = cx.execute(
            "SELECT id, username, password, role, full_name FROM users WHERE username=?",
            (username,)).fetchone()
    if row is None:
        return None
    uid, uname, stored_hash, role, full_name = row
    if not _verify_password(password, stored_hash):
        return None
    return User(id=uid, username=uname, role=role, full_name=full_name)


def add_user(username: str, password: str, role: str, full_name: str,
            db_path: str = DB_PATH) -> None:
    if role not in ROLES:
        raise ValueError(f"Rôle invalide : {role}. Choisir parmi {ROLES}.")
    ensure_users_table(db_path)
    with sqlite3.connect(db_path) as cx:
        cx.execute(
            "INSERT INTO users (username,password,role,full_name) VALUES (?,?,?,?)",
            (username, _hash_password(password), role, full_name))
        cx.commit()


def list_users(db_path: str = DB_PATH) -> list:
    ensure_users_table(db_path)
    with sqlite3.connect(db_path) as cx:
        rows = cx.execute(
            "SELECT id, username, role, full_name FROM users ORDER BY role, username"
        ).fetchall()
    return [User(id=r[0], username=r[1], role=r[2], full_name=r[3]) for r in rows]


# ── Permission helpers ──────────────────────────────────────────────────────
def can_see_raw_features(user: User) -> bool:
    return user.role in ("medical", "admin")

def can_see_shap(user: User) -> bool:
    return user.role in ("medical", "admin")

def can_manage_users(user: User) -> bool:
    return user.role == "admin"

def can_export_pdf(user: User) -> bool:
    return True  # both roles can export

def can_edit_players(user: User) -> bool:
    return user.role in ("medical", "admin")
