"""Accounts + favourites + watched list, saved in one local SQLite file (data/app.db).

Everything that touches the file lives here. To move to a hosted database later
(e.g. for Streamlit Cloud), only this file has to change.
"""
import hashlib
import hmac
import os
import re
import sqlite3
from contextlib import closing

DB_PATH = os.environ.get("MOVIE_APP_DB", "data/app.db")
_USER_RE = re.compile(r"^[A-Za-z0-9_]{3,20}$")


def _conn():
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.execute("PRAGMA foreign_keys = ON")
    return c


def init_db():
    with closing(_conn()) as c, c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS users(
            username TEXT PRIMARY KEY, salt BLOB NOT NULL, pw_hash BLOB NOT NULL,
            created TEXT DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE IF NOT EXISTS favorites(
            username TEXT, movie_id INTEGER, added TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(username, movie_id),
            FOREIGN KEY(username) REFERENCES users(username) ON DELETE CASCADE);
        CREATE TABLE IF NOT EXISTS watched(
            username TEXT, movie_id INTEGER, added TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(username, movie_id),
            FOREIGN KEY(username) REFERENCES users(username) ON DELETE CASCADE);
        """)


def _hash(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000)


def create_user(username, password):
    """Returns (ok, message)."""
    username = username.strip()
    if not _USER_RE.match(username):
        return False, "Username must be 3-20 characters: letters, numbers or underscore."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    salt = os.urandom(16)
    try:
        with closing(_conn()) as c, c:
            c.execute("INSERT INTO users(username, salt, pw_hash) VALUES(?,?,?)",
                      (username, salt, _hash(password, salt)))
    except sqlite3.IntegrityError:
        return False, "That username is already taken."
    return True, "Account created."


def check_login(username, password):
    with closing(_conn()) as c:
        row = c.execute("SELECT salt, pw_hash FROM users WHERE username=?",
                        (username.strip(),)).fetchone()
    if not row:
        return False
    return hmac.compare_digest(_hash(password, row[0]), row[1])


def _table(kind):
    assert kind in ("favorites", "watched")
    return kind


def toggle(username, kind, movie_id):
    """Adds the movie if absent, removes it if present. Returns True if now in the list."""
    t = _table(kind)
    with closing(_conn()) as c, c:
        cur = c.execute(f"DELETE FROM {t} WHERE username=? AND movie_id=?", (username, int(movie_id)))
        if cur.rowcount:
            return False
        c.execute(f"INSERT INTO {t}(username, movie_id) VALUES(?,?)", (username, int(movie_id)))
        return True


def add(username, kind, movie_id):
    t = _table(kind)
    with closing(_conn()) as c, c:
        c.execute(f"INSERT OR IGNORE INTO {t}(username, movie_id) VALUES(?,?)", (username, int(movie_id)))


def get_ids(username, kind):
    """Movie ids, newest first."""
    t = _table(kind)
    with closing(_conn()) as c:
        rows = c.execute(f"SELECT movie_id FROM {t} WHERE username=? ORDER BY added DESC, rowid DESC",
                         (username,)).fetchall()
    return [r[0] for r in rows]
