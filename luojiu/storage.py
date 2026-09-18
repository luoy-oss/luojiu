from __future__ import annotations

import contextlib
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS users(
 id TEXT PRIMARY KEY, name TEXT NOT NULL, password TEXT NOT NULL, created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS sessions(
 token TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id), expires REAL NOT NULL);
CREATE TABLE IF NOT EXISTS groups(
 id TEXT PRIMARY KEY, name TEXT NOT NULL, invite TEXT UNIQUE NOT NULL,
 owner TEXT NOT NULL REFERENCES users(id), created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS members(
 group_id TEXT NOT NULL REFERENCES groups(id), user_id TEXT NOT NULL REFERENCES users(id),
 PRIMARY KEY(group_id,user_id));
CREATE TABLE IF NOT EXISTS messages(
 id INTEGER PRIMARY KEY AUTOINCREMENT, group_id TEXT NOT NULL REFERENCES groups(id),
 user_id TEXT NOT NULL, name TEXT NOT NULL, text TEXT NOT NULL, created REAL NOT NULL,
 reply_to INTEGER REFERENCES messages(id), kind TEXT NOT NULL DEFAULT 'human',
 label_id INTEGER, confidence REAL, reason TEXT, query TEXT);
CREATE INDEX IF NOT EXISTS messages_group ON messages(group_id,id);
CREATE TABLE IF NOT EXISTS answers(
 id INTEGER PRIMARY KEY AUTOINCREMENT, group_id TEXT NOT NULL REFERENCES groups(id),
 text TEXT NOT NULL, created REAL NOT NULL, UNIQUE(group_id,text));
CREATE TABLE IF NOT EXISTS examples(
 id INTEGER PRIMARY KEY AUTOINCREMENT, group_id TEXT NOT NULL REFERENCES groups(id),
 question TEXT NOT NULL, answer_id INTEGER NOT NULL REFERENCES answers(id),
 author TEXT NOT NULL REFERENCES users(id), status TEXT NOT NULL,
 source TEXT NOT NULL, created REAL NOT NULL, rehearsed REAL NOT NULL DEFAULT 0,
 UNIQUE(group_id,question,answer_id));
CREATE INDEX IF NOT EXISTS examples_group ON examples(group_id,status);
CREATE TABLE IF NOT EXISTS models(
 group_id TEXT PRIMARY KEY REFERENCES groups(id), state TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS counterexamples(
 group_id TEXT NOT NULL REFERENCES groups(id), question TEXT NOT NULL,
 answer_id INTEGER NOT NULL REFERENCES answers(id), PRIMARY KEY(group_id,question,answer_id));
CREATE TABLE IF NOT EXISTS facts(
 id INTEGER PRIMARY KEY AUTOINCREMENT, group_id TEXT NOT NULL REFERENCES groups(id),
 user_id TEXT NOT NULL REFERENCES users(id), category TEXT NOT NULL, value TEXT NOT NULL,
 source_id INTEGER NOT NULL REFERENCES messages(id), active INTEGER NOT NULL DEFAULT 1,
 created REAL NOT NULL);
CREATE INDEX IF NOT EXISTS facts_lookup ON facts(group_id,user_id,active);
CREATE TABLE IF NOT EXISTS feedback(
 message_id INTEGER NOT NULL REFERENCES messages(id), user_id TEXT NOT NULL REFERENCES users(id),
 value INTEGER NOT NULL, style TEXT, created REAL NOT NULL, PRIMARY KEY(message_id,user_id));
CREATE TABLE IF NOT EXISTS traits(
 group_id TEXT NOT NULL REFERENCES groups(id), trait TEXT NOT NULL, evidence INTEGER NOT NULL,
 PRIMARY KEY(group_id,trait));
CREATE TABLE IF NOT EXISTS events(
 id INTEGER PRIMARY KEY AUTOINCREMENT, group_id TEXT NOT NULL REFERENCES groups(id),
 kind TEXT NOT NULL, detail TEXT NOT NULL, created REAL NOT NULL);
CREATE INDEX IF NOT EXISTS events_group ON events(group_id,id);
CREATE TABLE IF NOT EXISTS bot_state(
 group_id TEXT PRIMARY KEY REFERENCES groups(id),
 mood TEXT NOT NULL DEFAULT '平静', energy REAL NOT NULL DEFAULT 0.72,
 curiosity REAL NOT NULL DEFAULT 0.55, loneliness REAL NOT NULL DEFAULT 0.0,
 last_spoke REAL NOT NULL DEFAULT 0, last_seen REAL NOT NULL DEFAULT 0,
 turn_count INTEGER NOT NULL DEFAULT 0, inner_note TEXT NOT NULL DEFAULT '',
 updated REAL NOT NULL);
"""


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(SCHEMA)
            db.execute("INSERT OR IGNORE INTO meta VALUES('schema_version','1')")

    @contextlib.contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def backup(self, destination: str | Path):
        target = Path(destination).resolve()
        if target == self.path or target.exists():
            raise ValueError("备份目标必须是尚不存在的新文件")
        target.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as source:
            dest = sqlite3.connect(target)
            try:
                source.backup(dest)
            finally:
                dest.close()
        return target
