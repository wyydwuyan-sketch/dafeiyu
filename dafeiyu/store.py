"""SQLite 持久化：聊天记录、情景记忆、待跟进事项、运行状态。"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT NOT NULL,            -- user / assistant
    content TEXT NOT NULL,
    ts REAL NOT NULL,
    kind TEXT NOT NULL DEFAULT 'chat'  -- chat / proactive
);
CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    importance INTEGER NOT NULL DEFAULT 2,  -- 1~5
    created_at REAL NOT NULL,
    last_used REAL,
    active INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS followups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    due_at REAL NOT NULL,
    topic TEXT NOT NULL,
    done INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


@dataclass
class Message:
    id: int
    role: str
    content: str
    ts: float
    kind: str


@dataclass
class Note:
    id: int
    content: str
    importance: int
    created_at: float
    last_used: float | None


@dataclass
class Followup:
    id: int
    due_at: float
    topic: str


class Store:
    def __init__(self, path: Path | str):
        self.db = sqlite3.connect(str(path))
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.db.commit()

    # ---------- kv ----------
    def get(self, key: str, default: Any = None) -> Any:
        row = self.db.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def set(self, key: str, value: Any) -> None:
        self.db.execute("INSERT INTO kv(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                        (key, json.dumps(value, ensure_ascii=False)))
        self.db.commit()

    # ---------- messages ----------
    def add_message(self, role: str, content: str, kind: str = "chat", ts: float | None = None) -> int:
        cur = self.db.execute("INSERT INTO messages(role,content,ts,kind) VALUES(?,?,?,?)",
                              (role, content, ts or time.time(), kind))
        self.db.commit()
        return cur.lastrowid

    def messages_after(self, after_id: int, limit: int | None = None) -> list[Message]:
        sql = "SELECT * FROM messages WHERE id>? ORDER BY id"
        args: tuple = (after_id,)
        if limit:
            sql += " LIMIT ?"
            args += (limit,)
        return [Message(r["id"], r["role"], r["content"], r["ts"], r["kind"]) for r in self.db.execute(sql, args)]

    def update_message(self, msg_id: int, content: str) -> None:
        self.db.execute("UPDATE messages SET content=? WHERE id=?", (content, msg_id))
        self.db.commit()

    def recent_by_kind(self, kind: str, limit: int) -> list[Message]:
        rows = self.db.execute("SELECT * FROM messages WHERE kind=? ORDER BY id DESC LIMIT ?", (kind, limit)).fetchall()
        return [Message(r["id"], r["role"], r["content"], r["ts"], r["kind"]) for r in reversed(rows)]

    def last_message(self, role: str | None = None) -> Message | None:
        if role:
            r = self.db.execute("SELECT * FROM messages WHERE role=? ORDER BY id DESC LIMIT 1", (role,)).fetchone()
        else:
            r = self.db.execute("SELECT * FROM messages ORDER BY id DESC LIMIT 1").fetchone()
        return Message(r["id"], r["role"], r["content"], r["ts"], r["kind"]) if r else None

    # ---------- notes ----------
    def add_note(self, content: str, importance: int = 2) -> int:
        cur = self.db.execute("INSERT INTO notes(content,importance,created_at) VALUES(?,?,?)",
                              (content.strip(), max(1, min(5, int(importance))), time.time()))
        self.db.commit()
        return cur.lastrowid

    def active_notes(self) -> list[Note]:
        return [Note(r["id"], r["content"], r["importance"], r["created_at"], r["last_used"])
                for r in self.db.execute("SELECT * FROM notes WHERE active=1 ORDER BY id")]

    def notes_since(self, ts: float) -> list[Note]:
        return [Note(r["id"], r["content"], r["importance"], r["created_at"], r["last_used"])
                for r in self.db.execute("SELECT * FROM notes WHERE active=1 AND created_at>=? ORDER BY id", (ts,))]

    def touch_notes(self, ids: list[int]) -> None:
        if ids:
            now = time.time()
            self.db.executemany("UPDATE notes SET last_used=? WHERE id=?", [(now, i) for i in ids])
            self.db.commit()

    def deactivate_notes(self, ids: list[int]) -> None:
        if ids:
            self.db.executemany("UPDATE notes SET active=0 WHERE id=?", [(i,) for i in ids])
            self.db.commit()

    # ---------- followups ----------
    def add_followup(self, due_at: float, topic: str) -> int:
        cur = self.db.execute("INSERT INTO followups(due_at,topic,created_at) VALUES(?,?,?)",
                              (due_at, topic.strip(), time.time()))
        self.db.commit()
        return cur.lastrowid

    def pending_followups(self, until: float) -> list[Followup]:
        return [Followup(r["id"], r["due_at"], r["topic"]) for r in self.db.execute(
            "SELECT * FROM followups WHERE done=0 AND due_at<=? ORDER BY due_at", (until,))]

    def finish_followups(self, ids: list[int]) -> None:
        if ids:
            self.db.executemany("UPDATE followups SET done=1 WHERE id=?", [(i,) for i in ids])
            self.db.commit()
