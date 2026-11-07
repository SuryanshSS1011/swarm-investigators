"""Disk cache of successful model calls, in SQLite at data/cache/llm.sqlite (or SWARM_CACHE_PATH).

The key is sha256 over model, system, user, max_tokens, temperature and replicate, so a change to any of them
makes a new call. Route settings that are not request fields, such as Claude's effort and thinking floor or the
open-weight reasoning switches in swarm/llm.py, are not part of the key. If you change one of those, point
SWARM_CACHE_PATH at a new file. Errors are never cached, and each row keeps the cost of the call that made it.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

from swarm import REPO_ROOT

DEFAULT_PATH = REPO_ROOT / "data" / "cache" / "llm.sqlite"


def cache_key(model: str, system: Optional[str], user: str, max_tokens: int, temperature=None, replicate: int = 0) -> str:
    """sha256 hex digest of the request fields. Numbers are normalised, so temperature 0 and 0.0 match."""
    payload = {
        "model": model,
        "system": system or "",
        "user": user,
        "max_tokens": int(max_tokens),
        "temperature": None if temperature is None else float(temperature),
        "replicate": int(replicate),
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class Cache:
    """Thread-safe, and safe for several processes on one machine because it uses a WAL journal."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(self.path), timeout=60, isolation_level=None, check_same_thread=False)
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS calls (key TEXT PRIMARY KEY, model TEXT NOT NULL, result TEXT NOT NULL,"
                " cost_usd REAL NOT NULL, created_at REAL NOT NULL)"
            )

    def get(self, key: str) -> Optional[dict]:
        with self._lock:
            row = self._db.execute("SELECT result FROM calls WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, model: str, result: dict, cost_usd: float) -> None:
        """Store a successful result. The first answer stored under a key is the one that is kept."""
        blob = json.dumps(result, ensure_ascii=False)
        with self._lock:
            self._db.execute(
                "INSERT OR IGNORE INTO calls VALUES (?, ?, ?, ?, ?)", (key, model, blob, float(cost_usd), time.time())
            )

    def __len__(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM calls").fetchone()[0]

    def close(self) -> None:
        with self._lock:
            self._db.close()


_default: Optional[Cache] = None
_default_lock = threading.Lock()


def default_cache() -> Cache:
    """The process-wide cache at SWARM_CACHE_PATH, or data/cache/llm.sqlite in the repo."""
    global _default
    path = Path(os.environ.get("SWARM_CACHE_PATH") or DEFAULT_PATH)
    with _default_lock:
        if _default is None or _default.path != path:
            _default = Cache(path)
        return _default


def reset_default() -> None:
    global _default
    with _default_lock:
        if _default is not None:
            _default.close()
        _default = None
