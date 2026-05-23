"""Session history persistence -- standalone reimplementation of nanobot's session layer.

Architectural concept: each conversation is a Session keyed by "channel:chat_id";
SessionManager loads/saves them as JSONL on disk and keeps an in-memory cache.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class Session:
    key: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            **kwargs,
        })
        self.updated_at = datetime.now()

    def get_history(self) -> list[dict[str, Any]]:
        return list(self.messages)


class SessionManager:
    def __init__(self, workspace: Path) -> None:
        self.sessions_dir = workspace / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, Session] = {}

    def _path(self, key: str) -> Path:
        safe = key.replace(":", "_").replace("/", "_").replace("\\", "_")
        return self.sessions_dir / f"{safe}.jsonl"

    def get_or_create(self, key: str) -> Session:
        if key in self._cache:
            return self._cache[key]
        session = self._load(key) or Session(key=key)
        self._cache[key] = session
        return session

    def _load(self, key: str) -> Session | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            messages: list[dict] = []
            created_at = updated_at = None
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    if data.get("_type") == "metadata":
                        created_at = datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None
                        updated_at = datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None
                    else:
                        messages.append(data)
            return Session(
                key=key,
                messages=messages,
                created_at=created_at or datetime.now(),
                updated_at=updated_at or datetime.now(),
            )
        except Exception as e:
            print(f"[sessions] failed to load {key}: {e}")
            return None

    def save(self, session: Session) -> None:
        path = self._path(session.key)
        tmp = path.with_suffix(".jsonl.tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(json.dumps({
                    "_type": "metadata",
                    "key": session.key,
                    "created_at": session.created_at.isoformat(),
                    "updated_at": session.updated_at.isoformat(),
                }, ensure_ascii=False) + "\n")
                for msg in session.messages:
                    f.write(json.dumps(msg, ensure_ascii=False) + "\n")
            os.replace(tmp, path)
            self._cache[session.key] = session
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
