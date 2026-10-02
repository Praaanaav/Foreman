"""Long-term traveler memory.

Per-traveler store of:
- profile / preferences (structured)
- semantic-style memories (free text + tags + importance)
- completed trip records

Backed by JSON files on disk (one directory per traveler). The interface
is deliberately small so a Postgres + ChromaDB implementation can replace
the storage without touching callers.
"""
from __future__ import annotations

import json
import re
import tempfile
import time
import uuid
from pathlib import Path

from orchestrator import config

_TOKEN_RE = re.compile(r"[a-z0-9ऀ-ॿ]+")  # includes Devanagari


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _atomic_write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with open(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        tmp_path = Path(tmp)
        tmp_path.replace(path)
    finally:
        if Path(tmp).exists():
            Path(tmp).unlink(missing_ok=True)


def _read_json(path: Path, default):
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


class LongTermMemory:
    def __init__(self, data_dir: str | Path | None = None) -> None:
        self.dir = Path(data_dir or config.DATA_DIR)

    # -- paths ---------------------------------------------------------------

    def _traveler_dir(self, traveler_id: str) -> Path:
        safe = re.sub(r"[^a-zA-Z0-9_-]", "_", traveler_id) or "guest"
        return self.dir / "travelers" / safe

    def _profile_path(self, traveler_id: str) -> Path:
        return self._traveler_dir(traveler_id) / "profile.json"

    def _memories_path(self, traveler_id: str) -> Path:
        return self._traveler_dir(traveler_id) / "memories.json"

    def _trips_path(self, traveler_id: str) -> Path:
        return self._traveler_dir(traveler_id) / "trips.json"

    # -- profile ---------------------------------------------------------------

    def get_profile(self, traveler_id: str) -> dict:
        return _read_json(self._profile_path(traveler_id), {})

    def save_profile(self, traveler_id: str, profile: dict) -> None:
        _atomic_write(self._profile_path(traveler_id), profile)

    # -- memories ---------------------------------------------------------------

    def add_memory(
        self,
        traveler_id: str,
        text: str,
        tags: list[str] | None = None,
        importance: float = 0.5,
        source: str = "",
    ) -> dict:
        memories = _read_json(self._memories_path(traveler_id), [])
        memory = {
            "id": uuid.uuid4().hex[:10],
            "text": text,
            "tags": tags or [],
            "importance": min(1.0, max(0.0, importance)),
            "source": source,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        # cheap duplicate merge: near-identical text raises importance
        for existing in memories:
            if _similar(existing["text"], text):
                existing["importance"] = min(1.0, existing["importance"] + 0.2)
                existing["created_at"] = memory["created_at"]
                _atomic_write(self._memories_path(traveler_id), memories)
                return existing
        memories.append(memory)
        _atomic_write(self._memories_path(traveler_id), memories)
        return memory

    def get_memories(self, traveler_id: str, query: str, k: int = 5) -> list[dict]:
        """Relevance-ranked memories for a query (keyword overlap + recency + importance).

        A ChromaDB embedding-based retrieval would replace this ranking in
        phase 2; the caller interface stays the same.
        """
        memories = _read_json(self._memories_path(traveler_id), [])
        if not memories:
            return []
        q = _tokens(query)
        now = time.time()
        scored: list[tuple[float, dict]] = []
        for m in memories:
            overlap = len(q & _tokens(m["text"] + " " + " ".join(m["tags"])))
            overlap_score = overlap / max(len(q), 1)
            age_days = max(0.0, (now - time.mktime(
                time.strptime(m["created_at"], "%Y-%m-%d %H:%M:%S")
            )) / 86400)
            recency = 1.0 / (1.0 + age_days / 30)
            score = 2.0 * overlap_score + m["importance"] + 0.5 * recency
            scored.append((score, m))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [m for _, m in scored[:k]]

    def list_memories(self, traveler_id: str) -> list[dict]:
        return _read_json(self._memories_path(traveler_id), [])

    def delete_memory(self, traveler_id: str, memory_id: str) -> bool:
        path = self._memories_path(traveler_id)
        memories = _read_json(path, [])
        kept = [m for m in memories if m["id"] != memory_id]
        if len(kept) == len(memories):
            return False
        _atomic_write(path, kept)
        return True

    # -- trip history ---------------------------------------------------------------

    def save_trip(self, traveler_id: str, trip_id: str, record: dict) -> None:
        trips = _read_json(self._trips_path(traveler_id), {})
        trips[trip_id] = record
        _atomic_write(self._trips_path(traveler_id), trips)

    def get_trips(self, traveler_id: str) -> dict:
        return _read_json(self._trips_path(traveler_id), {})

    def previous_trips_summary(self, traveler_id: str, k: int = 3) -> str:
        """Compact text summary of the k most recent trips, for prompts."""
        trips = self.get_trips(traveler_id)
        if not trips:
            return "(no previous trips on record)"
        recent = sorted(
            trips.items(), key=lambda kv: kv[1].get("created_at", ""), reverse=True
        )[:k]
        lines = []
        for trip_id, rec in recent:
            lines.append(
                f"- {rec.get('title', trip_id)}: {rec.get('summary', '')[:300]}"
            )
        return "\n".join(lines)


def _similar(a: str, b: str) -> bool:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return False
    return len(ta & tb) / min(len(ta), len(tb)) > 0.85
