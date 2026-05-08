from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from openai import OpenAI

from sa_candidate_finder.config import Config
from sa_candidate_finder.models import CandidateMeta, RunTelemetry


def rank_candidates(
    jd_text: str,
    candidates: list[CandidateMeta],
    cfg: Config,
    telemetry: RunTelemetry,
) -> list[CandidateMeta]:
    """Embed JD and candidates, rank by cosine similarity, return sorted candidates."""
    client = OpenAI(api_key=cfg.openai_api_key, timeout=cfg.embedding_timeout_seconds)
    db_path = Path(cfg.embedding_cache_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = _init_db(db_path)

    ttl = timedelta(days=cfg.embedding_cache_ttl_days)

    # Embed JD
    jd_vec = _embed_text(client, cfg.embedding_model, jd_text)
    telemetry.embedding_tokens += len(jd_text.split())  # approximation; real count from API

    # Embed candidates (cache-aware)
    vecs: list[np.ndarray] = []
    for c in candidates:
        text = _candidate_text(c)
        cached = _get_cache(conn, c.id, ttl)
        if cached is not None:
            telemetry.cache_hits += 1
            vecs.append(cached)
        else:
            vec = _embed_text(client, cfg.embedding_model, text)
            _set_cache(conn, c.id, vec)
            if _is_stale(conn, c.id, ttl):
                telemetry.cache_stale_refreshes += 1
            else:
                telemetry.cache_misses += 1
            vecs.append(vec)

    conn.close()

    # Compute cosine similarity and rank
    jd_norm = jd_vec / (np.linalg.norm(jd_vec) + 1e-10)
    scores = []
    for vec in vecs:
        norm = vec / (np.linalg.norm(vec) + 1e-10)
        scores.append(float(np.dot(jd_norm, norm)))

    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    telemetry.post_ranking_count = len(ranked)

    # Attach embedding score to candidates via a side dict (used later by evaluator)
    for c, score in ranked:
        c.__dict__["_embedding_score"] = score  # transient field

    return [c for c, _ in ranked]


def _candidate_text(c: CandidateMeta) -> str:
    parts = [
        c.current_position,
        c.current_company,
        c.latest_degree,
        c.latest_university,
        c.location,
        " ".join(c.tags),
        " ".join(c.industries),
        c.description,
    ]
    return " | ".join(p for p in parts if p)


def _embed_text(client: OpenAI, model: str, text: str) -> np.ndarray:
    resp = client.embeddings.create(model=model, input=text)
    return np.array(resp.data[0].embedding, dtype=np.float32)


def _init_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS embeddings (
            candidate_id TEXT PRIMARY KEY,
            vector BLOB NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def _get_cache(conn: sqlite3.Connection, candidate_id: str, ttl: timedelta) -> np.ndarray | None:
    row = conn.execute(
        "SELECT vector, created_at FROM embeddings WHERE candidate_id = ?",
        (candidate_id,),
    ).fetchone()
    if row is None:
        return None
    created = datetime.fromisoformat(row[1])
    if datetime.now(timezone.utc) - created > ttl:
        return None  # stale
    return np.frombuffer(row[0], dtype=np.float32).copy()


def _is_stale(conn: sqlite3.Connection, candidate_id: str, ttl: timedelta) -> bool:
    row = conn.execute(
        "SELECT created_at FROM embeddings WHERE candidate_id = ?",
        (candidate_id,),
    ).fetchone()
    if row is None:
        return False
    created = datetime.fromisoformat(row[0])
    return datetime.now(timezone.utc) - created > ttl


def _set_cache(conn: sqlite3.Connection, candidate_id: str, vec: np.ndarray) -> None:
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT OR REPLACE INTO embeddings (candidate_id, vector, created_at) VALUES (?, ?, ?)",
        (candidate_id, vec.tobytes(), now),
    )
    conn.commit()
