from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sa_candidate_finder.config import Config
from sa_candidate_finder.models import CandidateMeta, RunTelemetry
from sa_candidate_finder.pipeline.embedder import rank_candidates, _init_db, _get_cache, _set_cache


@pytest.fixture
def cfg(tmp_path):
    return Config(
        openai_api_key="test",
        manatal_base_url="https://secret-manatal-url",
        embedding_cache_path=str(tmp_path / "embeddings.db"),
        embedding_cache_ttl_days=7,
    )


@pytest.fixture
def telemetry():
    return RunTelemetry(run_id="t", jd_hash="h", started_at="2026-01-01T00:00:00Z")


def _make_candidate(cid: str) -> CandidateMeta:
    return CandidateMeta(
        id=cid, name=f"Candidate {cid}", current_position="Engineer",
        current_company="Acme", latest_degree="BS", latest_university="MIT",
        location="NYC", tags=["python"], industries=["tech"], description="data engineer",
    )


def _mock_embed(vec: list[float]):
    resp = MagicMock()
    resp.data[0].embedding = vec
    return resp


def test_rank_candidates_returns_all(cfg, telemetry):
    candidates = [_make_candidate("1"), _make_candidate("2")]
    jd_vec = [1.0, 0.0]
    c1_vec = [1.0, 0.0]   # perfect match
    c2_vec = [0.0, 1.0]   # orthogonal

    call_count = 0
    def fake_embed(model, input):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _mock_embed(jd_vec)
        elif call_count == 2:
            return _mock_embed(c1_vec)
        return _mock_embed(c2_vec)

    with patch("sa_candidate_finder.pipeline.embedder.OpenAI") as mock_openai:
        mock_openai.return_value.embeddings.create.side_effect = fake_embed
        ranked = rank_candidates("JD text", candidates, cfg, telemetry)

    assert len(ranked) == 2
    assert ranked[0].id == "1"   # highest similarity to JD


def test_embedding_cache_hit(cfg, telemetry):
    """Second ranking for same candidate should use cached embedding."""
    candidates = [_make_candidate("cached-1")]
    vec = np.array([1.0, 0.0], dtype=np.float32)

    # Pre-populate cache
    conn = _init_db(Path(cfg.embedding_cache_path))
    _set_cache(conn, "cached-1", vec)
    conn.close()

    with patch("sa_candidate_finder.pipeline.embedder.OpenAI") as mock_openai:
        # Only one embed call expected (for JD), not for cached candidate
        mock_openai.return_value.embeddings.create.return_value = _mock_embed([1.0, 0.0])
        rank_candidates("JD text", candidates, cfg, telemetry)

    assert telemetry.cache_hits == 1
    assert mock_openai.return_value.embeddings.create.call_count == 1
