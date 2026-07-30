from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sa_candidate_finder.config import Config
from sa_candidate_finder.models import CandidateMeta, RunTelemetry, SoftCriteria
from sa_candidate_finder.pipeline.evaluator import evaluate_candidates


@pytest.fixture
def cfg():
    return Config(openai_api_key="test", manatal_base_url="https://secret-manatal-url")


@pytest.fixture
def telemetry():
    return RunTelemetry(run_id="t", jd_hash="h", started_at="2026-01-01T00:00:00Z")


def _make_candidate(cid: str, name: str) -> CandidateMeta:
    return CandidateMeta(
        id=cid, name=name, current_position="Engineer", current_company="Acme",
        latest_degree="BS", latest_university="MIT", location="NYC",
        tags=["python"], industries=["tech"], description=".", resume_text="Full resume text.",
    )


def test_evaluate_returns_ranked_results(cfg, telemetry):
    candidates = [_make_candidate("1", "Alice"), _make_candidate("2", "Bob")]
    llm_output = {
        "candidates": [
            {"candidate_id": "2", "rank": 1, "fit_score": 9.0, "strengths": ["a"], "risks": [], "rationale": "Best"},
            {"candidate_id": "1", "rank": 2, "fit_score": 7.5, "strengths": ["b"], "risks": ["c"], "rationale": "Good"},
        ]
    }

    with patch("sa_candidate_finder.pipeline.evaluator.OpenAI") as mock_openai:
        resp = MagicMock()
        resp.choices[0].message.content = json.dumps(llm_output)
        resp.usage.prompt_tokens = 100
        resp.usage.completion_tokens = 50
        mock_openai.return_value.chat.completions.create.return_value = resp

        results = evaluate_candidates("JD text", candidates, [], cfg, telemetry)

    assert results[0].rank == 1
    assert results[0].candidate.name == "Bob"
    assert results[0].fit_score == 9.0
    assert results[1].rank == 2


def test_evaluate_tracks_token_costs(cfg, telemetry):
    candidates = [_make_candidate("1", "Alice")]
    llm_output = {
        "candidates": [
            {"candidate_id": "1", "rank": 1, "fit_score": 8.0, "strengths": [], "risks": [], "rationale": "ok"},
        ]
    }

    with patch("sa_candidate_finder.pipeline.evaluator.OpenAI") as mock_openai:
        resp = MagicMock()
        resp.choices[0].message.content = json.dumps(llm_output)
        resp.usage.prompt_tokens = 500
        resp.usage.completion_tokens = 200
        mock_openai.return_value.chat.completions.create.return_value = resp

        evaluate_candidates("JD", candidates, [], cfg, telemetry)

    assert telemetry.llm_input_tokens == 500
    assert telemetry.llm_output_tokens == 200
    assert telemetry.total_cost_usd > 0


def test_evaluate_does_not_double_count_explicit_availability(cfg, telemetry):
    candidates = [_make_candidate("1", "Alice"), _make_candidate("2", "Bob")]
    llm_output = {
        "candidates": [
            {
                "candidate_id": "1",
                "rank": 1,
                "fit_score": 8.6,
                "strengths": [],
                "risks": [],
                "rationale": "Strong profile",
                "availability_signal": "unknown",
                "availability_evidence": "",
            },
            {
                "candidate_id": "2",
                "rank": 2,
                "fit_score": 8.3,
                "strengths": [],
                "risks": [],
                "rationale": "Slightly lower fit",
                "availability_signal": "available_now",
                "availability_evidence": "Available immediately",
            },
        ]
    }

    with patch("sa_candidate_finder.pipeline.evaluator.OpenAI") as mock_openai:
        resp = MagicMock()
        resp.choices[0].message.content = json.dumps(llm_output)
        resp.usage.prompt_tokens = 100
        resp.usage.completion_tokens = 50
        mock_openai.return_value.chat.completions.create.return_value = resp

        results = evaluate_candidates("JD text", candidates, [], cfg, telemetry)

    # Availability is already included in the Location & Availability rubric
    # score, so post-processing must not add another bonus or override fit order.
    assert results[0].candidate.id == "1"
    assert results[0].rank == 1
    assert results[1].candidate.id == "2"
    assert results[1].__dict__["availability_signal"] == "available_now"
    assert results[1].__dict__["availability_bonus"] == 0.0
