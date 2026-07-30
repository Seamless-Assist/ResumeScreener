from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from sa_candidate_finder.config import Config
from sa_candidate_finder.models import RunTelemetry
from sa_candidate_finder.pipeline.extractor import extract_constraints


@pytest.fixture
def cfg():
    return Config(
        openai_api_key="test",
        manatal_base_url="https://secret-manatal-url",
        max_hard_constraints=3,
        universal_dealbreakers=[],
    )


@pytest.fixture
def telemetry():
    return RunTelemetry(run_id="test", jd_hash="abc", started_at="2026-01-01T00:00:00Z")


@pytest.fixture(autouse=True)
def isolated_extraction_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "sa_candidate_finder.pipeline.extractor._keyword_cache_path",
        lambda jd_text: str(tmp_path / "extraction.json"),
    )


def _mock_openai_response(content: dict):
    response = MagicMock()
    response.choices[0].message.content = json.dumps(content)
    return response


def test_extract_constraints_keyword_cap_is_enforced(cfg, telemetry):
    llm_output = {
        "keywords": ["Python", "SQL", "AWS", "Terraform"],
        "dealbreakers": [],
    }

    with patch("sa_candidate_finder.pipeline.extractor.OpenAI") as mock_openai:
        mock_openai.return_value.chat.completions.create.return_value = _mock_openai_response(llm_output)
        extraction = extract_constraints(
            "Python SQL AWS Terraform engineer. Python experience required.",
            cfg,
            telemetry,
        )

    assert len(extraction["keywords"]) == cfg.max_hard_constraints
    assert telemetry.soft_criteria == extraction["keywords"]


def test_extract_constraints_drops_dealbreaker_without_jd_evidence(cfg, telemetry):
    llm_output = {
        "keywords": ["Python"],
        "dealbreakers": [
            {
                "id": "invented",
                "rule": "Candidate must own a car",
                "source_quote": "",
                "keyword_signals": [],
            },
            {
                "id": "required_license",
                "rule": "Candidate must hold an active license",
                "source_quote": "Active state license required",
                "keyword_signals": [],
            },
        ],
    }

    with patch("sa_candidate_finder.pipeline.extractor.OpenAI") as mock_openai:
        mock_openai.return_value.chat.completions.create.return_value = _mock_openai_response(llm_output)
        extraction = extract_constraints(
            "Python role. Active state license required.",
            cfg,
            telemetry,
        )

    assert [item["id"] for item in extraction["dealbreakers"]] == ["required_license"]


def test_extract_constraints_returns_current_payload_shape(cfg, telemetry):
    llm_output = {
        "keywords": ["SQL", "Python"],
        "relaxation_plan": [{"action": "drop", "target": "SQL"}],
        "dealbreakers": [],
    }

    with patch("sa_candidate_finder.pipeline.extractor.OpenAI") as mock_openai:
        mock_openai.return_value.chat.completions.create.return_value = _mock_openai_response(llm_output)
        extraction = extract_constraints("Python and SQL developer", cfg, telemetry)

    assert set(extraction) == {"keywords", "relaxation_plan", "dealbreakers"}
    assert set(extraction["keywords"]) == {"python", "sql"}
    assert extraction["relaxation_plan"] == [{"action": "drop", "target": "SQL"}]
