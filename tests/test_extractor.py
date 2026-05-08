from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from sa_candidate_finder.config import Config
from sa_candidate_finder.models import RunTelemetry
from sa_candidate_finder.pipeline.extractor import extract_constraints, _MANATAL_FILTERABLE_FIELDS


@pytest.fixture
def cfg():
    return Config(openai_api_key="test", manatal_base_url="https://secret-manatal-url", max_hard_constraints=3)


@pytest.fixture
def telemetry():
    return RunTelemetry(run_id="test", jd_hash="abc", started_at="2026-01-01T00:00:00Z")


def _mock_openai_response(content: dict):
    response = MagicMock()
    response.choices[0].message.content = json.dumps(content)
    return response


def test_extract_constraints_top_n_enforced(cfg, telemetry, monkeypatch):
    """LLM returning more than max_hard_constraints should be capped."""
    llm_output = {
        "hard_constraints": [
            {"field": "current_position", "value": "Engineer"},
            {"field": "location", "value": "NYC"},
            {"field": "latest_degree", "value": "BS"},
            {"field": "tags", "value": "python"},
        ],
    }

    with patch("sa_candidate_finder.pipeline.extractor.OpenAI") as mock_openai, \
            patch("sa_candidate_finder.pipeline.extractor._confirm_with_user", side_effect=lambda c, k: c):
        mock_openai.return_value.chat.completions.create.return_value = _mock_openai_response(llm_output)
        hard = extract_constraints("some JD text", cfg, telemetry)

    assert len(hard) == 3


def test_extract_constraints_unknown_fields_filtered(cfg, telemetry, monkeypatch):
    """Constraints with non-filterable fields must be dropped."""
    llm_output = {
        "hard_constraints": [
            {"field": "current_position", "value": "Engineer"},
            {"field": "salary", "value": "100k"},  # invalid
        ],
    }

    with patch("sa_candidate_finder.pipeline.extractor.OpenAI") as mock_openai, \
            patch("sa_candidate_finder.pipeline.extractor._confirm_with_user", side_effect=lambda c, k: c):
        mock_openai.return_value.chat.completions.create.return_value = _mock_openai_response(llm_output)
        hard = extract_constraints("some JD text", cfg, telemetry)

    assert all(c.field in _MANATAL_FILTERABLE_FIELDS for c in hard)
    assert len(hard) == 1


def test_extract_constraints_takes_first_n(cfg, telemetry):
    """Constraints should be capped at max_hard_constraints."""
    llm_output = {
        "hard_constraints": [
            {"field": "location", "value": "NYC"},
            {"field": "current_position", "value": "Eng"},
        ],
    }

    with patch("sa_candidate_finder.pipeline.extractor.OpenAI") as mock_openai, \
            patch("sa_candidate_finder.pipeline.extractor._confirm_with_user", side_effect=lambda c, k: c):
        mock_openai.return_value.chat.completions.create.return_value = _mock_openai_response(llm_output)
        hard = extract_constraints("JD", cfg, telemetry)

    assert len(hard) == 2
    assert hard[0].field == "location"
    assert hard[1].field == "current_position"
