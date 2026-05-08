from __future__ import annotations

from pathlib import Path

import pytest

from sa_candidate_finder.config import Config
from sa_candidate_finder.models import CandidateMeta, CandidateResult, HardConstraint
from sa_candidate_finder.output import write_results


@pytest.fixture
def cfg(tmp_path):
    return Config(
        openai_api_key="k",
        manatal_base_url="https://secret-manatal-url",
        results_dir=str(tmp_path / "results"),
    )


def _make_result(rank: int) -> CandidateResult:
    c = CandidateMeta(
        id=str(rank), name=f"Candidate {rank}", current_position="Engineer",
        current_company="Acme", latest_degree="BS", latest_university="MIT",
        location="NYC", tags=[], industries=[], description="",
        resume_url=f"https://example.com/resume{rank}.pdf",
    )
    return CandidateResult(
        rank=rank, candidate=c, fit_score=9.0 - rank,
        strengths=["Strong Python"], risks=["Limited domain exp"],
        rationale="Good overall fit.", embedding_score=0.85,
    )


def test_write_results_creates_file(cfg, tmp_path):
    jd_path = tmp_path / "senior_engineer.txt"
    results = [_make_result(1), _make_result(2)]
    constraints = [HardConstraint(field="location", value="NYC", confidence=0.9, source_quote="NYC")]

    output_path = write_results(results, jd_path, "run-001", constraints, cfg)

    assert output_path.exists()
    assert output_path.suffix == ".md"
    assert "senior_engineer" in output_path.name


def test_write_results_content(cfg, tmp_path):
    jd_path = tmp_path / "jd.txt"
    results = [_make_result(1)]
    constraints = [HardConstraint(field="current_position", value="Engineer", confidence=0.9, source_quote="x")]

    output_path = write_results(results, jd_path, "run-abc", constraints, cfg)
    content = output_path.read_text()

    assert "# Candidate Search Results" in content
    assert "run-abc" in content
    assert "Candidate 1" in content
    assert "8.0/10" in content
    assert "https://example.com/resume1.pdf" in content
    assert "Strong Python" in content
    assert "Limited domain exp" in content
    assert "Good overall fit." in content
