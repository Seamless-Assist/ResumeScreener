from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class HardConstraint:
    field: str
    value: str
    confidence: float  # 0.0 – 1.0
    source_quote: str  # verbatim phrase from JD


@dataclass
class SoftCriteria:
    description: str


@dataclass
class CandidateMeta:
    id: str
    name: str
    current_position: str
    current_company: str
    latest_degree: str
    latest_university: str
    location: str
    tags: list[str]
    industries: list[str]
    description: str
    resume_url: str = ""
    resume_text: str = ""
    manatal_stage: str = ""


@dataclass
class CandidateResult:
    rank: int
    candidate: CandidateMeta
    fit_score: float          # 0.0 – 10.0
    strengths: list[str]
    risks: list[str]
    rationale: str
    embedding_score: float = 0.0
    tier: str = ""            # A / B / C / D (D = hard-filter disqualified)
    disqualified: bool = False
    disqualify_reason: str = ""


@dataclass
class RunTelemetry:
    run_id: str
    jd_hash: str
    started_at: str
    finished_at: str = ""
    hard_constraints: list[dict[str, Any]] = field(default_factory=list)
    soft_criteria: list[str] = field(default_factory=list)
    pool_count: int = 0
    post_filter_count: int = 0
    post_ranking_count: int = 0
    llm_evaluated_count: int = 0
    final_returned_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_stale_refreshes: int = 0
    embedding_model: str = ""
    llm_model: str = ""
    embedding_tokens: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    embedding_cost_usd: float = 0.0
    llm_input_cost_usd: float = 0.0
    llm_output_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    errors: list[dict[str, Any]] = field(default_factory=list)
    status: str = "in_progress"  # in_progress | success | partial | failed
