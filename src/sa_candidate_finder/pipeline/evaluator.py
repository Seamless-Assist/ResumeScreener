from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from sa_candidate_finder.config import Config
from sa_candidate_finder.models import (
    CandidateMeta,
    CandidateResult,
    RunTelemetry,
    SoftCriteria,
)

_SYSTEM_PROMPT = """\
You are an expert recruiter. Evaluate each candidate against the job description.

Score each candidate across five dimensions. Sum the points to get a total out of 100,
then divide by 10 to produce fit_score (0.0–10.0).

Scoring rubric:

| Dimension              | Max Points | What to Evaluate |
|------------------------|------------|------------------|
| Experience Match       | 30         | Years of experience in role type, seniority level, industry relevance, complexity of past responsibilities |
| Skills Match           | 25         | Hard skills from required list present in resume, depth of each skill, evidence of use vs. just listing |
| Location & Availability| 20         | Country match to preferred locations, timezone compatibility (PST hours if required), availability date |
| English & AI Fluency   | 15         | Quality of written English in resume/cover letter, specific AI tools mentioned, evidence of tool use (not just listing) |
| Salary Fit             | 10         | Expected salary within client budget, stated or inferred from location + role level |

fit_score = (experience_score + skills_score + location_score + english_ai_score + salary_score) / 10

Availability rules:
- Only mark availability_signal when there is explicit evidence in candidate text.
- If no explicit evidence exists, use "unknown".

For each candidate return:
    - candidate_id (string: must exactly match the Candidate ID provided in the prompt)
  - rank (integer, 1 = best; disqualified candidates last)
  - fit_score (float 0.0-10.0, derived from rubric; 0.0 if disqualified)
  - tier (string: "A" for >=8.5, "B" for >=7.0, "C" for >=5.0, "D" if disqualified by a hard filter)
  - disqualified (boolean: true only if a hard filter rule was violated)
  - experience_score (float 0-30)
  - skills_score (float 0-25)
  - location_score (float 0-20)
  - english_ai_score (float 0-15)
  - salary_score (float 0-10)
  - strengths (list of strings)
  - risks (list of strings)
  - rationale (REQUIRED, never empty; one concise paragraph; if disqualified, name the violated rule and quote the exact candidate evidence)
  - availability_signal (one of: available_now, open_to_work, unknown)
  - availability_evidence (short quoted phrase from profile/resume, or empty string)

Return a JSON object with one key "candidates" containing a list of objects with those keys.
Order by rank ascending (rank 1 = best match).
"""


# Availability is now fully scored within the rubric (Location & Availability dimension).
# No additional bonus is applied — keeping zero values to avoid breaking existing call sites.
# Maps candidate_id -> disqualify_reason for the most recent evaluate_candidates call.
# Populated before returning so callers can read disqualification rationales.
disqualified_reasons: dict[str, str] = {}
# Maps candidate_id -> full CandidateResult for disqualified candidates.
disqualified_results: dict[str, "CandidateResult"] = {}

_AVAILABILITY_BONUS = {
        "available_now": 0.0,
        "open_to_work": 0.0,
        "unknown": 0.0,
}


def _build_system_prompt(dealbreakers: list[dict] | None) -> str:
    """Build the system prompt, injecting any dealbreaker rules at the top."""
    if not dealbreakers:
        dealbreaker_section = ""
    else:
        rules = "\n".join(
            (
                f"  - {d.get('rule', '')} | JD evidence: {d.get('source_quote', '')}"
                if d.get("source_quote")
                else f"  - {d.get('rule', '')}"
            )
            for d in dealbreakers
            if d.get("rule")
        )
        dealbreaker_section = (
            "\n\nHARD FILTERS (check FIRST, before scoring):\n"
            "If a candidate violates ANY of the following rules, set fit_score=0, tier=\"D\", "
            "disqualified=true, and do NOT apply the rubric scoring.\n"
            "Only disqualify when the candidate profile contains direct contradictory evidence or an explicit statement of absence. "
            "Do NOT disqualify because information is missing, unclear, or merely weaker than preferred. Missing evidence should lower rubric scores, not trigger a hard filter. "
            "The rationale field is REQUIRED for every candidate — never omit it or leave it empty. "
            "For disqualified candidates, the rationale MUST name the specific hard filter rule violated and quote the exact candidate evidence that triggered it. "
            "For non-disqualified candidates, the rationale must be a concise paragraph summarising fit.\n"
            f"{rules}\n"
        )
    return _SYSTEM_PROMPT + dealbreaker_section


_EVAL_BATCH_SIZE = 15  # Max candidates per LLM call; keeps output tokens well within model limits.


def _call_llm_batch(
    client: OpenAI,
    system_prompt: str,
    jd_text: str,
    batch: list[CandidateMeta],
    soft_criteria: list[SoftCriteria],
    cfg: Config,
    telemetry: RunTelemetry,
) -> list[dict[str, Any]]:
    """Send one batch to the LLM and return raw evaluation dicts."""
    user_content = _build_prompt(jd_text, batch, soft_criteria)
    response = client.chat.completions.create(
        model=cfg.llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
    )
    usage = response.usage
    if usage:
        telemetry.llm_input_tokens += usage.prompt_tokens
        telemetry.llm_output_tokens += usage.completion_tokens
        telemetry.llm_input_cost_usd += (
            usage.prompt_tokens / 1_000_000 * cfg.llm_input_price_per_million_tokens
        )
        telemetry.llm_output_cost_usd += (
            usage.completion_tokens / 1_000_000 * cfg.llm_output_price_per_million_tokens
        )
        telemetry.total_cost_usd = (
            telemetry.embedding_cost_usd
            + telemetry.llm_input_cost_usd
            + telemetry.llm_output_cost_usd
        )
    raw = json.loads(response.choices[0].message.content or "{}")
    return raw.get("candidates", [])


def evaluate_candidates(
    jd_text: str,
    candidates: list[CandidateMeta],
    soft_criteria: list[SoftCriteria],
    cfg: Config,
    telemetry: RunTelemetry,
    dealbreakers: list[dict] | None = None,
) -> list[CandidateResult]:
    client = OpenAI(api_key=cfg.openai_api_key, timeout=cfg.llm_timeout_seconds)
    system_prompt = _build_system_prompt(dealbreakers)

    # Split into batches so no single call risks hitting output token limits.
    batches = [
        candidates[i : i + _EVAL_BATCH_SIZE]
        for i in range(0, len(candidates), _EVAL_BATCH_SIZE)
    ]
    if len(batches) > 1:
        print(
            f"[Evaluator] Splitting {len(candidates)} candidates into {len(batches)} batches of <={_EVAL_BATCH_SIZE}.",
            flush=True,
        )

    evaluations: list[dict[str, Any]] = []
    for batch_idx, batch in enumerate(batches, 1):
        batch_evals = _call_llm_batch(client, system_prompt, jd_text, batch, soft_criteria, cfg, telemetry)
        evaluated_ids = {str(e.get("candidate_id", "")).strip() for e in batch_evals}
        missing = [c for c in batch if str(c.id) not in evaluated_ids]
        if missing:
            print(
                f"[Evaluator] Batch {batch_idx}: LLM omitted {len(missing)} candidate(s) "
                f"({', '.join(str(c.id) for c in missing)}). Retrying individually.",
                flush=True,
            )
            for c in missing:
                retry_evals = _call_llm_batch(client, system_prompt, jd_text, [c], soft_criteria, cfg, telemetry)
                if retry_evals:
                    batch_evals.extend(retry_evals)
                else:
                    print(f"[Evaluator] Candidate {c.id} still omitted after retry - skipping.", flush=True)
        evaluations.extend(batch_evals)

    candidate_map = {str(c.id): c for c in candidates}
    results: list[CandidateResult] = []

    ordered_candidates = list(candidates)
    for index, e in enumerate(evaluations):
        cid = str(e.get("candidate_id", "")).strip()
        c = candidate_map.get(cid)
        if c is None and cid.isdigit():
            c = candidate_map.get(str(int(cid)))
        if c is None and not cid and index < len(ordered_candidates):
            # Fallback for partially compliant model output: align by order.
            c = ordered_candidates[index]
        if c is None:
            continue
        signal = str(e.get("availability_signal", "unknown")).strip().lower()
        evidence = str(e.get("availability_evidence", "")).strip()
        availability_bonus = _availability_bonus(signal, evidence)
        fit_score = float(e.get("fit_score", 0.0))
        adjusted_score = fit_score + availability_bonus
        tier = str(e.get("tier", "")).strip().upper()
        # Enforce tier thresholds from spec regardless of what LLM returned
        llm_disqualified = bool(e.get("disqualified", False)) or tier == "D"
        if not llm_disqualified:
            score_for_tier = float(e.get("fit_score", 0.0))
            if score_for_tier >= 8.5:
                tier = "A"
            elif score_for_tier >= 7.0:
                tier = "B"
            elif score_for_tier >= 5.0:
                tier = "C"
            else:
                tier = "D"
        disqualified = llm_disqualified or tier == "D"
        raw_reason = str(e.get("rationale", "")).strip() if disqualified else ""
        if disqualified and not raw_reason:
            # Build a useful fallback when the LLM omitted a rationale.
            fit = float(e.get("fit_score", 0.0))
            if bool(e.get("disqualified", False)):
                raw_reason = "Disqualified by a hard filter rule (LLM did not provide a specific reason)."
            else:
                raw_reason = f"Score too low to rank (fit score {fit:.1f}/10, below the 5.0 threshold)."
        disqualify_reason = raw_reason

        results.append(
            CandidateResult(
                rank=int(e.get("rank", 0)),
                candidate=c,
                fit_score=fit_score,
                strengths=e.get("strengths", []),
                risks=e.get("risks", []),
                rationale=e.get("rationale", ""),
                embedding_score=c.__dict__.get("_embedding_score", 0.0),
                tier=tier,
                disqualified=disqualified,
                disqualify_reason=disqualify_reason if disqualified else "",
            )
        )
        # Keep details for transparent post-processing without changing public model.
        results[-1].__dict__["availability_signal"] = signal
        results[-1].__dict__["availability_evidence"] = evidence
        results[-1].__dict__["availability_bonus"] = availability_bonus
        results[-1].__dict__["adjusted_score"] = adjusted_score
        # Rubric dimension scores for transparency.
        results[-1].__dict__["experience_score"] = float(e.get("experience_score", 0.0))
        results[-1].__dict__["skills_score"] = float(e.get("skills_score", 0.0))
        results[-1].__dict__["location_score"] = float(e.get("location_score", 0.0))
        results[-1].__dict__["english_ai_score"] = float(e.get("english_ai_score", 0.0))
        results[-1].__dict__["salary_score"] = float(e.get("salary_score", 0.0))

    # Exclude hard-filter disqualified candidates (tier D) from ranked output.
    qualified = [r for r in results if not r.disqualified]
    # Publish disqualification rationales for transparent JSON serialization.
    global disqualified_reasons, disqualified_results
    disqualified_reasons = {
        str(r.candidate.id): r.disqualify_reason
        for r in results if r.disqualified
    }
    disqualified_results = {
        str(r.candidate.id): r
        for r in results if r.disqualified
    }

    # Deterministic rerank: preserve LLM fit ordering, then nudge by explicit availability evidence.
    qualified.sort(
        key=lambda r: (
            -float(r.__dict__.get("adjusted_score", r.fit_score)),
            r.rank,
        )
    )

    # Reassign rank after deterministic post-processing.
    for idx, result in enumerate(qualified, 1):
        result.rank = idx

    return qualified


def _build_prompt(
    jd_text: str,
    candidates: list[CandidateMeta],
    soft_criteria: list[SoftCriteria],
) -> str:
    soft_block = ""
    if soft_criteria:
        soft_block = "\n\nSoft criteria to consider:\n" + "\n".join(
            f"- {s.description}" for s in soft_criteria
        )

    candidates_block = "\n\n".join(
        f"Candidate ID: {c.id}\n"
        f"Name: {c.name}\n"
        f"Position: {c.current_position} @ {c.current_company}\n"
        f"Location: {c.location}\n"
        f"Degree: {c.latest_degree} — {c.latest_university}\n"
        f"Tags: {', '.join(c.tags)}\n"
        f"Industries: {', '.join(c.industries)}\n"
        f"Resume:\n{c.resume_text[:4000] if c.resume_text else '(no resume text available — evaluate from profile fields only)'}"
        for c in candidates
    )

    return f"Job Description:\n{jd_text}{soft_block}\n\nCandidates:\n{candidates_block}"


def _availability_bonus(signal: str, evidence: str) -> float:
    normalized = signal.strip().lower().replace(" ", "_")
    if not evidence:
        return 0.0
    return _AVAILABILITY_BONUS.get(normalized, 0.0)
