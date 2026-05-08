from __future__ import annotations

import hashlib
import json
import os
import re
import time
from typing import Any

from openai import OpenAI

from sa_candidate_finder.config import Config
from sa_candidate_finder.models import RunTelemetry


_SYSTEM_PROMPT = """
You are an expert AI recruiting agent. Given a job description, extract the most relevant candidate search filters, keywords, and hard disqualifying criteria.

Also extract a list of up to 10 keywords (1-3 words each, LLM-chosen, not just phrases from the JD) that would maximize candidate search quality for this job. These can be skills, responsibilities, or any other relevant terms.

Also extract a list of hard dealbreakers from the JD. These are absolute disqualifying criteria — if a candidate violates any one, they cannot be considered regardless of other strengths.

Only include a JD-specific dealbreaker when it is explicitly non-negotiable in the JD requirements (e.g., Required Qualifications, Requirements, Must-Have, Non-Negotiables). Use language like: must, required, mandatory, non-negotiable, dealbreaker, only, cannot be considered, not eligible, max budget, salary cap.

Prefer extracting JD-specific dealbreakers from explicit requirement lines. Do not derive dealbreakers from Responsibilities, Preferred/Nice-to-have sections, or general role context unless those lines explicitly state a non-negotiable requirement.

Good hard-dealbreaker examples:
- explicit location restrictions
- explicit timezone/work-hours requirements
- explicit salary ceiling or budget cap
- explicit mandatory tool, certification, or licensing requirement

Do NOT convert normal screening criteria, preferences, or scoring dimensions into hard dealbreakers. In particular, do NOT create hard dealbreakers for general experience level, domain exposure, bilingual preference, clinical background, cardiology exposure, or EMR familiarity unless the JD explicitly states they are mandatory/non-negotiable.

Return a JSON object with:
    "keywords": list of up to 10 keywords (string, 1-3 words each)
    "dealbreakers": list of objects, each with:
        - "id": short snake_case identifier (e.g. "no_philippines", "salary_over_1500")
        - "rule": one sentence describing the disqualifying condition
        - "source_quote": short verbatim quote from the JD proving it is explicit
        - "keyword_signals": list of 1-3 lowercase string patterns that would appear in a violating candidate's profile (empty list if not detectable by keyword)

Do not prompt the user.
"""


_KEYWORD_SYNONYMS = {
    "congestive heart failure": "chf",
    "heart failure": "chf",
    "ehr": "emr/ehr",
    "electronic health record": "emr/ehr",
    "electronic health records": "emr/ehr",
    "emr systems": "emr/ehr",
    "ehr systems": "emr/ehr",
    "telemedicine": "telehealth",
    "patient comms": "patient communication",
    "patient communications": "patient communication",
}

_DISPLAY_FORMS = {
    "chf": "CHF",
    "emr/ehr": "EMR/EHR",
    "ai": "AI",
    "hipaa": "HIPAA",
}

_NOISE_TERMS = {
    "team player",
    "hard worker",
    "fast paced",
    "dynamic environment",
}

_DOMAIN_PRIORITY = {
    "cardiology",
    "chf",
    "patient communication",
    "care coordination",
    "clinical documentation",
    "emr/ehr",
    "telehealth",
    "remote patient monitoring",
    "hipaa compliance",
    "medical billing",
}

_EXTRACTION_CACHE_VERSION = 2

_DEALBREAKER_EXPLICIT_MARKERS = {
    "must",
    "required",
    "mandatory",
    "non-negotiable",
    "dealbreaker",
    "cannot be considered",
    "not eligible",
    "salary cap",
    "budget",
    "max",
}

_DEALBREAKER_OBJECTIVE_TERMS = {
    "location",
    "country",
    "salary",
    "budget",
    "compensation",
    "certification",
    "license",
    "licensed",
    "tool",
    "authorization",
    "visa",
}

_DEALBREAKER_VAGUE_PATTERNS = {
    "experience",
    "background",
    "exposure",
    "familiarity",
    "proficiency",
    "timezone",
    "pst",
    "est",
    "cst",
    "work hours",
}


def _keyword_cache_path(jd_text: str) -> str:
    jd_hash = hashlib.sha256((jd_text or "").strip().encode("utf-8")).hexdigest()[:16]
    cache_dir = os.path.join(os.path.dirname(__file__), "../../cache/keyword_sets")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"jd_{jd_hash}.json")


def _load_cached_extraction(jd_text: str) -> dict | None:
    """Return cached keywords and dealbreakers, or None if not cached."""
    path = _keyword_cache_path(jd_text)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("cache_version") != _EXTRACTION_CACHE_VERSION:
            return None
        keywords = payload.get("keywords", [])
        dealbreakers = payload.get("dealbreakers", [])
        if isinstance(keywords, list):
            return {"keywords": [str(k) for k in keywords], "dealbreakers": dealbreakers}
    except Exception:
        return None
    return None


def _save_cached_extraction(jd_text: str, keywords: list[str], dealbreakers: list[dict]) -> None:
    path = _keyword_cache_path(jd_text)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "cache_version": _EXTRACTION_CACHE_VERSION,
                    "cached_at": time.time(),
                    "keywords": keywords,
                    "dealbreakers": dealbreakers,
                },
                f,
            )
    except Exception:
        pass


def _normalize_dealbreakers(raw_dealbreakers: list[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in raw_dealbreakers or []:
        if not isinstance(item, dict):
            continue
        rule = str(item.get("rule", "")).strip()
        source_quote = str(item.get("source_quote", "")).strip()
        if not rule or not source_quote:
            continue
        keyword_signals = item.get("keyword_signals", [])
        if not isinstance(keyword_signals, list):
            keyword_signals = []
        normalized.append(
            {
                "id": str(item.get("id", "")).strip() or "jd_dealbreaker",
                "rule": rule,
                "source_quote": source_quote,
                "keyword_signals": [str(signal).strip().lower() for signal in keyword_signals if str(signal).strip()],
            }
        )
    return normalized


def _merge_dealbreakers(universal_dealbreakers: list[dict[str, Any]], jd_dealbreakers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for item in (universal_dealbreakers or []) + (jd_dealbreakers or []):
        if not isinstance(item, dict):
            continue
        did = str(item.get("id", "")).strip() or "dealbreaker"
        if did in seen_ids:
            continue
        seen_ids.add(did)
        merged.append(item)
    return merged


def _is_explicit_objective_dealbreaker(dealbreaker: dict[str, Any]) -> bool:
    """Accept a JD-specific dealbreaker if the LLM supplied a non-empty source_quote
    AND the combined rule+quote text contains at least one explicit requirement marker.
    This trusts the LLM's content judgment while preventing hallucinations that lack
    any JD evidence."""
    source_quote = (dealbreaker.get("source_quote") or "").strip()
    if not source_quote:
        return False
    text = f"{dealbreaker.get('rule', '')} {source_quote}".lower()
    return any(marker in text for marker in _DEALBREAKER_EXPLICIT_MARKERS)


def _normalize_keyword(keyword: str) -> str:
    value = (keyword or "").strip().lower()
    value = re.sub(r"[\-_&]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    if not value:
        return ""
    value = _KEYWORD_SYNONYMS.get(value, value)
    if value in _NOISE_TERMS:
        return ""
    if len(value) < 2 and value not in {"ai"}:
        return ""
    return value


def _display_keyword(canonical: str) -> str:
    return _DISPLAY_FORMS.get(canonical, canonical)


def _score_keyword(canonical: str, jd_text: str, title_line: str) -> tuple[int, int]:
    jd = jd_text.lower()
    token = canonical.lower()
    freq = jd.count(token)
    score = 0
    if freq > 0:
        score += 3 * freq
    first_pos = jd.find(token)
    if first_pos != -1:
        quarter = max(1, len(jd) // 4)
        if first_pos <= quarter:
            score += 2
    else:
        first_pos = 10**9
    if token in title_line.lower():
        score += 4
    if token in _DOMAIN_PRIORITY:
        score += 2
    return score, first_pos


def _deterministic_keywords(jd_text: str, llm_keywords: list[str], max_keywords: int) -> list[str]:
    title_line = (jd_text or "").splitlines()[0] if jd_text else ""
    canonical_terms: dict[str, bool] = {}

    for term in llm_keywords or []:
        canonical = _normalize_keyword(term)
        if canonical:
            canonical_terms[canonical] = True

    ranked: list[tuple[str, int, int]] = []
    for canonical in canonical_terms:
        score, first_pos = _score_keyword(canonical, jd_text, title_line)
        if score >= 1:
            ranked.append((canonical, score, first_pos))

    ranked.sort(key=lambda x: (-x[1], x[2], -len(x[0]), x[0]))
    return [_display_keyword(c) for c, _, _ in ranked[:max_keywords]]


def _heuristic_keywords_from_jd(jd_text: str, max_keywords: int) -> list[str]:
    jd_lower = (jd_text or "").lower()
    title_line = (jd_text or "").splitlines()[0] if jd_text else ""

    ranked: list[tuple[str, int, int]] = []
    for canonical in sorted(_DOMAIN_PRIORITY):
        score, first_pos = _score_keyword(canonical, jd_text, title_line)
        if score > 0 or canonical in jd_lower:
            ranked.append((canonical, max(score, 1), first_pos))

    # Fallback to broad but useful defaults if nothing domain-specific matched.
    if not ranked:
        defaults = ["patient communication", "care coordination", "clinical documentation"]
        ranked = [(k, 1, jd_lower.find(k) if k in jd_lower else 10**9) for k in defaults]

    ranked.sort(key=lambda x: (-x[1], x[2], -len(x[0]), x[0]))
    return [_display_keyword(c) for c, _, _ in ranked[:max_keywords]]


def extract_constraints(
    jd_text: str,
    cfg: Config,
    telemetry: RunTelemetry | None = None,
) -> dict[str, Any]:
    cached = _load_cached_extraction(jd_text)

    client = OpenAI(api_key=cfg.openai_api_key, timeout=cfg.llm_timeout_seconds)

    raw: dict[str, Any] = {}
    llm_keywords: list[str] = []
    relaxation_plan: list[Any] = []
    jd_dealbreakers: list[dict[str, Any]] = []
    llm_error: str = ""
    for attempt in range(cfg.max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=cfg.llm_model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": jd_text},
                ],
                response_format={"type": "json_object"},
            )
            raw = json.loads(response.choices[0].message.content or "{}")
            llm_keywords = raw.get("keywords", [])
            relaxation_plan = raw.get("relaxation_plan", [])
            jd_dealbreakers = _normalize_dealbreakers(raw.get("dealbreakers", []))
            llm_error = ""
            break
        except Exception as exc:
            llm_error = f"LLM extraction failed (attempt {attempt + 1}/{cfg.max_retries + 1}): {exc}"
            if attempt < cfg.max_retries:
                backoff_s = 2 ** attempt if cfg.retry_backoff == "exponential" else 2
                time.sleep(backoff_s)

    if llm_error:
        print(f"[Extractor] {llm_error}. Falling back to cached/heuristic constraints.", flush=True)
        if telemetry is not None:
            telemetry.errors.append({"stage": "extract_constraints", "message": llm_error})

    if cached is not None:
        keywords = cached["keywords"]
        # Use cached JD-specific dealbreakers if present, otherwise use freshly extracted ones.
        if cached.get("dealbreakers"):
            jd_dealbreakers = cached["dealbreakers"]
    else:
        if llm_keywords:
            keywords = _deterministic_keywords(jd_text, llm_keywords, cfg.max_hard_constraints)
        else:
            keywords = _heuristic_keywords_from_jd(jd_text, cfg.max_hard_constraints)
        jd_dealbreakers = [d for d in jd_dealbreakers if _is_explicit_objective_dealbreaker(d)]
        _save_cached_extraction(jd_text, keywords, jd_dealbreakers)

    # Merge system-owned universal + JD-specific dealbreakers.
    dealbreakers = _merge_dealbreakers(
        cfg.universal_dealbreakers,
        [d for d in jd_dealbreakers if isinstance(d, dict)],
    )

    if telemetry is not None:
        telemetry.hard_constraints = []
        telemetry.soft_criteria = keywords

    return {
        "keywords": keywords,
        "relaxation_plan": relaxation_plan,
        "dealbreakers": dealbreakers,
    }

