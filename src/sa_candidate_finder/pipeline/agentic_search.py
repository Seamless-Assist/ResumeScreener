"""
Agentic candidate search and ranking loop for staged evaluation.
- Stage 1: Search among candidates who applied to the job.
- Stage 2: If not enough matches, adjust keywords and retry among applicants.
- Stage 3: If still not enough, expand to all candidates and repeat with best keyword set.
- Results are always ranked by best match.
"""
import re
from typing import List, Optional
from sa_candidate_finder.models import CandidateMeta, CandidateResult
from sa_candidate_finder.pipeline.extractor import extract_constraints
from sa_candidate_finder.config import Config
from sa_candidate_finder.mcp_client import search_candidates


def _normalize_keyword(value: str) -> str:
    token = (value or "").strip().lower()
    token = re.sub(r"[\-_&/]+", " ", token)
    token = re.sub(r"\s+", " ", token)
    return token


def _is_semantic_match(left: str, right: str) -> bool:
    a = _normalize_keyword(left)
    b = _normalize_keyword(right)
    if not a or not b:
        return False
    if a == b:
        return True
    if a in b or b in a:
        return True
    a_tokens = set(a.split())
    b_tokens = set(b.split())
    if not a_tokens or not b_tokens:
        return False
    overlap = a_tokens & b_tokens
    return len(overlap) >= min(2, len(a_tokens), len(b_tokens))


def reconcile_keywords(llm_keywords: List[str], user_keywords: Optional[List[str]], max_keywords: int) -> List[str]:
    final_keywords, _ = reconcile_keywords_with_audit(llm_keywords, user_keywords, max_keywords)
    return final_keywords


def reconcile_keywords_with_audit(
    llm_keywords: List[str],
    user_keywords: Optional[List[str]],
    max_keywords: int,
) -> tuple[List[str], dict]:
    """
    Reconcile user keyword directives with LLM keywords.

    User directive syntax (comma-separated via CLI):
    - plain term or +term: add
    - -term: remove semantically matching existing terms
    - old->new: replace semantically matching old with new
    - no input: keep LLM keywords unchanged
    """
    final_keywords = [k.strip() for k in (llm_keywords or []) if str(k).strip()]
    directives = [d.strip() for d in (user_keywords or []) if str(d).strip()]

    audit = {
        "directives": directives,
        "added": [],
        "removed": [],
        "replaced": [],
        "ignored": [],
        "trimmed": [],
    }

    if not directives:
        return final_keywords[:max_keywords], audit

    for directive in directives:
        if "->" in directive:
            old_value, new_value = directive.split("->", 1)
            old_value = old_value.strip()
            new_value = new_value.strip().lstrip("+")
            if not old_value or not new_value:
                audit["ignored"].append(directive)
                continue
            matched = [k for k in final_keywords if _is_semantic_match(k, old_value)]
            final_keywords = [k for k in final_keywords if not _is_semantic_match(k, old_value)]
            if matched:
                audit["removed"].extend(matched)
            added_new = False
            if not any(_is_semantic_match(k, new_value) for k in final_keywords):
                final_keywords.append(new_value)
                audit["added"].append(new_value)
                added_new = True
            audit["replaced"].append({"from": old_value, "to": new_value, "matched": matched, "added": added_new})
            continue

        if directive.startswith("-"):
            remove_value = directive[1:].strip()
            if not remove_value:
                audit["ignored"].append(directive)
                continue
            matched = [k for k in final_keywords if _is_semantic_match(k, remove_value)]
            final_keywords = [k for k in final_keywords if not _is_semantic_match(k, remove_value)]
            if matched:
                audit["removed"].extend(matched)
            else:
                audit["ignored"].append(directive)
            continue

        add_value = directive[1:].strip() if directive.startswith("+") else directive
        if not add_value:
            audit["ignored"].append(directive)
            continue
        if not any(_is_semantic_match(k, add_value) for k in final_keywords):
            final_keywords.append(add_value)
            audit["added"].append(add_value)
        else:
            audit["ignored"].append(directive)

    deduped: List[str] = []
    for keyword in final_keywords:
        if not any(_is_semantic_match(keyword, existing) for existing in deduped):
            deduped.append(keyword)

    if len(deduped) > max_keywords:
        audit["trimmed"] = deduped[max_keywords:]
    return deduped[:max_keywords], audit


def agentic_candidate_search(
    jd_text: str,
    applied_candidates: List[CandidateMeta],
    all_candidates: List[CandidateMeta],
    cfg: Config,
    user_keywords: Optional[List[str]] = None,
    required_keywords: Optional[List[str]] = None,
    target_count: int = 10,
    limit_results: bool = True,
    base_keywords: Optional[List[str]] = None,
    relaxation_plan: Optional[List[dict]] = None,
    dealbreakers: Optional[List[dict]] = None,
) -> List[CandidateResult]:
    # Reuse deterministic keywords from caller when provided to keep runs stable.
    if base_keywords is None or relaxation_plan is None:
        extraction = extract_constraints(jd_text, cfg)
        llm_keywords = extraction.get("keywords", [])
        keyword_set = reconcile_keywords(llm_keywords, user_keywords, cfg.max_keyword_set)
        plan = extraction.get("relaxation_plan", [])
    else:
        keyword_set = reconcile_keywords(base_keywords, user_keywords, cfg.max_keyword_set)
        plan = relaxation_plan

    def rank_candidates(candidates, keywords):
        return search_candidates(
            candidates,
            keywords,
            cfg,
            dealbreakers=dealbreakers,
            required_keywords=required_keywords,
        )

    def maybe_limit(results: List[CandidateResult]) -> List[CandidateResult]:
        if not limit_results:
            return results
        return results[:target_count]

    # Stage 1: Try full LLM-extracted keyword set on applied candidates
    results = rank_candidates(applied_candidates, keyword_set)
    if results:
        return maybe_limit(results)

    # Stage 2: Follow LLM's relaxation plan if no results
    current_keywords = keyword_set.copy()
    for step in plan:
        action = step.get("action")
        target = step.get("target")
        details = step.get("details", "")
        if action == "drop" and target in current_keywords:
            current_keywords = [k for k in current_keywords if k != target]
        elif action == "split" and target in current_keywords:
            parts = [w for w in target.split() if w]
            current_keywords = [k for k in current_keywords if k != target] + parts
        elif action == "replace" and target in current_keywords:
            replacement = details.strip() or target
            current_keywords = [k if k != target else replacement for k in current_keywords]
        elif action == "broaden":
            broadened = details.split(",") if details else []
            current_keywords += [b.strip() for b in broadened if b.strip() and b.strip() not in current_keywords]
        elif action == "stop":
            break
        # Remove duplicates and trim
        current_keywords = list(dict.fromkeys(current_keywords))[:cfg.max_keyword_set]
        results = rank_candidates(applied_candidates, current_keywords)
        if results:
            return maybe_limit(results)

    # Stage 3: Expand to all candidates with best keyword set
    results = rank_candidates(all_candidates, keyword_set)
    if results:
        return maybe_limit(results)

    # Stage 4: Keep relaxing for all candidates
    for n in range(len(keyword_set) - 1, 0, -1):
        reduced_keywords = keyword_set[:n]
        results = rank_candidates(all_candidates, reduced_keywords)
        if results:
            return maybe_limit(results)

    return []
