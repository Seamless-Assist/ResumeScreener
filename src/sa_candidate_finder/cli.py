from __future__ import annotations

from pathlib import Path

import typer

from sa_candidate_finder import settings


app = typer.Typer(
    name="SACandidateFinder",
    help="SeamlessAssist AI Candidate Finder — local agent for resume search.",
    add_completion=False,
)


def should_expand_global_pool(
    *,
    rerank_fast: bool,
    good_match_count: int,
    target_good_matches: int,
) -> bool:
    """Return whether this run should search candidates who did not apply.

    Web reranks intentionally use fast mode: recruiters expect the existing
    applicant pool to be rescored, not a fresh search across the full Manatal
    database.
    """
    return (not rerank_fast) and good_match_count < target_good_matches


@app.command("help")
def help_cmd() -> None:
    """Show all available commands and their inputs."""
    typer.echo(
        """
SACandidateFinder — SeamlessAssist AI Candidate Finder

Commands:
  help                          Show this help message
  search --jd <path> --top <n>  Run full candidate search pipeline for a job description file
  feedback --jd <path>          Provide feedback on the most recent search run for that JD
  learn                         Apply accumulated feedback to refine future searches

Examples:
  SACandidateFinder search --jd jobs/senior_engineer.txt --top 5
  SACandidateFinder feedback --jd jobs/senior_engineer.txt
  SACandidateFinder learn

Inputs:
  --jd   Path to job description file (.txt or .pdf)
  --top  Number of final candidates to return (default: 5)

Internal settings:
  Runtime settings are centralized in src/sa_candidate_finder/settings.py
  Secrets are centralized in src/sa_candidate_finder/secrets.py
"""
    )



# New agentic search command
@app.command("agentic-search")
def agentic_search_cmd(
    job_id: str = typer.Option(..., "--job-id", help="Manatal Job ID to search candidates for."),
    keywords: str = typer.Option(None, "--keywords", help="Optional keyword directives: add(plain/+term), remove(-term), modify(old->new)."),
    required_keywords: str = typer.Option(None, "--required-keywords", help="Deprecated (no-op). Use --keywords for session keyword directives."),
    disable_hard_filter_rules: str = typer.Option(None, "--disable-hard-filter-rules", help="Rule texts to disable, separated by ||."),
    rerank_fast: bool = typer.Option(False, "--rerank-fast", help="Fast rerank mode: skip global pool expansion and reduce LLM eval volume."),
    top: int = typer.Option(10, "--top", help="Number of candidates to return (default: 10)."),
) -> None:
    """Agentic candidate search for a given job ID."""
    import os
    import re
    from sa_candidate_finder.secrets import MANATAL_API_KEY
    from sa_candidate_finder.manatal_jobs import fetch_all_jobs, strip_html
    from sa_candidate_finder.pipeline.agentic_search import agentic_candidate_search
    from sa_candidate_finder.models import CandidateMeta
    from sa_candidate_finder.config import load_config

    print("[DEBUG] STARTING AGENTIC SEARCH", flush=True)
    cfg = load_config()
    # Print and log keywords and filters
    from pathlib import Path
    telemetry_log_path = Path(cfg.telemetry_log_path)
    telemetry_log_path.parent.mkdir(parents=True, exist_ok=True)

    # Logging helper
    def log(msg):
        with telemetry_log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    print(f"[DEBUG] Loaded config. Starting agentic search for Job ID: {job_id}", flush=True)
    log(f"[AgenticSearch] Starting agentic search for Job ID: {job_id}")
    # Fetch all jobs and find the selected one
    print("[DEBUG] Fetching all jobs...", flush=True)
    jobs = fetch_all_jobs(MANATAL_API_KEY)
    print("[DEBUG] Searching for job ID in jobs list...", flush=True)
    job = next((j for j in jobs if str(j.get("id")) == str(job_id)), None)
    if not job:
        print(f"[DEBUG] Job ID {job_id} not found in jobs list!", flush=True)
        log(f"Error: Job ID {job_id} not found.")
        raise typer.Exit(1)
    raw_desc = job.get("description") or ""
    jd_text = strip_html(raw_desc) if raw_desc else (job.get("position_name") or "")
    print(f"[DEBUG] Found job: {job.get('position_name')} (ID: {job_id})", flush=True)
    print(f"[DEBUG] JD text length after HTML strip: {len(jd_text)} chars", flush=True)
    if len(jd_text) < 50:
        print(f"[DEBUG] WARNING: JD text is very short ({len(jd_text)} chars). Description may be missing in Manatal.", flush=True)
        log(f"[AgenticSearch] WARNING: JD text is very short ({len(jd_text)} chars). Keyword extraction may be unreliable.")
    log(f"[AgenticSearch] Job: {job.get('position_name')} (ID: {job_id})")


    # Parse user keywords
    user_keywords = [k.strip() for k in keywords.split(",") if k.strip()] if keywords else None
    required_keyword_list = [k.strip() for k in required_keywords.split(",") if k.strip()] if required_keywords else None
    disabled_hard_filter_rules = [r.strip() for r in disable_hard_filter_rules.split("||") if r.strip()] if disable_hard_filter_rules else []


    # Fully agentic LLM-driven extraction
    from sa_candidate_finder.pipeline.extractor import extract_constraints
    print("[DEBUG] Extracting constraints from JD text...", flush=True)
    extraction = extract_constraints(jd_text, cfg)
    llm_keywords = extraction.get("keywords", [])
    relaxation_plan = extraction.get("relaxation_plan", [])
    dealbreakers = extraction.get("dealbreakers", [])
    if required_keyword_list:
        print(
            "[DEBUG] --required-keywords is deprecated and ignored. "
            "Use --keywords directives for step-1 keyword matching.",
            flush=True,
        )
    if disabled_hard_filter_rules:
        def _norm_rule(value: str) -> str:
            import re as _re
            return _re.sub(r"\s+", " ", (value or "").strip().lower())

        disabled_norm = {_norm_rule(r) for r in disabled_hard_filter_rules}
        dealbreakers = [
            d for d in dealbreakers
            if _norm_rule(str(d.get("rule", ""))) not in disabled_norm
        ]
    from sa_candidate_finder.pipeline.agentic_search import reconcile_keywords_with_audit
    keyword_set, keyword_audit = reconcile_keywords_with_audit(llm_keywords, user_keywords, cfg.max_keyword_set)

    def _effective_good_match_threshold(keyword_count: int) -> float:
        """Loosen quality gate when fewer keywords are active to keep candidate flow intuitive."""
        base = float(cfg.min_good_match_score)
        if keyword_count <= 1:
            return round(max(0.0, base - 2.0), 1)
        if keyword_count == 2:
            return round(max(0.0, base - 1.0), 1)
        return round(base, 1)

    effective_good_match_threshold = _effective_good_match_threshold(len(keyword_set))

    print(f"[DEBUG] User keywords: {user_keywords if user_keywords else 'None'}", flush=True)
    print(f"[DEBUG] Deprecated required filters input (ignored): {required_keyword_list if required_keyword_list else 'None'}", flush=True)
    print(f"[DEBUG] Disabled hard-filter rules: {disabled_hard_filter_rules if disabled_hard_filter_rules else 'None'}", flush=True)
    print(f"[DEBUG] LLM-extracted keywords: {llm_keywords}", flush=True)
    print(f"[DEBUG] Hard dealbreakers: {[d.get('id') for d in dealbreakers]}", flush=True)
    print(f"[DEBUG] Keyword directives: {keyword_audit.get('directives', []) or 'None'}", flush=True)
    print(f"[DEBUG] Keyword reconcile added: {keyword_audit.get('added', [])}", flush=True)
    print(f"[DEBUG] Keyword reconcile removed: {keyword_audit.get('removed', [])}", flush=True)
    print(f"[DEBUG] Keyword reconcile replaced: {keyword_audit.get('replaced', [])}", flush=True)
    print(f"[DEBUG] Keyword reconcile ignored: {keyword_audit.get('ignored', [])}", flush=True)
    if keyword_audit.get('trimmed'):
        print(f"[DEBUG] Keyword reconcile trimmed by max limit: {keyword_audit.get('trimmed')}", flush=True)
    print(f"[DEBUG] Final keyword set: {keyword_set}", flush=True)
    print(f"[DEBUG] Requested top count: {top}", flush=True)
    print(f"[DEBUG] Fast rerank mode: {rerank_fast}", flush=True)
    print(
        f"[DEBUG] Good match threshold: {effective_good_match_threshold}/10 "
        f"(base={cfg.min_good_match_score}/10, active_keywords={len(keyword_set)})",
        flush=True,
    )
    log(f"[AgenticSearch] User keywords: {user_keywords if user_keywords else 'None'}")
    log(f"[AgenticSearch] Deprecated required filters input (ignored): {required_keyword_list if required_keyword_list else 'None'}")
    log(f"[AgenticSearch] Disabled hard-filter rules: {disabled_hard_filter_rules if disabled_hard_filter_rules else 'None'}")
    log(f"[AgenticSearch] LLM-extracted keywords: {llm_keywords}")
    log(f"[AgenticSearch] Hard dealbreakers: {[d.get('id') for d in dealbreakers]}")
    log(f"[AgenticSearch] Keyword directives: {keyword_audit.get('directives', []) or 'None'}")
    log(f"[AgenticSearch] Keyword reconcile added: {keyword_audit.get('added', [])}")
    log(f"[AgenticSearch] Keyword reconcile removed: {keyword_audit.get('removed', [])}")
    log(f"[AgenticSearch] Keyword reconcile replaced: {keyword_audit.get('replaced', [])}")
    log(f"[AgenticSearch] Keyword reconcile ignored: {keyword_audit.get('ignored', [])}")
    if keyword_audit.get('trimmed'):
        log(f"[AgenticSearch] Keyword reconcile trimmed by max limit: {keyword_audit.get('trimmed')}")
    log(f"[AgenticSearch] Final keyword set: {keyword_set}")
    log(f"[AgenticSearch] Requested top count: {top}")
    log(f"[AgenticSearch] Fast rerank mode: {rerank_fast}")
    log(
        f"[AgenticSearch] Good match threshold: {effective_good_match_threshold}/10 "
        f"(base={cfg.min_good_match_score}/10, active_keywords={len(keyword_set)})"
    )


    from sa_candidate_finder.manatal_candidates import fetch_candidates_by_job, fetch_all_candidates

    import re as _re

    def _normalize_role(name: str) -> str:
        """Normalise a position name for role-sibling matching."""
        n = (name or "").lower().strip()
        n = _re.sub(r"[\s\-\u2013]+(?:t\d|tier\s*\d|remote|us|uk|ph|au|part.time|full.time|hourly)[^\w]*$", "", n)
        n = _re.sub(r"[^a-z0-9 ]+", " ", n)
        n = _re.sub(r"\s+", " ", n).strip()
        return n

    def _find_sibling_jobs(anchor_job: dict, all_jobs: list) -> list:
        anchor_norm = _normalize_role(anchor_job.get("position_name", ""))
        return [
            j for j in all_jobs
            if str(j.get("id")) != str(anchor_job.get("id"))
            and j.get("status") in ("active", "on_hold")
            and _normalize_role(j.get("position_name", "")) == anchor_norm
        ]

    def merge_results(result_lists):
        best_by_id = {}
        for result_list in result_lists:
            for r in result_list:
                cid = str(r.candidate.id)
                existing = best_by_id.get(cid)
                if existing is None or r.fit_score > existing.fit_score:
                    best_by_id[cid] = r
        merged = sorted(best_by_id.values(), key=lambda x: x.fit_score, reverse=True)
        for idx, r in enumerate(merged, start=1):
            r.rank = idx
        return merged

    def count_good_matches(result_list):
        return sum(1 for r in result_list if r.fit_score >= effective_good_match_threshold)

    # ── Discover all JDs for this role (normalised name match) ──────────────
    sibling_jobs = _find_sibling_jobs(job, jobs)
    all_role_job_ids = [str(job_id)] + [str(j["id"]) for j in sibling_jobs]
    role_name = job.get("position_name", f"Job {job_id}")
    print(f"[DEBUG] Role '{role_name}' spans {len(all_role_job_ids)} JDs: {all_role_job_ids}", flush=True)
    log(f"[AgenticSearch] Role '{role_name}' spans {len(all_role_job_ids)} JDs: {all_role_job_ids}")

    # ── Phase 1: fetch & dedup candidates from ALL role JDs in one pass ─────
    print("[DEBUG] Phase 1: Fetching candidates from all role JDs...", flush=True)
    log("[AgenticSearch] Phase 1: Fetching all role candidates...")
    all_role_candidates: list = []
    seen_ids: set = set()
    applied_candidate_ids: set = set()  # IDs of candidates who applied to this role (Phase 1)
    for jid in all_role_job_ids:
        for c in fetch_candidates_by_job(MANATAL_API_KEY, jid):
            if str(c.id) not in seen_ids:
                all_role_candidates.append(c)
                seen_ids.add(str(c.id))
                applied_candidate_ids.add(str(c.id))
    print(f"[DEBUG] Phase 1: {len(all_role_candidates)} unique candidates across all role JDs.", flush=True)
    log(f"[AgenticSearch] Phase 1: {len(all_role_candidates)} unique candidates across all role JDs.")

    results = agentic_candidate_search(
        jd_text=jd_text,
        applied_candidates=all_role_candidates,
        all_candidates=[],
        cfg=cfg,
        user_keywords=user_keywords,
        required_keywords=None,
        target_count=top,
        limit_results=False,
        base_keywords=llm_keywords,
        relaxation_plan=relaxation_plan,
        dealbreakers=dealbreakers,
    )
    print(f"[DEBUG] Phase 1: ranked candidates (keyword score > 0): {len(results)}", flush=True)
    print(
        f"[DEBUG] Phase 1: candidates above quality gate (score >= {effective_good_match_threshold}): {count_good_matches(results)}",
        flush=True,
    )
    log(f"[AgenticSearch] Phase 1: ranked candidates (keyword score > 0): {len(results)}")
    log(
        f"[AgenticSearch] Phase 1: candidates above quality gate (score >= {effective_good_match_threshold}): {count_good_matches(results)}"
    )

    # ── Phase 2: broader pool if quality-gated shortlist is still short ─────
    phase2_target = cfg.phase2_good_match_target
    phase1_good_match_count = count_good_matches(results)
    phase2_expanded = False
    if should_expand_global_pool(
        rerank_fast=rerank_fast,
        good_match_count=phase1_good_match_count,
        target_good_matches=phase2_target,
    ):
        phase2_expanded = True
        print("[DEBUG] Phase 2: Fetching broader candidate pool...", flush=True)
        log("[AgenticSearch] Phase 2: Fetching broader candidate pool...")
        all_candidates = fetch_all_candidates(MANATAL_API_KEY)
        extra = [c for c in all_candidates if str(c.id) not in seen_ids]
        print(f"[DEBUG] Phase 2: {len(extra)} additional candidates.", flush=True)
        log(f"[AgenticSearch] Phase 2: {len(extra)} additional candidates.")
        phase2_results = agentic_candidate_search(
            jd_text=jd_text,
            applied_candidates=[],
            all_candidates=extra,
            cfg=cfg,
            user_keywords=user_keywords,
            required_keywords=None,
            target_count=top,
            limit_results=False,
            base_keywords=llm_keywords,
            relaxation_plan=relaxation_plan,
            dealbreakers=dealbreakers,
        )
        results = merge_results([results, phase2_results])
        print(f"[DEBUG] Phase 2: cumulative good matches: {count_good_matches(results)}", flush=True)
        log(f"[AgenticSearch] Phase 2: cumulative good matches: {count_good_matches(results)}")
    elif rerank_fast and phase1_good_match_count < phase2_target:
        message = (
            "[DEBUG] Fast rerank: skipping Phase 2 global candidate search; "
            f"rescoring {len(all_role_candidates)} applied candidates only."
        )
        print(message, flush=True)
        log(message.removeprefix("[DEBUG] "))

    full_ranked_results = results
    # Snapshot matched keywords per candidate (r.strengths = matched keyword list from MCP stub,
    # before LLM evaluation overwrites strengths with qualitative bullet points).
    matched_keywords_by_id: dict[str, list[str]] = {
        str(r.candidate.id): list(r.strengths or []) for r in full_ranked_results
    }
    # Send all candidates who pass the Phase 1 quality gate to LLM evaluation.
    results = [r for r in full_ranked_results if r.fit_score >= effective_good_match_threshold]
    if rerank_fast and len(results) > cfg.shortlist_size:
        print(
            f"[DEBUG] Fast rerank: limiting LLM evaluation from {len(results)} "
            f"to the top {cfg.shortlist_size} candidates.",
            flush=True,
        )
        log(
            f"[AgenticSearch] Fast rerank: limiting LLM evaluation from {len(results)} "
            f"to the top {cfg.shortlist_size} candidates."
        )
        results = results[:cfg.shortlist_size]
    llm_eval_limit = len(results)
    print(
        f"[DEBUG] Search results before Claude evaluation: {len(results) if results else 0} "
        f"(from full pool: {len(full_ranked_results)})",
        flush=True,
    )
    print(
        f"[DEBUG] LLM eval policy: quality_gate score>={effective_good_match_threshold}; "
        f"selected={llm_eval_limit}",
        flush=True,
    )
    log(
        f"[AgenticSearch] LLM eval policy: quality_gate score>={effective_good_match_threshold}; "
        f"selected={llm_eval_limit}"
    )

    if not results:
        print("[DEBUG] No matching candidates found.", flush=True)
        log("[AgenticSearch] No matching candidates found.")
        return

    # --- Claude evaluation: apply 5-dimension rubric + hard dealbreakers ---
    print(f"[DEBUG] Running Claude evaluation on {len(results)} candidates...", flush=True)
    log(f"[AgenticSearch] Running Claude evaluation on {len(results)} candidates...")
    from sa_candidate_finder.pipeline.evaluator import evaluate_candidates
    from sa_candidate_finder.models import RunTelemetry
    import uuid, hashlib
    from datetime import datetime, timezone
    _telemetry = RunTelemetry(
        run_id=str(uuid.uuid4())[:8],
        jd_hash=hashlib.sha256(jd_text.encode()).hexdigest()[:16],
        started_at=datetime.now(timezone.utc).isoformat(),
        llm_model=cfg.llm_model,
    )
    def _normalize_resume_for_eval(resume_text: str) -> tuple[str, str]:
        rt = (resume_text or "").strip()
        if not rt:
            return "", "No resume text could be retrieved or parsed for this candidate — AI evaluation was skipped."

        if rt.startswith("__RESUME_PARSE_FAILED__"):
            return "", "Resume file could not be parsed into readable text — AI evaluation was skipped."

        head = rt[:256]
        if "%PDF-" in head:
            return "", "Resume parsing produced raw PDF stream data instead of readable text — AI evaluation was skipped."

        markers = ("/FlateDecode", "endobj", "stream", "xref", "trailer")
        marker_hits = sum(1 for m in markers if m in rt[:2000])
        if marker_hits >= 3:
            return "", "Resume parsing produced unreadable PDF structure text — AI evaluation was skipped."

        letter_count = len(re.findall(r"[A-Za-z]", rt[:2000]))
        if letter_count < 80:
            return "", "Resume text is too limited to evaluate reliably — AI evaluation was skipped."

        return rt, ""

    selected_candidates = [r.candidate for r in results]
    not_reviewed_by_id: dict[str, str] = {}
    candidates_to_eval = []
    for c in selected_candidates:
        normalized_resume_text, reason = _normalize_resume_for_eval(c.resume_text)
        if reason:
            not_reviewed_by_id[str(c.id)] = reason
        else:
            c.resume_text = normalized_resume_text
            candidates_to_eval.append(c)

    llm_reviewed_ids = {str(c.id) for c in candidates_to_eval}
    baseline_result_count = len(results)
    llm_disqualify_reasons = {}
    try:
        if candidates_to_eval:
            results = evaluate_candidates(
                jd_text=jd_text,
                candidates=candidates_to_eval,
                soft_criteria=[],
                cfg=cfg,
                telemetry=_telemetry,
                dealbreakers=dealbreakers,
            )
            from sa_candidate_finder.pipeline.evaluator import disqualified_reasons as _disqualified_reasons
            llm_disqualify_reasons = dict(_disqualified_reasons)  # capture first-pass reasons
            if not results:
                universal_only = [d for d in dealbreakers if d.get("universal")]
                if len(universal_only) != len(dealbreakers):
                    print(
                        "[DEBUG] All candidates were disqualified by JD-specific hard filters. Retrying Claude evaluation with universal hard filters only...",
                        flush=True,
                    )
                    log(
                        "[AgenticSearch] All candidates were disqualified by JD-specific hard filters. Retrying with universal hard filters only..."
                    )
                    results = evaluate_candidates(
                        jd_text=jd_text,
                        candidates=candidates_to_eval,
                        soft_criteria=[],
                        cfg=cfg,
                        telemetry=_telemetry,
                        dealbreakers=universal_only,
                    )
                    # Merge retry reasons; first-pass (JD-specific) reasons take precedence
                    from sa_candidate_finder.pipeline.evaluator import disqualified_reasons as _retry_reasons
                    merged = dict(_retry_reasons)
                    merged.update(llm_disqualify_reasons)  # first-pass wins for any overlap
                    llm_disqualify_reasons = merged
        else:
            results = []
    except Exception as exc:
        import traceback as _tb
        _exc_detail = f"{type(exc).__name__}: {exc}"
        print(f"[DEBUG] Claude evaluation failed — {_exc_detail}", flush=True)
        print(f"[DEBUG] Traceback:\n{_tb.format_exc()}", flush=True)
        log(f"[AgenticSearch] Claude evaluation failed: {_exc_detail}")
        from sa_candidate_finder.pipeline.evaluator import disqualified_reasons as _disqualified_reasons
        llm_disqualify_reasons = dict(_disqualified_reasons)
        # Reset so candidates aren't falsely marked as llm_qualified with no tier.
        results = []
        llm_reviewed_ids = set()
    llm_qualified_ids = {str(r.candidate.id) for r in results}
    from sa_candidate_finder.pipeline.evaluator import disqualified_results as _disqualified_results
    # Merge disqualified results into llm_result_by_id so rationale/scores are saved for all LLM-reviewed candidates
    llm_result_by_id = {str(r.candidate.id): r for r in results}
    for _cid, _dresult in _disqualified_results.items():
        if _cid not in llm_result_by_id:
            llm_result_by_id[_cid] = _dresult
    print(f"[DEBUG] After Claude evaluation: {len(results)} qualified candidates.", flush=True)
    log(f"[AgenticSearch] After Claude evaluation: {len(results)} qualified candidates.")

    if not results:
        if baseline_result_count > 0:
            print(
                "[DEBUG] Claude returned no qualified candidates. This can mean hard-filter disqualification or unusable model output.",
                flush=True,
            )
            log(
                "[AgenticSearch] Claude returned no qualified candidates. This can mean hard-filter disqualification or unusable model output."
            )
        else:
            print("[DEBUG] No qualified candidates after hard filters.", flush=True)
            log("[AgenticSearch] No qualified candidates after hard filters.")

    from sa_candidate_finder.manatal_candidates import fresh_resume_url
    from urllib.parse import parse_qs, urlparse as _urlparse

    def _url_is_fresh(url: str) -> bool:
        if not url:
            return False
        try:
            expires = parse_qs(_urlparse(url).query).get("Expires", [])
            if not expires:
                return True
            return int(expires[0]) > int(__import__('time').time()) + 300
        except Exception:
            return True

    display_results = results[:top]
    print(f"[DEBUG] Top {len(display_results)} candidates:", flush=True)
    log(f"[AgenticSearch] Top {len(display_results)} candidates:")
    for r in display_results:
        c = r.candidate
        matched = ", ".join(r.strengths[:5]) if r.strengths else "none"
        # Lazily refresh URL only at output time, only if expired
        if not _url_is_fresh(c.resume_url):
            fresh = fresh_resume_url(MANATAL_API_KEY, str(c.id))
            if fresh:
                c.resume_url = fresh
        resume_link = c.resume_url or "[no resume url]"
        tier_str = f" | Tier: {r.tier}" if r.tier else ""
        dimension_parts = [
            f"Experience {r.__dict__.get('experience_score', 0.0):.1f}/30",
            f"Skills {r.__dict__.get('skills_score', 0.0):.1f}/25",
            f"Location {r.__dict__.get('location_score', 0.0):.1f}/20",
            f"English/AI {r.__dict__.get('english_ai_score', 0.0):.1f}/15",
            f"Salary {r.__dict__.get('salary_score', 0.0):.1f}/10",
        ]
        risks = "; ".join(r.risks[:3]) if r.risks else "none"
        print(
            f"[DEBUG] [{r.rank}] {c.name or '[no name]'} | {c.current_position} | "
            f"Score: {r.fit_score}/10{tier_str} | Matched: {matched} | Resume: {resume_link}",
            flush=True,
        )
        print(
            f"[DEBUG]     Why this tier: {' | '.join(dimension_parts)} | Risks: {risks}",
            flush=True,
        )
        print(f"[DEBUG]     Rationale: {r.rationale}", flush=True)
        log(
            f"[{r.rank}] {c.name or '[no name]'} | {c.current_position} | "
            f"Score: {r.fit_score}/10{tier_str} | Matched: {matched} | Resume: {resume_link}"
        )
        log(f"    Why this tier: {' | '.join(dimension_parts)} | Risks: {risks}")
        log(f"    Rationale: {r.rationale}")

    # ── Persist results as JSON for web UI ───────────────────────────────────
    import json as _json
    from datetime import datetime, timezone as _tz
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    role_key = _re.sub(r"[^a-z0-9]+", "_", _normalize_role(role_name)).strip("_")
    json_path = results_dir / f"role_{role_key}.json"
    # llm_result_by_id already built above (includes both qualified + disqualified results)
    payload = {
        "role_name": role_name,
        "anchor_job_id": str(job_id),
        "all_job_ids": all_role_job_ids,
        "total_candidates_in_pool": len(all_role_candidates),
        "phase1_candidate_ids": [str(c.id) for c in all_role_candidates],
        "total_ranked_candidates": len(full_ranked_results),
        "llm_evaluated_candidates": len(llm_reviewed_ids),
        "llm_qualified_candidates": len(llm_qualified_ids),
        "llm_eval_policy": {
            "mode": "quality_gate",
            "fast_rerank": rerank_fast,
            "global_pool_expanded": phase2_expanded,
            "good_match_threshold": effective_good_match_threshold,
            "base_good_match_threshold": cfg.min_good_match_score,
            "active_keyword_count": len(keyword_set),
            "target_good_matches": phase2_target,
            "selected_eval_count": llm_eval_limit,
            "skipped_not_reviewed_count": len(not_reviewed_by_id),
            "ranked_pool_count": len(full_ranked_results),
            "good_match_count": llm_eval_limit,
        },
        "hard_filters": [
            {"rule": d.get("rule", ""), "universal": d.get("universal", False)}
            for d in dealbreakers
        ],
        "llm_keywords": llm_keywords,
        "user_keywords": user_keywords or [],
        "required_keywords": required_keyword_list or [],
        "final_keyword_set": list(keyword_set),
        "ran_at": datetime.now(_tz.utc).isoformat(),
        "candidates": [
            {
                "rank": r.rank,
                "id": str(r.candidate.id),
                "name": r.candidate.name,
                "current_position": r.candidate.current_position,
                "current_company": r.candidate.current_company,
                "location": r.candidate.location,
                "fit_score": round(llm_result_by_id[str(r.candidate.id)].fit_score, 2) if str(r.candidate.id) in llm_result_by_id else None,
                "keyword_score": round(r.fit_score, 2),
                "matched_keywords": matched_keywords_by_id.get(str(r.candidate.id), []),
                "tier": (
                    (llm_result_by_id[str(r.candidate.id)].tier or "")
                    if str(r.candidate.id) in llm_result_by_id and str(r.candidate.id) in llm_qualified_ids
                    else (
                        "D"
                        if str(r.candidate.id) in llm_reviewed_ids
                        and str(r.candidate.id) not in llm_qualified_ids
                        and (
                            llm_disqualify_reasons.get(str(r.candidate.id), "")
                            or (
                                llm_result_by_id[str(r.candidate.id)].disqualify_reason
                                if str(r.candidate.id) in llm_result_by_id
                                else ""
                            )
                        )
                        else ""
                    )
                ),
                "strengths": llm_result_by_id[str(r.candidate.id)].strengths if str(r.candidate.id) in llm_result_by_id else [],
                "risks": llm_result_by_id[str(r.candidate.id)].risks if str(r.candidate.id) in llm_result_by_id else [],
                "rationale": llm_result_by_id[str(r.candidate.id)].rationale if str(r.candidate.id) in llm_result_by_id else "",
                "resume_url": r.candidate.resume_url or "",
                "manatal_stage": r.candidate.manatal_stage or "",
                "experience_score": round(llm_result_by_id[str(r.candidate.id)].__dict__.get("experience_score", 0.0), 1) if str(r.candidate.id) in llm_result_by_id else None,
                "skills_score": round(llm_result_by_id[str(r.candidate.id)].__dict__.get("skills_score", 0.0), 1) if str(r.candidate.id) in llm_result_by_id else None,
                "location_score": round(llm_result_by_id[str(r.candidate.id)].__dict__.get("location_score", 0.0), 1) if str(r.candidate.id) in llm_result_by_id else None,
                "english_ai_score": round(llm_result_by_id[str(r.candidate.id)].__dict__.get("english_ai_score", 0.0), 1) if str(r.candidate.id) in llm_result_by_id else None,
                "salary_score": round(llm_result_by_id[str(r.candidate.id)].__dict__.get("salary_score", 0.0), 1) if str(r.candidate.id) in llm_result_by_id else None,
                "llm_reviewed": str(r.candidate.id) in llm_reviewed_ids,
                "llm_qualified": str(r.candidate.id) in llm_qualified_ids,
                "disqualify_reason": (
                    ""
                    if str(r.candidate.id) in not_reviewed_by_id
                    else (
                        llm_disqualify_reasons.get(str(r.candidate.id), "")
                        or (llm_result_by_id[str(r.candidate.id)].disqualify_reason if str(r.candidate.id) in llm_result_by_id else "")
                    )
                ),
                "not_reviewed_reason": (
                    not_reviewed_by_id.get(str(r.candidate.id), "")
                    or (
                        "The AI evaluator did not return a result for this candidate — likely omitted due to batch size. Re-rank to retry."
                        if str(r.candidate.id) in llm_reviewed_ids
                        and str(r.candidate.id) not in llm_qualified_ids
                        and not llm_disqualify_reasons.get(str(r.candidate.id))
                        and not (str(r.candidate.id) in llm_result_by_id and llm_result_by_id[str(r.candidate.id)].disqualify_reason)
                        else ""
                    )
                ),
            }
            for r in full_ranked_results
        ],
        "top_candidates": [
            {
                "rank": r.rank,
                "id": str(r.candidate.id),
                "name": r.candidate.name,
                "current_position": r.candidate.current_position,
                "current_company": r.candidate.current_company,
                "location": r.candidate.location,
                "fit_score": round(r.fit_score, 2),
                "tier": r.tier or "",
                "strengths": r.strengths,
                "risks": r.risks,
                "rationale": r.rationale,
                "resume_url": r.candidate.resume_url or "",
                "manatal_stage": r.candidate.manatal_stage or "",
                "experience_score": round(r.__dict__.get("experience_score", 0.0), 1),
                "skills_score": round(r.__dict__.get("skills_score", 0.0), 1),
                "location_score": round(r.__dict__.get("location_score", 0.0), 1),
                "english_ai_score": round(r.__dict__.get("english_ai_score", 0.0), 1),
                "salary_score": round(r.__dict__.get("salary_score", 0.0), 1),
            }
            for r in display_results
        ],
    }
    json_path.write_text(_json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Results] Saved JSON results -> {json_path}", flush=True)
    log(f"[AgenticSearch] JSON results saved to {json_path}")


@app.command("feedback")
def feedback_cmd(
    jd: Path = typer.Option(..., "--jd", help="Path to job description file used in the search run."),
) -> None:
    """Provide feedback on the most recent search run for a given JD."""
    from sa_candidate_finder.feedback import run_feedback

    if not jd.exists():
        typer.echo(f"Error: JD file not found: {jd}")
        raise typer.Exit(1)

    run_feedback(jd_path=jd)


@app.command("learn")
def learn_cmd() -> None:
    """Apply accumulated feedback to refine future searches."""
    from sa_candidate_finder.learn import run_learn

    run_learn()


if __name__ == "__main__":
    app()
