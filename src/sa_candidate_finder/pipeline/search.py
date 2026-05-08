from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path

import typer

from sa_candidate_finder.config import load_config
from sa_candidate_finder.models import RunTelemetry
from sa_candidate_finder.pipeline.extractor import extract_constraints
from sa_candidate_finder.pipeline.mcp_filter import filter_candidates
from sa_candidate_finder.pipeline.embedder import rank_candidates
from sa_candidate_finder.pipeline.retriever import fetch_resumes
from sa_candidate_finder.pipeline.evaluator import evaluate_candidates
from sa_candidate_finder.telemetry import save_telemetry
from sa_candidate_finder.output import write_results


def run_search(jd_path: Path, top_x: int) -> None:
    cfg = load_config()
    cfg.top_x = top_x

    jd_text = _read_jd(jd_path)
    jd_hash = hashlib.sha256(jd_text.encode()).hexdigest()[:16]
    run_id = str(uuid.uuid4())[:8]
    started_at = datetime.now(timezone.utc).isoformat()

    telemetry = RunTelemetry(
        run_id=run_id,
        jd_hash=jd_hash,
        started_at=started_at,
        embedding_model=cfg.embedding_model,
        llm_model=cfg.llm_model,
    )

    typer.echo(f"\n[SACandidateFinder] Run ID: {run_id}")
    typer.echo(f"[SACandidateFinder] JD: {jd_path}")

    # Stage 1: Extract and confirm keywords
    typer.echo("\n[1/5] Extracting keywords from JD...")
    hard_constraints = extract_constraints(jd_text, cfg, telemetry)
    dealbreakers = hard_constraints.get("dealbreakers", []) if isinstance(hard_constraints, dict) else []

    # Stage 2: MCP filter
    typer.echo("\n[2/5] Filtering candidates via Manatal MCP...")
    candidates = filter_candidates(hard_constraints, cfg, telemetry)
    typer.echo(f"      {telemetry.post_filter_count} candidates after filter.")

    if len(candidates) == 0:
        typer.echo("\n[SACandidateFinder] No candidates matched the constraints. Aborting.")
        telemetry.final_returned_count = 0
        telemetry.finished_at = datetime.now(timezone.utc).isoformat()
        telemetry.status = "no_candidates"
        output_path = write_results(
            [],
            jd_path,
            run_id,
            hard_constraints,
            cfg,
            evaluated_count=telemetry.post_filter_count,
        )
        save_telemetry(telemetry, cfg)
        typer.echo(f"[SACandidateFinder] Results saved to: {output_path}\n")
        return

    # Stage 3: Embedding + ranking
    typer.echo("\n[3/5] Embedding and ranking candidates...")
    ranked = rank_candidates(jd_text, candidates, cfg, telemetry)
    shortlist = ranked[: cfg.shortlist_size]
    typer.echo(f"      Cache hits: {telemetry.cache_hits}  Misses: {telemetry.cache_misses}")

    # Stage 4: Fetch resumes
    typer.echo(f"\n[4/5] Fetching profiles and resumes for top {len(shortlist)} candidates...")
    shortlist = fetch_resumes(shortlist, cfg, telemetry)

    # Stage 5: LLM evaluation
    typer.echo(f"\n[5/5] Running LLM evaluation on {len(shortlist)} candidates...")
    results = evaluate_candidates(jd_text, shortlist, [], cfg, telemetry, dealbreakers=dealbreakers)
    final = results[: cfg.top_x]

    # Output
    telemetry.final_returned_count = len(final)
    telemetry.finished_at = datetime.now(timezone.utc).isoformat()
    telemetry.status = "success"

    output_path = write_results(
        final,
        jd_path,
        run_id,
        hard_constraints,
        cfg,
        evaluated_count=telemetry.post_filter_count,
    )
    save_telemetry(telemetry, cfg)

    typer.echo(f"\n[SACandidateFinder] Done. {len(final)} candidates returned.")
    typer.echo(f"[SACandidateFinder] Results saved to: {output_path}\n")

    # Print summary to console
    for r in final:
        typer.echo(
            f"  #{r.rank}  {r.candidate.name}  —  Score: {r.fit_score}/10\n"
            f"       {r.candidate.current_position} @ {r.candidate.current_company}\n"
            f"       Resume: {r.candidate.resume_url}\n"
        )


def _read_jd(jd_path: Path) -> str:
    suffix = jd_path.suffix.lower()
    if suffix == ".pdf":
        import fitz  # PyMuPDF
        doc = fitz.open(str(jd_path))
        return "\n".join(page.get_text() for page in doc)
    return jd_path.read_text(encoding="utf-8")
