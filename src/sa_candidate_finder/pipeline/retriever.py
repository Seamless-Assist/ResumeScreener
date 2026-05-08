from __future__ import annotations

import time

import httpx
import typer

from sa_candidate_finder.config import Config
from sa_candidate_finder.mcp_client import call_tool
from sa_candidate_finder.models import CandidateMeta, RunTelemetry


def fetch_resumes(
    candidates: list[CandidateMeta],
    cfg: Config,
    telemetry: RunTelemetry,
) -> list[CandidateMeta]:
    """Fetch full candidate profiles and parse resume PDFs. Skips on failure."""
    fetched: list[CandidateMeta] = []

    for c in candidates:
        try:
            resume_url = _get_resume_url(c.id, cfg)
            c.resume_url = resume_url
            c.resume_text = _fetch_and_parse_pdf(resume_url, cfg)
            fetched.append(c)
        except Exception as exc:
            telemetry.errors.append({
                "stage": "retriever",
                "candidate_id": c.id,
                "error": str(exc),
            })
            typer.echo(f"  [Error] Skipping candidate {c.id} ({c.name}): {exc}")

    telemetry.llm_evaluated_count = len(fetched)
    return fetched


def _get_resume_url(candidate_id: str, cfg: Config) -> str:
    data = call_tool(name="candidates_read", arguments={"id": int(candidate_id)}, cfg=cfg, request_id=3)
    resume_url = data.get("resume") or data.get("resume_url") or data.get("cv_url", "")
    if not resume_url:
        raise ValueError(f"No resume URL in candidates_read response for candidate {candidate_id}")
    return resume_url


def _fetch_and_parse_pdf(resume_url: str, cfg: Config) -> str:
    import fitz  # PyMuPDF

    for attempt in range(cfg.max_retries + 1):
        try:
            resp = httpx.get(resume_url, timeout=cfg.mcp_timeout_seconds, follow_redirects=True)
            resp.raise_for_status()
            doc = fitz.open(stream=resp.content, filetype="pdf")
            return "\n".join(page.get_text() for page in doc)
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            if attempt == cfg.max_retries:
                raise
            wait = (2**attempt) if cfg.retry_backoff == "exponential" else 1
            time.sleep(wait)

    raise RuntimeError(f"Failed to fetch/parse resume: {resume_url}")
