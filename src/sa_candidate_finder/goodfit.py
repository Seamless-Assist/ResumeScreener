import re
import time
from typing import Optional

import httpx

GOODFIT_BASE_URL = "https://horizon.api.goodfit.studio/api/v1"
GOODFIT_APPLY_BASE = "https://app.goodfit.io/jobs"
_JOBS_CACHE_TTL = 3600  # 1 hour

_jobs_cache: list[dict] = []
_jobs_cache_at: float = 0.0


def _headers(api_key: str) -> dict:
    return {
        "x-api-key": api_key,
        "accept": "application/json",
        "content-type": "application/json",
    }


def _normalize_title(s: str) -> str:
    s = re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower().strip())
    return " ".join(s.split())


def list_goodfit_jobs(api_key: str, force_refresh: bool = False) -> list[dict]:
    """Return all active Goodfit jobs, cached in memory for 1 hour."""
    global _jobs_cache, _jobs_cache_at
    if not force_refresh and time.time() - _jobs_cache_at < _JOBS_CACHE_TTL and _jobs_cache:
        return _jobs_cache

    all_jobs: list[dict] = []
    page = 1
    while True:
        resp = httpx.get(
            f"{GOODFIT_BASE_URL}/jobs",
            headers=_headers(api_key),
            params={"page": page, "limit": 100, "status": "active"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        jobs = data.get("data", [])
        all_jobs.extend(jobs)
        if not data.get("pagination", {}).get("hasMore"):
            break
        page += 1

    _jobs_cache = all_jobs
    _jobs_cache_at = time.time()
    return all_jobs


def find_goodfit_job_by_title(api_key: str, role_title: str) -> Optional[dict]:
    """Find a Goodfit job matching the given role title (case-insensitive, normalized).

    Tries exact normalized match first, then substring match.
    """
    norm = _normalize_title(role_title)
    jobs = list_goodfit_jobs(api_key)
    for job in jobs:
        if _normalize_title(job.get("title", "")) == norm:
            return job
    for job in jobs:
        jt = _normalize_title(job.get("title", ""))
        if norm in jt or jt in norm:
            return job
    return None


def find_application_by_email(api_key: str, goodfit_job_id: str, candidate_email: str) -> Optional[dict]:
    """Search job applications for one matching candidate_email.

    Returns the application dict or None if not found.
    """
    norm_email = (candidate_email or "").strip().lower()
    page = 1
    while True:
        try:
            resp = httpx.get(
                f"{GOODFIT_BASE_URL}/applications",
                headers=_headers(api_key),
                params={"jobId": goodfit_job_id, "page": page, "limit": 100},
                timeout=15,
            )
            resp.raise_for_status()
        except Exception:
            break
        data = resp.json()
        items = data.get("data", [])
        for item in items:
            if isinstance(item, dict):
                app_email = (item.get("email") or item.get("candidateEmail") or "").strip().lower()
                if app_email == norm_email:
                    return item
        if not data.get("pagination", {}).get("hasMore"):
            break
        page += 1
    return None


def send_interview_invite(
    api_key: str,
    goodfit_job_id: str,
    candidate_name: str,
    candidate_email: str,
) -> dict:
    """Create a Goodfit application and send the candidate a magic-link invite email.

    Returns the raw application data dict from the API response.
    On 409 Conflict (application already exists), looks up and returns the existing application.
    """
    resp = httpx.post(
        f"{GOODFIT_BASE_URL}/applications",
        headers=_headers(api_key),
        json={
            "jobId": goodfit_job_id,
            "email": candidate_email,
            "name": candidate_name,
            "source": "individual_invite",
            "sendInvite": True,
        },
        timeout=15,
    )

    if resp.status_code == 409:
        # Application already exists for this email+job — look it up and return it.
        print(f"[Goodfit] 409 for {candidate_email} on job {goodfit_job_id} — fetching existing application.", flush=True)
        existing = find_application_by_email(api_key, goodfit_job_id, candidate_email)
        if existing:
            return existing
        # If the search also fails, surface a clear error instead of the raw 409.
        raise RuntimeError(
            f"A Goodfit application already exists for {candidate_email} on this job, "
            "but it could not be retrieved. Check the Goodfit dashboard."
        )

    resp.raise_for_status()
    return resp.json().get("data", {})


def get_application(api_key: str, application_id: str) -> dict:
    """Return full application details including current status."""
    resp = httpx.get(
        f"{GOODFIT_BASE_URL}/applications/{application_id}",
        headers=_headers(api_key),
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json().get("data", {})


def application_exists(api_key: str, application_id: str) -> bool:
    """Return True if the Goodfit application still exists (not deleted)."""
    try:
        resp = httpx.get(
            f"{GOODFIT_BASE_URL}/applications/{application_id}",
            headers=_headers(api_key),
            timeout=10,
        )
        return resp.status_code == 200
    except Exception:
        return False


def resend_invite(api_key: str, application_id: str) -> dict:
    """Resend the invite email for an existing Goodfit application."""
    resp = httpx.post(
        f"{GOODFIT_BASE_URL}/applications/{application_id}/resend-invite",
        headers=_headers(api_key),
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json().get("data", {})
