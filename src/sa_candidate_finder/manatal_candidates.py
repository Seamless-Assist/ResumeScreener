
import os
import json
import httpx
import time
from io import BytesIO
from urllib.parse import parse_qs, urlparse
from typing import Any, Dict, List, Optional
from sa_candidate_finder.models import CandidateMeta
from sa_candidate_finder.manatal_pagination import has_next_page


# ---------------------------------------------------------------------------
# Resume PDF disk cache  (cache/resumes/candidate_<id>.pdf)
# ---------------------------------------------------------------------------
_RESUME_PDF_CACHE_DIR = os.path.join(os.path.dirname(__file__), '../../cache/resumes')
os.makedirs(_RESUME_PDF_CACHE_DIR, exist_ok=True)


def _resume_pdf_cache_path(candidate_id) -> str:
    return os.path.join(_RESUME_PDF_CACHE_DIR, f"candidate_{candidate_id}.pdf")


def load_cached_resume_pdf(candidate_id) -> Optional[bytes]:
    """Return locally cached PDF bytes or None if not cached."""
    p = _resume_pdf_cache_path(candidate_id)
    try:
        if os.path.exists(p) and os.path.getsize(p) > 0:
            with open(p, 'rb') as f:
                return f.read()
    except Exception:
        pass
    return None


def save_resume_pdf(candidate_id, pdf_bytes: bytes) -> None:
    """Save PDF bytes to disk cache. Only saves actual PDF files (must start with %PDF-)."""
    if not pdf_bytes or not pdf_bytes.startswith(b'%PDF-'):
        return
    p = _resume_pdf_cache_path(candidate_id)
    try:
        with open(p, 'wb') as f:
            f.write(pdf_bytes)
    except Exception:
        pass


def _looks_like_pdf_stream_text(text: str) -> bool:
    if not text:
        return False
    head = text[:256]
    if "%PDF-" in head:
        return True
    markers = ("/FlateDecode", "endobj", "stream", "xref", "trailer")
    marker_hits = sum(1 for m in markers if m in text[:2000])
    return marker_hits >= 3


def _looks_like_meaningful_text(text: str) -> bool:
    if not text:
        return False
    letters = sum(ch.isalpha() for ch in text[:4000])
    return letters >= 120


def _extract_resume_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    extracted = ""

    # Pass 1: PyMuPDF text extraction
    try:
        import fitz

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        extracted = "\n".join(page.get_text("text") for page in doc).strip()
        if _looks_like_meaningful_text(extracted):
            return extracted
    except Exception:
        pass

    # Pass 2: pypdf text extraction (different parser behavior)
    try:
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(pdf_bytes))
        pypdf_text = "\n".join((page.extract_text() or "") for page in reader.pages).strip()
        if len(pypdf_text) > len(extracted):
            extracted = pypdf_text
        if _looks_like_meaningful_text(extracted):
            return extracted
    except Exception:
        pass

    # Pass 3: OCR first few pages if Tesseract is available locally
    try:
        import fitz
        import pytesseract
        from PIL import Image

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        ocr_chunks: list[str] = []
        for idx, page in enumerate(doc):
            if idx >= 3:
                break
            pix = page.get_pixmap(matrix=fitz.Matrix(1.8, 1.8), alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            ocr_chunks.append(pytesseract.image_to_string(img))
        ocr_text = "\n".join(ocr_chunks).strip()
        if len(ocr_text) > len(extracted):
            extracted = ocr_text
    except Exception:
        pass

    return extracted.strip()


def _sanitize_resume_text(resume_text: str) -> str:
    rt = (resume_text or "").strip()
    if rt.startswith("__RESUME_PARSE_FAILED__"):
        return "__RESUME_PARSE_FAILED__"
    if not rt:
        return ""
    # If cached/fetched text is actually raw PDF stream bytes, do not use it for evaluation.
    if _looks_like_pdf_stream_text(rt):
        return "__RESUME_PARSE_FAILED__"
    return rt

# Only candidates in these stages are eligible for ranking.
# Candidates in any other named stage (Goodfit, hired, phone screen, etc.) are excluded.
# Candidates with no stage set are also included (newly added, not yet staged).
_ALLOWED_STAGES: set[str] = {
    "new candidates",
    "filtered resume",
    "goodfit interview sent",
    "goodfit interview approved",
    "goodfit interview failed",
}


def _normalize_stage_name(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


# Synthetic stage name written into snapshots/caches when a Manatal /matches
# record is dropped (recruiter clicked "Drop" on the candidate for this job).
# Falls outside _ALLOWED_STAGES so is_excluded_stage() naturally excludes it.
DROPPED_STAGE_LABEL = "Dropped"


def is_dropped_match(match: Any) -> bool:
    """Return True if a raw Manatal /matches record represents a dropped candidate.

    A match is "dropped" when the recruiter clicked Drop on it. Manatal sets
    `dropped_at` to an ISO timestamp and `is_active` to False. Treat either signal
    as authoritative — both are written together by Manatal, but checking both
    guards against future API changes.
    """
    if not isinstance(match, dict):
        return False
    if match.get("dropped_at"):
        return True
    if match.get("is_active") is False:
        return True
    return False


def is_excluded_stage(value: Any) -> bool:
    """Return True if the candidate should be excluded from ranking.

    Candidates are included only when their stage is blank (not yet set) or
    matches one of the allowed stages exactly. The synthetic "Dropped" label
    written when a Manatal /matches record is dropped naturally falls outside
    _ALLOWED_STAGES.
    """
    normalized = _normalize_stage_name(value)
    if not normalized:
        return False  # no stage set — include (newly added candidates)
    return normalized not in _ALLOWED_STAGES


_STAGE_SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), '../../cache/stage_snapshots')
_STAGE_ID_CACHE_PATH = os.path.join(os.path.dirname(__file__), '../../cache/manatal_stage_ids.json')
_APPLIED_CACHE_DIR = os.path.join(os.path.dirname(__file__), '../../cache/applied_candidates')


def _load_stage_id_cache() -> Dict[str, int]:
    """Return {normalized_stage_name: stage_id} accumulated from prior syncs."""
    try:
        if os.path.exists(_STAGE_ID_CACHE_PATH):
            return json.load(open(_STAGE_ID_CACHE_PATH, encoding="utf-8"))
    except Exception:
        pass
    return {}


def _update_stage_id_cache(stage_obj: dict) -> None:
    """Merge one stage object {id, name} into the persistent cache."""
    if not isinstance(stage_obj, dict):
        return
    sid = stage_obj.get("id")
    name = stage_obj.get("name", "")
    if sid is None or not name:
        return
    norm = " ".join(str(name).strip().lower().split())
    try:
        cache = _load_stage_id_cache()
        if cache.get(norm) != sid:
            cache[norm] = sid
            os.makedirs(os.path.dirname(_STAGE_ID_CACHE_PATH), exist_ok=True)
            with open(_STAGE_ID_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(cache, f)
    except Exception:
        pass


def load_role_stage_snapshot(role_id: str) -> Dict[str, str]:
    """Read the most recent stage snapshot for a role from disk (no API call).

    Returns {candidate_id: stage_name} or empty dict if no snapshot exists.
    """
    os.makedirs(_STAGE_SNAPSHOT_DIR, exist_ok=True)
    path = os.path.join(_STAGE_SNAPSHOT_DIR, f"role_{role_id}.json")
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                snap = json.load(f)
            return snap.get('stages', {})
    except Exception:
        pass
    return {}


def is_full_sync_snapshot(role_id: str) -> bool:
    """Return True if the stage snapshot was written by a full sync_role_stages call.

    Partial snapshots (written by individual update_stage_snapshot_entry calls when
    Goodfit invites are sent) are not authoritative enough to use as a membership filter.
    """
    path = os.path.join(_STAGE_SNAPSHOT_DIR, f"role_{role_id}.json")
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                snap = json.load(f)
            return snap.get('source') == 'full_sync'
    except Exception:
        pass
    return False


def sync_role_stages(api_token: str, role_id: str, job_ids: List[str]) -> Dict[str, str]:
    """Fetch current pipeline stages for all candidates in the role from Manatal.

    Writes the result to a per-role snapshot file and returns {candidate_id: stage_name}.
    Throttles requests to avoid 429 rate limits.
    """
    os.makedirs(_STAGE_SNAPSHOT_DIR, exist_ok=True)
    headers = {"Authorization": f"Token {api_token}", "accept": "application/json"}
    stages: Dict[str, str] = {}
    _last_req = [0.0]

    def _throttle(min_gap: float = 0.7) -> None:
        elapsed = time.time() - _last_req[0]
        if elapsed < min_gap:
            time.sleep(min_gap - elapsed)
        _last_req[0] = time.time()

    for job_id in job_ids:
        page = 1
        while True:
            _throttle()
            try:
                resp = httpx.get(
                    f"https://api.manatal.com/open/v3/jobs/{job_id}/matches/",
                    headers=headers,
                    params={"page": page, "page_size": 100},
                    timeout=20,
                )
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", "10"))
                    print(f"[SyncStages] 429 on job {job_id} — waiting {retry_after}s", flush=True)
                    time.sleep(retry_after)
                    continue
                resp.raise_for_status()
            except Exception as e:
                print(f"[SyncStages] Error fetching job {job_id}: {e}", flush=True)
                break

            data = resp.json()
            results = data.get("results") or data.get("data") or []
            for match in results:
                if not isinstance(match, dict):
                    continue
                cand = match.get("candidate") or {}
                cand_id = cand.get("id") if isinstance(cand, dict) else cand
                stage_obj = match.get("stage") or match.get("job_pipeline_stage") or {}
                if isinstance(stage_obj, dict) and stage_obj.get("id"):
                    _update_stage_id_cache(stage_obj)
                stage_name = stage_obj.get("name", "") if isinstance(stage_obj, dict) else ""
                if is_dropped_match(match):
                    stage_name = DROPPED_STAGE_LABEL
                if cand_id is not None:
                    stages[str(cand_id)] = str(stage_name)
            if not has_next_page(data, results, 100):
                break
            page += 1

    path = os.path.join(_STAGE_SNAPSHOT_DIR, f"role_{role_id}.json")
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'fetched_at': time.time(), 'source': 'full_sync', 'stages': stages}, f)
    except Exception as e:
        print(f"[SyncStages] Failed to write snapshot: {e}", flush=True)

    return stages


def invalidate_stage_snapshot(role_id: str) -> None:
    """Delete the cached stage snapshot for a role so the next sync fetches fresh data."""
    path = os.path.join(_STAGE_SNAPSHOT_DIR, f"role_{role_id}.json")
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        print(f"[SyncStages] Could not invalidate snapshot for role {role_id}: {e}", flush=True)


def invalidate_applied_candidate_cache(job_id: str) -> None:
    """Delete the applied-candidate ID cache for a job.

    Forces the next fetch_candidates_by_job call to repaginate Manatal /matches
    instead of trusting the 7-day cached ID list. Use before a re-rank when the
    rerank pool must include candidates who applied since the cache was written.
    """
    path = os.path.join(_APPLIED_CACHE_DIR, f"job_{job_id}.json")
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        print(f"[Cache] Could not invalidate applied cache for job {job_id}: {e}", flush=True)


def update_stage_snapshot_entry(role_id: str, candidate_id: str, stage_name: str) -> None:
    """Update a single candidate's stage in the role snapshot without clearing other entries.

    If no snapshot exists yet, creates one with just this entry so the candidate is
    filtered out immediately on the next page load without requiring a full Sync Stages.
    """
    os.makedirs(_STAGE_SNAPSHOT_DIR, exist_ok=True)
    path = os.path.join(_STAGE_SNAPSHOT_DIR, f"role_{role_id}.json")
    try:
        snap: dict = {}
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                snap = json.load(f)
        stages: dict = snap.get('stages', {})
        stages[str(candidate_id)] = str(stage_name)
        snap['stages'] = stages
        snap['fetched_at'] = snap.get('fetched_at', time.time())
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(snap, f)
    except Exception as e:
        print(f"[SyncStages] Could not update snapshot entry for role {role_id}: {e}", flush=True)


def fetch_candidates_by_job(api_token: str, job_id: str, page_size: int = 100) -> List[CandidateMeta]:
    """
    Fetch candidates who have applied to a specific job from Manatal.
    """
    _is_excluded_stage = is_excluded_stage

    filtered_status_total = 0
    filtered_status_breakdown: dict[str, int] = {}

    base_url = f"https://api.manatal.com/open/v3/jobs/{job_id}/matches/"
    headers = {
        "Authorization": f"Token {api_token}",
        "accept": "application/json",
    }
    candidates = []
    page = 1
    # Applied candidates cache
    os.makedirs(_APPLIED_CACHE_DIR, exist_ok=True)
    applied_cache_path = os.path.join(_APPLIED_CACHE_DIR, f"job_{job_id}.json")
    # Try to load applied candidate IDs from cache (expires after 7 days)
    _APPLIED_CACHE_TTL = 7 * 24 * 3600
    applied_candidate_ids = None
    applied_candidate_stages: dict[str, str] = {}
    if os.path.exists(applied_cache_path):
        try:
            with open(applied_cache_path, 'r', encoding='utf-8') as f:
                _payload = json.load(f)
            # Support both old format (plain list) and new format (dict with cached_at)
            if isinstance(_payload, list):
                _ids = _payload
                _age = float('inf')  # old format has no timestamp — treat as expired
                _stages = {}
            else:
                _ids = _payload.get('ids', [])
                _age = time.time() - _payload.get('cached_at', 0)
                _stages = _payload.get('stages', {}) if isinstance(_payload.get('stages', {}), dict) else {}
            if _age < _APPLIED_CACHE_TTL:
                applied_candidate_ids = _ids
                applied_candidate_stages = {str(k): str(v) for k, v in _stages.items() if v}
                _days = int(_age // 3600 / 24)
                print(f"[Cache] Loaded applied candidate IDs for job {job_id} from cache (age: {_days}d).", flush=True)
                if applied_candidate_ids and not applied_candidate_stages:
                    # Legacy cache payloads stored only IDs, not stage metadata.
                    # Force a /matches refresh so Manatal stage can be shown in UI.
                    print(
                        f"[Cache] Applied candidate cache for job {job_id} has no stage metadata - refreshing from API.",
                        flush=True,
                    )
                    applied_candidate_ids = None
            else:
                print(f"[Cache] Applied candidate ID cache for job {job_id} is older than 7 days - will refresh from API.", flush=True)
        except Exception as e:
            print(f"[Cache] Failed to load applied candidate cache for job {job_id}: {e}", flush=True)
    import threading
    last_request_time = [0.0]
    base_min_interval = 0.6
    dynamic_min_interval = [base_min_interval]
    lock = threading.Lock()
    def throttle():
        with lock:
            import time as _time
            now = _time.time()
            elapsed = now - last_request_time[0]
            min_interval = dynamic_min_interval[0]
            if elapsed < min_interval:
                _time.sleep(min_interval - elapsed)
            last_request_time[0] = _time.time()

    # Removed redundant import of os, json, time
    CACHE_DIR = os.path.join(os.path.dirname(__file__), '../../cache/candidates')
    os.makedirs(CACHE_DIR, exist_ok=True)
    CACHE_EXPIRY = float('inf')  # TTL enforced per-candidate via CANDIDATE_CACHE_TTL
    CANDIDATE_CACHE_TTL = 30 * 24 * 3600  # 30 days
    PARSE_FAILURE_RETRY_INTERVAL = 14 * 24 * 3600  # re-try failed resume parsing every 14 days
    MAX_PARSE_FAILURE_REFRESH_PER_JOB = 4  # cap parse-failed API refreshes per job run
    parse_failure_refresh_budget = [MAX_PARSE_FAILURE_REFRESH_PER_JOB]
    parse_failure_refreshes = [0]
    parse_failure_cooldown_skips = [0]
    parse_failure_budget_skips = [0]

    def cache_path(candidate_id):
        return os.path.join(CACHE_DIR, f"candidate_{candidate_id}.json")

    def resume_url_is_fresh(url: str) -> bool:
        if not url:
            return False
        try:
            expires = parse_qs(urlparse(url).query).get("Expires", [])
            if not expires:
                return True
            return int(expires[0]) > int(time.time()) + 300
        except Exception:
            return True

    def write_candidate_cache(candidate_id, data, resume_text, applied_job_ids, manatal_stage: str = ""):
        """Write or update candidate cache, merging with existing fields."""
        path = cache_path(candidate_id)
        cache = {}
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
            except Exception:
                cache = {}
        # Merge resume_text
        if not resume_text:
            resume_text = cache.get('resume_text', '')
        # Merge applied_job_ids
        old_applied = set(cache.get('applied_job_ids', []))
        for jid in applied_job_ids:
            old_applied.add(jid)
        merged_applied = list(old_applied)
        now_ts = time.time()
        parse_failed_at = cache.get('resume_parse_failed_at')
        if str(resume_text).startswith("__RESUME_PARSE_FAILED__"):
            parse_failed_at = now_ts
        elif resume_text:
            parse_failed_at = None
        # Merge manatal_stage: keep most recent non-empty value
        merged_stage = manatal_stage or cache.get('manatal_stage', '')
        # Write
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(
                    {
                        'fetched_at': now_ts,
                        'data': data,
                        'resume_text': resume_text,
                        'applied_job_ids': merged_applied,
                        'resume_parse_failed_at': parse_failed_at,
                        'manatal_stage': merged_stage,
                    },
                    f,
                )
        except Exception as e:
            print(f"[Cache] Could not write cache for {candidate_id}: {e}")

    def fetch_candidate_by_id(candidate_id: int, manatal_stage: str = "") -> Optional[CandidateMeta]:
        path = cache_path(candidate_id)
        applied_job_ids = [job_id] if job_id else []
        # Try cache first
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if not content.strip():
                        raise ValueError("Cache file is empty")
                    cached = json.loads(content)
                if not isinstance(cached, dict):
                    raise ValueError(f"Cache for {candidate_id} is not a dict: {type(cached)}")
                cached_applied = set(cached.get('applied_job_ids', []))
                needs_write = job_id and job_id not in cached_applied
                if job_id:
                    cached_applied.add(job_id)
                age = time.time() - cached.get('fetched_at', 0)
                if age < CANDIDATE_CACHE_TTL:
                    resume_text = _sanitize_resume_text(cached.get('resume_text', ''))
                    # If previous extraction failed, re-fetch candidate to retry resume parsing.
                    # If it fails again, Not Reviewed logic will apply in CLI.
                    if str(resume_text).startswith("__RESUME_PARSE_FAILED__"):
                        parse_failed_at = float(cached.get('resume_parse_failed_at') or cached.get('fetched_at') or 0)
                        age_s = time.time() - parse_failed_at
                        # Avoid hammering the API; retry parse-failed resumes infrequently and with a small per-run budget.
                        if age_s >= PARSE_FAILURE_RETRY_INTERVAL and parse_failure_refresh_budget[0] > 0:
                            parse_failure_refresh_budget[0] -= 1
                            parse_failure_refreshes[0] += 1
                            print(
                                f"[Cache] Candidate {candidate_id} had prior resume parse failure — refreshing from API "
                                f"(budget left: {parse_failure_refresh_budget[0]}).",
                                flush=True,
                            )
                            applied_job_ids = list(cached_applied)
                        else:
                            if age_s < PARSE_FAILURE_RETRY_INTERVAL:
                                parse_failure_cooldown_skips[0] += 1
                            elif parse_failure_refresh_budget[0] <= 0:
                                parse_failure_budget_skips[0] += 1
                            meta = _parse_candidate(cached['data'])
                            meta.resume_text = resume_text or meta.description
                            meta.manatal_stage = manatal_stage or cached.get('manatal_stage', '')
                            if needs_write:
                                write_candidate_cache(
                                    candidate_id,
                                    cached['data'],
                                    cached.get('resume_text', ''),
                                    list(cached_applied),
                                    manatal_stage=meta.manatal_stage,
                                )
                            return meta
                    else:
                        meta = _parse_candidate(cached['data'])
                        meta.resume_text = resume_text or meta.description
                        meta.manatal_stage = manatal_stage or cached.get('manatal_stage', '')
                        if needs_write:
                            write_candidate_cache(
                                candidate_id,
                                cached['data'],
                                cached.get('resume_text', ''),
                                list(cached_applied),
                                manatal_stage=meta.manatal_stage,
                            )
                        # Return cached data for scoring even if URL is stale.
                        # Caller can refresh URL lazily at output time via fresh_resume_url().
                        return meta
                else:
                    print(f"[Cache] Candidate {candidate_id} cache is older than 30 days - refreshing.", flush=True)
                    applied_job_ids = list(cached_applied)
            except Exception as e:
                print(f"[Cache] Failed to load cache for {candidate_id}: {e}")
                try:
                    os.remove(path)
                    print(f"[Cache] Corrupted or empty cache for {candidate_id} deleted. Will refetch from API.", flush=True)
                except Exception as del_e:
                    print(f"[Cache] Could not delete corrupted cache for {candidate_id}: {del_e}", flush=True)
        # Fetch from API
        candidate_url = f"https://api.manatal.com/open/v3/candidates/{candidate_id}/"
        for attempt in range(3):
            try:
                throttle()
                resp = httpx.get(candidate_url, headers=headers, timeout=30)
                if resp.status_code == 429:
                    dynamic_min_interval[0] = min(2.0, dynamic_min_interval[0] + 0.25)
                    retry_after = resp.headers.get("Retry-After", "").strip()
                    wait_s = int(retry_after) if retry_after.isdigit() else min(15, 5 * (attempt + 1))
                    print(f"[Rate Limit] 429 on candidate {candidate_id}. Waiting {wait_s}s (attempt {attempt+1}/3, throttle {dynamic_min_interval[0]:.2f}s).", flush=True)
                    import time as _time
                    _time.sleep(wait_s)
                    continue
                resp.raise_for_status()
                dynamic_min_interval[0] = max(base_min_interval, dynamic_min_interval[0] - 0.05)
                data = resp.json()
                meta = _parse_candidate(data)
                resume_text = ''
                # Try to fetch resume text if resume_url is present
                resume_url = data.get('resume_url') or data.get('resume') or data.get('cv_url')
                if resume_url:
                    try:
                        throttle()
                        r = httpx.get(resume_url, timeout=30, follow_redirects=True)
                        if r.status_code == 200:
                            content_type = (r.headers.get("content-type") or "").lower()
                            is_pdf = (
                                "pdf" in content_type
                                or str(resume_url).lower().endswith(".pdf")
                                or r.content[:5] == b"%PDF-"
                            )
                            try:
                                if is_pdf:
                                    save_resume_pdf(candidate_id, r.content)
                                    resume_text = _extract_resume_text_from_pdf_bytes(r.content)
                                else:
                                    resume_text = r.text.strip()
                            except Exception:
                                if not is_pdf:
                                    resume_text = r.text.strip()
                    except Exception as e:
                        print(f"[Resume] Could not fetch resume for {candidate_id}: {e}")
                resume_text = _sanitize_resume_text(resume_text)
                # Add job_id to applied_job_ids if not present
                if job_id and job_id not in applied_job_ids:
                    applied_job_ids.append(job_id)
                # Save to cache (merge)
                write_candidate_cache(candidate_id, data, resume_text, applied_job_ids, manatal_stage=manatal_stage)
                meta.resume_text = resume_text or meta.description
                if manatal_stage:
                    meta.manatal_stage = manatal_stage
                return meta
            except httpx.RequestError as e:
                wait_s = min(8, 2 * (attempt + 1))
                print(f"[Network] Candidate {candidate_id} request failed: {e}. Retrying in {wait_s}s (attempt {attempt+1}/3).", flush=True)
                import time as _time
                _time.sleep(wait_s)
            except Exception as e:
                print(f"[Error] Fetching candidate {candidate_id}: {e}")
        print(f"[Warning] Could not fetch candidate details for ID: {candidate_id}")
        return None

    if applied_candidate_ids is not None:
        # Only fetch candidate metadata for cached IDs.
        total_cached_ids = len(applied_candidate_ids)
        for idx, candidate_id in enumerate(applied_candidate_ids, start=1):
            stage_hint = applied_candidate_stages.get(str(candidate_id), "")
            if _is_excluded_stage(stage_hint):
                filtered_status_total += 1
                stage_key = _normalize_stage_name(stage_hint)
                filtered_status_breakdown[stage_key] = filtered_status_breakdown.get(stage_key, 0) + 1
                print(
                    f"[Filter] Skipping candidate {candidate_id} for job {job_id} due to status '{stage_hint}'.",
                    flush=True,
                )
                continue
            print(
                f"[AgenticSearch] Job {job_id}: candidate {idx}/{total_cached_ids} (id: {candidate_id})",
                flush=True,
            )
            cand = fetch_candidate_by_id(candidate_id, manatal_stage=stage_hint)
            if cand:
                if stage_hint and not getattr(cand, "manatal_stage", ""):
                    cand.manatal_stage = stage_hint
                candidates.append(cand)
        if parse_failure_refreshes[0] or parse_failure_cooldown_skips[0] or parse_failure_budget_skips[0]:
            print(
                f"[Cache] Resume parse retry summary for job {job_id}: "
                f"refreshed={parse_failure_refreshes[0]}, "
                f"cooldown_skipped={parse_failure_cooldown_skips[0]}, "
                f"budget_skipped={parse_failure_budget_skips[0]}",
                flush=True,
            )
        print(
            f"[Filter] Status exclusion summary for job {job_id}: "
            f"filtered={filtered_status_total}, by_status={filtered_status_breakdown}",
            flush=True,
        )
        print(f"[Cache] Used applied candidate cache for job {job_id}. Returning {len(candidates)} candidates.", flush=True)
        return candidates

    import sys
    from pathlib import Path
    from sa_candidate_finder.config import load_config
    cfg = load_config()
    telemetry_log_path = Path(cfg.telemetry_log_path)
    telemetry_log_path.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    page_count = 0
    results = []  # Always defined
    max_retries = 5
    retry_count = 0
    unique_candidate_ids = set()
    got_api_response = False
    # --- Candidate to job mapping cache ---
    CANDIDATE_JOB_MAP_PATH = os.path.join(os.path.dirname(__file__), '../../cache/candidates/candidate_job_map.json')
    try:
        with open(CANDIDATE_JOB_MAP_PATH, 'r', encoding='utf-8') as f:
            candidate_job_map = json.load(f)
    except Exception:
        candidate_job_map = {}
    stage_by_candidate_id: dict[str, str] = {}
    # If not cached, fetch from API and cache the candidate IDs
    while True:
        params = {"page": page, "page_size": page_size}
        print(f"[DEBUG] Calling API: {base_url}", flush=True)
        print(f"[DEBUG] Params: {params}", flush=True)
        try:
            throttle()
            resp = httpx.get(base_url, headers=headers, params=params, timeout=30)
            resp.raise_for_status()
            retry_count = 0  # Reset on success
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                retry_count += 1
                dynamic_min_interval[0] = min(2.0, dynamic_min_interval[0] + 0.25)
                retry_after = e.response.headers.get("Retry-After", "").strip()
                wait_s = int(retry_after) if retry_after.isdigit() else min(30, 10 * retry_count)
                print(f"\n[Rate Limit] 429 Too Many Requests from Manatal /matches API! Attempt {retry_count}/{max_retries}", flush=True)
                print(f"[Rate Limit] Endpoint: {base_url}", flush=True)
                print(f"[Rate Limit] Params: {params}", flush=True)
                print(f"[Rate Limit] Waiting {wait_s}s before retrying (throttle {dynamic_min_interval[0]:.2f}s)...\n", flush=True)
                import time as _time
                _time.sleep(wait_s)
                if retry_count >= max_retries:
                    print(f"[Rate Limit] Exceeded maximum retries ({max_retries}). Exiting.", flush=True)
                    break
                continue
            print(f"[Error] HTTP {e.response.status_code} for URL: {resp.url}", flush=True)
            print(f"[Error] Response text: {resp.text}", flush=True)
            print(f"[Error] Headers: {resp.headers}", flush=True)
            raise
        except httpx.RequestError as e:
            retry_count += 1
            wait_s = min(20, 5 * retry_count)
            print(f"[Network] /matches request failed: {e}", flush=True)
            print(f"[Network] Waiting {wait_s}s before retrying...", flush=True)
            import time as _time
            _time.sleep(wait_s)
            if retry_count >= max_retries:
                print(f"[Network] Exceeded maximum retries ({max_retries}) for /matches. Exiting.", flush=True)
                break
            continue
        except Exception as e:
            print(f"[Error] Exception during candidate fetch: {e}", flush=True)
            raise
        data = resp.json()
        dynamic_min_interval[0] = max(base_min_interval, dynamic_min_interval[0] - 0.05)
        got_api_response = True
        response_count = data.get("count") if isinstance(data, dict) else None
        if response_count is not None:
            print(f"[DEBUG] /matches response count: {response_count}", flush=True)
            with telemetry_log_path.open("a", encoding="utf-8") as f:
                f.write(f"[DEBUG] /matches response count: {response_count}\n")
        if page == 1:
            # Log the full API response for diagnosis (log file only)
            import pprint
            with telemetry_log_path.open("a", encoding="utf-8") as f:
                f.write("[DEBUG] Full /matches/ API response (page 1):\n" + pprint.pformat(data) + "\n")
        results = data.get("results") or data.get("data") or data
        if not results or not isinstance(results, list):
            break
        print(f"[DEBUG] /matches page {page} results: {len(results)}", flush=True)
        with telemetry_log_path.open("a", encoding="utf-8") as f:
            f.write(f"[DEBUG] /matches page {page} results: {len(results)}\n")
        page_count += 1
        page_start = processed + 1
        page_end = processed + len(results)
        print(f"[AgenticSearch] Processing page {page} ({page_start}-{page_end})...", flush=True)
        with telemetry_log_path.open("a", encoding="utf-8") as f:
            f.write(f"[AgenticSearch] Processing page {page} ({page_start}-{page_end})...\n")
        for c in results:
            _match_stage = ''
            if isinstance(c, dict):
                _stage_obj = c.get('stage') or c.get('job_pipeline_stage') or {}
                if isinstance(_stage_obj, dict):
                    _match_stage = str(_stage_obj.get('name', '') or '')
                if is_dropped_match(c):
                    _match_stage = DROPPED_STAGE_LABEL
            if _is_excluded_stage(_match_stage):
                filtered_status_total += 1
                stage_key = _normalize_stage_name(_match_stage)
                filtered_status_breakdown[stage_key] = filtered_status_breakdown.get(stage_key, 0) + 1
                _candidate_data = c.get("candidate") if isinstance(c, dict) and "candidate" in c else c
                _candidate_id = (
                    _candidate_data.get("id")
                    if isinstance(_candidate_data, dict)
                    else _candidate_data
                    if isinstance(_candidate_data, int)
                    else "unknown"
                )
                print(
                    f"[Filter] Skipping candidate {_candidate_id} for job {job_id} due to status '{_match_stage}'.",
                    flush=True,
                )
                continue
            candidate_data = c.get("candidate") if isinstance(c, dict) and "candidate" in c else c
            candidate_id = None
            if isinstance(candidate_data, dict):
                candidate_id = candidate_data.get('id')
                candidate_job_id = candidate_data.get('job')
                # --- Cache candidate_id <-> job_id mapping ---
                if candidate_id is not None:
                    unique_candidate_ids.add(candidate_id)
                    if _match_stage:
                        stage_by_candidate_id[str(candidate_id)] = _match_stage
                    # Update mapping: candidate_id -> set of job_ids
                    cid_str = str(candidate_id)
                    job_set = set(candidate_job_map.get(cid_str, []))
                    if job_id not in job_set:
                        job_set.add(job_id)
                        candidate_job_map[cid_str] = list(job_set)
                    print(f"[AgenticSearch] Processed candidate ID: {candidate_id} (job: {candidate_job_id})", flush=True)
                    with telemetry_log_path.open("a", encoding="utf-8") as f:
                        f.write(f"[AgenticSearch] Processed candidate ID: {candidate_id} (job: {candidate_job_id})\n")
                meta = _parse_candidate(candidate_data)
                if _match_stage:
                    meta.manatal_stage = _match_stage
                candidates.append(meta)
                # Update cache with applied_job_ids and stage for every candidate (merge fields)
                if candidate_id:
                    write_candidate_cache(candidate_id, candidate_data, '', [job_id] if job_id else [], manatal_stage=_match_stage)
            elif isinstance(candidate_data, int):
                candidate_id = candidate_data
                unique_candidate_ids.add(candidate_id)
                if _match_stage:
                    stage_by_candidate_id[str(candidate_id)] = _match_stage
                cid_str = str(candidate_id)
                job_set = set(candidate_job_map.get(cid_str, []))
                if job_id not in job_set:
                    job_set.add(job_id)
                    candidate_job_map[cid_str] = list(job_set)
                print(f"[AgenticSearch] Processed candidate ID: {candidate_id}", flush=True)
                with telemetry_log_path.open("a", encoding="utf-8") as f:
                    f.write(f"[AgenticSearch] Processed candidate ID: {candidate_id}\n")
                cand = fetch_candidate_by_id(candidate_data, manatal_stage=_match_stage)
                if cand:
                    if _match_stage and not getattr(cand, "manatal_stage", ""):
                        cand.manatal_stage = _match_stage
                    candidates.append(cand)
                # Cache is handled inside fetch_candidate_by_id
            else:
                print(f"[Warning] Skipping candidate entry not a dict or int: {candidate_data}", flush=True)
            processed += 1
            if response_count is not None and int(response_count) > 0:
                print(
                    f"[AgenticSearch] Job {job_id}: candidate {processed}/{int(response_count)}",
                    flush=True,
                )
            msg = f"[AgenticSearch] Progress: {processed} processed."
            print(msg, flush=True)
            with telemetry_log_path.open("a", encoding="utf-8") as f:
                f.write(msg + "\n")
        # Save candidate-job mapping after each page
        try:
            with open(CANDIDATE_JOB_MAP_PATH, 'w', encoding='utf-8') as f:
                json.dump(candidate_job_map, f)
        except Exception as e:
            print(f"[Cache] Failed to save candidate-job map: {e}", flush=True)
        if not has_next_page(data, results, page_size):
            break
        page += 1
    print(f"[AgenticSearch] {len(unique_candidate_ids)} unique candidates have applied to this job.", flush=True)
    with telemetry_log_path.open("a", encoding="utf-8") as f:
        f.write(f"[AgenticSearch] {len(unique_candidate_ids)} unique candidates have applied to this job.\n")
    summary_msg = (
        f"[Filter] Status exclusion summary for job {job_id}: "
        f"filtered={filtered_status_total}, by_status={filtered_status_breakdown}"
    )
    print(summary_msg, flush=True)
    with telemetry_log_path.open("a", encoding="utf-8") as f:
        f.write(summary_msg + "\n")
    # Save applied candidate IDs to cache only when we got at least one successful API response.
    if got_api_response:
        try:
            with open(applied_cache_path, 'w', encoding='utf-8') as f:
                json.dump(
                    {
                        'cached_at': time.time(),
                        'ids': list(unique_candidate_ids),
                        'stages': stage_by_candidate_id,
                    },
                    f,
                )
            print(f"[Cache] Saved applied candidate IDs for job {job_id} to cache.", flush=True)
        except Exception as e:
            print(f"[Cache] Failed to save applied candidate cache for job {job_id}: {e}", flush=True)
    else:
        print(f"[Cache] Skipping applied candidate cache write for job {job_id} because no successful /matches response was received.", flush=True)
    return candidates

def fetch_all_candidates(api_token: str, page_size: int = 100) -> List[CandidateMeta]:
    """
    Fetch all candidates from Manatal.
    """
    base_url = "https://api.manatal.com/open/v3/candidates/"
    headers = {
        "Authorization": f"Token {api_token}",
        "accept": "application/json",
    }
    candidates = []
    import threading
    last_request_time = [0.0]
    lock = threading.Lock()
    def throttle():
        with lock:
            import time as _time
            now = _time.time()
            elapsed = now - last_request_time[0]
            min_interval = 1.2  # ~0.83 TPS — conservative to avoid 429s
            if elapsed < min_interval:
                _time.sleep(min_interval - elapsed)
            last_request_time[0] = _time.time()

    page = 1
    while True:
        params = {"page": page, "page_size": page_size}
        try:
            throttle()
            resp = httpx.get(base_url, headers=headers, params=params, timeout=30)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                retry_after = e.response.headers.get("Retry-After", "").strip()
                wait_s = int(retry_after) if retry_after.isdigit() else 15
                print(f"\n[Rate Limit] 429 Too Many Requests from Manatal /candidates API!", flush=True)
                print(f"[Rate Limit] Endpoint: {base_url}", flush=True)
                print(f"[Rate Limit] Params: {params}", flush=True)
                print(f"[Rate Limit] Waiting {wait_s}s before retrying...\n", flush=True)
                import time as _time
                _time.sleep(wait_s)
                continue
            print(f"[Error] HTTP {e.response.status_code} for URL: {resp.url}", flush=True)
            print(f"[Error] Response text: {resp.text}", flush=True)
            print(f"[Error] Headers: {resp.headers}", flush=True)
            raise
        except Exception as e:
            print(f"[Error] Exception during candidate fetch: {e}")
            raise
        data = resp.json()
        results = data.get("results") or data.get("data") or data
        if not results or not isinstance(results, list):
            break
        for c in results:
            candidates.append(_parse_candidate(c))
        if not has_next_page(data, results, page_size):
            break
        page += 1
    return candidates

def fresh_resume_url(api_token: str, candidate_id: str) -> str:
    """
    Fetch a fresh signed resume URL for a candidate from the Manatal API.
    Only called at output time for top-N results when the cached URL has expired.
    Returns empty string if unable to fetch.
    """
    from urllib.parse import parse_qs, urlparse
    candidate_url = f"https://api.manatal.com/open/v3/candidates/{candidate_id}/"
    headers = {"Authorization": f"Token {api_token}", "accept": "application/json"}
    try:
        resp = httpx.get(candidate_url, headers=headers, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            url = data.get('resume_url') or data.get('resume') or data.get('cv_url') or ''
            if url:
                # Update cached resume_url in the candidate's cache file
                cache_dir = os.path.join(os.path.dirname(__file__), '../../cache/candidates')
                path = os.path.join(cache_dir, f"candidate_{candidate_id}.json")
                if os.path.exists(path):
                    try:
                        with open(path, 'r', encoding='utf-8') as f:
                            cached = json.load(f)
                        if isinstance(cached, dict) and 'data' in cached:
                            cached['data']['resume_url'] = url
                            with open(path, 'w', encoding='utf-8') as f:
                                json.dump(cached, f)
                    except Exception:
                        pass
            return url
    except Exception as e:
        print(f"[Resume URL] Could not refresh URL for candidate {candidate_id}: {e}", flush=True)
    return ''


def fetch_candidate_contact(api_token: str, candidate_id: str) -> dict:
    """Return email and name for a candidate, fetched fresh from Manatal.

    Falls back to empty strings if unavailable.
    """
    url = f"https://api.manatal.com/open/v3/candidates/{candidate_id}/"
    headers = {"Authorization": f"Token {api_token}", "accept": "application/json"}
    try:
        resp = httpx.get(url, headers=headers, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            return {
                "email": data.get("email", ""),
                "name": data.get("name") or data.get("full_name", ""),
            }
    except Exception as e:
        print(f"[Manatal] Failed to fetch contact for candidate {candidate_id}: {e}", flush=True)
    return {"email": "", "name": ""}


def update_match_stage(api_token: str, job_id: str, candidate_id: str, stage_name: str) -> bool:
    """Move a candidate to a named pipeline stage in Manatal for the given job.

    Returns True on success, False if the update could not be completed.
    Failures are logged but do not raise.
    """
    headers = {
        "Authorization": f"Token {api_token}",
        "accept": "application/json",
        "content-type": "application/json",
    }
    base_url = f"https://api.manatal.com/open/v3/jobs/{job_id}/matches/"

    # Step 1: find the match record for this candidate; also collect stage IDs seen
    match_id: Optional[int] = None
    page = 1
    while True:
        try:
            resp = httpx.get(base_url, headers=headers, params={"page": page, "page_size": 100}, timeout=15)
            resp.raise_for_status()
        except Exception as e:
            print(f"[Manatal] Could not list matches for job {job_id}: {e}", flush=True)
            return False
        data = resp.json()
        results = data.get("results") or data.get("data") or []
        for match in results:
            if not isinstance(match, dict):
                continue
            stage_obj = match.get("stage") or match.get("job_pipeline_stage") or {}
            if isinstance(stage_obj, dict) and stage_obj.get("id"):
                _update_stage_id_cache(stage_obj)
            cand = match.get("candidate") or {}
            cand_id = cand.get("id") if isinstance(cand, dict) else cand
            if str(cand_id) == str(candidate_id):
                match_id = match.get("id")
        if match_id is not None:
            break
        if not has_next_page(data, results, 100):
            break
        page += 1

    if match_id is None:
        print(f"[Manatal] No match found for candidate {candidate_id} in job {job_id}", flush=True)
        return False

    # Step 2: look up the stage ID from the accumulated cache (pipeline-stages endpoint is not available)
    target_stage_id: Optional[int] = None
    target_norm = " ".join(stage_name.strip().lower().split())
    stage_id_cache = _load_stage_id_cache()
    target_stage_id = stage_id_cache.get(target_norm)
    if target_stage_id is None:
        print(f"[Manatal] Stage '{stage_name}' not in ID cache; will attempt name-based PATCH", flush=True)

    # Step 3: PATCH the match
    patch_url = f"https://api.manatal.com/open/v3/jobs/{job_id}/matches/{match_id}/"
    payload = {"stage": target_stage_id} if target_stage_id is not None else {"stage": {"name": stage_name}}
    try:
        resp = httpx.patch(patch_url, headers=headers, json=payload, timeout=15)
        if resp.status_code in (200, 204):
            print(f"[Manatal] Stage updated to '{stage_name}' for candidate {candidate_id} in job {job_id}", flush=True)
            return True
        print(f"[Manatal] Stage PATCH returned {resp.status_code}: {resp.text[:200]}", flush=True)
        return False
    except Exception as e:
        print(f"[Manatal] Exception patching match stage: {e}", flush=True)
        return False


def _parse_candidate(data: Dict[str, Any]) -> CandidateMeta:
    def _normalize_list(values: Any) -> list[str]:
        if not values:
            return []
        if not isinstance(values, list):
            return [str(values)]
        normalized: list[str] = []
        for item in values:
            if isinstance(item, dict):
                value = item.get("name") or item.get("value") or item.get("label") or ""
                if value:
                    normalized.append(str(value))
            elif item:
                normalized.append(str(item))
        return normalized

    return CandidateMeta(
        id=str(data.get("id", "")),
        name=data.get("name") or data.get("full_name", ""),
        current_position=data.get("current_position", ""),
        current_company=data.get("current_company", ""),
        latest_degree=data.get("latest_degree", ""),
        latest_university=data.get("latest_university", ""),
        location=data.get("location") or data.get("candidate_location") or data.get("address", ""),
        tags=_normalize_list(data.get("tags") or data.get("candidate_tags") or []),
        industries=_normalize_list(data.get("industries") or data.get("candidate_industries") or []),
        description=data.get("description", ""),
        resume_url=data.get("resume_url") or data.get("resume") or data.get("cv_url") or "",
        resume_text="",  # Can be filled in later if needed
    )
