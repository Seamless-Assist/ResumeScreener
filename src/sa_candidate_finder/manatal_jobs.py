import httpx
import re as _re
from typing import List, Dict, Any, Optional

from sa_candidate_finder.manatal_pagination import has_next_page, is_invalid_page_response


def strip_html(text: str) -> str:
    """Strip HTML tags and decode common entities, returning clean plain text."""
    if not text:
        return ""
    # Replace block-level tags with newlines so paragraphs don't run together
    text = _re.sub(r"<(?:br|p|li|h[1-6]|div|tr)[^>]*>", "\n", text, flags=_re.IGNORECASE)
    # Remove all remaining tags
    text = _re.sub(r"<[^>]+>", "", text)
    # Decode common HTML entities
    entities = {"&nbsp;": " ", "&amp;": "&", "&lt;": "<", "&gt;": ">",
                "&quot;": '"', "&#39;": "'", "&ndash;": "–", "&mdash;": "—"}
    for ent, repl in entities.items():
        text = text.replace(ent, repl)
    # Collapse whitespace
    text = _re.sub(r"\n{3,}", "\n\n", text)
    text = _re.sub(r"[ \t]+", " ", text)
    return text.strip()


def fetch_all_jobs(api_token: str, page_size: int = 100, force_refresh: bool = False) -> List[Dict[str, Any]]:
    """
    Fetch all jobs from Manatal using the jobs_list endpoint.
    Handles pagination and rate limiting (429) with exponential backoff.
    Args:
        api_token: Manatal API token (string)
        page_size: Number of jobs per page (default 100, max allowed by API)
    Returns:
        List of job dicts as returned by the API.
    """
    import time
    base_url = "https://api.manatal.com/open/v3/jobs/"
    headers = {
        "Authorization": f"Token {api_token}",
        "accept": "application/json",
    }
    import os, json
    default_cache_dir = os.path.join(os.path.dirname(__file__), '../../cache/jobs')
    CACHE_PATH = os.getenv(
        "SA_JOBS_CACHE_PATH",
        os.path.join(default_cache_dir, 'all_jobs.json'),
    )
    CACHE_DIR = os.path.dirname(CACHE_PATH)
    os.makedirs(CACHE_DIR, exist_ok=True)
    CACHE_EXPIRY = 7 * 24 * 3600  # 7 days
    # Try cache first unless caller forces refresh.
    if (not force_refresh) and os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, 'r', encoding='utf-8') as f:
                cached = json.load(f)
            if isinstance(cached, dict) and 'fetched_at' in cached and 'jobs' in cached:
                import time as _time
                if _time.time() - cached['fetched_at'] < CACHE_EXPIRY:
                    print(f"[Cache] Loaded jobs from cache (age: {(_time.time() - cached['fetched_at'])/3600:.1f} hours)", flush=True)
                    return cached['jobs']
        except Exception as e:
            print(f"[Cache] Failed to load jobs cache: {e}", flush=True)
    elif force_refresh:
        print("[Cache] Force refresh enabled: bypassing jobs cache.", flush=True)
    jobs = []
    page = 1
    max_retries = 5
    import threading
    last_request_time = [0.0]
    lock = threading.Lock()
    def throttle():
        with lock:
            import time as _time
            now = _time.time()
            elapsed = now - last_request_time[0]
            min_interval = 0.67  # 1.5 TPS
            if elapsed < min_interval:
                _time.sleep(min_interval - elapsed)
            last_request_time[0] = _time.time()

    while True:
        params = {"page": page, "page_size": page_size}
        last_exception = None
        exhausted = False
        for attempt in range(max_retries):
            try:
                throttle()
                resp = httpx.get(base_url, headers=headers, params=params, timeout=30)
                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After", "").strip()
                    wait_s = int(retry_after) if retry_after.isdigit() else min(15, 3 * (attempt + 1))
                    print(f"\n[Rate Limit] 429 Too Many Requests from Manatal /jobs API! Attempt {attempt+1}/{max_retries}", flush=True)
                    print(f"[Rate Limit] Endpoint: {base_url}", flush=True)
                    print(f"[Rate Limit] Params: {params}", flush=True)
                    print(f"[Rate Limit] Waiting {wait_s}s before retrying...\n", flush=True)
                    import time as _time
                    _time.sleep(wait_s)
                    continue
                if is_invalid_page_response(resp.status_code, page):
                    # Walked past the last page — Manatal answers 404 "Invalid page."
                    print(f"[Pagination] Page {page} does not exist; treating as end of jobs list.", flush=True)
                    exhausted = True
                    break
                if resp.status_code >= 400:
                    print(f"[HTTP Error] Status: {resp.status_code}, Content: {resp.text}")
                resp.raise_for_status()
                break
            except Exception as e:
                last_exception = e
                resp_obj = getattr(e, "response", None)
                if resp_obj is not None and getattr(resp_obj, "status_code", None) == 429 and attempt < max_retries - 1:
                    retry_after = resp_obj.headers.get("Retry-After", "").strip()
                    wait_s = int(retry_after) if retry_after.isdigit() else min(15, 3 * (attempt + 1))
                    print(f"\n[Rate Limit] 429 Too Many Requests from Manatal /jobs API! Exception retry {attempt+1}/{max_retries}", flush=True)
                    print(f"[Rate Limit] Endpoint: {base_url}", flush=True)
                    print(f"[Rate Limit] Params: {params}", flush=True)
                    print(f"[Rate Limit] Waiting {wait_s}s before retrying...\n", flush=True)
                    import time as _time
                    _time.sleep(wait_s)
                    continue
                print(f"[Exception] {e}")
                if resp_obj is not None and hasattr(resp_obj, 'text'):
                    print(f"[Response Text] {resp_obj.text}")
                if attempt == max_retries - 1:
                    raise
        else:
            print("[Error] Exceeded max retries due to rate limiting or errors.")
            if last_exception:
                raise last_exception
            raise RuntimeError("Exceeded max retries due to rate limiting or errors.")
        if exhausted:
            break
        data = resp.json()
        results = data.get("results") or data.get("data") or data
        if not results or not isinstance(results, list):
            break
        jobs.extend(results)
        if not has_next_page(data, results, page_size):
            break
        page += 1
    # Save to cache
    temp_cache_path = CACHE_PATH + ".tmp"
    try:
        with open(temp_cache_path, 'w', encoding='utf-8') as f:
            import time as _time
            json.dump({'fetched_at': _time.time(), 'jobs': jobs}, f)
        os.replace(temp_cache_path, CACHE_PATH)
        print(f"[Cache] Saved jobs to cache.", flush=True)
    except Exception as e:
        print(f"[Cache] Failed to save jobs cache: {e}", flush=True)
        try:
            if os.path.exists(temp_cache_path):
                os.unlink(temp_cache_path)
        except OSError:
            pass
    return jobs
