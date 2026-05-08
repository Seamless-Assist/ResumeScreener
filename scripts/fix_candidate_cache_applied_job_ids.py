import os
import json
from sa_candidate_finder.secrets import MANATAL_API_KEY
import httpx

CACHE_DIR = os.path.join(os.path.dirname(__file__), '../cache/candidates')
API_URL = "https://api.manatal.com/open/v3/candidates/"
HEADERS = {
    "Authorization": f"Token {MANATAL_API_KEY}",
    "accept": "application/json",
}

def update_cache_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        entry = json.load(f)
    # If already has applied_job_ids, skip
    if 'applied_job_ids' in entry:
        return False
    candidate_id = entry['data']['id'] if isinstance(entry['data'], dict) else entry['data']
    resp = httpx.get(f"{API_URL}{candidate_id}/", headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    applied_job_ids = data.get('applied_job_ids') or []
    # If not present, try to infer from jobs field
    if not applied_job_ids and 'jobs' in data:
        applied_job_ids = [str(j['id']) for j in data['jobs'] if 'id' in j]
    entry['applied_job_ids'] = applied_job_ids
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(entry, f, ensure_ascii=False, indent=2)
    print(f"Updated {os.path.basename(path)} with applied_job_ids: {applied_job_ids}")
    return True

def main():
    updated = 0
    for fname in os.listdir(CACHE_DIR):
        if not fname.endswith('.json'):
            continue
        path = os.path.join(CACHE_DIR, fname)
        try:
            if update_cache_file(path):
                updated += 1
        except Exception as e:
            print(f"Failed to update {fname}: {e}")
    print(f"Done. Updated {updated} files.")

if __name__ == "__main__":
    main()
