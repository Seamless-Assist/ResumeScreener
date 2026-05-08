import json
from collections import Counter

# Path to your log file
log_path = "logs/telemetry.jsonl"

candidate_ids = []

with open(log_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip().startswith("[DEBUG] Full /matches/ API response"):
            # The next line should be the JSON dict
            try:
                data = next(f)
                # Some logs may have pprint formatting, so try to parse as JSON
                # If this fails, skip
                try:
                    response = json.loads(data)
                except Exception:
                    continue
                results = response.get("results", [])
                for entry in results:
                    # Each entry may be a dict with 'candidate' field
                    if isinstance(entry, dict):
                        cid = entry.get("candidate")
                        if cid is not None:
                            candidate_ids.append(cid)
            except StopIteration:
                break

# Count duplicates
counter = Counter(candidate_ids)
duplicates = {cid: count for cid, count in counter.items() if count > 1}

print(f"Total candidate IDs processed: {len(candidate_ids)}")
print(f"Unique candidate IDs: {len(set(candidate_ids))}")
if duplicates:
    print("Duplicate candidate IDs and their counts:")
    for cid, count in duplicates.items():
        print(f"  {cid}: {count}")
else:
    print("No duplicate candidate IDs found.")
