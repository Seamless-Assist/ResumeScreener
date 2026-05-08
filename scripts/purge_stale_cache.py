import json
import os
import glob
import sys

sys.path.insert(0, "src")
from sa_candidate_finder.manatal_candidates import _looks_like_pdf_stream_text

bad = []
for p in glob.glob("cache/candidates/candidate_*.json"):
    try:
        d = json.load(open(p, encoding="utf-8"))
        rt = d.get("resume_text", "")
        if rt and (_looks_like_pdf_stream_text(rt) or rt.startswith("__RESUME_PARSE_FAILED__")):
            bad.append(p)
    except Exception:
        pass

print(f"Stale entries to delete: {len(bad)}")
for p in bad:
    os.remove(p)
print("Done.")
