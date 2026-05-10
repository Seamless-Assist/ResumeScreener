from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sa_candidate_finder.config import Config
from sa_candidate_finder.models import CandidateResult, HardConstraint


def write_results(
    results: list[CandidateResult],
    jd_path: Path,
    run_id: str,
    hard_constraints: list[HardConstraint],
    cfg: Config,
    evaluated_count: Optional[int] = None,
) -> Path:
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    stem = jd_path.stem
    output_path = results_dir / f"{stem}_{timestamp}.md"

    structured_constraints = [c for c in hard_constraints if c.field != "description"]
    keyword_constraints = [c for c in hard_constraints if c.field == "description"]

    constraints_str = ", ".join(f"{c.field}={c.value}" for c in structured_constraints) or "(none)"
    keywords_str = ", ".join(c.value for c in keyword_constraints) or "(none)"
    evaluated = evaluated_count if evaluated_count is not None else len(results)

    lines: list[str] = [
        "# Candidate Search Results",
        "",
        f"- **Job Description:** {jd_path}",
        f"- **Run ID:** {run_id}",
        f"- **Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"- **Hard Constraints:** {constraints_str}",
        f"- **Keywords:** {keywords_str}",
        f"- **Candidates Evaluated:** {evaluated}",
        f"- **Candidates Returned:** {len(results)}",
        "",
        "---",
        "",
    ]

    for r in results:
        c = r.candidate
        dimension_summary = (
            f"Experience {r.__dict__.get('experience_score', 0.0):.1f}/30; "
            f"Skills {r.__dict__.get('skills_score', 0.0):.1f}/25; "
            f"Location {r.__dict__.get('location_score', 0.0):.1f}/20; "
            f"English & AI {r.__dict__.get('english_ai_score', 0.0):.1f}/15; "
            f"Salary {r.__dict__.get('salary_score', 0.0):.1f}/10"
        )
        lines += [
            f"## Rank {r.rank} — {c.name}",
            "",
            f"- **Fit Score:** {r.fit_score}/10",
            f"- **Tier:** {r.tier or 'N/A'}",
            f"- **Why This Tier:** {dimension_summary}",
            f"- **Current Position:** {c.current_position} @ {c.current_company}",
            f"- **Location:** {c.location}",
            f"- **Resume:** {c.resume_url}",
            "",
            "### Strengths",
            *[f"- {s}" for s in r.strengths],
            "",
            "### Risks / Gaps",
            *[f"- {s}" for s in r.risks],
            "",
            "### Rationale",
            r.rationale,
            "",
            "---",
            "",
        ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
