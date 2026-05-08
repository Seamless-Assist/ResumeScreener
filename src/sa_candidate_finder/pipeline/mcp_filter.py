from __future__ import annotations

from typing import Any

from sa_candidate_finder.config import Config
from sa_candidate_finder.mcp_client import call_tool
from sa_candidate_finder.models import CandidateMeta, HardConstraint, RunTelemetry

# Maps internal field names to Manatal API query param names.
# Fields not in this map are passed through as-is (supporting free-form user additions).
_FIELD_MAP = {
    "current_position":  "current_position",
    "location":          "candidate_location",
    "latest_degree":     "latest_degree",
    "current_company":   "current_company",
    "latest_university": "latest_university",
    "description":       "description",
    "tags":              "candidate_tags",
    "industries":        "candidate_industries",
}


def filter_candidates(
    hard_constraints: list[HardConstraint],
    cfg: Config,
    telemetry: RunTelemetry,
) -> list[CandidateMeta]:
    """Call Manatal MCP Search Candidates with hard constraints.

    Automatically relaxes filters if 0 candidates are returned:
    1. Full filter set
    2. Drop description (keyword) constraints
    3. Drop latest_degree constraints
    4. Keep only core JD-derived constraints (source_quote != "user-added")
    """
    import typer

    fallback_stages = _build_fallback_stages(hard_constraints)

    for stage_label, stage_constraints in fallback_stages:
        params = _build_params(stage_constraints)
        candidates = _call_with_degree_alias_expansion(params, cfg)
        if candidates:
            if stage_label != "full":
                typer.echo(f"      Relaxed filters to '{stage_label}' — found {len(candidates)} candidates.")
            break
        if stage_label != fallback_stages[-1][0]:
            typer.echo(f"      0 results with '{stage_label}' filters — relaxing...")

    telemetry.post_filter_count = len(candidates)

    if len(candidates) < cfg.min_pool_size:
        typer.echo(
            f"  Warning: only {len(candidates)} candidates returned after filter "
            f"(min_pool_size={cfg.min_pool_size})."
        )

    return candidates


def _build_fallback_stages(
    constraints: list[HardConstraint],
) -> list[tuple[str, list[HardConstraint]]]:
    """Build progressively relaxed constraint lists for fallback retries."""
    stages: list[tuple[str, list[HardConstraint]]] = []

    # Stage 1: everything
    stages.append(("full", constraints))

    # Stage 2: drop user-added keyword (description) filters
    no_keywords = [c for c in constraints if c.field != "description"]
    if len(no_keywords) < len(constraints):
        stages.append(("no keywords", no_keywords))

    # Stage 3: also drop latest_degree
    no_degree = [c for c in no_keywords if c.field != "latest_degree"]
    if len(no_degree) < len(no_keywords):
        stages.append(("no degree", no_degree))

    # Stage 4: core JD constraints only (not user-added structured filters either)
    core = [c for c in no_degree if c.source_quote != "user-added"]
    if len(core) < len(no_degree):
        stages.append(("core only", core))

    # Deduplicate consecutive identical stages
    unique: list[tuple[str, list[HardConstraint]]] = []
    for label, cs in stages:
        if not unique or cs != unique[-1][1]:
            unique.append((label, cs))

    return unique


def _call_with_degree_alias_expansion(
    params: dict[str, Any],
    cfg: Config,
) -> list[CandidateMeta]:
    degree_value = params.get("latest_degree")
    if not isinstance(degree_value, str) or not degree_value.strip():
        return _call_mcp_search(params, cfg)

    aliases = _degree_aliases(degree_value)
    if len(aliases) <= 1:
        return _call_mcp_search(params, cfg)

    import typer

    typer.echo(f"      Expanding latest_degree '{degree_value}' to aliases: {', '.join(aliases)}")

    merged: list[CandidateMeta] = []
    seen_ids: set[str] = set()
    for alias in aliases:
        alias_params = dict(params)
        alias_params["latest_degree"] = alias
        candidates = _call_mcp_search(alias_params, cfg)
        for c in candidates:
            if c.id and c.id in seen_ids:
                continue
            if c.id:
                seen_ids.add(c.id)
            merged.append(c)

    return merged


def _degree_aliases(value: str) -> list[str]:
    normalized = value.strip().lower().replace(".", "")
    alias_map: dict[str, list[str]] = {
        "bachelor": ["Bachelor", "Bachelors", "BS", "BSc", "BA", "BEng", "BE", "BTech"],
        "bachelors": ["Bachelor", "Bachelors", "BS", "BSc", "BA", "BEng", "BE", "BTech"],
        "master": ["Master", "Masters", "MS", "MSc", "MA", "MEng"],
        "masters": ["Master", "Masters", "MS", "MSc", "MA", "MEng"],
        "mba": ["MBA", "Master of Business Administration"],
        "phd": ["PhD", "Doctorate", "Doctoral"],
        "doctorate": ["PhD", "Doctorate", "Doctoral"],
    }
    aliases = alias_map.get(normalized)
    if not aliases:
        return [value]
    return aliases


def _build_params(hard_constraints: list[HardConstraint]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for c in hard_constraints:
        # Use mapped name if known, otherwise pass through as-is (free-form user fields)
        api_field = _FIELD_MAP.get(c.field, c.field)
        params[api_field] = c.value

    params.setdefault("page", 1)
    params.setdefault("page_size", 100)  # Manatal API caps at 100 per page
    return params


def _call_mcp_search(
    params: dict[str, Any],
    cfg: Config,
) -> list[CandidateMeta]:
    import typer

    page_size = int(params.get("page_size", 100))
    page = int(params.get("page", 1))
    max_pool_size = max(page_size, int(cfg.max_filter_pool_size))
    progress_every = max(1, int(getattr(cfg, "filter_progress_every_pages", 5)))

    all_candidates: list[CandidateMeta] = []
    seen_ids: set[str] = set()

    while len(all_candidates) < max_pool_size:
        page_params = dict(params)
        page_params["page"] = page
        page_params["page_size"] = page_size

        payload = call_tool(name="candidates_list", arguments=page_params, cfg=cfg, request_id=2)
        page_candidates = _parse_candidates(payload)

        if not page_candidates:
            typer.echo(f"      Page {page}: 0 candidates (no more pages)")
            break

        new_on_page = 0
        for candidate in page_candidates:
            if candidate.id and candidate.id in seen_ids:
                continue
            if candidate.id:
                seen_ids.add(candidate.id)
            all_candidates.append(candidate)
            new_on_page += 1
            if len(all_candidates) >= max_pool_size:
                break

        if page == 1 or page % progress_every == 0:
            typer.echo(
                f"      Page {page}: +{new_on_page} candidates "
                f"(total {len(all_candidates)}/{max_pool_size})"
            )

        # Guard against looping when API repeats page content.
        if new_on_page == 0:
            typer.echo("      Pagination stopped: repeated page content detected.")
            break

        if not _has_more_pages(payload, len(page_candidates), page_size, page):
            break

        page += 1

    if len(all_candidates) >= max_pool_size:
        typer.echo(f"      Reached max filter pool size: {max_pool_size} candidates.")

    return all_candidates


def _has_more_pages(
    payload: dict[str, Any],
    page_result_count: int,
    page_size: int,
    current_page: int,
) -> bool:
    next_link = payload.get("next")
    if next_link not in (None, "", False):
        return True

    total_count = payload.get("count")
    if isinstance(total_count, int) and total_count >= 0:
        return current_page * page_size < total_count

    # Fallback when pagination metadata is absent.
    return page_result_count >= page_size


def _parse_candidates(data: dict[str, Any]) -> list[CandidateMeta]:
    results = data.get("results", data.get("candidates", []))
    candidates = []
    for r in results:
        tags = r.get("candidate_tags", r.get("tags", []))
        industries = r.get("candidate_industries", r.get("industries", []))

        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]
        if isinstance(industries, str):
            industries = [i.strip() for i in industries.split(",") if i.strip()]

        candidates.append(
            CandidateMeta(
                id=str(r.get("id", "")),
                name=r.get("full_name", r.get("name", "")),
                current_position=r.get("current_position", ""),
                current_company=r.get("current_company", ""),
                latest_degree=r.get("latest_degree", ""),
                latest_university=r.get("latest_university", ""),
                location=r.get("candidate_location", r.get("location", "")),
                tags=tags,
                industries=industries,
                description=r.get("description", ""),
                resume_url=r.get("resume", ""),
            )
        )
    return candidates
