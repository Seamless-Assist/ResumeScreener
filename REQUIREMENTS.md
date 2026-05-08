# SeamlessAssist Resume Finder Requirements

## Version And Change Control
- Version: 1.0
- Last Updated: 2026-04-15
- Status: Approved. Implementation may proceed.
- Rule: No implementation starts until this revision is explicitly approved.
- This is a living specification and will be updated as decisions evolve.

## Deployment Model
- CLI-only executable for v1. No API or web interface.
- Packaged as a portable executable that runs on any local machine.
- Executable name: `SACandidateFinder`.
- Connects to remote endpoints (Manatal MCP, OpenAI) as needed.
- Each user has their own independent local installation.

## CLI Interface
### Commands
| Command | Description |
|---|---|
| `SACandidateFinder help` | Show all commands and their inputs |
| `SACandidateFinder search --jd path/to/job.txt --top 5` | Run full candidate search pipeline for a given JD |
| `SACandidateFinder feedback --jd path/to/job.txt` | Provide feedback on the most recent run for that JD |
| `SACandidateFinder learn` | Manually trigger learning update from accumulated feedback |

### Input Format
- Job description input: file path only (`.txt` or `.pdf`).
- All modes follow the consistent format: `SACandidateFinder <mode> --<input>`.
- `search` accepts `--top` as the user-facing input for final result count (default 5).

### Output
- `search` results are printed to console and saved to a Markdown file.
- Results file is saved in a configurable output directory (default: `results/` alongside the executable).
- Results file is named automatically: `<jd_filename>_<YYYYMMDD>_<HHMM>.md`.

### Results File Structure
Each results file contains:
- Header: JD file path, run ID, date, hard constraints used, candidates evaluated, candidates returned.
- One section per returned candidate containing:
  - Rank and full name
  - Fit score
  - Current position and location
  - Resume URL
  - Strengths
  - Risks/gaps
  - Rationale summary

## Configuration Model
- No user-editable runtime config file in v1.
- End-user input is limited to CLI arguments (notably `--top`).
- Internal non-secret settings are centralized in `src/sa_candidate_finder/settings.py`.
- Secrets are centralized in `src/sa_candidate_finder/secrets.py`.

## Problem Statement
There are 30,000 resumes in a database accessed through an MCP interface. Given a job description, the system must return the top X matching candidates while balancing:
- Cost
- Search speed
- Match quality

Key constraints:
- Do not vectorize the entire 30K database upfront.
- Do not run slow full-database search on every query.
- Keep ongoing operational cost manageable.

## Goals
1. Return top X candidates for a job description with clear reasoning.
2. Use a staged pipeline where expensive processing runs only on a small shortlist.
3. Improve repeated-search performance through lazy cache growth.
4. Keep cost inputs and operational knobs configurable to support tuning over time.
5. Provide strong observability for both internal telemetry and user-facing progress updates.

## Baseline Architecture Requirement
The required baseline is Option 4 plus Option 5.

### Option 4 (Required): Hybrid Filter + Embed + LLM
Three stages with increasing per-item cost and decreasing candidate volume:
1. Structured filtering through MCP to reduce 30K to hundreds.
2. Embedding-based semantic ranking on the filtered set.
3. LLM evaluation only on top N shortlisted candidates.

### Option 5 (Required): Lazy Embedding Cache
1. Create embedding only when candidate appears in filtered results and cache is missing.
2. Reuse cached embedding on subsequent searches.
3. Apply cache TTL of 7 days by default, with configurable value.
4. Refresh stale embeddings on access.

## MCP And Data Assumptions (Manatal)
Source: https://support.manatal.com/docs/mcp-server-actions

### Structured Search Inputs (Used For Pre-Filter)
- Current position
- Current company
- Latest degree
- Latest university
- Location and address
- Candidate tags
- Candidate industries
- Description free text
- Source type and created/updated date ranges

### Candidate Data For Ranking
Search Candidates metadata is used for embedding and semantic ranking:
- Full name
- Current position
- Current company
- Latest degree
- Latest university
- Location
- Tags
- Industries
- Description

### Full Resume Retrieval
View Candidate Profile returns resume URL (PDF). Full resume fetch and parsing is required only for final LLM evaluation on shortlist.

## End-To-End Required Flow
### Stage 1: Agentic JD Constraint Extraction (Interactive)
1. User provides a job description as input.
2. One LLM call extracts proposed hard constraints from the JD.
3. Constraints are restricted to Manatal MCP-filterable fields only:
   - Current position
   - Location
   - Latest degree
   - Candidate tags
   - Candidate industries
4. LLM ranks proposed constraints by confidence and selects the top N (default 3, configurable via `max_hard_constraints` agent config setting).
5. If LLM proposes more than `max_hard_constraints`, excess constraints are shown separately as soft criteria suggestions for the embedding/LLM stages.
6. Each proposed hard constraint must include:
   - Field name
   - Extracted value
   - Confidence level
   - Exact supporting phrase quoted from the JD
7. CLI presents constraints as a numbered list with actions: accept, edit, remove, add.
8. User reviews and adjusts constraints interactively.
9. User must issue an explicit confirm command to proceed.
10. Approved constraints and soft criteria are logged in telemetry.

### Stage 2: MCP Structured Filter
1. Call Manatal MCP Search Candidates with approved hard constraints.
2. Target a minimum candidate pool after filtering (configurable via `min_pool_size`, default 200).
3. If pool is below `min_pool_size`, log a warning and continue with available candidates.
4. Only Manatal MCP is supported in v1.

### Stage 3: Embedding Cache Check And Ranking
1. For each filtered candidate, check local embedding cache.
2. Embed cache misses using candidate metadata (description, position, company, degree, tags, industries).
3. Store new embeddings in cache with TTL metadata (default 7 days, configurable via `embedding_cache_ttl_days`).
4. Refresh stale embeddings on access.
5. Embed the job description.
6. Rank all filtered candidates by cosine similarity to the JD embedding.

### Stage 4: Shortlist Retrieval
1. Select top N candidates by embedding rank (default 20, configurable via `shortlist_size`).
2. For each shortlisted candidate, call Manatal View Candidate Profile to get resume URL.
3. Fetch and parse full resume PDF.
4. On fetch or parse failure: log an error, exclude candidate, continue with remaining shortlist.

### Stage 5: LLM Evaluation
1. Send full resumes plus job description to LLM for detailed evaluation.
2. Soft criteria (from Stage 1 overflow) are passed to LLM as additional guidance.
3. LLM alone determines final ranking of shortlisted candidates.
4. Return final top X candidates (default 10, configurable via `top_x`).

## Output Contract Requirements
For each returned candidate, output must include:
1. Final rank.
2. Overall fit score.
3. Explanation including:
   - Strengths
   - Risks or gaps
   - Rationale summary
4. Link or URL to full resume.

## Default Providers And Models
- Embedding provider: OpenAI
- Embedding model: `text-embedding-3-small` (configurable)
- LLM provider: OpenAI
- LLM model: `gpt-4.1-mini` (configurable)

## Configurability Requirements
All cost-relevant and pipeline-relevant parameters must be in the config file.

### Required Config Parameters
| Parameter | Default | Description |
|---|---|---|
| `max_hard_constraints` | 3 | Max hard MCP filter constraints extracted per JD |
| `min_pool_size` | 200 | Minimum candidate pool after MCP filter |
| `embedding_cache_ttl_days` | 7 | Embedding cache TTL in days |
| `shortlist_size` | 20 | Top N candidates sent to LLM |
| `top_x` | 10 | Final candidates returned to user |
| `embedding_provider` | openai | Embedding provider |
| `embedding_model` | text-embedding-3-small | Embedding model |
| `embedding_price_per_token` | configurable | For cost computation |
| `llm_provider` | openai | LLM provider |
| `llm_model` | gpt-4.1-mini | LLM model |
| `llm_input_price_per_token` | configurable | For cost computation |
| `llm_output_price_per_token` | configurable | For cost computation |
| `mcp_timeout_seconds` | configurable | MCP call timeout |
| `embedding_timeout_seconds` | configurable | Embedding call timeout |
| `llm_timeout_seconds` | configurable | LLM call timeout |
| `max_retries` | 2 | Max retries for MCP, embedding, LLM calls |
| `retry_backoff` | exponential | Retry backoff strategy |
| `telemetry_log_path` | configurable | Path to JSONL telemetry log |

## Resilience Requirements
- Retry policy applies to MCP, embedding, and LLM calls.
- Default: 2 retries with exponential backoff.
- All retry parameters are configurable.
- On resume fetch/parse failure: log error, exclude candidate, continue.

## Feedback And Self-Learning Requirements
### Feedback Mode
- Optional, user-invoked as a dedicated agent mode.
- Available after any completed search run.
- User provides per-candidate feedback referencing a prior run ID.

### Feedback Schema (per candidate)
- Run ID
- Candidate ID
- Signal: positive (selected) or negative (rejected)
- Optional free-text note (for example strengths, gaps, reasons)
- Timestamp

### Learning Updates
- Feedback stored immediately in local JSONL feedback log.
- Learning updates are manual-only in v1.
- User invokes a dedicated update command to apply accumulated feedback.
- No scheduled or automatic updates in v1.
- No cross-user sync in v1. Each installation learns independently.
- Feedback schema is designed to support future shared sync without format changes.

## Observability And Logging Requirements
### 1. Internal Structured Telemetry
Persisted as JSONL log. Per search run, capture at minimum:
- Run ID and timestamps
- JD fingerprint or hash
- Approved hard constraints and soft criteria
- Candidate counts by stage: initial pool, post-filter, post-ranking, LLM-evaluated, final returned
- Cache metrics: hits, misses, stale refreshes
- Model and pricing configuration used
- Token usage and cost estimates: embedding, LLM input, LLM output, total
- Error details and retry outcomes by stage
- Final status: success, partial, or failed

### 2. User-Facing Console Progress Events
During each run, print clear stage updates to console:
- Parsing and extracting constraints from JD
- Awaiting user constraint confirmation
- Filtering candidates via MCP (with count)
- Checking embedding cache and ranking candidates (with hit/miss counts)
- Fetching shortlisted profiles and resumes (with count)
- Running LLM evaluation (with count)
- Returning final ranked results (with count)

Each event includes stage name, item counts where applicable, and status.

## Non-Functional Requirements
1. CLI-only interface, packaged as a portable local executable.
2. Deterministic ranking behavior for identical inputs when external responses are stable.
3. Graceful degradation when one stage fails with logged errors and partial results where safe.
4. Clear typed interfaces and modular components for each pipeline stage.
5. External call resiliency via configurable timeouts, retries, and failure classification.
6. Python 3.11 or newer compatibility.

## Acceptance Criteria (v0.3)
1. Spec defines CLI-only portable executable as the deployment target.
2. Spec defines no user-editable runtime config file; secrets and internal settings are centralized in code for v1.
3. Spec defines agentic LLM-based constraint extraction with interactive CLI confirmation per JD.
4. Spec caps hard constraints at `max_hard_constraints` (default 3).
5. Spec restricts hard constraints to MCP-filterable fields only.
6. Spec defines three-stage pipeline: MCP filter, embedding ranking, LLM shortlist evaluation.
7. Spec mandates configurable `embedding_cache_ttl_days` (default 7).
8. Spec mandates internal shortlist control and user-facing top-X input via `search --top` (default 5).
9. Spec defines output contract: rank, fit score, explanation, resume URL.
10. Spec pins default models: text-embedding-3-small and gpt-4.1-mini.
11. Spec mandates JSONL telemetry with cost computation fields.
12. Spec mandates console progress events for user-facing updates.
13. Spec defines optional feedback mode with positive, negative, and free-text signals.
14. Spec defines manual-only local learning updates for v1.
15. Spec defines retry policy: 2 retries with exponential backoff (configurable).

## Open Questions
None. All pre-implementation questions resolved as of 2026-04-15.

## Decision Log
- 2026-04-15: Requirements-first process confirmed; approval required before implementation.
- 2026-04-15: Engineering-ready living spec format approved.
- 2026-04-15: Option 4 plus Option 5 selected as baseline architecture.
- 2026-04-15: Weekly cache TTL accepted, configurable.
- 2026-04-15: Default shortlist N=20, final X=10, both configurable.
- 2026-04-15: LLM output must include ranking, score, explanation, resume URL.
- 2026-04-15: Performance targets accepted as initial and tunable.
- 2026-04-15: No hard budget cap; full cost computation capability via configurable inputs.
- 2026-04-15: Dual logging: internal JSONL telemetry and console progress updates.
- 2026-04-15: CLI-only, portable local executable with no user runtime config file.
- 2026-04-15: Shared secrets are centralized constants in code (temporary), not in config and not env vars.
- 2026-04-15: Agentic LLM constraint extraction with interactive CLI confirmation per JD.
- 2026-04-15: Max hard constraints = 3 (agent config, not user-facing per-run setting).
- 2026-04-15: Hard constraints restricted to MCP-filterable fields only.
- 2026-04-15: Excess constraints auto-demoted to soft criteria suggestions.
- 2026-04-15: Each constraint must include source quote from JD.
- 2026-04-15: Manatal MCP only in v1; provider abstraction deferred.
- 2026-04-15: JSONL log files for telemetry.
- 2026-04-15: Console-only user-facing progress events.
- 2026-04-15: LLM alone decides final ranking.
- 2026-04-15: Optional feedback mode with positive, negative, free-text signals.
- 2026-04-15: Manual-only learning updates in v1.
- 2026-04-15: Local-only learning, no cross-user sync.
- 2026-04-15: Resume fetch/parse failure: log error, exclude, continue.
- 2026-04-15: OpenAI default for both embedding and LLM.
- 2026-04-15: Default embedding model text-embedding-3-small, default LLM gpt-4.1-mini.
- 2026-04-15: Retry policy: 2 retries with exponential backoff, configurable.
- 2026-04-15: Internal settings centralized in code for controlled testing.
- 2026-04-15: OpenAI API key and Manatal secret URL sourced from constants in code.
- 2026-04-15: Windows Credential Manager integration added as TODO for a later phase.
- 2026-04-15: CLI framework: Typer.
- 2026-04-15: PDF parsing library: PyMuPDF (fitz).
- 2026-04-15: Embedding cache storage: SQLite (single file, structured, queryable).
- 2026-04-15: Executable name: SACandidateFinder.
- 2026-04-15: CLI format: SACandidateFinder <mode> --<input>.
- 2026-04-15: Modes: help, search, feedback, learn.
- 2026-04-15: help mode shows all commands and inputs in one view.
- 2026-04-15: `search` accepts user-facing `--top` input (default 5).
- 2026-04-15: JD input is file path only (.txt or .pdf).
- 2026-04-15: feedback defaults to most recent run for that JD; --list dropped for v1.
- 2026-04-15: search saves results to Markdown file and prints to console.
- 2026-04-15: Results directory configurable, default results/ alongside executable.
- 2026-04-15: Results file named: <jd_filename>_<YYYYMMDD>_<HHMM>.md.
- 2026-04-15: Results file structure: header, then per-candidate rank, score, position, location, resume URL, strengths, risks, rationale.
