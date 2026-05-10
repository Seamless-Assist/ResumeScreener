# --- MCP Candidate Search Tool Wrapper ---
from __future__ import annotations
import re
from sa_candidate_finder.models import CandidateResult, CandidateMeta
from typing import Any, List, Optional


def _keyword_matches(keyword: str, haystack: str) -> bool:
    """Return True if every word in the keyword phrase appears in haystack at a word boundary.

    This allows "care coordination" to match "coordination of patient care" (words present,
    not necessarily adjacent or in order), while still avoiding false positives like
    "care" matching inside "healthcare" as a substring.
    """
    return all(re.search(r'\b' + re.escape(w) + r'\b', haystack) for w in keyword.split())

def search_candidates(
    candidates: List[CandidateMeta],
    keywords: List[str],
    cfg: Config,
    dealbreakers: Optional[List[dict]] = None,
    required_keywords: Optional[List[str]] = None,
) -> List[CandidateResult]:
    """
    Call the MCP candidate search tool to score and rank candidates by fit to keywords.
    Hard filters are applied first: disqualified candidates are excluded from results.
    This is a stub: replace with real MCP tool call and result parsing.
    """
    results = []
    normalized_keywords = [k.strip().lower() for k in keywords if k and k.strip()]
    normalized_required_keywords = [k.strip().lower() for k in (required_keywords or []) if k and k.strip()]
    active_dealbreakers = dealbreakers or []

    for c in candidates:
        haystack = " ".join(
            part
            for part in [
                c.name or "",
                c.current_position or "",
                c.current_company or "",
                c.latest_degree or "",
                c.latest_university or "",
                c.location or "",
                " ".join(c.tags or []),
                " ".join(c.industries or []),
                c.description or "",
                c.resume_text or "",
            ]
            if part
        ).lower()

        # --- Hard filter check (applied before scoring) ---
        disqualify_reason = _check_dealbreakers(c, haystack, active_dealbreakers)
        if disqualify_reason:
            # Score = 0, tier = D, excluded from ranked results.
            continue

        # User-specified required keywords are treated as session hard filters.
        if normalized_required_keywords and not all(_keyword_matches(req, haystack) for req in normalized_required_keywords):
            continue

        matched_keywords = [k for k in normalized_keywords if _keyword_matches(k, haystack)]
        if not normalized_keywords:
            score = 0.0
        else:
            score = round((len(matched_keywords) / len(normalized_keywords)) * 10, 1)
        if score <= 0:
            continue
        results.append(
            CandidateResult(
                rank=0,
                candidate=c,
                fit_score=score,
                strengths=matched_keywords,
                risks=[],
                rationale="Normalized stub keyword scoring",
            )
        )
    results.sort(key=lambda r: r.fit_score, reverse=True)
    for i, r in enumerate(results):
        r.rank = i + 1
    return results


def _check_dealbreakers(candidate: CandidateMeta, haystack: str, dealbreakers: List[dict]) -> str:
    """
    Check a candidate against all dealbreakers.
    Returns empty string because all hard-filter evaluation is delegated to the LLM evaluator.
    Search-time keyword filtering should not disqualify candidates on hard filters.
    """
    _ = candidate
    _ = haystack
    _ = dealbreakers
    return ""

import json
import time
from typing import Any

import httpx

from sa_candidate_finder.config import Config


def call_tool(name: str, arguments: dict[str, Any], cfg: Config, request_id: int = 1) -> dict[str, Any]:
    """Call a Manatal MCP tool over streamable HTTP and return parsed JSON payload.

    Manatal responds with SSE frames where the MCP result is embedded in
    result.content[0].text as a JSON string.
    """
    for attempt in range(cfg.max_retries + 1):
        try:
            return _call_tool_once(name=name, arguments=arguments, cfg=cfg, request_id=request_id)
        except Exception:
            if attempt == cfg.max_retries:
                raise
            wait = (2**attempt) if cfg.retry_backoff == "exponential" else 1
            time.sleep(wait)
    raise RuntimeError("Unreachable")


def _call_tool_once(name: str, arguments: dict[str, Any], cfg: Config, request_id: int) -> dict[str, Any]:
    base_url = cfg.manatal_base_url.rstrip("/")
    headers = {
        "accept": "application/json, text/event-stream",
        "content-type": "application/json",
    }

    with httpx.Client(timeout=cfg.mcp_timeout_seconds) as client:
        # 1) initialize
        init_payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "sa-candidate-finder", "version": "0.1"},
            },
        }
        init_resp = client.post(base_url, json=init_payload, headers=headers)
        init_resp.raise_for_status()

        session_id = init_resp.headers.get("mcp-session-id", "")
        if not session_id:
            raise RuntimeError("Missing mcp-session-id in initialize response")

        session_headers = {**headers, "mcp-session-id": session_id}

        # 2) initialized notification
        client.post(
            base_url,
            json={"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
            headers=session_headers,
        )

        # 3) tools/call
        call_payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
        call_resp = client.post(base_url, json=call_payload, headers=session_headers)
        call_resp.raise_for_status()

    # Parse event-stream body -> data JSON-RPC -> result.content[0].text JSON
    rpc = _parse_sse_jsonrpc(call_resp.text)
    result = rpc.get("result", {})
    content = result.get("content", [])
    if not content:
        return {}

    text = content[0].get("text", "")
    if not text:
        return {}

    return json.loads(text)


def _parse_sse_jsonrpc(raw_text: str) -> dict[str, Any]:
    # SSE events may span multiple data lines. Collect complete event payloads
    # (delimited by a blank line), then decode the first valid JSON-RPC object.
    current_data_lines: list[str] = []
    in_data_event = False

    def _try_decode_event(data_lines: list[str]) -> Optional[dict[str, Any]]:
        if not data_lines:
            return None
        payload = "\n".join(data_lines).strip()
        if not payload or payload == "[DONE]":
            return None
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            return None
        if isinstance(obj, dict):
            return obj
        return None

    def _decode_first_json_dict(text: str) -> Optional[dict[str, Any]]:
        decoder = json.JSONDecoder()
        for i, ch in enumerate(text):
            if ch != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(text[i:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
        return None

    for raw_line in raw_text.splitlines():
        line = raw_line.rstrip("\r")

        # Blank line = end of an SSE event.
        if line == "":
            event_obj = _try_decode_event(current_data_lines)
            if event_obj is not None:
                return event_obj
            current_data_lines = []
            in_data_event = False
            continue

        if line.startswith("data:"):
            in_data_event = True
            data_chunk = line[5:]
            if data_chunk.startswith(" "):
                data_chunk = data_chunk[1:]
            current_data_lines.append(data_chunk)
            continue

        # Ignore known SSE metadata lines.
        if line.startswith(":") or line.startswith("event:") or line.startswith("id:") or line.startswith("retry:"):
            continue

        # Some servers send a blank `data:` line followed by raw JSON continuation
        # lines without `data:` prefix. Treat these as continuation of current event.
        if in_data_event:
            current_data_lines.append(line)

    # Handle final event if stream does not end with a blank line.
    event_obj = _try_decode_event(current_data_lines)
    if event_obj is not None:
        return event_obj

    # Backward-compatible fallback for single-line payloads.
    for line in raw_text.splitlines():
        if line.startswith("data:"):
            candidate = line[5:].lstrip()
            try:
                obj = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj

    # Last-resort fallback: strip common SSE prefixes and try to decode first JSON object.
    cleaned_lines: list[str] = []
    for raw_line in raw_text.splitlines():
        line = raw_line.rstrip("\r")
        if line.startswith("data:"):
            cleaned_lines.append(line[5:].lstrip())
            continue
        if line.startswith(":") or line.startswith("event:") or line.startswith("id:") or line.startswith("retry:"):
            continue
        cleaned_lines.append(line)

    maybe_obj = _decode_first_json_dict("\n".join(cleaned_lines))
    if maybe_obj is not None:
        return maybe_obj

    raise RuntimeError(f"Unable to parse SSE JSON-RPC payload: {raw_text[:400]}")
