from __future__ import annotations

import copy
import os

import yaml
from pydantic import BaseModel, field_validator
from sa_candidate_finder import secrets
from sa_candidate_finder import settings


_DEFAULT_UNIVERSAL_DEALBREAKERS: list[dict] = [
    {
        "id": "no_professional_experience",
        "rule": "Candidate has no professional work experience at all",
        "keyword_signals": [],
        "universal": True,
    },
    {
        "id": "completely_unrelated_field",
        "rule": "Candidate's entire background is in a completely unrelated field (e.g., manufacturing laborer applying for EA role)",
        "keyword_signals": [],
        "universal": True,
    },
]


class Config(BaseModel):
    # Pipeline
    max_hard_constraints: int = settings.MAX_HARD_CONSTRAINTS
    max_keyword_set: int = settings.MAX_KEYWORD_SET
    min_good_match_score: float = settings.MIN_GOOD_MATCH_SCORE
    min_pool_size: int = settings.MIN_POOL_SIZE
    max_filter_pool_size: int = settings.MAX_FILTER_POOL_SIZE
    filter_progress_every_pages: int = settings.FILTER_PROGRESS_EVERY_PAGES
    shortlist_size: int = settings.SHORTLIST_SIZE
    phase2_good_match_target: int = settings.PHASE2_GOOD_MATCH_TARGET
    top_x: int = settings.DEFAULT_TOP_X

    # Embedding cache
    embedding_cache_ttl_days: int = settings.EMBEDDING_CACHE_TTL_DAYS
    embedding_cache_path: str = settings.EMBEDDING_CACHE_PATH

    # Output
    results_dir: str = settings.RESULTS_DIR

    # Telemetry
    telemetry_log_path: str = settings.TELEMETRY_LOG_PATH
    feedback_log_path: str = settings.FEEDBACK_LOG_PATH

    # Embedding
    embedding_provider: str = settings.EMBEDDING_PROVIDER
    embedding_model: str = settings.EMBEDDING_MODEL
    embedding_price_per_million_tokens: float = settings.EMBEDDING_PRICE_PER_MILLION_TOKENS

    # LLM
    llm_provider: str = settings.LLM_PROVIDER
    llm_model: str = settings.LLM_MODEL
    llm_input_price_per_million_tokens: float = settings.LLM_INPUT_PRICE_PER_MILLION_TOKENS
    llm_output_price_per_million_tokens: float = settings.LLM_OUTPUT_PRICE_PER_MILLION_TOKENS

    # System-owned universal dealbreakers (operator configurable)
    universal_dealbreakers: list[dict] = copy.deepcopy(_DEFAULT_UNIVERSAL_DEALBREAKERS)

    # Manatal MCP
    manatal_base_url: str = ""

    # Resilience
    mcp_timeout_seconds: int = settings.MCP_TIMEOUT_SECONDS
    embedding_timeout_seconds: int = settings.EMBEDDING_TIMEOUT_SECONDS
    llm_timeout_seconds: int = settings.LLM_TIMEOUT_SECONDS
    max_retries: int = settings.MAX_RETRIES
    retry_backoff: str = settings.RETRY_BACKOFF

    # Secrets loaded from centralized constants
    openai_api_key: str = ""

    @field_validator("retry_backoff")
    @classmethod
    def _valid_backoff(cls, v: str) -> str:
        if v not in ("exponential", "fixed"):
            raise ValueError("retry_backoff must be 'exponential' or 'fixed'")
        return v

    @field_validator("universal_dealbreakers")
    @classmethod
    def _valid_universal_dealbreakers(cls, value: list[dict]) -> list[dict]:
        normalized: list[dict] = []
        for item in value or []:
            if not isinstance(item, dict):
                continue
            did = str(item.get("id", "")).strip() or "system_dealbreaker"
            rule = str(item.get("rule", "")).strip()
            if not rule:
                continue
            signals = item.get("keyword_signals", [])
            if not isinstance(signals, list):
                signals = []
            normalized.append(
                {
                    "id": did,
                    "rule": rule,
                    "keyword_signals": [str(s).strip().lower() for s in signals if str(s).strip()],
                    "universal": True,
                }
            )
        return normalized


def load_config() -> Config:
    """Build runtime config from config.yaml plus centralized secrets."""
    config_path = os.getenv("SA_CANDIDATE_FINDER_CONFIG", "config.yaml")
    file_values: dict = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"Config file must contain a YAML object at top level: {config_path}")
            file_values = loaded

    cfg = Config(**file_values)

    # Allow callers (web rerank worker) to route output JSON into a custom directory.
    results_dir_override = os.getenv("SA_RESULTS_DIR", "").strip()
    if results_dir_override:
        cfg.results_dir = results_dir_override

    if not secrets.OPENAI_API_KEY or secrets.OPENAI_API_KEY.startswith("REPLACE_WITH_"):
        raise ValueError(
            "OPENAI_API_KEY constant is not set in src/sa_candidate_finder/secrets.py."
        )
    cfg.openai_api_key = secrets.OPENAI_API_KEY

    if not secrets.MANATAL_BASE_URL or secrets.MANATAL_BASE_URL.startswith("REPLACE_WITH_"):
        raise ValueError(
            "MANATAL_BASE_URL constant is not set in src/sa_candidate_finder/secrets.py."
        )
    cfg.manatal_base_url = secrets.MANATAL_BASE_URL

    return cfg
